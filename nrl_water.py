"""Load the released NRL models without initializer or training dependencies."""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def model_index():
    return json.loads((ROOT/'model-index.json').read_text())

def load_model(model_id, weights_root, device='cpu'):
    """Verify a released checkpoint and strictly load its original FP32 graph."""
    spec = next((r for r in model_index() if r['model_id']==model_id), None)
    if spec is None: raise ValueError(f'Unknown model: {model_id}')
    checkpoint = Path(weights_root)/spec['checkpoint']
    if sha256(checkpoint)!=spec['sha256']: raise ValueError(f'Checkpoint checksum failed: {checkpoint}')
    tag = 'nafnet' if spec['family']=='nafnet' else 'legacy'
    module_spec = importlib.util.spec_from_file_location('nrl_arch_'+tag, ROOT/'architectures'/tag/'campaign_models.py')
    module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
    # The archived constructor first loads scratch initialization for training.
    # Inference builds the identical graph and loads selected weights directly.
    net = module.Restoration.__new__(module.Restoration)
    nn.Module.__init__(net); net.family=spec['family']; net.mode=spec['mode']
    if tag=='nafnet':
        arch=json.loads((ROOT/'architectures/nafnet/config.json').read_text())
        net.network=module.NAFNetCore({'nafnet_architecture':arch})
    elif spec['family']=='restormer': net.network=module.RestormerCore()
    else: net.network=module.EDVRCore()
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    net.load_state_dict(state['state_dict'],strict=True)
    if sum(p.numel() for p in net.parameters())!=spec['parameters']: raise ValueError('Unexpected model capacity')
    return net.to(device).eval()

def predict(net, pixels, device='cpu'):
    """Return unclipped float32 RGB from uint8 HWC still or eight-frame THWC burst."""
    a=np.asarray(pixels)
    if a.dtype!=np.uint8: raise ValueError('Input must be uint8 RGB, encoded values from 0 to 255')
    if a.ndim==3 and a.shape[-1]==3: a=np.repeat(a[None],8,axis=0)
    if a.ndim!=4 or a.shape[0]!=8 or a.shape[-1]!=3: raise ValueError('Expected (8,H,W,3) burst or (H,W,3) still')
    h,w=a.shape[1:3]
    if min(h,w)<32: raise ValueError('Each spatial dimension must be at least 32 pixels')
    x=torch.from_numpy(a.copy()).permute(0,3,1,2).float().div_(255)[None].to(device)
    # NAFNet has four downsampling stages; other families also accept multiples of 32.
    ph,pw=(-h)%32,(-w)%32
    if ph or pw:
        x=F.pad(x.reshape(8,3,h,w),(0,pw,0,ph),mode='reflect').reshape(1,8,3,h+ph,w+pw)
    with torch.inference_mode(): result=net(x)[0,:,:h,:w]
    a=result.float().cpu().permute(1,2,0).numpy().copy()
    if not np.isfinite(a).all(): raise ValueError('Model produced nonfinite output')
    return a
