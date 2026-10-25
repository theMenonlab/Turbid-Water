"""Portable versions of the existing EDVR/Restormer input-control graphs."""
from pathlib import Path
import hashlib, importlib, importlib.util, json, sys, types
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import torchvision

CODE = Path(__file__).resolve().parent
EXTERNAL = CODE / 'vendor/external'

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def state_hash(net):
    h = hashlib.sha256()
    for key, value in net.state_dict().items():
        h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

class EDVRCore(nn.Module):
    def __init__(self):
        super().__init__()
        for name in list(sys.modules):
            if name == 'basicsr' or name.startswith('basicsr.'): del sys.modules[name]
        repo = EXTERNAL / 'BasicSR'
        module = types.ModuleType('basicsr'); module.__path__ = [str(repo / 'basicsr')]; module.__package__ = 'basicsr'
        sys.modules['basicsr'] = module
        arch = importlib.import_module('basicsr.archs.edvr_arch')
        self.network = arch.EDVR(num_in_ch=3, num_out_ch=3, num_feat=32, num_frame=8,
                                deformable_groups=4, num_extract_block=2, num_reconstruct_block=3,
                                hr_in=True, with_predeblur=True, with_tsa=True)
    def forward(self, burst): return self.network(burst)

def execute_stage(module, x, policy):
    if policy == 'stages':
        return checkpoint(module, x, use_reentrant=False, preserve_rng_state=True)
    if policy == 'blocks':
        for layer in module:
            x = checkpoint(layer, x, use_reentrant=False, preserve_rng_state=True)
        return x
    assert policy == 'none'
    return module(x)

def decode(net, embedded, policy='none'):
    enc1 = execute_stage(net.encoder_level1, embedded, policy)
    enc2 = execute_stage(net.encoder_level2, net.down1_2(enc1), policy)
    enc3 = execute_stage(net.encoder_level3, net.down2_3(enc2), policy)
    latent = execute_stage(net.latent, net.down3_4(enc3), policy)
    dec3 = execute_stage(net.decoder_level3, net.reduce_chan_level3(torch.cat([net.up4_3(latent), enc3], 1)), policy)
    dec2 = execute_stage(net.decoder_level2, net.reduce_chan_level2(torch.cat([net.up3_2(dec3), enc2], 1)), policy)
    dec1 = execute_stage(net.decoder_level1, torch.cat([net.up2_1(dec2), enc1], 1), policy)
    return net.output(execute_stage(net.refinement, dec1, policy))

class RestormerCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.checkpoint_policy = 'none'
        module = load_module('campaign_restormer_arch', EXTERNAL / 'Restormer/basicsr/models/archs/restormer_arch.py')
        self.network = module.Restormer(inp_channels=24, out_channels=3)
    def forward(self, burst):
        n, _, _, h, w = burst.shape
        policy = self.checkpoint_policy if self.training and torch.is_grad_enabled() else 'none'
        return decode(self.network, self.network.patch_embed(burst.reshape(n, 24, h, w)), policy) + burst.mean(1)

class Restoration(nn.Module):
    def __init__(self, family, mode, root, config):
        super().__init__(); self.family = family; self.mode = mode
        torch.manual_seed(config['restoration_seed'])
        self.network = EDVRCore() if family == 'edvr' else RestormerCore()
        asset = config['assets'][family]; path = Path(root) / asset['path']
        assert digest(path) == asset['sha256']
        saved = torch.load(path, map_location='cpu', weights_only=False)
        self.network.load_state_dict(saved['state_dict'], strict=True)
        assert sum(v.numel() for v in self.parameters()) == (900019 if family == 'edvr' else 26135716)
    def forward(self, burst):
        assert burst.ndim == 5 and burst.shape[1:3] == (8, 3)
        if self.mode == 'single': burst = burst[:, 4:5].repeat(1, 8, 1, 1, 1)
        elif self.mode == 'mean': burst = burst.mean(1, keepdim=True).repeat(1, 8, 1, 1, 1)
        return self.network(burst)

class Classifier(nn.Module):
    def __init__(self, root, config, seed):
        super().__init__(); torch.manual_seed(seed)
        asset = config['assets']['resnet50']; path = Path(root) / asset['path']
        assert digest(path) == asset['sha256']
        self.backbone = torchvision.models.resnet50(weights=None)
        self.backbone.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        self.backbone.fc = nn.Identity(); self.head = nn.Linear(2048, 5)
        assert sum(v.numel() for v in self.parameters()) == 23518277
    def set_stage(self, full):
        for p in self.backbone.parameters(): p.requires_grad_(full)
        for p in self.head.parameters(): p.requires_grad_(True)
        self.train()
        for m in self.backbone.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm): m.eval()
    def forward(self, x):
        n, k, c, h, w = x.shape; a = x.reshape(n * k, c, h, w)
        z = checkpoint(self.backbone, a, use_reentrant=False) if self.training and next(self.backbone.parameters()).requires_grad else self.backbone(a)
        return self.head(z.reshape(n, k, 2048).mean(1))
