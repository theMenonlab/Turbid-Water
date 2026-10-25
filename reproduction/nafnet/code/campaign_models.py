"""Official NAFNet with only 8-frame stem and RGB residual adaptation."""
from pathlib import Path
import hashlib, importlib.util, logging, sys, types
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

CODE = Path(__file__).resolve().parent

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

def official_module():
    name = 'basicsr.models.archs.NAFNet_arch'
    if name in sys.modules: return sys.modules[name]
    # Import just the official architecture dependencies, avoiding BasicSR's
    # dataset/trainer package discovery. The logger is unused by NAFNet.
    for package in ['basicsr', 'basicsr.models', 'basicsr.models.archs']:
        m = types.ModuleType(package); m.__path__ = []; sys.modules[package] = m
    utils = types.ModuleType('basicsr.utils')
    utils.get_root_logger = logging.getLogger
    sys.modules['basicsr.utils'] = utils
    for short in ['arch_util', 'local_arch', 'NAFNet_arch']:
        full = 'basicsr.models.archs.' + short
        spec = importlib.util.spec_from_file_location(full, CODE / 'vendor/NAFNet' / (short + '.py'))
        module = importlib.util.module_from_spec(spec); sys.modules[full] = module
        spec.loader.exec_module(module)
    return sys.modules[name]

class NAFNetCore(nn.Module):
    def __init__(self, config):
        super().__init__()
        a = config['nafnet_architecture']
        self.network = official_module().NAFNet(img_channel=3, width=a['width'],
            middle_blk_num=a['middle_blk_num'], enc_blk_nums=a['enc_blk_nums'], dec_blk_nums=a['dec_blk_nums'])
        self.network.intro = nn.Conv2d(24, a['width'], 3, padding=1, bias=True)

    def forward(self, burst):
        n, _, _, h, w = burst.shape
        net = self.network
        inp = net.check_image_size(burst.reshape(n, 24, h, w))
        x = net.intro(inp); encs = []
        for encoder, down in zip(net.encoders, net.downs):
            x = encoder(x); encs.append(x); x = down(x)
        x = net.middle_blks(x)
        for decoder, up, enc_skip in zip(net.decoders, net.ups, encs[::-1]):
            x = up(x); x = x + enc_skip; x = decoder(x)
        x = net.ending(x)
        x = x + inp.reshape(n, 8, 3, *inp.shape[-2:]).mean(1)
        return x[:, :, :h, :w]

class Restoration(nn.Module):
    def __init__(self, family, mode, root, config):
        super().__init__()
        assert family == 'nafnet' and mode in ['single', 'mean', 'burst']
        self.family = family; self.mode = mode
        torch.manual_seed(config['restoration_seed'])
        self.network = NAFNetCore(config)
        asset = config['assets'][family]; path = Path(root) / asset['path']
        assert digest(path) == asset['sha256']
        self.network.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['state_dict'], strict=True)
        assert sum(p.numel() for p in self.parameters()) == config['nafnet_architecture']['parameters']

    def forward(self, burst):
        assert burst.ndim == 5 and burst.shape[1:3] == (8, 3)
        if self.mode == 'single': burst = burst[:, 4:5].repeat(1, 8, 1, 1, 1)
        elif self.mode == 'mean': burst = burst.mean(1, keepdim=True).repeat(1, 8, 1, 1, 1)
        return self.network(burst)
