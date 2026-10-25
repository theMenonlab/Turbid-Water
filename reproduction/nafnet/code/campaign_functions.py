import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF

def ssim_value(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    kernel = 11
    padding = kernel // 2
    mu_x = F.avg_pool2d(prediction, kernel, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, kernel, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(prediction * prediction, kernel, stride=1, padding=padding) - mu_x.square()
    sigma_y = F.avg_pool2d(target * target, kernel, stride=1, padding=padding) - mu_y.square()
    sigma_xy = F.avg_pool2d(prediction * target, kernel, stride=1, padding=padding) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2) + 1e-12
    )
    return score.mean()

def fidelity_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    pixel_loss: str = "charbonnier",
    ssim_weight: float = 0.05,
    gradient_weight: float = 0.05,
    pixel_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if pixel_loss == "l1":
        pixel = F.l1_loss(prediction, target)
    elif pixel_loss == "charbonnier":
        pixel = torch.sqrt((prediction - target).square() + 1e-6).mean()
    else:
        raise ValueError(pixel_loss)
    ssim_loss = 1.0 - ssim_value(prediction, target)
    pred_dx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    pred_dy = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    gradient = 0.5 * (F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy))
    total = pixel_weight * pixel + ssim_weight * ssim_loss + gradient_weight * gradient
    return total, {
        "pixel_loss": float(pixel.detach()),
        "ssim_loss": float(ssim_loss.detach()),
        "gradient_loss": float(gradient.detach()),
    }

def classifier_tensor(a,r,seed=1337,epoch=1,training=True):
 x=torch.from_numpy(a.copy()).permute(0,3,1,2).float()/255
 if training:
  key=int.from_bytes(hashlib.sha256(f'{seed}|{epoch}|{r["global_identity"]}'.encode()).digest()[:8],'big')%(2**63)
  with torch.random.fork_rng(devices=[]):
   torch.random.default_generator.manual_seed(key);top,left,h,w=T.RandomResizedCrop.get_params(x,scale=(.9,1.),ratio=(.95,1.05));flip=bool(torch.rand(1)<.5);order,b,c,s,hue=T.ColorJitter.get_params((.9,1.1),(.9,1.1),(.9,1.1),(-.02,.02))
  x=TF.resized_crop(x,top,left,h,w,[224,224],antialias=True)
  if flip:x=TF.hflip(x)
  for op in order.tolist():x=TF.adjust_brightness(x,b) if op==0 else TF.adjust_contrast(x,c) if op==1 else TF.adjust_saturation(x,s) if op==2 else TF.adjust_hue(x,hue)
  x=x.clamp(0,1)
 else:x=TF.resize(x,[224,224],antialias=True)
 x=TF.normalize(x,[.485,.456,.406],[.229,.224,.225]);assert torch.isfinite(x).all();return x
