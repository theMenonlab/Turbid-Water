"""TRAIN-informed encoded-RGB scatter augmentation, not a calibrated water solver."""
from pathlib import Path
import json
import cv2
import numpy as np
D=Path(__file__).resolve().parents[1]
def smooth_field(rng):
 # Same medium field shared by every frame. No outdoor horizon/depth prior.
 coarse=rng.random((5,5)).astype(np.float32)
 return np.clip(cv2.resize(coarse,(256,256),interpolation=cv2.INTER_CUBIC),0,1)
def render(clean_uint8,seed,force=None):
 assert clean_uint8.shape==(256,256,3) and clean_uint8.dtype==np.uint8
 rng=np.random.default_rng(seed);stats=json.loads((D/'audit/TRAIN_IMAGE_DOMAIN_INITIAL_STATISTICS.json').read_text())['conditions'];condition=stats[int(rng.integers(len(stats)))];kind='identity' if rng.random()<.10 else 'light' if rng.random()<.30 else 'scatter'
 if force is not None:kind=force
 J=clean_uint8.astype(np.float32)/255
 if kind=='identity':return np.repeat(clean_uint8[None],8,axis=0),{'seed':seed,'kind':kind,'one_underlying_scene':True,'condition_proxy':condition['condition']}
 a=rng.random();gain=np.asarray(condition['effective_gain_quantiles'][0])*(1-a)+np.asarray(condition['effective_gain_quantiles'][2])*a;gain=np.clip(gain,.035,.90)
 a=rng.random();offset=np.asarray(condition['offset_quantiles'][0])*(1-a)+np.asarray(condition['offset_quantiles'][2])*a;offset=np.clip(offset,.01,.75);A=np.clip(offset/np.maximum(1-gain,.1),.01,.95)
 if kind=='light':gain=rng.uniform(.75,.98,size=3)
 direct=-np.log(gain).astype(np.float32);back=(direct*rng.uniform(.8,1.2,size=3)).astype(np.float32)
 field=1+.15*(2*smooth_field(rng)-1);air=np.clip(A[None,None,:]+.012*(smooth_field(rng)[:,:,None]-.5),0,1)
 sigma=float(rng.uniform(.0,.5) if kind=='light' else rng.uniform(.2,2.0));blur=cv2.GaussianBlur(J,(0,0),max(sigma,.01),borderType=cv2.BORDER_REFLECT101)
 noise=float(np.clip(.5*condition['temporal_variation_quantiles'][1],.001,.008));phase=rng.uniform(0,2*np.pi);omega=rng.uniform(.12,.35);output=[]
 for frame in range(8):
  drift=1+.02*np.sin(phase+omega*(frame-4));path=field[:,:,None]*drift;transmission=np.exp(-direct[None,None,:]*path);backscatter=air*(1-np.exp(-back[None,None,:]*path));image=blur*transmission+backscatter;image+=rng.normal(0,noise,image.shape).astype(np.float32);output.append(np.rint(np.clip(image,0,1)*255).astype(np.uint8))
 burst=np.stack(output);assert burst.shape==(8,256,256,3)
 return burst,{'seed':seed,'kind':kind,'condition_proxy':condition['condition'],'effective_direct_coeff_RGB':direct.tolist(),'distinct_effective_backscatter_coeff_RGB':back.tolist(),'airlight_encoded_RGB':A.tolist(),'blur_sigma_px':sigma,'noise_sigma_encoded':noise,'one_underlying_scene':True,'temporal_medium':'same spatialfield, smooth2percentdrift, independentlow sensornoise','physical_scope':'Encoded image-domaincoefficients only; no extinction inversion, bubbles, refraction or scene-depth estimates'}
