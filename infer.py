"""Restore one RGB image or a released eight-frame NumPy burst."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import torch
from nrl_water import load_model, model_index, predict

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='restormer_RGB_mean_mixed_seed1337',choices=[r['model_id'] for r in model_index()])
    p.add_argument('--weights-root',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True,help='New .png output; a float .npy file is also saved')
    p.add_argument('--device',default='cpu');p.add_argument('--max-side',type=int,default=256,help='Still images only; 0 retains native size')
    args=p.parse_args();torch.set_num_threads(2)
    if args.output.suffix.lower()!='.png':p.error('--output must end in .png')
    if args.output.exists() or args.output.with_suffix('.npy').exists():p.error('Output already exists')
    if args.input.suffix.lower()=='.npy': pixels=np.load(args.input,allow_pickle=False)
    else:
        im=Image.open(args.input).convert('RGB')
        if args.max_side and max(im.size)>args.max_side:
            factor=args.max_side/max(im.size);im=im.resize(tuple(round(v*factor) for v in im.size),Image.Resampling.LANCZOS)
        pixels=np.asarray(im)
    net=load_model(args.model,args.weights_root,args.device)
    out=predict(net,pixels,args.device)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.save(args.output.with_suffix('.npy'),out)
    Image.fromarray(np.rint(np.clip(out,0,1)*255).astype(np.uint8)).save(args.output)
    print(args.output.resolve())

if __name__=='__main__':main()
