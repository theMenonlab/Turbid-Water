"""Evaluate a selected model on the frozen paired evaluation cohort."""
from pathlib import Path
from collections import defaultdict
import argparse,csv,json
import numpy as np
import torch
from skimage.metrics import structural_similarity
from nrl_water import load_model,predict,sha256

def aggregate(rows,key):
    groups=defaultdict(lambda:defaultdict(list))
    for r in rows:groups[r['global_identity']][r['condition']].append(r[key])
    return float(np.mean([np.mean([np.mean(x) for x in g.values()]) for g in groups.values()]))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--weights-root',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--limit',type=int,help='Smoke check only; omit for the full 329-record benchmark')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    manifest=a.data_root/'data/RESTORATION_PAIRED_MANIFEST_v1.json'
    if sha256(manifest)!='12cedc7d910c922c642693959dae31e962ce84c4aa0c09dafc97a5f76c36cb05':raise ValueError('Unexpected paired manifest')
    rows=[r for r in json.loads(manifest.read_text())['rows'] if r['role']=='eval']
    if a.limit is not None:
        if a.limit<1:p.error('--limit must be positive')
        rows=rows[:a.limit]
    net=load_model(a.model,a.weights_root,a.device);scores=[]
    for i,r in enumerate(rows):
        for field in ['input_npy','target_npy']:
            if sha256(a.data_root/r[field])!=r[field+'_sha256']:raise ValueError('Array checksum failed: '+r[field])
        x=np.load(a.data_root/r['input_npy'],allow_pickle=False)
        y=np.load(a.data_root/r['target_npy'],allow_pickle=False).astype(np.float32)/255
        z=predict(net,x,a.device).clip(0,1);mse=np.mean((z.astype(np.float64)-y.astype(np.float64))**2)
        scores.append({k:r[k] for k in ['sample_uid','global_identity','category','condition']}|dict(
            ssim=float(structural_similarity(z,y,channel_axis=2,data_range=1,win_size=7,gaussian_weights=False,use_sample_covariance=True)),
            psnr_db=float(-10*np.log10(mse)) if mse else float('inf'),mae=float(np.abs(z-y).mean())))
        print(i+1,'/',len(rows),r['sample_uid'],flush=True)
    with (a.output/'per_record.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(scores[0]));w.writeheader();w.writerows(scores)
    summary=dict(model=a.model,records=len(scores),identities=len({r['global_identity'] for r in scores}),
        smoke_check=a.limit is not None,aggregation='recordings within condition, conditions within identity, then identities equally',
        **{k:aggregate(scores,k) for k in ['ssim','psnr_db','mae']})
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
