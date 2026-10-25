"""One paired evaluation pass per model after all three selections freeze."""
from pathlib import Path
from collections import defaultdict
import csv,json,time
import numpy as np
import torch
from skimage.metrics import structural_similarity
from PIL import Image,ImageDraw
from campaign_common import read,write,tensors,seed,utc
from campaign_models import Restoration,digest
R=Path(__file__).resolve().parents[1]

def metrics(raw,target):
    a=np.clip(raw,0,1);error=a-target
    return {'ssim':float(structural_similarity(a,target,data_range=1,channel_axis=2)),
        'psnr_db':float(-10*np.log10(max(np.mean(error.astype(np.float64)**2),1e-12))),
        'mae':float(np.abs(error).mean()),'clipping_fraction':float(np.mean((raw<0)|(raw>1)))}

def aggregate(rows):
    groups=defaultdict(list);ids=defaultdict(list)
    for row in rows:groups[(row['global_identity'],row['acquisition_condition'])].append(row)
    for (gid,_),rs in groups.items():ids[gid].append({k:np.mean([r[k] for r in rs]) for k in ['ssim','psnr_db','mae']})
    return {'records':len(rows),'identities':len(ids),
        **{k:float(np.mean([np.mean([v[k] for v in vals]) for vals in ids.values()])) for k in ['ssim','psnr_db','mae']}}

def csvwrite(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    cfg=read(R/'config/CAMPAIGN_v1.json');seed(1337,cfg)
    frozen=read(R/'results/ALL_THREE_SELECTIONS_FROZEN.json');assert len(frozen['runs'])==3
    data=Path(read(R/'audit/DATA_PREFLIGHT.json')['data_root'])
    rows=[r for r in read(data/cfg['restoration_manifest'])['rows'] if r['role']=='eval']
    assert len(rows)==329 and len({r['global_identity'] for r in rows})==72
    out=R/'results/evaluation';out.mkdir(exist_ok=True);start=time.monotonic()
    with (out/'INFERENCE_STARTED.json').open('x') as f:
        json.dump({'UTC':utc(),'all_three_freeze_sha256':digest(R/'results/ALL_THREE_SELECTIONS_FROZEN.json')},f)
    summaries=[];all_scores=[]
    for fit in frozen['runs']:
        run=fit['run_id'];mode=run.split('_')[2];cp=R/'results/restoration'/run/'best.pth'
        assert digest(cp)==fit['best_sha256']
        net=Restoration('nafnet',mode,R,cfg).cuda().eval()
        net.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['state_dict'],strict=True)
        array=np.empty((329,256,256,3),dtype=np.float32);scores=[]
        with torch.inference_mode():
            for i,row in enumerate(rows):
                x,y=tensors(data,row,'cuda');raw=net(x)[0].cpu().permute(1,2,0).numpy();assert np.isfinite(raw).all()
                array[i]=raw;target=y[0].cpu().permute(1,2,0).numpy()
                scores.append({'run_id':run,**{k:row[k] for k in ['sample_uid','global_identity','category']},
                    'acquisition_condition':row['condition'],**metrics(raw,target)})
        p=out/(run+'_float32.npy');np.save(p,array)
        # Independent CPU replay verifies metrics without repeating inference.
        max_error=0.
        for i,row in enumerate(rows):
            target=np.load(data/row['target_npy']).astype(np.float32)/255
            for key,v in metrics(array[i],target).items():
                error=abs(v-scores[i][key]);assert error<2e-6;max_error=max(max_error,error)
        scope=[('All',scores)] + [(k,[s for s in scores if s['acquisition_condition']==k]) for k in sorted({s['acquisition_condition'] for s in scores})]
        for name,subset in scope:summaries.append({'run_id':run,'cohort':name,**aggregate(subset)})
        # Fixed first/middle/last evaluation records, never chosen by score.
        canvas=Image.new('RGB',(768,3*280),'white');draw=ImageDraw.Draw(canvas)
        for j,i in enumerate([0,len(rows)//2,len(rows)-1]):
            row=rows[i];x=np.load(data/row['input_npy']);y=np.load(data/row['target_npy'])
            for k,(a,label) in enumerate([(x[4],'Camera frame5'),(np.rint(np.clip(array[i],0,1)*255).astype(np.uint8),run),(y,'Digital reference')]):
                canvas.paste(Image.fromarray(a),(k*256,j*280+24));draw.text((k*256+3,j*280+3),label,fill='black')
        canvas.save(out/(run+'_fixed_examples.png'))
        write(out/(run+'_VERIFIED.json'),{'UTC':utc(),'checkpoint_sha256':digest(cp),'float32_sha256':digest(p),
            'records':329,'identities':72,'CPU_metric_replay_max_error':max_error,'heldout_inference_passes':1,
            'selection_precedes_inference':True,'evaluation_did_not_change_selection':True})
        all_scores+=scores;del net,array;torch.cuda.empty_cache()
    csvwrite(out/'PAIRED_PER_RECORD.csv',all_scores);csvwrite(out/'PAIRED_SUMMARY.csv',summaries)
    write(out/'SUMMARY.json',{'UTC':utc(),'scope':'Historically reused 329-record/72-identity paired digital-reference evaluation; no new heldout or condition-generalization claim.',
        'overall':[s for s in summaries if s['cohort']=='All'],'all_three_selected_before_inference':True,
        'heldout_used_for_selection':False,'elapsed_seconds':time.monotonic()-start})
    print('POST_FREEZE_EVALUATION_COMPLETE',json.dumps([s for s in summaries if s['cohort']=='All']),flush=True)

if __name__=='__main__':main()
