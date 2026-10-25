"""Sequential fits under the existing global GPU guard; stop on any failure."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
import torch
from campaign_common import read,write,utc
from campaign_models import digest
from verify_fit import verify
R=Path(__file__).resolve().parents[1]

def main(args):
    torch.set_num_threads(2)
    cfg=read(R/'config/CAMPAIGN_v1.json');runs=[v['run_id'] for v in cfg['restoration_runs']]
    for rel,sha in read(R/'audit/SCIENCE_FILE_SHA256.json')['files'].items():assert digest(R/rel)==sha,rel
    status={'UTC':utc(),'PID':os.getpid(),'phase':args.phase,'runs':runs,'protocol_sha256':digest(R/'config/CAMPAIGN_v1.json')}
    write(R/'audit/WORKFLOW_OWNER.json',status)
    expected=128 if args.phase=='gate' else 65536
    if args.phase=='full':assert read(R/'audit/ALL_THREE_GATES_PASSED.json')['all_finite']
    results=[]
    for run in runs:
        out=R/'results/restoration'/run
        assert not (out/'COMPLETE.json').exists(),'No stale completed fit may be reused'
        if args.phase=='gate':assert not (out/'resume.pth').exists(),'Fresh gate required'
        else:verify(run,128)
        cmd=[sys.executable,'-B','-u',str(R/'code/guarded_train.py'),'--run',run]
        if args.phase=='gate':cmd+=['--stop-at','128']
        status.update(UTC=utc(),current_run=run,phase=args.phase);write(R/'audit/WORKFLOW_STATUS.json',status)
        started=time.monotonic()
        with (R/'logs'/(run+'_'+args.phase+'.log')).open('a') as log:
            proc=subprocess.run(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT)
        assert proc.returncode==(75 if args.phase=='gate' else 0),f'{run} failed: {proc.returncode}'
        receipt=verify(run,expected);receipt['stage_wall_seconds']=time.monotonic()-started;results.append(receipt)
        print('VERIFIED',json.dumps(receipt),flush=True)
    if args.phase=='gate':
        elapsed=sum(v['stage_wall_seconds'] for v in results)
        # Conservative measured projection includes complete step-zero tuning,
        # data loading, optimizer checks and checkpoint save overhead.
        rates=[]
        for v in results:
            out=R/'results/restoration'/v['run_id'];curve=[json.loads(x) for x in (out/'training_curve.jsonl').read_text().splitlines()]
            train_seconds=sum(x['seconds'] for x in curve)
            overhead=max(0,v['stage_wall_seconds']-train_seconds)
            rates.append({'run_id':v['run_id'],'measured_guarded_train_seconds_per_update':train_seconds/128,
                'measured_gate_wall_seconds':v['stage_wall_seconds'],'measured_non_update_gate_seconds':overhead,
                'projected_full_seconds':train_seconds/128*65536+overhead*35})
        write(R/'audit/ALL_THREE_GATES_PASSED.json',{'UTC':utc(),'all_finite':True,'gate_updates_each':128,
            'measured_synthetic_each':[96,32],'results':results,'ETA_measurements':rates,
            'projected_total_seconds':sum(v['projected_full_seconds'] for v in rates),
            'projection_note':'Training-step timings exclude loader/log overhead; non-update gate time multiplied by35 covers tuning/checkpoint/process overhead conservatively. Reestimate using full-run wall progress.'})
    else:
        assert len({r['initial_state_sha256'] for r in results})==1
        write(R/'results/ALL_THREE_SELECTIONS_FROZEN.json',{'UTC':utc(),'runs':results,'heldout_used':False})
        # Existing fitting code never evaluates the held-out cohort. Evaluation
        # can run only after all three selected models have frozen receipts.
        evaluated=subprocess.run([sys.executable,'-B','-u',str(R/'code/evaluate_frozen.py')],cwd=R)
        assert evaluated.returncode==0,'Post-freeze evaluation failed; no automatic retry'
        write(R/'audit/ALL_THREE_COMPLETE.json',{'UTC':utc(),'fits':results,'evaluation':read(R/'results/evaluation/SUMMARY.json')})
    status.update(UTC=utc(),phase=args.phase+'_complete');write(R/'audit/WORKFLOW_STATUS.json',status)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['gate','full'],required=True)
    try:main(p.parse_args())
    except BaseException as exc:
        import traceback
        write(R/'audit/WORKFLOW_STOPPED.json',{'UTC':utc(),'error':repr(exc),'traceback':traceback.format_exc(),'automatic_retry':False})
        raise
