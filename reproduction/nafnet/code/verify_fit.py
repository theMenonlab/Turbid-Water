"""CPU verification of exact draws, finite states and tuning-only selection."""
from pathlib import Path
import json
import numpy as np
import torch
from campaign_common import read, write, identity_mean, utc
from campaign_models import digest
R=Path(__file__).resolve().parents[1]

def verify(run,expected):
    cfg=read(R/'config/CAMPAIGN_v1.json');sha=digest(R/'config/CAMPAIGN_v1.json')
    out=R/'results/restoration'/run;s=torch.load(out/'resume.pth',map_location='cpu',weights_only=False)
    assert s['updates']==expected and s['protocol_sha256']==sha and s['run_id']==run
    assert s['synthetic_updates']==expected//4 and s['measured_updates']==expected-expected//4
    assert all(bool(torch.isfinite(v).all()) for v in s['model'].values())
    states=s['optimizer']['state'];assert len(states)==len(s['model'])
    for v in states.values():
        assert int(v['step'])==expected
        assert all(bool(torch.isfinite(a).all()) for a in v.values() if torch.is_tensor(a))
    curve=[json.loads(line) for line in (out/'training_curve.jsonl').read_text().splitlines()]
    assert [v['step'] for v in curve]==list(range(1,expected+1))
    assert all(np.isfinite(v['loss']) and v['seconds']>0 for v in curve)
    draws=read(R/cfg['draws']);i=j=0
    for row in curve:
        if row['step']%4==0:
            assert row['kind']=='synthetic' and row['identity']=='natural:'+str(draws['synthetic'][j]['image_id']);j+=1
        else:
            assert row['kind']=='measured' and row['identity']==draws['measured_UIDs'][i];i+=1
    data=Path(read(R/'audit/DATA_PREFLIGHT.json')['data_root'])
    tune=[r for r in read(data/cfg['restoration_manifest'])['rows'] if r['role']=='tune']
    tunes=[read(out/f'tune_{step:06d}.json') for step in cfg['restoration_tune_steps'] if step<=expected]
    for t in tunes:
        assert [v['sample_uid'] for v in t['scores']]==[v['sample_uid'] for v in tune]
        assert np.isfinite([v['ssim'] for v in t['scores']]).all()
        assert identity_mean(t['scores'],'ssim')==t['identity_condition_ssim']
    best=max(tunes,key=lambda t:t['identity_condition_ssim'])
    assert s['selected_step']==best['step'] and s['best_tune_ssim']==best['identity_condition_ssim']
    cp=torch.load(out/'best.pth',map_location='cpu',weights_only=False)
    assert cp['step']==best['step'] and cp['protocol_sha256']==sha
    assert all(bool(torch.isfinite(v).all()) for v in cp['state_dict'].values())
    assert not (out/'NUMERICAL_FAILURE.json').exists() and not (out/'RUN_EXCEPTION.json').exists()
    if expected==65536:
        complete=read(out/'COMPLETE.json');assert complete==read(out/'SELECTION.json')
        assert complete['selected_checkpoint_sha256']==digest(out/'best.pth')
        assert complete['selected_step']==best['step'] and complete['updates']==expected
    receipt={'UTC':utc(),'run_id':run,'updates':expected,'weights_and_optimizer_finite':True,
        'optimizer_parameter_states':len(states),'curve_complete_exact_draws_verified':True,
        'tuning_records_per_milestone':365,'tuning_milestones':len(tunes),
        'earliest_tune_maximum_verified':True,'selected_step':best['step'],'best_tune_ssim':best['identity_condition_ssim'],
        'best_sha256':digest(out/'best.pth'),'final_resume_sha256':digest(out/'resume.pth'),
        'initial_state_sha256':s['initial_state_sha256'],'protocol_sha256':sha,'heldout_used_for_selection':False}
    write(out/f'VERIFIED_{expected:06d}.json',receipt);return receipt
