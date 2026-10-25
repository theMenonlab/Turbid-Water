from pathlib import Path
import ast, json, shutil, time
import numpy as np
import torch
from campaign_common import read, write, roles, validate_inputs, array, utc
from campaign_models import digest, Restoration, state_hash

R=Path(__file__).resolve().parents[1]
BASE=Path('/media/al/ingold_20260515/nrl_final_paper_v3_20261014')
DATA=Path('/home/al/Documents/ChatGPT/NRL/turbid_water/final_paper_local_backup_20261014')

def main():
    started=time.monotonic();cfg=read(R/'config/CAMPAIGN_v1.json')
    for name,expected in cfg['manifest_sha256'].items():
        assert digest(DATA/name)==expected and digest(BASE/name)==expected,name
    m=read(DATA/cfg['restoration_manifest']);rr=roles(m['rows'])
    assert m['illustrations_excluded'] and m['global_roles_unchanged'] and m['classification_only133_excluded_from_restoration']
    assert m['new_pairs']==260
    assert [len(rr[k]) for k in ['train','tune','eval']]==[2648,365,329]
    assert [len({r['global_identity'] for r in rr[k]}) for k in ['train','tune','eval']]==[580,76,72]
    for role,rows in rr.items():
        for row in rows:
            assert not row.get('full_current_presentation_reservation',False),row['sample_uid']
            assert not row.get('new_model_reserved_evaluation',False),row['sample_uid']
    validate_inputs(DATA,m['rows'])
    for row in m['rows']:
        array(DATA,row,'input_npy');array(DATA,row,'target_npy')
    natural=read(DATA/cfg['natural_manifest'])['records']
    assert len([r for r in natural if r['experiment_role']=='train'])==96
    for row in natural:
        if row['experiment_role']=='train':
            p=DATA/row['crop'];a=np.load(p)
            assert a.shape==(256,256,3) and a.dtype==np.uint8
    draws=read(R/cfg['draws']);assert digest(R/cfg['draws'])==cfg['draws_sha256']
    assert set(draws['measured_UIDs'])<={r['sample_uid'] for r in rr['train']}
    assert {d['image_id'] for d in draws['synthetic']}<={r['image_id'] for r in natural if r['experiment_role']=='train'}
    assert digest(R/'code/water_generator.py')==cfg['generator_sha256']
    assert digest(R/'audit/TRAIN_IMAGE_DOMAIN_INITIAL_STATISTICS.json')==cfg['generator_stats_sha256']
    torch.set_num_threads(2);hashes={}
    for mode in ['single','mean','burst']:
        net=Restoration('nafnet',mode,R,cfg);hashes[mode]=state_hash(net)
    assert len(set(hashes.values()))==1
    for p in (R/'code').glob('*.py'):ast.parse(p.read_text())
    # Under 2 GiB for all initializers/best/final resume states and evaluation
    # arrays; require a generous margin without copying the paired dataset.
    assert shutil.disk_usage(R).free>5*1024**3
    receipt={'UTC':utc(),'data_root':str(DATA),'downloaded_final_release':str(BASE),
        'manifest_sha256':cfg['manifest_sha256'],'same_local_and_downloaded_final_manifests':True,
        'all_3342_paired_input_target_hashes_shapes_dtypes_verified':True,
        'roles':m['roles'],'new_validated_pairs':260,'classification_only_133_excluded':True,
        'frozen_global_roles_reservations_preserved':True,'natural_train_crops':96,
        'draws_sha256':cfg['draws_sha256'],'initial_state_sha256_per_mode':hashes,
        'internal_free_bytes':shutil.disk_usage(R).free,'external_free_bytes':shutil.disk_usage(BASE).free,
        'elapsed_seconds':time.monotonic()-started,'GPU_training_performed':False}
    write(R/'audit/DATA_PREFLIGHT.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
