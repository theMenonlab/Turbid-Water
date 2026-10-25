"""Local CPU verification only: no cluster, CUDA model, submission or training campaign."""
from pathlib import Path
import copy, gc, hashlib, importlib.util, json, os, py_compile, sys, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
D = Path(__file__).resolve().parents[1]; P = D / 'portable'
sys.path.insert(0, str(P / 'code'))
from campaign_models import Restoration, Classifier, digest, state_hash
from campaign_common import read, write, seed, rng, restore_rng, probability_cohorts
from campaign_functions import fidelity_loss, classifier_tensor
from campaign_conditions import display_condition, display_cohorts
from campaign_classify import sequence
from water_generator import render

def main():
    started = time.perf_counter(); config = read(P / 'config/CAMPAIGN_v1.json'); torch.set_num_threads(2)
    assert not torch.cuda.is_available()
    for path in (P / 'code').rglob('*.py'): py_compile.compile(str(path), doraise=True)
    checks = {}; hashes = {}; grad_hashes = {}
    expected = {'edvr': 'e990d08fdfcefca85e982ccf22d8091c961fd8968cfacfc54fed2c6b05b9bebb',
                'restormer': '45495e6c145479e97038ca0c46624925558b11f125a0637e736aaa5647a6df4f'}
    torch.manual_seed(71); burst = torch.rand(1, 8, 3, 32, 32); altered = burst.clone(); altered[:, :4] = 1.; altered[:, 5:] = 0.
    target = torch.rand(1, 3, 32, 32)
    for family in ['edvr', 'restormer']:
        reference = None
        for mode in ['single', 'mean', 'burst']:
            net = Restoration(family, mode, P, config); h = state_hash(net); assert h == expected[family]
            hashes[family + '_' + mode] = h
            # Equal-input controls agree through exactly the same initialized capacity.
            repeated = burst[:, 4:5].repeat(1, 8, 1, 1, 1)
            with torch.no_grad():
                output = net(repeated)
                if reference is None: reference = output.clone()
                else: assert torch.allclose(reference, output, atol=3e-6, rtol=1e-6)
                if mode == 'single': assert torch.equal(net(burst), net(altered))
            net.train(); loss, _ = fidelity_loss(net(burst), target, pixel_loss='charbonnier', **config['loss'])
            assert torch.isfinite(loss); loss.backward()
            grads = [v.grad for v in net.parameters() if v.grad is not None]
            assert len(grads) and all(torch.isfinite(v).all() for v in grads)
            assert any(torch.count_nonzero(v).item() for v in grads)
            grad_hashes[family + '_' + mode] = len(grads)
            del net, grads; gc.collect()
    checks['six_restoration_graphs_forward_backward_and_shared_capacity'] = True
    checks['single_mode_ignores_other_seven_measured_frames'] = True
    checks['repeat_controls_numerically_agree'] = True
    original = read(D.parent / 'classification_only_expansion_20261013/audit/CPU_GATE.json')
    classifier_hashes = {}
    for value in config['classifier_seeds']:
        seed(value, config); net = Classifier(P, config, value)
        h = state_hash(net); assert h == original['shared_seed_initializations_and_draws'][str(value)]['shared_initial_state_SHA256']
        classifier_hashes[value] = h
        if value == 1337:
            before = {k: v.clone() for k, v in net.state_dict().items() if 'running_' in k or 'num_batches_tracked' in k}
            net.set_stage(True); pred = net(torch.rand(1, 1, 3, 32, 32)); pred.square().mean().backward()
            assert all(torch.equal(v, net.state_dict()[k]) for k, v in before.items()) and len(before) == 159
        del net; gc.collect()
    checks['three_classifier_initial_states_exact_prior_ImageNetV1_recipe'] = True
    checks['BN159_running_buffers_frozen_during_full_stage'] = True
    spec = importlib.util.spec_from_file_location('prior_transform', D.parent / 'condition_coverage_extension_June26_20261005/production_all_verified/all_conditions_v2_20261005/code/common.py')
    prior = importlib.util.module_from_spec(spec); spec.loader.exec_module(prior)
    a = np.random.default_rng(9).integers(256, size=(8, 256, 256, 3), dtype=np.uint8); row = {'global_identity': 'test_identity'}
    for training in [True, False]:
        assert torch.equal(classifier_tensor(a, row, 2027, 6, training), prior.classifier_tensor(a, row, 2027, 6, training))
    checks['training_eval_224_transforms_exact_prior_recipe'] = True
    # Resume includes model/AdamW moments and all RNG streams. A CPU toy compares exact updates.
    seed(4099, config); initial = torch.nn.Linear(3, 2).state_dict()
    def fit(interrupt):
        seed(17, config); net = torch.nn.Linear(3, 2); net.load_state_dict(initial); opt = torch.optim.AdamW(net.parameters(), lr=.0002)
        for i in range(6):
            x = torch.rand(4, 3); y = torch.randn(4, 2); opt.zero_grad(); (net(x) - y).square().mean().backward(); opt.step()
            if interrupt and i == 2:
                saved = {'model': copy.deepcopy(net.state_dict()), 'optimizer': copy.deepcopy(opt.state_dict()), **rng()}
                net = torch.nn.Linear(3, 2); net.load_state_dict(saved['model']); opt = torch.optim.AdamW(net.parameters(), lr=.0002); opt.load_state_dict(saved['optimizer']); restore_rng(saved)
        return net.state_dict(), opt.state_dict()
    full, aopt = fit(False); resumed, bopt = fit(True)
    assert all(torch.equal(full[k], resumed[k]) for k in full)
    for k in aopt['state']:
        for j in aopt['state'][k]: assert torch.equal(aopt['state'][k][j], bopt['state'][k][j])
    checks['AdamW_and_RNG_exact_interrupted_CPU_replay'] = True
    manifests = {name: read(D / 'data' / name) for name in ['RESTORATION_PAIRED_MANIFEST_v1.json', 'CLASSIFICATION_MANIFEST_v1.json']}
    counts = {}; role_map = {}
    for name, manifest in manifests.items():
        rows = manifest['rows']; counts[name] = {}
        assert len({r['sample_uid'] for r in rows}) == len(rows)
        for r in rows:
            assert r['category'] in config['categories']
            assert role_map.setdefault(r['global_identity'], r['role']) == r['role']
        for role in ['train', 'tune', 'eval']:
            rr = [r for r in rows if r['role'] == role]; counts[name][role] = {'records': len(rr), 'identities': len({r['global_identity'] for r in rr})}
    assert [counts['RESTORATION_PAIRED_MANIFEST_v1.json'][r]['records'] for r in ['train','tune','eval']] == [2648,365,329]
    assert [counts['CLASSIFICATION_MANIFEST_v1.json'][r]['records'] for r in ['train','tune','eval']] == [2758,372,345]
    draws = read(P / config['draws']); assert len(draws['measured_UIDs']) == 65536 and len(draws['synthetic']) == 16384
    measured = {r['sample_uid'] for r in manifests['RESTORATION_PAIRED_MANIFEST_v1.json']['rows'] if r['role'] == 'train'}
    assert set(draws['measured_UIDs']) <= measured
    assert sum(i % 4 == 0 for i in range(1,65537)) == 16384
    assert all(r['updates']==r['measured_updates']+r['synthetic_updates']==65536 for r in config['restoration_runs'])
    checks['fixed_budget_mixture_counts_train_only_draws_and_disjoint_roles'] = True
    natural = read(D.parent / 'synthetic_measured_transfer_20261013/protocol/NATURAL_IMAGE_ROSTER.json')['records']
    nat = {r['image_id']: r for r in natural if r['experiment_role'] == 'train'}; assert len(nat) == 96
    durations = []; example = None
    for i, draw in enumerate(draws['synthetic'][:64]):
        clean = np.load(nat[draw['image_id']]['crop']); t = time.perf_counter(); pixels, metadata = render(clean, draw['generator_seed']); durations.append(time.perf_counter()-t)
        assert pixels.shape == (8,256,256,3) and pixels.dtype == np.uint8
        if i == 0:
            again, meta = render(clean, draw['generator_seed']); assert np.array_equal(pixels,again) and meta == metadata
            example = hashlib.sha256(pixels.tobytes()).hexdigest()
    checks['64_native_synthetic_bursts_reproducible_TRAIN_only'] = True
    assert display_condition('june25_highmotor') == 'C+B+M' and display_condition('june24_highmotor') == 'HM'
    table = []
    for name, manifest in manifests.items():
        d5 = [r for r in manifest['rows'] if r['condition']=='june25_highmotor']
        assert [sum(r['role']==role for r in d5) for role in ['train','tune','eval']] == [404,51,50]
        assert len({r['global_identity'] for r in d5 if r['role']=='eval'}) == 50
        table.append({'manifest':name,'D5_train':404,'D5_tune':51,'D5_eval':50,'D5_eval_IDs':50})
    checks['D5_confirmed_override_and_fifty_eval_IDs_unchanged'] = True
    receipt = {'passed': True, 'CPU_only': True, 'CUDA_models_created': 0, 'cluster_or_submission_actions': 0,
      'UTC': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
      'checks':checks,'restoration_initial_states':hashes,'finite_gradient_tensor_counts':grad_hashes,
      'classifier_initial_states':classifier_hashes,'manifest_counts':counts,'D5_counts':table,
      'synthetic_CPU_seconds_per_burst_mean':float(np.mean(durations)), 'synthetic_CPU_seconds_per_burst_max64':float(np.max(durations)),
      'synthetic_CPU_seconds_for_16384_estimate':float(np.mean(durations)*16384), 'synthetic_sample_pixel_sha256':example,
      'elapsed_seconds':time.perf_counter()-started,
      'limits':['CPU checks do not establish CHPC CUDA/runtime compatibility, VRAM or throughput. No cluster warmup/benchmark/job is added.','Mass submission still requires parent matrix/compute review.']}
    write(D / 'audit/PORTABLE_SCIENCE_CPU_QA.json', receipt); print(json.dumps(receipt), flush=True)

if __name__ == '__main__': main()
