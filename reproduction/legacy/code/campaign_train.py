"""One independent, fixed-budget restoration run; no scheduling or test evaluation."""
from pathlib import Path
import argparse, gc, inspect, json, os, time
import numpy as np
import torch
from skimage.metrics import structural_similarity
from campaign_models import Restoration, state_hash, digest, checkpoint
import campaign_common as common
from campaign_common import read, write, save, roles, tensors, seed, rng, restore_rng, identity_mean, utc
from campaign_functions import fidelity_loss
from water_generator import render

def tune(net, rows, root, device):
    net.eval(); records = []
    with torch.inference_mode():
        for r in rows:
            x, y = tensors(root, r, device); pred = net(x)
            assert torch.isfinite(pred).all()
            a = pred[0].clamp(0, 1).cpu().permute(1, 2, 0).numpy(); b = y[0].cpu().permute(1, 2, 0).numpy()
            records.append({k: r[k] for k in ['sample_uid', 'global_identity', 'condition', 'category']} |
                           {'ssim': float(structural_similarity(a, b, data_range=1, channel_axis=2))})
    return records

def run(args):
    root = Path(args.data_root).resolve(); protocol = Path(args.protocol).resolve(); config = read(protocol); protocol_sha = digest(protocol)
    assert config['mass_submission_reviewed'], 'Parent matrix/compute review is required before a full run.'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    code_root = Path(__file__).resolve().parent.parent
    common.verify_release(code_root)
    for name, expected in config['manifest_sha256'].items(): assert digest(root / name) == expected
    assert digest(code_root / config['draws']) == config['draws_sha256']
    assert digest(code_root / 'audit/TRAIN_IMAGE_DOMAIN_INITIAL_STATISTICS.json') == config['generator_stats_sha256']
    assert digest(Path(__file__).parent / 'water_generator.py') == config['generator_sha256']
    spec = next(r for r in config['restoration_runs'] if r['run_id'] == args.run)
    family, _, mode = spec['model'].split('_'); mixed = spec['synthetic_updates'] > 0
    checkpoint_policy = config['restoration_activation_checkpointing'] if family == 'restormer' else 'none'
    assert checkpoint_policy in ['none', 'stages'], 'Unsupported checkpoint policy; no automatic fallback.'
    if checkpoint_policy == 'stages':
        assert 'use_reentrant' in inspect.signature(checkpoint).parameters, 'PyTorch must support explicit non-reentrant checkpointing; no fallback is allowed.'
    minimum_GiB = (10 if checkpoint_policy == 'stages' else 15) if family == 'restormer' else 1
    assert torch.cuda.get_device_properties(0).total_memory >= minimum_GiB * 1024 ** 3, 'Allocated GPU is too small for the reviewed checkpoint policy.'
    rr = roles(read(root / config['restoration_manifest'])['rows']); assert [len(rr[k]) for k in ['train', 'tune', 'eval']] == [2648, 365, 329]
    lookup = {r['sample_uid']: r for r in rr['train']}; natural = {r['image_id']: r for r in read(root / config['natural_manifest'])['records'] if r['experiment_role'] == 'train'}
    assert len(natural) == 96
    draws = read(code_root / config['draws']); assert len(draws['measured_UIDs']) == 65536 and len(draws['synthetic']) == 16384
    assert set(draws['measured_UIDs']) <= set(lookup)
    output_root = Path(args.results_root or os.environ['RESULTS_DIR']).resolve(); out = output_root / 'restoration' / args.run; out.mkdir(parents=True, exist_ok=True)
    if (out / 'COMPLETE.json').exists():
        done = read(out / 'COMPLETE.json'); assert done['protocol_sha256'] == protocol_sha and done['updates'] == 65536
        assert digest(out / 'best.pth') == done['selected_checkpoint_sha256']; print('ALREADY_COMPLETE', args.run); return 0
    seed(config['restoration_seed'], config); net = Restoration(family, mode, code_root, config).cuda()
    if family == 'restormer': net.network.checkpoint_policy = checkpoint_policy
    initial_hash = state_hash(net); opt = torch.optim.AdamW(net.parameters(), lr=.0002, weight_decay=0)
    first = real_count = synthetic_count = 0; best = -float('inf'); selected_step = None; elapsed_prior = 0.
    resume = out / 'resume.pth'
    if resume.exists():
        s = torch.load(resume, map_location='cpu', weights_only=False)
        assert s['protocol_sha256'] == protocol_sha and s['run_id'] == args.run and s['initial_state_sha256'] == initial_hash
        net.load_state_dict(s['model'], strict=True); opt.load_state_dict(s['optimizer']); restore_rng(s)
        first, real_count, synthetic_count, best, selected_step, elapsed_prior = [s[k] for k in ['updates', 'measured_updates', 'synthetic_updates', 'best_tune_ssim', 'selected_step', 'elapsed_seconds']]
        del s
    write(out / 'CONFIG.json', {'run_id': args.run, 'protocol_sha256': protocol_sha, 'initial_state_sha256': initial_hash,
        'initial_asset_sha256': config['assets'][family]['sha256'], 'budget': spec, 'draws_sha256': config['draws_sha256'],
        'data_manifest_sha256': config['manifest_sha256'][config['restoration_manifest']], 'GPU': torch.cuda.get_device_name(),
        'precision': 'FP32', 'numeric': config['numeric'], 'resume_step': first,
        'activation_checkpointing': checkpoint_policy, 'keep_step_checkpoints': config['restoration_keep_step_checkpoints'], 'torch': torch.__version__})
    common.signals(); started = time.perf_counter()
    def snapshot(step):
        return {'model': {k: v.detach().cpu() for k, v in net.state_dict().items()}, 'optimizer': opt.state_dict(),
                'protocol_sha256': protocol_sha, 'run_id': args.run, 'initial_state_sha256': initial_hash,
                'updates': step, 'measured_updates': real_count, 'synthetic_updates': synthetic_count,
                'best_tune_ssim': best, 'selected_step': selected_step, 'elapsed_seconds': elapsed_prior + time.perf_counter() - started,
                'sampler_position': {'measured': real_count, 'synthetic': synthetic_count}, **rng()}
    if first == 0:
        scores = tune(net, rr['tune'], root, 'cuda'); best = identity_mean(scores, 'ssim'); selected_step = 0
        save(out / 'best.pth', {'state_dict': {k: v.detach().cpu() for k, v in net.state_dict().items()}, 'step': 0, 'run_id': args.run, 'protocol_sha256': protocol_sha, 'tuning_ssim': best})
        write(out / 'tune_000000.json', {'step': 0, 'scores': scores, 'identity_condition_ssim': best})
    for step in range(first + 1, 65537):
        net.train(); syn = mixed and step % 4 == 0
        if syn:
            draw = draws['synthetic'][synthetic_count]; row = natural[draw['image_id']]
            clean = np.load(root / row['crop']); burst, meta = render(clean, draw['generator_seed'])
            x = torch.from_numpy(burst.copy()).permute(0, 3, 1, 2).float().div_(255)[None].cuda()
            y = torch.from_numpy(clean.copy()).permute(2, 0, 1).float().div_(255)[None].cuda(); synthetic_count += 1
            identity = 'natural:' + str(draw['image_id'])
        else:
            identity = draws['measured_UIDs'][real_count]; x, y = tensors(root, lookup[identity], 'cuda'); real_count += 1
        opt.zero_grad(set_to_none=True); begin = time.perf_counter(); pred = net(x)
        loss, _ = fidelity_loss(pred, y, pixel_loss='charbonnier', **config['loss'])
        assert torch.isfinite(pred).all() and torch.isfinite(loss)
        loss.backward(); assert all(torch.isfinite(v.grad).all() for v in net.parameters() if v.grad is not None)
        opt.step(); torch.cuda.synchronize()
        with (out / 'training_curve.jsonl').open('a') as f:
            f.write(json.dumps({'step': step, 'kind': 'synthetic' if syn else 'measured', 'identity': identity, 'loss': float(loss.detach()), 'seconds': time.perf_counter() - begin}) + '\n')
        del x, y, pred, loss
        if step in config['restoration_tune_steps']:
            scores = tune(net, rr['tune'], root, 'cuda'); score = identity_mean(scores, 'ssim')
            cp = {'state_dict': {k: v.detach().cpu() for k, v in net.state_dict().items()}, 'step': step, 'run_id': args.run, 'protocol_sha256': protocol_sha, 'tuning_ssim': score}
            if config['restoration_keep_step_checkpoints']: save(out / f'step_{step:06d}.pth', cp)
            if score > best: best = score; selected_step = step; save(out / 'best.pth', cp)
            write(out / f'tune_{step:06d}.json', {'step': step, 'scores': scores, 'identity_condition_ssim': score, 'UTC': utc()})
            print('TUNING', args.run, step, score, 'selected', selected_step, flush=True)
        if step % 256 == 0 or common.STOP: save(resume, snapshot(step))
        if step % 64 == 0: write(out / 'PROGRESS.json', {'updates': step, 'measured': real_count, 'synthetic': synthetic_count, 'UTC': utc()})
        if common.STOP:
            print('CHECKPOINTED_FOR_MANUAL_RESUME', args.run, step, flush=True); return 75
    assert real_count == spec['measured_updates'] and synthetic_count == spec['synthetic_updates']
    receipt = {'run_id': args.run, 'protocol_sha256': protocol_sha, 'updates': 65536, 'measured_updates': real_count,
               'synthetic_updates': synthetic_count, 'initial_state_sha256': initial_hash, 'selected_step': selected_step,
               'selected_checkpoint': 'best.pth', 'selected_checkpoint_sha256': digest(out / 'best.pth'),
               'selection': 'measured_tune_only_earliest_SSIM_maximum', 'best_tune_ssim': best, 'elapsed_seconds': elapsed_prior + time.perf_counter() - started,
               'UTC': utc(), 'test_evaluation_performed': False}
    write(out / 'SELECTION.json', receipt); write(out / 'COMPLETE.json', receipt)
    print('COMPLETE', args.run, 65536, flush=True); return 0

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', required=True); p.add_argument('--protocol', required=True)
    p.add_argument('--data-root', required=True); p.add_argument('--results-root')
    raise SystemExit(run(p.parse_args()))
