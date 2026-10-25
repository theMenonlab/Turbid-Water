"""Manually launched matched classifier fits; evaluation is a later manual command."""
from pathlib import Path
import argparse, gc, json, os, time
import numpy as np
import torch
from campaign_models import Restoration, Classifier, digest, state_hash
import campaign_common as common
from campaign_common import read, write, save, seed, rng, restore_rng, roles, array, probability_cohorts, utc
from campaign_functions import classifier_tensor

def sequence(rows, value):
    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(list))
    for r in sorted(rows, key=lambda r: r['sample_uid']): grouped[r['global_identity']][r['condition']].append(r)
    ids = sorted(grouped); random = np.random.default_rng(value); result = []
    for _ in range(32000):
        uid = ids[int(random.integers(len(ids)))]; conditions = sorted(grouped[uid]); condition = conditions[int(random.integers(len(conditions)))]; records = grouped[uid][condition]
        result.append(records[int(random.integers(len(records)))])
    return result

def restoration_freeze(config, results, protocol_sha):
    selected = {}
    for r in config['restoration_runs']:
        folder = results / 'restoration' / r['run_id']; s = read(folder / 'COMPLETE.json')
        assert s['protocol_sha256'] == protocol_sha and s['updates'] == r['updates'] == 65536
        assert s['measured_updates'] == r['measured_updates'] and s['synthetic_updates'] == r['synthetic_updates']
        assert digest(folder / 'best.pth') == s['selected_checkpoint_sha256']
        selected[r['run_id']] = s
    assert len(selected) == 7
    return selected

class Store:
    def __init__(self, arm, root, rows, code_root, config, results, selected):
        self.arm = arm; self.root = root; self.rows = rows; self.index = {r['sample_uid']: i for i, r in enumerate(rows)}; self.cache = None
        if arm.startswith('raw'): return
        directory = results / 'classifier_representations' / arm; directory.mkdir(parents=True, exist_ok=True)
        path = directory / 'all_roles_uint8.npy'; receipt = directory / 'CACHE_READY.json'
        if receipt.exists():
            info = read(receipt); assert info['UIDs'] == [r['sample_uid'] for r in rows]
            assert info['classification_manifest_sha256'] == config['manifest_sha256'][config['classification_manifest']]
            assert info['restorer_checkpoint_sha256'] == selected[arm]['selected_checkpoint_sha256'] and digest(path) == info['array_sha256']
            self.cache = np.load(path, mmap_mode='r'); return
        spec = next(r for r in config['restoration_runs'] if r['run_id'] == arm); family, _, mode = spec['model'].split('_')
        net = Restoration(family, mode, code_root, config).cuda().eval(); cp = torch.load(results / 'restoration' / arm / 'best.pth', map_location='cpu', weights_only=False)
        net.load_state_dict(cp['state_dict'], strict=True); del cp
        # The cache is disposable until its complete hash receipt exists.
        output = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8, shape=(len(rows), 256, 256, 3))
        with torch.inference_mode():
            for i, r in enumerate(rows):
                x = torch.from_numpy(array(root, r, 'input_npy').copy()).permute(0, 3, 1, 2).float().div_(255)[None].cuda()
                pred = net(x)[0].float().cpu().permute(1, 2, 0).numpy(); assert np.isfinite(pred).all()
                output[i] = np.rint(np.clip(pred, 0, 1) * 255).astype(np.uint8)
                if (i + 1) % 200 == 0: print('CLASSIFIER_CACHE', arm, i + 1, len(rows), flush=True)
                if common.STOP:
                    output.flush()
                    print('PARTIAL_CACHE_DISPOSABLE_MANUAL_RESTART', arm, i + 1, flush=True)
                    raise SystemExit(75)
        output.flush(); del output, net, x, pred; gc.collect(); torch.cuda.empty_cache()
        write(receipt, {'UIDs': [r['sample_uid'] for r in rows], 'array_sha256': digest(path), 'restorer_checkpoint_sha256': selected[arm]['selected_checkpoint_sha256'],
                        'classification_manifest_sha256': config['manifest_sha256'][config['classification_manifest']],
                        'quantization': 'float32 prediction clamped0to1, np.rint255 ties-even uint8', 'classification_outcomes_used': False})
        self.cache = np.load(path, mmap_mode='r')
    def get(self, r):
        if self.cache is not None: return self.cache[self.index[r['sample_uid']]][None].copy()
        a = array(self.root, r, 'input_npy')
        if self.arm == 'raw1': return a[4:5]
        if self.arm == 'rawmean': return np.rint(a.astype(np.float32).mean(0, dtype=np.float32)).astype(np.uint8)[None]
        return a

def infer(net, rows, store):
    records = []; net.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), 4):
            rr = rows[start:start + 4]; x = torch.stack([classifier_tensor(store.get(r), r, training=False) for r in rr]).cuda()
            probs = net(x).softmax(1).cpu().numpy(); assert np.isfinite(probs).all()
            records.extend({k: r[k] for k in ['sample_uid', 'global_identity', 'category', 'condition', 'label']} | {'probabilities': p.tolist()} for r, p in zip(rr, probs))
    return records

def train_one(arm, value, config, protocol_sha, code_root, rr, store, results):
    out = results / 'classifiers' / arm / f'seed{value}'; out.mkdir(parents=True, exist_ok=True)
    if (out / 'COMPLETE.json').exists():
        s = read(out / 'COMPLETE.json'); assert s['protocol_sha256'] == protocol_sha and s['updates'] == 1000 and digest(out / 'best.pth') == s['checkpoint_sha256']; return s
    seed(value, config); net = Classifier(code_root, config, value).cuda(); initial = state_hash(net)
    draws = sequence(rr['train'], value); draw_sha = __import__('hashlib').sha256('\n'.join(r['sample_uid'] for r in draws).encode()).hexdigest()
    first = 1; updates = 0; best = -1.; selected = None; opt = scheduler = None; prior = None; elapsed_prior = 0.
    resume = out / 'resume.pth'
    if resume.exists():
        prior = torch.load(resume, map_location='cpu', weights_only=False)
        assert prior['protocol_sha256'] == protocol_sha and prior['arm'] == arm and prior['seed'] == value
        assert prior['initial_state_sha256'] == initial and prior['draws_UID_sha256'] == draw_sha
        net.load_state_dict(prior['model'], strict=True); first = prior['next_epoch']; updates = prior['updates']; best = prior['best']; selected = prior['selected_epoch']; elapsed_prior = prior['elapsed_seconds']
    write(out / 'CONFIG.json', {'arm': arm, 'seed': value, 'initial_state_sha256': initial, 'draws_UID_sha256': draw_sha,
                              'protocol_sha256': protocol_sha, 'updates': 1000, 'batch_size': 32, 'microbatch': 4})
    started = time.perf_counter()
    def snapshot(next_epoch, next_batch, losses, full):
        save(resume, {'model': {k: v.detach().cpu() for k, v in net.state_dict().items()}, 'optimizer': opt.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None, 'next_epoch': next_epoch, 'next_batch': next_batch,
            'losses': losses, 'full': full, 'updates': updates, 'sampler_position': updates * 32, 'best': best,
            'selected_epoch': selected, 'elapsed_seconds': elapsed_prior + time.perf_counter() - started,
            'initial_state_sha256': initial, 'draws_UID_sha256': draw_sha, 'protocol_sha256': protocol_sha, 'arm': arm, 'seed': value, **rng()})
    for epoch in range(first, 26):
        net.set_stage(epoch >= 6)
        if opt is None or epoch == 6:
            opt = torch.optim.AdamW([v for v in net.parameters() if v.requires_grad], lr=.001 if epoch < 6 else .0001, weight_decay=.0001)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20) if epoch >= 6 else None
            if prior is not None and prior['full'] == (epoch >= 6):
                opt.load_state_dict(prior['optimizer'])
                if scheduler: scheduler.load_state_dict(prior['scheduler'])
        batch_start = prior['next_batch'] if prior is not None else 0
        losses = prior['losses'] if prior is not None else []
        if prior is not None: restore_rng(prior); prior = None
        for j in range(batch_start, 40):
            batch = draws[((epoch - 1) * 40 + j) * 32:((epoch - 1) * 40 + j + 1) * 32]
            x = torch.stack([classifier_tensor(store.get(r), r, value, epoch) for r in batch]); y = torch.tensor([r['label'] for r in batch]); opt.zero_grad(set_to_none=True); loss_sum = 0.
            for k in range(0, 32, 4):
                loss = torch.nn.functional.cross_entropy(net(x[k:k + 4].cuda()), y[k:k + 4].cuda(), reduction='sum') / 32
                assert torch.isfinite(loss); loss.backward(); loss_sum += float(loss.detach())
            assert all(torch.isfinite(v.grad).all() for v in net.parameters() if v.grad is not None)
            opt.step(); updates += 1; losses.append(loss_sum)
            if common.STOP:
                snapshot(epoch, j + 1, losses, epoch >= 6)
                return None
        scores = infer(net, rr['tune'], store); cohorts = probability_cohorts(scores); metric = cohorts['overall']['metrics']['macro_f1']
        if metric > best:
            best = metric; selected = epoch
            save(out / 'best.pth', {'state_dict': {k: v.detach().cpu() for k, v in net.state_dict().items()}, 'epoch': epoch, 'updates': updates,
                                  'protocol_sha256': protocol_sha, 'arm': arm, 'seed': value, 'tuning_macro_f1': metric})
        write(out / f'tune_epoch_{epoch:02d}.json', {'epoch': epoch, 'updates': updates, 'loss_mean': float(np.mean(losses)), 'macro_f1': metric, 'scores': scores, 'cohorts': cohorts})
        if scheduler: scheduler.step()
        elapsed = elapsed_prior + time.perf_counter() - started
        snapshot(epoch + 1, 0, [], epoch >= 6)
        print('CLASSIFIER_TUNING', arm, value, epoch, updates, metric, flush=True)
        if common.STOP: return None
    assert updates == 1000
    result = {'arm': arm, 'seed': value, 'updates': 1000, 'protocol_sha256': protocol_sha, 'initial_state_sha256': initial,
              'draws_UID_sha256': draw_sha, 'selected_epoch': selected, 'best_tuning_macro_f1': best, 'checkpoint_sha256': digest(out / 'best.pth'),
              'elapsed_seconds': elapsed_prior + time.perf_counter() - started, 'UTC': utc(), 'classifier_eval_performed': False}
    write(out / 'COMPLETE.json', result); del net, opt, scheduler; gc.collect(); torch.cuda.empty_cache(); return result

def main(args):
    root = Path(args.data_root).resolve(); code_root = Path(__file__).resolve().parent.parent; config = read(args.protocol); protocol_sha = digest(args.protocol)
    assert config['mass_submission_reviewed']
    common.verify_release(code_root)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    for name, h in config['manifest_sha256'].items(): assert digest(root / name) == h
    assert args.arm in {r['arm'] for r in config['classifier_arms']}
    results = Path(args.results_root or os.environ['RESULTS_DIR']).resolve()
    selected = restoration_freeze(config, results, protocol_sha)
    rows = read(root / config['classification_manifest'])['rows']; rr = roles(rows)
    assert [len(rr[k]) for k in ['train', 'tune', 'eval']] == [2758, 372, 345]
    common.signals(); store = Store(args.arm, root, rows, code_root, config, results, selected)
    for value in config['classifier_seeds']:
        s = train_one(args.arm, value, config, protocol_sha, code_root, rr, store, results)
        if s is None: print('CHECKPOINTED_FOR_MANUAL_RESUME', args.arm, value); return 75
    print('THREE_CLASSIFIER_FITS_COMPLETE_NO_EVALUATION', args.arm, flush=True); return 0

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--arm', required=True); p.add_argument('--protocol', required=True)
    p.add_argument('--data-root', required=True); p.add_argument('--results-root')
    raise SystemExit(main(p.parse_args()))
