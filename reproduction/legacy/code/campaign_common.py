"""Portable array contracts, paired aggregation and resumable training state."""
from pathlib import Path
from collections import defaultdict
import datetime, hashlib, json, os, random, signal
import numpy as np
import torch
from campaign_models import digest

STOP = False
VERIFIED_ARRAY_FILES = set()
def stopping(sig, frame):
    global STOP
    STOP = True
def signals():
    for sig in [signal.SIGUSR1, signal.SIGTERM, signal.SIGINT]: signal.signal(sig, stopping)
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def verify_release(code_root):
    """Check the reviewed science payload before constructing any GPU model."""
    inventory = read(Path(code_root) / 'audit/SCIENCE_FILE_SHA256.json')
    for relative, expected in inventory['files'].items():
        assert digest(Path(code_root) / relative) == expected, relative
    return inventory
def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.writing'); temp.write_text(json.dumps(data, indent=2) + '\n'); temp.replace(path)
def save(path, data):
    path = Path(path); temp = path.with_suffix(path.suffix + '.writing'); torch.save(data, temp); temp.replace(path)
def seed(value, config):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)
    torch.set_num_threads(2)
    for k in ['benchmark', 'deterministic', 'allow_tf32']:
        setattr(torch.backends.cudnn, k, config['numeric']['cudnn_' + k])
    torch.backends.cuda.matmul.allow_tf32 = config['numeric']['matmul_allow_tf32']
def rng():
    return {'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
def restore_rng(value):
    random.setstate(value['python_rng']); np.random.set_state(value['numpy_rng']); torch.set_rng_state(value['torch_rng'])
    if torch.cuda.is_available(): torch.cuda.set_rng_state_all(value['cuda_rng'])
def roles(rows): return {role: [r for r in rows if r['role'] == role] for role in ['train', 'tune', 'eval']}
def array(root, row, field):
    path = Path(root) / row[field]
    if path not in VERIFIED_ARRAY_FILES:
        assert digest(path) == row[field + '_sha256'], path
        VERIFIED_ARRAY_FILES.add(path)
    a = np.load(path)
    expected = (8, 256, 256, 3) if field == 'input_npy' else (256, 256, 3)
    assert a.shape == expected and a.dtype == np.uint8
    return a
def tensors(root, row, device):
    x = torch.from_numpy(array(root, row, 'input_npy').copy()).permute(0, 3, 1, 2).float().div_(255)[None].to(device)
    y = torch.from_numpy(array(root, row, 'target_npy').copy()).permute(2, 0, 1).float().div_(255)[None].to(device)
    return x, y
def identity_mean(rows, key):
    g = defaultdict(lambda: defaultdict(list))
    for r in rows: g[r['global_identity']][r['condition']].append(r[key])
    return float(np.mean([np.mean([np.mean(v) for v in conditions.values()]) for conditions in g.values()]))
def probability_cohorts(rows):
    grouped = defaultdict(lambda: defaultdict(list)); labels = {}; categories = {}
    for row in rows:
        uid = row['global_identity']; labels.setdefault(uid, row['label']); categories.setdefault(uid, row['category'])
        assert labels[uid] == row['label'] and categories[uid] == row['category']
        p = np.asarray(row['probabilities'], dtype=float); assert p.shape == (5,) and np.isfinite(p).all() and abs(p.sum() - 1) < 1e-5
        grouped[uid][row['condition']].append(p)
    output = {}
    for scope in ['overall'] + sorted({r['condition'] for r in rows}):
        ids = sorted(uid for uid, cond in grouped.items() if scope == 'overall' or scope in cond)
        p = np.stack([np.mean([np.mean(v, axis=0) for condition, v in grouped[uid].items() if scope == 'overall' or condition == scope], axis=0) for uid in ids])
        truth = np.array([labels[uid] for uid in ids]); predictions = p.argmax(1)
        cm = np.bincount(truth * 5 + predictions, minlength=25).reshape(5, 5); tp = cm.diagonal(); count = cm.sum(1); guesses = cm.sum(0)
        recall = np.divide(tp, count, out=np.zeros(5), where=count > 0); f1 = np.divide(2 * tp, count + guesses, out=np.zeros(5), where=count + guesses > 0)
        metric_labels = list(range(5)) if scope == 'overall' else sorted(set(truth.tolist()))
        output[scope] = {'ids': ids, 'labels': truth.tolist(), 'categories': [categories[uid] for uid in ids], 'probabilities': p.tolist(), 'predictions': predictions.tolist(),
             'metrics': {'accuracy': float(tp.sum() / len(ids)), 'macro_f1': float(f1[metric_labels].mean()), 'balanced_accuracy': float(recall[metric_labels].mean()),
                         'metric_labels': metric_labels, 'confusion': cm.tolist(), 'per_class_recall': recall.tolist(), 'true_class_counts': count.tolist()}}
    return output

def validate_inputs(root, rows):
    mapping = {}; categories = {}; hashes = defaultdict(set)
    for r in rows:
        assert mapping.setdefault(r['global_identity'], r['role']) == r['role']
        assert categories.setdefault(r['global_identity'], r['category']) == r['category']
        for field in ['input_npy', 'target_npy']:
            if field not in r: continue
            assert digest(Path(root) / r[field]) == r[field + '_sha256']
        hashes[r['input_npy_sha256']].add(r['role'])
    assert all(len(v) == 1 for v in hashes.values())
