"""One manual evaluation command after all scientific checkpoint selections freeze."""
from pathlib import Path
import argparse, gc, os
import numpy as np
import torch
from skimage.metrics import structural_similarity
from campaign_models import Restoration, Classifier, digest
from campaign_common import read, write, roles, tensors, probability_cohorts, utc, verify_release
from campaign_conditions import display_condition, display_cohorts
from campaign_classify import restoration_freeze, Store, infer

def main(args):
    root = Path(args.data_root).resolve(); code_root = Path(__file__).resolve().parent.parent
    config = read(args.protocol); ph = digest(args.protocol); results = Path(args.results_root or os.environ['RESULTS_DIR']).resolve()
    verify_release(code_root)
    for name, expected in config['manifest_sha256'].items(): assert digest(root / name) == expected
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    restoration = restoration_freeze(config, results, ph); classifiers = []
    for arm in config['classifier_arms']:
        for value in config['classifier_seeds']:
            folder = results / 'classifiers' / arm['arm'] / f'seed{value}'; s = read(folder / 'COMPLETE.json')
            assert s['protocol_sha256'] == ph and s['updates'] == 1000
            assert digest(folder / 'best.pth') == s['checkpoint_sha256']; classifiers.append(s)
    assert len(classifiers) == 30
    frozen = results / 'ALL_SELECTIONS_FROZEN.json'
    if frozen.exists():
        prior = read(frozen); assert prior['restoration'] == restoration and prior['classifiers'] == classifiers and prior['protocol_sha256'] == ph
    else: write(frozen, {'UTC': utc(), 'protocol_sha256': ph, 'restoration': restoration, 'classifiers': classifiers})
    eval_dir = results / 'evaluation'; eval_dir.mkdir(exist_ok=True)
    rr = roles(read(root / config['restoration_manifest'])['rows']); assert len(rr['eval']) == 329
    class_rows = read(root / config['classification_manifest'])['rows']; class_roles = roles(class_rows); assert len(class_roles['eval']) == 345
    paired_UIDs = {r['sample_uid'] for r in rr['eval']}
    for spec in config['restoration_runs']:
        key = spec['run_id']; path = eval_dir / f'restoration_{key}.json'
        if path.exists(): assert read(path)['freeze_sha256'] == digest(frozen); continue
        family, _, mode = spec['model'].split('_'); net = Restoration(family, mode, code_root, config).cuda().eval()
        saved = torch.load(results / 'restoration' / key / 'best.pth', map_location='cpu', weights_only=False); net.load_state_dict(saved['state_dict']); del saved
        scores = []; pixels = []
        with torch.inference_mode():
            for r in rr['eval']:
                x, y = tensors(root, r, 'cuda'); raw = net(x)[0].float().cpu().permute(1, 2, 0).numpy(); assert np.isfinite(raw).all()
                pred = np.clip(raw, 0, 1); target = y[0].cpu().permute(1, 2, 0).numpy(); error = pred - target
                scores.append({k: r[k] for k in ['sample_uid', 'global_identity', 'condition', 'category']} |
                    {'display_condition': display_condition(r['condition']), 'ssim': float(structural_similarity(pred, target, data_range=1, channel_axis=2)),
                     'psnr_db': float(-10 * np.log10(max(np.mean(error.astype(np.float64) ** 2), 1e-12))), 'mae': float(np.abs(error).mean()),
                     'clipping_fraction': float(np.mean((raw < 0) | (raw > 1)))})
                pixels.append(raw)
        fp = eval_dir / f'restoration_{key}_float32.npy'; np.save(fp, np.stack(pixels))
        write(path, {'run_id': key, 'checkpoint_sha256': restoration[key]['selected_checkpoint_sha256'], 'freeze_sha256': digest(frozen),
                     'scores': scores, 'UIDs': [r['sample_uid'] for r in rr['eval']], 'float32_sha256': digest(fp),
                     'scope': 'Archived digital display targets, approximate frozen source geometry. No calibrated optical/physical noGT accuracy claim.'})
        del net, pixels; gc.collect(); torch.cuda.empty_cache()
    for arm in config['classifier_arms']:
        key = arm['arm']; store = Store(key, root, class_rows, code_root, config, results, restoration)
        for value in config['classifier_seeds']:
            path = eval_dir / f'classification_{key}_seed{value}.json'
            if path.exists(): assert read(path)['freeze_sha256'] == digest(frozen); continue
            net = Classifier(code_root, config, value).cuda().eval()
            cp = results / 'classifiers' / key / f'seed{value}' / 'best.pth'; saved = torch.load(cp, map_location='cpu', weights_only=False); net.load_state_dict(saved['state_dict']); del saved
            scores = infer(net, class_roles['eval'], store); paired = [r for r in scores if r['sample_uid'] in paired_UIDs]
            assert len(scores) == 345 and len(paired) == 329
            write(path, {'arm': key, 'seed': value, 'checkpoint_sha256': digest(cp), 'freeze_sha256': digest(frozen), 'scores': scores,
                         'cohorts345': probability_cohorts(scores), 'paired329cohorts': probability_cohorts(paired),
                         'display_cohorts345': display_cohorts(scores), 'paired329_display_cohorts': display_cohorts(paired),
                         'scope': 'Recorded categories/acquisitions, historically reused evaluation identities. No unseen-condition semantic generalization claim.'})
            del net; gc.collect(); torch.cuda.empty_cache()
        del store
    if args.physical:
        physical = read(code_root / 'audit/PHYSICAL24_ROSTER.json')['cases']; assert len(physical) == 24
        for spec in config['restoration_runs']:
            key = spec['run_id']; folder = eval_dir / 'physical24'; folder.mkdir(exist_ok=True); path = folder / (key + '.json')
            if path.exists():
                receipt = read(path); assert receipt['freeze_sha256'] == digest(frozen) and digest(folder / (key + '.npy')) == receipt['array_sha256']; continue
            family, _, mode = spec['model'].split('_'); net = Restoration(family, mode, code_root, config).cuda().eval()
            cp = torch.load(results / 'restoration' / key / 'best.pth', map_location='cpu', weights_only=False); net.load_state_dict(cp['state_dict']); del cp
            outputs = []
            with torch.inference_mode():
                for row in physical:
                    source = code_root / row['input']; assert digest(source) == row['input_sha256']
                    a = np.load(source); assert a.shape == (8,256,256,3) and a.dtype == np.uint8
                    x = torch.from_numpy(a.copy()).permute(0,3,1,2).float().div_(255)[None].cuda()
                    pixels = net(x)[0].cpu().permute(1,2,0).numpy(); assert np.isfinite(pixels).all(); outputs.append(pixels)
            array_path = folder / (key + '.npy'); np.save(array_path, np.stack(outputs))
            write(path, {'run_id':key,'UIDs':[r['case_id'] for r in physical],'checkpoint_sha256':restoration[key]['selected_checkpoint_sha256'],
                         'array_sha256':digest(array_path),'freeze_sha256':digest(frozen),'GT_available':False,'accuracy_metrics_computed':False,
                         'scope':'Fixed physical target visual diagnostics only; no physical PSNR, SSIM or recognition accuracy.'})
            del net, outputs; gc.collect(); torch.cuda.empty_cache()
    write(eval_dir / 'NUMERICALLY_COMPLETE.json', {'UTC': utc(), 'restoration_runs': 7, 'restoration_records_each': 329,
          'classifier_fits': 30, 'classifier_records_each': 345, 'paired_classifier_records_each': 329,
          'freeze_sha256': digest(frozen), 'independent_QA_and_native_review_pending': True})
    print('NUMERICAL_EVALUATION_COMPLETE_PENDING_QA', flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--protocol', required=True); p.add_argument('--data-root', required=True); p.add_argument('--results-root'); p.add_argument('--physical', action='store_true')
    main(p.parse_args())
