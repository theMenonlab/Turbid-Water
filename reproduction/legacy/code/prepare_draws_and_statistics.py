"""Pin TRAIN-only synthetic coefficients and shared full-budget draw schedules."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import hashlib, json, shutil, tarfile, time
import numpy as np
from admit_final_data import D, P, read, sha, write

def draws(rows, n, seed):
    grouped = defaultdict(lambda: defaultdict(list))
    for r in sorted(rows, key=lambda r: r['sample_uid']): grouped[r['global_identity']][r['condition']].append(r['sample_uid'])
    ids = sorted(grouped); rng = np.random.default_rng(seed); result = []
    for _ in range(n):
        uid = ids[int(rng.integers(len(ids)))]; conditions = sorted(grouped[uid]); condition = conditions[int(rng.integers(len(conditions)))]; clips = grouped[uid][condition]
        result.append(clips[int(rng.integers(len(clips)))])
    return result

def main():
    B = D / 'portable'; C = B / 'config'; A = B / 'audit'; start = time.perf_counter()
    original = read(D / 'data/RESTORATION_PAIRED_MANIFEST_v1.json'); train = [r for r in original['rows'] if r['role'] == 'train']; assert len(train) == 2648
    grouped = defaultdict(lambda: defaultdict(list)); proxies = []
    for i, r in enumerate(train):
        x = np.load(r['input_npy']).astype(np.float64) / 255; y = np.load(r['target_npy']).astype(np.float64) / 255
        measured = x.mean(0); gain = measured.std((0, 1), ddof=0) / np.maximum(y.std((0, 1), ddof=0), 1 / 255)
        offset = measured.mean((0, 1)) - gain * y.mean((0, 1)); temporal = float(x.std(0, ddof=0).mean())
        v = {'sample_uid': r['sample_uid'], 'global_identity': r['global_identity'], 'condition': r['condition'], 'gain_RGB': gain.tolist(), 'offset_RGB': offset.tolist(), 'temporal': temporal}
        assert np.isfinite(gain).all() and np.isfinite(offset).all() and np.isfinite(temporal)
        grouped[r['condition']][r['global_identity']].append(v); proxies.append(v)
        if (i + 1) % 500 == 0: print('TRAIN_ONLY_STATISTICS', i + 1, len(train), flush=True)
    conditions = []
    for condition, identities in sorted(grouped.items()):
        g = np.array([np.mean([v['gain_RGB'] for v in values], axis=0) for values in identities.values()])
        o = np.array([np.mean([v['offset_RGB'] for v in values], axis=0) for values in identities.values()])
        t = np.array([np.mean([v['temporal'] for v in values]) for values in identities.values()])
        conditions.append({'condition': condition, 'identities': len(identities), 'records': sum(map(len, identities.values())),
            'effective_gain_quantiles': np.quantile(g, [.1, .5, .9], axis=0, method='linear').tolist(),
            'offset_quantiles': np.quantile(o, [.1, .5, .9], axis=0, method='linear').tolist(),
            'temporal_variation_quantiles': np.quantile(t, [.1, .5, .9], method='linear').tolist()})
    stats = {'UTC': datetime.now(timezone.utc).isoformat(), 'source_manifest_sha256': sha(D / 'data/RESTORATION_PAIRED_MANIFEST_v1.json'),
      'roles_used': ['train'], 'records': 2648, 'global_IDs': 580, 'conditions': conditions,
      'quantile_levels': [.1, .5, .9], 'quantile_method': 'numpy.linear', 'population_std_ddof': 0,
      'formula': {'gain_RGB': 'std_pixels(float64mean8(encodedRGB/255))/max(std_pixels(float64targetRGB/255),1/255)',
                  'offset_RGB': 'mean_pixels(mean8)-gain_RGB*mean_pixels(target)',
                  'temporal_proxy': 'mean_pixels_and_channels(population_std_across8slots(encodedRGB/255))',
                  'aggregation': 'Mean interval proxies within each globalID/recordedcondition, then quantiles across IDs per condition.'},
      'scope': 'Image-domain augmentation proxies under frozen approximate geometry; no physical extinction, radiance or registration calibration. No tune/eval/reserved statistics.',
      'source_code_sha256': sha(__file__), 'computation_seconds': time.perf_counter() - start}
    write(A / 'TRAIN_IMAGE_DOMAIN_INITIAL_STATISTICS.json', stats)
    write(A / 'TRAIN_STATISTICS_RECORD_PROXIES.json', {'rows': proxies, 'stats_sha256': sha(A / 'TRAIN_IMAGE_DOMAIN_INITIAL_STATISTICS.json')})
    measured = draws(train, 65536, 61013)
    natural_source = P / 'synthetic_measured_transfer_20261013/protocol/NATURAL_IMAGE_ROSTER.json'; natural = read(natural_source)
    ids = sorted(r['image_id'] for r in natural['records'] if r['experiment_role'] == 'train'); assert len(ids) == 96
    rng = np.random.default_rng(6101301)
    synthetic = [{'image_id': ids[int(rng.integers(len(ids)))], 'generator_seed': int(rng.integers(0, 2 ** 63 - 1))} for _ in range(16384)]
    write(C / 'draws.json', {'seed': 61013, 'measured_UIDs': measured, 'synthetic': synthetic,
                           'shared_mixed_measured_prefix_count': 49152, 'control_measured_count': 65536})
    for r in natural['records']:
        source = Path(r['original']); assert sha(source) == r['original_sha256']
        dest = B / 'synthetic_source_originals' / source.name; dest.parent.mkdir(exist_ok=True); shutil.copyfile(source, dest)
    write(A / 'SYNTHETIC_SOURCE_METADATA.json', {'records': [{k: v for k, v in r.items() if k not in ['crop', 'original']} |
            {'source_original': 'synthetic_source_originals/' + Path(r['original']).name,
             'crop': 'data/objects/' + r['crop_sha256'] + '.npy'} for r in natural['records']],
            'source_roster_sha256': sha(natural_source), 'split': natural['role_counts'],
            'source_originals_are_provenance': 'Renderer consumes immutable256RGBcrops in the data archive; these originalJPEGs preserve source/attribution lineage.'})
    config_path = C / 'CAMPAIGN_v1.json'; prior = config_path.read_bytes(); config_path.with_name('CAMPAIGN_v1.prepared_assets.bak.json').write_bytes(prior)
    config = json.loads(prior)
    archive = P / 'chpc_full_campaign_package_20261013/NRL_FINAL_PAIRED_AND_SYNTHETIC_DATA_20261013.tar'
    manifests = ['data/RESTORATION_PAIRED_MANIFEST_v1.json', 'data/CLASSIFICATION_MANIFEST_v1.json', 'data/NATURAL_IMAGE_ROSTER.json']
    with tarfile.open(archive, 'r') as t:
        hashes = {name: hashlib.sha256(t.extractfile(name).read()).hexdigest() for name in manifests}
    config['restoration_manifest'], config['classification_manifest'], config['natural_manifest'] = manifests
    config['manifest_sha256'] = hashes; config['draws'] = 'config/draws.json'; config['draws_sha256'] = sha(C / 'draws.json')
    config['generator_stats_sha256'] = sha(A / 'TRAIN_IMAGE_DOMAIN_INITIAL_STATISTICS.json')
    config['generator_sha256'] = sha(B / 'code/water_generator.py')
    config['mass_submission_reviewed'] = False; config['scientific_adapters_authorized'] = True
    config['data_archive_sha256'] = read(P / 'chpc_full_campaign_package_20261013/audit/FINAL_DATA_UPLOAD_VERIFIED.json')['sha256']
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    write(D / 'handoffs/SYNTHETIC_DEPENDENCY_HANDOFF.json', {'data_archive': str(archive), 'data_archive_immutable': True,
        'renderer': str(B / 'code/water_generator.py'), 'renderer_sha256': config['generator_sha256'],
        'frozen_new_TRAIN_only_constants': str(A / 'TRAIN_IMAGE_DOMAIN_INITIAL_STATISTICS.json'), 'constants_sha256': config['generator_stats_sha256'],
        'draws': str(C / 'draws.json'), 'draws_sha256': config['draws_sha256'], 'source_original_JPEGs': str(B / 'synthetic_source_originals'),
        'source_split_and_attribution': str(A / 'SYNTHETIC_SOURCE_METADATA.json'), 'split': natural['role_counts'],
        'six_mixed_runs': [r['run_id'] for r in config['restoration_runs'] if r['synthetic_updates']],
        'only_measured_control': 'restormer_RGB_burst_measured', 'no_data_archive_changes': True})
    print('STATISTICS_AND_DRAW_CONTRACT_READY', time.perf_counter() - start, flush=True)

if __name__ == '__main__': main()
