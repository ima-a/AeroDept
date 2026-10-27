"""Frozen protocol execution. No training, policy search, or test-driven selection."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .data import FrameRecord, MidAirDataset
from .inference import file_sha256, load_inference_model
from .metrics import DepthMetricAccumulator
from .replay import _critical_depth
from .repro import seed_everything
from .research import clean, summarize
from .scheduler_v2 import SchedulerV2

POLICIES = ('every', 'fixed2', 'fixed5', 'v2_selected', 'fixed_validation_budget')


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')


def read(path):
    return json.loads(Path(path).read_text())


def table(path, rows):
    import csv
    if rows:
        with Path(path).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(clean(rows))


def protocol_load(path):
    path = Path(path)
    digest = file_sha256(path)
    expected = path.with_suffix('.sha256').read_text().split()[0]
    if digest != expected:
        raise ValueError('Frozen protocol hash mismatch')
    p = read(path)
    if file_sha256(p['model']['weights']) != p['model']['sha256']:
        raise ValueError('Frozen model mismatch')
    if file_sha256('models/mobilenet_v3_small_epoch4.json') != p['model']['metadata_sha256']:
        raise ValueError('Frozen metadata mismatch')
    if file_sha256('results/scheduler_v2/selection.json') != p['v2']['selection_sha256']:
        raise ValueError('Frozen V2 selection mismatch')
    raw = gzip.decompress(Path(p['manifest']['path']).read_bytes())
    if hashlib.sha256(raw).hexdigest() != p['manifest']['uncompressed_sha256']:
        raise ValueError('Frozen manifest mismatch')
    return p, json.loads(raw), digest


def path_group(trajectory):
    environment, _, name = trajectory.split('/')
    return f"{environment.split('_')[0]}/trajectory_{int(name.split('_')[-1]) % 1000:04d}"


def isolation(manifest, pilot):
    """Metadata only: inspect all equivalent weather variants, never image bytes."""
    splits = manifest['splits']
    groups = {s: sorted({path_group(t) for t in ts}) for s, ts in splits.items()}
    problems = []
    for left, right in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        common = sorted(set(groups[left]) & set(groups[right]))
        if common:
            problems.append({'left': left, 'right': right, 'equivalent_paths': common})
    pilot_groups = sorted({path_group(pilot['dataset'][k]) for k in ('train_trajectory', 'validation_trajectory')})
    if set(groups['test']) & set(pilot_groups):
        problems.append({'pilot_test_overlap': sorted(set(groups['test']) & set(pilot_groups))})
    if set(manifest['pilot_paths_excluded_from_test']) != set(pilot_groups):
        problems.append({'pilot_exclusion_metadata_mismatch': True})
    declared = {t: g for g, ts in manifest['path_groups'].items() for t in ts}
    assigned = [tr for ts in splits.values() for tr in ts]
    if len(assigned) != len(set(assigned)):
        problems.append({'trajectory_assigned_multiple_times': True})
    record_keys = [(r['trajectory_id'],r['frame_index']) for r in manifest['records']]
    if len(record_keys) != len(set(record_keys)):
        problems.append({'duplicate_trajectory_frame': True})
    if len(manifest['records']) != sum(manifest['frame_counts'].values()):
        problems.append({'total_record_count_mismatch': True})
    for modality in ['rgb_path','depth_path']:
        paths = [r[modality] for r in manifest['records']]
        if len(paths) != len(set(paths)):
            problems.append({'duplicate_modality_path':modality})
    if any(r['trajectory_id'] not in assigned for r in manifest['records']):
        problems.append({'unassigned_record': True})
    seen = Counter()
    for split, ts in splits.items():
        for tr in ts:
            if declared.get(tr) != path_group(tr):
                problems.append({'incorrect_path_group': tr})
        records = [r for r in manifest['records'] if r['trajectory_id'] in ts]
        if len(records) != manifest['frame_counts'][split]:
            problems.append({'frame_count_mismatch': split})
        for r in records:
            seen[r['trajectory_id']] += 1
            for key, modality in [('rgb_path', 'color_left'), ('depth_path', 'depth')]:
                tr = r['trajectory_id'].split('/')
                expected = '/'.join([*tr[:2], modality, tr[2]]) + '/'
                if not r[key].startswith(expected) or '..' in Path(r[key]).parts or Path(r[key]).is_absolute():
                    problems.append({'invalid_record_path': r[key]})
    if set(seen) != set(declared):
        problems.append({'unassigned_or_absent_trajectory': True})
    return {'passed': not problems, 'split_path_groups': groups, 'pilot_exclusions_verified': pilot_groups,
            'trajectory_frame_counts': dict(seen), 'problems': problems,
            'method': 'Recomputed environment plus ID modulo1000 for every manifest record/split, including every climate variant. Original pilot metadata independently checked. No RGB/depth reads.'}


def sensor_path_audit(root, manifest):
    """Independent position-stream identity checks, including unselected climates."""
    import h5py
    assigned = {path_group(tr): split for split, ts in manifest['splits'].items() for tr in ts}
    streams, problems = [], []
    by_digest = {}
    for environment in sorted({tr.split('/')[0] for ts in manifest['splits'].values() for tr in ts}):
        for sensor in sorted((root/environment).glob('*/sensor_records.hdf5')):
            with h5py.File(sensor, 'r') as h:
                for name in sorted(h):
                    if not name.startswith('trajectory_') or name+'/groundtruth/position' not in h:
                        continue
                    tr = f'{environment}/{sensor.parent.name}/{name}'
                    group = path_group(tr)
                    a = np.asarray(h[name+'/groundtruth/position'][:], dtype='<f8')
                    digest = hashlib.sha256(a.tobytes()).hexdigest()
                    entry = {'trajectory':tr,'path_group':group,'split':assigned.get(group),
                             'position_shape':list(a.shape),'position_sha256':digest}
                    streams.append(entry);by_digest.setdefault(digest,[]).append(entry)
    for values in by_digest.values():
        used = {v['split'] for v in values if v['split'] is not None}
        if len(used)>1:
            problems.append({'identical_position_stream_cross_split':values})
    found = {x['trajectory'] for x in streams}
    for ts in manifest['splits'].values():
        for tr in ts:
            if tr not in found:problems.append({'missing_position_stream':tr})
    return {'passed':not problems,'position_streams':streams,'problems':problems,
            'method':'Canonical weather-group exclusion plus independent exact float64 position-stream SHA-256 across all available climates; sensors only, no images/depth. Exact stream hashes do not prove geographic scene independence.'}


def positions(n, k):
    if not 1 <= k <= n:
        raise ValueError('Invalid fixed call budget')
    return {(i * n + k - 1) // k for i in range(k)}


def budget_positions(n, p):
    b = p['fixed_budget']
    k = (n * b['numerator'] + b['denominator'] - 1) // b['denominator']
    return positions(n, k)


def records_for(manifest, split, root):
    """Read only selected split sensor groups; strip privileged navigation."""
    import h5py
    records, provenance = [], []
    for tr in sorted(manifest['splits'][split]):
        selected = sorted((r for r in manifest['records'] if r['trajectory_id'] == tr), key=lambda r: r['frame_index'])
        parent, name = tr.rsplit('/', 1)
        sensor = root / parent / 'sensor_records.hdf5'
        with h5py.File(sensor, 'r') as handle:
            gps = handle[name + '/gps/velocity']
            hz = float(gps.attrs['sampling_frequency'])
            values = np.asarray(gps[:])
            if hz != 1.0 or values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
                raise ValueError('Unexpected simulated GPS contract: ' + tr)
            # Independently verify the manifest IMU against its original source.
            for leaf, keys in [('imu/accelerometer', ('ax', 'ay', 'az')), ('imu/gyroscope', ('wx', 'wy', 'wz'))]:
                source = handle[name + '/' + leaf]
                freq = float(source.attrs['sampling_frequency'])
                if freq != 100.0:
                    raise ValueError('Unexpected IMU frequency')
                for ordinal, rec in enumerate(selected):
                    expected = np.asarray(source[int(ordinal * freq / 25)])
                    actual = np.array([rec['navigation'][key] for key in keys])
                    if not np.array_equal(actual, expected) or not np.isfinite(actual).all():
                        raise ValueError('Manifest IMU/source disagreement: ' + tr)
        for ordinal, rec in enumerate(selected):
            ix = int(ordinal * hz / 25)
            if ix >= len(values):
                raise ValueError('Insufficient GPS samples')
            nav = {k: rec['navigation'][k] for k in ('ax', 'ay', 'az', 'wx', 'wy', 'wz')}
            nav.update(dict(zip(('gps_vx', 'gps_vy', 'gps_vz'), map(float, values[ix]))))
            records.append(replace(FrameRecord(**rec), navigation=nav))
        provenance.append({'trajectory': tr, 'sensor_file': str(sensor), 'sensor_sha256': file_sha256(sensor),
                           'gps_hz': hz, 'gps_samples': len(values), 'imu_hz': 100, 'imu_manifest_exact_match': True,
                           'alignment': 'floor(camera ordinal * sensor frequency/25), past-only zero-order hold; ideal availability'})
    return records, provenance


def environment():
    import importlib.metadata
    return {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
            'torchvision': importlib.metadata.version('torchvision'), 'device': 'cpu', 'precision': 'FP32',
            'threads': torch.get_num_threads(), 'platform': platform.platform(),
            'cpu': next((s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')), platform.processor()),
            'cuda_available': torch.cuda.is_available(), 'time_utc': datetime.now(timezone.utc).isoformat()}


@torch.inference_mode()
def warmup(model, rgb):
    for _ in range(10):
        model(rgb[None].to('cpu')).cpu()


@torch.inference_mode()
def forward(model, rgb):
    start = time.perf_counter()
    pred = model(rgb[None].to('cpu')).cpu()[0]
    elapsed = time.perf_counter() - start
    if not torch.isfinite(pred).all() or (pred < .1).any():
        raise ValueError('Invalid model output')
    return pred, elapsed


def validation_timing(p, m, digest, root, out):
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'started.json', {'protocol_sha256': digest, 'environment': environment()})
    records, provenance = records_for(m, 'val', root)
    write(out / 'sensor_provenance.json', provenance)
    ds = MidAirDataset(root, records, read('models/mobilenet_v3_small_epoch4.json')['training_config']['data'])
    cache = np.lib.format.open_memmap(out / 'validation_rgb.npy', mode='w+', dtype='float32', shape=(len(ds), 3, 192, 320))
    start = time.perf_counter()
    for i in range(len(ds)):
        cache[i] = ds[i]['rgb'].numpy()
        if (i + 1) % 500 == 0:
            print(f'Validation RGB preparation {i+1}/{len(ds)}', flush=True)
    cache.flush()
    write(out / 'preparation.json', {'seconds_excluded': time.perf_counter() - start, 'frames': len(ds), 'cache_sha256': file_sha256(out / 'validation_rgb.npy')})
    model, _ = load_inference_model(p['model']['weights'], 'models/mobilenet_v3_small_epoch4.json')
    lengths = Counter(r.trajectory_id for r in records)
    fixed = {tr: positions(n, p['timing']['validation_matched_calls'][tr]) for tr, n in lengths.items()}
    runs = []
    for rep, order in enumerate(p['timing']['validation_order'], 1):
        for order_index, policy in enumerate(order):
            warmup(model, torch.from_numpy(np.array(cache[0])))
            sched = SchedulerV2(p['v2']['config'])
            last_tr, previous, age, pos = None, None, 0, 0
            rows = []
            wall = time.perf_counter()
            for i, rec in enumerate(records):
                tr = rec.trajectory_id
                if tr != last_tr:
                    sched.reset(); previous = None; age = 0; pos = 0; last_tr = tr
                rgb = torch.from_numpy(np.array(cache[i]))
                start = time.perf_counter()
                if policy == 'v2_selected':
                    decision = asdict(sched.decide(rgb, rec.navigation, previous)); infer = decision['infer']
                else:
                    infer = pos in fixed[tr]
                    decision = {'reason': 'fixed_budget' if infer else 'fixed_skip'}
                overhead = time.perf_counter() - start
                latency = 0.
                if infer:
                    previous, latency = forward(model, rgb)
                age = 0 if infer else age + 1
                rows.append({'index': i, 'trajectory': tr, 'position': pos, 'infer': infer,
                             'reason': decision['reason'], 'prediction_age': age, 'scheduler_ms': overhead * 1000,
                             'inference_ms': latency * 1000})
                pos += 1
            wall = time.perf_counter() - wall
            table(out / f'rep{rep}_{policy}_frames.csv', rows)
            for scope in ['Overall', *lengths]:
                rr = rows if scope == 'Overall' else [r for r in rows if r['trajectory'] == scope]
                calls = sum(r['infer'] for r in rr)
                expected = sum(p['timing']['validation_matched_calls'].values()) if scope == 'Overall' else p['timing']['validation_matched_calls'][scope]
                if calls != expected:
                    raise ValueError(f'Frozen validation call-count disagreement: {scope}: {calls} != {expected}; stop, do not tune')
                runs.append({'repetition': rep, 'order_index': order_index, 'policy': policy, 'scope': scope,
                             'calls': calls, 'scheduler_seconds': sum(r['scheduler_ms'] for r in rr)/1000,
                             'inference_seconds': sum(r['inference_ms'] for r in rr)/1000,
                             'compute_seconds': sum(r['scheduler_ms']+r['inference_ms'] for r in rr)/1000,
                             'wall_seconds_overall': wall if scope == 'Overall' else None, 'warmup_calls_excluded_overall': 10 if scope == 'Overall' else 0})
            print(f'Validation repetition {rep} {policy}: {runs[-3]["compute_seconds"]:.3f}s compute', flush=True)
            table(out / 'individual_results.csv', runs)
    spreads = []
    for policy in p['timing']['validation_order'][0]:
        for scope in ['Overall', *lengths]:
            rr = [r for r in runs if r['policy'] == policy and r['scope'] == scope]
            for metric in ['scheduler_seconds', 'inference_seconds', 'compute_seconds']:
                v = [r[metric] for r in rr]
                spreads.append({'policy': policy, 'scope': scope, 'metric': metric, 'repetitions': len(v),
                                'median': float(np.median(v)), 'min': min(v), 'max': max(v), 'spread_max_minus_min': max(v)-min(v)})
    table(out / 'median_spread.csv', spreads)
    write(out / 'completed.json', {'passed': True, 'protocol_sha256': digest, 'environment': environment(),
                                 'repetitions_per_policy': 3, 'actual_measured_calls': sum(r['calls'] for r in runs if r['scope'] == 'Overall'),
                                 'excluded_warmup_calls': 60, 'individual_results_sha256': file_sha256(out/'individual_results.csv'),
                                 'frame_logs_sha256': {f.name:file_sha256(f) for f in sorted(out.glob('*_frames.csv'))}, 'summary': spreads})
    timing_report(out)


def timing_report(out):
    """Render saved measurements; never execute policies or select parameters."""
    import csv
    with (out/'individual_results.csv').open() as h:
        rows = list(csv.DictReader(h))
    completion = read(out/'completed.json')
    overall = [r for r in rows if r['scope']=='Overall']
    text = ['# Validation timing verification', '', f"Protocol SHA-256: `{completion['protocol_sha256']}`.", '',
            'Three measured repetitions per policy; predeclared alternating order. FP32 batch1 CPU, two torch threads, ten identical excluded warmup calls before each run. Actual calls, including all scheduler decision overhead, are measured. Cache generation and I/O are excluded from compute; offline replay wall is separate.', '',
            '| Rep | Order in pair | Policy | Calls | Model s | Scheduler s | Compute s |',
            '|---|---|---|---:|---:|---:|---:|']
    for r in overall:
        text.append(f"| {r['repetition']} | {int(r['order_index'])+1} | {r['policy']} | {r['calls']} | {float(r['inference_seconds']):.6f} | {float(r['scheduler_seconds']):.6f} | {float(r['compute_seconds']):.6f} |")
    text += ['', '| Policy | Median compute s | Min s | Max s | Spread s |', '|---|---:|---:|---:|---:|']
    for r in completion['summary']:
        if r['scope']=='Overall' and r['metric']=='compute_seconds':
            text.append(f"| {r['policy']} | {r['median']:.6f} | {r['min']:.6f} | {r['max']:.6f} | {r['spread_max_minus_min']:.6f} |")
    text += ['', 'Each policy executes 2,455 measured calls per repetition (905 Kite; 1,550 PLE); 14,730 measured calls plus 60 excluded warmups overall. This matched comparator uses the original per-trajectory validation budgets. The final-test primary comparator instead uses the separately frozen pooled validation fraction, identically applied to each unseen path. No timing result changes any policy.', '',
             'Individual per-trajectory results and median/min/max/spread are in CSV; every actual call and scheduler duration is retained in frame logs. Three repetitions quantify this run\'s spread, not a population confidence interval. CPU results cannot establish GTX1650 speedup, energy, battery or flight-safety improvement.', '']
    (out/'REPORT.md').write_text('\n'.join(text))
    os.environ.setdefault('MPLCONFIGDIR','/tmp/aerodepth-final-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,4))
    for policy in ['v2_selected','fixed_validation_matched']:
        rr = [r for r in overall if r['policy']==policy]
        ax.plot([int(r['repetition']) for r in rr],[float(r['compute_seconds']) for r in rr],marker='o',label=policy)
    ax.set(xlabel='Measured repetition',ylabel='Inference + scheduler seconds',xticks=[1,2,3]);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'repeated_compute.png',dpi=160);plt.close(fig)


def aggregate(logs, accs):
    metrics = {tr: a.compute() | {'near_valid_pixels': a.near_count} for tr, a in accs.items()}
    result = summarize(logs, metrics)
    total = DepthMetricAccumulator()
    for a in accs.values():
        for k in vars(total):
            setattr(total, k, getattr(total, k) + getattr(a, k))
    result['overall']['metrics'] = total.compute() | {'near_valid_pixels': total.near_count}
    for x in [result['overall'], *result['per_trajectory'].values()]:
        x['compute_seconds'] = x['scheduler_seconds'] + x['inference_seconds']
        x['critical_events']['observed_delays_seconds'] = [v/25 for v in x['critical_events']['observed_delays_frames']]
    return result


@torch.inference_mode()
def final_test(p, m, digest, root, out, timing_dir, allow):
    if not allow:
        raise ValueError('Final test requires explicit --allow-final-test')
    completion = read(timing_dir / 'completed.json')
    if not completion['passed'] or completion['protocol_sha256'] != digest or completion['repetitions_per_policy'] < 3:
        raise ValueError('Complete frozen validation timing first')
    if file_sha256(timing_dir/'individual_results.csv') != completion['individual_results_sha256']:
        raise ValueError('Timing evidence mismatch')
    for name, expected in completion['frame_logs_sha256'].items():
        if file_sha256(timing_dir/name) != expected:
            raise ValueError('Timing frame evidence mismatch')
    audit = isolation(m, read('evidence/real_midair_pilot/metadata.json'))
    audit['sensor_path_audit'] = sensor_path_audit(root, m)
    audit['passed'] = audit['passed'] and audit['sensor_path_audit']['passed']
    if not audit['passed']:
        write(out.parent / 'isolation_FAILED.json', audit)
        raise ValueError('Leakage discovered: stop without replacement')
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'isolation.json', audit)
    records, provenance = records_for(m, 'test', root)
    write(out / 'sensor_provenance.json', provenance)
    model, _ = load_inference_model(p['model']['weights'], 'models/mobilenet_v3_small_epoch4.json')
    ds = MidAirDataset(root, records, read('models/mobilenet_v3_small_epoch4.json')['training_config']['data'])
    lengths = Counter(r.trajectory_id for r in records)
    fixed = {tr: budget_positions(n, p) for tr, n in lengths.items()}
    write(out / 'fixed_validation_budget_positions.json', {tr: sorted(v) for tr, v in fixed.items()})
    # Exclusive ledger lives beside the protocol, so changing --output cannot bypass it.
    ledger = Path('results/final_test/EXPOSURE.json')
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open('x') as handle:
        json.dump({'protocol_sha256': digest, 'output': str(out.resolve()), 'started_utc': datetime.now(timezone.utc).isoformat(),
                   'state': 'Exposure authorized; any failure/partial output must be retained. No automatic retry.'}, handle, indent=2)
    write(out / 'execution.json', {'protocol_sha256': digest, 'timing_completion_sha256': file_sha256(timing_dir/'completed.json'),
                                 'environment': environment(), 'source_hashes': {str(f):file_sha256(f) for f in sorted(Path('src/aerodepth_lite').glob('*.py'))}})
    first = ds[0]['rgb']  # First reserved image/depth access occurs only after the ledger.
    for _ in POLICIES:
        warmup(model, first)
    sched = SchedulerV2(p['v2']['config'])
    logs = {k: [] for k in POLICIES}; accs = {k: {} for k in POLICIES}
    previous = {}; sources = {}; last_tr = None; last_gt = math.inf; pos = 0
    wall = time.perf_counter()
    # Append progress immediately; failed studies retain every completed frame.
    import csv
    from contextlib import ExitStack
    with ExitStack() as stack:
        handles = {k: stack.enter_context((out/f'{k}_frames.csv').open('x', newline='')) for k in POLICIES}
        writers = {}
        for i, rec in enumerate(records):
            sample = ds[i]; rgb = sample['rgb']; target = sample['depth']; valid = sample['valid']; tr = rec.trajectory_id
            if tr != last_tr:
                sched.reset(); previous = {}; sources = {}; last_tr = tr; pos = 0; last_gt = math.inf
                for k in POLICIES:
                    accs[k][tr] = DepthMetricAccumulator()
            critical = _critical_depth(target, valid)
            event = critical < 8 and (last_gt >= 8 or last_gt - critical >= 2)
            last_gt = critical
            shift = i % len(POLICIES)
            for policy in POLICIES[shift:] + POLICIES[:shift]:
                start = time.perf_counter()
                if policy == 'v2_selected':
                    d = asdict(sched.decide(rgb, rec.navigation, previous.get(policy))); infer = d['infer']
                else:
                    infer = pos in fixed[tr] if policy == 'fixed_validation_budget' else pos % {'every':1,'fixed2':2,'fixed5':5}[policy] == 0
                    d = {'reason': 'first_frame' if pos == 0 else 'fixed_refresh' if infer else 'fixed_skip'}
                overhead = time.perf_counter() - start
                latency = 0.
                if infer:
                    previous[policy], latency = forward(model, rgb); sources[policy] = i
                prediction = previous[policy]; age = i - sources[policy]
                if age > (p['freshness']['v2_max_age_frames'] if policy == 'v2_selected' else 4):
                    raise ValueError('Unexpected freshness implementation violation')
                accs[policy][tr].update(prediction, target, valid, 10.)
                row = {'index':i, 'trajectory':tr, 'position':pos, 'frame_index':rec.frame_index, 'timestamp_seconds':pos/25,
                       'infer':infer, 'reason':d['reason'], 'source_index':sources[policy], 'prediction_age':age,
                       'scheduler_ms':overhead*1000, 'inference_ms':latency*1000, 'critical_event':event,
                       'gt_p10':critical, 'offline_predicted_p10_on_event':_critical_depth(prediction,valid) if event else None}
                logs[policy].append(row)
                if policy not in writers:
                    writers[policy] = csv.DictWriter(handles[policy], fieldnames=list(row)); writers[policy].writeheader()
                writers[policy].writerow(clean(row))
            pos += 1
            if (i+1)%250 == 0:
                for h in handles.values(): h.flush()
                print(f'Final test {i+1}/{len(records)} current frames scored for all policies; wall {time.perf_counter()-wall:.1f}s',flush=True)
    results = {policy: aggregate(logs[policy], accs[policy]) for policy in POLICIES}
    for policy, result in results.items():
        result.update({'mode':'actual FP32 calls; every current frame scored', 'warmup_calls_excluded':10})
        write(out/f'{policy}.json', result)
    write(out/'completed.json', {'passed':True, 'protocol_sha256':digest, 'frames':len(records), 'policies':list(POLICIES),
                                'wall_seconds_including_IO_scoring_instrumentation':time.perf_counter()-wall, 'excluded_warmup_calls':50,
                                'actual_measured_model_calls':sum(r['overall']['inference_calls'] for r in results.values())})
    final_report(out, results, digest)


def final_report(out, results, digest):
    rows = []
    for policy, result in results.items():
        for scope, x in [('Overall',result['overall']), *result['per_trajectory'].items()]:
            e = x['critical_events']
            rows.append({'policy':policy,'scope':scope,**x['metrics'],'frames':x['frames'],'calls':x['inference_calls'],
                         'skip_rate':x['skip_rate'],'mean_age':x['mean_prediction_age'],'max_age':x['max_prediction_age'],
                         'event_count':e['count'],'fresh_on_event':e['fresh_on_event'],'missed_update_rate':e['missed_update_rate'],
                         'mean_delay_frames':e['mean_observed_delay_frames'],'max_delay_frames':e['max_observed_delay_frames'],
                         'censored_count':e['right_censored_count'],'scheduler_seconds':x['scheduler_seconds'],
                         'inference_seconds':x['inference_seconds'],'compute_seconds':x['compute_seconds']})
    table(out/'comparison.csv', rows)
    comparisons = []
    for scope in ['Overall',*results['every']['per_trajectory']]:
        def scoped(policy):
            return results[policy]['overall'] if scope == 'Overall' else results[policy]['per_trajectory'][scope]
        for comparator in ['fixed_validation_budget','every','fixed2','fixed5']:
            a,b = scoped('v2_selected'),scoped(comparator)
            for metric in ['absrel','rmse','near_absrel','near_rmse']:
                av,bv = a['metrics'][metric],b['metrics'][metric]
                comparisons.append({'scope':scope,'comparator':comparator,'metric':metric,'v2':av,'comparator_value':bv,
                                    'v2_change_percent':100*(av/bv-1) if bv else None})
    table(out/'v2_comparisons.csv', comparisons)
    os.environ.setdefault('MPLCONFIGDIR','/tmp/aerodepth-final-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    for ax,tr in zip(axes,results['every']['per_trajectory']):
        for policy,r in results.items():
            x = r['per_trajectory'][tr]
            ax.scatter(x['inference_calls'],x['metrics']['near_rmse'],label=policy)
        ax.set(title=tr,xlabel='Actual model calls',ylabel='Near-field RMSE (m)');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'quality_vs_calls.png',dpi=160);plt.close(fig)
    fig, axes = plt.subplots(len(results['every']['per_trajectory']),1,figsize=(12,6),squeeze=False)
    for ax,tr in zip(axes[:,0],results['every']['per_trajectory']):
        import csv
        for policy in ['v2_selected','fixed_validation_budget']:
            with (out/f'{policy}_frames.csv').open() as h: rr=[r for r in csv.DictReader(h) if r['trajectory']==tr]
            ax.plot([int(r['position'])/25 for r in rr],[int(r['prediction_age']) for r in rr],label=policy,alpha=.65,linewidth=.7)
        for r in rr:
            if r['critical_event']=='True':ax.axvline(int(r['position'])/25,color='red',alpha=.4)
        ax.set(title=tr,xlabel='Camera time (s)',ylabel='Prediction age (frames)');ax.legend()
    fig.tight_layout();fig.savefig(out/'freshness_events.png',dpi=160);plt.close(fig)
    lines=['# Frozen independent final test','',f'Protocol SHA-256: `{digest}`. One study; no tuning or retraining.','',
           'All current frames are scored, including reused predictions. Overall metrics pool valid pixels. See comparison.csv for deltas/counts/timing and per-policy JSON/CSV for event delays and frame evidence.','',
           '| Policy | Scope | Calls | AbsRel | RMSE m | Near AbsRel | Near RMSE m | Events | Missed update | Max delay | Compute s |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        fmt=lambda v: 'undefined' if v is None else f'{v:.6g}'
        lines.append('| '+' | '.join(str(r[k]) if k in ['policy','scope'] else fmt(r[k]) for k in ['policy','scope','calls','absrel','rmse','near_absrel','near_rmse','event_count','missed_update_rate','max_delay_frames','compute_seconds'])+' |')
    lines += ['', 'Primary comparison: V2 versus fixed_validation_budget (same pooled validation fraction for each test trajectory, frozen before exposure). Positive error changes are regressions; all remain in v2_comparisons.csv. No retrospective test-call-matched policy was evaluated.','',
              'Fresh inference is an update, not evidence of correct obstacle detection. Zero-event rates are undefined; terminal delays are right censored. Two simulation paths and validation-selected model/policy limit generalization. Compute excludes I/O, preprocessing and scoring; this is CPU timing, not measured energy, battery or flight safety.','']
    (out/'REPORT.md').write_text('\n'.join(lines))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['audit','validation-timing','test'],required=True)
    parser.add_argument('--dataset-root',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,default=Path('results/final_test/protocol.json'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--timing-dir',type=Path,default=Path('artifacts/final_study/validation_timing'))
    parser.add_argument('--allow-final-test',action='store_true')
    a=parser.parse_args(argv)
    torch.set_num_threads(2);seed_everything(42)
    p,m,digest=protocol_load(a.protocol)
    root=a.dataset_root.resolve();out=a.output.resolve()
    if out==root or root in out.parents:
        raise ValueError('Output cannot be inside dataset')
    if a.stage=='audit':
        if out.exists():raise FileExistsError('Preserve existing audit')
        audit=isolation(m,read('evidence/real_midair_pilot/metadata.json'))
        audit['sensor_path_audit']=sensor_path_audit(root,m)
        audit['passed']=audit['passed'] and audit['sensor_path_audit']['passed'];write(out,audit)
        if not audit['passed']:raise ValueError('Leakage discovered; stop')
    elif a.stage=='validation-timing':validation_timing(p,m,digest,root,out)
    else:final_test(p,m,digest,root,out,a.timing_dir,a.allow_final_test)


if __name__=='__main__':
    main()
