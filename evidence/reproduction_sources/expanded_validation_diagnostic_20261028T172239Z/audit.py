"""Frozen-identity FP32 validation audit. No optimizer, backward or training calls."""
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import csv
import gc
import hashlib
import json
import math
import os
import sys
import time

REPO = Path('/home/Ima/work/Projects/DL-projects/AeroDept')
ART = Path(__file__).resolve().parent
RES = REPO / 'results/expanded_validation_diagnostic_20261028T172239Z'
USB = Path('/run/media/Ima/AERODEPTH/AeroDepth-rescue')
ROOT = USB / 'MidAir'
BUNDLE = USB / 'project/artifacts/hdd_recovery_20261028T140852Z/expanded-experiments/20261026T114024_recovery'
os.environ['MPLCONFIGDIR'] = str(ART / 'mpl-cache')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(REPO / 'src'))
import numpy as np
import torch
from aerodepth_lite import expanded as e
from aerodepth_lite.metrics import DepthMetricAccumulator
from aerodepth_lite.model import build_model, model_contract

NAMES = ['epoch4', 'student_A', 'teacher_B']
BANDS = [('0.1-10m', .1, 10.), ('10-50m', 10., 50.), ('50-100m', 50., 100.),
         ('100-300m', 100., 300.), ('300-1000m', 300., 1000.)]
START = time.monotonic()


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [clean(v) for v in x]
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def save(name, data):
    p = RES / name
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(clean(data), indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def progress(stage, **extra):
    d = dict(stage=stage, utc=datetime.now(timezone.utc).isoformat(),
             seconds=time.monotonic() - START, **extra)
    save('status.json', d)
    print(json.dumps(clean(d)), flush=True)


class ValGuard(e.ImageReadGuard):
    def __init__(self, root, records):
        super().__init__(root, records)
        self.opens = defaultdict(int)
        self.rejected = []
        self.audit_log = ART / 'validation_image_opens.jsonl'
        self.audit_fd = os.open(self.audit_log, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o600)

    def check(self, event, args):
        if event in ('open', 'os.scandir', 'os.listdir') and args and not isinstance(args[0], int):
            p = os.path.abspath(os.fsdecode(args[0]))
            if p == '/run/media/Ima/Low storage' or p.startswith('/run/media/Ima/Low storage/'):
                self.rejected.append((event, p))
                raise PermissionError('Failing HDD and all its inputs are forbidden for this audit')
            super().check(event, args)
            if event == 'open' and p in self.allowed:
                self.opens[p] += 1
                os.write(self.audit_fd, (json.dumps({'pid': os.getpid(), 'path': p}) + '\n').encode())


class Stats:
    def __init__(self):
        self.n = self.floor = self.over2 = self.under2 = self.pred_far = 0
        self.err = self.ae = self.sse = self.ar = self.rel_bias = self.pred = self.gt = 0.
        self.deltas = np.zeros(3, dtype=np.int64)

    def add(self, p, g):
        assert p.dtype == g.dtype == np.float64
        if not g.size:
            return
        err = p - g
        ratio = np.maximum(p / g, g / p)
        self.n += len(g)
        self.floor += int(np.count_nonzero(p <= .101))
        self.over2 += int(np.count_nonzero(err >= 2.))
        self.under2 += int(np.count_nonzero(err <= -2.))
        self.pred_far += int(np.count_nonzero(p >= 10.))
        self.err += float(err.sum())
        self.ae += float(np.abs(err).sum())
        self.sse += float(np.square(err).sum())
        self.ar += float((np.abs(err) / g).sum())
        self.rel_bias += float((err / g).sum())
        self.pred += float(p.sum())
        self.gt += float(g.sum())
        self.deltas += [np.count_nonzero(ratio < 1.25**k) for k in (1, 2, 3)]

    def result(self):
        if not self.n:
            return {'valid_pixels': 0, 'absrel': None, 'rmse': None, 'bias_m': None}
        return dict(valid_pixels=self.n, absrel=self.ar / self.n, rmse=math.sqrt(self.sse / self.n),
                    mae_m=self.ae / self.n, bias_m=self.err / self.n,
                    mean_signed_relative_error=self.rel_bias / self.n,
                    mean_prediction_m=self.pred / self.n, mean_gt_m=self.gt / self.n,
                    floor_count=self.floor, floor_rate=self.floor / self.n,
                    overestimate_ge_2m_rate=self.over2 / self.n,
                    underestimate_ge_2m_rate=self.under2 / self.n,
                    prediction_ge_10m_rate=self.pred_far / self.n,
                    delta1=self.deltas[0] / self.n, delta2=self.deltas[1] / self.n,
                    delta3=self.deltas[2] / self.n)


def self_check():
    g = np.array([.1, 1., 9., 10., 50., 100., 300., 1000.], dtype=np.float64)
    p = np.array([.1, 2., 12., 11., 60., 80., 350., 900.], dtype=np.float64)
    s = Stats();s.add(p, g)
    a = DepthMetricAccumulator();a.update(torch.from_numpy(p), torch.from_numpy(g), torch.ones(8, dtype=torch.bool))
    for k in ['absrel', 'rmse', 'delta1', 'delta2', 'delta3']:
        assert math.isclose(s.result()[k], a.compute()[k], abs_tol=1e-12)
    coverage = np.zeros(len(g), dtype=int)
    for _, lo, hi in BANDS:
        coverage += (g >= lo) & ((g < hi) if hi != 1000 else (g <= hi))
    assert np.all(coverage == 1)
    assert Stats().result()['absrel'] is None
    save('supplementary_arithmetic_checks.json', {'passed': True, 'synthetic_pixels': 8,
         'checks': 'Stats agrees with existing accumulator; band boundaries partition .1 through inclusive 1000; empty masks undefined.'})


def main():
    progress('preflight')
    torch.set_num_threads(2)
    e.seed_everything(42)
    assert torch.cuda.is_available(), 'FP32 CUDA validation audit requires available CUDA'
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.hub.download_url_to_file = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('Audit downloads forbidden'))
    assert json.loads((USB / 'READY.json').read_text())['dataset_ready']
    status = json.loads((BUNDLE / 'status.json').read_text())
    assert status['status'] == 'teacher_rejected' and not status['teacher_gate_passed']
    assert not (BUNDLE / 'runs/D_kd').exists()
    gate = json.loads((BUNDLE / 'teacher_gate.json').read_text())
    assert not gate['passed']
    reference = json.loads((REPO / 'results/expanded_validation/validation_metrics.json').read_text())
    records, manifest = e.load_inputs(e.PREP / 'manifest_train_val.json.gz', e.PREP / 'reservations.json', ROOT)
    val = records['val']
    trajectories = sorted(set(r.trajectory_id for r in val))
    assert len(val) == 4410 and len(trajectories) == 2
    guard = ValGuard(ROOT, val)
    # Install before any image access or dataset iterator construction.
    sys.addaudithook(guard.check)
    for forbidden in [ROOT / 'Kite_training/sunny/color_left/trajectory_0004/000000.JPEG',
                      ROOT / 'Kite_training/sunny/color_left/trajectory_0007/000000.JPEG']:
        try:
            e.ImageReadGuard.check(guard, 'open', (str(forbidden), 'rb', 0))
        except PermissionError:
            pass
        else:
            raise AssertionError('Held-out read guard ineffective')
    self_check()
    paths = dict(epoch4=REPO / 'models/mobilenet_v3_small_epoch4.state_dict.pt',
                 student_A=BUNDLE / 'runs/A_supervised/best.pt', teacher_B=BUNDLE / 'runs/B_teacher/best.pt')
    configs, models, identities, histories, payload_metrics = {}, {}, {}, {}, {}
    for name in NAMES:
        p = paths[name]
        digest = sha(p)
        if name == 'epoch4':
            assert digest == e.INITIAL_SHA
            config = json.loads((REPO / 'models/mobilenet_v3_small_epoch4.json').read_text())['training_config']
            payload = torch.load(p, map_location='cpu', weights_only=True)
            state = payload
            expected_metrics = {'overall': reference['metrics'], 'per_trajectory': reference['per_trajectory']}
            receipt = {'epoch_completed': 4, 'validation_absrel': reference['metrics']['absrel']}
        else:
            arm = 'A_supervised' if name == 'student_A' else 'B_teacher'
            budget = 10 if name == 'student_A' else 20
            steps = budget * 22049
            completion = json.loads((BUNDLE / 'completion' / (arm + '.json')).read_text())
            for ck, item in completion['checkpoints'].items():
                assert sha(BUNDLE / 'runs' / arm / ck) == item['sha256']
            assert completion['checkpoints']['last.pt']['epoch_completed'] == budget
            assert completion['checkpoints']['last.pt']['global_step'] == steps
            history = json.loads((BUNDLE / 'runs' / arm / 'history.json').read_text())
            assert len(history) == budget
            selected = min(range(len(history)), key=lambda i: history[i]['validation']['overall']['absrel']) + 1
            receipt = completion['checkpoints']['best.pt']
            assert receipt['epoch_completed'] == selected
            payload = torch.load(p, map_location='cpu', weights_only=False)
            extra = payload['extra']
            config = extra['config']
            assert extra['identity']['scope'] == 'full'
            assert extra['identity']['manifest_sha256'] == e.MANIFEST_SHA
            assert extra['identity']['source_sha256'] == e.source_hashes()
            assert extra['model_contract'] == model_contract(config)
            assert payload['epoch'] + 1 == selected
            assert extra['validation']['precision'] == 'FP32'
            assert receipt['validation_absrel'] == history[selected - 1]['validation']['overall']['absrel']
            expected_metrics = extra['validation']
            state = payload['model']
            histories[name] = history
            if name == 'student_A':
                assert digest == gate['baseline_sha256']
                assert extra['identity']['starting_weights']['sha256'] == e.INITIAL_SHA
            else:
                assert digest == gate['teacher_sha256']
                assert extra['identity']['starting_weights']['kind'] == 'ResNet50_Weights.IMAGENET1K_V2'
        config = dict(config)
        config['device'] = 'cuda'
        configs[name] = config
        models[name] = build_model(config['model'] | {'pretrained': False, 'min_depth': .1})
        models[name].load_state_dict(state, strict=True)
        models[name].to(device).requires_grad_(False).eval()
        assert all(not m.training for m in models[name].modules())
        counters = {k: int(v) for k, v in state.items() if k.endswith('num_batches_tracked')}
        variances = [v.detach().float().flatten() for k, v in state.items() if k.endswith('running_var')]
        variances = torch.cat(variances)
        identities[name] = dict(checkpoint=str(p), sha256=digest, selection_receipt=receipt,
                                contract=model_contract(config), state_digest_before=e.state_digest(models[name]),
                                batchnorm_counter_values=sorted(set(counters.values())),
                                batchnorm_running_variance_min=float(variances.min()),
                                batchnorm_running_variance_max=float(variances.max()),
                                model_tensors_finite=all(torch.isfinite(v).all().item() for v in state.values()))
        payload_metrics[name] = expected_metrics
        del payload, state
        gc.collect()
    assert configs['epoch4']['data'] | {} == configs['student_A']['data'] | {}  # workers/layout contract also matches
    assert model_contract(configs['student_A'])['preprocessing'] == model_contract(configs['teacher_B'])['preprocessing']
    recomputed_gate = e.teacher_gate(payload_metrics['student_A'], payload_metrics['teacher_B'])
    for k in recomputed_gate:
        assert recomputed_gate[k] == gate[k], k
    save('identities.json', identities)
    save('gate_verification.json', dict(passed=True, teacher_gate_passed=False,
         original_gate_sha256=sha(BUNDLE / 'teacher_gate.json'), recomputed_from_saved_metrics=recomputed_gate,
         source_sha256=e.source_hashes(), histories_epochs={k: len(v) for k, v in histories.items()},
         no_kd_output=True, original_baseline_path_is_historical_not_rewritten=gate['baseline_checkpoint']))
    save('runtime.json', dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
         cudnn=torch.backends.cudnn.version(), batch_size=2, threads=2, autocast=False,
         deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
         source_sha256=e.source_hashes(), audit_script_sha256=sha(__file__)))

    overall = {n: DepthMetricAccumulator() for n in NAMES}
    per = {n: {tr: DepthMetricAccumulator() for tr in trajectories} for n in NAMES}
    bands = {n: {tr: {label: Stats() for label, _, _ in BANDS} for tr in trajectories} for n in NAMES}
    full = {n: {tr: Stats() for tr in trajectories} for n in NAMES}
    near = {n: {tr: Stats() for tr in trajectories} for n in NAMES}
    maps = {tr: dict(count=np.zeros((192, 320), dtype=np.uint32),
                    **{n: np.zeros((4, 192, 320), dtype=np.float64) for n in NAMES}) for tr in trajectories}
    (ART / 'near_arrays').mkdir()
    arrays = {tr: np.lib.format.open_memmap(ART / 'near_arrays' / (tr.replace('/', '__') + '.npy'), mode='w+',
              dtype=np.float32, shape=(reference['per_trajectory'][tr]['near_valid_pixels'], 4)) for tr in trajectories}
    offsets = {tr: 0 for tr in trajectories}
    frame_rows = []
    normalization = {}
    current_rgb = [None]
    handles = []
    for name in NAMES:
        model = models[name]
        conv = model.encoder['conv1'] if name == 'teacher_B' else model.encoder[0][0]
        def hook(module, args, name=name, model=model):
            expected = (current_rgb[0] - model.rgb_mean) / model.rgb_std
            assert torch.equal(args[0], expected), 'Normalization differs'
            normalization[name] = dict(exact_encoder_input_match=True,
                max_absolute_difference=float((args[0] - expected).abs().max()), input_dtype=str(args[0].dtype))
        handles.append(conv.register_forward_pre_hook(hook))
    teacher_shapes = []
    for layer in models['teacher_B'].projections:
        handles.append(layer.register_forward_hook(lambda m, a, o: teacher_shapes.append(dict(input=list(a[0].shape), output=list(o.shape)))))
    cfg = configs['student_A'] | {'train': configs['student_A']['train'] | {'batch_size': 2}}
    dl = e.loader(ROOT, val, cfg, 'val', False)
    # Preserve the process audit hook and its atomic append log in loader workers.
    dl.multiprocessing_context = 'fork'
    seen = 0
    rgb_min, rgb_max = 1., 0.
    progress('validation', frames=0, total_frames=len(val))
    with torch.inference_mode():
        for batch in dl:
            rgb = batch['rgb'].to(device)
            target, valid = batch['depth'], batch['valid']
            assert rgb.dtype == target.dtype == torch.float32
            assert valid.dtype == torch.bool and target.shape[1:] == (1, 192, 320)
            rgb_min = min(rgb_min, float(rgb.min()));rgb_max = max(rgb_max, float(rgb.max()))
            assert 0 <= rgb_min <= rgb_max <= 1
            assert torch.isfinite(target[valid]).all() and ((target[valid] >= .1) & (target[valid] <= 1000)).all()
            current_rgb[0] = rgb
            predictions = {}
            for name, model in models.items():
                assert not model.training and all(not m.training for m in model.modules())
                with torch.amp.autocast('cuda', enabled=False):
                    pred = model(rgb.float())
                assert pred.dtype == torch.float32 and pred.shape == target.shape
                assert torch.isfinite(pred).all() and (pred >= .1).all()
                predictions[name] = pred.cpu()
                overall[name].update(predictions[name], target, valid)
            if handles:
                for h in handles:h.remove()
                handles.clear()
            for j, record in enumerate(batch['record']):
                tr = record['trajectory_id']
                gt = target[j, 0].numpy();mask = valid[j, 0].numpy()
                nm = mask & (gt < 10.)
                count = int(nm.sum());off = offsets[tr]
                arrays[tr][off:off + count, 0] = gt[nm]
                maps[tr]['count'] += nm
                g = gt[mask].astype(np.float64)
                for k, name in enumerate(NAMES, 1):
                    pc = predictions[name][j:j + 1]
                    per[name][tr].update(pc, target[j:j + 1], valid[j:j + 1])
                    pp = pc[0, 0].numpy();p = pp[mask].astype(np.float64)
                    full[name][tr].add(p, g)
                    pg = pp[nm].astype(np.float64);gg = gt[nm].astype(np.float64)
                    near[name][tr].add(pg, gg)
                    arrays[tr][off:off + count, k] = pp[nm]
                    err = pp.astype(np.float64) - gt.astype(np.float64)
                    maps[tr][name][0] += np.where(nm, err, 0.)
                    maps[tr][name][1] += np.where(nm, err**2, 0.)
                    maps[tr][name][2] += np.where(nm, np.abs(err) / np.maximum(gt, 1e-6), 0.)
                    maps[tr][name][3] += nm & (pp <= .101)
                    for label, lo, hi in BANDS:
                        use = (g >= lo) & ((g < hi) if hi != 1000 else (g <= hi))
                        bands[name][tr][label].add(p[use], g[use])
                    fm = Stats();fm.add(p, g)
                    nn = Stats();nn.add(pg, gg)
                    row = dict(model=name, trajectory=tr, frame_index=record['frame_index'],
                               near_valid_pixels=count, **fm.result(),
                               near_sse=nn.sse, near_absrel_sum=nn.ar,
                               near_absrel=nn.ar / count if count else None,
                               near_rmse=math.sqrt(nn.sse / count) if count else None,
                               near_bias_m=nn.err / count if count else None,
                               near_mae_m=nn.ae / count if count else None,
                               near_floor_rate=nn.floor / count if count else None)
                    frame_rows.append(row)
                offsets[tr] += count
            seen += len(batch['record'])
            if seen % 100 == 0 or seen == len(val):
                progress('validation', frames=seen, total_frames=len(val))
    assert seen == len(val)
    os.fsync(guard.audit_fd)
    audited_opens = defaultdict(int)
    with guard.audit_log.open() as handle:
        for line in handle:
            row = json.loads(line)
            assert row['path'] in guard.allowed
            audited_opens[row['path']] += 1
    assert len(audited_opens) == len(guard.allowed) == 8820
    metrics = {}
    for name in NAMES:
        overall_metrics = overall[name].compute() | {'near_valid_pixels': overall[name].near_count}
        pt = {tr: a.compute() | {'near_valid_pixels': a.near_count} for tr, a in per[name].items()}
        expected = payload_metrics[name]
        deviations = {}
        for scope, actual in [('overall', overall_metrics), *pt.items()]:
            ref = expected['overall'] if scope == 'overall' else expected['per_trajectory'][scope]
            deviations[scope] = {}
            for key in ['absrel', 'rmse', 'delta1', 'delta2', 'delta3', 'near_absrel', 'near_rmse', 'valid_pixels', 'near_valid_pixels']:
                tolerance = 0 if key.endswith('pixels') else (1e-3 if key.endswith('rmse') else 1e-5)
                delta = actual[key] - ref[key]
                assert abs(delta) <= tolerance, (name, scope, key, delta)
                deviations[scope][key] = delta
        metrics[name] = dict(overall=overall_metrics, per_trajectory=pt, saved_reference_differences=deviations,
            supplementary={tr: {'all_valid': full[name][tr].result(), 'near': near[name][tr].result(),
                               'depth_bands': {label: stat.result() for label, stat in bands[name][tr].items()}}
                           for tr in trajectories})
        identities[name]['state_digest_after'] = e.state_digest(models[name])
        assert identities[name]['state_digest_before'] == identities[name]['state_digest_after']
    save('metrics.json', metrics)
    save('identities.json', identities)
    with (RES / 'per_frame.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(frame_rows[0]));writer.writeheader();writer.writerows(frame_rows)
    packed_maps = {}
    for tr in trajectories:
        assert offsets[tr] == reference['per_trajectory'][tr]['near_valid_pixels']
        arrays[tr].flush()
        for key, arr in maps[tr].items():packed_maps[tr.replace('/', '__') + '__' + key] = arr
    np.savez_compressed(RES / 'near_spatial_maps.npz', **packed_maps)
    save('contract_checks.json', dict(passed=True, normalization=normalization,
         teacher_first_batch_projection_shapes=teacher_shapes, rgb_range=[rgb_min, rgb_max],
         all_outputs_finite_positive_fp32_aligned=True, all_models_eval=True,
         all_model_and_bn_buffers_unchanged=True, no_augmentation=True,
         output_units='Euclidean range metres; no upper clamp or fitted scale',
         mask='GT finite inclusive [.1,1000] before nearest resize; near GT<10',
         floor_definition='Prediction <=0.101 m on VALID GT pixels',
         guard_distinct_image_paths=len(audited_opens), guard_allowed_image_paths=len(guard.allowed),
         guard_total_image_opens=sum(audited_opens.values()), unexpected_dataset_opens=guard.rejected,
         blocked_test_probes_before_io=2, failing_hdd_reads=0))
    save('evaluation_completion.json', dict(passed=True, frames=seen, models=NAMES,
         seconds=time.monotonic() - START, selected_checkpoints_unchanged=True,
         near_arrays={tr: str(ART / 'near_arrays' / (tr.replace('/', '__') + '.npy')) for tr in trajectories},
         scope='FP32 validation only; no optimizer/training, no test inputs, no alternate checkpoint selection'))
    progress('evaluated', frames=seen, total_frames=len(val))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        progress('failed', error=repr(exc))
        raise
