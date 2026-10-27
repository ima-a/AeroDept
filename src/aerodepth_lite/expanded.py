"""Guarded expanded-data workflow, separate from every historical runner.

Checkpoints are epoch-boundary only. Bounded software checks use a fixed subset
of real TRAIN inputs; they cannot become full-run checkpoints or teacher gates.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from .data import FrameRecord, MidAirDataset, collate_samples
from .kaggle import path_group
from .losses import depth_loss, masked_log_range_consistency
from .metrics import DepthMetricAccumulator
from .model import build_model, model_contract, model_stats, save_checkpoint
from .repro import seed_everything, seed_worker, sha256_file, run_metadata

PROJECT = Path(__file__).resolve().parents[2]
MANIFEST_SHA = '67c6cad39767d847a2087a81d1ecc4de2ad8c9ecc0fa01e491f3950b25f8dd3a'
INITIAL_SHA = 'b40c37db3507cd22d39fcfeba6e23067fb619aee4e3ca61735a7698515006aa6'
PREP = PROJECT / 'results/expanded_preparation_20261026'
TRAIN_GROUPS = {f'{e}/trajectory_{n:04d}' for e, ns in
                [('Kite', [0, 1, 2, 5, 6, 21]), ('PLE', [1, 2, 4, 5, 6, 7])] for n in ns}
VAL_GROUPS = {'Kite/trajectory_0003', 'PLE/trajectory_0000'}
TEST_GROUPS = {'Kite/trajectory_0007', 'Kite/trajectory_0008', 'PLE/trajectory_0008',
               'PLE/trajectory_0009', 'Kite/trajectory_0004', 'PLE/trajectory_0003'}
SOURCE_FILES = ['src/aerodepth_lite/' + n + '.py' for n in
                ('expanded', 'model', 'losses', 'data', 'metrics', 'repro', 'kaggle')]
GRADIENT_POLICY = {'name': 'same_batch_amp_backoff_v1', 'max_overflow_retries': 8,
                   'restore_before_retry': 'all model buffers and training/data RNG',
                   'optimizer_updates': 'one finite update per batch; no skipped batches',
                   'nonfinite_prediction_or_loss': 'fatal', 'persistent_nonfinite_gradients': 'fatal'}


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def raw_manifest(path):
    b = Path(path).read_bytes()
    return gzip.decompress(b) if b.startswith(b'\x1f\x8b') else b


def check_record(record, split, root, check_files=True):
    """Called before Image.open, including in loader workers. No discovery."""
    allowed = TRAIN_GROUPS if split == 'train' else VAL_GROUPS if split == 'val' else set()
    if path_group(record.trajectory_id) not in allowed:
        raise ValueError('Forbidden image group/split')
    env, climate, trajectory = record.trajectory_id.split('/')
    for field, stream, extension in [('rgb_path', 'color_left', 'JPEG'), ('depth_path', 'depth', 'PNG')]:
        rel = Path(getattr(record, field))
        if rel.is_absolute() or '..' in rel.parts or rel.parts[-5:] != (
                env, climate, stream, trajectory, f'{record.frame_index:06d}.{extension}'):
            raise ValueError('Unsafe or mismatched image path')
        if not check_files:
            continue
        p = Path(root) / rel
        for ancestor in (p, *p.parents):
            if ancestor.is_symlink():
                raise ValueError('Dataset symlinks are forbidden')
            if ancestor == Path(root):
                break


def load_inputs(manifest, reservations, root):
    raw = raw_manifest(manifest)
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA:
        raise ValueError('Frozen expanded manifest SHA-256 mismatch')
    m = json.loads(raw)
    if sha256_file(reservations) != m['reservations_sha256']:
        raise ValueError('Reservation SHA-256 mismatch')
    r = json.loads(Path(reservations).read_text())
    expected = {'train': TRAIN_GROUPS, 'val': VAL_GROUPS,
                'new_test': {'Kite/trajectory_0007', 'Kite/trajectory_0008', 'PLE/trajectory_0008', 'PLE/trajectory_0009'},
                'legacy_test': {'Kite/trajectory_0004', 'PLE/trajectory_0003'}}
    for split, groups in expected.items():
        actual = {g.replace('/', '/trajectory_') for g in r['group_assignments'][split]}
        if actual != groups:
            raise ValueError('Grouped reservations changed')
    if m['splits']['test'] or m['frame_counts'] != {'train': 44098, 'val': 4410, 'test': 0}:
        raise ValueError('Unexpected image inventory/test records')
    records = {}
    for split, groups in [('train', TRAIN_GROUPS), ('val', VAL_GROUPS)]:
        names = set(m['splits'][split])
        if {path_group(t) for t in names} != groups:
            raise ValueError('Grouped split mismatch')
        rows = [FrameRecord(**v) for v in m['records'] if v['trajectory_id'] in names]
        if len(rows) != m['frame_counts'][split]:
            raise ValueError('Frame count mismatch')
        for record in rows:
            check_record(record, split, root, check_files=False)
        records[split] = rows
    if sum(map(len, records.values())) != len(m['records']):
        raise ValueError('Unassigned image records')
    return records, m


class ImageReadGuard:
    """Process-wide protection; dataset-level checks also cover spawn workers."""
    def __init__(self, root, records):
        self.root = os.path.abspath(root)
        self.allowed = {os.path.abspath(Path(root) / getattr(r, k)) for r in records
                        for k in ('rgb_path', 'depth_path')}

    def check(self, event, args):
        if event not in ('open', 'os.scandir', 'os.listdir') or not args or isinstance(args[0], int):
            return
        p = os.path.abspath(os.fsdecode(args[0]))
        if p != self.root and not p.startswith(self.root + '/'):
            return
        if event != 'open' or p not in self.allowed:
            raise PermissionError('Expanded runner rejected dataset discovery/held-out read')
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        if flags & 3:
            raise PermissionError('Dataset writes forbidden')


class GuardedDataset(MidAirDataset):
    def __init__(self, root, records, data, split, augment):
        if split not in ('train', 'val'):
            raise ValueError('Test inputs are not supported')
        self.split = split
        for r in records:
            check_record(r, split, root, check_files=False)
        super().__init__(root, records, data, training=augment)

    def __getitem__(self, index):
        check_record(self.records[index], self.split, self.root)
        return super().__getitem__(index)


def fixed_training_subset(records, count=64):
    by = defaultdict(list)
    for r in records:
        by[r.trajectory_id].append(r)
    candidates = []
    for fraction in (.5, 0., 1., .25):
        for tr in sorted(by):
            rs = by[tr]
            candidate = rs[int((len(rs) - 1) * fraction)]
            if candidate not in candidates:
                candidates.append(candidate)
    if len(candidates) < count:
        raise ValueError('Not enough distinct real TRAIN frames')
    return candidates[:count]


def loader(root, records, config, split, augment):
    dataset = GuardedDataset(root, records, config['data'], split, augment)
    return DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=augment,
                      num_workers=config['data']['workers'], collate_fn=collate_samples,
                      worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(config['seed'] + (split == 'val')),
                      persistent_workers=False, pin_memory=config['device'] == 'cuda')


def state_digest(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def initialize_student(model, path):
    if sha256_file(path) != INITIAL_SHA:
        raise ValueError('Epoch-4 initialization SHA-256 mismatch')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not all(torch.is_tensor(v) and torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Invalid initialization state dict')
    model.load_state_dict(state, strict=True)  # Includes every BatchNorm buffer.


def strict_supervised(prediction, target, valid, config):
    if prediction.shape != target.shape or valid.shape != target.shape:
        raise ValueError('Prediction/target/mask alignment mismatch')
    if not torch.isfinite(prediction).all() or not (prediction > 0).all():
        raise FloatingPointError('Nonfinite/nonpositive model prediction')
    if not torch.isfinite(target[valid]).all() or not (target[valid] > 0).all():
        raise FloatingPointError('Invalid supervised targets inside valid mask')
    loss, terms = depth_loss(prediction.float(), target.float(), valid, config)
    return loss + prediction.float().sum() * 0., terms


def require_identity(actual, expected):
    if actual != expected:
        keys = sorted(k for k in set(actual) | set(expected) if actual.get(k) != expected.get(k))
        raise ValueError('Incompatible expanded checkpoint identity: ' + ', '.join(keys))


def backoff_scaler(scaler):
    """Reduce AMP scale without calling optimizer.step on bad gradients.

    update(new_scale) resets the per-optimizer unscale bookkeeping. Loading the
    public state dict also resets growth tracking, as a normal overflow does.
    """
    state = scaler.state_dict()
    reduced = state['scale'] * scaler.get_backoff_factor()
    if not math.isfinite(reduced) or reduced <= 0:
        raise FloatingPointError('AMP loss scale cannot be reduced safely')
    scaler.update(new_scale=reduced)
    reduced = scaler.get_scale()  # Stored as float32; reject underflow to zero.
    if not math.isfinite(reduced) or reduced <= 0:
        raise FloatingPointError('AMP loss scale cannot be reduced safely')
    state.update(scale=reduced, _growth_tracker=0)
    scaler.load_state_dict(state)
    return reduced


def copy_checkpoint_exact(source, destination, expected_sha):
    """Atomic byte copy, retaining the source and its serialization identity."""
    source, destination = Path(source), Path(destination)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + destination.name, dir=destination.parent)
    try:
        with os.fdopen(descriptor, 'wb') as output, source.open('rb') as incoming:
            shutil.copyfileobj(incoming, output)
            output.flush()
            os.fsync(output.fileno())
        if sha256_file(temporary) != expected_sha:
            raise ValueError('Checkpoint copy identity changed')
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def ensure_prior_best(directory, expected_sha):
    """Recover a best/last pair interrupted between their atomic writes.

    last.pt pins one exact best hash. Keep that prior best while promoting an
    improvement so a killed writer cannot lose the only matching checkpoint.
    """
    directory = Path(directory)
    best, previous = directory / 'best.pt', directory / 'previous_best.pt'
    if not expected_sha:
        raise ValueError('Resume requires a pinned prior best checkpoint')
    if best.is_file() and sha256_file(best) == expected_sha:
        return
    if not previous.is_file() or sha256_file(previous) != expected_sha:
        raise ValueError('Resume requires the exact prior best.pt or recovery copy')
    copy_checkpoint_exact(previous, best, expected_sha)


def teacher_gate(baseline_metrics, teacher_metrics):
    checks = {}
    keys = ('absrel', 'rmse', 'near_absrel', 'near_rmse')
    if set(baseline_metrics['per_trajectory']) != set(teacher_metrics['per_trajectory']):
        raise ValueError('Teacher gate validation trajectory mismatch')
    for tr, b in baseline_metrics['per_trajectory'].items():
        t = teacher_metrics['per_trajectory'][tr]
        checks[tr] = all(isinstance(t[k], (int, float)) and math.isfinite(t[k]) and
                         isinstance(b[k], (int, float)) and math.isfinite(b[k]) and t[k] <= b[k] for k in keys)
        checks[tr] &= t['valid_pixels'] == b['valid_pixels'] and t['near_valid_pixels'] == b['near_valid_pixels']
    b, t = baseline_metrics['overall'], teacher_metrics['overall']
    checks['pooled_absrel_strictly_better'] = (isinstance(t['absrel'], (int, float)) and
        isinstance(b['absrel'], (int, float)) and math.isfinite(t['absrel']) and
        math.isfinite(b['absrel']) and t['absrel'] < b['absrel'])
    return {'passed': all(checks.values()), 'checks': checks,
            'rule': 'Pooled FP32 validation AbsRel strictly better; each validation path overall/near AbsRel and RMSE no worse; identical valid/near pixel counts.'}


def source_hashes():
    return {p: sha256_file(PROJECT / p) for p in SOURCE_FILES}


def check_config(config):
    if config['arm'] not in ('supervised', 'teacher', 'kd'):
        raise ValueError('Unknown arm')
    d, t = config['data'], config['train']
    if {k: d[k] for k in ('image_size', 'min_depth', 'max_depth', 'depth_encoding', 'depth_scale')} != {
        'image_size': [192, 320], 'min_depth': .1, 'max_depth': 1000.,
        'depth_encoding': 'midair_float16_bits', 'depth_scale': 1.}:
        raise ValueError('Expanded preprocessing/range contract changed')
    if t['effective_batch_size'] != t['batch_size'] or t['accumulation_steps'] != 1:
        raise ValueError('Only matched physical batches without accumulation are supported')
    if config['loss'] != {'l1_weight': 1., 'log_weight': .2, 'gradient_weight': .1,
                          'near_weight': 0., 'near_threshold': 10.}:
        raise ValueError('Supervised loss contract changed')
    expected_model = 'resnet50_depth' if config['arm'] == 'teacher' else 'mobilenet_v3_small_depth'
    if config['model']['name'] != expected_model or t['lr_schedule']['name'] != 'cosine':
        raise ValueError('Architecture/schedule mismatch')


class Experiment:
    def __init__(self, config, records, root, manifest_info, reservations, scope, initialize=None,
                 resume=None, teacher=None, teacher_sha=None, gate=None):
        check_config(config)
        self.config, self.scope = config, scope
        self.device = torch.device(config['device'])
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        checkpoint = torch.load(resume, map_location='cpu', weights_only=False) if resume else None
        if resume and initialize:
            raise ValueError('--initialize and --resume are mutually exclusive')
        if checkpoint and 'identity' not in checkpoint.get('extra', {}):
            raise ValueError('Resume requires a complete expanded-workflow checkpoint')
        seed_everything(config['seed'])
        opts = config['model'] | {'pretrained': config['arm'] == 'teacher' and not resume,
                                  'min_depth': config['data']['min_depth']}
        self.model = build_model(opts).to(self.device)
        if config['arm'] == 'teacher':
            if initialize:
                raise ValueError('Teacher uses ImageNet encoder initialization, not student export')
            from torchvision.models import ResNet50_Weights
            enum = ResNet50_Weights.IMAGENET1K_V2
            cached = Path(torch.hub.get_dir()) / 'checkpoints' / Path(enum.url).name
            self.start_identity = {'kind': 'ResNet50_Weights.IMAGENET1K_V2',
                                   'sha256': sha256_file(cached) if cached.exists() else None}
            if checkpoint:
                prior = checkpoint['extra']['identity']['starting_weights']
                if prior['kind'] != self.start_identity['kind'] or (cached.exists() and prior != self.start_identity):
                    raise ValueError('Teacher initialization provenance differs')
                self.start_identity = prior
        else:
            self.start_identity = {'kind': 'published_MobileNet_epoch4', 'sha256': INITIAL_SHA}
            if not resume:
                initialize_student(self.model, initialize or PROJECT / 'models/mobilenet_v3_small_epoch4.state_dict.pt')
        self.teacher = None
        teacher_identity = None
        gate_identity = None
        if config['arm'] == 'kd':
            if not teacher or not teacher_sha or sha256_file(teacher) != teacher_sha:
                raise ValueError('KD requires a pinned teacher checkpoint SHA-256')
            tp = torch.load(teacher, map_location='cpu', weights_only=False)
            ti = tp['extra']['identity']
            if ti['arm'] != 'teacher' or ti['manifest_sha256'] != MANIFEST_SHA:
                raise ValueError('Incompatible teacher provenance')
            require_identity(ti['source_sha256'], source_hashes())
            check_config(tp['extra']['config'])
            teacher_contract = model_contract(tp['extra']['config'])
            if (teacher_contract['architecture'] != 'resnet50_depth' or
                    tp['extra']['model_contract'] != teacher_contract or
                    ti['model_contract'] != teacher_contract or
                    teacher_contract['preprocessing'] != model_contract(config)['preprocessing']):
                raise ValueError('Teacher architecture/range/preprocessing contract differs')
            if ti['starting_weights']['kind'] != 'ResNet50_Weights.IMAGENET1K_V2':
                raise ValueError('Teacher encoder initialization changed')
            self.teacher = build_model(tp['extra']['config']['model'] | {'pretrained': False, 'min_depth': .1})
            self.teacher.load_state_dict(tp['model'], strict=True)
            self.teacher.to(self.device).requires_grad_(False).eval()
            teacher_identity = {'sha256': teacher_sha, 'model_contract': tp['extra']['model_contract'], 'precision': 'FP32'}
            if scope == 'full':
                if not gate or ti['scope'] != 'full':
                    raise ValueError('Full KD requires a full-run teacher and passing validation gate')
                g = json.loads(Path(gate).read_text())
                if not g.get('passed') or g['teacher_sha256'] != teacher_sha or g['manifest_sha256'] != MANIFEST_SHA:
                    raise ValueError('Teacher gate does not authorize this teacher/manifest')
                if sha256_file(g['baseline_checkpoint']) != g['baseline_sha256']:
                    raise ValueError('Teacher gate baseline identity changed')
                expected_gate = make_gate(g['baseline_checkpoint'], teacher)
                if g != expected_gate:
                    raise ValueError('Teacher gate contents/policy changed')
                baseline = torch.load(g['baseline_checkpoint'], map_location='cpu', weights_only=False)
                require_identity(baseline['extra']['identity']['source_sha256'], source_hashes())
                for section in ('data', 'train', 'seed', 'model', 'loss', 'device'):
                    if baseline['extra']['config'][section] != config[section]:
                        raise ValueError('Supervised/KD student arms differ: ' + section)
                gate_identity = sha256_file(gate)
        elif teacher or teacher_sha or gate:
            raise ValueError('Teacher/gate arguments are only valid for KD')
        t = config['train']
        if config['arm'] == 'teacher':
            enc = list(self.model.encoder.parameters())
            enc_ids = {id(p) for p in enc}
            groups = [{'params': enc, 'lr': t['encoder_learning_rate']},
                      {'params': [p for p in self.model.parameters() if id(p) not in enc_ids], 'lr': t['learning_rate']}]
        else:
            groups = self.model.parameters()
        self.optimizer = torch.optim.AdamW(groups, lr=t['learning_rate'], weight_decay=t['weight_decay'])
        if self.teacher is not None:
            assert not {id(p) for p in self.teacher.parameters()} & {
                id(p) for g in self.optimizer.param_groups for p in g['params']}
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=t['lr_schedule']['t_max'], eta_min=t['lr_schedule']['eta_min'])
        self.amp = bool(t['amp']) and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp, init_scale=t['amp_initial_scale'])
        self.identity = {'format': 2, 'arm': config['arm'], 'scope': scope, 'manifest_sha256': MANIFEST_SHA,
                         'reservations_sha256': sha256_file(reservations), 'starting_weights': self.start_identity,
                         'teacher': teacher_identity, 'gate_sha256': gate_identity, 'config': config,
                         'model_contract': model_contract(config), 'source_sha256': source_hashes(),
                         'gradient_policy': GRADIENT_POLICY.copy()}
        self.epoch, self.global_step, self.history, self.best = 0, 0, [], float('inf')
        self.overflow_retries_total = 0
        self.best_checkpoint_sha256 = None
        self.train_loader = loader(root, records['train'], config, 'train', True)
        self.val_records = records.get('val', [])
        self.root = root
        # Crucial: teacher construction must not perturb student/data RNG.
        seed_everything(config['seed'])
        if checkpoint:
            require_identity(checkpoint['extra']['identity'], self.identity)
            self.model.load_state_dict(checkpoint['model'], strict=True)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            extra = checkpoint['extra']
            self.scheduler.load_state_dict(extra['scheduler'])
            self.scaler.load_state_dict(extra['scaler'])
            self.epoch, self.global_step = checkpoint['epoch'] + 1, extra['global_step']
            self.overflow_retries_total = extra['overflow_retries_total']
            self.history, self.best = extra['history'], extra['best_validation_absrel']
            self.best_checkpoint_sha256 = extra['best_checkpoint_sha256']
            if scope == 'full':
                ensure_prior_best(Path(resume).parent, self.best_checkpoint_sha256)
            self.restore_rng(extra['rng'])
        self.teacher_digest = state_digest(self.teacher) if self.teacher is not None else None

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def rng(self):
        return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else None,
                'loader': self.train_loader.generator.get_state()}

    def restore_rng(self, r):
        random.setstate(r['python'])
        np.random.set_state(r['numpy'])
        torch.set_rng_state(r['torch'].cpu())
        if self.device.type == 'cuda':
            torch.cuda.set_rng_state_all([x.cpu() for x in r['cuda']])
        self.train_loader.generator.set_state(r['loader'].cpu())

    def step(self, batch, trace=False):
        self.model.train()
        if self.teacher is not None:
            self.teacher.eval()
        self.sync()
        start = time.perf_counter()
        rgb, target, valid = (batch[k].to(self.device) for k in ('rgb', 'depth', 'valid'))
        teacher_seconds = 0.
        teacher_prediction = None
        if self.teacher is not None:
            self.sync()
            ts = time.perf_counter()
            with torch.no_grad(), torch.amp.autocast(self.device.type, enabled=False):
                teacher_prediction = self.teacher(rgb.float())
            self.sync()
            teacher_seconds = time.perf_counter() - ts
        # An overflowing trial must not count as another BN update or consume
        # another dropout RNG stream. Parameters/optimizer are unchanged until
        # one finite backward has completed. The already augmented batch and
        # frozen FP32 teacher output are reused unchanged.
        buffers = {k: b.detach().clone() for k, b in self.model.named_buffers()} if self.amp else None
        replay_rng = self.rng() if self.amp else None
        scale_start = self.scaler.get_scale()
        overflows = []

        def restore_trial():
            if buffers is not None:
                with torch.no_grad():
                    for name, value in self.model.named_buffers():
                        value.copy_(buffers[name])
                self.restore_rng(replay_rng)

        try:
            for attempt in range(GRADIENT_POLICY['max_overflow_retries'] + 1):
                if attempt:
                    restore_trial()
                self.optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(self.device.type, enabled=self.amp):
                    pred = self.model(rgb)
                supervised, _ = strict_supervised(pred, target, valid, self.config['loss'])
                kd = masked_log_range_consistency(pred, teacher_prediction, valid) if teacher_prediction is not None else pred.float().sum() * 0.
                loss = supervised + self.config['kd_weight'] * kd
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                bad = [name for name, p in self.model.named_parameters()
                       if p.grad is not None and not torch.isfinite(p.grad).all()]
                if not bad:
                    break
                if not self.amp or attempt == GRADIENT_POLICY['max_overflow_retries']:
                    raise FloatingPointError('Persistent nonfinite unscaled gradients: ' + ', '.join(bad))
                old_scale = self.scaler.get_scale()
                new_scale = backoff_scaler(self.scaler)
                self.overflow_retries_total += 1
                event = {'event': 'amp_gradient_overflow_retry', 'global_step': self.global_step,
                         'retry': attempt + 1, 'scale_before': old_scale, 'scale_after': new_scale,
                         'bad_parameters': bad,
                         'frames': [(r['trajectory_id'], r['frame_index']) for r in batch['record']]}
                overflows.append(event)
                print(json.dumps(event), flush=True)
        except Exception:
            restore_trial()
            self.optimizer.zero_grad(set_to_none=True)
            raise
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['train']['grad_clip'], error_if_nonfinite=True)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.sync()
        duration = time.perf_counter() - start
        self.global_step += 1
        row = {'loss': float(loss.detach()), 'supervised_loss': float(supervised.detach()), 'kd_loss': float(kd.detach()),
               'examples': len(rgb), 'compute_seconds': duration, 'teacher_seconds': teacher_seconds,
               'overflow_retries': len(overflows), 'overflow_events': overflows,
               'amp_scale_start': scale_start, 'amp_scale_end': self.scaler.get_scale()}
        if trace:
            row.update(rgb_sha256=hashlib.sha256(batch['rgb'].numpy().tobytes()).hexdigest(),
                       frames=[(r['trajectory_id'], r['frame_index']) for r in batch['record']])
        return row

    def train_epoch(self, limit=None, trace=False):
        rows = []
        iterator = iter(self.train_loader)
        try:
            for i in range(len(self.train_loader) if limit is None else limit):
                start = time.perf_counter()
                batch = next(iterator)
                waited = time.perf_counter() - start
                row = self.step(batch, trace)
                row['loader_wait_seconds'] = waited
                rows.append(row)
                if i == 0 or (i + 1) % 250 == 0:
                    print(json.dumps({'epoch': self.epoch + 1, 'batches': i + 1,
                                      'global_step': self.global_step, 'loss': row['loss'],
                                      'amp_scale': self.scaler.get_scale(),
                                      'overflow_retries_total': self.overflow_retries_total}), flush=True)
        finally:
            del iterator
        self.scheduler.step()
        self.epoch += 1
        return rows

    @torch.no_grad()
    def evaluate(self, records, split, limit=None):
        self.model.eval()
        dl = loader(self.root, records, self.config, split, False)
        pooled, per = DepthMetricAccumulator(), defaultdict(DepthMetricAccumulator)
        start = time.perf_counter()
        examples = 0
        for i, batch in enumerate(dl):
            if limit is not None and i >= limit:
                break
            rgb, target, valid = (batch[k].to(self.device) for k in ('rgb', 'depth', 'valid'))
            # Explicit FP32; KD teacher is never called for validation selection.
            with torch.amp.autocast(self.device.type, enabled=False):
                pred = self.model(rgb.float())
                strict_supervised(pred, target, valid, self.config['loss'])
            pooled.update(pred, target, valid)
            for j, rec in enumerate(batch['record']):
                per[rec['trajectory_id']].update(pred[j:j+1], target[j:j+1], valid[j:j+1])
            examples += len(rgb)
        self.sync()
        def metrics(a):
            return a.compute() | {'near_valid_pixels': a.near_count}
        return clean_json({'overall': metrics(pooled), 'per_trajectory': {tr: metrics(a) for tr, a in per.items()},
                           'seconds': time.perf_counter() - start, 'examples': examples, 'precision': 'FP32'})

    def assert_teacher_frozen(self):
        if self.teacher is not None:
            if self.teacher.training or state_digest(self.teacher) != self.teacher_digest or any(
                    p.requires_grad or p.grad is not None for p in self.teacher.parameters()):
                raise AssertionError('Teacher weights, gradients or BatchNorm state changed')

    def save(self, path, validation=None):
        self.assert_teacher_frozen()
        start = time.perf_counter()
        save_checkpoint(path, self.model, self.optimizer, self.epoch - 1, {
            'identity': self.identity, 'config': self.config, 'model_contract': model_contract(self.config),
            'scheduler': self.scheduler.state_dict(), 'scaler': self.scaler.state_dict(),
            'global_step': self.global_step, 'history': self.history, 'best_validation_absrel': self.best,
            'overflow_retries_total': self.overflow_retries_total,
            'rng': self.rng(), 'validation': validation, 'model_stats': model_stats(self.model),
            'best_checkpoint_sha256': self.best_checkpoint_sha256})
        return time.perf_counter() - start


def compare_states(first, second):
    differences = {k: float((v.cpu().double() - second[k].cpu().double()).abs().max())
                   for k, v in first.items()}
    passed = all(torch.allclose(v.cpu(), second[k].cpu(), atol=1e-6, rtol=1e-5)
                 for k, v in first.items())
    return {'passed': passed, 'max_absolute_difference': max(differences.values()),
            'atol': 1e-6, 'rtol': 1e-5}


def smoke(factory, out):
    engine = factory()
    initial = state_digest(engine.model)
    initial_params = {k: v.detach().cpu().clone() for k, v in engine.model.named_parameters()
                      if k in ('encoder.conv1.weight', 'encoder.0.0.weight', 'head.weight')}
    assert not engine.optimizer.state and engine.global_step == engine.epoch == 0 and not engine.history
    traces = engine.train_epoch(limit=2, trace=True)
    engine.history.append({'software_epoch': engine.epoch, 'steps': engine.global_step})
    engine.save(out / 'last.pt')
    # Continue the same run, then independently reconstruct/restore the boundary.
    continuation = engine.train_epoch(limit=2, trace=True)
    expected = {k: v.detach().cpu().clone() for k, v in engine.model.state_dict().items()}
    expected_steps = engine.global_step
    engine.assert_teacher_frozen()
    updates = {k: not torch.equal(v, dict(engine.model.named_parameters())[k].detach().cpu())
               for k, v in initial_params.items()}
    assert updates and all(updates.values()), 'Encoder/head did not update'
    del engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    resumed = factory(resume=out / 'last.pt')
    assert resumed.epoch == 1 and resumed.global_step == 2 and len(resumed.history) == 1
    actual = resumed.train_epoch(limit=2, trace=True)
    match = compare_states(expected, resumed.model.state_dict())
    assert match['passed'], match
    assert expected_steps == resumed.global_step == 4
    assert [(r['rgb_sha256'], r['frames']) for r in continuation] == [(r['rgb_sha256'], r['frames']) for r in actual]
    resumed.history.append({'software_epoch': resumed.epoch, 'steps': resumed.global_step})
    resumed.save(out / 'last.pt')
    resumed.assert_teacher_frozen()
    result = {'passed': True, 'scope': 'software check using fixed real TRAIN inputs; no accuracy evidence',
              'initial_model_state_sha256': initial, 'identity': resumed.identity,
              'encoder_head_updates': updates, 'resume_weight_comparison': match,
              'optimizer_state_restored': bool(resumed.optimizer.state), 'global_step': resumed.global_step,
              'teacher_frozen': True if resumed.teacher is not None else None,
              'first_epoch': traces, 'second_epoch': actual, 'model': model_stats(resumed.model)}
    write_json(out / 'smoke.json', result)
    return result


def benchmark(engine, out, warmup=2, measured=8):
    if engine.device.type != 'cuda':
        raise ValueError('Laptop benchmark requires CUDA; no CPU timing substituted')
    iterator = iter(engine.train_loader)
    rows = []
    try:
        for i in range(warmup + measured):
            if i == warmup:
                engine.sync()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
            wait_start = time.perf_counter()
            batch = next(iterator)
            waited = time.perf_counter() - wait_start
            row = engine.step(batch)
            row['loader_wait_seconds'] = waited
            if i >= warmup:
                rows.append(row)
        engine.sync()
        wall = time.perf_counter() - started
        peak = {'allocated_bytes': torch.cuda.max_memory_allocated(), 'reserved_bytes': torch.cuda.max_memory_reserved()}
    finally:
        del iterator
    engine.assert_teacher_frozen()
    # Evaluation timing uses TRAIN images only; it is an overhead proxy, not VAL accuracy.
    eval_timing = engine.evaluate(list(engine.train_loader.dataset.records), 'train', limit=4)
    engine.epoch = 1
    checkpoint_seconds = engine.save(out / 'benchmark.pt')
    recovery_start = time.perf_counter()
    saved_sha = sha256_file(out / 'benchmark.pt')
    copy_checkpoint_exact(out / 'benchmark.pt', out / 'previous_best.pt', saved_sha)
    assert sha256_file(out / 'benchmark.pt') == saved_sha
    recovery_seconds = time.perf_counter() - recovery_start
    examples = sum(r['examples'] for r in rows)
    train_epoch = 44098 / (examples / wall)
    validation_proxy = 4410 / (eval_timing['examples'] / eval_timing['seconds'])
    epoch_total = train_epoch + validation_proxy + 2 * checkpoint_seconds + recovery_seconds
    result = {'scope': 'bounded real TRAIN-only timing, no model quality claim', 'identity': engine.identity,
              'hardware': torch.cuda.get_device_name(), 'model': model_stats(engine.model),
              'warmup_batches_excluded': warmup, 'measured_batches': measured, 'physical_batch_size': engine.config['train']['batch_size'],
              'effective_batch_size': engine.config['train']['effective_batch_size'], 'amp': engine.amp,
              'teacher_precision': 'FP32' if engine.teacher is not None else None,
              'examples': examples, 'wall_seconds': wall, 'examples_per_second': examples / wall,
              'teacher_compute_seconds': sum(r['teacher_seconds'] for r in rows),
              'compute_seconds': sum(r['compute_seconds'] for r in rows),
              'loader_wait_seconds': sum(r['loader_wait_seconds'] for r in rows), 'peak_gpu': peak,
              'checkpoint_write_seconds_measured': checkpoint_seconds,
              'checkpoint_recovery_overhead_seconds_measured': recovery_seconds,
              'fp32_train_input_eval_proxy': {'examples': eval_timing['examples'], 'seconds': eval_timing['seconds']},
              'extrapolation': {'train_epoch_hours': train_epoch / 3600,
                  'validation_proxy_hours': validation_proxy / 3600,
                  'two_checkpoint_writes_hours': 2 * checkpoint_seconds / 3600,
                  'checkpoint_recovery_hours': recovery_seconds / 3600,
                  'epoch_total_hours': epoch_total / 3600, 'candidate_epochs': engine.config['train']['epochs'],
                  'candidate_total_hours': epoch_total * engine.config['train']['epochs'] / 3600,
                  'candidate_total_hours_plus_20_percent': epoch_total * engine.config['train']['epochs'] / 3000},
              'conditions': 'Two loader workers; external HDD; OS cache uncontrolled; same fixed 64 TRAIN frames. Short timing is not full-dataset I/O measurement. CUDA synchronized per step. Validation overhead extrapolates FP32 TRAIN inference, not validation performance. Two full checkpoint writes and one atomic prior-best recovery copy plus hash checks per epoch are conservatively assumed.',
              'batches': rows}
    write_json(out / 'benchmark.json', result)
    return result


def full_training(engine, out):
    if engine.scope != 'full':
        raise ValueError('Only full scope can launch training')
    if not engine.val_records:
        raise ValueError('Full training requires unchanged validation')
    for _ in range(engine.epoch, engine.config['train']['epochs']):
        started = time.perf_counter()
        rows = engine.train_epoch()
        validation = engine.evaluate(engine.val_records, 'val')
        reference = json.loads((PROJECT / 'results/expanded_validation/validation_metrics.json').read_text())
        for tr, metrics in validation['per_trajectory'].items():
            for field in ('valid_pixels', 'near_valid_pixels'):
                if metrics[field] != reference['per_trajectory'][tr][field]:
                    raise ValueError('Frozen validation pixel counts changed')
        value = validation['overall']['absrel']
        if value is None or not math.isfinite(value):
            raise FloatingPointError('Undefined validation selection metric')
        improved = value < engine.best
        engine.best = min(engine.best, value)
        engine.history.append({'epoch': engine.epoch, 'global_step': engine.global_step,
                               'train_mean_loss': sum(x['loss'] for x in rows) / len(rows),
                               'teacher_compute_seconds': sum(x['teacher_seconds'] for x in rows),
                               'overflow_retries': sum(x['overflow_retries'] for x in rows),
                               'seconds': time.perf_counter() - started, 'validation': validation})
        if improved:
            if engine.best_checkpoint_sha256:
                ensure_prior_best(out, engine.best_checkpoint_sha256)
                copy_checkpoint_exact(out / 'best.pt', out / 'previous_best.pt', engine.best_checkpoint_sha256)
            engine.save(out / 'best.pt', validation)
            engine.best_checkpoint_sha256 = sha256_file(out / 'best.pt')
        engine.save(out / 'last.pt', validation)
        write_json(out / 'history.json', engine.history)


def make_gate(baseline, teacher):
    b = torch.load(baseline, map_location='cpu', weights_only=False)
    t = torch.load(teacher, map_location='cpu', weights_only=False)
    for p, arm in [(b, 'supervised'), (t, 'teacher')]:
        identity = p['extra']['identity']
        if identity['scope'] != 'full' or identity['arm'] != arm or identity['manifest_sha256'] != MANIFEST_SHA:
            raise ValueError('Gate requires expanded full-run baseline/teacher checkpoints')
        require_identity(identity['source_sha256'], source_hashes())
        if not p['extra'].get('validation') or p['extra']['validation']['precision'] != 'FP32':
            raise ValueError('Gate requires recorded FP32 validation')
        if p['extra']['validation']['overall']['absrel'] != p['extra']['best_validation_absrel']:
            raise ValueError('Gate requires validation-selected best checkpoints')
        expected = {'Kite_training/foggy/trajectory_2003', 'PLE_training/spring/trajectory_5000'}
        if set(p['extra']['validation']['per_trajectory']) != expected:
            raise ValueError('Gate trajectory scope changed')
    result = teacher_gate(b['extra']['validation'], t['extra']['validation'])
    return result | {'teacher_sha256': sha256_file(teacher), 'baseline_sha256': sha256_file(baseline),
                     'baseline_checkpoint': str(Path(baseline).absolute()), 'manifest_sha256': MANIFEST_SHA,
                     'teacher_validation': t['extra']['validation'], 'baseline_validation': b['extra']['validation']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('smoke', 'benchmark', 'train', 'gate'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--manifest', type=Path, default=PREP / 'manifest_train_val.json.gz')
    parser.add_argument('--reservations', type=Path, default=PREP / 'reservations.json')
    parser.add_argument('--output', type=Path, required=True)
    init = parser.add_mutually_exclusive_group()
    init.add_argument('--initialize', type=Path)
    init.add_argument('--resume', type=Path)
    parser.add_argument('--teacher', type=Path)
    parser.add_argument('--teacher-sha256')
    parser.add_argument('--gate', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--confirm-full-training', action='store_true', help='Explicit operator launch after protocol review; never used by smoke/benchmark')
    a = parser.parse_args(argv)
    out = a.output.absolute()
    if a.mode == 'gate':
        if not a.baseline or not a.teacher or out.exists():
            parser.error('gate requires --baseline, --teacher and a new output JSON')
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json(out, make_gate(a.baseline, a.teacher))
        return
    if not a.config or not a.dataset_root:
        parser.error('--config and --dataset-root required')
    root = a.dataset_root.absolute()
    if out == root or root in out.parents:
        parser.error('Outputs cannot be inside the dataset')
    if a.mode == 'train' and not a.confirm_full_training:
        parser.error('Full training requires explicit --confirm-full-training after protocol review')
    if a.resume and a.mode != 'train':
        parser.error('External resume is only accepted for full mode; smoke checks resume internally')
    if a.resume:
        if a.resume.absolute() != out / 'last.pt':
            parser.error('Resume the output directory own last.pt; choose a fresh run otherwise')
    elif out.exists():
        raise FileExistsError('Use a fresh output directory; preserve existing work')
    config = yaml.safe_load(a.config.read_text())
    config['device'] = a.device
    check_config(config)
    records, m = load_inputs(a.manifest, a.reservations, root)
    scope = 'full' if a.mode == 'train' else a.mode
    if scope != 'full':
        records = {'train': fixed_training_subset(records['train']), 'val': []}
    guard = ImageReadGuard(root, records['train'] + records['val'])
    sys.addaudithook(guard.check)
    out.mkdir(parents=True, exist_ok=bool(a.resume))
    if scope != 'full':
        write_json(out / 'fixed_training_subset.json', [asdict(r) for r in records['train']])
    def factory(resume=None):
        return Experiment(config, records, root, m, a.reservations, scope,
                          initialize=a.initialize if not resume else None, resume=resume,
                          teacher=a.teacher, teacher_sha=a.teacher_sha256, gate=a.gate)
    # Validate resume identity before changing any existing run evidence.
    resumed_engine = factory(a.resume) if a.resume else None
    provenance_path = out / ('resume_provenance_' + str(time.time_ns()) + '.json' if a.resume else 'provenance.json')
    write_json(provenance_path, run_metadata(config, a.manifest) | {
        'manifest_uncompressed_sha256': MANIFEST_SHA, 'reservations_sha256': sha256_file(a.reservations),
        'scope': scope, 'source_sha256': source_hashes(), 'dataset_root': str(root)})
    if a.mode == 'smoke':
        result = smoke(factory, out)
    elif a.mode == 'benchmark':
        result = benchmark(factory(), out)
    else:
        full_training(resumed_engine or factory(), out)
        result = {'status': 'full_run_complete'}
    print(json.dumps(clean_json({k: result[k] for k in result if k in
        ('passed', 'status', 'model', 'examples_per_second', 'peak_gpu', 'extrapolation')})), flush=True)
