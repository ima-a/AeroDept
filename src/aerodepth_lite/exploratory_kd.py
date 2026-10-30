"""Explicitly authorized rejected-teacher exploration, separate from gated KD.

The parent full KD constructor and gate are unchanged. This distinct scope uses
its verified training step, FP32 evaluation and atomic checkpoint serialization.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import torch
import yaml

from . import expanded as e
from .model import model_contract

ARM = 'exploratory_kd_rejected_teacher_v1'
SCOPE = 'exploratory_full_rejected_teacher_v1'
FILES = ('src/aerodepth_lite/exploratory_kd.py', 'scripts/exploratory_kd_training.py')


def implementation_hashes():
    return {name: e.sha256_file(e.PROJECT / name) for name in FILES}


def read_authorization(path, expected_sha):
    if not expected_sha or e.sha256_file(path) != expected_sha:
        raise ValueError('Exploratory authorization SHA-256 mismatch')
    auth = json.loads(Path(path).read_text())
    if (auth.get('arm') != ARM or auth.get('scope') != SCOPE or
            auth.get('explicit_rejected_teacher_authorization') is not True or
            auth.get('original_gate_passed') is not False or
            auth.get('allow_teacher_repair') is not False or
            auth.get('allow_test_access') is not False or auth.get('maximum_runs') != 1):
        raise ValueError('Missing/scoped exploratory authorization')
    e.require_identity(auth['parent_source_sha256'], e.source_hashes())
    e.require_identity(auth['exploratory_source_sha256'], implementation_hashes())
    for name, item in auth['inputs'].items():
        if e.sha256_file(item['path']) != item['sha256']:
            raise ValueError('Exploratory pinned input changed: ' + name)
    return auth


# Lineage
def validate_lineage(auth, config):
    """Metadata only. Recompute rejection without rewriting historical paths."""
    e.check_config(config)
    if config['arm'] != 'kd' or config.get('exploratory_arm') != ARM:
        raise ValueError('Exploratory config/arm mismatch')
    inputs = auth['inputs']
    if e.sha256_file(inputs['config']['path']) != auth['config_sha256']:
        raise ValueError('Exploratory configuration identity changed')
    if yaml.safe_load(Path(inputs['config']['path']).read_text()) != config:
        raise ValueError('Runtime configuration differs from frozen exploratory config')
    b = torch.load(inputs['baseline_best']['path'], map_location='cpu', weights_only=False)
    t = torch.load(inputs['teacher_best']['path'], map_location='cpu', weights_only=False)
    for payload, arm, receipt_name, final_epochs in [
            (b, 'supervised', 'baseline_completion', 10),
            (t, 'teacher', 'teacher_completion', 20)]:
        identity = payload['extra']['identity']
        if identity['arm'] != arm or identity['scope'] != 'full' or identity['manifest_sha256'] != e.MANIFEST_SHA:
            raise ValueError('Parent checkpoint provenance changed')
        e.require_identity(identity['source_sha256'], auth['parent_source_sha256'])
        receipt = json.loads(Path(inputs[receipt_name]['path']).read_text())
        key = 'baseline_best' if arm == 'supervised' else 'teacher_best'
        best = receipt['checkpoints']['best.pt']
        if (best['sha256'] != inputs[key]['sha256'] or payload['epoch'] + 1 != best['epoch_completed'] or
                receipt['checkpoints']['last.pt']['epoch_completed'] != final_epochs or
                receipt['checkpoints']['last.pt']['global_step'] != final_epochs * 22049 or
                receipt['freeze_sha256'] != auth['parent_freeze_sha256']):
            raise ValueError('Completion/selected checkpoint identity mismatch')
        if payload['extra']['validation']['overall']['absrel'] != payload['extra']['best_validation_absrel']:
            raise ValueError('Parent checkpoint was not validation-selected best')
    if t['epoch'] != 2 or inputs['teacher_best']['sha256'] != auth['official_teacher_best_sha256']:
        raise ValueError('Exploration requires official selected epoch-3 teacher B best')
    if b['extra']['identity']['starting_weights'] != {'kind': 'published_MobileNet_epoch4', 'sha256': e.INITIAL_SHA}:
        raise ValueError('Supervised A initialization lineage changed')
    for section in ('data', 'train', 'seed', 'model', 'loss', 'device'):
        if b['extra']['config'][section] != config[section]:
            raise ValueError('Matched A/exploratory KD controls differ: ' + section)
    if config['train']['epochs'] != 10 or config['kd_weight'] != .2 or config['seed'] != 42:
        raise ValueError('Exploratory budget/weight/seed changed')
    if config['train'] != auth['matched_train_config']:
        raise ValueError('Matched training policy changed')
    if config['loss']['near_weight'] != 0 or inputs['initialization']['sha256'] != e.INITIAL_SHA:
        raise ValueError('Initialization or supervised loss changed')
    gate = json.loads(Path(inputs['failed_gate']['path']).read_text())
    if gate.get('passed') is not False or gate['teacher_sha256'] != inputs['teacher_best']['sha256']:
        raise ValueError('Original rejected gate identity/status changed')
    recomputed = e.make_gate(inputs['baseline_best']['path'], inputs['teacher_best']['path'])
    # The receipt keeps its original HDD baseline path. Compare only that path
    # after resolving the byte-identical recovery copy; never write a new gate.
    recomputed['baseline_checkpoint'] = gate['baseline_checkpoint']
    if recomputed != gate or recomputed['passed']:
        raise ValueError('Historical gate no longer reproduces the same rejection')
    return {'original_gate_still_rejected': True, 'gate_sha256': inputs['failed_gate']['sha256'],
            'baseline_path_relocation_only': True, 'checks': gate['checks']}


# Exploration
class ExploratoryExperiment(e.Experiment):
    def __init__(self, config, records, root, manifest_info, reservations, authorization,
                 authorization_sha, resume=None, progress=None):
        self.authorization = read_authorization(authorization, authorization_sha)
        self.lineage = validate_lineage(self.authorization, config)
        inputs = self.authorization['inputs']
        # This explicit, separately authorized scope is never represented as an
        # accepted full gated KD arm. Parent source/checkpoint guards still run.
        super().__init__(config, records, root, manifest_info, reservations, SCOPE,
                         initialize=inputs['initialization']['path'],
                         teacher=inputs['teacher_best']['path'], teacher_sha=inputs['teacher_best']['sha256'])
        self.identity.update(arm=ARM, experiment_id=self.authorization['experiment_id'],
                             exploratory_authorization_sha256=authorization_sha,
                             exploratory_source_sha256=implementation_hashes(),
                             rejected_gate_sha256=inputs['failed_gate']['sha256'],
                             original_gate_passed=False,
                             parent_baseline_sha256=inputs['baseline_best']['sha256'],
                             parent_freeze_sha256=self.authorization['parent_freeze_sha256'],
                             dataset_root=str(Path(root).absolute()))
        if state_hash := self.authorization.get('initial_student_state_sha256'):
            if e.state_digest(self.model) != state_hash:
                raise ValueError('Initial complete student/BN state differs from original export')
        if self.optimizer.state or self.epoch != 0 or self.global_step != 0 or self.history:
            raise ValueError('Fresh student training state was not empty')
        if self.scaler.get_scale() != config['train']['amp_initial_scale']:
            raise ValueError('Initial scaler state differs')
        self.initial_student_digest = e.state_digest(self.model)
        self.on_progress = progress
        self.initial_input_contract_checked = False
        if resume:
            self.restore_exploratory_checkpoint(resume)
        self.assert_teacher_frozen()

    def restore_exploratory_checkpoint(self, path):
        cp = torch.load(path, map_location='cpu', weights_only=False)
        extra = cp['extra']
        e.require_identity(extra['identity'], self.identity)
        epoch, step = cp['epoch'] + 1, extra['global_step']
        if not (1 <= epoch <= 10 and step == epoch * 22049 and len(extra['history']) == epoch):
            raise ValueError('Exploratory resume requires a complete own epoch boundary')
        e.ensure_prior_best(Path(path).parent, extra['best_checkpoint_sha256'])
        self.model.load_state_dict(cp['model'], strict=True)
        self.optimizer.load_state_dict(cp['optimizer'])
        self.scheduler.load_state_dict(extra['scheduler'])
        self.scaler.load_state_dict(extra['scaler'])
        self.epoch, self.global_step = epoch, step
        self.overflow_retries_total = extra['overflow_retries_total']
        self.history, self.best = extra['history'], extra['best_validation_absrel']
        self.best_checkpoint_sha256 = extra['best_checkpoint_sha256']
        self.restore_rng(extra['rng'])

    def step(self, batch, trace=False):
        prior = self.global_step
        seen, handles = {}, []
        if not self.initial_input_contract_checked:
            def observe(name):
                def hook(module, args, output):
                    x = args[0]
                    seen.setdefault(name, []).append({
                        'rgb_sha256': hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                        'input_dtype': str(x.dtype), 'device': str(x.device),
                        'output_dtype': str(output.dtype), 'output_requires_grad': output.requires_grad,
                        'mode_training': module.training})
                return hook
            handles = [self.model.register_forward_hook(observe('student')),
                       self.teacher.register_forward_hook(observe('teacher'))]
        try:
            row = super().step(batch, trace)
        finally:
            for handle in handles:
                handle.remove()
        if self.global_step != prior + 1 or not all(math.isfinite(row[k]) for k in ('loss', 'supervised_loss', 'kd_loss')):
            raise AssertionError('Successful-update counter or finite loss invariant failed')
        if seen:
            teacher = seen['teacher']
            if len(teacher) != 1 or teacher[0]['output_dtype'] != 'torch.float32' or teacher[0]['output_requires_grad'] or teacher[0]['mode_training']:
                raise AssertionError('Teacher FP32/no-grad/eval contract failed')
            if any(x['rgb_sha256'] != teacher[0]['rgb_sha256'] for x in seen['student']):
                raise AssertionError('Teacher and student did not receive identical augmented RGB')
            self.initial_input_contract_checked = True
            self.input_contract_observation = seen
        if self.global_step <= 3 or self.global_step % 250 == 0:
            self.assert_teacher_frozen()
        if self.on_progress and (self.global_step <= 3 or self.global_step % 25 == 0 and self.global_step <= 100 or self.global_step % 250 == 0):
            self.on_progress(self, row)
        return row


# Training
def train_exploratory(engine, out):
    """Same epoch orchestration as parent full_training, with distinct scope."""
    if engine.scope != SCOPE or engine.identity['arm'] != ARM or not engine.val_records:
        raise ValueError('Exploratory full scope/validation required')
    for _ in range(engine.epoch, engine.config['train']['epochs']):
        started = time.perf_counter()
        rows = engine.train_epoch()
        validation = engine.evaluate(engine.val_records, 'val')
        reference = json.loads((e.PROJECT / 'results/expanded_validation/validation_metrics.json').read_text())
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
                e.ensure_prior_best(out, engine.best_checkpoint_sha256)
                e.copy_checkpoint_exact(out / 'best.pt', out / 'previous_best.pt', engine.best_checkpoint_sha256)
            engine.save(out / 'best.pt', validation)
            engine.best_checkpoint_sha256 = e.sha256_file(out / 'best.pt')
        engine.save(out / 'last.pt', validation)
        e.write_json(out / 'history.json', engine.history)
        if engine.on_progress:
            engine.on_progress(engine, {'epoch_completed': engine.epoch, 'validation': validation})
    if engine.epoch != 10 or engine.global_step != 220490:
        raise ValueError('Exploratory completed budget mismatch')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--authorization-sha256', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    bundle = args.bundle.absolute(); out = bundle / 'run'
    lock = (bundle / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'arm': ARM, 'scope': SCOPE, 'trainer_pid': os.getpid(), 'unit': os.environ.get('AERODEPTH_UNIT'),
             'original_gate_passed': False, 'status': 'preflight', 'manual_resume': args.resume}
    def event(**values):
        state.update(values);state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        e.write_json(bundle / 'status.json', state)
        print(json.dumps(e.clean_json(state)), flush=True)
    def stopping(signum, frame):
        event(status='stopped', signal=signum, checkpoint_policy='Resume own last.pt at last completed epoch; unfinished epoch replays')
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stopping);signal.signal(signal.SIGINT, stopping)
    try:
        freeze = json.loads((bundle / 'freeze.json').read_text())
        if e.sha256_file(bundle / 'freeze.json') != (bundle / 'freeze.sha256').read_text().split()[0]:
            raise ValueError('Freeze index identity changed')
        if not os.path.ismount(freeze['external_mount']):
            raise RuntimeError('Rescue USB mount unavailable')
        import subprocess
        actual_uuid = subprocess.check_output(['findmnt', '-n', '-o', 'UUID', '--target', str(bundle)], text=True).strip()
        if actual_uuid != freeze['external_uuid']:
            raise RuntimeError('Rescue USB mount identity changed')
        mains = [p.parent for p in Path('/sys/class/power_supply').glob('*/type')
                 if p.read_text().strip() == 'Mains']
        if not mains or not any((p / 'online').read_text().strip() == '1' for p in mains):
            raise RuntimeError('AC power is not observably connected')
        for name, sha in freeze['files_sha256'].items():
            if e.sha256_file(bundle / name) != sha:
                raise ValueError('Frozen source/input changed: ' + name)
        if os.statvfs(bundle).f_bavail * os.statvfs(bundle).f_frsize < 5 * 1024 ** 3:
            raise RuntimeError('Less than 5 GiB free for safe checkpoints')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
        torch.set_num_threads(2)
        auth_path = bundle / 'authorization.json'
        auth = read_authorization(auth_path, args.authorization_sha256)
        if Path(auth['authorized_bundle']).absolute() != bundle:
            raise ValueError('Authorization is restricted to its exact output bundle')
        config = yaml.safe_load(Path(auth['inputs']['config']['path']).read_text())
        root = Path(auth['dataset_root'])
        records, manifest = e.load_inputs(e.PREP / 'manifest_train_val.json.gz', e.PREP / 'reservations.json', root)
        guard = e.ImageReadGuard(root, records['train'] + records['val'])
        sys.addaudithook(guard.check)
        resume = out / 'last.pt' if args.resume else None
        if args.resume:
            if not resume.is_file():
                raise ValueError('No own last.pt; incomplete first epoch cannot resume')
        elif out.exists():
            raise FileExistsError('Exploratory output already exists; never restart/overwrite it')
        def progress(engine, row):
            event(status='running', epoch=engine.epoch + (0 if 'epoch_completed' in row else 1),
                  global_step=engine.global_step, total_updates=220490, completed_epochs=engine.epoch,
                  progress=row, overflow_retries_total=engine.overflow_retries_total,
                  amp_scale=engine.scaler.get_scale(), cuda_device=str(engine.device),
                  input_contract_verified=engine.initial_input_contract_checked)
        engine = ExploratoryExperiment(config, records, root, manifest, e.PREP / 'reservations.json',
                                       auth_path, args.authorization_sha256, resume=resume, progress=progress)
        out.mkdir(exist_ok=args.resume)
        provenance = {'authorization_sha256':args.authorization_sha256,'identity':engine.identity,
                      'initial_student_state_sha256':engine.initial_student_digest,'teacher_state_sha256':engine.teacher_digest,
                      'fresh_optimizer_on_new_run':not args.resume and not engine.optimizer.state,
                      'initialization_is_original_epoch4':True,'initialization_is_completed_A':False,
                      'cuda':torch.version.cuda,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
                      'lineage':engine.lineage,'train_frames':len(records['train']),'val_frames':len(records['val'])}
        e.write_json(out / ('resume_provenance_'+str(time.time_ns())+'.json' if args.resume else 'provenance.json'), provenance)
        event(status='running', epoch=engine.epoch+1, global_step=engine.global_step, initial_state_verified=True)
        train_exploratory(engine, out)
        engine.assert_teacher_frozen()
        e.write_json(bundle / 'completion.json', {'status':'complete_exploratory_rejected_teacher_KD',
                     'epochs':engine.epoch,'global_step':engine.global_step,'identity':engine.identity,
                     'original_gate_passed':False,'deployment_promotion':False,
                     'checkpoints':{n:e.sha256_file(out/n) for n in ['best.pt','last.pt']}})
        event(status='complete', completed_epochs=10, global_step=220490)
    except SystemExit:
        raise
    except BaseException as exc:
        event(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
