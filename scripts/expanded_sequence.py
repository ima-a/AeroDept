"""Sequential supervisor for a frozen, explicitly authorized expanded experiment.

Calls the verified expanded_training.py CLI. Never retries, changes budgets or
opens dataset images. Manual --resume-sequence uses only existing last.pt files.
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
import subprocess
import sys
import traceback


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--resume-sequence', action='store_true')
    args = parser.parse_args()
    bundle = args.bundle.absolute()
    frozen = bundle / 'frozen'
    freeze = read(bundle / 'freeze.json')
    lock = (bundle / 'sequence.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'supervisor_pid': os.getpid(), 'stage': 'preflight', 'status': 'starting',
             'manual_resume': args.resume_sequence, 'freeze_sha256': sha(bundle / 'freeze.json'),
             'systemd_unit': os.environ.get('AERODEPTH_UNIT'), 'child_pid': None}
    child = None

    def event(**values):
        state.update(values)
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        write(bundle / 'status.json', state)
        with (bundle / 'events.jsonl').open('a') as handle:
            handle.write(json.dumps(state, allow_nan=False) + '\n')
        print(json.dumps(state, allow_nan=False), flush=True)

    def stopping(signum, _frame):
        event(status='stopped', stop_signal=signum)
        if child is not None and child.poll() is None:
            child.terminate()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stopping)
    signal.signal(signal.SIGINT, stopping)

    def verify_freeze():
        if sha(bundle / 'freeze.json') != (bundle / 'freeze.sha256').read_text().split()[0]:
            raise RuntimeError('Frozen protocol index identity changed')
        if not os.path.ismount(freeze['external_mount']):
            raise RuntimeError('External disk mount is unavailable')
        uuid = subprocess.check_output(['findmnt', '-n', '-o', 'UUID', '--target', str(bundle)], text=True).strip()
        if uuid != freeze['external_uuid']:
            raise RuntimeError('External disk identity changed')
        available = os.statvfs(bundle).f_bavail * os.statvfs(bundle).f_frsize
        if available < 5 * 1024 ** 3:
            raise RuntimeError('Less than 5 GiB available for safe checkpoint writes')
        for name, expected in freeze['files_sha256'].items():
            if sha(bundle / name) != expected:
                raise RuntimeError('Frozen input/source identity changed: ' + name)

    def command(argv, stage):
        nonlocal child
        log = bundle / 'logs' / (stage + '.log')
        event(stage=stage, status='launching', command=argv, log=str(log), child_pid=None)
        with log.open('a', buffering=1) as output:
            output.write('\n' + json.dumps({'launch_utc': state['updated_utc'], 'argv': argv}) + '\n')
            child = subprocess.Popen(argv, cwd=frozen, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
            event(status='running', child_pid=child.pid)
            for line in child.stdout:
                output.write(line)
                output.flush()
                try:
                    progress = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if isinstance(progress, dict) and 'batches' in progress:
                    event(progress=progress, finite_checks_completed_through_reported_batch=True)
            code = child.wait()
            event(child_exit_code=code, child_pid=None)
            child = None
            if code:
                raise RuntimeError(f'{stage} exited with status {code}; no retry or later stage launched')

    def validate_completed(arm, directory):
        import torch
        import yaml
        from aerodepth_lite import expanded as e
        config = yaml.safe_load((frozen / f'configs/expanded_{arm}.yaml').read_text()) | {'device': 'cuda'}
        result = {}
        for name in ('last.pt', 'best.pt'):
            path = directory / name
            payload = torch.load(path, map_location='cpu', weights_only=False)
            extra, expected_epochs = payload['extra'], config['train']['epochs']
            identity = extra['identity']
            if (identity['scope'] != 'full' or identity['arm'] != arm or identity['config'] != config or
                    identity['manifest_sha256'] != e.MANIFEST_SHA or
                    identity['reservations_sha256'] != freeze['reservations_sha256'] or
                    identity['source_sha256'] != e.source_hashes() or
                    identity['model_contract'] != e.model_contract(config)):
                raise RuntimeError('Completed checkpoint identity mismatch: ' + str(path))
            expected_start = freeze['teacher_initialization'] if arm == 'teacher' else {
                'kind': 'published_MobileNet_epoch4', 'sha256': e.INITIAL_SHA}
            if identity['starting_weights'] != expected_start:
                raise RuntimeError('Starting weights changed: ' + str(path))
            if not extra['validation'] or extra['validation']['precision'] != 'FP32':
                raise RuntimeError('Missing FP32 validation: ' + str(path))
            metric = extra['validation']['overall']['absrel']
            if not isinstance(metric, (int, float)) or not math.isfinite(metric):
                raise RuntimeError('Nonfinite validation checkpoint metric')
            if name == 'last.pt':
                if (payload['epoch'] + 1 != expected_epochs or
                        extra['global_step'] != expected_epochs * 22049 or
                        len(extra['history']) != expected_epochs):
                    raise RuntimeError('Training budget incomplete: ' + str(path))
                expected_best = extra['best_checkpoint_sha256']
            elif metric != extra['best_validation_absrel']:
                raise RuntimeError('best.pt was not selected by validation AbsRel')
            result[name] = {'sha256': sha(path), 'epoch_completed': payload['epoch'] + 1,
                            'global_step': extra['global_step'], 'validation_absrel': metric}
            del payload, extra, identity
        if result['best.pt']['sha256'] != expected_best:
            raise RuntimeError('best.pt differs from last.pt pinned identity')
        return result

    try:
        event()
        verify_freeze()
        for arm, stage in [('supervised', 'A_supervised'), ('teacher', 'B_teacher'), ('kd', 'D_kd')]:
            verify_freeze()
            if arm == 'kd':
                gate_path = bundle / 'teacher_gate.json'
                if not gate_path.exists():
                    command([sys.executable, '-u', '-B', str(frozen / 'scripts/expanded_training.py'),
                             'gate', '--baseline', str(bundle / 'runs/A_supervised/best.pt'),
                             '--teacher', str(bundle / 'runs/B_teacher/best.pt'), '--output', str(gate_path)], 'C_teacher_gate')
                from aerodepth_lite.expanded import make_gate
                gate = read(gate_path)
                if gate != make_gate(bundle / 'runs/A_supervised/best.pt', bundle / 'runs/B_teacher/best.pt'):
                    raise RuntimeError('Saved teacher gate identity/policy mismatch')
                event(stage='C_teacher_gate', teacher_gate_passed=gate['passed'], teacher_gate_sha256=sha(gate_path))
                if not gate['passed']:
                    event(status='teacher_rejected', child_pid=None)
                    return 2
            directory = bundle / 'runs' / stage
            completion = bundle / 'completion' / (stage + '.json')
            if completion.exists():
                if not args.resume_sequence or read(completion)['checkpoints'] != validate_completed(arm, directory):
                    raise RuntimeError('Existing completion cannot be reused or its identity changed')
                event(stage=stage, status='verified_complete_skipped_on_manual_resume')
                continue
            argv = [sys.executable, '-u', '-B', str(frozen / 'scripts/expanded_training.py'), 'train',
                    '--config', str(frozen / f'configs/expanded_{arm}.yaml'),
                    '--dataset-root', freeze['dataset_root'],
                    '--manifest', str(frozen / 'results/expanded_preparation_20261026/manifest_train_val.json.gz'),
                    '--reservations', str(frozen / 'results/expanded_preparation_20261026/reservations.json'),
                    '--device', 'cuda', '--output', str(directory), '--confirm-full-training']
            if directory.exists():
                if not args.resume_sequence or not (directory / 'last.pt').is_file():
                    raise RuntimeError('Existing output has no permitted resume. Before first checkpoint a fresh destination is required: ' + str(directory))
                argv += ['--resume', str(directory / 'last.pt')]
            elif arm != 'teacher':
                argv += ['--initialize', str(frozen / 'models/mobilenet_v3_small_epoch4.state_dict.pt')]
            if arm == 'kd':
                argv += ['--teacher', str(bundle / 'runs/B_teacher/best.pt'),
                         '--teacher-sha256', gate['teacher_sha256'], '--gate', str(gate_path)]
            command(argv, stage)
            verified = validate_completed(arm, directory)
            write(completion, {'arm': arm, 'checkpoints': verified, 'freeze_sha256': sha(bundle / 'freeze.json')})
            event(stage=stage, status='verified_complete', checkpoints=verified, progress=None)
        event(stage='complete', status='complete', child_pid=None)
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        event(status='failed', error=str(exc), child_pid=None)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
