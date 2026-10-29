"""Focused authorization/lineage/resume checks; no training or image evaluation."""
import ast
import copy
import inspect
import json
import os
from pathlib import Path

import pytest
import torch
import yaml

from aerodepth_lite import expanded as e
from aerodepth_lite import exploratory_kd as x


def test_exploratory_epoch_math_matches_existing_full_runner():
    original = next(n for n in ast.parse(inspect.getsource(e.full_training)).body[0].body if isinstance(n, ast.For))
    current = next(n for n in ast.parse(inspect.getsource(x.train_exploratory)).body[0].body if isinstance(n, ast.For))
    current = copy.deepcopy(current)
    assert ast.unparse(current.body[-1].test) == 'engine.on_progress'
    current.body.pop()  # Additional progress reporting has no training effect.
    class Normalize(ast.NodeTransformer):
        def visit_Attribute(self, node):
            self.generic_visit(node)
            if isinstance(node.value, ast.Name) and node.value.id == 'e':
                return ast.Name(id=node.attr, ctx=node.ctx)
            return node
    assert ast.dump(original) == ast.dump(Normalize().visit(current))


@pytest.mark.parametrize('key', ['arm', 'scope', 'exploratory_authorization_sha256',
                                  'exploratory_source_sha256', 'rejected_gate_sha256',
                                  'parent_baseline_sha256', 'config', 'starting_weights'])
def test_exploratory_resume_identity_fails_before_restoration(key, monkeypatch):
    engine = object.__new__(x.ExploratoryExperiment)
    engine.identity = {key: 'pinned'}
    monkeypatch.setattr(torch, 'load', lambda *a, **k: {'extra': {'identity': {key: 'changed'}}})
    with pytest.raises(ValueError, match='identity'):
        engine.restore_exploratory_checkpoint('not-opened.pt')


@pytest.fixture(scope='module')
def frozen():
    path = os.environ.get('AERODEPTH_EXPLORATORY_BUNDLE')
    if not path:
        pytest.skip('Requires explicitly prepared exploratory authorization bundle')
    bundle = Path(path)
    auth_path = bundle / 'authorization.json'
    auth_sha = e.sha256_file(auth_path)
    auth = x.read_authorization(auth_path, auth_sha)
    config = yaml.safe_load(Path(auth['inputs']['config']['path']).read_text())
    return bundle, auth_path, auth_sha, auth, config


def test_actual_selected_lineage_recomputes_failed_gate(frozen):
    _, _, _, auth, config = frozen
    result = x.validate_lineage(auth, config)
    assert result['original_gate_still_rejected']
    assert not result['checks']['Kite_training/foggy/trajectory_2003']


def test_authorization_and_config_fail_closed(frozen, tmp_path):
    _, path, sha, auth, config = frozen
    with pytest.raises(ValueError, match='authorization SHA'):
        x.read_authorization(path, '0' * 64)
    for key, value in [('explicit_rejected_teacher_authorization', False),
                       ('original_gate_passed', True), ('allow_teacher_repair', True),
                       ('maximum_runs', 2)]:
        changed = auth | {key: value}
        bad = tmp_path / 'changed_auth.json';bad.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match='scoped exploratory'):
            x.read_authorization(bad, e.sha256_file(bad))
    with pytest.raises(ValueError, match='configuration differs'):
        x.validate_lineage(auth, config | {'kd_weight': .3})


def test_original_full_kd_still_rejects_actual_teacher(frozen):
    _, _, _, auth, config = frozen
    parent = config | {'device': 'cpu'}
    with pytest.raises(ValueError, match='Teacher gate does not authorize'):
        e.Experiment(parent, {'train': [], 'val': []}, Path(auth['dataset_root']), {},
                     e.PREP / 'reservations.json', 'full',
                     initialize=auth['inputs']['initialization']['path'],
                     teacher=auth['inputs']['teacher_best']['path'],
                     teacher_sha=auth['inputs']['teacher_best']['sha256'],
                     gate=auth['inputs']['failed_gate']['path'])


def test_fresh_initialization_frozen_teacher_serialization_and_boundary_guard(frozen, tmp_path):
    _, path, sha, auth, config = frozen
    records, manifest = e.load_inputs(e.PREP / 'manifest_train_val.json.gz', e.PREP / 'reservations.json', Path(auth['dataset_root']))
    records = {'train': e.fixed_training_subset(records['train']), 'val': []}
    engine = x.ExploratoryExperiment(config, records, Path(auth['dataset_root']), manifest,
                                     e.PREP / 'reservations.json', path, sha)
    expected = torch.load(auth['inputs']['initialization']['path'], map_location='cpu', weights_only=True)
    assert all(torch.equal(v.cpu(), expected[k]) for k, v in engine.model.state_dict().items())
    assert e.state_digest(engine.model) == auth['initial_student_state_sha256']
    assert not engine.optimizer.state and engine.global_step == engine.epoch == 0 and not engine.history
    assert engine.scheduler.state_dict()['last_epoch'] == 0
    assert engine.scaler.get_scale() == 1024
    assert engine.identity['arm'] == x.ARM and engine.identity['scope'] == x.SCOPE
    assert engine.identity['gate_sha256'] is None and engine.identity['original_gate_passed'] is False
    assert all(not p.requires_grad and p.grad is None and p.dtype == torch.float32 for p in engine.teacher.parameters())
    assert not any(m.training for m in engine.teacher.modules())
    assert not {id(p) for p in engine.teacher.parameters()} & {id(p) for g in engine.optimizer.param_groups for p in g['params']}
    before = e.state_digest(engine.model);teacher_before = e.state_digest(engine.teacher)
    fixture = tmp_path / 'zero_update_software_serialization.pt'
    engine.save(fixture)
    cp = torch.load(fixture, map_location='cpu', weights_only=False)
    assert cp['extra']['identity'] == engine.identity and cp['extra']['global_step'] == 0
    assert set(['rng', 'scheduler', 'scaler', 'history', 'best_checkpoint_sha256']) <= set(cp['extra'])
    assert cp['optimizer']['state'] == {}
    with pytest.raises(ValueError, match='complete own epoch boundary'):
        engine.restore_exploratory_checkpoint(fixture)
    assert e.state_digest(engine.model) == before and e.state_digest(engine.teacher) == teacher_before
    engine.assert_teacher_frozen()
