from __future__ import annotations

import torch

from aerodepth_lite.scheduler import AdaptiveScheduler, motion_score, near_field_risk, visual_change


def config():
    return {"max_skip": 2, "decision_threshold": 2.0, "motion_scale": 10, "visual_scale": 1, "near_threshold": 3, "critical_risk": 2, "extreme_motion": 2, "extreme_visual": 2}


def test_visual_change_normalized_and_deterministic():
    black, white = torch.zeros(3, 50, 80), torch.ones(3, 50, 80)
    assert visual_change(black, black) == 0.0
    assert visual_change(black, white) == 1.0


def test_motion_normalization_and_missing_signal():
    assert motion_score({}, 5) == (0.0, False)
    score, available = motion_score({"vx": 3, "vy": 4, "vz": 0}, 10)
    assert score == 0.5 and available


def test_robust_near_field_risk():
    far = torch.full((1, 20, 30), 20.0)
    far[0, 10, 15] = 0.01
    close = torch.full((1, 20, 30), 1.0)
    far_risk, _ = near_field_risk(far, 3)
    close_risk, _ = near_field_risk(close, 3)
    assert far_risk < 0.1
    assert close_risk > 0.8


def test_first_frame_and_max_skip_are_forced():
    scheduler = AdaptiveScheduler(config())
    rgb, depth = torch.zeros(3, 20, 30), torch.full((1, 20, 30), 20.0)
    first = scheduler.decide(rgb, {}, None)
    assert first.infer and first.reason == "first_frame"
    assert not scheduler.decide(rgb, {}, depth).infer
    assert not scheduler.decide(rgb, {}, depth).infer
    forced = scheduler.decide(rgb, {}, depth)
    assert forced.infer and forced.max_skip_forced and forced.reason == "max_skip"


def test_reset_forces_new_trajectory_first_frame():
    scheduler = AdaptiveScheduler(config())
    rgb = torch.zeros(3, 20, 30)
    scheduler.decide(rgb, {}, None)
    scheduler.reset()
    assert scheduler.decide(rgb, {}, None).reason == "first_frame"

