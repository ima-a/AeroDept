"""Versioned presentation scheduler. Historical V2 is deliberately untouched."""
from __future__ import annotations
import numpy as np
from .scheduler import Decision, near_field_risk
from .scheduler_v2 import SchedulerV2, thumbnail

SIGNALS = frozenset({'motion', 'visual', 'risk', 'age'})
POLICIES = ('every', 'fixed2', 'fixed5', 'budget_fixed', 'v2',
            'v2_no_motion', 'v2_no_visual', 'v2_no_risk', 'v2_no_age')
DISABLED = {'v2': (), 'v2_no_motion': ('motion',), 'v2_no_visual': ('visual',),
            'v2_no_risk': ('risk',), 'v2_no_age': ('age',)}


class PresentationScheduler(SchedulerV2):
    """Explicit fresh-prediction notification; age is never part of risk cache."""
    def __init__(self, config, disabled=(), cache_risk=True):
        self.disabled = frozenset(disabled)
        if self.disabled - SIGNALS:
            raise ValueError('Unknown ablation signal')
        self.cache_risk = cache_risk
        super().__init__(dict(config))

    def reset(self):
        super().reset()
        self.risk_value, self.risk_available = 0., False
        self.prediction_token = None
        self.risk_updates = 0

    def update_prediction(self, depth, token):
        if token is None:
            raise ValueError('Fresh prediction requires an explicit source token')
        if token == self.prediction_token:
            return
        self.prediction_token = token
        if self.cache_risk and 'risk' not in self.disabled:
            self.risk_value, self.risk_available = near_field_risk(depth, 10.)
            self.risk_updates += 1
        else:
            self.risk_value, self.risk_available = 0., False

    def decide(self, rgb, navigation, previous_depth=None):
        if 'risk' in self.disabled:
            risk, available = 0., False
        elif self.cache_risk:
            risk, available = self.risk_value, self.risk_available
        else:
            risk, available = near_field_risk(previous_depth, 10.)
        return self.decide_features(thumbnail(rgb), navigation, risk, available)

    def decide_features(self, gray, navigation, risk, risk_available):
        c = self.config
        def vector(keys):
            values = [navigation.get(k) for k in keys]
            return np.array(values, dtype=float) if all(v is not None and np.isfinite(v) for v in values) else None
        acceleration = vector(('ax', 'ay', 'az'))
        gyro = vector(('wx', 'wy', 'wz'))
        gps = vector(('gps_vx', 'gps_vy', 'gps_vz'))
        change = float(np.linalg.norm(acceleration-self.previous_acceleration)) if acceleration is not None and self.previous_acceleration is not None else 0.
        motion = max(min(change/c['accel_change_scale'],1.), min(float(np.linalg.norm(gyro))/c['gyro_scale'],1.) if gyro is not None else 0.)
        motion = max(motion,min(float(np.linalg.norm(gps))/c.get('gps_speed_scale',20.),1.) if gps is not None else 0.)
        motion_available = acceleration is not None or gyro is not None or gps is not None
        visual_raw = float(np.abs(gray-self.last_inferred_gray).mean()) if self.last_inferred_gray is not None else 0.
        visual = min(visual_raw/c['visual_scale'],1.)
        active = SIGNALS-self.disabled
        if 'motion' not in active: motion, motion_available = 0., False
        if 'visual' not in active: visual = 0.
        if 'risk' not in active: risk, risk_available = 0., False
        age = self.age
        score = (c['motion_weight']*motion+c['visual_weight']*visual+c['risk_weight']*risk
                 + (c['age_weight']*min(age/max(c['max_skip'],1),1.) if 'age' in active else 0.))
        forced = False
        if self.frames == 0: infer, reason = True, 'first_frame'
        elif age >= c['max_skip']: infer, reason, forced = True, 'max_skip', True
        elif 'risk' in active and risk_available and risk >= c['critical_risk']: infer, reason = True, 'critical_risk'
        elif 'motion' in active and motion_available and motion >= c['extreme_motion']: infer, reason = True, 'extreme_motion'
        elif 'visual' in active and visual >= c['extreme_visual']: infer, reason = True, 'extreme_visual'
        elif score >= c['decision_threshold']: infer, reason = True, 'combined_score'
        else: infer, reason = False, 'stable'
        self.age = 0 if infer else age+1
        self.frames += 1
        self.previous_acceleration = acceleration
        if infer:self.last_inferred_gray = np.array(gray,copy=True)
        return Decision(infer,reason,score,motion,motion_available,visual,risk,risk_available,age,self.age,forced)


def fixed_positions(policy, length, numerator=2455, denominator=4410):
    import math
    if policy == 'budget_fixed':
        k = math.ceil(length*numerator/denominator)
        return {math.ceil(j*length/k) for j in range(k)}
    interval = {'every':1,'fixed2':2,'fixed5':5}[policy]
    return set(range(0,length,interval))
