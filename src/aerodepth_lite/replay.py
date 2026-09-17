from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any
import csv
import json
import math

import torch

from .data import MidAirDataset
from .metrics import DepthMetricAccumulator
from .scheduler import AdaptiveScheduler


LOG_FIELDS = [
    "trajectory", "frame_index", "timestamp", "infer", "reason", "score", "motion",
    "motion_available", "visual", "risk", "risk_available", "decision_threshold",
    "age_before", "age_after", "max_skip_forced", "inference_latency_ms",
    "scheduler_latency_ms", "prediction_source_frame", "critical_event",
]


def _critical_depth(target: torch.Tensor, valid: torch.Tensor) -> float:
    values = target[valid & torch.isfinite(target) & (target > 0)]
    return float(torch.quantile(values, 0.10)) if values.numel() else math.inf


@torch.inference_mode()
def replay_policy(
    model: torch.nn.Module,
    dataset: MidAirDataset,
    policy: str,
    scheduler_config: dict[str, Any],
    replay_config: dict[str, Any],
    device: torch.device,
    log_path: str | Path,
) -> dict[str, Any]:
    model.eval()
    scheduler = AdaptiveScheduler(scheduler_config)
    metrics = DepthMetricAccumulator()
    previous_prediction = None
    previous_trajectory = None
    source_frame = None
    per_trajectory: dict[str, DepthMetricAccumulator] = defaultdict(DepthMetricAccumulator)
    logs: list[dict[str, Any]] = []
    inference_calls = 0
    inference_seconds = 0.0
    scheduler_seconds = 0.0
    reasons: Counter[str] = Counter()
    trajectory_position = 0
    fixed_age = 0
    last_critical_depth = math.inf
    pending_events: list[int] = []
    event_delays: list[int] = []
    critical_count = inferred_on_time = 0

    for sample_index in range(len(dataset)):
        sample = dataset[sample_index]
        record = sample["record"]
        trajectory = record["trajectory_id"]
        if trajectory != previous_trajectory:
            if previous_trajectory is not None and pending_events:
                event_delays.extend(trajectory_position - event for event in pending_events)
            scheduler.reset()
            previous_prediction = None
            source_frame = None
            trajectory_position = 0
            fixed_age = 0
            last_critical_depth = math.inf
            pending_events = []
            previous_trajectory = trajectory
        rgb = sample["rgb"]
        start = perf_counter()
        if policy == "adaptive":
            decision = scheduler.decide(rgb, record.get("navigation", {}), previous_prediction)
            infer = decision.infer
            decision_data = asdict(decision)
        else:
            interval = 1 if policy == "every" else int(policy.removeprefix("fixed"))
            infer = trajectory_position % interval == 0
            age_before = fixed_age
            age_after = 0 if infer else fixed_age + 1
            decision_data = {
                "reason": "first_frame" if trajectory_position == 0 else ("fixed_interval" if infer else "fixed_skip"),
                "score": 0.0, "motion": 0.0, "motion_available": False, "visual": 0.0,
                "risk": 0.0, "risk_available": previous_prediction is not None,
                "age_before": age_before, "age_after": age_after, "max_skip_forced": False,
            }
            fixed_age = age_after
        scheduler_latency = perf_counter() - start
        scheduler_seconds += scheduler_latency
        latency = 0.0
        if infer:
            start = perf_counter()
            previous_prediction = model(rgb.unsqueeze(0).to(device)).cpu().squeeze(0)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latency = perf_counter() - start
            inference_seconds += latency
            inference_calls += 1
            source_frame = trajectory_position
            if pending_events:
                event_delays.extend(trajectory_position - event for event in pending_events)
                pending_events = []
        if previous_prediction is None:
            raise RuntimeError("Policy failed to infer on first trajectory frame")
        target, valid = sample["depth"], sample["valid"]
        metrics.update(previous_prediction, target, valid, float(scheduler_config.get("near_threshold", 10.0)))
        per_trajectory[trajectory].update(previous_prediction, target, valid, float(scheduler_config.get("near_threshold", 10.0)))
        critical_depth = _critical_depth(target, valid)
        near_limit = float(replay_config.get("critical_near_threshold", 8.0))
        change_limit = float(replay_config.get("critical_change_threshold", 2.0))
        critical = critical_depth < near_limit and (last_critical_depth >= near_limit or last_critical_depth - critical_depth >= change_limit)
        if critical:
            critical_count += 1
            if infer:
                inferred_on_time += 1
                event_delays.append(0)
            else:
                pending_events.append(trajectory_position)
        last_critical_depth = critical_depth
        reasons[decision_data["reason"]] += 1
        logs.append({
            "trajectory": trajectory,
            "frame_index": record["frame_index"],
            "timestamp": record.get("timestamp"),
            "infer": infer,
            "reason": decision_data["reason"],
            "score": decision_data["score"],
            "motion": decision_data["motion"],
            "motion_available": decision_data["motion_available"],
            "visual": decision_data["visual"],
            "risk": decision_data["risk"],
            "risk_available": decision_data["risk_available"],
            "decision_threshold": scheduler_config.get("decision_threshold", 0.62),
            "age_before": decision_data["age_before"],
            "age_after": decision_data["age_after"],
            "max_skip_forced": decision_data["max_skip_forced"],
            "inference_latency_ms": latency * 1000,
            "scheduler_latency_ms": scheduler_latency * 1000,
            "prediction_source_frame": source_frame,
            "critical_event": critical,
        })
        trajectory_position += 1

    if pending_events:
        event_delays.extend(trajectory_position - event for event in pending_events)

    output = Path(log_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(logs)
    total = len(logs)
    ages = [int(row["age_after"]) for row in logs]
    result = {
        "policy": policy,
        "total_frames": total,
        "inference_calls": inference_calls,
        "inference_call_reduction": 1.0 - inference_calls / total if total else 0.0,
        "skip_rate": 1.0 - inference_calls / total if total else 0.0,
        "maximum_prediction_age": max(ages, default=0),
        "mean_prediction_age": sum(ages) / total if total else 0.0,
        "inference_runtime_seconds": inference_seconds,
        "scheduler_runtime_seconds": scheduler_seconds,
        "metrics": metrics.compute(),
        "per_trajectory_metrics": {k: v.compute() for k, v in per_trajectory.items()},
        "inference_reasons": dict(reasons),
        "critical_events": {
            "definition": "Provisional: 10th-percentile valid GT depth enters near zone or drops by configured threshold while in zone.",
            "count": critical_count,
            "inferred_on_time": inferred_on_time,
            "delayed_update_count": critical_count - inferred_on_time,
            "missed_critical_update_rate": (critical_count - inferred_on_time) / critical_count if critical_count else 0.0,
            "worst_update_delay_frames": max(event_delays, default=0),
        },
        "log_path": str(output),
    }
    return result


def save_replay_results(results: list[dict[str, Any]], output: str | Path) -> None:
    Path(output).write_text(json.dumps({"results": results}, indent=2, allow_nan=True) + "\n", encoding="utf-8")
