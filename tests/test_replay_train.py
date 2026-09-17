from __future__ import annotations

import csv
import json
import torch

from aerodepth_lite.data import MidAirDataset, load_manifest
from aerodepth_lite.model import build_model
from aerodepth_lite.replay import LOG_FIELDS, replay_policy
from aerodepth_lite.train import train_model


def test_fixed_rate_replay_accounting_and_logging(fixture_dataset, smoke_config, tmp_path):
    root, manifest = fixture_dataset
    records, _ = load_manifest(manifest, "val")
    dataset = MidAirDataset(root, records, smoke_config["data"])
    model = build_model(smoke_config["model"])
    every = replay_policy(model, dataset, "every", smoke_config["scheduler"], smoke_config["replay"], torch.device("cpu"), tmp_path / "every.csv")
    fixed2 = replay_policy(model, dataset, "fixed2", smoke_config["scheduler"], smoke_config["replay"], torch.device("cpu"), tmp_path / "fixed2.csv")
    fixed5 = replay_policy(model, dataset, "fixed5", smoke_config["scheduler"], smoke_config["replay"], torch.device("cpu"), tmp_path / "fixed5.csv")
    assert every["inference_calls"] == every["total_frames"] == 4
    assert fixed2["inference_calls"] == 2
    assert fixed5["inference_calls"] == 1
    assert fixed2["maximum_prediction_age"] == 1
    assert fixed5["maximum_prediction_age"] == 3
    assert fixed2["metrics"]["valid_pixels"] == every["metrics"]["valid_pixels"]
    with (tmp_path / "fixed2.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == LOG_FIELDS
    assert [row["infer"] for row in rows] == ["True", "False", "True", "False"]
    assert rows[1]["prediction_source_frame"] == "0"


def test_adaptive_max_skip_accounting(fixture_dataset, smoke_config, tmp_path):
    root, manifest = fixture_dataset
    records, _ = load_manifest(manifest, "val")
    dataset = MidAirDataset(root, records, smoke_config["data"])
    model = build_model(smoke_config["model"])
    result = replay_policy(model, dataset, "adaptive", smoke_config["scheduler"], smoke_config["replay"], torch.device("cpu"), tmp_path / "adaptive.csv")
    assert result["total_frames"] == 4
    assert result["maximum_prediction_age"] <= smoke_config["scheduler"]["max_skip"]
    assert sum(result["inference_reasons"].values()) == 4


def test_smoke_training_and_evaluation(fixture_dataset, smoke_config, tmp_path):
    root, manifest = fixture_dataset
    output = tmp_path / "run"
    result = train_model(smoke_config, root, manifest, output)
    assert (output / "best.pt").exists()
    assert (output / "history.csv").exists()
    assert result["epochs_completed"] == 1
    assert result["best_validation_absrel"] >= 0
    assert result["model"]["parameters"] > 0
