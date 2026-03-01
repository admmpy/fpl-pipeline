"""Tests for atomic model registry promotion and rollback behaviour."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils import model_registry


class ConstantModel:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, frame: pd.DataFrame):
        return np.full(len(frame), self.value)


def _payload(value: float) -> dict:
    return {
        "model": ConstantModel(value),
        "metadata": {
            "feature_cols": ["feature_a", "feature_b"],
            "league_mean": 2.0,
            "shrinkage_alpha": 0.1,
        },
    }


def test_promote_atomically_creates_active_pointer_and_model_bin(tmp_path):
    logs_dir = tmp_path / "logs"
    result = model_registry.promote_atomically(
        model_payload=_payload(3.5),
        candidate_metrics={"mae": 1.2},
        run_id="run-1",
        logs_dir=logs_dir,
    )

    assert result["version"].startswith("model_")
    assert (logs_dir / "model.bin").exists()
    assert (logs_dir / "model_registry.json").exists()

    active = model_registry.get_active_model(logs_dir)
    assert active is not None
    assert active["version"] == result["version"]
    assert model_registry.verify_artifact(active["path"], active["checksum"])


def test_failed_promotion_rolls_back_to_previous_active(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"

    first = model_registry.promote_atomically(
        model_payload=_payload(1.5),
        candidate_metrics={"mae": 2.0},
        run_id="run-old",
        logs_dir=logs_dir,
    )
    previous_version = first["version"]

    def fail_switch(*args, **kwargs):
        raise RuntimeError("forced pointer failure")

    monkeypatch.setattr(model_registry, "_switch_active_pointer", fail_switch)

    with pytest.raises(model_registry.PromotionError):
        model_registry.promote_atomically(
            model_payload=_payload(0.9),
            candidate_metrics={"mae": 1.0},
            run_id="run-new",
            logs_dir=logs_dir,
        )

    active = model_registry.get_active_model(logs_dir)
    assert active is not None
    assert active["version"] == previous_version

    registry = json.loads((logs_dir / "model_registry.json").read_text(encoding="utf-8"))
    assert registry["active_version"] == previous_version


def test_rollback_to_specific_version(tmp_path):
    logs_dir = tmp_path / "logs"

    v1 = model_registry.promote_atomically(
        model_payload=_payload(2.2),
        candidate_metrics={"mae": 2.0},
        run_id="v1",
        logs_dir=logs_dir,
    )
    v2 = model_registry.promote_atomically(
        model_payload=_payload(1.1),
        candidate_metrics={"mae": 1.0},
        run_id="v2",
        logs_dir=logs_dir,
    )

    assert v1["version"] != v2["version"]

    restored = model_registry.rollback_to(v1["version"], logs_dir=logs_dir)
    assert restored["version"] == v1["version"]

    active = model_registry.get_active_model(logs_dir)
    assert active is not None
    assert active["version"] == v1["version"]
    assert Path(active["path"]).exists()


def test_allowlist_rejects_prefix_path_bypass(tmp_path):
    logs_dir = tmp_path / "logs"
    bypass_path = tmp_path / "logs_evil" / "model.bin"
    bypass_path.parent.mkdir(parents=True, exist_ok=True)
    bypass_path.write_bytes(b"payload")

    with pytest.raises(model_registry.RegistryError):
        model_registry._ensure_allowlisted(bypass_path, logs_dir)
