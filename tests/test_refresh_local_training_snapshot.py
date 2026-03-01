"""Tests for local training snapshot refresh workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import refresh_local_training_snapshot as refresh
from utils import local_data


def _sample_df(seed: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [seed, seed + 1],
            "gameweek_id": [8, 8],
            "position_id": [2, 3],
            "total_points": [5.0, 6.0],
            "minutes_played": [90.0, 75.0],
            "ict_index": [7.0, 8.0],
            "now_cost": [5.5, 7.0],
        }
    )


def _fake_write_parquet(df: pd.DataFrame, target_path: Path) -> None:
    payload = f"rows={len(df)} cols={len(df.columns)}".encode("utf-8")
    tmp_path = target_path.with_suffix(".parquet.tmp")
    tmp_path.write_bytes(payload)
    tmp_path.replace(target_path)


def test_refresh_writes_versioned_and_latest_files(monkeypatch, tmp_path):
    monkeypatch.setattr(local_data, "ALLOWLIST_ROOT", tmp_path)
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(refresh, "_fetch_from_snowflake", lambda _table: _sample_df())
    monkeypatch.setattr(refresh, "_write_parquet_atomic", _fake_write_parquet)
    monkeypatch.setattr(refresh, "_utc_stamp", lambda: "2026-03-01T120000Z")

    summary = refresh.refresh_snapshot(output_dir=str(tmp_path), force=True, retention_count=3)

    assert summary["status"] == "refreshed"
    assert (tmp_path / local_data.LATEST_SNAPSHOT_NAME).exists()
    assert (tmp_path / local_data.LATEST_MANIFEST_NAME).exists()
    assert (tmp_path / "features_2026-03-01T120000Z.parquet").exists()
    assert (tmp_path / "features_2026-03-01T120000Z.manifest.json").exists()

    latest_manifest = json.loads((tmp_path / local_data.LATEST_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert latest_manifest["required_columns_present"] is True
    assert latest_manifest["schema_version"] == local_data.SCHEMA_VERSION


def test_refresh_fails_when_lock_already_held(monkeypatch, tmp_path):
    monkeypatch.setattr(local_data, "ALLOWLIST_ROOT", tmp_path)
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))
    lock_path = tmp_path / ".refresh.lock"
    lock_path.write_text("held", encoding="utf-8")

    monkeypatch.setattr(refresh, "_fetch_from_snowflake", lambda _table: _sample_df())
    monkeypatch.setattr(refresh, "_write_parquet_atomic", _fake_write_parquet)

    with pytest.raises(RuntimeError, match="lock already held"):
        refresh.refresh_snapshot(output_dir=str(tmp_path), force=True, retention_count=3)


def test_refresh_applies_retention_policy(monkeypatch, tmp_path):
    monkeypatch.setattr(local_data, "ALLOWLIST_ROOT", tmp_path)
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(refresh, "_fetch_from_snowflake", lambda _table: _sample_df())
    monkeypatch.setattr(refresh, "_write_parquet_atomic", _fake_write_parquet)

    stamps = iter(
        [
            "2026-03-01T120000Z",
            "2026-03-08T120000Z",
            "2026-03-15T120000Z",
        ]
    )
    monkeypatch.setattr(refresh, "_utc_stamp", lambda: next(stamps))

    refresh.refresh_snapshot(output_dir=str(tmp_path), force=True, retention_count=2)
    refresh.refresh_snapshot(output_dir=str(tmp_path), force=True, retention_count=2)
    refresh.refresh_snapshot(output_dir=str(tmp_path), force=True, retention_count=2)

    versioned = sorted(tmp_path.glob("features_*.parquet"))
    assert len(versioned) == 2
    assert versioned[-1].name == "features_2026-03-15T120000Z.parquet"
