"""Unit tests for local-first data source and snapshot policy behaviour."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from utils import local_data


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [1, 2],
            "gameweek_id": [10, 10],
            "position_id": [2, 3],
            "total_points": [5.0, 6.0],
            "minutes_played": [90.0, 80.0],
            "ict_index": [7.0, 8.0],
            "now_cost": [5.5, 7.0],
        }
    )


def _write_snapshot_bundle(
    *,
    root: Path,
    df: pd.DataFrame,
    created_at_utc: str,
    schema_version: str = local_data.SCHEMA_VERSION,
) -> tuple[Path, Path]:
    snapshot_path = root / local_data.LATEST_SNAPSHOT_NAME
    snapshot_path.write_bytes(b"fake-parquet-bytes")

    contract_spec = local_data.build_contract_spec(df)
    contract_hash = local_data.build_contract_hash(
        schema_version=schema_version,
        contract_spec=contract_spec,
    )
    manifest = {
        "manifest_version": local_data.MANIFEST_VERSION,
        "schema_version": schema_version,
        "created_at_utc": created_at_utc,
        "source": "snowflake",
        "query_fingerprint": "q123",
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "min_gameweek": int(df["gameweek_id"].min()),
        "max_gameweek": int(df["gameweek_id"].max()),
        "required_columns_present": True,
        "contract_hash": contract_hash,
        "file_sha256": local_data.hash_file(snapshot_path),
        "refresh_run_id": "refresh_test",
    }
    manifest_path = root / local_data.LATEST_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return snapshot_path, manifest_path


def test_resolve_source_and_policy_precedence(monkeypatch, tmp_path):
    monkeypatch.setattr(local_data, "ALLOWLIST_ROOT", tmp_path)
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TRAINING_DATA_SOURCE", "snowflake")
    monkeypatch.setenv("TRAINING_DATA_POLICY", "LOCAL_THEN_SNOWFLAKE")

    resolved = local_data.resolve_data_resolution(source="local", policy="LOCAL_ONLY")

    assert resolved.source == "local"
    assert resolved.policy == "LOCAL_ONLY"


def test_allowlist_enforced(monkeypatch, tmp_path):
    monkeypatch.setattr(local_data, "ALLOWLIST_ROOT", tmp_path)

    with pytest.raises(local_data.LocalDataError) as exc:
        local_data.ensure_allowed_path(Path("/tmp/not-allowed"), reason="read snapshot")

    assert exc.value.code == "ALLOWLIST_VIOLATION"


def test_local_only_stale_snapshot_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(local_data, "ALLOWLIST_ROOT", tmp_path)
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))
    df = _base_df()
    stale_created = (datetime.now(UTC) - timedelta(days=30)).isoformat()
    _write_snapshot_bundle(root=tmp_path, df=df, created_at_utc=stale_created)
    monkeypatch.setattr(local_data, "_read_parquet", lambda _path: df.copy())

    with pytest.raises(local_data.LocalDataError) as exc:
        local_data.load_training_dataframe(
            table_name="fct_ml_player_features",
            source="local",
            policy="LOCAL_ONLY",
            max_age_days=8,
            snowflake_loader=lambda _table: df.copy(),
        )

    assert exc.value.code == "SNAPSHOT_STALE"


def test_local_then_snowflake_fallback_on_stale_snapshot(monkeypatch, tmp_path):
    monkeypatch.setattr(local_data, "ALLOWLIST_ROOT", tmp_path)
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))
    local_df = _base_df()
    snowflake_df = _base_df().assign(player_id=[10, 11])
    stale_created = (datetime.now(UTC) - timedelta(days=20)).isoformat()
    _write_snapshot_bundle(root=tmp_path, df=local_df, created_at_utc=stale_created)
    monkeypatch.setattr(local_data, "_read_parquet", lambda _path: local_df.copy())

    loaded_df, metadata = local_data.load_training_dataframe(
        table_name="fct_ml_player_features",
        source="local",
        policy="LOCAL_THEN_SNOWFLAKE",
        max_age_days=8,
        snowflake_loader=lambda _table: snowflake_df.copy(),
    )

    assert loaded_df["player_id"].tolist() == [10, 11]
    assert metadata["data_source_selected"] == "snowflake"
    assert metadata["fallback_taken"] is True
    assert "stale" in (metadata["fallback_reason"] or "").lower()


def test_major_schema_mismatch_never_falls_back(monkeypatch, tmp_path):
    monkeypatch.setattr(local_data, "ALLOWLIST_ROOT", tmp_path)
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))
    df = _base_df()
    fresh_created = datetime.now(UTC).isoformat()
    _write_snapshot_bundle(
        root=tmp_path,
        df=df,
        created_at_utc=fresh_created,
        schema_version="2.0.0",
    )
    monkeypatch.setattr(local_data, "_read_parquet", lambda _path: df.copy())

    fallback_called = {"value": False}

    def _loader(_table: str) -> pd.DataFrame:
        fallback_called["value"] = True
        return df.copy()

    with pytest.raises(local_data.LocalDataError) as exc:
        local_data.load_training_dataframe(
            table_name="fct_ml_player_features",
            source="local",
            policy="LOCAL_THEN_SNOWFLAKE",
            max_age_days=8,
            snowflake_loader=_loader,
        )

    assert exc.value.code == "SCHEMA_MAJOR_MISMATCH"
    assert fallback_called["value"] is False
