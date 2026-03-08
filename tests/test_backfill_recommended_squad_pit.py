"""Tests for point-in-time backfill trust validation metadata."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import backfill_recommended_squad_pit as pit


class _DummyModel:
    def predict(self, frame: pd.DataFrame):
        return np.full(len(frame), 5.0)


def _feature_df() -> pd.DataFrame:
    rows = []
    for gw in range(1, 7):
        for player_id in range(1, 18):
            rows.append(
                {
                    "player_id": player_id,
                    "web_name": f"P{player_id}",
                    "position_id": (player_id % 4) + 1,
                    "team_id": (player_id % 8) + 1,
                    "now_cost": 5.0 + (player_id % 3),
                    "gameweek_id": gw,
                    "minutes_played": 90,
                    "total_points": 3.0 + player_id * 0.01,
                    "ict_index": 5.0,
                }
            )
    return pd.DataFrame(rows)


def test_validate_backfill_integrity_passes_for_prior_data_only():
    train_df = pd.DataFrame(
        {
            "target_gameweek_id": [5, 6, 6],
        }
    )
    validation = pit.validate_backfill_integrity(
        train_df=train_df,
        feature_df=_feature_df(),
        target_gw=7,
        validation_version="v-test",
    )
    assert validation["trusted"] is True
    assert validation["status"] == "validated_trusted"


def test_backfill_gameweek_adds_trust_metadata(monkeypatch):
    monkeypatch.setattr(
        pit,
        "_train_point_in_time_model",
        lambda train_df, feature_cols: (
            _DummyModel(),
            {
                "league_mean": 4.0,
                "shrinkage_alpha": 0.0,
                "selected_calibration_variant": "none",
                "calibration": None,
                "position_calibration": None,
                "position_caps": {},
                "use_log_target": False,
            },
        ),
    )

    def _fake_optimize(predictions):
        squad = []
        for row in predictions[:15]:
            squad.append(
                {
                    **row,
                    "is_in_squad": True,
                    "is_starter": len(squad) < 11,
                    "is_captain": len(squad) == 0,
                    "is_vice_captain": len(squad) == 1,
                    "expected_points_5_gw": row["expected_points_next_gw"] * 5,
                }
            )
        return squad

    monkeypatch.setattr(pit.optimize_squad_task, "fn", _fake_optimize)

    result = pit.backfill_gameweek(
        feature_df=_feature_df(),
        target_gw=7,
        dry_run=True,
        timestamp_prefix="2026-01-01 00:00:00",
        validation_version="policy-v1",
    )

    assert result["backfill_trusted"] is True
    assert result["backfill_validation_status"] == "validated_trusted"
    assert result["backfill_validation_version"] == "policy-v1"


def test_backfill_prediction_path_uses_shared_post_processing():
    feature_df = _feature_df()
    train_df, feature_cols, zscore_stats = pit._prepare_training_data(feature_df, target_gw=7)
    model, metadata = pit._train_point_in_time_model(train_df, feature_cols)

    predictions = pit._build_predictions_for_target_gw(
        feature_df=feature_df,
        target_gw=7,
        model=model,
        feature_cols=feature_cols,
        zscore_stats=zscore_stats,
        metadata=metadata,
    )

    assert predictions
    assert metadata["selected_calibration_variant"] == "none"
    assert set(metadata["position_caps"]) == {1, 2, 3, 4}


def test_backfill_gameweek_replaces_existing_rows(monkeypatch):
    deleted = []
    loaded = []

    monkeypatch.setattr(
        pit,
        "_train_point_in_time_model",
        lambda train_df, feature_cols: (
            _DummyModel(),
            {
                "league_mean": 4.0,
                "shrinkage_alpha": 0.0,
                "selected_calibration_variant": "none",
                "calibration": None,
                "position_calibration": None,
                "position_caps": {},
                "use_log_target": False,
            },
        ),
    )
    monkeypatch.setattr(pit, "_delete_existing_gameweek_rows", lambda target_gw: deleted.append(target_gw) or 15)
    monkeypatch.setattr(pit, "insert_typed_records", lambda table_name, squad: loaded.append((table_name, len(squad))) or len(squad))

    def _fake_optimize(predictions):
        squad = []
        for row in predictions[:15]:
            squad.append(
                {
                    **row,
                    "is_in_squad": True,
                    "is_starter": len(squad) < 11,
                    "is_captain": len(squad) == 0,
                    "is_vice_captain": len(squad) == 1,
                    "expected_points_5_gw": row["expected_points_next_gw"] * 5,
                }
            )
        return squad

    monkeypatch.setattr(pit.optimize_squad_task, "fn", _fake_optimize)

    result = pit.backfill_gameweek(
        feature_df=_feature_df(),
        target_gw=7,
        dry_run=False,
        timestamp_prefix="2026-01-01 00:00:00",
        validation_version="policy-v1",
    )

    assert deleted == [7]
    assert loaded == [("recommended_squad", 15)]
    assert result["loaded_rows"] == 15
