"""Unit tests for gameweek quality filtering and position-aware calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.train_model import (
    add_baseline_comparison_to_report,
    build_calibration_report,
    build_weekly_backtest_report,
    compute_focus_position_abs_bias,
    apply_position_aware_calibration,
    fit_position_aware_calibration,
    split_train_holdout,
)
from utils.gameweek_quality import effective_excluded_gameweeks


def _policy() -> dict:
    return {
        "policy_version": "test-v1",
        "excluded_gameweeks": [2],
        "backfilled_but_untrusted_gameweeks": [5],
        "trusted_backfilled_gameweeks": [],
        "eligible_training_gameweeks": [1, 3, 4],
        "eligible_holdout_backtest_gameweeks": [3, 4, 5, 6],
    }


def _frame() -> pd.DataFrame:
    rows = []
    for gw in range(1, 7):
        for player_id in range(1, 4):
            rows.append(
                {
                    "player_id": player_id,
                    "gameweek_id": gw - 1,
                    "target_gameweek_id": gw,
                    "target_next_gw_points": float(gw + player_id),
                    "minutes_played": 90.0,
                    "ict_index": 5.0,
                    "total_points": 3.0,
                    "position_id": (player_id % 4) + 1,
                }
            )
    return pd.DataFrame(rows)


def test_split_train_holdout_excludes_contaminated_gameweeks():
    train_df, holdout_df = split_train_holdout(_frame(), 2, gameweek_policy=_policy())
    assert 2 not in train_df["target_gameweek_id"].unique()
    assert 5 not in holdout_df["target_gameweek_id"].unique()
    assert 5 not in train_df["target_gameweek_id"].unique()


def test_trusted_backfilled_week_reenters_when_flagged():
    policy = _policy()
    policy["trusted_backfilled_gameweeks"] = [5]
    excluded = effective_excluded_gameweeks(policy)
    assert 5 not in excluded


def test_position_aware_calibration_fallback_for_small_groups():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([2.0, 2.0, 2.0, 2.0])
    position_ids = np.array([1, 2, 3, 4])

    payload = fit_position_aware_calibration(
        y_true,
        y_pred,
        position_ids,
        strength=0.8,
        min_samples=3,
    )
    assert payload["by_position"]["1"]["fallback"] == "global"
    calibrated = apply_position_aware_calibration(y_pred, position_ids, payload)
    assert calibrated.shape == y_pred.shape


def test_position_aware_calibration_applies_position_coefficients():
    y_true = np.array([2.0, 4.0, 10.0, 12.0])
    y_pred = np.array([1.0, 2.0, 5.0, 6.0])
    position_ids = np.array([1, 1, 2, 2])

    payload = fit_position_aware_calibration(
        y_true,
        y_pred,
        position_ids,
        strength=1.0,
        min_samples=2,
    )
    calibrated = apply_position_aware_calibration(y_pred, position_ids, payload)

    assert calibrated[0] == calibrated[1] / 2
    assert calibrated[2] == calibrated[3] * (5.0 / 6.0)


def test_position_aware_calibration_reduces_gk_def_bias_on_biased_sample():
    y_true = np.array([2.0, 3.0, 6.0, 7.0, 9.0, 10.0])
    y_pred = np.array([5.0, 6.0, 9.0, 10.0, 9.1, 10.1])
    position_ids = np.array([1, 2, 1, 2, 3, 4])

    global_bias = compute_focus_position_abs_bias(position_ids, y_true, y_pred)

    payload = fit_position_aware_calibration(
        y_true,
        y_pred,
        position_ids,
        strength=1.0,
        min_samples=2,
    )
    calibrated = apply_position_aware_calibration(y_pred, position_ids, payload)
    calibrated_bias = compute_focus_position_abs_bias(position_ids, y_true, calibrated)

    assert calibrated_bias <= global_bias


def test_calibration_report_contains_pre_post_and_delta():
    y_true = np.array([2.0, 4.0, 6.0, 8.0])
    pre = np.array([3.0, 5.0, 7.0, 9.0])
    post_global = np.array([2.5, 4.5, 6.5, 8.5])
    position = np.array([2.0, 4.0, 6.0, 8.0])
    pos_ids = np.array([1, 2, 3, 4])

    report = build_calibration_report(
        y_true,
        pre,
        global_calibration_pred=post_global,
        position_calibration_pred=position,
        selected_variant="position_aware",
        position_ids=pos_ids,
    )

    assert report["pre_calibration"]["mae"] > report["selected_post"]["mae"]
    assert report["post_global"] is not None
    assert report["post_position_aware"] is not None
    assert report["selected_delta"]["mae"] < 0
    assert "gk_def_abs_bias_pre" in report


def test_weekly_backtest_reports_instability_bias_flips():
    holdout_df = pd.DataFrame(
        {
            "player_id": [1, 2, 1, 2],
            "target_gameweek_id": [10, 10, 11, 11],
            "minutes_played": [90, 90, 90, 90],
            "position_id": [1, 2, 1, 2],
        }
    )
    y_true = np.array([20.0, 20.0, 0.0, 0.0])
    y_pred = np.array([0.0, 0.0, 20.0, 20.0])

    report = build_weekly_backtest_report(
        holdout_df,
        y_true,
        y_pred,
        model_rules={"instability_bias_abs_threshold": 5.0},
    )
    summary = report["summary"]

    assert summary["bias_flip_weeks"] == 1
    assert summary["bias_flip_detected"] is True
    assert summary["bias_flip_gameweek_pairs"] == [[10, 11]]


def test_baseline_comparison_marks_required_baseline_failure():
    holdout_df = pd.DataFrame(
        {
            "player_id": [1, 2, 1, 2],
            "target_gameweek_id": [10, 10, 11, 11],
            "target_next_gw_points": [10.0, 1.0, 10.0, 1.0],
            "minutes_played": [90, 90, 90, 90],
            "position_id": [3, 3, 3, 3],
            "five_week_players_roll_avg_points": [1.0, 10.0, 1.0, 10.0],
        }
    )
    y_true = np.array([10.0, 1.0, 10.0, 1.0])
    y_pred = np.array([1.0, 10.0, 1.0, 10.0])

    report = build_weekly_backtest_report(holdout_df, y_true, y_pred)
    report = add_baseline_comparison_to_report(
        report,
        holdout_df.assign(predicted_points=y_pred),
        y_true,
        model_rules={
            "required_baseline": "five_week_players_roll_avg_points",
            "baseline_gate_metrics": ["top_k_hit_rate_mean", "selected_xi_regret_mean"],
        },
    )

    assert report["required_baseline"] == "five_week_players_roll_avg_points"
    assert report["required_baseline_comparison"]["baseline_gate_passed"] is False
