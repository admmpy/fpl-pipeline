"""
Unit tests for pure ML helper functions in train_model.py and ml_tasks.py.
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

# Allow imports from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_model import (
    apply_shrinkage as train_apply_shrinkage,
    fit_calibration,
    blend_calibration,
    apply_calibration as train_apply_calibration,
    compute_global_stats,
    FEATURES_TO_SCALE,
)
from tasks.ml_tasks import (
    apply_shrinkage as infer_apply_shrinkage,
    apply_calibration as infer_apply_calibration,
    apply_global_z_scores,
)


# ---------------------------------------------------------------------------
# apply_shrinkage
# ---------------------------------------------------------------------------

class TestApplyShrinkage:
    def test_alpha_zero_passthrough(self):
        preds = np.array([1.0, 2.0, 3.0])
        result = train_apply_shrinkage(preds, league_mean=5.0, alpha=0.0)
        np.testing.assert_array_almost_equal(result, preds)

    def test_alpha_one_all_mean(self):
        preds = np.array([1.0, 2.0, 3.0])
        result = train_apply_shrinkage(preds, league_mean=5.0, alpha=1.0)
        np.testing.assert_array_almost_equal(result, [5.0, 5.0, 5.0])

    def test_alpha_half(self):
        preds = np.array([10.0])
        result = train_apply_shrinkage(preds, league_mean=0.0, alpha=0.5)
        np.testing.assert_array_almost_equal(result, [5.0])

    def test_empty_array(self):
        preds = np.array([])
        result = train_apply_shrinkage(preds, league_mean=3.0, alpha=0.5)
        assert len(result) == 0

    def test_train_inference_parity(self):
        preds = np.array([2.0, 4.0, 6.0])
        train_result = train_apply_shrinkage(preds, 3.0, 0.3)
        infer_result = infer_apply_shrinkage(preds, 3.0, 0.3)
        np.testing.assert_array_almost_equal(train_result, infer_result)


# ---------------------------------------------------------------------------
# fit_calibration
# ---------------------------------------------------------------------------

class TestFitCalibration:
    def test_identical_inputs_identity(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        a, b = fit_calibration(y, y)
        assert a == pytest.approx(1.0)
        assert b == pytest.approx(0.0)

    def test_constant_predictions_guard(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([5.0, 5.0, 5.0])
        a, b = fit_calibration(y_true, y_pred)
        assert a == 1.0
        assert b == 0.0

    def test_scaled_distribution(self):
        y_true = np.array([2.0, 4.0, 6.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        a, b = fit_calibration(y_true, y_pred)
        # true_std / pred_std = 2.0, so a ≈ 2.0
        assert a == pytest.approx(2.0)
        # b = mean(y_true) - a * mean(y_pred) = 4 - 2*2 = 0
        assert b == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# blend_calibration
# ---------------------------------------------------------------------------

class TestBlendCalibration:
    def test_strength_zero_identity(self):
        a, b = blend_calibration(2.0, 5.0, strength=0.0)
        assert a == pytest.approx(1.0)
        assert b == pytest.approx(0.0)

    def test_strength_one_original(self):
        a, b = blend_calibration(2.0, 5.0, strength=1.0)
        assert a == pytest.approx(2.0)
        assert b == pytest.approx(5.0)

    def test_strength_half(self):
        a, b = blend_calibration(3.0, 4.0, strength=0.5)
        # blended_a = 1 + 0.5*(3-1) = 2.0
        assert a == pytest.approx(2.0)
        # blended_b = 0.5 * 4 = 2.0
        assert b == pytest.approx(2.0)

    def test_clamp_above_one(self):
        a, b = blend_calibration(2.0, 5.0, strength=1.5)
        # strength clamped to 1.0
        assert a == pytest.approx(2.0)
        assert b == pytest.approx(5.0)

    def test_clamp_below_zero(self):
        a, b = blend_calibration(2.0, 5.0, strength=-0.5)
        # strength clamped to 0.0
        assert a == pytest.approx(1.0)
        assert b == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# apply_calibration
# ---------------------------------------------------------------------------

class TestApplyCalibration:
    def test_identity(self):
        preds = np.array([1.0, 2.0, 3.0])
        result = train_apply_calibration(preds, a=1.0, b=0.0)
        np.testing.assert_array_almost_equal(result, preds)

    def test_scale_and_offset(self):
        preds = np.array([1.0, 2.0, 3.0])
        result = train_apply_calibration(preds, a=2.0, b=1.0)
        np.testing.assert_array_almost_equal(result, [3.0, 5.0, 7.0])

    def test_empty_array(self):
        preds = np.array([])
        result = train_apply_calibration(preds, a=2.0, b=1.0)
        assert len(result) == 0

    def test_train_inference_parity(self):
        preds = np.array([5.0, 10.0])
        train_result = train_apply_calibration(preds, 0.9, 0.5)
        infer_result = infer_apply_calibration(preds, 0.9, 0.5)
        np.testing.assert_array_almost_equal(train_result, infer_result)


# ---------------------------------------------------------------------------
# apply_global_z_scores (tasks/ml_tasks.py)
# ---------------------------------------------------------------------------

class TestApplyGlobalZScores:
    def test_basic_computation(self):
        df = pd.DataFrame({"total_points": [10.0, 20.0, 30.0]})
        stats = {"total_points": {"mean": 20.0, "std": 10.0}}
        result = apply_global_z_scores(df, stats, ["total_points"])
        expected = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(
            result["total_points_z_score"].to_numpy(), expected
        )

    def test_missing_feature_skipped(self):
        df = pd.DataFrame({"total_points": [1.0, 2.0]})
        stats = {"total_points": {"mean": 1.5, "std": 0.5}}
        result = apply_global_z_scores(df, stats, ["total_points", "not_here"])
        assert "total_points_z_score" in result.columns
        assert "not_here_z_score" not in result.columns

    def test_nan_filled_with_zero(self):
        df = pd.DataFrame({"total_points": [np.nan, 10.0]})
        stats = {"total_points": {"mean": 10.0, "std": 5.0}}
        result = apply_global_z_scores(df, stats, ["total_points"])
        assert result["total_points_z_score"].iloc[0] == 0.0

    def test_zero_std_guard(self):
        df = pd.DataFrame({"total_points": [5.0, 15.0]})
        stats = {"total_points": {"mean": 10.0, "std": 0.0}}
        result = apply_global_z_scores(df, stats, ["total_points"])
        # std=0 should fall back to 1.0
        expected = np.array([-5.0, 5.0])
        np.testing.assert_array_almost_equal(
            result["total_points_z_score"].to_numpy(), expected
        )

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"total_points": [10.0, 20.0]})
        stats = {"total_points": {"mean": 15.0, "std": 5.0}}
        apply_global_z_scores(df, stats, ["total_points"])
        assert "total_points_z_score" not in df.columns


# ---------------------------------------------------------------------------
# compute_global_stats (scripts/train_model.py)
# ---------------------------------------------------------------------------

class TestComputeGlobalStats:
    def test_normal_data(self):
        df = pd.DataFrame({
            "total_points": [10.0, 20.0, 30.0],
            "minutes_played": [45.0, 90.0, 67.0],
            "ict_index": [3.0, 6.0, 9.0],
        })
        stats = compute_global_stats(df)
        for feature in FEATURES_TO_SCALE:
            assert "mean" in stats[feature]
            assert "std" in stats[feature]
            assert stats[feature]["std"] > 0

    def test_constant_column_returns_std_one(self):
        df = pd.DataFrame({
            "total_points": [5.0, 5.0, 5.0],
            "minutes_played": [90.0, 90.0, 90.0],
            "ict_index": [4.0, 4.0, 4.0],
        })
        stats = compute_global_stats(df)
        for feature in FEATURES_TO_SCALE:
            assert stats[feature]["std"] == 1.0

    def test_single_row(self):
        df = pd.DataFrame({
            "total_points": [7.0],
            "minutes_played": [60.0],
            "ict_index": [2.0],
        })
        stats = compute_global_stats(df)
        for feature in FEATURES_TO_SCALE:
            # std of single value is NaN → should fall back to 1.0
            assert stats[feature]["std"] == 1.0
