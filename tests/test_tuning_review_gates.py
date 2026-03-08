"""Tests for dual-agent tuning gate enforcement."""

from __future__ import annotations

from agents.tuning_nodes_review import _evaluate_gates


def _sanity() -> dict:
    return {
        "actual_stats": {"std": 4.0},
        "position_exceed_rate": 0.0,
        "player_exceed_rate": 0.0,
    }


def _summary() -> dict:
    return {
        "squad_total_bias_abs_mean": 2.0,
        "max_position_bias_abs": 1.0,
        "top_k_hit_rate_mean": 0.6,
        "rank_correlation_mean": 0.4,
        "selected_xi_regret_mean": 5.0,
        "prediction_collapse_weeks": 0,
        "bias_flip_weeks": 0,
    }


def test_reviewer_rejects_when_ranking_gate_fails(monkeypatch):
    monkeypatch.setenv("TUNING_MIN_TOP_K_HIT", "0.5")

    summary = _summary()
    summary["top_k_hit_rate_mean"] = 0.1

    passed, gates = _evaluate_gates(
        holdout_mae=2.0,
        holdout_rmse=2.1,
        holdout_bias=0.1,
        sanity_metrics=_sanity(),
        backtest_summary=summary,
        best_holdout_mae=2.0,
        best_holdout_rmse=2.1,
        best_candidate_score=2.0,
        candidate_score=2.0,
    )

    assert passed is False
    assert gates["ranking_gate"] is False


def test_reviewer_rejects_when_instability_gate_fails(monkeypatch):
    monkeypatch.setenv("TUNING_MAX_BIAS_FLIP_WEEKS", "0")

    summary = _summary()
    summary["bias_flip_weeks"] = 1

    passed, gates = _evaluate_gates(
        holdout_mae=2.0,
        holdout_rmse=2.1,
        holdout_bias=0.1,
        sanity_metrics=_sanity(),
        backtest_summary=summary,
        best_holdout_mae=2.0,
        best_holdout_rmse=2.1,
        best_candidate_score=2.0,
        candidate_score=2.0,
    )

    assert passed is False
    assert gates["instability_gate"] is False


def test_reviewer_rejects_when_required_baseline_gate_fails():
    summary = _summary()
    report = {
        "summary": summary,
        "required_baseline_comparison": {
            "baseline_gate_passed": False,
            "baseline_summary": {
                "top_k_hit_rate_mean": 0.7,
                "selected_xi_regret_mean": 4.0,
            },
            "model_summary": summary,
        },
    }

    passed, gates = _evaluate_gates(
        holdout_mae=2.0,
        holdout_rmse=2.1,
        holdout_bias=0.1,
        sanity_metrics=_sanity(),
        backtest_summary=report,
        best_holdout_mae=2.0,
        best_holdout_rmse=2.1,
        best_candidate_score=(),
        candidate_score=(5.0, -0.6, -0.4, 2.1, 2.0),
    )

    assert passed is False
    assert gates["baseline_gate"] is False
