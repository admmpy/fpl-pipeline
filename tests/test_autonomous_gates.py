"""Unit tests for balanced autonomous promotion gates."""

from __future__ import annotations

from pathlib import Path

import yaml

from agents.autonomous_nodes import apply_domain_rules


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RULES = ROOT / "config" / "domain_rules.yaml"


def _rules_path(tmp_path: Path) -> Path:
    rules = yaml.safe_load(DEFAULT_RULES.read_text(encoding="utf-8"))
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump(rules), encoding="utf-8")
    return path


def _base_state(tmp_path: Path) -> dict:
    rules_path = _rules_path(tmp_path)
    candidate_backtest = {
        "summary": {
            "top_k_hit_rate_mean": 0.6,
            "rank_correlation_mean": 0.4,
            "selected_xi_regret_mean": 6.0,
            "squad_total_bias_abs_mean": 4.0,
            "max_position_bias_abs": 1.0,
            "prediction_collapse_weeks": 0,
            "bias_flip_weeks": 0,
        },
        "per_week": [
            {"gameweek_id": 25, "prediction_to_actual_ratio": 1.0, "prediction_collapse": False}
        ],
    }
    return {
        "state": "EVALUATED",
        "snapshot_meta": {"rules_path": str(rules_path)},
        "drift_report": {"drift_triggered": True},
        "candidate_metrics": {
            "candidate": {
                "mae": 2.0,
                "rmse": 2.3,
                "bias": 0.1,
                "position_exceed_rate": 0.01,
                "player_exceed_rate": 0.01,
                "candidate_score": 2.4,
                "backtest_summary": candidate_backtest["summary"],
                "backtest": candidate_backtest,
                "calibration_comparison": {"position_calibration_gain": 0.05},
            },
            "active_baseline": {
                "mae": 3.0,
                "rmse": 3.1,
                "candidate_score": 3.4,
            },
        },
    }


def test_rejects_on_excessive_squad_total_bias(tmp_path):
    state = _base_state(tmp_path)
    state["candidate_metrics"]["candidate"]["backtest_summary"]["squad_total_bias_abs_mean"] = 99.0
    update = apply_domain_rules(state)
    assert update["rule_eval"]["passed"] is False
    assert update["rule_eval"]["gates"]["max_squad_total_bias"] is False


def test_rejects_on_excessive_position_bias(tmp_path):
    state = _base_state(tmp_path)
    state["candidate_metrics"]["candidate"]["backtest_summary"]["max_position_bias_abs"] = 99.0
    update = apply_domain_rules(state)
    assert update["rule_eval"]["passed"] is False
    assert update["rule_eval"]["gates"]["max_position_bias"] is False


def test_rejects_on_prediction_collapse(tmp_path):
    state = _base_state(tmp_path)
    state["candidate_metrics"]["candidate"]["backtest_summary"]["prediction_collapse_weeks"] = 1
    update = apply_domain_rules(state)
    assert update["rule_eval"]["passed"] is False
    assert update["rule_eval"]["gates"]["max_prediction_collapse_weeks"] is False


def test_rejects_on_instability_bias_flips(tmp_path):
    state = _base_state(tmp_path)
    state["candidate_metrics"]["candidate"]["backtest_summary"]["bias_flip_weeks"] = 2
    update = apply_domain_rules(state)
    assert update["rule_eval"]["passed"] is False
    assert update["rule_eval"]["gates"]["max_bias_flip_weeks"] is False


def test_rejects_on_ranking_degradation_even_with_mae_gain(tmp_path):
    state = _base_state(tmp_path)
    state["candidate_metrics"]["candidate"]["backtest_summary"]["top_k_hit_rate_mean"] = 0.0
    state["candidate_metrics"]["candidate"]["backtest_summary"]["rank_correlation_mean"] = -0.3
    update = apply_domain_rules(state)
    assert update["rule_eval"]["passed"] is False
    assert update["rule_eval"]["gates"]["min_top_k_hit_rate"] is False


def test_rejects_when_required_baseline_gate_fails(tmp_path):
    state = _base_state(tmp_path)
    state["candidate_metrics"]["candidate"]["backtest"] = {
        "summary": state["candidate_metrics"]["candidate"]["backtest_summary"],
        "per_week": [{"gameweek_id": 25, "prediction_to_actual_ratio": 1.0, "prediction_collapse": False}],
        "required_baseline_comparison": {
            "baseline_gate_passed": False,
            "baseline_summary": {
                "top_k_hit_rate_mean": 0.7,
                "selected_xi_regret_mean": 5.0,
            },
        },
    }
    update = apply_domain_rules(state)
    assert update["rule_eval"]["passed"] is False
    assert update["rule_eval"]["gates"]["baseline_gate_passed"] is False
