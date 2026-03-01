"""Integration tests for autonomous optimisation graph execution paths."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from agents.autonomous_graph import build_autonomous_graph
from agents.autonomous_state import create_initial_state
from tasks.ml_tasks import run_ml_inference


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RULES = ROOT / "config" / "domain_rules.yaml"


def _dataset(n_players: int = 8, n_gameweeks: int = 14) -> pd.DataFrame:
    rows: list[dict] = []
    rng = np.random.default_rng(42)

    for player_id in range(1, n_players + 1):
        position_id = (player_id % 4) + 1
        for gameweek_id in range(1, n_gameweeks + 1):
            total_points = 1.5 + 0.25 * gameweek_id + 0.1 * player_id
            minutes_played = 60 + (player_id % 3) * 10
            ict_index = 3.0 + 0.15 * gameweek_id
            noise = float(rng.normal(0, 0.01))
            target_next = 0.55 * total_points + 0.03 * minutes_played + noise

            rows.append(
                {
                    "player_id": int(player_id),
                    "gameweek_id": int(gameweek_id),
                    "target_gameweek_id": int(gameweek_id + 1),
                    "target_next_gw_points": float(target_next),
                    "position_id": int(position_id),
                    "total_points": float(total_points),
                    "minutes_played": float(minutes_played),
                    "ict_index": float(ict_index),
                    "three_week_players_roll_avg_points": float(total_points * 0.9),
                    "five_week_players_roll_avg_points": float(total_points * 0.85),
                    "total_games_played": int(gameweek_id),
                    "goals_scored": 0,
                    "assists": 0,
                    "clean_sheets": 0,
                    "goals_conceded": 1,
                    "bonus": 0,
                    "influence": float(ict_index * 10),
                    "creativity": float(ict_index * 6),
                    "threat": float(ict_index * 4),
                    "expected_goals": 0.1,
                    "expected_assists": 0.05,
                    "expected_goals_conceded": 0.3,
                    "expected_goal_involvements": 0.15,
                    "team_roll_avg_goals_scored": 1.2,
                    "team_roll_avg_xg": 1.3,
                    "team_roll_avg_clean_sheets": 0.2,
                    "team_roll_avg_wins_pct": 0.45,
                    "opponent_roll_avg_goals_conceded": 1.1,
                    "opponent_roll_avg_xg": 1.2,
                    "opponent_defence_strength": 3,
                    "team_attack_strength": 3,
                    "team_position": 10,
                    "opponent_team_position": 8,
                    "team_position_difference": 2,
                    "form": float(total_points * 0.8),
                    "now_cost": 6.0,
                    "minutes_band_0_30": 0,
                    "minutes_band_31_60": 0,
                    "minutes_band_61_90": 1,
                    "is_gk": int(position_id == 1),
                    "is_def": int(position_id == 2),
                    "is_mid": int(position_id == 3),
                    "is_fwd": int(position_id == 4),
                }
            )

    return pd.DataFrame(rows)


def _rules_file(tmp_path: Path, mutate: dict | None = None) -> Path:
    rules = yaml.safe_load(DEFAULT_RULES.read_text(encoding="utf-8"))
    if mutate:
        for key, value in mutate.items():
            section, item = key.split(".", 1)
            rules[section][item] = value
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump(rules), encoding="utf-8")
    return path


def _patch_runtime_dirs(monkeypatch, tmp_path: Path):
    from agents import autonomous_nodes

    logs_dir = tmp_path / "logs"
    autonomous_dir = logs_dir / "autonomous"
    monkeypatch.setattr(autonomous_nodes, "DEFAULT_LOGS_DIR", logs_dir)
    monkeypatch.setattr(autonomous_nodes, "AUTONOMOUS_LOG_DIR", autonomous_dir)
    monkeypatch.setattr(autonomous_nodes, "AUTONOMOUS_EVENTS_PATH", autonomous_dir / "events.jsonl")
    return logs_dir


def test_no_drift_path_exits_without_retrain(tmp_path, monkeypatch):
    _patch_runtime_dirs(monkeypatch, tmp_path)
    rules_path = _rules_file(tmp_path)

    graph = build_autonomous_graph()
    state = create_initial_state(
        "run-no-drift",
        snapshot_meta={
            "dataframe": _dataset(),
            "rules_path": str(rules_path),
            "force_no_drift": True,
        },
    )

    final_state = graph.invoke(state)

    assert final_state["state"] == "RECORDED"
    assert final_state["promotion_decision"]["decision"] == "no_action"
    assert Path(final_state["evidence_path"]).exists()
    assert final_state.get("optuna_result", {}) == {}


def test_validation_failure_routes_to_record_evidence_without_drift_node_error(tmp_path, monkeypatch):
    _patch_runtime_dirs(monkeypatch, tmp_path)
    rules_path = _rules_file(tmp_path)

    broken_df = _dataset().drop(columns=["target_next_gw_points"])

    graph = build_autonomous_graph()
    state = create_initial_state(
        "run-validation-fail",
        snapshot_meta={
            "dataframe": broken_df,
            "rules_path": str(rules_path),
        },
    )

    final_state = graph.invoke(state)

    assert final_state["state"] == "FAILED"
    assert final_state["error"]["type"] in {"ValidationError", "ValueError"}
    assert final_state["error"]["node"] != "detect_drift"
    assert Path(final_state["evidence_path"]).exists()


def test_drift_path_can_promote_deterministically(tmp_path, monkeypatch):
    logs_dir = _patch_runtime_dirs(monkeypatch, tmp_path)
    rules_path = _rules_file(tmp_path, mutate={"optimisation.optuna_trials": 2})

    graph = build_autonomous_graph()
    state = create_initial_state(
        "run-promote",
        snapshot_meta={
            "dataframe": _dataset(),
            "rules_path": str(rules_path),
            "force_drift": True,
        },
    )

    final_state = graph.invoke(state)

    assert final_state["state"] == "RECORDED"
    assert final_state["rule_eval"]["passed"] is True
    assert final_state["active_model_version"]
    assert (logs_dir / "model.bin").exists()


def test_drift_path_can_reject_deterministically(tmp_path, monkeypatch):
    _patch_runtime_dirs(monkeypatch, tmp_path)
    rules_path = _rules_file(
        tmp_path,
        mutate={
            "optimisation.optuna_trials": 2,
            "model.max_mae": 0.0,
            "model.max_rmse": 0.0,
        },
    )

    graph = build_autonomous_graph()
    state = create_initial_state(
        "run-reject",
        snapshot_meta={
            "dataframe": _dataset(),
            "rules_path": str(rules_path),
            "force_drift": True,
        },
    )

    final_state = graph.invoke(state)

    assert final_state["state"] == "RECORDED"
    assert final_state["rule_eval"]["passed"] is False
    assert final_state["promotion_decision"]["decision"] == "reject"


def test_inference_compatibility_reads_active_model_bin(tmp_path, monkeypatch):
    logs_dir = _patch_runtime_dirs(monkeypatch, tmp_path)
    rules_path = _rules_file(tmp_path, mutate={"optimisation.optuna_trials": 2})

    graph = build_autonomous_graph()
    state = create_initial_state(
        "run-inference",
        snapshot_meta={
            "dataframe": _dataset(),
            "rules_path": str(rules_path),
            "force_drift": True,
        },
    )

    final_state = graph.invoke(state)
    assert final_state["state"] == "RECORDED"
    assert (logs_dir / "model.bin").exists()

    inference_df = pd.DataFrame(
        [
            {
                "gameweek_id": 25,
                "player_id": 101,
                "web_name": "Player",
                "position_id": 3,
                "team_id": 2,
                "now_cost": 7.5,
                "three_week_players_roll_avg_points": 4.2,
                "total_points": 5.0,
                "minutes_played": 75,
                "ict_index": 8.2,
            }
        ]
    )

    monkeypatch.chdir(tmp_path)
    predictions = run_ml_inference.fn(inference_df)

    assert len(predictions) == 1
    assert "expected_points_next_gw" in predictions[0]
    assert predictions[0]["expected_points_next_gw"] >= 0
