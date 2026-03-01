"""Tests for autonomous loop runner exit semantics."""

from __future__ import annotations

from types import SimpleNamespace

import scripts.run_autonomous_loop as runner


class _FakeGraph:
    def __init__(self, state: str):
        self._state = state

    def invoke(self, _state):
        return {
            "run_id": "run-test",
            "state": self._state,
            "evidence_path": None,
            "active_model_version": None,
            "error": None,
        }


def test_main_returns_non_zero_when_final_state_failed(monkeypatch):
    monkeypatch.setattr(
        runner,
        "_parse_args",
        lambda: SimpleNamespace(
            run_id="run-test",
            rules_path="config/domain_rules.yaml",
            snapshot_path=None,
            local_snapshot_path=None,
            data_source=None,
            data_policy=None,
            force_drift=False,
            force_no_drift=False,
            optuna_trials=None,
        ),
    )
    monkeypatch.setattr(runner, "load_domain_rules", lambda _path: {"version": "1.0.0"})
    monkeypatch.setattr(runner, "build_autonomous_graph", lambda: _FakeGraph("FAILED"))

    assert runner.main() == 1


def test_main_returns_zero_when_final_state_recorded(monkeypatch):
    monkeypatch.setattr(
        runner,
        "_parse_args",
        lambda: SimpleNamespace(
            run_id="run-test",
            rules_path="config/domain_rules.yaml",
            snapshot_path=None,
            local_snapshot_path=None,
            data_source=None,
            data_policy=None,
            force_drift=False,
            force_no_drift=False,
            optuna_trials=None,
        ),
    )
    monkeypatch.setattr(runner, "load_domain_rules", lambda _path: {"version": "1.0.0"})
    monkeypatch.setattr(runner, "build_autonomous_graph", lambda: _FakeGraph("RECORDED"))

    assert runner.main() == 0
