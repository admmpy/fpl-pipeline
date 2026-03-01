"""Tests for autonomous lifecycle transition guards."""

import pytest

from agents.autonomous_state import (
    StateGuardError,
    TransitionError,
    create_initial_state,
    ensure_transition,
    guard_node_state,
)


def test_initial_transition_must_start_with_ingested():
    with pytest.raises(TransitionError):
        ensure_transition(None, "VALIDATED")


def test_allowed_transition_sequence():
    ensure_transition(None, "INGESTED")
    ensure_transition("INGESTED", "VALIDATED")
    ensure_transition("VALIDATED", "DRIFT_TRIGGERED")
    ensure_transition("DRIFT_TRIGGERED", "SEARCHED")
    ensure_transition("SEARCHED", "TRAINED")
    ensure_transition("TRAINED", "EVALUATED")
    ensure_transition("EVALUATED", "RULES_PASSED")
    ensure_transition("RULES_PASSED", "PROMOTED")
    ensure_transition("PROMOTED", "RECORDED")


def test_invalid_transition_raises():
    with pytest.raises(TransitionError):
        ensure_transition("INGESTED", "SEARCHED")


def test_any_state_can_fail():
    ensure_transition("TRAINED", "FAILED")
    ensure_transition("RECORDED", "FAILED")


def test_guard_node_state_enforces_entry_contract():
    state = create_initial_state("run-1")
    state["state"] = "INGESTED"
    guard_node_state(state, node_name="validate_snapshot", allowed_states={"INGESTED"})

    with pytest.raises(StateGuardError):
        guard_node_state(state, node_name="train_best_candidate", allowed_states={"SEARCHED"})
