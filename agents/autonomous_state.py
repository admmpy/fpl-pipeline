"""State contracts and lifecycle guards for the autonomous optimisation loop."""

from __future__ import annotations

from typing import Any, Literal, Optional
from typing_extensions import TypedDict


LifecycleState = Literal[
    "INGESTED",
    "VALIDATED",
    "DRIFT_TRIGGERED",
    "SEARCHED",
    "TRAINED",
    "EVALUATED",
    "RULES_PASSED",
    "PROMOTED",
    "REJECTED",
    "ROLLED_BACK",
    "RECORDED",
    "FAILED",
]


ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "INGESTED": {"VALIDATED", "FAILED"},
    "VALIDATED": {"DRIFT_TRIGGERED", "RECORDED", "FAILED"},
    "DRIFT_TRIGGERED": {"SEARCHED", "FAILED"},
    "SEARCHED": {"TRAINED", "FAILED"},
    "TRAINED": {"EVALUATED", "FAILED"},
    "EVALUATED": {"RULES_PASSED", "REJECTED", "FAILED"},
    "RULES_PASSED": {"PROMOTED", "ROLLED_BACK", "FAILED"},
    "PROMOTED": {"RECORDED", "FAILED"},
    "REJECTED": {"RECORDED", "FAILED"},
    "ROLLED_BACK": {"RECORDED", "FAILED"},
    "RECORDED": set(),
    "FAILED": set(),
}


class AutonomousState(TypedDict, total=False):
    """Typed state for autonomous optimisation runs."""

    run_id: str
    state: LifecycleState
    snapshot_meta: dict[str, Any]
    validation_report: dict[str, Any]
    drift_report: dict[str, Any]
    optuna_result: dict[str, Any]
    candidate_metrics: dict[str, Any]
    rule_eval: dict[str, Any]
    promotion_decision: dict[str, Any]
    active_model_version: Optional[str]
    previous_model_version: Optional[str]
    evidence_path: Optional[str]
    error: Optional[dict[str, Any]]


class TransitionError(ValueError):
    """Raised when a lifecycle transition is invalid."""


class StateGuardError(ValueError):
    """Raised when a node is entered with an invalid state."""


def create_initial_state(run_id: str, snapshot_meta: Optional[dict[str, Any]] = None) -> AutonomousState:
    """Create an initial autonomous state before graph execution."""

    return AutonomousState(
        run_id=run_id,
        snapshot_meta=snapshot_meta or {},
        validation_report={},
        drift_report={},
        optuna_result={},
        candidate_metrics={},
        rule_eval={},
        promotion_decision={},
        active_model_version=None,
        previous_model_version=None,
        evidence_path=None,
        error=None,
    )


def guard_node_state(
    state: AutonomousState,
    *,
    node_name: str,
    allowed_states: set[str],
) -> None:
    """Validate node entry state and raise if invalid."""

    current = state.get("state")
    if current not in allowed_states:
        raise StateGuardError(
            f"{node_name} received invalid state '{current}'. Allowed: {sorted(allowed_states)}"
        )


def ensure_transition(current: Optional[str], target: str) -> None:
    """Validate lifecycle transitions, including fail-fast to FAILED."""

    if target == "FAILED":
        return
    if current is None:
        if target != "INGESTED":
            raise TransitionError(
                f"Initial transition must target INGESTED, got '{target}'"
            )
        return

    allowed = ALLOWED_TRANSITIONS.get(current)
    if allowed is None:
        raise TransitionError(f"Unknown current state '{current}'")
    if target not in allowed:
        raise TransitionError(
            f"Invalid transition '{current}' -> '{target}'. Allowed: {sorted(allowed)}"
        )


def transition_update(state: AutonomousState, target: LifecycleState) -> dict[str, str]:
    """Return a state update after validating transition legality."""

    ensure_transition(state.get("state"), target)
    return {"state": target}
