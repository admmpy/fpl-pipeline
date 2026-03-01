"""LangGraph assembly for autonomous optimisation lifecycle orchestration."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from agents.autonomous_nodes import (
    apply_domain_rules,
    detect_drift,
    evaluate_candidate,
    ingest_snapshot,
    promote_model_transaction,
    record_evidence,
    rollback_active_model,
    run_optuna_search,
    train_best_candidate,
    validate_snapshot,
)
from agents.autonomous_state import AutonomousState


def _route_after_drift(state: AutonomousState) -> str:
    current = state.get("state")
    if current == "DRIFT_TRIGGERED":
        return "run_optuna_search"
    if current in {"RECORDED", "FAILED"}:
        return "record_evidence"
    return "record_evidence"


def _route_after_rules(state: AutonomousState) -> str:
    current = state.get("state")
    if current == "RULES_PASSED":
        return "promote_model_transaction"
    return "record_evidence"


def _route_after_promotion(state: AutonomousState) -> str:
    current = state.get("state")
    if current == "PROMOTED":
        return "record_evidence"
    if current == "ROLLED_BACK":
        return "rollback_active_model"
    return "record_evidence"


def build_autonomous_graph():
    """Build and compile the autonomous optimisation graph."""

    graph = StateGraph(AutonomousState)

    graph.add_node("ingest_snapshot", ingest_snapshot)
    graph.add_node("validate_snapshot", validate_snapshot)
    graph.add_node("detect_drift", detect_drift)
    graph.add_node("run_optuna_search", run_optuna_search)
    graph.add_node("train_best_candidate", train_best_candidate)
    graph.add_node("evaluate_candidate", evaluate_candidate)
    graph.add_node("apply_domain_rules", apply_domain_rules)
    graph.add_node("promote_model_transaction", promote_model_transaction)
    graph.add_node("rollback_active_model", rollback_active_model)
    graph.add_node("record_evidence", record_evidence)

    graph.set_entry_point("ingest_snapshot")
    graph.add_edge("ingest_snapshot", "validate_snapshot")
    graph.add_edge("validate_snapshot", "detect_drift")

    graph.add_conditional_edges(
        "detect_drift",
        _route_after_drift,
        {
            "run_optuna_search": "run_optuna_search",
            "record_evidence": "record_evidence",
        },
    )

    graph.add_edge("run_optuna_search", "train_best_candidate")
    graph.add_edge("train_best_candidate", "evaluate_candidate")
    graph.add_edge("evaluate_candidate", "apply_domain_rules")

    graph.add_conditional_edges(
        "apply_domain_rules",
        _route_after_rules,
        {
            "promote_model_transaction": "promote_model_transaction",
            "record_evidence": "record_evidence",
        },
    )

    graph.add_conditional_edges(
        "promote_model_transaction",
        _route_after_promotion,
        {
            "rollback_active_model": "rollback_active_model",
            "record_evidence": "record_evidence",
        },
    )

    graph.add_edge("rollback_active_model", "record_evidence")
    graph.add_edge("record_evidence", END)

    return graph.compile()
