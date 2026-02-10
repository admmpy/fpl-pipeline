"""LangGraph assembly for the dual-agent tuning workflow."""

from langgraph.graph import StateGraph, END

from agents.tuning_state import TuningState
from agents.tuning_nodes_review import (
    fetch_data,
    train_evaluate,
    propose_params,
    review_candidate,
    check_convergence,
    route_after_review,
    save_best,
)


def build_tuning_graph_dual():
    """
    Build and compile the dual-agent tuning graph.

    Flow:
        START → fetch_data → train_evaluate → review_candidate → check_convergence
                                   ↑                              │
                                   └──── propose_params ──────────┘
    """
    graph = StateGraph(TuningState)

    graph.add_node("fetch_data", fetch_data)
    graph.add_node("train_evaluate", train_evaluate)
    graph.add_node("propose_params", propose_params)
    graph.add_node("review_candidate", review_candidate)
    graph.add_node("check_convergence", lambda state: state)
    graph.add_node("save_best", save_best)

    graph.set_entry_point("fetch_data")
    graph.add_edge("fetch_data", "train_evaluate")
    graph.add_edge("train_evaluate", "review_candidate")

    graph.add_conditional_edges(
        "review_candidate",
        route_after_review,
        {
            "check_convergence": "check_convergence",
            "propose_params": "propose_params",
            "save_best": "save_best",
        },
    )

    graph.add_edge("propose_params", "train_evaluate")

    graph.add_conditional_edges(
        "check_convergence",
        check_convergence,
        {
            "propose_params": "propose_params",
            "save_best": "save_best",
        },
    )

    graph.add_edge("save_best", END)

    return graph.compile()
