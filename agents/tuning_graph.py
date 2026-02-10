"""LangGraph assembly for the hyperparameter tuning agent."""

from langgraph.graph import StateGraph, END

from agents.tuning_state import TuningState
from agents.tuning_nodes import (
    fetch_data,
    train_evaluate,
    analyze_suggest,
    check_convergence,
    save_best,
)


def build_tuning_graph():
    """
    Build and compile the tuning agent graph.

    Flow:
        START → fetch_data → train_evaluate → analyze_suggest → check_convergence
                                  ↑                                     │
                                  └──── (not converged) ────────────────┘
                                                                        │
                                                             (converged)│
                                                                        ▼
                                                                   save_best → END
    """
    graph = StateGraph(TuningState)

    # Add nodes
    graph.add_node("fetch_data", fetch_data)
    graph.add_node("train_evaluate", train_evaluate)
    graph.add_node("analyze_suggest", analyze_suggest)
    graph.add_node("check_convergence", lambda state: state)  # routing-only node
    graph.add_node("save_best", save_best)

    # Edges
    graph.set_entry_point("fetch_data")
    graph.add_edge("fetch_data", "train_evaluate")
    graph.add_edge("train_evaluate", "analyze_suggest")
    graph.add_edge("analyze_suggest", "check_convergence")

    # Conditional edge from check_convergence
    graph.add_conditional_edges(
        "check_convergence",
        check_convergence,
        {
            "train_evaluate": "train_evaluate",
            "save_best": "save_best",
        },
    )

    graph.add_edge("save_best", END)

    return graph.compile()
