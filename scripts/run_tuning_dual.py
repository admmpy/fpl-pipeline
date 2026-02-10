"""
Entry point for the dual-agent LangGraph hyperparameter tuning workflow.

Usage:
    python scripts/run_tuning_dual.py [--max-iterations 10]
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

# Add pipeline root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run dual-agent tuning workflow")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum number of tuning iterations (default: 10)",
    )
    args = parser.parse_args()

    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(env_path)

    if not os.environ.get("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("FPL DUAL-AGENT TUNING WORKFLOW")
    logger.info("=" * 60)
    logger.info(f"Max iterations: {args.max_iterations}")

    from agents.tuning_graph_dual import build_tuning_graph_dual

    graph = build_tuning_graph_dual()

    result = graph.invoke(
        {"max_iterations": args.max_iterations},
        {"recursion_limit": args.max_iterations * 6 + 10},
    )

    logger.info("=" * 60)
    logger.info("TUNING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total experiments: {len(result['experiments'])}")
    logger.info(f"Best holdout MAE: {result['best_holdout_mae']:.4f}")
    logger.info(f"Best params: {result['best_params']}")

    if result.get("analysis"):
        logger.info(f"\nFinal LLM analysis:\n{result['analysis']}")


if __name__ == "__main__":
    main()
