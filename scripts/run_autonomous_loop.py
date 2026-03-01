"""Run the autonomous optimisation LangGraph loop from the command line."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid

from pathlib import Path

# Add pipeline root so relative imports work when executed as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.autonomous_graph import build_autonomous_graph
from agents.autonomous_nodes import DomainRulesError, load_domain_rules
from agents.autonomous_state import create_initial_state


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run autonomous optimisation loop")
    parser.add_argument("--run-id", default=f"autonomous_{uuid.uuid4().hex[:12]}")
    parser.add_argument("--rules-path", default="config/domain_rules.yaml")
    parser.add_argument("--snapshot-path", default=None)
    parser.add_argument("--local-snapshot-path", default=None)
    parser.add_argument("--data-source", choices=["local", "snowflake"], default=None)
    parser.add_argument(
        "--data-policy",
        choices=["LOCAL_ONLY", "LOCAL_THEN_SNOWFLAKE"],
        default=None,
    )
    parser.add_argument("--force-drift", action="store_true")
    parser.add_argument("--force-no-drift", action="store_true")
    parser.add_argument("--optuna-trials", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    try:
        rules = load_domain_rules(args.rules_path)
    except DomainRulesError as exc:
        print(f"Rules validation failed before run start: {exc}", file=sys.stderr)
        return 2

    snapshot_meta = {
        "rules_path": str(Path(args.rules_path)),
        "force_drift": bool(args.force_drift),
        "force_no_drift": bool(args.force_no_drift),
    }
    resolved_local_snapshot = args.local_snapshot_path or args.snapshot_path
    if resolved_local_snapshot:
        snapshot_meta["local_snapshot_path"] = resolved_local_snapshot
        snapshot_meta["snapshot_path"] = resolved_local_snapshot
    if args.data_source:
        snapshot_meta["data_source"] = args.data_source
    if args.data_policy:
        snapshot_meta["data_policy"] = args.data_policy
    if args.optuna_trials:
        snapshot_meta["optuna_trials"] = args.optuna_trials

    graph = build_autonomous_graph()
    state = create_initial_state(run_id=args.run_id, snapshot_meta=snapshot_meta)
    final_state = graph.invoke(state)

    output = {
        "run_id": final_state.get("run_id"),
        "state": final_state.get("state"),
        "rules_version": rules.get("version"),
        "evidence_path": final_state.get("evidence_path"),
        "active_model_version": final_state.get("active_model_version"),
        "error": final_state.get("error"),
    }
    print(json.dumps(output, indent=2, sort_keys=True))

    return 0 if final_state.get("state") == "RECORDED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
