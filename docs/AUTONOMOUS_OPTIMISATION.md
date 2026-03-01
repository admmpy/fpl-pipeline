# Autonomous Optimisation Loop

## Overview

The autonomous optimiser runs a guarded, deterministic lifecycle on top of LangGraph:

1. Ingest snapshot
2. Validate snapshot and rule pack
3. Detect drift
4. Run Optuna search (when drift is triggered)
5. Train/evaluate candidate
6. Apply hard domain rules
7. Promote atomically or rollback
8. Record evidence

## Contracts

- State contract: `agents/autonomous_state.py`
- Graph: `agents/autonomous_graph.py`
- Nodes: `agents/autonomous_nodes.py`
- Domain rules: `config/domain_rules.yaml`
- Rules schema: `config/domain_rules.schema.json`
- Registry: `utils/model_registry.py`

## Run

```bash
python scripts/run_autonomous_loop.py \
  --rules-path config/domain_rules.yaml \
  --force-drift
```

Useful flags:

- `--snapshot-path path/to/snapshot.csv`
- `--optuna-trials 4`
- `--force-no-drift`
- `--run-id autonomous_manual_001`

## Evidence and Logs

- Evidence bundle: `logs/autonomous/<run_id>.json`
- Structured node events: `logs/autonomous/events.jsonl`
- Model registry: `logs/model_registry.json`
- Active model compatibility path: `logs/model.bin`

## Security Controls

- Secret-like keys containing `KEY`, `TOKEN`, or `SECRET` are redacted in evidence and structured logs.
- Registry and artifact operations are path-allowlisted under `logs/`.
- Rules are validated against JSON schema before execution starts.
