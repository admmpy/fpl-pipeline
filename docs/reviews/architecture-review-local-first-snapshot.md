# Architecture Review: Local-First Snapshot System

Date: 2026-03-01
Branch: `feat/local-first-snapshot-v2`
Scope: `utils/local_data.py`, `scripts/refresh_local_training_snapshot.py`, integration points in training/tasks/autonomous/runtime CLI

## Severity-Ranked Findings

### Critical
1. **AR-006: Parquet runtime dependency missing from repository requirements**
- Location: `scripts/refresh_local_training_snapshot.py:112`, `requirements.txt`
- Impact: A clean environment can execute Snowflake fetch successfully, then fail at snapshot write (`to_parquet`) due missing engine, preventing weekly refresh and breaking local-first rollout.
- Status: **Fixed** in this branch.
- Fix: Added `pyarrow==23.0.1` to [requirements.txt](/Users/am/Sync/fpl-workspace/pipeline/requirements.txt).
- Tradeoff: Slightly larger environment footprint and installation time.

### High
- No high-severity architecture findings identified.

### Medium
1. **AR-007: Refresh lock lacks stale lock recovery path**
- Location: `scripts/refresh_local_training_snapshot.py:73`
- Risk: Unexpected process termination can leave a stale lock and require manual intervention to resume refreshes.
- Decision: Deferred (non-critical in current single-operator/local runtime model).
- Recommended follow-up: Add stale lock metadata TTL and `--force-unlock` guarded path.

2. **AR-008: Policy/configuration defaults span env + CLI + code and require operational discipline**
- Location: `utils/local_data.py:239`, `scripts/run_autonomous_loop.py:23`, `scripts/train_model.py:80`, `tasks/ml_tasks.py:107`
- Risk: Divergent operational usage can occur if environments are not standardised.
- Decision: Deferred.
- Recommended follow-up: Add a single startup diagnostics command to print effective source/policy/path values for all entrypoints.

### Low
- No low-severity architecture findings with immediate action required.

## Gate Decision
- Architecture gate: **PASS WITH CRITICAL FIX APPLIED**
- Critical fix implemented and re-verified.
