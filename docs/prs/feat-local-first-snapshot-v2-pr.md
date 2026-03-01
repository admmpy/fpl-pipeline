# Pull Request: feat/local-first-snapshot-v2

## Template Status
Repository PR template file `.github/pull_request_template.md` is not present in this repository. This PR body uses a best-effort structure aligned to the senior review gate requirements.

## Summary
Implements a local-first training data snapshot system with deterministic refresh, atomic `latest.*` pointer updates, schema/contract validation, explicit runtime fallback policies, observability metadata, and operational documentation.

Includes one critical architecture fix identified during gate review:
- pin `pyarrow` for deterministic Parquet snapshot write/read support in clean environments (`AR-006`)

## Changes
- Added local-first data utilities:
  - `utils/local_data.py`
- Added snapshot refresh command:
  - `scripts/refresh_local_training_snapshot.py`
- Integrated source/policy loader controls:
  - `scripts/train_model.py`
  - `tasks/ml_tasks.py`
  - `agents/autonomous_nodes.py`
  - `scripts/run_autonomous_loop.py`
- Added tests:
  - `tests/test_local_data.py`
  - `tests/test_refresh_local_training_snapshot.py`
  - `tests/test_run_autonomous_loop.py` (updated args)
- Added docs:
  - `docs/LOCAL_DATA_WORKFLOW.md`
  - `docs/reviews/architecture-review-local-first-snapshot.md`
  - `security_best_practices_report.md` (new review section)
  - `docs/reviews/critical-findings-log.md` (AR-006 entry)
  - `README.md`
  - `docs/ML_TRAINING.md`
  - `ENV_EXAMPLE.txt`
- Added snapshot storage git hygiene:
  - `.gitignore`
  - `data/.gitkeep`
  - `data/training/.gitkeep`
- Added dependency pin:
  - `requirements.txt` (`pyarrow==23.0.1`)

## Review Outcomes
### Security audit
- Report: `security_best_practices_report.md`
- Critical findings in this scope: none.

### Architecture review
- Report: `docs/reviews/architecture-review-local-first-snapshot.md`
- Critical finding fixed: `AR-006` (Parquet dependency pinning).

### Critical fixes log
- Updated: `docs/reviews/critical-findings-log.md` with `AR-006`.

## Test Evidence
Commands executed:
- `venv/bin/python scripts/refresh_local_training_snapshot.py --output-dir pipeline/data/training --force`
  - Result: refreshed successfully (`row_count=21523`, `status=refreshed`)
- `venv/bin/pytest -q tests/test_run_autonomous_loop.py tests/test_local_data.py tests/test_refresh_local_training_snapshot.py tests/test_autonomous_loop.py`
  - Result: `15 passed, 3 warnings`

## Risk and Rollback
- Main risk:
  - Operational drift if environments set inconsistent source/policy variables.
  - Manual intervention required for stale lock scenarios.
- Rollback strategy:
  - Set `TRAINING_DATA_SOURCE=snowflake` for immediate runtime fallback away from local snapshots.
  - Revert branch commits if required.
  - Local snapshot pointer rollback by restoring prior versioned `features_*.parquet` and its manifest to `latest.*`.

## Checklist
- [x] Security review completed
- [x] Architecture review completed
- [x] Critical findings fixed (where applicable)
- [x] Tests passed for touched scope
- [x] Rollback strategy documented
