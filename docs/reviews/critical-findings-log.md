# Critical Findings Log

## Entry
- id: `SBP-001`
- date: `2026-03-01`
- severity: `Critical`
- area: `security`
- location: `utils/model_registry.py:35`
- issue: `Allowlist guard used prefix string matching, permitting path-prefix bypass outside logs root.`
- fix: `Replaced prefix check with parent-chain containment check using resolved path parents.`
- tradeoff: `Slightly stricter path validation; no functional regression for valid in-root paths.`
- verification: `pytest -q tests/test_model_registry.py` -> `4 passed`
- commit: `f857d5e`

## Entry
- id: `AR-005`
- date: `2026-03-01`
- severity: `Critical`
- area: `architecture`
- location: `scripts/run_autonomous_loop.py:65`
- issue: `CLI returned success exit code for FAILED terminal state, masking unrecoverable run failures in automation.`
- fix: `Updated runner to return success only for RECORDED state; FAILED now returns non-zero.`
- tradeoff: `Automation may now fail faster; callers depending on previous behaviour must handle non-zero status.`
- verification: `pytest -q tests/test_run_autonomous_loop.py` -> `2 passed`
- commit: `f857d5e`

## Entry
- id: `AR-006`
- date: `2026-03-01`
- severity: `Critical`
- area: `architecture`
- location: `scripts/refresh_local_training_snapshot.py:112`
- issue: `Parquet snapshot refresh depended on optional engines not pinned in repository requirements, causing refresh failure in clean environments.`
- fix: `Added pyarrow as a pinned runtime dependency in requirements.txt.`
- tradeoff: `Increases environment size and install time, but guarantees deterministic Parquet support for snapshot write/read operations.`
- verification: `venv/bin/python scripts/refresh_local_training_snapshot.py --output-dir pipeline/data/training --force` -> `status=refreshed`; `venv/bin/pytest -q tests/test_run_autonomous_loop.py tests/test_local_data.py tests/test_refresh_local_training_snapshot.py tests/test_autonomous_loop.py` -> `15 passed`
- commit: `2b5958c`
