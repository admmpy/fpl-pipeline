# Pull Request: feat/autonomous-optimiser-phase1

## Template Status
Repository PR template file `.github/pull_request_template.md` is not present in this repository. This PR body uses a best-effort structure aligned to the release gate requirements.

## Summary
Implements an autonomous optimisation loop with strict lifecycle transitions, hard-gated rule validation, deterministic drift-triggered retraining, atomic model promotion/rollback, evidence bundles, and security redaction controls.

Also applies two critical gate fixes discovered during senior review:
- path allowlist bypass hardening in model registry
- non-zero CLI exit on FAILED runs

## Changes
- Added autonomous lifecycle/state contracts:
  - `agents/autonomous_state.py`
  - `agents/autonomous_graph.py`
  - `agents/autonomous_nodes.py`
- Added domain rule contracts:
  - `config/domain_rules.yaml`
  - `config/domain_rules.schema.json`
- Added model registry with atomic promotion and rollback:
  - `utils/model_registry.py`
- Added runner and docs:
  - `scripts/run_autonomous_loop.py`
  - `docs/AUTONOMOUS_OPTIMISATION.md`
- Added security and architecture review artefacts:
  - `security_best_practices_report.md`
  - `docs/reviews/architecture-review-autonomous.md`
  - `docs/reviews/critical-findings-log.md`
- Added test coverage:
  - `tests/test_autonomous_state.py`
  - `tests/test_autonomous_rules.py`
  - `tests/test_model_registry.py`
  - `tests/test_autonomous_security.py`
  - `tests/test_autonomous_loop.py`
  - `tests/test_run_autonomous_loop.py`
  - `tests/conftest.py`

## Review Outcomes
### Security audit
- Report: `security_best_practices_report.md`
- Critical findings fixed: `SBP-001`, `SBP-002`

### Architecture review
- Report: `docs/reviews/architecture-review-autonomous.md`
- No outstanding critical issues.

### Critical fixes log
- Updated: `docs/reviews/critical-findings-log.md`

## Test Evidence
Commands executed:
- `pytest -q tests/test_model_registry.py tests/test_run_autonomous_loop.py tests/test_autonomous_state.py tests/test_autonomous_rules.py tests/test_autonomous_security.py tests/test_autonomous_loop.py`
  - Result: `19 passed, 3 warnings`
- `pytest -q tests/test_ml_functions.py tests/test_autonomous_state.py tests/test_autonomous_rules.py tests/test_model_registry.py tests/test_autonomous_security.py tests/test_autonomous_loop.py tests/test_run_autonomous_loop.py`
  - Result: `49 passed, 3 warnings`

## Risk and Rollback
- Main risk: local file-based registry contention in multi-run environments.
- Rollback strategy:
  - Git rollback to prior commit anchors on this branch.
  - Runtime model rollback via `utils/model_registry.rollback_to(version)`.

## Checklist
- [x] Security review completed
- [x] Architecture review completed
- [x] Critical findings fixed
- [x] Tests passed for touched scope
- [x] Rollback strategy documented
