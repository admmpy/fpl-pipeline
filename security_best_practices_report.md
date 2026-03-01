# Security Best Practices Report

Date: 2026-03-01
Branch: `feat/autonomous-optimiser-phase1`
Scope: autonomous optimisation loop additions (`agents/autonomous_*`, `utils/model_registry.py`, `scripts/run_autonomous_loop.py`, related tests)

## Severity-Ranked Findings

### Critical
1. **SBP-001: Path allowlist prefix-bypass in model registry filesystem guard**
- Location: `utils/model_registry.py:35`
- Impact: A path such as `.../logs_evil/...` could pass a naive string-prefix check intended to constrain reads/writes to `logs/`, weakening the artefact and registry confinement control.
- Status: **Fixed** in commit `f857d5e`.
- Fix: Replaced prefix string matching with parent-chain containment check (`resolved == logs_resolved or logs_resolved in resolved.parents`).
- Verification: `pytest -q tests/test_model_registry.py` includes `test_allowlist_rejects_prefix_path_bypass`.

2. **SBP-002: Runner process returned success code on unrecoverable FAILED state**
- Location: `scripts/run_autonomous_loop.py:65`
- Impact: Automation/CI could treat a failed autonomous run as successful, suppressing incident handling and rollback responses.
- Status: **Fixed** in commit `f857d5e`.
- Fix: Runner now returns `0` only for `RECORDED`, otherwise non-zero.
- Verification: `pytest -q tests/test_run_autonomous_loop.py` validates FAILED -> exit `1`.

### High
- No high-severity findings remaining in reviewed scope.

### Medium
1. **SBP-003: Unpickling remains a trusted-boundary assumption**
- Location: `utils/model_registry.py:171`, `agents/autonomous_nodes.py:815`
- Risk: `pickle.load` is unsafe for untrusted artefacts. Current design assumes `logs/` artefacts are controlled.
- Decision: Deferred (non-critical under current local-registry trust model).
- Recommended follow-up: Signed artefacts or a safer serialisation strategy for model metadata and model object loading boundaries.

### Low
1. **SBP-004: Broad exception capture masks failure classes in some nodes**
- Location: `agents/autonomous_nodes.py` (multiple node `except Exception` blocks)
- Risk: Reduces triage granularity; does not currently bypass fail-safe path to `FAILED`.
- Decision: Deferred.

## Overall Gate Decision
- Security gate: **PASS WITH CRITICAL FIXES APPLIED**
- Critical findings fixed and regression tested.
