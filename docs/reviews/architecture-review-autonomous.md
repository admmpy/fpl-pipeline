# Architecture Review: Autonomous Optimisation Loop

Date: 2026-03-01
Branch: `feat/autonomous-optimiser-phase1`

## Severity-Ranked Findings

### Critical
- No outstanding critical architecture findings after fixes in commit `f857d5e`.

### High
1. **AR-001: State payload carries full DataFrames across graph nodes**
- Location: `agents/autonomous_nodes.py:425`, `agents/autonomous_nodes.py:524`
- Risk: Memory pressure for large snapshots, tighter coupling between orchestration and data payload lifecycle.
- Tradeoff: Current approach keeps deterministic, single-process behaviour simple.
- Recommendation: Move large frames to ephemeral, versioned run artefacts with references in state.

### Medium
1. **AR-002: Evidence contract split between precheck and terminal write**
- Location: `agents/autonomous_nodes.py:925`, `agents/autonomous_nodes.py:396`
- Risk: Promotion is gated by precheck evidence write, while terminal evidence can still fail later.
- Tradeoff: Prevents promote-without-any-evidence, but does not guarantee complete post-decision evidence in all scenarios.
- Recommendation: Atomic two-phase evidence transaction with explicit completion marker.

2. **AR-003: Operational assumptions are local-node centric**
- Location: `utils/model_registry.py`, `agents/autonomous_nodes.py`
- Risk: Concurrent multi-run behaviour may need stronger file-lock semantics if moved beyond single-node execution.
- Tradeoff: Current phase scope targets local deterministic execution.
- Recommendation: Add advisory lock around registry writes for multi-run deployment.

### Low
1. **AR-004: Repeated broad exception capture reduces observability precision**
- Location: `agents/autonomous_nodes.py` node handlers
- Risk: Harder root-cause categorisation; state safety remains intact.
- Recommendation: Introduce structured error codes per failure class.

## Gate Decision
- Architecture gate: **PASS**
- Rationale: No unresolved critical defects; current risks are acceptable for the declared single-node phase scope.
