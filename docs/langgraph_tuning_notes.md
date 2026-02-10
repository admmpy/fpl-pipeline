# Dual-Agent LangGraph Tuning Notes

## Overview
This repo includes a dual-agent tuning workflow that optimizes XGBoost hyperparameters while enforcing realism and bias-variance checks. The optimizer proposes new parameters and a reviewer validates feasibility against historical player score distributions and gate thresholds.

## Entry Point
- `python scripts/run_tuning_dual.py --max-iterations 10`

This runs the dual-agent workflow and saves the best feasible model.

## Environment Variables
The workflow uses OpenRouter for LLM calls:
- `OPENROUTER_API_KEY`
- `OPENROUTER_BASE_URL` (default: `https://openrouter.ai/api/v1`)
- `OPENROUTER_MODEL` (default: `anthropic/claude-sonnet-4-5`)

Optional gate controls (see `ENV_EXAMPLE.txt` for defaults):
- `TUNING_BIAS_SD_MULT`
- `TUNING_POS_P95_EXCEED_PCT`
- `TUNING_PLAYER_P95_EXCEED_PCT`
- `TUNING_RMSE_WORSE_PCT`
- `TUNING_REVIEW_MAX_REJECTS`
- `TUNING_NO_IMPROVE_ITERS`
- `TUNING_PLAYER_MIN_HISTORY`
- `TUNING_LOG_TARGET` (default: on; signed log transform for targets)

## Fast Run (Quick Validation)
Use a short run with stricter gates:

```bash
TUNING_NO_IMPROVE_ITERS=2 \
TUNING_POS_P95_EXCEED_PCT=0.03 \
TUNING_PLAYER_P95_EXCEED_PCT=0.03 \
python scripts/run_tuning_dual.py --max-iterations 3
```

## Notes
- `scripts/run_once.py` is separate. When you use it, run with `RUN_ONCE_FAST=1` to avoid long full runs.
- The dual-agent workflow only updates best params when gates pass.
- Targets use a signed log transform by default (`LOG_TARGET=1` for training; `TUNING_LOG_TARGET=1` for tuning).
- Minutes-band features are included: `minutes_band_0_30`, `minutes_band_31_60`, `minutes_band_61_90`.
- Tuning search space now includes `min_child_weight`, `gamma`, and `max_delta_step`.
