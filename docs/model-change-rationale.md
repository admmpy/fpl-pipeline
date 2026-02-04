# Model Change Rationale (Brief)

## Problem
Recent recommendations show unrealistic point totals (e.g., 30–40 points for multiple players). This indicates systematic over‑prediction.

## Key Decisions
- **Predict mean points** (switch to `reg:squarederror`) to remove the built‑in upward bias from quantile loss.
- **Align train/inference preprocessing** by using training‑derived global z‑score stats and removing `is_home` until fixtures are integrated.
- **Add guardrails** with per‑position 95th‑percentile caps to prevent extreme outputs.
- **Use a 5‑GW backtest window** to match rolling‑average features and validate bias.

## Expected Impact
- Predictions centered around realistic weekly outcomes.
- Reduced extreme outliers while preserving rank order.
- More stable and explainable model behavior week‑to‑week.

## How to Validate
- Backtest bias near 0 on last 5 GWs.
- No prediction exceeds per‑position caps.
- Recommended squad totals fall within plausible historical ranges.
