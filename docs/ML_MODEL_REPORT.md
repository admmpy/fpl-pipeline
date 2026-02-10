# ML Model Report: FPL XGBoost (Formal)

Date: 2026-02-04
Scope: Review of the model training and inference pipeline in `scripts/train_model.py` and `tasks/ml_tasks.py` for FPL next-gameweek points prediction.

## 1. Executive Summary
The current model is an XGBoost regressor trained to predict next gameweek points using current gameweek player, team, and opponent features. The pipeline is generally sound with time-aware validation and leakage controls, but there are material mismatches between the model objective and evaluation, and between training and inference features. These issues can introduce bias and reduce operational accuracy.

## 2. Model Overview
Algorithm: XGBoost Regressor
Objective: Quantile regression (`reg:quantileerror`) with `quantile_alpha=0.8`
Evaluation Metrics: MAE, RMSE, R²
Validation: TimeSeriesSplit CV (5 folds) and a holdout split of the last 3 gameweeks
Artifact: `logs/model.bin` (model + metadata)

## 3. Data and Feature Pipeline
Source: `fct_ml_player_features` (Snowflake), joined with `dim_players` for `position_id`.
Target: `target_next_gw_points` created by shifting `total_points` forward per player.
Feature Engineering:
- Position one-hot: `is_gk`, `is_def`, `is_mid`, `is_fwd`
- Home flag: `is_home`
- Z-scores for `total_points`, `minutes_played`, `ict_index`
- Rolling, team, opponent, and strength metrics

## 4. Key Findings

### 4.1 Objective / Evaluation Mismatch (High Impact)
The model optimizes a 0.8 quantile objective but is evaluated using MAE/RMSE against actuals. This systematically shifts predictions upward (by design) and makes MAE/RMSE less interpretable. This can be partially offset by shrinkage, but it conflates model behavior with post-processing.

Impact:
- Predictions are biased high relative to the mean outcome.
- Reported MAE does not measure the objective the model is trained for.

### 4.2 Training vs Inference Feature Mismatch (High Impact)
During inference, `is_home` is hard-coded to 0 (neutral), while training uses the historical `was_home`. This changes feature distributions and may shift predictions, especially for home/away-sensitive players or teams.

Impact:
- Model receives systematically different inputs at inference than during training.
- Potential accuracy loss and unstable weekly behavior.

### 4.3 Feature Schema Not Persisted (Medium Impact)
The trained artifact does not store a feature list or preprocessing spec. Inference relies on a hardcoded list in `tasks/ml_tasks.py`. Feature additions/removals can cause silent drift or incorrect alignment.

Impact:
- Increased risk of training/inference drift.
- Harder debugging when schema changes.

### 4.4 Quantile Model with Mean Shrinkage (Medium Impact)
Shrinkage targets the mean of the training distribution, while the model outputs a quantile. This mixes statistical targets and can distort the quantile interpretation.

Impact:
- Predictions become a hybrid of quantile and mean estimates.
- Bias behavior is harder to reason about.

## 5. Strengths
- Time-series-aware cross-validation reduces leakage risk.
- Target construction (next-GW shift) aligns features correctly.
- Holdout split by future gameweeks is appropriate for temporal forecasting.
- Z-score normalization uses training-only statistics for holdout (no leakage).

## 6. Recommendations

### Priority A (Correctness / Bias)
1. Align objective and evaluation:
   - If the goal is mean prediction, use a mean loss (e.g., `reg:squarederror`) and keep MAE/RMSE.
   - If the goal is quantile prediction, report pinball loss and quantile coverage metrics.
2. Use actual next-gameweek fixtures to set `is_home` at inference.

### Priority B (Stability / Maintainability)
3. Persist the feature list and preprocessing metadata in the model artifact and load it during inference.
4. If keeping quantile objective, consider quantile-aware shrinkage or remove mean shrinkage.

## 7. Residual Risks
- Data drift (team form, injuries) can degrade performance between retrains.
- Inference performance depends on upstream data quality and timely feature refresh.

## 8. Proposed Next Steps
1. Confirm whether the prediction target should be mean or upper-quantile.
2. Implement feature schema persistence and fixture-based `is_home` inference.
3. Add a quantile-appropriate evaluation section (pinball loss) if quantiles are retained.

