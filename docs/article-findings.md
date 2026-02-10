# Article Findings: ML Interview Concepts and Project Alignment

This document summarizes the user-provided article excerpts (Concepts 1–5) and maps them to the FPL pipeline's ML model and workflow.

## Summary of Concepts

1. Bias-Variance Trade-off
High variance = overfitting (train high, validation low). Use learning curves to diagnose. Fix with regularization, reducing model complexity, more data, and cross-validation.

2. Class Imbalance
Accuracy can be misleading with skewed classes. Use precision/recall/F1/ROC-AUC, class weighting, or resampling (e.g., SMOTE).

3. Feature Engineering vs Feature Selection
Feature engineering (domain-driven) often beats brute selection. Feature selection reduces complexity and overfitting risk.

4. Cross-Validation
Single train/test split can be misleading. Use K-fold (stratified if imbalanced), check variance across folds, and consider time-based holdouts.

5. Regularization (L1/L2/ElasticNet)
L2 shrinks coefficients (keeps all features), L1 selects features (zeros out some), ElasticNet combines both. Choose based on feature relevance and model complexity.

## How This Maps to the FPL ML Model

### Bias-Variance Trade-off
- The training script uses a constrained XGBoost configuration (`max_depth`, `reg_alpha`, `reg_lambda`) to manage variance. See `pipeline/scripts/train_model.py`.
- The model’s evaluation includes cross-validation and a time-based holdout. That reduces the risk of hidden variance and aligns with the article’s diagnostic stance.

### Class Imbalance (Analogous Risks in Regression)
- The model is a regression task (points prediction), so class imbalance does not directly apply.
- The closest analogue is skewed target distributions and sparse minutes. The training pipeline filters to `minutes_played > 0` to avoid heavy noise from non-playing data. See `pipeline/scripts/train_model.py`.
- In inference, all players are retained, which may introduce additional variance for low-minute players. This is a deliberate tradeoff for recommendation coverage. See `pipeline/tasks/ml_tasks.py`.

### Feature Engineering vs Feature Selection
- The model relies on extensive engineered features (rolling averages, opponent strength, z-scores). See `pipeline/scripts/train_model.py` and `pipeline/tasks/ml_tasks.py`.
- There is no explicit feature selection step. Instead, feature importance is reported post-training to validate signal quality.

### Cross-Validation and Holdouts
- The training uses `TimeSeriesSplit` cross-validation and a gameweek-based holdout, which fits the temporal nature of FPL data. See `pipeline/scripts/train_model.py`.
- This directly matches the article’s guidance on avoiding a single lucky split and preferring time-based validation when data is temporal.

### Regularization
- Regularization is explicitly configured in training (`reg_alpha`, `reg_lambda`) and the model complexity is bounded (`max_depth`). See `pipeline/scripts/train_model.py`.
- These settings align with the article’s guidance: control complexity to reduce variance while keeping predictive signal.

## Gaps and Opportunities (Based on the Concepts)

1. Train/Inference Preprocessing Consistency
- Training uses gameweek-based z-score stats computed from training data, while inference recomputes z-scores on live data per gameweek. This is defensible but not identical.
- Option: Persist training z-score statistics and apply them in inference for stricter alignment.

2. Drift Monitoring
- The pipeline does not currently log or alert on feature distribution drift or prediction drift.
- Option: add a lightweight drift check (e.g., z-score mean/variance shifts) in the orchestration step and alert via Slack if thresholds are crossed.

3. Feature Selection Review
- There is no formal pruning of weak features.
- Option: Use feature importance reports or ablation runs to remove consistently low-signal features, improving model stability and interpretability.

## References to Project Files

- Training: `pipeline/scripts/train_model.py`
- Inference: `pipeline/tasks/ml_tasks.py`
- Orchestration: `pipeline/flows/fpl_orchestration.py`
