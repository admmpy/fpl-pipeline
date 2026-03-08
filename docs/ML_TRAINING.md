# XGBoost Model Training for FPL Predictions

This document explains how to train and use the XGBoost model for Fantasy Premier League player points prediction.

## Overview

The ML pipeline:
1. Fetches historical player features from `fct_ml_player_features` in Snowflake
2. Engineers temporal features and creates training targets
3. Trains an XGBoost regressor with time series cross-validation
4. Evaluates a temporal holdout set and reports bias, calibration deltas, and weekly realism/ranking metrics
5. Saves the model and metadata to `model.bin` for inference

## Training the Model

### Prerequisites

1. **Data Requirements:**
   - Local snapshot refreshed weekly (`python scripts/refresh_local_training_snapshot.py`) or Snowflake connection configured for direct mode
   - At least 10+ gameweeks of historical data in `fct_ml_player_features`
   - dbt models built: `dbt build` in `fpl_development/`

2. **Python Dependencies:**
   ```bash
   pip install xgboost scikit-learn
   ```

### Run Training

```bash
cd /Users/am/Sync/fpl-workspace/pipeline
source venv/bin/activate  # or activate your environment
python scripts/train_model.py
```

### Explicit Source Modes

```bash
# Local snapshot only (default policy for development)
TRAINING_DATA_SOURCE=local TRAINING_DATA_POLICY=LOCAL_ONLY python scripts/train_model.py

# Local-first with Snowflake fallback (rollout resilience mode)
TRAINING_DATA_SOURCE=local TRAINING_DATA_POLICY=LOCAL_THEN_SNOWFLAKE python scripts/train_model.py

# Direct Snowflake mode
TRAINING_DATA_SOURCE=snowflake python scripts/train_model.py
```

### Expected Output

```
============================================================
FPL XGBOOST MODEL TRAINING
============================================================
Fetching training data from Snowflake...
Fetched 15000 training samples across 20 gameweeks
Players: 750, Date range: GW4-GW23

Engineering features...
Feature engineering complete. 14250 samples remain after creating target.

Training data shape: (14250, 45)
Training target range: [0.0, 24.0] points
Training target mean: 3.45 points

Training XGBoost model...
Running time series cross-validation...
Cross-validation MAE: 2.35 (+/- 0.12)

Training final model on full dataset...

============================================================
MODEL EVALUATION
============================================================

Overall Metrics:
  MAE:  2.12 points
  RMSE: 3.45 points
  R²:   0.34

Top 10 Most Important Features:
  three_week_players_roll_avg_points       0.1234
  total_points                              0.0987
  form                                      0.0876
  ...

============================================================
HOLDOUT EVALUATION
============================================================

Holdout Metrics:
  MAE:  2.45 points
  RMSE: 3.80 points
  Bias: -0.15 points (actual - predicted)

Holdout bias by position_id:
  1 | count=... pred=... actual=... mae=... bias=...
  ...

Holdout bias by minutes_band:
  0-30 | count=... pred=... actual=... mae=... bias=...
  ...

✅ Model saved to: logs/model.bin
✅ Metrics saved to: logs/model.metrics.txt
```

## Using the Model

### Automatic Inference

Once `logs/model.bin` exists, the pipeline automatically uses it:

```bash
python scripts/run_once.py
```

The `run_ml_inference` task will:
1. Check if `model.bin` exists
2. Load the trained model
3. Apply shrinkage (and optional calibration) using saved metadata
4. Generate predictions for all players
5. Output predictions to `recommended_squad` table

If the active artefact fails the local publication gates saved in model metadata, inference now refuses to emit forward predictions. This is intentional: the pipeline keeps the model available for retrospective evaluation, but blocks forward squad publication until the ranking and regret gates pass.

### Fallback Behaviour

If `logs/model.bin` doesn't exist, the pipeline falls back to heuristic predictions:
```python
predicted_points = 0.7 * rolling_avg + 1.5 * z_score + 2.0
```

The missing-model heuristic is not used when an existing model artefact is explicitly marked invalid for forward publication. In that case the pipeline returns no predictions so a weak model cannot silently continue driving squad output.

## March 2026 Stabilisation Changes

The current recovery pass adds a local-only safety layer around training, replay, and inference:

- target engineering now drops missing shifted targets before numeric fill and fails fast if invalid target rows survive
- duplicate `player_id`/`gameweek_id` rows are collapsed before target shift so double-gameweek fixture rows do not poison the target
- `form` has been removed from the active feature set for the stabilisation cycle
- calibration selection now treats `none` as the baseline and rejects harmful calibrated variants
- point-in-time backfill and live inference now share the same post-processing contract
- weekly backtest output now includes oracle-versus-selected XI evidence needed for re-enable decisions
- forward publication is blocked unless the rebuilt artefact clears the local ranking, regret, collapse, and calibration gates

Validated locally after the recovery pass:

- local-only training completed from `pipeline/data/training/latest.parquet`
- test suite passed for ML helpers, calibration/reporting, PIT backfill, and gate enforcement
- the rebuilt artefact selected `none` for calibration and removed prediction collapse
- the rebuilt artefact still failed the strict re-enable bar on ranking, so `forward_publish_ready=false` remains the correct state

## Model Features

### Input Features (45 total)

**Current Gameweek Stats:**
- `total_points`, `minutes_played`, `goals_scored`, `assists`
- `clean_sheets`, `goals_conceded`, `bonus`, `ict_index`
- `influence`, `creativity`, `threat`

**Expected Metrics:**
- `expected_goals`, `expected_assists`, `expected_goals_conceded`
- `expected_goal_involvements`

**Player Rolling Stats:**
- `three_week_players_roll_avg_points`
- `five_week_players_roll_avg_points`
- `total_games_played`

**Team Context:**
- `team_roll_avg_goals_scored`, `team_roll_avg_xg`
- `team_roll_avg_clean_sheets`, `team_roll_avg_wins_pct`

**Opponent Context:**
- `opponent_roll_avg_goals_conceded`, `opponent_roll_avg_xg`
- `opponent_defence_strength`, `team_attack_strength`

**League Position:**
- `team_position`, `opponent_team_position`, `team_position_difference`

**Player Metadata:**
- `now_cost`

**Engineered Features:**
- Z-scores for `total_points`, `minutes_played`, `ict_index`
- Position encoding: `is_gk`, `is_def`, `is_mid`, `is_fwd`
- Home advantage: `is_home`

### Target Variable

`target_next_gw_points` - The actual total points scored by the player in the **next gameweek**.

## Model Architecture

- **Algorithm:** XGBoost Regressor
- **Objective:** Quantile regression (upper quantile)
- **Evaluation Metric:** Mean Absolute Error (MAE)
- **Cross-Validation:** Time Series Split (5 folds)
- **Hyperparameters:**
  - `n_estimators=200`
  - `max_depth=5`
  - `learning_rate=0.1`
  - `subsample=0.8`
  - `colsample_bytree=0.8`
  - `quantile_alpha=0.8`
  - `reg_alpha=0.1`
  - `reg_lambda=1.0`

### Bias Control

- **Shrinkage toward league mean** is applied at inference using metadata saved with the model.
- **Calibration** is optional and only applied if coefficients are saved.
 
Current defaults: `shrinkage_alpha=0.3`, `calibration=None` (disabled).

## Performance Expectations

### Realistic Benchmarks

- **MAE:** 2.0 - 2.5 points (good for FPL prediction)
- **RMSE:** 3.0 - 4.0 points
- **R²:** 0.25 - 0.40 (player points are inherently noisy)

### Context

FPL points prediction is difficult because:
- High variance (0-20+ points per gameweek)
- Unpredictable events (red cards, injuries, bonus points)
- Rotation and manager decisions
- Match-specific factors

An MAE of 2.5 points means the model is typically within 1-2 points of actual performance, which is **excellent** for FPL.

## Retraining

Retrain the model:
1. **Weekly** - After each gameweek to incorporate new data
2. **Monthly** - For major meta shifts (injuries, form changes)
3. **Ad-hoc** - When model performance degrades

```bash
# Quick retrain
python scripts/train_model.py

# Then run pipeline to generate new predictions
python scripts/run_once.py
```

## Manual Gameweek Trust Workflow

Gameweek trust is managed manually in `config/domain_rules.yaml` under `gameweek_quality`:

1. Keep polluted weeks in `excluded_gameweeks` and `backfilled_but_untrusted_gameweeks`.
2. Rebuild with point-in-time backfill:
   ```bash
   python scripts/backfill_recommended_squad_pit.py --gameweeks 27 28
   ```
3. Review trust metadata in `recommended_squad` (`backfill_trusted`, `backfill_validation_status`, `backfill_validation_details`).
4. Manually update `config/domain_rules.yaml`:
   - remove validated weeks from `excluded_gameweeks`
   - remove validated weeks from `backfilled_but_untrusted_gameweeks`
   - add validated weeks to `trusted_backfilled_gameweeks`
5. Re-run training and autonomous evaluation after the manual rule update.

No automatic trust promotion is performed.

## Weekly Baseline Artifacts

Promotion decisions should use pipeline artifacts, not dashboard-only review:

- `logs/model_weekly_report.json` from `scripts/train_model.py`
- `logs/autonomous/<run_id>.weekly_report.json` from autonomous evaluation
- autonomous evidence bundle with per-week/per-position metrics and calibration report

`model_weekly_report.json` now includes prediction collapse and instability (`bias_flip_weeks`) signals.

## Monitoring

Check `model.metrics.txt` after training:
- Monitor train and holdout MAE trend over time
- Track holdout bias (actual - predicted)
- Review feature importance for drift

Compare predictions to actuals in dbt:
```sql
SELECT 
    AVG(absolute_error) AS mae,
    AVG(error_bias) AS bias
FROM fct_model_analysis
WHERE recommended_at >= CURRENT_DATE - 30
```

## Troubleshooting

### Issue: "Fetched 0 training samples"
**Solution:** Run `dbt build` to create `fct_ml_player_features`

### Issue: "Missing X features"
**Solution:** Update `fct_ml_player_features.sql` to include missing columns

### Issue: MAE > 4.0
**Solution:** Check data quality, consider feature engineering, tune hyperparameters

### Issue: Model file not found during inference
**Solution:** Run `python train_model.py` first, or let pipeline use fallback

## Files

- `scripts/train_model.py` - Training script
- `logs/model.bin` - Serialised XGBoost model (gitignored)
- `logs/model.metrics.txt` - Model performance metrics
- `tasks/ml_tasks.py` - Inference functions
- `scripts/run_once.py` - Full pipeline (ingestion → dbt → ML → optimisation)
