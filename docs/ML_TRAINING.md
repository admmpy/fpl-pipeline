# XGBoost Model Training for FPL Predictions

This document explains how to train and use the XGBoost model for Fantasy Premier League player points prediction.

## Overview

The ML pipeline:
1. Fetches historical player features from `fct_ml_player_features` in Snowflake
2. Engineers temporal features and creates training targets
3. Trains an XGBoost regressor with time series cross-validation
4. Saves the model to `model.bin` for inference

## Training the Model

### Prerequisites

1. **Data Requirements:**
   - Snowflake connection configured in `.env`
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
Target range: [0.0, 24.0] points
Target mean: 3.45 points

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
3. Generate predictions for all players
4. Output predictions to `recommended_squad` table

### Fallback Behaviour

If `logs/model.bin` doesn't exist, the pipeline falls back to heuristic predictions:
```python
predicted_points = 0.7 * rolling_avg + 1.5 * z_score + 2.0
```

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
- `form`, `now_cost`

**Engineered Features:**
- Z-scores for `total_points`, `minutes_played`, `ict_index`
- Position encoding: `is_gk`, `is_def`, `is_mid`, `is_fwd`
- Home advantage: `is_home`

### Target Variable

`target_next_gw_points` - The actual total points scored by the player in the **next gameweek**.

## Model Architecture

- **Algorithm:** XGBoost Regressor
- **Objective:** Squared error regression
- **Evaluation Metric:** Mean Absolute Error (MAE)
- **Cross-Validation:** Time Series Split (5 folds)
- **Hyperparameters:**
  - `n_estimators=200`
  - `max_depth=6`
  - `learning_rate=0.1`
  - `subsample=0.8`
  - `colsample_bytree=0.8`

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

## Monitoring

Check `model.metrics.txt` after training:
- Monitor MAE trend over time
- Compare to baseline heuristic
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
