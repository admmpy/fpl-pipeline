"""
Train XGBoost model for FPL player points prediction.

This script:
1. Fetches historical player features from Snowflake
2. Engineers features and prepares training data
3. Trains an XGBoost regressor with cross-validation
4. Evaluates model performance
5. Saves the trained model to disk
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from pathlib import Path
from typing import Dict, Tuple
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.snowflake_client import get_snowflake_connection
from config import get_snowflake_config

HOLDOUT_GAMEWEEKS = 3
QUANTILE_ALPHA = 0.8
SHRINKAGE_ALPHA = 0.3
REG_ALPHA = 0.1
REG_LAMBDA = 1.0
MAX_DEPTH = 5
ENABLE_CALIBRATION = False
FEATURES_TO_SCALE = ['total_points', 'minutes_played', 'ict_index']


def fetch_training_data() -> pd.DataFrame:
    """
    Fetch historical player features from Snowflake.
    
    Returns:
        DataFrame with player features and target variable (total_points)
    """
    print("Fetching training data from Snowflake...")
    
    if get_snowflake_config() is None:
        raise ValueError("Snowflake configuration not found. Cannot fetch training data.")
    
    query = """
    SELECT 
        f.* EXCLUDE (now_cost),
        COALESCE(f.now_cost, p.current_value) AS now_cost,
        p.position_id
    FROM fct_ml_player_features f
    LEFT JOIN dim_players p ON f.player_id = p.player_id
    WHERE f.gameweek_id > 3  -- Skip first 3 GWs for rolling stat stability
        AND f.minutes_played > 0  -- Only players who actually played
    """
    
    with get_snowflake_connection() as conn:
        df = pd.read_sql(query, conn)
    
    # Standardise column names to lowercase
    df.columns = [c.lower() for c in df.columns]
    
    print(f"Fetched {len(df)} training samples across {df['gameweek_id'].nunique()} gameweeks")
    print(f"Players: {df['player_id'].nunique()}, Date range: GW{df['gameweek_id'].min()}-{df['gameweek_id'].max()}")
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer additional features and handle missing values.
    
    Args:
        df: Raw feature DataFrame
        
    Returns:
        DataFrame with engineered features (excluding z-scores)
    """
    print("\nEngineering features...")
    
    df = df.copy()
    
    # 1. Create target variables (next gameweek points and gameweek id)
    # Sort by player and gameweek, then shift total_points forward
    df = df.sort_values(['player_id', 'gameweek_id'])
    df['target_next_gw_points'] = df.groupby('player_id')['total_points'].shift(-1)
    df['target_gameweek_id'] = df.groupby('player_id')['gameweek_id'].shift(-1)
    
    # 2. Position encoding (one-hot)
    df['is_gk'] = (df['position_id'] == 1).astype(int)
    df['is_def'] = (df['position_id'] == 2).astype(int)
    df['is_mid'] = (df['position_id'] == 3).astype(int)
    df['is_fwd'] = (df['position_id'] == 4).astype(int)
    
    # 3. Home advantage flag
    df['is_home'] = df['was_home'].astype(int)
    
    # 4. Fill missing values with sensible defaults
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # 5. Drop rows without target (last GW for each player)
    df = df.dropna(subset=['target_next_gw_points'])
    
    print(f"Feature engineering complete. {len(df)} samples remain after creating target.")
    
    return df


def compute_gw_stats(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Compute per-gameweek and overall statistics for z-score normalisation.
    
    Args:
        df: Training DataFrame only (no holdout rows)
        
    Returns:
        Tuple of (per-gameweek stats DataFrame, overall stats dict)
    """
    gw_stats = df.groupby('gameweek_id')[FEATURES_TO_SCALE].agg(['mean', 'std']).reset_index()
    gw_stats.columns = [
        'gameweek_id',
        *[f'{feature}_gw_{stat}' for feature in FEATURES_TO_SCALE for stat in ['mean', 'std']]
    ]
    
    overall_stats: Dict[str, Dict[str, float]] = {}
    for feature in FEATURES_TO_SCALE:
        overall_stats[feature] = {
            'mean': float(df[feature].mean()),
            'std': float(df[feature].std() or 1.0),
        }
    
    return gw_stats, overall_stats


def add_z_scores(
    df: pd.DataFrame,
    gw_stats: pd.DataFrame,
    overall_stats: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """
    Add z-score features using training-only statistics.
    
    Args:
        df: DataFrame to transform
        gw_stats: Per-gameweek stats computed from training data
        overall_stats: Overall stats computed from training data
        
    Returns:
        DataFrame with z-score columns added
    """
    df = df.merge(gw_stats, on='gameweek_id', how='left')
    
    for feature in FEATURES_TO_SCALE:
        mean_col = f'{feature}_gw_mean'
        std_col = f'{feature}_gw_std'
        df[mean_col] = df[mean_col].fillna(overall_stats[feature]['mean'])
        df[std_col] = df[std_col].fillna(overall_stats[feature]['std'])
        df[f'{feature}_z_score'] = (df[feature] - df[mean_col]) / df[std_col].replace(0, np.nan)
        df[f'{feature}_z_score'] = df[f'{feature}_z_score'].fillna(0)
    
    df = df.drop(columns=[f'{feature}_gw_{stat}' for feature in FEATURES_TO_SCALE for stat in ['mean', 'std']])
    return df


def split_train_holdout(df: pd.DataFrame, holdout_gameweeks: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and holdout sets based on target gameweek.
    
    Args:
        df: Feature DataFrame with target_gameweek_id
        holdout_gameweeks: Number of gameweeks to hold out
        
    Returns:
        Tuple of (train_df, holdout_df)
    """
    target_gameweeks = sorted(df['target_gameweek_id'].dropna().unique())
    
    if len(target_gameweeks) <= holdout_gameweeks:
        raise ValueError("Not enough gameweeks to create a holdout set.")
    
    holdout_gws = set(target_gameweeks[-holdout_gameweeks:])
    train_df = df[~df['target_gameweek_id'].isin(holdout_gws)].copy()
    holdout_df = df[df['target_gameweek_id'].isin(holdout_gws)].copy()
    
    return train_df, holdout_df


def apply_shrinkage(predictions: np.ndarray, league_mean: float, alpha: float) -> np.ndarray:
    """
    Shrink predictions toward the league mean.
    
    Args:
        predictions: Raw model predictions
        league_mean: Mean of target on training set
        alpha: Shrinkage weight
        
    Returns:
        Shrunk predictions
    """
    return (1 - alpha) * predictions + alpha * league_mean


def fit_calibration(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Fit a linear calibration model: actual ≈ a * predicted + b.
    """
    if np.std(y_pred) == 0:
        return 1.0, 0.0
    
    a, b = np.polyfit(y_pred, y_true, 1)
    return float(a), float(b)


def apply_calibration(predictions: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Apply linear calibration to predictions.
    """
    return a * predictions + b


def print_group_bias(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, group_col: str, label: str) -> None:
    """
    Print bias metrics grouped by a column.
    """
    report_df = df.copy()
    report_df['actual_points'] = y_true
    report_df['predicted_points'] = y_pred
    report_df['absolute_error'] = np.abs(y_true - y_pred)
    report_df['error_bias'] = y_true - y_pred
    
    summary = report_df.groupby(group_col).agg(
        player_count=('player_id', 'count'),
        avg_predicted_points=('predicted_points', 'mean'),
        avg_actual_points=('actual_points', 'mean'),
        mean_absolute_error=('absolute_error', 'mean'),
        mean_error_bias=('error_bias', 'mean')
    ).reset_index()
    
    print(f"\n{label} bias by {group_col}:")
    for _, row in summary.iterrows():
        print(
            f"  {row[group_col]} | count={int(row['player_count'])} "
            f"pred={row['avg_predicted_points']:.2f} "
            f"actual={row['avg_actual_points']:.2f} "
            f"mae={row['mean_absolute_error']:.2f} "
            f"bias={row['mean_error_bias']:.2f}"
        )


def select_features() -> list:
    """
    Define the feature set for training.
    
    Returns:
        List of feature column names
    """
    features = [
        # Current gameweek performance
        'total_points',
        'minutes_played',
        'goals_scored',
        'assists',
        'clean_sheets',
        'goals_conceded',
        'bonus',
        'ict_index',
        'influence',
        'creativity',
        'threat',
        
        # Expected metrics
        'expected_goals',
        'expected_assists',
        'expected_goals_conceded',
        'expected_goal_involvements',
        
        # Player rolling stats
        'three_week_players_roll_avg_points',
        'five_week_players_roll_avg_points',
        'total_games_played',
        
        # Team rolling stats
        'team_roll_avg_goals_scored',
        'team_roll_avg_xg',
        'team_roll_avg_clean_sheets',
        'team_roll_avg_wins_pct',
        
        # Opponent rolling stats
        'opponent_roll_avg_goals_conceded',
        'opponent_roll_avg_xg',
        
        # Strength metrics
        'opponent_defence_strength',
        'team_attack_strength',
        
        # Position context
        'team_position',
        'opponent_team_position',
        'team_position_difference',
        
        # Player metadata
        'form',
        'now_cost',
        
        # Z-scores
        'total_points_z_score',
        'minutes_played_z_score',
        'ict_index_z_score',
        
        # Position encoding
        'is_gk',
        'is_def',
        'is_mid',
        'is_fwd',
        
        # Home advantage
        'is_home',
    ]
    
    return features


def train_xgboost_model(X: pd.DataFrame, y: pd.Series) -> xgb.XGBRegressor:
    """
    Train XGBoost model with optimal hyperparameters.
    
    Args:
        X: Feature matrix
        y: Target variable
        
    Returns:
        Trained XGBoost model
    """
    print("\nTraining XGBoost model...")
    
    # Define model with sensible hyperparameters
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=MAX_DEPTH,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        objective='reg:quantileerror',
        quantile_alpha=QUANTILE_ALPHA,
        reg_alpha=REG_ALPHA,
        reg_lambda=REG_LAMBDA,
        eval_metric='mae'
    )
    
    # Time series cross-validation (respects temporal order)
    tscv = TimeSeriesSplit(n_splits=5)
    
    print("Running time series cross-validation...")
    cv_scores = cross_val_score(
        model, X, y, 
        cv=tscv, 
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    
    print(f"Cross-validation MAE: {-cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    # Train final model on all data
    print("Training final model on full dataset...")
    model.fit(X, y)
    
    return model


def evaluate_model(model: xgb.XGBRegressor, X: pd.DataFrame, y: pd.Series):
    """
    Evaluate model performance on training data.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: True target values
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Generate predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    print(f"\nOverall Metrics:")
    print(f"  MAE:  {mae:.3f} points")
    print(f"  RMSE: {rmse:.3f} points")
    print(f"  R²:   {r2:.3f}")
    
    # Feature importance
    print(f"\nTop 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:40s} {row['importance']:.4f}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'feature_importance': feature_importance
    }


def save_model(model: xgb.XGBRegressor, metrics: dict, output_path: str = "logs/model.bin"):
    """
    Save trained model and metadata to disk.
    
    Args:
        model: Trained model
        metrics: Dictionary of evaluation metrics
        output_path: Path to save model
    """
    model_dir = Path(output_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model and metadata together
    with open(output_path, 'wb') as f:
        pickle.dump(
            {
                'model': model,
                'metadata': metrics.get('metadata', {})
            },
            f
        )
    
    print(f"\n✅ Model saved to: {output_path}")
    
    # Save metrics
    metrics_path = Path(output_path).with_suffix('.metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("FPL XGBoost Model Metrics\n")
        f.write("="*60 + "\n\n")
        f.write(f"Train MAE:  {metrics['mae']:.3f} points\n")
        f.write(f"Train RMSE: {metrics['rmse']:.3f} points\n")
        f.write(f"Train R²:   {metrics['r2']:.3f}\n\n")
        if 'holdout_mae' in metrics:
            f.write(f"Holdout MAE:  {metrics['holdout_mae']:.3f} points\n")
            f.write(f"Holdout RMSE: {metrics['holdout_rmse']:.3f} points\n")
            f.write(f"Holdout Bias: {metrics['holdout_bias']:.3f} points\n\n")
        f.write("Top 10 Features:\n")
        for idx, row in metrics['feature_importance'].head(10).iterrows():
            f.write(f"  {row['feature']:40s} {row['importance']:.4f}\n")
    
    print(f"✅ Metrics saved to: {metrics_path}")


def main():
    """
    Main training pipeline.
    """
    print("="*60)
    print("FPL XGBOOST MODEL TRAINING")
    print("="*60)
    
    # 1. Fetch data
    df = fetch_training_data()
    
    # 2. Engineer base features
    df = engineer_features(df)
    
    # 3. Train/holdout split by target gameweek
    train_df, holdout_df = split_train_holdout(df, HOLDOUT_GAMEWEEKS)
    
    # 4. Z-score normalisation using training-only stats
    gw_stats, overall_stats = compute_gw_stats(train_df)
    train_df = add_z_scores(train_df, gw_stats, overall_stats)
    holdout_df = add_z_scores(holdout_df, gw_stats, overall_stats)
    
    # 5. Select features and target
    features = select_features()
    
    # Check which features exist in the data
    available_features = [f for f in features if f in train_df.columns]
    missing_features = [f for f in features if f not in train_df.columns]
    
    if missing_features:
        print(f"\n⚠️  Warning: {len(missing_features)} features not found in data:")
        for f in missing_features[:5]:  # Show first 5
            print(f"  - {f}")
        if len(missing_features) > 5:
            print(f"  ... and {len(missing_features) - 5} more")
    
    if missing_features:
        for feature in missing_features:
            train_df[feature] = 0
            holdout_df[feature] = 0
        available_features = features
    
    X_train = train_df[available_features]
    y_train = train_df['target_next_gw_points']
    
    X_holdout = holdout_df[available_features]
    y_holdout = holdout_df['target_next_gw_points']
    
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Training target shape: {y_train.shape}")
    print(f"Training target range: [{y_train.min():.1f}, {y_train.max():.1f}] points")
    print(f"Training target mean: {y_train.mean():.2f} points")
    
    # 6. Train model
    model = train_xgboost_model(X_train, y_train)
    
    # 7. Evaluate on training data
    metrics = evaluate_model(model, X_train, y_train)
    
    # 8. Holdout evaluation with shrinkage and optional calibration
    league_mean = float(y_train.mean())
    holdout_pred = model.predict(X_holdout)
    holdout_pred = apply_shrinkage(holdout_pred, league_mean, SHRINKAGE_ALPHA)
    
    calibration = None
    if ENABLE_CALIBRATION:
        a, b = fit_calibration(y_holdout.to_numpy(), holdout_pred)
        holdout_pred = apply_calibration(holdout_pred, a, b)
        calibration = {'a': a, 'b': b}
    
    holdout_pred = np.maximum(holdout_pred, 0)
    
    holdout_mae = mean_absolute_error(y_holdout, holdout_pred)
    holdout_rmse = np.sqrt(mean_squared_error(y_holdout, holdout_pred))
    holdout_bias = float((y_holdout - holdout_pred).mean())
    
    print("\n" + "="*60)
    print("HOLDOUT EVALUATION")
    print("="*60)
    print(f"\nHoldout Metrics:")
    print(f"  MAE:  {holdout_mae:.3f} points")
    print(f"  RMSE: {holdout_rmse:.3f} points")
    print(f"  Bias: {holdout_bias:.3f} points (actual - predicted)")
    
    holdout_context = holdout_df[['player_id', 'position_id', 'minutes_played']].copy()
    holdout_context['minutes_band'] = pd.cut(
        holdout_context['minutes_played'],
        bins=[-1, 30, 60, 1_000_000],
        labels=['0-30', '31-60', '61-90']
    )
    
    print_group_bias(holdout_context, y_holdout.to_numpy(), holdout_pred, 'position_id', "Holdout")
    print_group_bias(holdout_context, y_holdout.to_numpy(), holdout_pred, 'minutes_band', "Holdout")
    
    metrics['holdout_mae'] = holdout_mae
    metrics['holdout_rmse'] = holdout_rmse
    metrics['holdout_bias'] = holdout_bias
    metrics['metadata'] = {
        'league_mean': league_mean,
        'shrinkage_alpha': SHRINKAGE_ALPHA,
        'quantile_alpha': QUANTILE_ALPHA,
        'reg_alpha': REG_ALPHA,
        'reg_lambda': REG_LAMBDA,
        'max_depth': MAX_DEPTH,
        'holdout_gameweeks': HOLDOUT_GAMEWEEKS,
        'calibration': calibration
    }
    
    # 9. Save
    save_model(model, metrics, output_path="logs/model.bin")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Review model metrics above")
    print("  2. Run pipeline: python run_once.py")
    print("  3. Check predictions in Snowflake: SELECT * FROM recommended_squad")


if __name__ == "__main__":
    main()
