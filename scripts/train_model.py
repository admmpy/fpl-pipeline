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
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.snowflake_client import get_snowflake_connection
from config import get_snowflake_config


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
        DataFrame with engineered features
    """
    print("\nEngineering features...")
    
    df = df.copy()
    
    # 1. Z-Score Normalization (relative to gameweek)
    features_to_scale = ['total_points', 'minutes_played', 'ict_index']
    
    for feature in features_to_scale:
        if feature in df.columns:
            gw_mean = df.groupby('gameweek_id')[feature].transform('mean')
            gw_std = df.groupby('gameweek_id')[feature].transform('std')
            df[f'{feature}_z_score'] = (df[feature] - gw_mean) / gw_std.replace(0, np.nan)
            df[f'{feature}_z_score'] = df[f'{feature}_z_score'].fillna(0)
    
    # 2. Create target variable (next gameweek points)
    # Sort by player and gameweek, then shift total_points forward
    df = df.sort_values(['player_id', 'gameweek_id'])
    df['target_next_gw_points'] = df.groupby('player_id')['total_points'].shift(-1)
    
    # 3. Position encoding (one-hot)
    df['is_gk'] = (df['position_id'] == 1).astype(int)
    df['is_def'] = (df['position_id'] == 2).astype(int)
    df['is_mid'] = (df['position_id'] == 3).astype(int)
    df['is_fwd'] = (df['position_id'] == 4).astype(int)
    
    # 4. Home advantage flag
    df['is_home'] = df['was_home'].astype(int)
    
    # 5. Fill missing values with sensible defaults
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # 6. Drop rows without target (last GW for each player)
    df = df.dropna(subset=['target_next_gw_points'])
    
    print(f"Feature engineering complete. {len(df)} samples remain after creating target.")
    
    return df


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
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        objective='reg:squarederror',
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
    
    # Save model
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\n✅ Model saved to: {output_path}")
    
    # Save metrics
    metrics_path = Path(output_path).with_suffix('.metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("FPL XGBoost Model Metrics\n")
        f.write("="*60 + "\n\n")
        f.write(f"MAE:  {metrics['mae']:.3f} points\n")
        f.write(f"RMSE: {metrics['rmse']:.3f} points\n")
        f.write(f"R²:   {metrics['r2']:.3f}\n\n")
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
    
    # 2. Engineer features
    df = engineer_features(df)
    
    # 3. Select features and target
    features = select_features()
    
    # Check which features exist in the data
    available_features = [f for f in features if f in df.columns]
    missing_features = [f for f in features if f not in df.columns]
    
    if missing_features:
        print(f"\n⚠️  Warning: {len(missing_features)} features not found in data:")
        for f in missing_features[:5]:  # Show first 5
            print(f"  - {f}")
        if len(missing_features) > 5:
            print(f"  ... and {len(missing_features) - 5} more")
    
    X = df[available_features]
    y = df['target_next_gw_points']
    
    print(f"\nTraining data shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target range: [{y.min():.1f}, {y.max():.1f}] points")
    print(f"Target mean: {y.mean():.2f} points")
    
    # 4. Train model
    model = train_xgboost_model(X, y)
    
    # 5. Evaluate
    metrics = evaluate_model(model, X, y)
    
    # 6. Save
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
