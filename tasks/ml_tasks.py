"""
Tasks for ML inference and data preparation.
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from prefect import task, get_run_logger
from utils.snowflake_client import get_snowflake_connection
import os

DEFAULT_SHRINKAGE_ALPHA = 0.3


def apply_shrinkage(predictions: np.ndarray, league_mean: float, alpha: float) -> np.ndarray:
    """
    Shrink predictions toward the league mean.
    """
    return (1 - alpha) * predictions + alpha * league_mean


def apply_calibration(predictions: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Apply linear calibration to predictions.
    """
    return a * predictions + b

@task
def fetch_training_data(table_name: str = "fct_ml_player_features") -> pd.DataFrame:
    """
    Fetch the denormalized feature table from Snowflake for training/inference.
    """
    logger = get_run_logger()
    logger.info(f"Fetching features from {table_name}...")
    
    query = f"""
    SELECT 
        f.* EXCLUDE (now_cost),
        COALESCE(f.now_cost, p.current_value) AS now_cost,
        p.position_id
    FROM {table_name} f
    LEFT JOIN dim_players p ON f.player_id = p.player_id
    """
    
    with get_snowflake_connection() as conn:
        df = pd.read_sql(query, conn)
    
    # Standardize column names to lowercase (Snowflake returns UPPERCASE by default)
    df.columns = [c.lower() for c in df.columns]
        
    logger.info(f"Fetched {len(df)} rows.")
    return df

@task
def prepare_inference_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Z-score normalization and handle the Cold Start problem.
    
    Logic:
    1. Calculate Z-scores for key metrics per Gameweek (to handle Feature Drift).
    2. Filter for players with minutes_played > 0 to avoid selection bias.
    3. Drop GW 1-3 to ensure rolling averages are more stable.
    """
    logger = get_run_logger()
    
    # 1. Z-Score Normalization (Relative Form)
    # We do this before dropping GW 1-3 to use the full context for normalization
    # but we only use active players to calculate the mean/std
    features_to_scale = ['total_points', 'minutes_played', 'ict_index']
    
    for feature in features_to_scale:
        if feature in df.columns:
            # Only include players with minutes > 0 to avoid selection bias in the baseline
            active_players = df[df['minutes_played'] > 0]
            
            gw_stats = active_players.groupby('gameweek_id')[feature].agg(['mean', 'std']).reset_index()
            gw_stats.columns = ['gameweek_id', f'{feature}_gw_mean', f'{feature}_gw_std']
            
            df = df.merge(gw_stats, on='gameweek_id', how='left')
            
            # Calculate Z-score (standardizing relative to the gameweek)
            df[f'{feature}_z_score'] = (df[feature] - df[f'{feature}_gw_mean']) / df[f'{feature}_gw_std'].replace(0, np.nan)
            df[f'{feature}_z_score'] = df[f'{feature}_z_score'].fillna(0)
            
            # Drop the helper columns
            df = df.drop(columns=[f'{feature}_gw_mean', f'{feature}_gw_std'])

    # 2. Cold Start: Drop GW 1-3
    logger.info("Dropping Gameweeks 1-3 to handle Cold Start problem.")
    df_clean = df[df['gameweek_id'] > 3].copy()
    
    logger.info(f"Data preparation complete. {len(df_clean)} rows remaining.")
    return df_clean

@task
def run_ml_inference(df: pd.DataFrame, model_path: str = "logs/model.bin") -> List[Dict[str, Any]]:
    """
    Run ML inference using the trained XGBoost model.
    
    Args:
        df: Prepared DataFrame with features.
        model_path: Path to the pickled model artifact.
        
    Returns:
        List of dictionaries with player predictions for the next gameweek.
    """
    logger = get_run_logger()
    
    # Identify the target gameweek (max in data + 1)
    current_gw = df['gameweek_id'].max()
    target_gw = current_gw + 1
    
    logger.info(f"Running ML inference for GW {target_gw}...")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        logger.warning(f"Model file not found at {model_path}")
        logger.warning("Falling back to heuristic predictions. Run 'python train_model.py' to create model.")
        
        # Fallback: heuristic prediction
        df['expected_points_next_gw'] = (
            df['three_week_players_roll_avg_points'].fillna(2.0) * 0.7 + 
            df['total_points_z_score'].fillna(0) * 1.5 + 
            2.0
        ).clip(lower=0)
    else:
        # Load trained model
        logger.info(f"Loading model from {model_path}...")
        import pickle
        with open(model_path, 'rb') as f:
            model_payload = pickle.load(f)
        
        if isinstance(model_payload, dict) and 'model' in model_payload:
            model = model_payload['model']
            metadata = model_payload.get('metadata', {})
        else:
            model = model_payload
            metadata = {}
        
        # Define feature set (must match training)
        feature_cols = [
            'total_points', 'minutes_played', 'goals_scored', 'assists', 
            'clean_sheets', 'goals_conceded', 'bonus', 'ict_index', 
            'influence', 'creativity', 'threat',
            'expected_goals', 'expected_assists', 'expected_goals_conceded', 
            'expected_goal_involvements',
            'three_week_players_roll_avg_points', 'five_week_players_roll_avg_points', 
            'total_games_played',
            'team_roll_avg_goals_scored', 'team_roll_avg_xg', 
            'team_roll_avg_clean_sheets', 'team_roll_avg_wins_pct',
            'opponent_roll_avg_goals_conceded', 'opponent_roll_avg_xg',
            'opponent_defence_strength', 'team_attack_strength',
            'team_position', 'opponent_team_position', 'team_position_difference',
            'form', 'now_cost',
            'total_points_z_score', 'minutes_played_z_score', 'ict_index_z_score',
            'is_gk', 'is_def', 'is_mid', 'is_fwd',
            'is_home'
        ]
        
        # Engineer additional features needed for inference
        # Position encoding
        df['is_gk'] = (df['position_id'] == 1).astype(int)
        df['is_def'] = (df['position_id'] == 2).astype(int)
        df['is_mid'] = (df['position_id'] == 3).astype(int)
        df['is_fwd'] = (df['position_id'] == 4).astype(int)
        
        # Home advantage (assume neutral for future prediction)
        df['is_home'] = 0
        
        # Filter to only features that exist in the dataframe
        available_features = [f for f in feature_cols if f in df.columns]
        missing_features = [f for f in feature_cols if f not in df.columns]
        
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features, filling with zeros: {missing_features[:5]}")
            for feat in missing_features:
                df[feat] = 0
            available_features = feature_cols  # Use all after filling
        
        # Get latest stats for each player
        latest_stats = df.sort_values('gameweek_id').groupby('player_id').tail(1).copy()
        
        # Make predictions
        X_inference = latest_stats[available_features].fillna(0)
        predictions_array = model.predict(X_inference)
        predictions_array = np.maximum(predictions_array, 0)

        league_mean = metadata.get('league_mean')
        shrinkage_alpha = metadata.get('shrinkage_alpha', DEFAULT_SHRINKAGE_ALPHA)
        if league_mean is not None:
            predictions_array = apply_shrinkage(predictions_array, league_mean, shrinkage_alpha)
            logger.info(f"Applied shrinkage: alpha={shrinkage_alpha}, league_mean={league_mean:.2f}")
        
        calibration = metadata.get('calibration')
        if calibration:
            predictions_array = apply_calibration(
                predictions_array,
                calibration.get('a', 1.0),
                calibration.get('b', 0.0)
            )
            logger.info("Applied calibration to predictions.")
        
        # Clip negative predictions to zero after shrinkage/calibration
        predictions_array = np.maximum(predictions_array, 0)

        latest_stats['expected_points_next_gw'] = predictions_array
        
        logger.info(f"Model predictions: min={predictions_array.min():.2f}, max={predictions_array.max():.2f}, mean={predictions_array.mean():.2f}")
        
        df = latest_stats
    
    # Get only the latest available stats for each player to predict the next GW
    latest_stats = df.sort_values('gameweek_id').groupby('player_id').tail(1)
    
    predictions = []
    for _, row in latest_stats.iterrows():
        predictions.append({
            "player_id": int(row['player_id']),
            "web_name": row['web_name'],
            "position_id": int(row['position_id']),
            "team_id": int(row['team_id']),
            "now_cost": float(row['now_cost']),
            "expected_points_next_gw": float(row['expected_points_next_gw']),
            "gameweek_id": int(target_gw)
        })
        
    logger.info(f"Generated predictions for {len(predictions)} players.")
    return predictions
