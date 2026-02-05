"""
Tasks for ML inference and data preparation.
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import logging
import pickle
from prefect import task, get_run_logger
from prefect.exceptions import MissingContextError
from utils.snowflake_client import get_snowflake_connection
import os

DEFAULT_SHRINKAGE_ALPHA = 0.0


def get_logger():
    """
    Return a Prefect run logger when available, otherwise a standard logger.
    """
    try:
        return get_run_logger()
    except MissingContextError:
        return logging.getLogger(__name__)


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


def apply_global_z_scores(
    df: pd.DataFrame,
    stats: Dict[str, Dict[str, float]],
    features_to_scale: List[str]
) -> pd.DataFrame:
    """
    Apply global z-score normalization using training statistics.
    """
    df = df.copy()
    for feature in features_to_scale:
        if feature in df.columns and feature in stats:
            mean_val = stats[feature].get('mean', 0.0)
            raw_std = stats[feature].get('std', 1.0)
            std_val = raw_std if raw_std > 0 else 1.0
            df[f'{feature}_z_score'] = (df[feature] - mean_val) / std_val
            df[f'{feature}_z_score'] = df[f'{feature}_z_score'].fillna(0)
    return df

@task
def fetch_training_data(table_name: str = "fct_ml_player_features") -> pd.DataFrame:
    """
    Fetch the denormalized feature table from Snowflake for training/inference.
    """
    logger = get_logger()
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
    Prepare inference data and handle the Cold Start problem.
    
    Logic:
    1. Drop GW 1-3 to ensure rolling averages are more stable.
    """
    logger = get_logger()

    # Cold Start: Drop GW 1-3
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
    logger = get_logger()
    
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
        try:
            with open(model_path, 'rb') as f:
                model_payload = pickle.load(f)
        except Exception as exc:
            logger.error(f"Failed to load model from {model_path}: {exc}")
            return []

        if isinstance(model_payload, dict) and 'model' in model_payload:
            model = model_payload['model']
            metadata = model_payload.get('metadata', {})
        else:
            model = model_payload
            metadata = {}

        if not hasattr(model, 'predict'):
            logger.error("Loaded model object does not have a predict method")
            return []
        
        # Define default feature set (overridden by model metadata if present)
        default_feature_cols = [
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
            'is_gk', 'is_def', 'is_mid', 'is_fwd'
        ]
        feature_cols = metadata.get('feature_cols') or default_feature_cols
        
        # Engineer additional features needed for inference
        # Position encoding
        df['is_gk'] = (df['position_id'] == 1).astype(int)
        df['is_def'] = (df['position_id'] == 2).astype(int)
        df['is_mid'] = (df['position_id'] == 3).astype(int)
        df['is_fwd'] = (df['position_id'] == 4).astype(int)

        # Apply global z-scores using training stats, if available
        zscore_stats = metadata.get('zscore_stats', {})
        if zscore_stats:
            df = apply_global_z_scores(df, zscore_stats, ['total_points', 'minutes_played', 'ict_index'])
        
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
        try:
            predictions_array = model.predict(X_inference)
        except Exception as exc:
            logger.error(f"model.predict() failed: {exc}")
            return []
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

        # Apply per-position caps if available
        position_caps = metadata.get('position_caps', {})
        if position_caps:
            cap_values = latest_stats['position_id'].map(position_caps).fillna(np.inf).to_numpy()
            before_clip = latest_stats['expected_points_next_gw'].to_numpy()
            latest_stats['expected_points_next_gw'] = np.minimum(before_clip, cap_values)
            clipped = (before_clip > cap_values).sum()
            if clipped:
                logger.info(f"Applied position caps: {clipped} predictions clipped")
        
        clipped_stats = latest_stats['expected_points_next_gw']
        logger.info(
            "Model predictions: "
            f"min={clipped_stats.min():.2f}, "
            f"max={clipped_stats.max():.2f}, "
            f"mean={clipped_stats.mean():.2f}"
        )
        
        df = latest_stats
    
    # Get only the latest available stats for each player to predict the next GW
    latest_stats = df.sort_values('gameweek_id').groupby('player_id').tail(1)

    if latest_stats.empty:
        logger.warning("DataFrame is empty after filtering — returning no predictions")
        return []

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
