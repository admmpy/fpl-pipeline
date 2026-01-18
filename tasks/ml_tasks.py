"""
Tasks for ML inference and data preparation.
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from prefect import task, get_run_logger
from utils.snowflake_client import get_snowflake_connection
import os

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
def run_ml_inference(df: pd.DataFrame, model_path: str = "model.bin") -> List[Dict[str, Any]]:
    """
    Run ML inference using the trained XGBoost/LightGBM model.
    
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
    
    # In a real scenario, you would do something like:
    # model = load_model(model_path)
    # df['predicted_points'] = model.predict(df[features])
    
    # SIMULATION: Using a weighted average of rolling points and Z-scores for now
    # until the actual model.bin is available.
    df['expected_points_next_gw'] = (
        df['three_week_players_roll_avg_points'].fillna(2.0) * 0.7 + 
        df['total_points_z_score'].fillna(0) * 1.5 + 
        2.0 # Baseline
    ).clip(lower=0)
    
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
