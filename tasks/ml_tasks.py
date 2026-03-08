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
from utils.local_data import (
    build_training_query,
    emit_selection_log,
    load_training_dataframe,
)
from config import get_snowflake_config
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import train_model

DEFAULT_SHRINKAGE_ALPHA = 0.0
ALLOW_INVALID_FORWARD_PUBLISH = os.getenv("ALLOW_INVALID_FORWARD_PUBLISH", "").lower() in {"1", "true", "yes"}


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


def ensure_z_score_columns(
    df: pd.DataFrame,
    features_to_scale: List[str],
    stats: Optional[Dict[str, Dict[str, float]]] = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Ensure z-score columns exist for requested features, using training stats when available.
    """
    logger = logger or logging.getLogger(__name__)
    df = df.copy()

    if stats:
        df = apply_global_z_scores(df, stats, features_to_scale)
        for feature in features_to_scale:
            z_col = f'{feature}_z_score'
            if z_col not in df.columns:
                if feature in df.columns:
                    logger.warning(f"Missing stats for {feature}; defaulting {z_col} to 0")
                else:
                    logger.warning(f"Missing {feature} and {z_col}; defaulting {z_col} to 0")
                df[z_col] = 0.0
        return df

    for feature in features_to_scale:
        z_col = f'{feature}_z_score'
        if z_col in df.columns:
            continue
        if feature in df.columns:
            mean_val = df[feature].mean()
            std_val = df[feature].std()
            std_val = std_val if std_val and std_val > 0 else 1.0
            df[z_col] = ((df[feature] - mean_val) / std_val).fillna(0)
            logger.warning(f"Missing {z_col}; computed from {feature}")
        else:
            logger.warning(f"Missing {feature} and {z_col}; defaulting {z_col} to 0")
            df[z_col] = 0.0

    return df

@task
def fetch_training_data(
    table_name: str = "fct_ml_player_features",
    source: Optional[str] = None,
    local_path: Optional[str] = None,
    policy: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch the denormalized feature table using local-first policy controls.
    """
    logger = get_logger()
    logger.info(f"Fetching features from {table_name}...")

    def _snowflake_loader(source_table: str) -> pd.DataFrame:
        if get_snowflake_config() is None:
            raise ValueError("Snowflake configuration not found. Cannot fetch training data.")
        query = build_training_query(source_table, apply_training_filters=False)
        with get_snowflake_connection() as conn:
            frame = pd.read_sql(query, conn)
        frame.columns = [str(column).lower() for column in frame.columns]
        return frame

    df, metadata = load_training_dataframe(
        table_name=table_name,
        source=source,
        local_path=local_path,
        policy=policy,
        snowflake_loader=_snowflake_loader,
        logger=logger,
    )
    emit_selection_log(logger, context="tasks.ml_tasks.fetch_training_data", metadata=metadata)

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
        if 'three_week_players_roll_avg_points' in df.columns:
            roll_avg = df['three_week_players_roll_avg_points'].fillna(2.0)
        else:
            logger.warning("Missing three_week_players_roll_avg_points; defaulting to 2.0")
            roll_avg = pd.Series(2.0, index=df.index)

        df = ensure_z_score_columns(
            df,
            ['total_points'],
            stats=None,
            logger=logger
        )
        if 'total_points_z_score' not in df.columns:
            logger.warning("total_points_z_score still missing after ensure_z_score_columns; defaulting to 0")
            df['total_points_z_score'] = 0.0
        total_points_z = df['total_points_z_score'].fillna(0)

        df['expected_points_next_gw'] = (
            roll_avg * 0.7 +
            total_points_z * 1.5 +
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

        if not isinstance(model, dict) and not hasattr(model, 'predict'):
            logger.error("Loaded model object does not expose a supported prediction interface")
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
            'now_cost',
            'total_points_z_score', 'minutes_played_z_score', 'ict_index_z_score',
            'is_gk', 'is_def', 'is_mid', 'is_fwd'
        ]
        feature_cols = metadata.get('feature_cols') or default_feature_cols
        if not bool(metadata.get("forward_publish_ready", True)) and not ALLOW_INVALID_FORWARD_PUBLISH:
            logger.error(
                "Active model is marked invalid for forward publication; refusing to emit predictions. "
                "Reasons: %s",
                metadata.get("forward_publish_reasons", []),
            )
            return []
        
        # Engineer additional features needed for inference.
        df['is_gk'] = (df['position_id'] == 1).astype(int)
        df['is_def'] = (df['position_id'] == 2).astype(int)
        df['is_mid'] = (df['position_id'] == 3).astype(int)
        df['is_fwd'] = (df['position_id'] == 4).astype(int)

        # Recreate minute-band one-hot features expected by tuned models.
        if 'minutes_played' in df.columns:
            minute_band = pd.cut(
                df['minutes_played'].fillna(0),
                bins=[-1, 30, 60, float('inf')],
                labels=['0_30', '31_60', '61_90']
            )
            for label in ['0_30', '31_60', '61_90']:
                col = f'minutes_band_{label}'
                if col not in df.columns:
                    df[col] = (minute_band == label).astype(int)
        df = train_model.engineer_upside_features(df)

        # Apply global z-scores using training stats, if available
        zscore_features = ['total_points', 'minutes_played', 'ict_index']
        zscore_stats = metadata.get('zscore_stats', {})
        df = ensure_z_score_columns(
            df,
            zscore_features,
            stats=zscore_stats if zscore_stats else None,
            logger=logger
        )
        
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
        try:
            predictions_array = train_model.predict_prediction_bundle(
                model,
                latest_stats,
                available_features,
                use_log_target=bool(metadata.get("use_log_target", False)),
            )
        except Exception as exc:
            logger.error(f"model.predict() failed: {exc}")
            return []

        league_mean = metadata.get('league_mean')
        shrinkage_alpha = metadata.get('shrinkage_alpha', DEFAULT_SHRINKAGE_ALPHA)
        if league_mean is not None:
            logger.info(f"Applied shrinkage: alpha={shrinkage_alpha}, league_mean={league_mean:.2f}")

        selected_variant = metadata.get("selected_calibration_variant", "none")
        predictions_array = train_model.apply_prediction_post_processing(
            np.asarray(predictions_array),
            league_mean=league_mean,
            shrinkage_alpha=shrinkage_alpha,
            calibration=metadata.get('calibration'),
            position_calibration=metadata.get("position_calibration"),
            selected_variant=selected_variant,
            position_ids=latest_stats["position_id"].to_numpy() if "position_id" in latest_stats.columns else None,
            position_caps=metadata.get('position_caps', {}),
        )
        if selected_variant == "position_aware":
            logger.info("Applied position-aware calibration to predictions.")
        elif selected_variant == "global":
            logger.info("Applied global calibration to predictions.")

        latest_stats['expected_points_next_gw'] = predictions_array
        
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
