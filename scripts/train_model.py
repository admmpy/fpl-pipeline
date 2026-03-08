"""
Train XGBoost model for FPL player points prediction.

This script:
1. Fetches historical player features from Snowflake
2. Engineers features and prepares training data
3. Trains an XGBoost regressor with cross-validation
4. Evaluates model performance
5. Saves the trained model to disk
"""
import logging
import json
import yaml
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.snowflake_client import get_snowflake_connection
from config import get_snowflake_config
from utils.local_data import (
    build_training_query,
    emit_selection_log,
    load_training_dataframe,
)
from utils.gameweek_quality import (
    classify_gameweek,
    load_gameweek_quality_policy,
    split_train_holdout_by_policy,
)

HOLDOUT_GAMEWEEKS = 5
SHRINKAGE_ALPHA = 0.0
REG_ALPHA = 0.1
REG_LAMBDA = 1.0
MAX_DEPTH = 5
ENABLE_CALIBRATION = True
CALIBRATION_STRENGTH = 0.8
POSITION_CALIBRATION_MIN_SAMPLES = 25
FEATURES_TO_SCALE = ['total_points', 'minutes_played', 'ict_index']
LOG_TARGET = os.getenv("LOG_TARGET", "1").lower() in {"1", "true", "yes"}
CALIBRATION_METRIC_TOLERANCE = float(os.getenv("CALIBRATION_METRIC_TOLERANCE", "0.0"))
DISABLED_FEATURES = {"form"}
SUM_AGG_COLUMNS = {
    "total_points",
    "minutes_played",
    "goals_scored",
    "expected_goals",
    "expected_goal_involvements",
    "assists",
    "expected_assists",
    "clean_sheets",
    "goals_conceded",
    "expected_goals_conceded",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "influence",
    "creativity",
    "threat",
    "ict_index",
}
MEAN_AGG_COLUMNS = {
    "opponent_defence_strength",
    "team_attack_strength",
    "team_roll_avg_goals_scored",
    "team_roll_avg_xg",
    "team_roll_avg_clean_sheets",
    "team_roll_avg_wins_pct",
    "opponent_roll_avg_goals_conceded",
    "opponent_roll_avg_xg",
    "opponent_team_position",
    "team_position_difference",
    "three_week_players_roll_avg_points",
    "five_week_players_roll_avg_points",
}


def _transform_target(y: pd.Series) -> pd.Series:
    return np.sign(y) * np.log1p(np.abs(y))


def _inverse_transform(pred: np.ndarray) -> np.ndarray:
    return np.sign(pred) * np.expm1(np.abs(pred))


def _fetch_training_data_from_snowflake(table_name: str) -> pd.DataFrame:
    if get_snowflake_config() is None:
        raise ValueError("Snowflake configuration not found. Cannot fetch training data.")

    query = build_training_query(table_name, apply_training_filters=False)
    with get_snowflake_connection() as conn:
        df = pd.read_sql(query, conn)

    df.columns = [str(column).lower() for column in df.columns]
    return df


def _apply_training_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()
    if "gameweek_id" in filtered.columns:
        filtered = filtered[filtered["gameweek_id"] > 3].copy()
    if "minutes_played" in filtered.columns:
        filtered = filtered[filtered["minutes_played"] > 0].copy()
    return filtered


def fetch_training_data(
    table_name: str = "fct_ml_player_features",
    source: Optional[str] = None,
    local_path: Optional[str] = None,
    policy: Optional[str] = None,
    max_age_days: Optional[int] = None,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """
    Fetch historical player features using local-first policy controls.
    
    Returns:
        DataFrame with player features and target variable (total_points)
    """
    df, metadata = load_training_dataframe(
        table_name=table_name,
        source=source,
        local_path=local_path,
        policy=policy,
        max_age_days=max_age_days,
        snowflake_loader=_fetch_training_data_from_snowflake,
        logger=logger,
    )
    df = _apply_training_filters(df)

    emit_selection_log(logger, context="scripts.train_model.fetch_training_data", metadata=metadata)
    if "gameweek_id" in df.columns and "player_id" in df.columns:
        logger.info(
            "Fetched %s training samples across %s gameweeks",
            len(df),
            df["gameweek_id"].nunique(),
        )
        logger.info(
            "Players: %s, Date range: GW%s-GW%s",
            df["player_id"].nunique(),
            df["gameweek_id"].min(),
            df["gameweek_id"].max(),
        )
    else:
        logger.info("Fetched %s training samples", len(df))

    if return_metadata:
        return df, metadata
    return df


def load_model_rules(path: str = "config/domain_rules.yaml") -> dict[str, Any]:
    """Load model gate rules for reporting thresholds."""

    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        return {}
    return dict(raw.get("model") or {})


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer additional features and handle missing values.
    
    Args:
        df: Raw feature DataFrame
        
    Returns:
        DataFrame with engineered features (excluding z-scores)
    """
    logger.info("Engineering features...")
    
    df = df.copy()
    
    # Canonicalise to one row per player/gameweek before shifting the target.
    df = collapse_player_gameweek_rows(df)

    # 1. Create target variables (next gameweek points and gameweek id)
    # Sort by player and gameweek, then shift total_points forward
    df = df.sort_values(['player_id', 'gameweek_id'])
    df['target_next_gw_points'] = df.groupby('player_id')['total_points'].shift(-1)
    df['target_gameweek_id'] = df.groupby('player_id')['gameweek_id'].shift(-1)
    
    # Drop terminal rows before any generic fill touches target columns.
    df = df.dropna(subset=['target_next_gw_points', 'target_gameweek_id']).copy()

    # 2. Position encoding (one-hot)
    df['is_gk'] = (df['position_id'] == 1).astype(int)
    df['is_def'] = (df['position_id'] == 2).astype(int)
    df['is_mid'] = (df['position_id'] == 3).astype(int)
    df['is_fwd'] = (df['position_id'] == 4).astype(int)
    
    # 3. Minutes bands (one-hot)
    if 'minutes_played' in df.columns:
        df['minutes_band'] = pd.cut(
            df['minutes_played'],
            bins=[-1, 30, 60, 1_000_000],
            labels=['0_30', '31_60', '61_90'],
        )
        band_dummies = pd.get_dummies(df['minutes_band'], prefix='minutes_band')
        df = pd.concat([df, band_dummies], axis=1)

    # 4. Fill missing values with sensible defaults for non-target numeric features only.
    target_cols = {'target_next_gw_points', 'target_gameweek_id'}
    numeric_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in target_cols
    ]
    df[numeric_cols] = df[numeric_cols].fillna(0)

    validate_engineered_features(df)
    
    logger.info(f"Feature engineering complete. {len(df)} samples remain after creating target.")
    
    return df


def validate_engineered_features(df: pd.DataFrame) -> None:
    """Fail fast when target engineering has produced invalid training rows."""

    required = {'gameweek_id', 'target_gameweek_id', 'target_next_gw_points'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing engineered target columns: {sorted(missing)}")

    if df['target_next_gw_points'].isna().any():
        raise ValueError("target_next_gw_points contains NaNs after engineering")
    if df['target_gameweek_id'].isna().any():
        raise ValueError("target_gameweek_id contains NaNs after engineering")
    if (df['target_gameweek_id'] <= 0).any():
        raise ValueError("target_gameweek_id must be > 0 after engineering")
    if (df['target_gameweek_id'] <= df['gameweek_id']).any():
        raise ValueError("target_gameweek_id must be strictly greater than gameweek_id")


def collapse_player_gameweek_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse double-gameweek fixture rows into a single player-gameweek row."""

    if "player_id" not in df.columns or "gameweek_id" not in df.columns:
        return df.copy()

    duplicate_mask = df.duplicated(subset=["player_id", "gameweek_id"], keep=False)
    if not duplicate_mask.any():
        return df.copy()

    logger.warning(
        "Collapsing %s duplicate player/gameweek rows before target engineering",
        int(duplicate_mask.sum()),
    )
    frame = df.sort_values(["player_id", "gameweek_id", "fixture_id"] if "fixture_id" in df.columns else ["player_id", "gameweek_id"]).copy()
    aggregations: dict[str, str] = {}
    for column in frame.columns:
        if column in {"player_id", "gameweek_id"}:
            continue
        if column in SUM_AGG_COLUMNS:
            aggregations[column] = "sum"
        elif column in MEAN_AGG_COLUMNS:
            aggregations[column] = "mean"
        else:
            aggregations[column] = "last"

    collapsed = (
        frame.groupby(["player_id", "gameweek_id"], as_index=False)
        .agg(aggregations)
    )
    return collapsed


def compute_global_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute global statistics for z-score normalisation.
    
    Args:
        df: Training DataFrame only (no holdout rows)
        
    Returns:
        Dict of global mean/std for each feature
    """
    global_stats: Dict[str, Dict[str, float]] = {}
    for feature in FEATURES_TO_SCALE:
        raw_std = float(df[feature].std())
        global_stats[feature] = {
            'mean': float(df[feature].mean()),
            'std': raw_std if raw_std > 0 else 1.0,
        }
    return global_stats


def add_z_scores(
    df: pd.DataFrame,
    global_stats: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """
    Add z-score features using global training statistics.
    
    Args:
        df: DataFrame to transform
        global_stats: Global stats computed from training data
        
    Returns:
        DataFrame with z-score columns added
    """
    for feature in FEATURES_TO_SCALE:
        mean_val = global_stats[feature]['mean']
        std_val = global_stats[feature]['std']
        df[f'{feature}_z_score'] = (df[feature] - mean_val) / std_val
        df[f'{feature}_z_score'] = df[f'{feature}_z_score'].fillna(0)
    return df


def split_train_holdout(
    df: pd.DataFrame,
    holdout_gameweeks: int,
    *,
    gameweek_policy: Optional[dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training and holdout sets based on target gameweek."""

    if gameweek_policy:
        train_df, holdout_df, _ = split_train_holdout_by_policy(
            df,
            holdout_gameweeks=holdout_gameweeks,
            policy=gameweek_policy,
            target_gameweek_col="target_gameweek_id",
        )
        return train_df, holdout_df

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
    Fit a linear calibration model using mean/std matching.
    """
    pred_std = float(np.std(y_pred))
    true_std = float(np.std(y_true))
    if pred_std == 0:
        return 1.0, 0.0

    a = true_std / pred_std
    b = float(np.mean(y_true)) - a * float(np.mean(y_pred))
    return float(a), float(b)


def blend_calibration(a: float, b: float, strength: float) -> Tuple[float, float]:
    """
    Blend calibration toward no-op (a=1, b=0) to avoid over-correction.
    """
    strength = max(0.0, min(1.0, strength))
    blended_a = 1.0 + strength * (a - 1.0)
    blended_b = strength * b
    return float(blended_a), float(blended_b)


def apply_calibration(predictions: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Apply linear calibration to predictions.
    """
    return a * predictions + b


def fit_position_aware_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    position_ids: np.ndarray,
    *,
    strength: float,
    min_samples: int = POSITION_CALIBRATION_MIN_SAMPLES,
) -> dict[str, Any]:
    """Fit global and per-position calibration using training-only data."""

    global_a, global_b = fit_calibration(y_true, y_pred)
    global_a, global_b = blend_calibration(global_a, global_b, strength)

    payload: dict[str, Any] = {
        "type": "position_aware_linear_v1",
        "global": {
            "a": float(global_a),
            "b": float(global_b),
            "strength": float(strength),
        },
        "min_samples": int(min_samples),
        "by_position": {},
    }

    frame = pd.DataFrame(
        {
            "position_id": np.asarray(position_ids).astype(int),
            "actual": np.asarray(y_true, dtype=float),
            "predicted": np.asarray(y_pred, dtype=float),
        }
    )
    for position_id, group in frame.groupby("position_id"):
        count = int(len(group))
        if count < int(min_samples):
            payload["by_position"][str(int(position_id))] = {
                "a": float(global_a),
                "b": float(global_b),
                "count": count,
                "fallback": "global",
            }
            continue

        pos_a, pos_b = fit_calibration(group["actual"].to_numpy(), group["predicted"].to_numpy())
        pos_a, pos_b = blend_calibration(pos_a, pos_b, strength)
        payload["by_position"][str(int(position_id))] = {
            "a": float(pos_a),
            "b": float(pos_b),
            "count": count,
            "fallback": None,
        }
    return payload


def apply_position_aware_calibration(
    predictions: np.ndarray,
    position_ids: np.ndarray,
    calibration_payload: Optional[dict[str, Any]],
) -> np.ndarray:
    """Apply per-position calibration with global fallback."""

    pred = np.asarray(predictions, dtype=float).copy()
    if not calibration_payload:
        return pred

    global_cfg = calibration_payload.get("global") or {}
    global_a = float(global_cfg.get("a", 1.0))
    global_b = float(global_cfg.get("b", 0.0))
    by_position = calibration_payload.get("by_position") or {}

    positions = np.asarray(position_ids).astype(int)
    for idx, position_id in enumerate(positions):
        cfg = by_position.get(str(int(position_id))) or {}
        a = float(cfg.get("a", global_a))
        b = float(cfg.get("b", global_b))
        pred[idx] = a * pred[idx] + b
    return pred


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 2 or len(y) < 2:
        return None
    x_rank = pd.Series(x).rank(method="average")
    y_rank = pd.Series(y).rank(method="average")
    corr = x_rank.corr(y_rank, method="pearson")
    if pd.isna(corr):
        return None
    return float(corr)


def compute_focus_position_abs_bias(
    positions: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    focus_positions: tuple[int, ...] = (1, 2),
) -> float:
    """Compute mean absolute bias for focus positions (default GK/DEF)."""

    frame = pd.DataFrame(
        {
            "position_id": np.asarray(positions).astype(int),
            "actual": np.asarray(y_true, dtype=float),
            "predicted": np.asarray(y_pred, dtype=float),
        }
    )
    focus = frame[frame["position_id"].isin(set(focus_positions))]
    if focus.empty:
        return 0.0
    grouped = (
        focus.assign(error_bias=focus["actual"] - focus["predicted"])
        .groupby("position_id")["error_bias"]
        .mean()
    )
    return float(np.mean(np.abs(grouped.to_numpy(dtype=float))))


def compute_prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute core error metrics for a prediction vector."""

    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "bias": float((true - pred).mean()),
    }


def _calibration_candidate_score(
    metrics: dict[str, float],
    *,
    focus_bias: Optional[float],
) -> tuple[float, float, float, float]:
    return (
        float(metrics["mae"]),
        float(metrics["rmse"]),
        abs(float(metrics["bias"])),
        float(focus_bias if focus_bias is not None else 0.0),
    )


def select_calibration_variant(
    *,
    y_train: np.ndarray,
    train_pred: np.ndarray,
    y_holdout: np.ndarray,
    holdout_pred: np.ndarray,
    train_position_ids: Optional[np.ndarray] = None,
    holdout_position_ids: Optional[np.ndarray] = None,
    strength: float,
    min_samples: int = POSITION_CALIBRATION_MIN_SAMPLES,
    tolerance: float = CALIBRATION_METRIC_TOLERANCE,
) -> dict[str, Any]:
    """Fit optional calibration variants and select a safe winner against `none`."""

    pre_pred = np.asarray(holdout_pred, dtype=float)
    baseline_metrics = compute_prediction_metrics(y_holdout, pre_pred)
    global_pred = None
    position_pred = None
    calibration = None
    position_calibration = None
    calibration_comparison = {
        "selected": "none",
        "gk_def_abs_bias_global": None,
        "gk_def_abs_bias_position": None,
        "position_calibration_gain": None,
    }
    selected_variant = "none"
    selected_pred = pre_pred

    candidates: list[tuple[tuple[float, float, float, float], str, np.ndarray]] = [
        (_calibration_candidate_score(baseline_metrics, focus_bias=None), "none", pre_pred)
    ]

    if strength > 0:
        a, b = fit_calibration(np.asarray(y_train, dtype=float), np.asarray(train_pred, dtype=float))
        a, b = blend_calibration(a, b, strength)
        calibration = {"a": float(a), "b": float(b), "strength": float(strength)}
        global_pred = apply_calibration(pre_pred, a, b)
        global_metrics = compute_prediction_metrics(y_holdout, global_pred)
        global_ok = (
            global_metrics["mae"] <= baseline_metrics["mae"] + tolerance
            and global_metrics["rmse"] <= baseline_metrics["rmse"] + tolerance
        )
        if global_ok:
            focus_bias = None
            if holdout_position_ids is not None:
                focus_bias = compute_focus_position_abs_bias(holdout_position_ids, y_holdout, global_pred)
            candidates.append((_calibration_candidate_score(global_metrics, focus_bias=focus_bias), "global", global_pred))

        if train_position_ids is not None and holdout_position_ids is not None:
            position_calibration = fit_position_aware_calibration(
                np.asarray(y_train, dtype=float),
                np.asarray(train_pred, dtype=float),
                np.asarray(train_position_ids, dtype=int),
                strength=strength,
                min_samples=min_samples,
            )
            position_pred = apply_position_aware_calibration(
                pre_pred,
                np.asarray(holdout_position_ids, dtype=int),
                position_calibration,
            )
            if global_pred is not None:
                global_abs = compute_focus_position_abs_bias(holdout_position_ids, y_holdout, global_pred)
                position_abs = compute_focus_position_abs_bias(holdout_position_ids, y_holdout, position_pred)
                calibration_comparison.update(
                    {
                        "gk_def_abs_bias_global": float(global_abs),
                        "gk_def_abs_bias_position": float(position_abs),
                        "position_calibration_gain": float(global_abs - position_abs),
                    }
                )
            position_metrics = compute_prediction_metrics(y_holdout, position_pred)
            position_ok = (
                position_metrics["mae"] <= baseline_metrics["mae"] + tolerance
                and position_metrics["rmse"] <= baseline_metrics["rmse"] + tolerance
            )
            if position_ok:
                focus_bias = None
                if holdout_position_ids is not None:
                    focus_bias = compute_focus_position_abs_bias(holdout_position_ids, y_holdout, position_pred)
                candidates.append(
                    (_calibration_candidate_score(position_metrics, focus_bias=focus_bias), "position_aware", position_pred)
                )

    selected_score, selected_variant, selected_pred = min(candidates, key=lambda item: item[0])
    _ = selected_score
    calibration_comparison["selected"] = selected_variant
    calibration_report = build_calibration_report(
        np.asarray(y_holdout, dtype=float),
        pre_pred,
        global_calibration_pred=np.asarray(global_pred) if global_pred is not None else None,
        position_calibration_pred=np.asarray(position_pred) if position_pred is not None else None,
        selected_variant=selected_variant,
        position_ids=np.asarray(holdout_position_ids, dtype=int) if holdout_position_ids is not None else None,
    )
    if calibration_report.get("gk_def_abs_bias_global") is not None:
        calibration_comparison["gk_def_abs_bias_global"] = calibration_report.get("gk_def_abs_bias_global")
    if calibration_report.get("gk_def_abs_bias_position") is not None:
        calibration_comparison["gk_def_abs_bias_position"] = calibration_report.get("gk_def_abs_bias_position")
        calibration_comparison["position_calibration_gain"] = float(
            calibration_report["gk_def_abs_bias_global"] - calibration_report["gk_def_abs_bias_position"]
        )

    if selected_variant == "none":
        calibration = None
        position_calibration = None
    elif selected_variant == "global":
        position_calibration = None
    elif selected_variant == "position_aware":
        calibration = None

    return {
        "selected_variant": selected_variant,
        "selected_pred": np.asarray(selected_pred, dtype=float),
        "calibration": calibration,
        "position_calibration": position_calibration,
        "calibration_comparison": calibration_comparison,
        "calibration_report": calibration_report,
    }


def apply_prediction_post_processing(
    predictions: np.ndarray,
    *,
    league_mean: Optional[float] = None,
    shrinkage_alpha: float = 0.0,
    calibration: Optional[dict[str, Any]] = None,
    position_calibration: Optional[dict[str, Any]] = None,
    selected_variant: str = "none",
    position_ids: Optional[np.ndarray] = None,
    position_caps: Optional[dict[Any, float]] = None,
) -> np.ndarray:
    """Apply the shared live/PIT prediction post-processing contract."""

    pred = np.asarray(predictions, dtype=float).copy()
    if league_mean is not None:
        pred = apply_shrinkage(pred, float(league_mean), float(shrinkage_alpha))

    if selected_variant == "position_aware" and position_calibration is not None and position_ids is not None:
        pred = apply_position_aware_calibration(pred, np.asarray(position_ids, dtype=int), position_calibration)
    elif selected_variant == "global" and calibration is not None:
        pred = apply_calibration(pred, float(calibration.get("a", 1.0)), float(calibration.get("b", 0.0)))

    pred = np.maximum(pred, 0)

    if position_caps and position_ids is not None:
        cap_series = pd.Series(np.asarray(position_ids, dtype=int)).map(position_caps).fillna(np.inf)
        pred = np.minimum(pred, cap_series.to_numpy(dtype=float))

    return pred


def build_calibration_report(
    y_true: np.ndarray,
    pre_calibration_pred: np.ndarray,
    *,
    global_calibration_pred: Optional[np.ndarray] = None,
    position_calibration_pred: Optional[np.ndarray] = None,
    selected_variant: str = "none",
    position_ids: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Build pre/post calibration metrics and deltas for auditing."""

    pre = compute_prediction_metrics(y_true, pre_calibration_pred)
    report: dict[str, Any] = {
        "selected_variant": selected_variant,
        "pre_calibration": pre,
        "post_global": None,
        "post_position_aware": None,
        "selected_post": pre,
        "selected_delta": {"mae": 0.0, "rmse": 0.0, "bias": 0.0},
    }

    if global_calibration_pred is not None:
        global_metrics = compute_prediction_metrics(y_true, global_calibration_pred)
        report["post_global"] = global_metrics
        report["delta_global_vs_pre"] = {
            key: float(global_metrics[key] - pre[key]) for key in ("mae", "rmse", "bias")
        }

    if position_calibration_pred is not None:
        position_metrics = compute_prediction_metrics(y_true, position_calibration_pred)
        report["post_position_aware"] = position_metrics
        report["delta_position_vs_pre"] = {
            key: float(position_metrics[key] - pre[key]) for key in ("mae", "rmse", "bias")
        }

    selected_map: dict[str, Optional[np.ndarray]] = {
        "none": pre_calibration_pred,
        "global": global_calibration_pred,
        "position_aware": position_calibration_pred,
    }
    selected_pred = selected_map.get(selected_variant)
    if selected_pred is None:
        selected_pred = pre_calibration_pred
    selected_metrics = compute_prediction_metrics(y_true, selected_pred)
    report["selected_post"] = selected_metrics
    report["selected_delta"] = {
        key: float(selected_metrics[key] - pre[key]) for key in ("mae", "rmse", "bias")
    }

    if position_ids is not None:
        positions = np.asarray(position_ids).astype(int)
        pre_focus = compute_focus_position_abs_bias(positions, y_true, pre_calibration_pred)
        report["gk_def_abs_bias_pre"] = float(pre_focus)
        if global_calibration_pred is not None:
            report["gk_def_abs_bias_global"] = float(
                compute_focus_position_abs_bias(positions, y_true, global_calibration_pred)
            )
        if position_calibration_pred is not None:
            report["gk_def_abs_bias_position"] = float(
                compute_focus_position_abs_bias(positions, y_true, position_calibration_pred)
            )

    return report


def build_weekly_backtest_report(
    holdout_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    gameweek_policy: Optional[dict[str, Any]] = None,
    model_rules: Optional[dict[str, Any]] = None,
    top_k: int = 11,
) -> dict[str, Any]:
    """Build rolling per-gameweek backtest report for trusted weeks."""

    if "target_gameweek_id" not in holdout_df.columns:
        raise ValueError("holdout_df must include target_gameweek_id for weekly backtest report")

    frame = holdout_df.copy().reset_index(drop=True)
    frame["actual_points"] = np.asarray(y_true, dtype=float)
    frame["predicted_points"] = np.asarray(y_pred, dtype=float)
    frame["abs_error"] = np.abs(frame["actual_points"] - frame["predicted_points"])
    frame["error_bias"] = frame["actual_points"] - frame["predicted_points"]

    min_ratio = float((model_rules or {}).get("min_prediction_to_actual_ratio", 0.6))
    max_ratio = float((model_rules or {}).get("max_prediction_to_actual_ratio", 1.6))
    instability_bias_threshold = float((model_rules or {}).get("instability_bias_abs_threshold", 10.0))

    per_week: list[dict[str, Any]] = []
    for gameweek_id, group in frame.groupby("target_gameweek_id"):
        actual_total = float(group["actual_points"].sum())
        predicted_total = float(group["predicted_points"].sum())
        ratio = float(predicted_total / max(actual_total, 1e-9))
        mae = float(group["abs_error"].mean())
        rmse = float(np.sqrt(np.mean(np.square(group["error_bias"]))))
        bias = float(group["error_bias"].mean())

        likely = group[group["minutes_played"] >= 60].copy() if "minutes_played" in group.columns else group.copy()
        k = min(int(top_k), int(len(likely)))

        top_k_hit_rate = None
        rank_corr = None
        if k > 0:
            pred_top = likely.nlargest(k, "predicted_points")
            actual_top = likely.nlargest(k, "actual_points")
            pred_ids = set(pred_top["player_id"].tolist()) if "player_id" in likely.columns else set(pred_top.index.tolist())
            actual_ids = set(actual_top["player_id"].tolist()) if "player_id" in likely.columns else set(actual_top.index.tolist())
            top_k_hit_rate = float(len(pred_ids & actual_ids) / k)
            rank_corr = _safe_spearman(
                likely["predicted_points"].to_numpy(),
                likely["actual_points"].to_numpy(),
            )

        selected_xi = group.nlargest(min(11, len(group)), "predicted_points")
        optimal_xi = group.nlargest(min(11, len(group)), "actual_points")
        selected_xi_actual = float(selected_xi["actual_points"].sum())
        optimal_xi_actual = float(optimal_xi["actual_points"].sum())
        selected_xi_regret = float(optimal_xi_actual - selected_xi_actual)

        selected_squad = group.nlargest(min(15, len(group)), "predicted_points")
        selected_squad_pred_total = float(selected_squad["predicted_points"].sum())
        selected_squad_actual_total = float(selected_squad["actual_points"].sum())

        if "position_id" in group.columns:
            position_totals = (
                group.groupby("position_id")[["predicted_points", "actual_points"]]
                .sum()
                .reset_index()
                .to_dict(orient="records")
            )
            for row in position_totals:
                row["position_id"] = int(row["position_id"])
                row["predicted_points"] = float(row["predicted_points"])
                row["actual_points"] = float(row["actual_points"])
                row["bias"] = float(row["actual_points"] - row["predicted_points"])
        else:
            position_totals = []

        largest_misses_cols = ["player_id", "actual_points", "predicted_points", "abs_error"]
        if "web_name" in group.columns:
            largest_misses_cols.insert(1, "web_name")
        largest_misses = group.nlargest(min(5, len(group)), "abs_error")[largest_misses_cols].to_dict(orient="records")
        for row in largest_misses:
            row["player_id"] = int(row.get("player_id", 0))
            row["actual_points"] = float(row["actual_points"])
            row["predicted_points"] = float(row["predicted_points"])
            row["abs_error"] = float(row["abs_error"])

        actual_range = float(group["actual_points"].max() - group["actual_points"].min())
        predicted_range = float(group["predicted_points"].max() - group["predicted_points"].min())
        range_ratio = float(predicted_range / max(actual_range, 1e-9))

        collapse = ratio < min_ratio or ratio > max_ratio
        quality = classify_gameweek(int(gameweek_id), gameweek_policy or {}) if gameweek_policy else {"trusted": True, "status": "trusted"}

        per_week.append(
            {
                "gameweek_id": int(gameweek_id),
                "trusted": bool(quality["trusted"]),
                "status": quality["status"],
                "prediction_total": predicted_total,
                "actual_total": actual_total,
                "prediction_to_actual_ratio": ratio,
                "mae": mae,
                "rmse": rmse,
                "bias": bias,
                "prediction_range_ratio": range_ratio,
                "selected_squad_predicted_total": selected_squad_pred_total,
                "selected_squad_actual_total": selected_squad_actual_total,
                "selected_squad_bias": float(selected_squad_actual_total - selected_squad_pred_total),
                "selected_xi_actual_total": selected_xi_actual,
                "oracle_xi_actual_total": optimal_xi_actual,
                "top_k_hit_rate": top_k_hit_rate,
                "rank_correlation": rank_corr,
                "selected_xi_regret": selected_xi_regret,
                "prediction_collapse": bool(collapse),
                "largest_player_misses": largest_misses,
                "position_totals": position_totals,
            }
        )

    if "position_id" in frame.columns:
        position_summary = (
            frame.groupby("position_id")
            .agg(
                count=("position_id", "count"),
                mae=("abs_error", "mean"),
                bias=("error_bias", "mean"),
            )
            .reset_index()
            .to_dict(orient="records")
        )
        for row in position_summary:
            row["position_id"] = int(row["position_id"])
            row["count"] = int(row["count"])
            row["mae"] = float(row["mae"])
            row["bias"] = float(row["bias"])
    else:
        position_summary = []

    top_k_values = [row["top_k_hit_rate"] for row in per_week if row["top_k_hit_rate"] is not None]
    rank_values = [row["rank_correlation"] for row in per_week if row["rank_correlation"] is not None]
    regret_values = [row["selected_xi_regret"] for row in per_week]
    squad_bias_values = [abs(row["selected_squad_bias"]) for row in per_week]
    selected_xi_totals = [row["selected_xi_actual_total"] for row in per_week]
    oracle_xi_totals = [row["oracle_xi_actual_total"] for row in per_week]
    prediction_ratios = [row["prediction_to_actual_ratio"] for row in per_week]
    collapse_weeks = [row["gameweek_id"] for row in per_week if row["prediction_collapse"]]
    max_position_bias = max((abs(row["bias"]) for row in position_summary), default=0.0)

    per_week_sorted = sorted(per_week, key=lambda row: row["gameweek_id"])

    bias_flip_pairs: list[list[int]] = []
    for idx in range(1, len(per_week_sorted)):
        prev = per_week_sorted[idx - 1]
        curr = per_week_sorted[idx]
        prev_bias = float(prev["bias"])
        curr_bias = float(curr["bias"])
        prev_sign = np.sign(prev_bias)
        curr_sign = np.sign(curr_bias)
        if prev_sign == 0 or curr_sign == 0 or prev_sign == curr_sign:
            continue
        if abs(prev_bias) < instability_bias_threshold or abs(curr_bias) < instability_bias_threshold:
            continue
        bias_flip_pairs.append([int(prev["gameweek_id"]), int(curr["gameweek_id"])])

    summary = {
        "weekly_count": int(len(per_week)),
        "mae": float(frame["abs_error"].mean()),
        "rmse": float(np.sqrt(np.mean(np.square(frame["error_bias"])) )),
        "bias": float(frame["error_bias"].mean()),
        "top_k_hit_rate_mean": float(np.mean(top_k_values)) if top_k_values else 0.0,
        "rank_correlation_mean": float(np.mean(rank_values)) if rank_values else 0.0,
        "selected_xi_regret_mean": float(np.mean(regret_values)) if regret_values else 0.0,
        "selected_xi_actual_total_mean": float(np.mean(selected_xi_totals)) if selected_xi_totals else 0.0,
        "oracle_xi_actual_total_mean": float(np.mean(oracle_xi_totals)) if oracle_xi_totals else 0.0,
        "squad_total_bias_abs_mean": float(np.mean(squad_bias_values)) if squad_bias_values else 0.0,
        "max_position_bias_abs": float(max_position_bias),
        "prediction_to_actual_ratio_min": float(min(prediction_ratios)) if prediction_ratios else 1.0,
        "prediction_to_actual_ratio_max": float(max(prediction_ratios)) if prediction_ratios else 1.0,
        "prediction_collapse_weeks": int(len(collapse_weeks)),
        "prediction_collapse_detected": bool(collapse_weeks),
        "prediction_collapse_gameweeks": collapse_weeks,
        "instability_bias_abs_threshold": float(instability_bias_threshold),
        "bias_flip_weeks": int(len(bias_flip_pairs)),
        "bias_flip_detected": bool(bias_flip_pairs),
        "bias_flip_gameweek_pairs": bias_flip_pairs,
    }

    return {
        "summary": summary,
        "per_week": per_week_sorted,
        "per_position": sorted(position_summary, key=lambda row: row["position_id"]),
    }


def evaluate_publication_readiness(
    backtest_report: dict[str, Any],
    *,
    calibration_report: Optional[dict[str, Any]] = None,
    model_rules: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Determine whether forward publication should be allowed for the active artefact."""

    summary = (backtest_report or {}).get("summary") or {}
    calibration_delta = (calibration_report or {}).get("selected_delta") or {}
    gates = {
        "max_prediction_collapse_weeks": int(summary.get("prediction_collapse_weeks", 0))
        <= int((model_rules or {}).get("max_prediction_collapse_weeks", 0)),
        "max_bias_flip_weeks": int(summary.get("bias_flip_weeks", 0))
        <= int((model_rules or {}).get("max_bias_flip_weeks", 0)),
        "min_top_k_hit_rate": float(summary.get("top_k_hit_rate_mean", 0.0))
        >= float((model_rules or {}).get("min_top_k_hit_rate", 0.0)),
        "min_rank_correlation": float(summary.get("rank_correlation_mean", 0.0))
        >= float((model_rules or {}).get("min_rank_correlation", 0.0)),
        "max_selected_xi_regret": float(summary.get("selected_xi_regret_mean", 0.0))
        <= float((model_rules or {}).get("max_selected_xi_regret", float("inf"))),
        "prediction_ratio_min": float(summary.get("prediction_to_actual_ratio_min", 1.0))
        >= float((model_rules or {}).get("min_prediction_to_actual_ratio", 0.0)),
        "prediction_ratio_max": float(summary.get("prediction_to_actual_ratio_max", 1.0))
        <= float((model_rules or {}).get("max_prediction_to_actual_ratio", float("inf"))),
        "calibration_safe_mae": float(calibration_delta.get("mae", 0.0)) <= CALIBRATION_METRIC_TOLERANCE,
        "calibration_safe_rmse": float(calibration_delta.get("rmse", 0.0)) <= CALIBRATION_METRIC_TOLERANCE,
    }
    ready = all(gates.values())
    reasons = [name for name, passed in gates.items() if not passed]
    return {"ready": ready, "gates": gates, "reasons": reasons}


def write_weekly_backtest_report(report: dict[str, Any], path: str) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return path


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
    
    logger.info(f"{label} bias by {group_col}:")
    for _, row in summary.iterrows():
        logger.info(
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

        # Minutes bands
        'minutes_band_0_30',
        'minutes_band_31_60',
        'minutes_band_61_90',
        
    ]
    
    return [feature for feature in features if feature not in DISABLED_FEATURES]


def train_xgboost_model(X: pd.DataFrame, y: pd.Series) -> xgb.XGBRegressor:
    """
    Train XGBoost model with optimal hyperparameters.
    
    Args:
        X: Feature matrix
        y: Target variable
        
    Returns:
        Trained XGBoost model
    """
    logger.info("Training XGBoost model...")
    
    # Define model with sensible hyperparameters
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=MAX_DEPTH,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        objective='reg:squarederror',
        reg_alpha=REG_ALPHA,
        reg_lambda=REG_LAMBDA,
        eval_metric='mae'
    )
    
    # Time series cross-validation (respects temporal order)
    tscv = TimeSeriesSplit(n_splits=5)
    
    logger.info("Running time series cross-validation...")
    y_train = _transform_target(y) if LOG_TARGET else y
    cv_mae_scores = []
    for train_idx, val_idx in tscv.split(X):
        model_cv = xgb.XGBRegressor(**model.get_params())
        model_cv.fit(X.iloc[train_idx], y_train.iloc[train_idx])
        preds = model_cv.predict(X.iloc[val_idx])
        if LOG_TARGET:
            preds = _inverse_transform(preds)
        cv_mae_scores.append(mean_absolute_error(y.iloc[val_idx], preds))
    cv_mae = float(np.mean(cv_mae_scores))
    cv_std = float(np.std(cv_mae_scores))
    logger.info(f"Cross-validation MAE: {cv_mae:.3f} (+/- {cv_std:.3f})")
    
    # Train final model on all data
    logger.info("Training final model on full dataset...")
    model.fit(X, y_train)
    
    return model


def evaluate_model(model: xgb.XGBRegressor, X: pd.DataFrame, y: pd.Series):
    """
    Evaluate model performance on training data.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: True target values
    """
    logger.info("=" * 60)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 60)
    
    # Generate predictions
    y_pred = model.predict(X)
    if LOG_TARGET:
        y_pred = _inverse_transform(y_pred)
    
    # Calculate metrics
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    bias = float((y - y_pred).mean())
    
    logger.info("Overall Metrics:")
    logger.info(f"  MAE:  {mae:.3f} points")
    logger.info(f"  RMSE: {rmse:.3f} points")
    logger.info(f"  R2:   {r2:.3f}")
    logger.info(f"  Bias: {bias:.3f} points (actual - predicted)")
    
    # Feature importance
    logger.info("Top 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for idx, row in feature_importance.head(10).iterrows():
        logger.info(f"  {row['feature']:40s} {row['importance']:.4f}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'feature_importance': feature_importance,
        'bias': bias
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
    payload_metadata = dict(metrics.get('metadata', {}))
    payload_metadata.update(
        {
            "feature_cols": metrics.get("feature_cols", []),
            "zscore_stats": metrics.get("zscore_stats", {}),
            "position_caps": metrics.get("position_caps", {}),
            "position_calibration": metrics.get("position_calibration", {}),
            "calibration_report": metrics.get("calibration_report", {}),
            "train_target_stats": metrics.get("train_target_stats", {}),
            "training_window": metrics.get("training_window", {}),
            "evaluation_window_summary": metrics.get("evaluation_window_summary", {}),
            "trusted_gameweek_policy_version": metrics.get("trusted_gameweek_policy_version"),
        }
    )

    with open(output_path, 'wb') as f:
        pickle.dump(
            {
                'model': model,
                'metadata': payload_metadata
            },
            f
        )
    
    logger.info(f"Model saved to: {output_path}")
    
    # Save metrics
    metrics_path = Path(output_path).with_suffix('.metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("FPL XGBoost Model Metrics\n")
        f.write("="*60 + "\n\n")
        f.write(f"Train MAE:  {metrics['mae']:.3f} points\n")
        f.write(f"Train RMSE: {metrics['rmse']:.3f} points\n")
        f.write(f"Train R²:   {metrics['r2']:.3f}\n")
        f.write(f"Train Bias: {metrics['bias']:.3f} points (actual - predicted)\n\n")
        if 'holdout_mae' in metrics:
            f.write(f"Holdout MAE:  {metrics['holdout_mae']:.3f} points\n")
            f.write(f"Holdout RMSE: {metrics['holdout_rmse']:.3f} points\n")
            f.write(f"Holdout Bias: {metrics['holdout_bias']:.3f} points\n\n")
        if 'holdout_p95_by_position' in metrics:
            f.write("Holdout 95th Percentile by Position (actual vs predicted):\n")
            for row in metrics['holdout_p95_by_position']:
                f.write(
                    f"  position_id={row['position_id']} "
                    f"actual_p95={row['actual_p95']:.2f} "
                    f"predicted_p95={row['predicted_p95']:.2f} "
                    f"count={row['count']}\n"
                )
            f.write("\n")
        weekly_summary = metrics.get("evaluation_window_summary") or {}
        if weekly_summary:
            f.write("Weekly Backtest Summary:\n")
            f.write(f"  weeks={weekly_summary.get('weekly_count', 0)}\n")
            f.write(f"  top_k_hit_rate_mean={weekly_summary.get('top_k_hit_rate_mean', 0.0):.3f}\n")
            f.write(f"  rank_correlation_mean={weekly_summary.get('rank_correlation_mean', 0.0):.3f}\n")
            f.write(f"  selected_xi_regret_mean={weekly_summary.get('selected_xi_regret_mean', 0.0):.3f}\n")
            f.write(
                f"  prediction_collapse_weeks={weekly_summary.get('prediction_collapse_weeks', 0)}\n\n"
            )
        calibration_report = metrics.get("calibration_report") or {}
        if calibration_report:
            selected_delta = calibration_report.get("selected_delta") or {}
            f.write("Calibration Summary:\n")
            f.write(f"  selected_variant={calibration_report.get('selected_variant', 'none')}\n")
            f.write(f"  delta_mae={float(selected_delta.get('mae', 0.0)):.3f}\n")
            f.write(f"  delta_rmse={float(selected_delta.get('rmse', 0.0)):.3f}\n")
            f.write(f"  delta_bias={float(selected_delta.get('bias', 0.0)):.3f}\n\n")
        f.write("Top 10 Features:\n")
        for idx, row in metrics['feature_importance'].head(10).iterrows():
            f.write(f"  {row['feature']:40s} {row['importance']:.4f}\n")
    
    logger.info(f"Metrics saved to: {metrics_path}")


def main():
    """
    Main training pipeline.
    """
    logger.info("=" * 60)
    logger.info("FPL XGBOOST MODEL TRAINING")
    logger.info("=" * 60)
    
    # 1. Fetch data
    df = fetch_training_data()
    
    # 2. Engineer base features
    df = engineer_features(df)
    
    # 3. Train/holdout split by trusted gameweek policy
    rules_path = os.getenv("DOMAIN_RULES_PATH", "config/domain_rules.yaml")
    gameweek_policy = load_gameweek_quality_policy(rules_path)
    train_df, holdout_df = split_train_holdout(
        df,
        HOLDOUT_GAMEWEEKS,
        gameweek_policy=gameweek_policy,
    )
    logger.info(
        "Applied gameweek quality policy version=%s excluded=%s untrusted=%s",
        gameweek_policy.get("policy_version"),
        gameweek_policy.get("excluded_gameweeks"),
        gameweek_policy.get("backfilled_but_untrusted_gameweeks"),
    )
    
    # 4. Z-score normalisation using training-only global stats
    global_stats = compute_global_stats(train_df)
    train_df = add_z_scores(train_df, global_stats)
    holdout_df = add_z_scores(holdout_df, global_stats)
    
    # 5. Select features and target
    features = select_features()
    
    # Check which features exist in the data
    available_features = [f for f in features if f in train_df.columns]
    missing_features = [f for f in features if f not in train_df.columns]
    
    if missing_features:
        logger.warning(f"{len(missing_features)} features not found in data:")
        for f in missing_features[:5]:  # Show first 5
            logger.warning(f"  - {f}")
        if len(missing_features) > 5:
            logger.warning(f"  ... and {len(missing_features) - 5} more")
    
    if missing_features:
        for feature in missing_features:
            train_df[feature] = 0
            holdout_df[feature] = 0
        available_features = features
    
    X_train = train_df[available_features]
    y_train = train_df['target_next_gw_points']
    
    X_holdout = holdout_df[available_features]
    y_holdout = holdout_df['target_next_gw_points']
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Training target shape: {y_train.shape}")
    logger.info(f"Training target range: [{y_train.min():.1f}, {y_train.max():.1f}] points")
    logger.info(f"Training target mean: {y_train.mean():.2f} points")
    
    # 6. Train model
    model = train_xgboost_model(X_train, y_train)
    
    # 7. Evaluate on training data
    metrics = evaluate_model(model, X_train, y_train)
    
    # 8. Holdout evaluation with shrinkage and optional calibration
    league_mean = float(y_train.mean())
    holdout_pred_raw = model.predict(X_holdout)
    if LOG_TARGET:
        holdout_pred_raw = _inverse_transform(holdout_pred_raw)
    holdout_pred_raw = apply_shrinkage(holdout_pred_raw, league_mean, SHRINKAGE_ALPHA)

    train_pred_raw = model.predict(X_train)
    if LOG_TARGET:
        train_pred_raw = _inverse_transform(train_pred_raw)
    train_pred_raw = apply_shrinkage(train_pred_raw, league_mean, SHRINKAGE_ALPHA)

    calibration_result = select_calibration_variant(
        y_train=y_train.to_numpy(),
        train_pred=np.asarray(train_pred_raw),
        y_holdout=y_holdout.to_numpy(),
        holdout_pred=np.asarray(holdout_pred_raw),
        train_position_ids=train_df['position_id'].to_numpy() if 'position_id' in train_df.columns else None,
        holdout_position_ids=holdout_df['position_id'].to_numpy() if 'position_id' in holdout_df.columns else None,
        strength=CALIBRATION_STRENGTH if ENABLE_CALIBRATION else 0.0,
        min_samples=POSITION_CALIBRATION_MIN_SAMPLES,
    )
    calibration = calibration_result["calibration"]
    position_calibration = calibration_result["position_calibration"]
    calibration_comparison = calibration_result["calibration_comparison"]
    calibration_report = calibration_result["calibration_report"]
    selected_variant = calibration_result["selected_variant"]
    holdout_pred = apply_prediction_post_processing(
        np.asarray(holdout_pred_raw),
        league_mean=None,
        shrinkage_alpha=0.0,
        calibration=calibration,
        position_calibration=position_calibration,
        selected_variant=selected_variant,
        position_ids=holdout_df['position_id'].to_numpy() if 'position_id' in holdout_df.columns else None,
        position_caps=None,
    )
    
    holdout_mae = mean_absolute_error(y_holdout, holdout_pred)
    holdout_rmse = np.sqrt(mean_squared_error(y_holdout, holdout_pred))
    holdout_bias = float((y_holdout - holdout_pred).mean())

    holdout_p95_by_position = []
    if 'position_id' in holdout_df.columns:
        holdout_context = holdout_df[['position_id']].copy()
        holdout_context['actual'] = y_holdout.to_numpy()
        holdout_context['predicted'] = holdout_pred
        for position_id, group in holdout_context.groupby('position_id'):
            holdout_p95_by_position.append(
                {
                    "position_id": int(position_id),
                    "actual_p95": float(np.percentile(group['actual'], 95)),
                    "predicted_p95": float(np.percentile(group['predicted'], 95)),
                    "count": int(len(group)),
                }
            )
    
    logger.info("=" * 60)
    logger.info("HOLDOUT EVALUATION")
    logger.info("=" * 60)
    logger.info("Holdout Metrics:")
    logger.info(f"  MAE:  {holdout_mae:.3f} points")
    logger.info(f"  RMSE: {holdout_rmse:.3f} points")
    logger.info(f"  Bias: {holdout_bias:.3f} points (actual - predicted)")
    
    holdout_context = holdout_df[['player_id', 'position_id', 'minutes_played']].copy()
    holdout_context['minutes_band'] = pd.cut(
        holdout_context['minutes_played'],
        bins=[-1, 30, 60, 1_000_000],
        labels=['0-30', '31-60', '61-90']
    )
    
    print_group_bias(holdout_context, y_holdout.to_numpy(), holdout_pred, 'position_id', "Holdout")
    print_group_bias(holdout_context, y_holdout.to_numpy(), holdout_pred, 'minutes_band', "Holdout")
    
    backtest_report = build_weekly_backtest_report(
        holdout_df,
        y_holdout.to_numpy(),
        holdout_pred,
        gameweek_policy=gameweek_policy,
        model_rules=load_model_rules(rules_path),
    )
    publication_status = evaluate_publication_readiness(
        backtest_report,
        calibration_report=calibration_report,
        model_rules=load_model_rules(rules_path),
    )
    report_path = write_weekly_backtest_report(backtest_report, "logs/model_weekly_report.json")
    logger.info("Weekly backtest report written to %s", report_path)
    logger.info("Weekly backtest summary: %s", backtest_report["summary"])
    logger.info("Forward publication ready=%s reasons=%s", publication_status["ready"], publication_status["reasons"])

    metrics['holdout_mae'] = holdout_mae
    metrics['holdout_rmse'] = holdout_rmse
    metrics['holdout_bias'] = holdout_bias
    metrics['holdout_p95_by_position'] = holdout_p95_by_position
    metrics['zscore_stats'] = global_stats
    metrics['feature_cols'] = available_features
    metrics['position_calibration'] = position_calibration
    metrics['calibration_report'] = calibration_report
    metrics['evaluation_window_summary'] = backtest_report['summary']
    metrics['trusted_gameweek_policy_version'] = gameweek_policy.get('policy_version')

    position_caps = {}
    if 'position_id' in train_df.columns:
        for position_id, group in train_df.groupby('position_id'):
            position_caps[int(position_id)] = float(np.percentile(group['target_next_gw_points'], 95))
    metrics['position_caps'] = position_caps

    metrics['train_target_stats'] = {
        "mean": float(y_train.mean()),
        "std": float(y_train.std()),
        "min": float(y_train.min()),
        "max": float(y_train.max()),
    }
    metrics['training_window'] = {
        "min_gameweek": int(train_df['target_gameweek_id'].min()),
        "max_gameweek": int(train_df['target_gameweek_id'].max()),
    }
    metrics['metadata'] = {
        'league_mean': league_mean,
        'shrinkage_alpha': SHRINKAGE_ALPHA,
        'reg_alpha': REG_ALPHA,
        'reg_lambda': REG_LAMBDA,
        'max_depth': MAX_DEPTH,
        'holdout_gameweeks': HOLDOUT_GAMEWEEKS,
        'calibration': calibration,
        'calibration_comparison': calibration_comparison,
        'calibration_report': calibration_report,
        'backtest_report_path': report_path,
        'gameweek_policy_version': gameweek_policy.get('policy_version'),
        'use_log_target': LOG_TARGET,
        'selected_calibration_variant': selected_variant,
        'forward_publish_ready': publication_status['ready'],
        'forward_publish_gates': publication_status['gates'],
        'forward_publish_reasons': publication_status['reasons'],
    }
    
    # 9. Save
    save_model(model, metrics, output_path="logs/model.bin")
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info("Next steps:")
    logger.info("  1. Review model metrics above")
    logger.info("  2. Run pipeline: python run_once.py")
    logger.info("  3. Check predictions in Snowflake: SELECT * FROM recommended_squad")


if __name__ == "__main__":
    main()
