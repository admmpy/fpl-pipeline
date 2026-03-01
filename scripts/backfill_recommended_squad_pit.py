"""
Point-in-time backfill for historical recommended squads.

For each target gameweek, this script:
1) Trains a model using data with target_gameweek_id < target gameweek
2) Runs inference using features available up to target gameweek - 1
3) Optimises a 15-player squad
4) Loads rows into recommended_squad with a backfill_method marker
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
import os
import sys

import numpy as np
import pandas as pd

# Add project root to Python path (align with other pipeline entry points).
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TABLE_SCHEMAS
from scripts import train_model
from tasks.ml_tasks import fetch_training_data, ensure_z_score_columns
from tasks.optimizer_tasks import optimize_squad_task, add_recommendation_metadata
from utils.snowflake_client import create_typed_table, get_snowflake_connection, insert_typed_records

LOGGER = logging.getLogger(__name__)
BACKFILL_METHOD = "pit_retrain_v1"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill recommended_squad with point-in-time retraining."
    )
    parser.add_argument(
        "--gameweeks",
        nargs="+",
        type=int,
        required=True,
        help="Target gameweeks to backfill, e.g. --gameweeks 27 28",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run end-to-end but do not write to Snowflake.",
    )
    parser.add_argument(
        "--recommended-at-prefix",
        type=str,
        default=None,
        help="Optional prefix for recommended_at timestamps.",
    )
    return parser.parse_args()


def _add_minutes_band_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep training and inference feature shapes aligned.
    """
    df = df.copy()
    if "minutes_played" in df.columns:
        df["minutes_band"] = pd.cut(
            df["minutes_played"],
            bins=[-1, 30, 60, 1_000_000],
            labels=["0_30", "31_60", "61_90"],
        )
        band_dummies = pd.get_dummies(df["minutes_band"], prefix="minutes_band")
        df = pd.concat([df, band_dummies], axis=1)
    return df


def _prepare_training_data(feature_df: pd.DataFrame, target_gw: int) -> tuple[pd.DataFrame, List[str], Dict[str, Dict[str, float]]]:
    hist_df = feature_df[feature_df["gameweek_id"] < target_gw].copy()
    if hist_df.empty:
        raise ValueError(f"No historical rows found before GW{target_gw}")

    engineered = train_model.engineer_features(hist_df)
    train_df = engineered[engineered["target_gameweek_id"] < target_gw].copy()
    if train_df.empty:
        raise ValueError(f"No trainable rows found for GW{target_gw}")

    zscore_stats = train_model.compute_global_stats(train_df)
    train_df = train_model.add_z_scores(train_df, zscore_stats)

    feature_cols = train_model.select_features()
    missing_features = [f for f in feature_cols if f not in train_df.columns]
    for feature in missing_features:
        train_df[feature] = 0

    return train_df, feature_cols, zscore_stats


def _train_point_in_time_model(
    train_df: pd.DataFrame,
    feature_cols: List[str],
) -> tuple[Any, float]:
    X_train = train_df[feature_cols]
    y_train = train_df["target_next_gw_points"]
    model = train_model.train_xgboost_model(X_train, y_train)
    league_mean = float(y_train.mean())
    return model, league_mean


def _build_predictions_for_target_gw(
    feature_df: pd.DataFrame,
    target_gw: int,
    model: Any,
    feature_cols: List[str],
    zscore_stats: Dict[str, Dict[str, float]],
    league_mean: float,
) -> List[Dict[str, Any]]:
    inference_df = feature_df[feature_df["gameweek_id"] <= (target_gw - 1)].copy()
    if inference_df.empty:
        raise ValueError(f"No inference rows available for GW{target_gw}")

    # Mirror feature engineering used in training.
    inference_df["is_gk"] = (inference_df["position_id"] == 1).astype(int)
    inference_df["is_def"] = (inference_df["position_id"] == 2).astype(int)
    inference_df["is_mid"] = (inference_df["position_id"] == 3).astype(int)
    inference_df["is_fwd"] = (inference_df["position_id"] == 4).astype(int)
    inference_df = _add_minutes_band_features(inference_df)
    inference_df = ensure_z_score_columns(
        inference_df,
        train_model.FEATURES_TO_SCALE,
        stats=zscore_stats,
        logger=LOGGER,
    )

    latest_stats = inference_df.sort_values("gameweek_id").groupby("player_id").tail(1).copy()
    for feature in feature_cols:
        if feature not in latest_stats.columns:
            latest_stats[feature] = 0

    X_inference = latest_stats[feature_cols].fillna(0)
    preds = model.predict(X_inference)
    if train_model.LOG_TARGET:
        preds = train_model._inverse_transform(preds)  # Existing helper from training module
    preds = train_model.apply_shrinkage(preds, league_mean, train_model.SHRINKAGE_ALPHA)
    preds = np.maximum(preds, 0)

    latest_stats["expected_points_next_gw"] = preds
    predictions: List[Dict[str, Any]] = []
    for _, row in latest_stats.iterrows():
        predictions.append(
            {
                "player_id": int(row["player_id"]),
                "web_name": row["web_name"],
                "position_id": int(row["position_id"]),
                "team_id": int(row["team_id"]),
                "now_cost": float(row["now_cost"]),
                "expected_points_next_gw": float(row["expected_points_next_gw"]),
                "gameweek_id": int(target_gw),
            }
        )
    return predictions


def _ensure_table_and_marker_column() -> None:
    create_typed_table("recommended_squad", TABLE_SCHEMAS["recommended_squad"])
    with get_snowflake_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "ALTER TABLE recommended_squad ADD COLUMN IF NOT EXISTS backfill_method VARCHAR(100)"
        )
        conn.commit()
        cursor.close()


def _parse_timestamp_prefix(raw_value: str) -> datetime:
    try:
        return datetime.strptime(raw_value, "%Y-%m-%d %H:%M:%S")
    except ValueError as exc:
        raise ValueError(
            "recommended-at-prefix must use format 'YYYY-MM-DD HH:MM:SS'"
        ) from exc


def backfill_gameweek(
    feature_df: pd.DataFrame,
    target_gw: int,
    dry_run: bool,
    timestamp_prefix: str,
) -> Dict[str, Any]:
    LOGGER.info("Backfilling GW%s with point-in-time retraining...", target_gw)
    train_df, feature_cols, zscore_stats = _prepare_training_data(feature_df, target_gw)
    model, league_mean = _train_point_in_time_model(train_df, feature_cols)
    predictions = _build_predictions_for_target_gw(
        feature_df=feature_df,
        target_gw=target_gw,
        model=model,
        feature_cols=feature_cols,
        zscore_stats=zscore_stats,
        league_mean=league_mean,
    )
    squad = optimize_squad_task.fn(predictions)

    base_ts = _parse_timestamp_prefix(timestamp_prefix)
    recommended_at = (base_ts + timedelta(minutes=target_gw)).strftime("%Y-%m-%d %H:%M:%S")
    squad = add_recommendation_metadata(squad, recommended_at=recommended_at)
    for row in squad:
        row["gameweek_id"] = target_gw
        row["backfill_method"] = BACKFILL_METHOD

    loaded = 0
    if not dry_run:
        loaded = insert_typed_records("recommended_squad", squad)

    return {
        "gameweek_id": target_gw,
        "training_rows": int(len(train_df)),
        "prediction_rows": int(len(predictions)),
        "squad_rows": int(len(squad)),
        "loaded_rows": int(loaded),
        "backfill_method": BACKFILL_METHOD,
        "recommended_at": recommended_at,
        "dry_run": dry_run,
    }


def main() -> None:
    configure_logging()
    args = parse_args()
    target_gameweeks = sorted(set(args.gameweeks))
    timestamp_prefix = args.recommended_at_prefix or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    LOGGER.info("Fetching feature data once from Snowflake...")
    feature_df = fetch_training_data.fn()

    if not args.dry_run:
        _ensure_table_and_marker_column()

    results: List[Dict[str, Any]] = []
    for target_gw in target_gameweeks:
        result = backfill_gameweek(
            feature_df=feature_df,
            target_gw=target_gw,
            dry_run=args.dry_run,
            timestamp_prefix=timestamp_prefix,
        )
        results.append(result)
        LOGGER.info(
            "GW%s complete: loaded=%s squad_rows=%s method=%s",
            result["gameweek_id"],
            result["loaded_rows"],
            result["squad_rows"],
            result["backfill_method"],
        )

    LOGGER.info("Backfill summary:")
    for result in results:
        LOGGER.info("%s", result)


if __name__ == "__main__":
    main()
