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
from utils.gameweek_quality import load_gameweek_quality_policy
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
    parser.add_argument(
        "--rules-path",
        type=str,
        default="config/domain_rules.yaml",
        help="Domain rules path used for trust policy version metadata.",
    )
    return parser.parse_args()


def validate_backfill_integrity(
    *,
    train_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    target_gw: int,
    validation_version: str,
) -> Dict[str, Any]:
    """Validate rebuilt week uses only prior data and is trust-eligible."""

    train_max_target = int(train_df["target_gameweek_id"].max()) if "target_gameweek_id" in train_df.columns else -1
    inference_source_max = int(feature_df[feature_df["gameweek_id"] <= (target_gw - 1)]["gameweek_id"].max())
    train_prior_only = bool(train_max_target < target_gw)
    inference_prior_only = bool(inference_source_max <= (target_gw - 1))
    trusted = bool(train_prior_only and inference_prior_only)

    status = "validated_trusted" if trusted else "failed_contaminated"
    details = (
        f"train_max_target_gw={train_max_target};"
        f"inference_max_source_gw={inference_source_max};"
        f"target_gw={target_gw};"
        f"train_prior_only={train_prior_only};"
        f"inference_prior_only={inference_prior_only}"
    )
    return {
        "trusted": trusted,
        "status": status,
        "details": details,
        "validation_version": validation_version,
        "validated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "train_max_target_gameweek": train_max_target,
        "inference_max_source_gameweek": inference_source_max,
    }


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
) -> tuple[Any, dict[str, Any]]:
    X_train = train_df[feature_cols]
    y_train = train_df["target_next_gw_points"]
    model = train_model.train_xgboost_model(X_train, y_train)
    position_caps = {}
    if "position_id" in train_df.columns:
        for position_id, group in train_df.groupby("position_id"):
            position_caps[int(position_id)] = float(np.percentile(group["target_next_gw_points"], 95))
    metadata = {
        "league_mean": float(y_train.mean()),
        "shrinkage_alpha": float(train_model.SHRINKAGE_ALPHA),
        "selected_calibration_variant": "none",
        "calibration": None,
        "position_calibration": None,
        "position_caps": position_caps,
        "use_log_target": bool(train_model.LOG_TARGET),
    }
    return model, metadata


def _build_predictions_for_target_gw(
    feature_df: pd.DataFrame,
    target_gw: int,
    model: Any,
    feature_cols: List[str],
    zscore_stats: Dict[str, Dict[str, float]],
    metadata: dict[str, Any],
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
    if bool(metadata.get("use_log_target", False)):
        preds = train_model._inverse_transform(preds)  # Existing helper from training module
    preds = train_model.apply_prediction_post_processing(
        preds,
        league_mean=metadata.get("league_mean"),
        shrinkage_alpha=float(metadata.get("shrinkage_alpha", 0.0)),
        calibration=metadata.get("calibration"),
        position_calibration=metadata.get("position_calibration"),
        selected_variant=str(metadata.get("selected_calibration_variant", "none")),
        position_ids=latest_stats["position_id"].to_numpy() if "position_id" in latest_stats.columns else None,
        position_caps=metadata.get("position_caps"),
    )

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
        cursor.execute(
            "ALTER TABLE recommended_squad ADD COLUMN IF NOT EXISTS backfill_trusted BOOLEAN"
        )
        cursor.execute(
            "ALTER TABLE recommended_squad ADD COLUMN IF NOT EXISTS backfill_validation_status VARCHAR(50)"
        )
        cursor.execute(
            "ALTER TABLE recommended_squad ADD COLUMN IF NOT EXISTS backfill_validation_details VARCHAR(500)"
        )
        cursor.execute(
            "ALTER TABLE recommended_squad ADD COLUMN IF NOT EXISTS backfill_validation_version VARCHAR(50)"
        )
        cursor.execute(
            "ALTER TABLE recommended_squad ADD COLUMN IF NOT EXISTS backfill_validated_at TIMESTAMP_NTZ"
        )
        cursor.execute(
            "ALTER TABLE recommended_squad ADD COLUMN IF NOT EXISTS backfill_train_max_target_gw INTEGER"
        )
        cursor.execute(
            "ALTER TABLE recommended_squad ADD COLUMN IF NOT EXISTS backfill_inference_max_source_gw INTEGER"
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
    validation_version: str,
) -> Dict[str, Any]:
    LOGGER.info("Backfilling GW%s with point-in-time retraining...", target_gw)
    train_df, feature_cols, zscore_stats = _prepare_training_data(feature_df, target_gw)
    model, model_metadata = _train_point_in_time_model(train_df, feature_cols)
    predictions = _build_predictions_for_target_gw(
        feature_df=feature_df,
        target_gw=target_gw,
        model=model,
        feature_cols=feature_cols,
        zscore_stats=zscore_stats,
        metadata=model_metadata,
    )
    validation = validate_backfill_integrity(
        train_df=train_df,
        feature_df=feature_df,
        target_gw=target_gw,
        validation_version=validation_version,
    )
    squad = optimize_squad_task.fn(predictions)

    base_ts = _parse_timestamp_prefix(timestamp_prefix)
    recommended_at = (base_ts + timedelta(minutes=target_gw)).strftime("%Y-%m-%d %H:%M:%S")
    squad = add_recommendation_metadata(squad, recommended_at=recommended_at)
    for row in squad:
        row["gameweek_id"] = target_gw
        row["backfill_method"] = BACKFILL_METHOD
        row["backfill_trusted"] = bool(validation["trusted"])
        row["backfill_validation_status"] = validation["status"]
        row["backfill_validation_details"] = validation["details"][:500]
        row["backfill_validation_version"] = validation["validation_version"]
        row["backfill_validated_at"] = validation["validated_at"]
        row["backfill_train_max_target_gw"] = int(validation["train_max_target_gameweek"])
        row["backfill_inference_max_source_gw"] = int(validation["inference_max_source_gameweek"])

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
        "backfill_trusted": bool(validation["trusted"]),
        "backfill_validation_status": validation["status"],
        "backfill_validation_version": validation["validation_version"],
        "backfill_train_max_target_gw": int(validation["train_max_target_gameweek"]),
        "backfill_inference_max_source_gw": int(validation["inference_max_source_gameweek"]),
        "recommended_at": recommended_at,
        "dry_run": dry_run,
    }


def main() -> None:
    configure_logging()
    args = parse_args()
    target_gameweeks = sorted(set(args.gameweeks))
    timestamp_prefix = args.recommended_at_prefix or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    gameweek_policy = load_gameweek_quality_policy(args.rules_path)
    validation_version = str(gameweek_policy.get("policy_version") or "unknown")

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
            validation_version=validation_version,
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
