"""Run staged local-only ranking experiments against the latest snapshot."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import train_model
from utils.gameweek_quality import load_gameweek_quality_policy


EXPERIMENT_LOG_DIR = Path("logs/ranking_experiments")
EXPERIMENT_DOC_DIR = Path("docs/ranking_experiments")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staged local-only ranking experiments")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=list(train_model.EXPERIMENT_SEQUENCE),
        help="Experiment variants to run in order",
    )
    parser.add_argument("--rules-path", default="config/domain_rules.yaml")
    parser.add_argument("--local-path", default=None)
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def _ensure_dirs() -> None:
    EXPERIMENT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_DOC_DIR.mkdir(parents=True, exist_ok=True)


def _write_markdown_note(variant: str, report: dict[str, Any], publication_status: dict[str, Any]) -> None:
    summary = report.get("summary") or {}
    baseline = report.get("required_baseline_comparison") or {}
    metric_deltas = baseline.get("metric_deltas") or {}
    per_week = report.get("per_week") or []
    misses = []
    for week in per_week:
        for miss in week.get("largest_player_misses", []):
            misses.append(
                {
                    "gameweek_id": week.get("gameweek_id"),
                    "web_name": miss.get("web_name", str(miss.get("player_id"))),
                    "abs_error": float(miss.get("abs_error", 0.0)),
                }
            )
    misses = sorted(misses, key=lambda row: row["abs_error"], reverse=True)[:5]

    lines = [
        f"# Ranking Experiment: {variant}",
        "",
        f"- Forward publish ready: `{publication_status['ready']}`",
        f"- Failed gates: `{publication_status['reasons']}`",
        f"- Recent validation gameweeks: `{report.get('recent_validation_gameweeks', [])}`",
        f"- Top-k hit rate mean: `{summary.get('top_k_hit_rate_mean', 0.0):.3f}`",
        f"- Rank correlation mean: `{summary.get('rank_correlation_mean', 0.0):.3f}`",
        f"- Selected XI regret mean: `{summary.get('selected_xi_regret_mean', 0.0):.3f}`",
        f"- Required baseline: `{report.get('required_baseline', 'unknown')}`",
        f"- Baseline gate passed: `{baseline.get('baseline_gate_passed', False)}`",
        f"- Baseline deltas: `{metric_deltas}`",
        "",
        "## Worst misses",
        "",
    ]
    for miss in misses:
        lines.append(f"- GW{miss['gameweek_id']}: {miss['web_name']} abs_error={miss['abs_error']:.2f}")

    (EXPERIMENT_DOC_DIR / f"{variant}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_variant(df, variant: str, rules_path: str) -> dict[str, Any]:
    rules = train_model.load_model_rules(rules_path)
    gameweek_policy = load_gameweek_quality_policy(rules_path)

    engineered = train_model.engineer_features(df)
    train_df, holdout_df = train_model.split_train_holdout(
        engineered,
        train_model.HOLDOUT_GAMEWEEKS,
        gameweek_policy=gameweek_policy,
    )
    global_stats = train_model.compute_global_stats(train_df)
    train_df = train_model.add_z_scores(train_df, global_stats)
    holdout_df = train_model.add_z_scores(holdout_df, global_stats)

    feature_cols = train_model.select_features(variant)
    for feature in feature_cols:
        if feature not in train_df.columns:
            train_df[feature] = 0
            holdout_df[feature] = 0

    model = train_model.train_prediction_bundle(train_df, feature_cols, variant=variant, use_log_target=train_model.LOG_TARGET)
    y_train = train_df["target_next_gw_points"]
    y_holdout = holdout_df["target_next_gw_points"]
    league_mean = float(y_train.mean())

    train_pred_raw = train_model.predict_prediction_bundle(model, train_df, feature_cols, use_log_target=train_model.LOG_TARGET)
    holdout_pred_raw = train_model.predict_prediction_bundle(model, holdout_df, feature_cols, use_log_target=train_model.LOG_TARGET)
    train_pred_raw = train_model.apply_shrinkage(train_pred_raw, league_mean, train_model.SHRINKAGE_ALPHA)
    holdout_pred_raw = train_model.apply_shrinkage(holdout_pred_raw, league_mean, train_model.SHRINKAGE_ALPHA)

    calibration_result = train_model.select_calibration_variant(
        y_train=y_train.to_numpy(),
        train_pred=np.asarray(train_pred_raw),
        y_holdout=y_holdout.to_numpy(),
        holdout_pred=np.asarray(holdout_pred_raw),
        train_position_ids=train_df["position_id"].to_numpy() if "position_id" in train_df.columns else None,
        holdout_position_ids=holdout_df["position_id"].to_numpy() if "position_id" in holdout_df.columns else None,
        strength=train_model.CALIBRATION_STRENGTH if train_model.ENABLE_CALIBRATION else 0.0,
        min_samples=train_model.POSITION_CALIBRATION_MIN_SAMPLES,
    )
    holdout_pred = train_model.apply_prediction_post_processing(
        np.asarray(holdout_pred_raw),
        calibration=calibration_result["calibration"],
        position_calibration=calibration_result["position_calibration"],
        selected_variant=calibration_result["selected_variant"],
        position_ids=holdout_df["position_id"].to_numpy() if "position_id" in holdout_df.columns else None,
    )

    report = train_model.build_weekly_backtest_report(
        holdout_df,
        y_holdout.to_numpy(),
        holdout_pred,
        gameweek_policy=gameweek_policy,
        model_rules=rules,
    )
    report = train_model.add_baseline_comparison_to_report(
        report,
        holdout_df.assign(predicted_points=holdout_pred),
        y_holdout.to_numpy(),
        gameweek_policy=gameweek_policy,
        model_rules=rules,
    )
    publication_status = train_model.evaluate_publication_readiness(
        report,
        calibration_report=calibration_result["calibration_report"],
        model_rules=rules,
    )

    result = {
        "variant": variant,
        "holdout_mae": float(mean_absolute_error(y_holdout, holdout_pred)),
        "holdout_rmse": float(np.sqrt(mean_squared_error(y_holdout, holdout_pred))),
        "holdout_bias": float((y_holdout - holdout_pred).mean()),
        "report": report,
        "publication_status": publication_status,
    }
    return result


def main() -> int:
    args = _parse_args()
    _ensure_dirs()

    os.environ["DOMAIN_RULES_PATH"] = args.rules_path
    os.environ["TRAINING_DATA_SOURCE"] = "local"
    os.environ["TRAINING_DATA_POLICY"] = "LOCAL_ONLY"
    if args.local_path:
        os.environ["TRAINING_DATA_LOCAL_PATH"] = args.local_path

    df = train_model.fetch_training_data()
    results = []
    for variant in args.variants:
        os.environ["EXPERIMENT_VARIANT"] = variant
        result = _run_variant(df, variant, args.rules_path)
        results.append(
            {
                "variant": variant,
                "holdout_mae": result["holdout_mae"],
                "holdout_rmse": result["holdout_rmse"],
                "holdout_bias": result["holdout_bias"],
                "publication_status": result["publication_status"],
                "required_baseline_comparison": result["report"].get("required_baseline_comparison", {}),
            }
        )
        (EXPERIMENT_LOG_DIR / f"{variant}.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        _write_markdown_note(variant, result["report"], result["publication_status"])
        if result["publication_status"]["ready"] and not args.keep_going:
            break

    print(json.dumps({"results": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
