"""Node functions for the hyperparameter tuning agent."""

import json
import logging
import re
import sys
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Add pipeline root to path so we can import train_model utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from agents.tuning_state import TuningState, SEARCH_SPACE, DEFAULT_PARAMS
import train_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Target transform helpers
# ---------------------------------------------------------------------------

def _use_log_target() -> bool:
    return os.environ.get("TUNING_LOG_TARGET", "1").lower() in {"1", "true", "yes"}


def _transform_target(y: pd.Series) -> pd.Series:
    return np.sign(y) * np.log1p(np.abs(y))


def _inverse_transform(pred: np.ndarray) -> np.ndarray:
    return np.sign(pred) * np.expm1(np.abs(pred))


# ---------------------------------------------------------------------------
# Node: fetch_data
# ---------------------------------------------------------------------------

def fetch_data(state: TuningState) -> dict:
    """Fetch and prepare data from Snowflake. Runs once at the start."""
    logger.info("=" * 60)
    logger.info("FETCH DATA")
    logger.info("=" * 60)

    df = train_model.fetch_training_data()
    df = train_model.engineer_features(df)

    train_df, holdout_df = train_model.split_train_holdout(
        df, train_model.HOLDOUT_GAMEWEEKS
    )

    global_stats = train_model.compute_global_stats(train_df)
    train_df = train_model.add_z_scores(train_df, global_stats)
    holdout_df = train_model.add_z_scores(holdout_df, global_stats)

    features = train_model.select_features()

    # Ensure all feature columns exist
    for feature in features:
        if feature not in train_df.columns:
            train_df[feature] = 0
            holdout_df[feature] = 0

    logger.info(
        f"Data ready: {len(train_df)} train, {len(holdout_df)} holdout samples, "
        f"{len(features)} features"
    )

    return {
        "train_df": train_df,
        "holdout_df": holdout_df,
        "features": features,
        "global_stats": global_stats,
        "experiments": [],
        "current_params": DEFAULT_PARAMS.copy(),
        "best_params": DEFAULT_PARAMS.copy(),
        "best_holdout_mae": float("inf"),
        "iteration": 0,
        "converged": False,
        "analysis": "",
    }


# ---------------------------------------------------------------------------
# Node: train_evaluate
# ---------------------------------------------------------------------------

def train_evaluate(state: TuningState) -> dict:
    """Train XGBoost with current_params and evaluate on holdout."""
    iteration = state["iteration"] + 1
    params = state["current_params"]

    logger.info("=" * 60)
    logger.info(f"ITERATION {iteration}: TRAIN & EVALUATE")
    logger.info("=" * 60)
    logger.info(f"Params: {json.dumps(params, indent=2)}")

    train_df = state["train_df"]
    holdout_df = state["holdout_df"]
    features = state["features"]

    X_train = train_df[features]
    y_train = train_df["target_next_gw_points"]
    X_holdout = holdout_df[features]
    y_holdout = holdout_df["target_next_gw_points"]

    # Separate XGBoost params from post-processing params
    shrinkage_alpha = params.get("shrinkage_alpha", 0.0)
    calibration_strength = params.get("calibration_strength", 0.8)

    xgb_params = {
        k: v
        for k, v in params.items()
        if k not in ("shrinkage_alpha", "calibration_strength")
    }

    model = xgb.XGBRegressor(
        **xgb_params,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
        eval_metric="mae",
    )

    # Cross-validation
    use_log_target = _use_log_target()
    y_train_trans = _transform_target(y_train) if use_log_target else y_train
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    for train_idx, val_idx in tscv.split(X_train):
        model_cv = xgb.XGBRegressor(**model.get_params())
        model_cv.fit(X_train.iloc[train_idx], y_train_trans.iloc[train_idx])
        preds = model_cv.predict(X_train.iloc[val_idx])
        if use_log_target:
            preds = _inverse_transform(preds)
        cv_scores.append(mean_absolute_error(y_train.iloc[val_idx], preds))
    cv_mae = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))
    logger.info(f"CV MAE: {cv_mae:.4f} (+/- {cv_std:.4f})")

    # Train on full training set
    model.fit(X_train, y_train_trans)

    # Holdout evaluation with shrinkage + calibration
    league_mean = float(y_train.mean())
    holdout_pred = model.predict(X_holdout)
    if use_log_target:
        holdout_pred = _inverse_transform(holdout_pred)
    holdout_pred = train_model.apply_shrinkage(holdout_pred, league_mean, shrinkage_alpha)

    if calibration_strength > 0:
        a, b = train_model.fit_calibration(y_holdout.to_numpy(), holdout_pred)
        a, b = train_model.blend_calibration(a, b, calibration_strength)
        holdout_pred = train_model.apply_calibration(holdout_pred, a, b)

    holdout_pred = np.maximum(holdout_pred, 0)

    holdout_mae = float(mean_absolute_error(y_holdout, holdout_pred))
    holdout_rmse = float(np.sqrt(mean_squared_error(y_holdout, holdout_pred)))
    holdout_bias = float((y_holdout - holdout_pred).mean())

    logger.info(f"Holdout MAE: {holdout_mae:.4f}  RMSE: {holdout_rmse:.4f}  Bias: {holdout_bias:.4f}")

    # Record experiment
    experiment = {
        "iteration": iteration,
        "params": params.copy(),
        "cv_mae": cv_mae,
        "holdout_mae": holdout_mae,
        "holdout_rmse": holdout_rmse,
        "holdout_bias": holdout_bias,
    }

    experiments = list(state["experiments"]) + [experiment]

    # Track best
    best_params = state["best_params"]
    best_holdout_mae = state["best_holdout_mae"]
    if holdout_mae < best_holdout_mae:
        best_params = params.copy()
        best_holdout_mae = holdout_mae
        logger.info(f"*** New best holdout MAE: {holdout_mae:.4f} ***")

    return {
        "experiments": experiments,
        "iteration": iteration,
        "best_params": best_params,
        "best_holdout_mae": best_holdout_mae,
    }


# ---------------------------------------------------------------------------
# Node: analyze_suggest (LLM)
# ---------------------------------------------------------------------------

def analyze_suggest(state: TuningState) -> dict:
    """Use Claude to analyze experiment history and suggest next hyperparameters."""
    logger.info("=" * 60)
    logger.info("LLM ANALYSIS & SUGGESTION")
    logger.info("=" * 60)

    experiments = state["experiments"]
    best_params = state["best_params"]
    best_holdout_mae = state["best_holdout_mae"]

    # Build experiment history table
    history_lines = []
    for exp in experiments:
        p = exp["params"]
        history_lines.append(
            f"  Iter {exp['iteration']}: "
            f"CV_MAE={exp['cv_mae']:.4f} "
            f"Holdout_MAE={exp['holdout_mae']:.4f} "
            f"RMSE={exp['holdout_rmse']:.4f} "
            f"Bias={exp['holdout_bias']:.4f} | "
        f"n_est={p['n_estimators']} depth={p['max_depth']} "
        f"lr={p['learning_rate']} sub={p['subsample']} "
        f"col={p['colsample_bytree']} alpha={p['reg_alpha']} "
        f"lambda={p['reg_lambda']} min_child={p['min_child_weight']} "
        f"gamma={p['gamma']} delta={p['max_delta_step']} "
        f"shrink={p['shrinkage_alpha']} calib={p['calibration_strength']}"
        )
    history_text = "\n".join(history_lines)

    # Build search space description
    space_lines = []
    for param, bounds in SEARCH_SPACE.items():
        space_lines.append(
            f"  {param}: [{bounds['min']}, {bounds['max']}] ({bounds['type']})"
        )
    space_text = "\n".join(space_lines)

    prompt = f"""You are an XGBoost hyperparameter optimization expert for a Fantasy Premier League (FPL) points prediction model.

## Search Space
{space_text}

## Experiment History
{history_text}

## Current Best
  Holdout MAE: {best_holdout_mae:.4f}
  Params: {json.dumps(best_params, indent=2)}

## Task
Analyze the experiment history:
1. Identify which parameter changes improved or worsened performance.
2. Note any patterns (e.g., overfitting signals when CV_MAE << Holdout_MAE).
3. Propose the NEXT set of hyperparameters to try, aiming to minimize holdout MAE.

Respond with:
- A brief analysis paragraph (2-4 sentences).
- Then a JSON block with the exact next parameters to try.

IMPORTANT: Return ALL 12 parameters in the JSON. Use this exact format:
```json
{{
  "n_estimators": <int>,
  "max_depth": <int>,
  "learning_rate": <float>,
  "subsample": <float>,
  "colsample_bytree": <float>,
  "reg_alpha": <float>,
  "reg_lambda": <float>,
  "min_child_weight": <float>,
  "gamma": <float>,
  "max_delta_step": <float>,
  "shrinkage_alpha": <float>,
  "calibration_strength": <float>
}}
```"""

    llm = ChatOpenAI(
        model=os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5"),
        openai_api_key=os.environ["OPENROUTER_API_KEY"],
        openai_api_base=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        max_tokens=1024,
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    response_text = response.content

    logger.info(f"LLM response:\n{response_text}")

    # Parse JSON from response
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if not json_match:
        # Fallback: try to find raw JSON object
        json_match = re.search(r"\{[^{}]*\"n_estimators\"[^{}]*\}", response_text, re.DOTALL)
        if json_match:
            raw_json = json_match.group(0)
        else:
            logger.warning("Could not parse LLM suggestion, reusing best params")
            return {"analysis": response_text, "current_params": best_params.copy()}
    else:
        raw_json = json_match.group(1)

    suggested = json.loads(raw_json)

    # Clamp to search space bounds and cast types
    clamped = {}
    for param, bounds in SEARCH_SPACE.items():
        val = suggested.get(param, DEFAULT_PARAMS[param])
        if bounds["type"] == "int":
            val = int(round(val))
        else:
            val = float(val)
        val = max(bounds["min"], min(bounds["max"], val))
        clamped[param] = val

    logger.info(f"Next params (clamped): {json.dumps(clamped, indent=2)}")

    # Extract analysis text (everything before the JSON block)
    analysis = response_text.split("```")[0].strip() if "```" in response_text else response_text

    return {"analysis": analysis, "current_params": clamped}


# ---------------------------------------------------------------------------
# Node: check_convergence (conditional edge)
# ---------------------------------------------------------------------------

def check_convergence(state: TuningState) -> str:
    """Decide whether to continue or save the best model."""
    iteration = state["iteration"]
    max_iterations = state["max_iterations"]
    experiments = state["experiments"]

    # Condition 1: hit max iterations
    if iteration >= max_iterations:
        logger.info(f"Reached max iterations ({max_iterations}). Stopping.")
        return "save_best"

    # Condition 2: no improvement in last 3 iterations
    if len(experiments) >= 4:
        recent = experiments[-3:]
        best_recent = min(e["holdout_mae"] for e in recent)
        best_before = min(e["holdout_mae"] for e in experiments[:-3])
        if best_recent >= best_before:
            logger.info("No improvement in last 3 iterations. Stopping.")
            return "save_best"

    return "train_evaluate"


# ---------------------------------------------------------------------------
# Node: save_best
# ---------------------------------------------------------------------------

def save_best(state: TuningState) -> dict:
    """Retrain with best params and save the model."""
    logger.info("=" * 60)
    logger.info("SAVING BEST MODEL")
    logger.info("=" * 60)

    best_params = state["best_params"]
    train_df = state["train_df"]
    holdout_df = state["holdout_df"]
    features = state["features"]
    global_stats = state["global_stats"]

    logger.info(f"Best params: {json.dumps(best_params, indent=2)}")
    logger.info(f"Best holdout MAE: {state['best_holdout_mae']:.4f}")

    X_train = train_df[features]
    y_train = train_df["target_next_gw_points"]
    X_holdout = holdout_df[features]
    y_holdout = holdout_df["target_next_gw_points"]

    shrinkage_alpha = best_params.get("shrinkage_alpha", 0.0)
    calibration_strength = best_params.get("calibration_strength", 0.8)

    xgb_params = {
        k: v
        for k, v in best_params.items()
        if k not in ("shrinkage_alpha", "calibration_strength")
    }

    model = xgb.XGBRegressor(
        **xgb_params,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
        eval_metric="mae",
    )
    use_log_target = _use_log_target()
    y_train_trans = _transform_target(y_train) if use_log_target else y_train
    model.fit(X_train, y_train_trans)

    # Evaluate on training set for metrics
    train_pred = model.predict(X_train)
    if use_log_target:
        train_pred = _inverse_transform(train_pred)
    train_mae = float(mean_absolute_error(y_train, train_pred))
    train_rmse = float(np.sqrt(mean_squared_error(y_train, train_pred)))

    # Holdout evaluation
    league_mean = float(y_train.mean())
    holdout_pred = model.predict(X_holdout)
    if use_log_target:
        holdout_pred = _inverse_transform(holdout_pred)
    holdout_pred = train_model.apply_shrinkage(holdout_pred, league_mean, shrinkage_alpha)

    calibration = None
    if calibration_strength > 0:
        a, b = train_model.fit_calibration(y_holdout.to_numpy(), holdout_pred)
        a, b = train_model.blend_calibration(a, b, calibration_strength)
        holdout_pred = train_model.apply_calibration(holdout_pred, a, b)
        calibration = {"a": a, "b": b}

    holdout_pred = np.maximum(holdout_pred, 0)

    holdout_mae = float(mean_absolute_error(y_holdout, holdout_pred))
    holdout_rmse = float(np.sqrt(mean_squared_error(y_holdout, holdout_pred)))
    holdout_bias = float((y_holdout - holdout_pred).mean())

    # Build feature importance
    feature_importance = pd.DataFrame(
        {"feature": features, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    # Position caps
    position_caps = {}
    if "position_id" in train_df.columns:
        for position_id, group in train_df.groupby("position_id"):
            position_caps[int(position_id)] = float(
                np.percentile(group["target_next_gw_points"], 95)
            )

    from sklearn.metrics import r2_score

    metrics = {
        "mae": train_mae,
        "rmse": train_rmse,
        "r2": float(r2_score(y_train, train_pred)),
        "bias": float((y_train - train_pred).mean()),
        "feature_importance": feature_importance,
        "holdout_mae": holdout_mae,
        "holdout_rmse": holdout_rmse,
        "holdout_bias": holdout_bias,
        "zscore_stats": global_stats,
        "feature_cols": features,
        "position_caps": position_caps,
        "train_target_stats": {
            "mean": float(y_train.mean()),
            "std": float(y_train.std()),
            "min": float(y_train.min()),
            "max": float(y_train.max()),
        },
        "training_window": {
            "min_gameweek": int(train_df["target_gameweek_id"].min()),
            "max_gameweek": int(train_df["target_gameweek_id"].max()),
        },
        "metadata": {
            "league_mean": league_mean,
            "shrinkage_alpha": shrinkage_alpha,
            "reg_alpha": best_params.get("reg_alpha", 0.1),
            "reg_lambda": best_params.get("reg_lambda", 1.0),
            "max_depth": best_params.get("max_depth", 5),
            "holdout_gameweeks": train_model.HOLDOUT_GAMEWEEKS,
            "calibration": calibration,
        },
    }

    train_model.save_model(model, metrics, output_path="logs/model.bin")

    # Print experiment summary
    logger.info("=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    logger.info(
        f"{'Iter':>4} | {'CV MAE':>8} | {'Hold MAE':>9} | {'Hold RMSE':>10} | {'Bias':>7} | Best Params Changed"
    )
    logger.info("-" * 70)
    for exp in state["experiments"]:
        logger.info(
            f"{exp['iteration']:4d} | {exp['cv_mae']:8.4f} | {exp['holdout_mae']:9.4f} | "
            f"{exp['holdout_rmse']:10.4f} | {exp['holdout_bias']:7.4f}"
        )

    logger.info(f"\nBest holdout MAE: {state['best_holdout_mae']:.4f}")

    return {"converged": True}
