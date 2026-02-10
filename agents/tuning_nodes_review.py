"""Dual-agent tuning nodes with reviewer gates."""

import json
import logging
import os
import re
import sys
from typing import Tuple

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
from agents.tuning_nodes import fetch_data as base_fetch_data, save_best
import train_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is None or val == "":
        return float(default)
    try:
        return float(val)
    except ValueError:
        logger.warning("Invalid %s=%r, using default %s", name, val, default)
        return float(default)


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None or val == "":
        return int(default)
    try:
        return int(val)
    except ValueError:
        logger.warning("Invalid %s=%r, using default %s", name, val, default)
        return int(default)


def _compute_group_bias(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, group_col: str) -> list[dict]:
    report_df = df.copy()
    report_df["actual_points"] = y_true
    report_df["predicted_points"] = y_pred
    report_df["absolute_error"] = np.abs(y_true - y_pred)
    report_df["error_bias"] = y_true - y_pred

    summary = (
        report_df.groupby(group_col)
        .agg(
            count=(group_col, "count"),
            avg_predicted_points=("predicted_points", "mean"),
            avg_actual_points=("actual_points", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
            mean_error_bias=("error_bias", "mean"),
        )
        .reset_index()
    )

    results = []
    for _, row in summary.iterrows():
        results.append(
            {
                "group": row[group_col],
                "count": int(row["count"]),
                "avg_predicted_points": float(row["avg_predicted_points"]),
                "avg_actual_points": float(row["avg_actual_points"]),
                "mae": float(row["mean_absolute_error"]),
                "bias": float(row["mean_error_bias"]),
            }
        )
    return results


def _compute_caps(train_df: pd.DataFrame, group_col: str, target_col: str, min_count: int = 1) -> dict:
    caps = {}
    for key, group in train_df.groupby(group_col):
        if len(group) < min_count:
            continue
        caps[key] = float(np.percentile(group[target_col], 95))
    return caps


def _exceed_rate(holdout_df: pd.DataFrame, holdout_pred: np.ndarray, caps: dict, key_col: str) -> Tuple[float, int]:
    if key_col not in holdout_df.columns or not caps:
        return 0.0, 0
    keys = holdout_df[key_col].to_numpy()
    cap_values = np.array([caps.get(k) for k in keys], dtype=float)
    mask = ~np.isnan(cap_values)
    if not mask.any():
        return 0.0, 0
    exceeds = holdout_pred[mask] > cap_values[mask]
    return float(exceeds.mean()), int(mask.sum())


def _compute_sanity_metrics(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    y_holdout: pd.Series,
    holdout_pred: np.ndarray,
) -> dict:
    actual = y_holdout.to_numpy()
    actual_stats = {
        "mean": float(np.mean(actual)),
        "std": float(np.std(actual)),
        "min": float(np.min(actual)),
        "max": float(np.max(actual)),
    }
    pred_stats = {
        "mean": float(np.mean(holdout_pred)),
        "std": float(np.std(holdout_pred)),
        "min": float(np.min(holdout_pred)),
        "max": float(np.max(holdout_pred)),
    }

    position_caps = {}
    position_exceed_rate = None
    position_exceed_n = 0
    if "position_id" in train_df.columns and "position_id" in holdout_df.columns:
        position_caps = _compute_caps(train_df, "position_id", "target_next_gw_points")
        position_exceed_rate, position_exceed_n = _exceed_rate(
            holdout_df, holdout_pred, position_caps, "position_id"
        )

    min_hist = _env_int("TUNING_PLAYER_MIN_HISTORY", 10)
    player_caps = {}
    player_exceed_rate = None
    player_exceed_n = 0
    if "player_id" in train_df.columns and "player_id" in holdout_df.columns:
        player_caps = _compute_caps(train_df, "player_id", "target_next_gw_points", min_count=min_hist)
        player_exceed_rate, player_exceed_n = _exceed_rate(
            holdout_df, holdout_pred, player_caps, "player_id"
        )

    minutes_bias = []
    if "minutes_played" in holdout_df.columns:
        context = holdout_df[["player_id", "minutes_played"]].copy()
        context["minutes_band"] = pd.cut(
            context["minutes_played"],
            bins=[-1, 30, 60, 1_000_000],
            labels=["0-30", "31-60", "61-90"],
        )
        minutes_bias = _compute_group_bias(context, actual, holdout_pred, "minutes_band")

    return {
        "actual_stats": actual_stats,
        "pred_stats": pred_stats,
        "position_caps": position_caps,
        "position_exceed_rate": position_exceed_rate,
        "position_exceed_n": position_exceed_n,
        "player_caps_count": len(player_caps),
        "player_exceed_rate": player_exceed_rate,
        "player_exceed_n": player_exceed_n,
        "minutes_band_bias": minutes_bias,
    }


def _evaluate_gates(
    holdout_mae: float,
    holdout_rmse: float,
    holdout_bias: float,
    sanity_metrics: dict,
    best_holdout_mae: float,
    best_holdout_rmse: float,
) -> Tuple[bool, dict]:
    bias_sd_mult = _env_float("TUNING_BIAS_SD_MULT", 0.25)
    pos_exceed_pct = _env_float("TUNING_POS_P95_EXCEED_PCT", 0.05)
    player_exceed_pct = _env_float("TUNING_PLAYER_P95_EXCEED_PCT", 0.05)
    rmse_worse_pct = _env_float("TUNING_RMSE_WORSE_PCT", 0.05)

    actual_std = sanity_metrics.get("actual_stats", {}).get("std", 0.0)
    bias_gate = abs(holdout_bias) <= bias_sd_mult * actual_std if actual_std > 0 else True

    pos_rate = sanity_metrics.get("position_exceed_rate")
    if pos_rate is None:
        pos_gate = True
    else:
        pos_gate = pos_rate <= pos_exceed_pct

    player_rate = sanity_metrics.get("player_exceed_rate")
    if player_rate is None:
        player_gate = True
    else:
        player_gate = player_rate <= player_exceed_pct

    if best_holdout_rmse < float("inf"):
        rmse_gate = holdout_rmse <= best_holdout_rmse * (1.0 + rmse_worse_pct)
    else:
        rmse_gate = True

    if best_holdout_mae < float("inf"):
        mae_gate = holdout_mae <= best_holdout_mae * (1.0 + 0.01)
    else:
        mae_gate = True

    gates = {
        "bias_gate": bias_gate,
        "pos_cap_gate": pos_gate,
        "player_cap_gate": player_gate,
        "rmse_gate": rmse_gate,
        "mae_gate": mae_gate,
        "bias_sd_mult": bias_sd_mult,
        "pos_exceed_pct": pos_exceed_pct,
        "player_exceed_pct": player_exceed_pct,
        "rmse_worse_pct": rmse_worse_pct,
    }

    gates_passed = all([bias_gate, pos_gate, player_gate, rmse_gate, mae_gate])
    return gates_passed, gates


# ---------------------------------------------------------------------------
# Node: fetch_data
# ---------------------------------------------------------------------------

def fetch_data(state: TuningState) -> dict:
    """Fetch data and initialize reviewer-specific fields."""
    base = base_fetch_data(state)
    base.update(
        {
            "best_holdout_rmse": float("inf"),
            "sanity_metrics": {},
            "gates_passed": False,
            "gate_details": {},
            "review_decision": "",
            "review_feedback": "",
            "reject_count": 0,
        }
    )
    return base


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

    shrinkage_alpha = params.get("shrinkage_alpha", 0.0)
    calibration_strength = params.get("calibration_strength", 0.8)

    xgb_params = {
        k: v for k, v in params.items() if k not in ("shrinkage_alpha", "calibration_strength")
    }

    model = xgb.XGBRegressor(
        **xgb_params,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
        eval_metric="mae",
    )

    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    cv_mae = float(-cv_scores.mean())
    logger.info(f"CV MAE: {cv_mae:.4f} (+/- {cv_scores.std():.4f})")

    model.fit(X_train, y_train)

    league_mean = float(y_train.mean())
    holdout_pred = model.predict(X_holdout)
    holdout_pred = train_model.apply_shrinkage(holdout_pred, league_mean, shrinkage_alpha)

    if calibration_strength > 0:
        a, b = train_model.fit_calibration(y_holdout.to_numpy(), holdout_pred)
        a, b = train_model.blend_calibration(a, b, calibration_strength)
        holdout_pred = train_model.apply_calibration(holdout_pred, a, b)

    holdout_pred = np.maximum(holdout_pred, 0)

    holdout_mae = float(mean_absolute_error(y_holdout, holdout_pred))
    holdout_rmse = float(np.sqrt(mean_squared_error(y_holdout, holdout_pred)))
    holdout_bias = float((y_holdout - holdout_pred).mean())

    logger.info(
        f"Holdout MAE: {holdout_mae:.4f}  RMSE: {holdout_rmse:.4f}  Bias: {holdout_bias:.4f}"
    )

    sanity_metrics = _compute_sanity_metrics(train_df, holdout_df, y_holdout, holdout_pred)
    gates_passed, gate_details = _evaluate_gates(
        holdout_mae,
        holdout_rmse,
        holdout_bias,
        sanity_metrics,
        state["best_holdout_mae"],
        state.get("best_holdout_rmse", float("inf")),
    )

    experiment = {
        "iteration": iteration,
        "params": params.copy(),
        "cv_mae": cv_mae,
        "holdout_mae": holdout_mae,
        "holdout_rmse": holdout_rmse,
        "holdout_bias": holdout_bias,
        "gates_passed": gates_passed,
        "gate_details": gate_details,
    }

    experiments = list(state["experiments"]) + [experiment]

    best_params = state["best_params"]
    best_holdout_mae = state["best_holdout_mae"]
    best_holdout_rmse = state.get("best_holdout_rmse", float("inf"))

    if gates_passed and holdout_mae < best_holdout_mae:
        best_params = params.copy()
        best_holdout_mae = holdout_mae
        best_holdout_rmse = holdout_rmse
        logger.info(f"*** New best feasible holdout MAE: {holdout_mae:.4f} ***")
    elif not gates_passed:
        logger.info("Gates failed. Not updating best params.")

    return {
        "experiments": experiments,
        "iteration": iteration,
        "best_params": best_params,
        "best_holdout_mae": best_holdout_mae,
        "best_holdout_rmse": best_holdout_rmse,
        "sanity_metrics": sanity_metrics,
        "gates_passed": gates_passed,
        "gate_details": gate_details,
    }


# ---------------------------------------------------------------------------
# Node: propose_params (LLM)
# ---------------------------------------------------------------------------

def propose_params(state: TuningState) -> dict:
    """Optimizer LLM proposes next hyperparameters."""
    logger.info("=" * 60)
    logger.info("LLM PROPOSE PARAMS")
    logger.info("=" * 60)

    experiments = state["experiments"]
    best_params = state["best_params"]
    best_holdout_mae = state["best_holdout_mae"]
    sanity_metrics = state.get("sanity_metrics", {})
    review_feedback = state.get("review_feedback", "")

    history_lines = []
    for exp in experiments:
        p = exp["params"]
        history_lines.append(
            f"  Iter {exp['iteration']}: "
            f"CV_MAE={exp['cv_mae']:.4f} "
            f"Holdout_MAE={exp['holdout_mae']:.4f} "
            f"RMSE={exp['holdout_rmse']:.4f} "
            f"Bias={exp['holdout_bias']:.4f} "
            f"Gates={'PASS' if exp.get('gates_passed') else 'FAIL'} | "
            f"n_est={p['n_estimators']} depth={p['max_depth']} "
            f"lr={p['learning_rate']} sub={p['subsample']} "
            f"col={p['colsample_bytree']} alpha={p['reg_alpha']} "
            f"lambda={p['reg_lambda']} shrink={p['shrinkage_alpha']} "
            f"calib={p['calibration_strength']}"
        )
    history_text = "\n".join(history_lines)

    space_lines = []
    for param, bounds in SEARCH_SPACE.items():
        space_lines.append(f"  {param}: [{bounds['min']}, {bounds['max']}] ({bounds['type']})")
    space_text = "\n".join(space_lines)

    prompt = f"""You are optimizing XGBoost hyperparameters for an FPL next-gameweek points model.

## Search Space
{space_text}

## Experiment History
{history_text}

## Current Best (feasible)
  Holdout MAE: {best_holdout_mae:.4f}
  Params: {json.dumps(best_params, indent=2)}

## Sanity Metrics (latest)
{json.dumps(sanity_metrics, indent=2)}

## Reviewer Feedback (latest)
{review_feedback}

## Task
Propose the NEXT hyperparameters to try. Prioritize holdout MAE while keeping predictions realistic and reliable.
Respect bias-variance tradeoff signals (e.g., CV_MAE << Holdout_MAE suggests overfit).
Use best judgment for feasibility and sensible outputs.

Respond with:
- A brief analysis paragraph (2-4 sentences).
- Then a JSON block with the exact next parameters to try.

IMPORTANT: Return ALL 9 parameters in the JSON. Use this exact format:
```json
{{
  \"n_estimators\": <int>,
  \"max_depth\": <int>,
  \"learning_rate\": <float>,
  \"subsample\": <float>,
  \"colsample_bytree\": <float>,
  \"reg_alpha\": <float>,
  \"reg_lambda\": <float>,
  \"shrinkage_alpha\": <float>,
  \"calibration_strength\": <float>
}}
```
"""

    llm = ChatOpenAI(
        model=os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5"),
        openai_api_key=os.environ["OPENROUTER_API_KEY"],
        openai_api_base=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        max_tokens=1024,
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    response_text = response.content

    logger.info(f"LLM response:\n{response_text}")

    json_match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if not json_match:
        json_match = re.search(r"\{[^{}]*\"n_estimators\"[^{}]*\}", response_text, re.DOTALL)
        if json_match:
            raw_json = json_match.group(0)
        else:
            logger.warning("Could not parse LLM suggestion, reusing best params")
            return {"analysis": response_text, "current_params": best_params.copy()}
    else:
        raw_json = json_match.group(1)

    suggested = json.loads(raw_json)

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

    analysis = response_text.split("```")[0].strip() if "```" in response_text else response_text

    return {"analysis": analysis, "current_params": clamped}


# ---------------------------------------------------------------------------
# Node: review_candidate (LLM + gates)
# ---------------------------------------------------------------------------

def review_candidate(state: TuningState) -> dict:
    """Reviewer evaluates latest results for realism and feasibility."""
    logger.info("=" * 60)
    logger.info("REVIEWER CHECK")
    logger.info("=" * 60)

    experiments = state["experiments"]
    latest = experiments[-1] if experiments else {}
    sanity_metrics = state.get("sanity_metrics", {})
    gate_details = state.get("gate_details", {})
    gates_passed = state.get("gates_passed", False)

    prompt = f"""You are a strict reviewer for an FPL prediction model.

## Latest Metrics
{json.dumps(latest, indent=2)}

## Sanity Metrics
{json.dumps(sanity_metrics, indent=2)}

## Gate Details
{json.dumps(gate_details, indent=2)}

## Task
Decide if the latest model is realistic, feasible, sensible, and reliable.
Consider bias-variance tradeoff and historical player score distributions.
Return a decision and feedback.

Respond in JSON:
```json
{{
  \"decision\": \"accept|revise|veto\",
  \"rationale\": \"short explanation\",
  \"suggested_changes\": \"what to adjust next, if any\"
}}
```
"""

    llm = ChatOpenAI(
        model=os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5"),
        openai_api_key=os.environ["OPENROUTER_API_KEY"],
        openai_api_base=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        max_tokens=512,
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    response_text = response.content

    logger.info(f"Reviewer response:\n{response_text}")

    decision = "revise"
    rationale = response_text
    suggested_changes = ""

    json_match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if not json_match:
        json_match = re.search(r"\{[^{}]*\"decision\"[^{}]*\}", response_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1) if "```" in response_text else json_match.group(0))
            decision = str(parsed.get("decision", decision)).lower()
            rationale = parsed.get("rationale", rationale)
            suggested_changes = parsed.get("suggested_changes", "")
        except json.JSONDecodeError:
            pass

    max_rejects = _env_int("TUNING_REVIEW_MAX_REJECTS", 3)
    reject_count = state.get("reject_count", 0)

    if not gates_passed:
        if reject_count + 1 >= max_rejects:
            decision = "veto"
            rationale = f"Gates failed and max rejects reached ({max_rejects})."
        else:
            decision = "revise"
            rationale = "Gates failed. Revise parameters to meet realism gates."

    if decision not in {"accept", "revise", "veto"}:
        decision = "revise"

    if decision in {"revise", "veto"}:
        reject_count += 1
    else:
        reject_count = 0

    feedback = rationale
    if suggested_changes:
        feedback = f"{rationale}\nSuggested changes: {suggested_changes}"

    return {
        "review_decision": decision,
        "review_feedback": feedback,
        "reject_count": reject_count,
    }


# ---------------------------------------------------------------------------
# Node: check_convergence
# ---------------------------------------------------------------------------

def check_convergence(state: TuningState) -> str:
    """Decide whether to continue or save the best model."""
    iteration = state["iteration"]
    max_iterations = state["max_iterations"]
    experiments = state["experiments"]
    no_improve_iters = _env_int("TUNING_NO_IMPROVE_ITERS", 3)

    if iteration >= max_iterations:
        logger.info(f"Reached max iterations ({max_iterations}). Stopping.")
        return "save_best"

    best_holdout_mae = state.get("best_holdout_mae", float("inf"))
    if best_holdout_mae == float("inf"):
        return "propose_params"

    if len(experiments) >= no_improve_iters + 1:
        recent = experiments[-no_improve_iters:]
        prior = experiments[:-no_improve_iters]
        recent_feasible = [e["holdout_mae"] for e in recent if e.get("gates_passed")]
        prior_feasible = [e["holdout_mae"] for e in prior if e.get("gates_passed")]

        if recent_feasible and prior_feasible:
            best_recent = min(recent_feasible)
            best_before = min(prior_feasible)
            if best_recent >= best_before:
                logger.info("No improvement in last %d iterations. Stopping.", no_improve_iters)
                return "save_best"

    return "propose_params"


def route_after_review(state: TuningState) -> str:
    decision = state.get("review_decision", "revise")
    if decision == "accept":
        return "check_convergence"
    if decision == "veto":
        return "save_best"
    return "propose_params"


__all__ = [
    "fetch_data",
    "train_evaluate",
    "propose_params",
    "review_candidate",
    "check_convergence",
    "route_after_review",
    "save_best",
]
