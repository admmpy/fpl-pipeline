"""Node implementations for the autonomous optimisation LangGraph workflow."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from jsonschema import ValidationError as JsonSchemaValidationError, validate as jsonschema_validate
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add pipeline root/scripts to path so we can import train_model and peer modules.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from agents.autonomous_state import (
    AutonomousState,
    guard_node_state,
    transition_update,
)
from agents.tuning_nodes_review import _compute_sanity_metrics
import train_model
from utils.model_registry import (
    DEFAULT_LOGS_DIR,
    get_active_model,
    promote_atomically,
    rollback_to,
)

logger = logging.getLogger(__name__)

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RULES_PATH = PIPELINE_ROOT / "config" / "domain_rules.yaml"
DEFAULT_RULES_SCHEMA_PATH = PIPELINE_ROOT / "config" / "domain_rules.schema.json"
AUTONOMOUS_LOG_DIR = DEFAULT_LOGS_DIR / "autonomous"
AUTONOMOUS_EVENTS_PATH = AUTONOMOUS_LOG_DIR / "events.jsonl"

SENSITIVE_KEY_PATTERN = ("KEY", "TOKEN", "SECRET")


class DomainRulesError(ValueError):
    """Raised when domain rules are invalid or missing."""


def _now_utc() -> str:
    return datetime.now(UTC).isoformat()


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PIPELINE_ROOT, stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def _ensure_dirs() -> None:
    AUTONOMOUS_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _contains_sensitive_key(key: str) -> bool:
    upper = key.upper()
    return any(pattern in upper for pattern in SENSITIVE_KEY_PATTERN)


def redact_sensitive(payload: Any) -> Any:
    """Recursively redact keys that match *KEY*, *TOKEN*, *SECRET*."""

    if isinstance(payload, dict):
        redacted: dict[str, Any] = {}
        for key, value in payload.items():
            if _contains_sensitive_key(str(key)):
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = redact_sensitive(value)
        return redacted
    if isinstance(payload, list):
        return [redact_sensitive(item) for item in payload]
    return payload


def _safe_for_json(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return {
            "rows": int(len(value)),
            "columns": list(value.columns),
        }
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _safe_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_for_json(v) for v in value]
    return value


def _node_log(
    *,
    node: str,
    run_id: str,
    state: Optional[str],
    duration_ms: int,
    status: str,
    detail: Optional[dict[str, Any]] = None,
) -> None:
    _ensure_dirs()
    event = redact_sensitive(
        {
            "timestamp_utc": _now_utc(),
            "run_id": run_id,
            "node": node,
            "state": state,
            "duration_ms": duration_ms,
            "status": status,
            "detail": _safe_for_json(detail or {}),
        }
    )
    with AUTONOMOUS_EVENTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def _failure_update(node: str, state: AutonomousState, exc: Exception) -> dict[str, Any]:
    duration_ms = 0
    error_payload = {
        "type": type(exc).__name__,
        "message": str(exc),
        "node": node,
        "timestamp_utc": _now_utc(),
    }
    _node_log(
        node=node,
        run_id=state.get("run_id", "unknown"),
        state="FAILED",
        duration_ms=duration_ms,
        status="failed",
        detail={"error": error_payload},
    )
    return {
        "state": "FAILED",
        "error": error_payload,
    }


def _state_run_id(state: AutonomousState) -> str:
    run_id = state.get("run_id")
    if run_id:
        return run_id
    return f"autonomous_{uuid.uuid4().hex[:12]}"


def _hash_dataframe(df: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(df, index=True)
    digest = hashlib.sha256(hashed.values.tobytes())
    return digest.hexdigest()


def load_domain_rules(path: str | Path = DEFAULT_RULES_PATH) -> dict[str, Any]:
    """Load and validate domain rules against the JSON schema."""

    rules_path = Path(path)
    if not rules_path.exists():
        raise DomainRulesError(f"Rules file not found: {rules_path}")

    schema_path = DEFAULT_RULES_SCHEMA_PATH
    if not schema_path.exists():
        raise DomainRulesError(f"Rules schema file not found: {schema_path}")

    with rules_path.open("r", encoding="utf-8") as handle:
        rules = yaml.safe_load(handle) or {}
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)

    try:
        jsonschema_validate(instance=rules, schema=schema)
    except JsonSchemaValidationError as exc:
        raise DomainRulesError(f"Rules validation failed: {exc.message}") from exc

    return rules


def _rules_from_state(state: AutonomousState) -> dict[str, Any]:
    snapshot_meta = state.get("snapshot_meta") or {}
    rules_path = snapshot_meta.get("rules_path") or str(DEFAULT_RULES_PATH)
    return load_domain_rules(rules_path)


def _load_snapshot_dataframe(state: AutonomousState) -> pd.DataFrame:
    snapshot_meta = state.get("snapshot_meta") or {}

    explicit_df = snapshot_meta.get("dataframe")
    if isinstance(explicit_df, pd.DataFrame):
        return explicit_df.copy()

    snapshot_path = snapshot_meta.get("snapshot_path")
    if snapshot_path:
        return pd.read_csv(snapshot_path)

    df = train_model.fetch_training_data()
    return train_model.engineer_features(df)


def _dtype_matches(series: pd.Series, expected: str) -> bool:
    if expected == "integer":
        if pd.api.types.is_integer_dtype(series):
            return True
        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if non_null.empty:
                return True
            return bool(((non_null % 1) == 0).all())
        return False
    if expected == "number":
        return pd.api.types.is_numeric_dtype(series)
    if expected == "string":
        return pd.api.types.is_string_dtype(series)
    if expected == "boolean":
        return pd.api.types.is_bool_dtype(series)
    return False


def _ensure_targets(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    if "target_next_gw_points" not in prepared.columns or "target_gameweek_id" not in prepared.columns:
        prepared = train_model.engineer_features(prepared)
    if "target_next_gw_points" not in prepared.columns or "target_gameweek_id" not in prepared.columns:
        raise ValueError("Snapshot is missing target columns and could not be engineered")
    prepared = prepared.dropna(subset=["target_next_gw_points", "target_gameweek_id"])
    prepared = prepared[prepared["target_gameweek_id"] > prepared["gameweek_id"]].copy()
    if prepared.empty:
        raise ValueError("No rows remain after filtering non-future target rows")
    return prepared


def _prepare_train_holdout(df: pd.DataFrame, holdout_gameweeks: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    train_df, holdout_df = train_model.split_train_holdout(df, holdout_gameweeks)
    global_stats = train_model.compute_global_stats(train_df)
    train_df = train_model.add_z_scores(train_df, global_stats)
    holdout_df = train_model.add_z_scores(holdout_df, global_stats)

    features = train_model.select_features()
    for feature in features:
        if feature not in train_df.columns:
            train_df[feature] = 0
            holdout_df[feature] = 0

    return train_df, holdout_df, features, global_stats


def _fit_predict(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    features: list[str],
    params: dict[str, Any],
) -> tuple[xgb.XGBRegressor, np.ndarray, dict[str, Any]]:
    X_train = train_df[features].fillna(0)
    y_train = train_df["target_next_gw_points"]
    X_holdout = holdout_df[features].fillna(0)
    y_holdout = holdout_df["target_next_gw_points"]

    shrinkage_alpha = float(params.get("shrinkage_alpha", 0.0))
    calibration_strength = float(params.get("calibration_strength", 0.8))

    xgb_params = {
        "n_estimators": int(params.get("n_estimators", 200)),
        "max_depth": int(params.get("max_depth", 5)),
        "learning_rate": float(params.get("learning_rate", 0.1)),
        "subsample": float(params.get("subsample", 0.8)),
        "colsample_bytree": float(params.get("colsample_bytree", 0.8)),
        "reg_alpha": float(params.get("reg_alpha", 0.1)),
        "reg_lambda": float(params.get("reg_lambda", 1.0)),
        "min_child_weight": float(params.get("min_child_weight", 1.0)),
        "gamma": float(params.get("gamma", 0.0)),
        "max_delta_step": float(params.get("max_delta_step", 0.0)),
        "random_state": 42,
        "n_jobs": 1,
        "objective": "reg:squarederror",
        "eval_metric": "mae",
    }

    model = xgb.XGBRegressor(**xgb_params)

    use_log_target = os.environ.get("TUNING_LOG_TARGET", "1").lower() in {"1", "true", "yes"}
    y_train_trans = train_model._transform_target(y_train) if use_log_target else y_train
    model.fit(X_train, y_train_trans)

    holdout_pred = model.predict(X_holdout)
    if use_log_target:
        holdout_pred = train_model._inverse_transform(holdout_pred)

    league_mean = float(y_train.mean())
    holdout_pred = train_model.apply_shrinkage(holdout_pred, league_mean, shrinkage_alpha)

    calibration_payload = {"a": 1.0, "b": 0.0, "strength": calibration_strength}
    if calibration_strength > 0:
        a, b = train_model.fit_calibration(y_holdout.to_numpy(), holdout_pred)
        a, b = train_model.blend_calibration(a, b, calibration_strength)
        holdout_pred = train_model.apply_calibration(holdout_pred, a, b)
        calibration_payload = {"a": float(a), "b": float(b), "strength": calibration_strength}

    holdout_pred = np.maximum(holdout_pred, 0)

    post = {
        "league_mean": league_mean,
        "shrinkage_alpha": shrinkage_alpha,
        "calibration": calibration_payload,
    }
    return model, holdout_pred, post


def _evaluate_payload(
    payload: Any,
    holdout_df: pd.DataFrame,
    features: list[str],
) -> dict[str, float]:
    if isinstance(payload, dict):
        model = payload.get("model")
        metadata = payload.get("metadata", {})
    else:
        model = payload
        metadata = {}

    if model is None or not hasattr(model, "predict"):
        raise ValueError("Active model payload is invalid")

    X_holdout = holdout_df[features].fillna(0)
    y_holdout = holdout_df["target_next_gw_points"]
    pred = model.predict(X_holdout)

    league_mean = metadata.get("league_mean")
    if league_mean is not None:
        pred = train_model.apply_shrinkage(
            np.asarray(pred), float(league_mean), float(metadata.get("shrinkage_alpha", 0.0))
        )

    calibration = metadata.get("calibration")
    if calibration:
        pred = train_model.apply_calibration(
            np.asarray(pred),
            float(calibration.get("a", 1.0)),
            float(calibration.get("b", 0.0)),
        )

    pred = np.maximum(pred, 0)
    return {
        "mae": float(mean_absolute_error(y_holdout, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_holdout, pred))),
        "bias": float((y_holdout - pred).mean()),
    }


def _evidence_payload(state: AutonomousState) -> dict[str, Any]:
    snapshot_meta = state.get("snapshot_meta") or {}
    rules_path = snapshot_meta.get("rules_path") or str(DEFAULT_RULES_PATH)
    rules_version = "unknown"
    try:
        rules_version = load_domain_rules(rules_path).get("version", "unknown")
    except Exception:
        rules_version = "unknown"

    return redact_sensitive(
        {
            "run_id": state.get("run_id"),
            "timestamps_utc": {
                "started_at": snapshot_meta.get("started_at_utc"),
                "completed_at": _now_utc(),
            },
            "code_version": _git_commit(),
            "input_snapshot_hash": snapshot_meta.get("snapshot_hash"),
            "rules_version": rules_version,
            "active_model": {
                "pre": state.get("previous_model_version"),
                "post": state.get("active_model_version"),
            },
            "metrics": _safe_for_json(state.get("candidate_metrics") or {}),
            "gate_outcomes": _safe_for_json(state.get("rule_eval") or {}),
            "decision_rationale": _safe_for_json(state.get("promotion_decision") or {}),
            "rollback_target": state.get("previous_model_version"),
            "final_state": state.get("state"),
            "error": _safe_for_json(state.get("error")),
        }
    )


def _write_evidence(state: AutonomousState, suffix: str = "") -> str:
    _ensure_dirs()
    run_id = state.get("run_id") or "unknown"
    path = AUTONOMOUS_LOG_DIR / f"{run_id}{suffix}.json"
    payload = _evidence_payload(state)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return str(path)


def ingest_snapshot(state: AutonomousState) -> dict[str, Any]:
    """Ingest and fingerprint the latest training snapshot."""

    start = time.perf_counter()
    run_id = _state_run_id(state)
    try:
        rules = _rules_from_state(state)
        df = _load_snapshot_dataframe(state)
        df = _ensure_targets(df)

        snapshot_meta = dict(state.get("snapshot_meta") or {})
        snapshot_meta.update(
            {
                "started_at_utc": snapshot_meta.get("started_at_utc") or _now_utc(),
                "snapshot_rows": int(len(df)),
                "snapshot_columns": list(df.columns),
                "snapshot_hash": _hash_dataframe(df),
                "rules_path": snapshot_meta.get("rules_path") or str(DEFAULT_RULES_PATH),
                "rules_version": rules.get("version"),
                "dataframe": df,
            }
        )

        update = {
            "run_id": run_id,
            "snapshot_meta": snapshot_meta,
            "error": None,
            **transition_update(state, "INGESTED"),
        }
        _node_log(
            node="ingest_snapshot",
            run_id=run_id,
            state="INGESTED",
            duration_ms=int((time.perf_counter() - start) * 1000),
            status="ok",
            detail={"rows": len(df)},
        )
        return update
    except Exception as exc:
        return {"run_id": run_id, **_failure_update("ingest_snapshot", {**state, "run_id": run_id}, exc)}


def validate_snapshot(state: AutonomousState) -> dict[str, Any]:
    """Apply hard gate schema validation and training split checks."""

    start = time.perf_counter()
    run_id = _state_run_id(state)
    try:
        guard_node_state(state, node_name="validate_snapshot", allowed_states={"INGESTED"})
        rules = _rules_from_state(state)
        schema_rules = rules["schema"]
        split_rules = rules["split"]

        df = (state.get("snapshot_meta") or {}).get("dataframe")
        if not isinstance(df, pd.DataFrame):
            raise ValueError("snapshot_meta.dataframe is missing")

        failures: list[str] = []

        required_columns = schema_rules["required_columns"]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            failures.append(f"Missing required columns: {missing}")

        for col, dtype_name in schema_rules["dtypes"].items():
            if col in df.columns and not _dtype_matches(df[col], dtype_name):
                failures.append(f"Column '{col}' failed dtype check: expected {dtype_name}")

        for col, max_ratio in schema_rules.get("null_limits", {}).items():
            if col in df.columns:
                ratio = float(df[col].isna().mean())
                if ratio > float(max_ratio):
                    failures.append(
                        f"Column '{col}' null ratio {ratio:.4f} exceeds max {float(max_ratio):.4f}"
                    )

        duplicate_cfg = schema_rules["duplicate_key_limits"]
        duplicate_keys = duplicate_cfg["keys"]
        if all(key in df.columns for key in duplicate_keys):
            duplicate_ratio = float(df.duplicated(subset=duplicate_keys).mean())
            if duplicate_ratio > float(duplicate_cfg["max_ratio"]):
                failures.append(
                    f"Duplicate ratio {duplicate_ratio:.4f} exceeds max {float(duplicate_cfg['max_ratio']):.4f}"
                )

        leakage_cfg = split_rules["temporal_leakage"]
        src_col = leakage_cfg["source_column"]
        tgt_col = leakage_cfg["target_column"]
        if src_col in df.columns and tgt_col in df.columns:
            if leakage_cfg["allow_equal"]:
                leakage_violations = int((df[tgt_col] < df[src_col]).sum())
            else:
                leakage_violations = int((df[tgt_col] <= df[src_col]).sum())
            if leakage_violations > 0:
                failures.append(f"Temporal leakage violations: {leakage_violations}")
        else:
            failures.append(
                f"Temporal leakage columns missing: source='{src_col}', target='{tgt_col}'"
            )

        if failures:
            report = {
                "ok": False,
                "failures": failures,
                "checked_at_utc": _now_utc(),
            }
            return {
                "validation_report": report,
                "error": {"type": "ValidationError", "message": "; ".join(failures)},
                "state": "FAILED",
            }

        holdout_gameweeks = int(split_rules["holdout_gameweeks"])
        train_df, holdout_df, features, global_stats = _prepare_train_holdout(df, holdout_gameweeks)

        snapshot_meta = dict(state.get("snapshot_meta") or {})
        snapshot_meta.update(
            {
                "train_df": train_df,
                "holdout_df": holdout_df,
                "features": features,
                "global_stats": global_stats,
            }
        )

        report = {
            "ok": True,
            "failures": [],
            "checked_at_utc": _now_utc(),
            "train_rows": int(len(train_df)),
            "holdout_rows": int(len(holdout_df)),
        }

        _node_log(
            node="validate_snapshot",
            run_id=run_id,
            state="VALIDATED",
            duration_ms=int((time.perf_counter() - start) * 1000),
            status="ok",
            detail={"train_rows": len(train_df), "holdout_rows": len(holdout_df)},
        )
        return {
            "snapshot_meta": snapshot_meta,
            "validation_report": report,
            "error": None,
            **transition_update(state, "VALIDATED"),
        }
    except Exception as exc:
        return _failure_update("validate_snapshot", {**state, "run_id": run_id}, exc)


def detect_drift(state: AutonomousState) -> dict[str, Any]:
    """Run deterministic drift checks and decide whether retraining is required."""

    start = time.perf_counter()
    run_id = _state_run_id(state)
    try:
        guard_node_state(state, node_name="detect_drift", allowed_states={"VALIDATED"})
        rules = _rules_from_state(state)
        drift_cfg = rules["drift"]

        snapshot_meta = dict(state.get("snapshot_meta") or {})
        force_drift = bool(snapshot_meta.get("force_drift", False))
        force_no_drift = bool(snapshot_meta.get("force_no_drift", False))

        train_df = snapshot_meta.get("train_df")
        holdout_df = snapshot_meta.get("holdout_df")
        if not isinstance(train_df, pd.DataFrame) or not isinstance(holdout_df, pd.DataFrame):
            raise ValueError("Train/holdout frames are missing for drift detection")

        checks: list[dict[str, Any]] = []
        breached = 0
        columns = list(drift_cfg["columns"])

        if force_no_drift:
            breached = 0
        elif force_drift:
            breached = len(columns)
        else:
            ks_threshold = float(drift_cfg["ks_threshold"])
            for col in columns:
                if col not in train_df.columns or col not in holdout_df.columns:
                    continue
                stat, p_value = ks_2samp(
                    train_df[col].fillna(0).to_numpy(), holdout_df[col].fillna(0).to_numpy()
                )
                is_breach = float(stat) > ks_threshold
                if is_breach:
                    breached += 1
                checks.append(
                    {
                        "column": col,
                        "ks_stat": float(stat),
                        "p_value": float(p_value),
                        "breach": is_breach,
                    }
                )

        total_cols = max(len(columns), 1)
        breach_ratio = breached / total_cols
        trigger_threshold = float(drift_cfg["breach_ratio"])
        drift_triggered = breach_ratio >= trigger_threshold

        drift_report = {
            "checked_at_utc": _now_utc(),
            "columns": columns,
            "checks": checks,
            "breached": breached,
            "breach_ratio": float(breach_ratio),
            "trigger_threshold": trigger_threshold,
            "drift_triggered": drift_triggered,
            "force_drift": force_drift,
            "force_no_drift": force_no_drift,
        }

        next_state = "DRIFT_TRIGGERED" if drift_triggered else "RECORDED"
        _node_log(
            node="detect_drift",
            run_id=run_id,
            state=next_state,
            duration_ms=int((time.perf_counter() - start) * 1000),
            status="ok",
            detail={"drift_triggered": drift_triggered, "breach_ratio": breach_ratio},
        )
        return {
            "drift_report": drift_report,
            "promotion_decision": {
                "decision": "retrain" if drift_triggered else "no_action",
                "rationale": "Drift threshold exceeded" if drift_triggered else "No drift trigger",
            },
            **transition_update(state, next_state),
        }
    except Exception as exc:
        return _failure_update("detect_drift", {**state, "run_id": run_id}, exc)


def run_optuna_search(state: AutonomousState) -> dict[str, Any]:
    """Run deterministic Optuna search over XGBoost and post-processing parameters."""

    start = time.perf_counter()
    run_id = _state_run_id(state)
    try:
        guard_node_state(state, node_name="run_optuna_search", allowed_states={"DRIFT_TRIGGERED"})
        rules = _rules_from_state(state)

        try:
            import optuna
        except ImportError as exc:
            raise RuntimeError("Optuna is required for autonomous search") from exc

        snapshot_meta = dict(state.get("snapshot_meta") or {})
        train_df = snapshot_meta.get("train_df")
        holdout_df = snapshot_meta.get("holdout_df")
        features = snapshot_meta.get("features")
        if not isinstance(train_df, pd.DataFrame) or not isinstance(holdout_df, pd.DataFrame):
            raise ValueError("Train/holdout frames missing for Optuna search")
        if not isinstance(features, list) or not features:
            raise ValueError("Feature list missing for Optuna search")

        trials = int(snapshot_meta.get("optuna_trials") or rules["optimisation"]["optuna_trials"])
        seed = int(rules["optimisation"]["random_seed"])
        y_holdout = holdout_df["target_next_gw_points"]

        def objective(trial: Any) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 3.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 5.0),
                "gamma": trial.suggest_float("gamma", 0.0, 2.0),
                "max_delta_step": trial.suggest_float("max_delta_step", 0.0, 5.0),
                "shrinkage_alpha": trial.suggest_float("shrinkage_alpha", 0.0, 0.5),
                "calibration_strength": trial.suggest_float("calibration_strength", 0.0, 1.0),
            }
            _, pred, _ = _fit_predict(train_df, holdout_df, features, params)
            return float(mean_absolute_error(y_holdout, pred))

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=trials)

        result = {
            "best_params": study.best_trial.params,
            "best_value": float(study.best_value),
            "n_trials": trials,
            "seed": seed,
        }

        _node_log(
            node="run_optuna_search",
            run_id=run_id,
            state="SEARCHED",
            duration_ms=int((time.perf_counter() - start) * 1000),
            status="ok",
            detail={"best_value": result["best_value"], "n_trials": trials},
        )
        return {
            "optuna_result": result,
            **transition_update(state, "SEARCHED"),
        }
    except Exception as exc:
        return _failure_update("run_optuna_search", {**state, "run_id": run_id}, exc)


def train_best_candidate(state: AutonomousState) -> dict[str, Any]:
    """Train the best candidate returned by Optuna."""

    start = time.perf_counter()
    run_id = _state_run_id(state)
    try:
        guard_node_state(state, node_name="train_best_candidate", allowed_states={"SEARCHED"})

        snapshot_meta = dict(state.get("snapshot_meta") or {})
        train_df = snapshot_meta.get("train_df")
        holdout_df = snapshot_meta.get("holdout_df")
        features = snapshot_meta.get("features")

        if not isinstance(train_df, pd.DataFrame) or not isinstance(holdout_df, pd.DataFrame):
            raise ValueError("Train/holdout frames missing for training")
        if not isinstance(features, list) or not features:
            raise ValueError("Feature list missing for training")

        best_params = (state.get("optuna_result") or {}).get("best_params")
        if not isinstance(best_params, dict) or not best_params:
            raise ValueError("No Optuna best parameters available")

        model, holdout_pred, post = _fit_predict(train_df, holdout_df, features, best_params)

        snapshot_meta["candidate_model_payload"] = {
            "model": model,
            "metadata": {
                "feature_cols": features,
                "zscore_stats": snapshot_meta.get("global_stats", {}),
                "league_mean": post["league_mean"],
                "shrinkage_alpha": post["shrinkage_alpha"],
                "calibration": post["calibration"],
            },
        }
        snapshot_meta["candidate_holdout_pred"] = holdout_pred

        _node_log(
            node="train_best_candidate",
            run_id=run_id,
            state="TRAINED",
            duration_ms=int((time.perf_counter() - start) * 1000),
            status="ok",
        )
        return {
            "snapshot_meta": snapshot_meta,
            **transition_update(state, "TRAINED"),
        }
    except Exception as exc:
        return _failure_update("train_best_candidate", {**state, "run_id": run_id}, exc)


def evaluate_candidate(state: AutonomousState) -> dict[str, Any]:
    """Evaluate candidate metrics against current active baseline."""

    start = time.perf_counter()
    run_id = _state_run_id(state)
    try:
        guard_node_state(state, node_name="evaluate_candidate", allowed_states={"TRAINED"})

        snapshot_meta = dict(state.get("snapshot_meta") or {})
        holdout_df = snapshot_meta.get("holdout_df")
        train_df = snapshot_meta.get("train_df")
        features = snapshot_meta.get("features")
        holdout_pred = snapshot_meta.get("candidate_holdout_pred")

        if not isinstance(holdout_df, pd.DataFrame) or not isinstance(train_df, pd.DataFrame):
            raise ValueError("Train/holdout frames missing for evaluation")
        if not isinstance(features, list) or not features:
            raise ValueError("Feature list missing for evaluation")
        if holdout_pred is None:
            raise ValueError("Candidate predictions missing for evaluation")

        y_holdout = holdout_df["target_next_gw_points"]
        holdout_pred_arr = np.asarray(holdout_pred)

        candidate_sanity = _compute_sanity_metrics(
            train_df=train_df,
            holdout_df=holdout_df,
            y_holdout=y_holdout,
            holdout_pred=holdout_pred_arr,
        )

        candidate_metrics = {
            "mae": float(mean_absolute_error(y_holdout, holdout_pred_arr)),
            "rmse": float(np.sqrt(mean_squared_error(y_holdout, holdout_pred_arr))),
            "bias": float((y_holdout - holdout_pred_arr).mean()),
            "position_exceed_rate": float(candidate_sanity.get("position_exceed_rate") or 0.0),
            "player_exceed_rate": float(candidate_sanity.get("player_exceed_rate") or 0.0),
            "sanity": candidate_sanity,
        }

        active_baseline = {
            "mae": float("inf"),
            "rmse": float("inf"),
            "bias": 0.0,
        }
        active_model_version = None
        active = get_active_model(DEFAULT_LOGS_DIR)
        if active:
            active_model_version = active["version"]
            with Path(active["path"]).open("rb") as handle:
                active_payload = pickle.load(handle)
            active_baseline = _evaluate_payload(active_payload, holdout_df, features)

        combined = {
            "candidate": candidate_metrics,
            "active_baseline": active_baseline,
            "evaluated_at_utc": _now_utc(),
        }

        _node_log(
            node="evaluate_candidate",
            run_id=run_id,
            state="EVALUATED",
            duration_ms=int((time.perf_counter() - start) * 1000),
            status="ok",
            detail={"candidate_mae": candidate_metrics["mae"], "active_mae": active_baseline["mae"]},
        )
        return {
            "candidate_metrics": combined,
            "active_model_version": active_model_version,
            **transition_update(state, "EVALUATED"),
        }
    except Exception as exc:
        return _failure_update("evaluate_candidate", {**state, "run_id": run_id}, exc)


def apply_domain_rules(state: AutonomousState) -> dict[str, Any]:
    """Apply hard promotion gates from domain rules."""

    start = time.perf_counter()
    run_id = _state_run_id(state)
    try:
        guard_node_state(state, node_name="apply_domain_rules", allowed_states={"EVALUATED"})
        rules = _rules_from_state(state)
        model_rules = rules["model"]

        metrics = state.get("candidate_metrics") or {}
        candidate = metrics.get("candidate") or {}
        baseline = metrics.get("active_baseline") or {}
        drift_report = state.get("drift_report") or {}

        candidate_mae = float(candidate.get("mae", float("inf")))
        candidate_rmse = float(candidate.get("rmse", float("inf")))
        candidate_bias = float(candidate.get("bias", 0.0))
        position_exceed_rate = float(candidate.get("position_exceed_rate", 0.0))
        player_exceed_rate = float(candidate.get("player_exceed_rate", 0.0))

        active_mae = float(baseline.get("mae", float("inf")))
        active_rmse = float(baseline.get("rmse", float("inf")))

        gates = {
            "drift_context_present": bool(drift_report.get("drift_triggered") is True),
            "max_mae": candidate_mae <= float(model_rules["max_mae"]),
            "max_rmse": candidate_rmse <= float(model_rules["max_rmse"]),
            "max_abs_bias": abs(candidate_bias) <= float(model_rules["max_abs_bias"]),
            "position_exceed_cap": position_exceed_rate <= float(model_rules["max_position_exceed_rate"]),
            "player_exceed_cap": player_exceed_rate <= float(model_rules["max_player_exceed_rate"]),
            "mae_improves_active": candidate_mae < active_mae,
            "rmse_within_deterioration": (
                True
                if not np.isfinite(active_rmse)
                else candidate_rmse <= active_rmse * (1.0 + float(model_rules["rmse_deterioration_pct"]))
            ),
        }
        passed = all(gates.values())

        rationale = [
            f"{name}={'pass' if outcome else 'fail'}"
            for name, outcome in gates.items()
        ]

        next_state = "RULES_PASSED" if passed else "REJECTED"
        _node_log(
            node="apply_domain_rules",
            run_id=run_id,
            state=next_state,
            duration_ms=int((time.perf_counter() - start) * 1000),
            status="ok",
            detail={"passed": passed, "gates": gates},
        )
        return {
            "rule_eval": {
                "passed": passed,
                "gates": gates,
                "checked_at_utc": _now_utc(),
                "rules_version": rules.get("version"),
            },
            "promotion_decision": {
                "decision": "promote" if passed else "reject",
                "rationale": rationale,
            },
            **transition_update(state, next_state),
        }
    except Exception as exc:
        return _failure_update("apply_domain_rules", {**state, "run_id": run_id}, exc)


def promote_model_transaction(state: AutonomousState) -> dict[str, Any]:
    """Promote model atomically and capture rollback anchor metadata."""

    start = time.perf_counter()
    run_id = _state_run_id(state)
    try:
        guard_node_state(state, node_name="promote_model_transaction", allowed_states={"RULES_PASSED"})

        snapshot_meta = dict(state.get("snapshot_meta") or {})
        payload = snapshot_meta.get("candidate_model_payload")
        if not isinstance(payload, dict) or "model" not in payload:
            raise ValueError("Candidate model payload missing for promotion")

        precheck_path = _write_evidence(state, suffix=".precheck")
        if not Path(precheck_path).exists():
            raise RuntimeError("Evidence precheck write failed")

        promoted = promote_atomically(
            model_payload=payload,
            candidate_metrics=state.get("candidate_metrics") or {},
            run_id=run_id,
            logs_dir=DEFAULT_LOGS_DIR,
        )

        _node_log(
            node="promote_model_transaction",
            run_id=run_id,
            state="PROMOTED",
            duration_ms=int((time.perf_counter() - start) * 1000),
            status="ok",
            detail={"active_version": promoted.get("version")},
        )
        return {
            "active_model_version": promoted.get("version"),
            "previous_model_version": promoted.get("previous_model_version"),
            "promotion_decision": {
                **(state.get("promotion_decision") or {}),
                "registry_entry": promoted.get("entry"),
                "evidence_precheck_path": precheck_path,
            },
            **transition_update(state, "PROMOTED"),
        }
    except Exception as exc:
        error_update = _failure_update("promote_model_transaction", {**state, "run_id": run_id}, exc)
        # RULES_PASSED -> ROLLED_BACK is an allowed deterministic failure path.
        if state.get("state") == "RULES_PASSED":
            return {
                "state": "ROLLED_BACK",
                "error": error_update.get("error"),
                "promotion_decision": {
                    **(state.get("promotion_decision") or {}),
                    "decision": "rollback",
                    "rationale": [str(exc)],
                },
            }
        return error_update


def rollback_active_model(state: AutonomousState) -> dict[str, Any]:
    """Rollback active pointer to previous model version after failed promotion."""

    start = time.perf_counter()
    run_id = _state_run_id(state)
    try:
        guard_node_state(state, node_name="rollback_active_model", allowed_states={"ROLLED_BACK"})
        previous_version = state.get("previous_model_version")
        if previous_version:
            restored = rollback_to(previous_version, logs_dir=DEFAULT_LOGS_DIR)
            active_model_version = restored.get("version")
        else:
            active = get_active_model(DEFAULT_LOGS_DIR)
            active_model_version = active.get("version") if active else None

        _node_log(
            node="rollback_active_model",
            run_id=run_id,
            state="ROLLED_BACK",
            duration_ms=int((time.perf_counter() - start) * 1000),
            status="ok",
            detail={"restored_version": active_model_version},
        )
        return {
            "active_model_version": active_model_version,
            "promotion_decision": {
                **(state.get("promotion_decision") or {}),
                "decision": "rolled_back",
            },
        }
    except Exception as exc:
        return _failure_update("rollback_active_model", {**state, "run_id": run_id}, exc)


def record_evidence(state: AutonomousState) -> dict[str, Any]:
    """Persist per-run evidence bundle and finalise terminal states."""

    start = time.perf_counter()
    run_id = _state_run_id(state)
    try:
        guard_node_state(
            state,
            node_name="record_evidence",
            allowed_states={"PROMOTED", "REJECTED", "ROLLED_BACK", "RECORDED", "FAILED"},
        )

        evidence_path = _write_evidence(state)

        current_state = state.get("state")
        update: dict[str, Any] = {
            "evidence_path": evidence_path,
        }
        if current_state in {"PROMOTED", "REJECTED", "ROLLED_BACK"}:
            update.update(transition_update(state, "RECORDED"))

        _node_log(
            node="record_evidence",
            run_id=run_id,
            state=update.get("state", current_state),
            duration_ms=int((time.perf_counter() - start) * 1000),
            status="ok",
            detail={"evidence_path": evidence_path},
        )
        return update
    except Exception as exc:
        return _failure_update("record_evidence", {**state, "run_id": run_id}, exc)
