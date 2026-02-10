"""State definition for the hyperparameter tuning agent."""

from typing import Optional
from typing_extensions import Annotated, TypedDict

import pandas as pd


class TuningState(TypedDict):
    # Cached data (set once by fetch_data node)
    train_df: Annotated[Optional[pd.DataFrame], "Training data"]
    holdout_df: Annotated[Optional[pd.DataFrame], "Holdout data"]
    features: list[str]
    global_stats: dict

    # Experiment tracking
    experiments: list[dict]  # [{params, cv_mae, holdout_mae, holdout_rmse, holdout_bias, iteration}]
    current_params: dict  # Params for the next experiment
    best_params: dict
    best_holdout_mae: float

    # Control
    iteration: int
    max_iterations: int  # Default 10
    converged: bool

    # LLM context
    analysis: str  # LLM's reasoning about latest results


# Hyperparameter search space bounds
SEARCH_SPACE = {
    "n_estimators": {"min": 50, "max": 1000, "type": "int"},
    "max_depth": {"min": 2, "max": 10, "type": "int"},
    "learning_rate": {"min": 0.01, "max": 0.3, "type": "float"},
    "subsample": {"min": 0.5, "max": 1.0, "type": "float"},
    "colsample_bytree": {"min": 0.5, "max": 1.0, "type": "float"},
    "reg_alpha": {"min": 0.0, "max": 2.0, "type": "float"},
    "reg_lambda": {"min": 0.1, "max": 5.0, "type": "float"},
    "shrinkage_alpha": {"min": 0.0, "max": 0.5, "type": "float"},
    "calibration_strength": {"min": 0.0, "max": 1.0, "type": "float"},
}

# Default starting hyperparameters (current production values)
DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "shrinkage_alpha": 0.0,
    "calibration_strength": 0.8,
}
