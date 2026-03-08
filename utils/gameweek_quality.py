"""Shared gameweek quality policy loading and filtering utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


DEFAULT_RULES_PATH = Path(__file__).resolve().parents[1] / "config" / "domain_rules.yaml"


def _normalise_week_list(value: Any) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Gameweek policy lists must be arrays of integers")
    weeks: list[int] = []
    for item in value:
        week = int(item)
        if week <= 0:
            raise ValueError(f"Invalid gameweek id in policy: {item}")
        weeks.append(week)
    return sorted(set(weeks))


def load_gameweek_quality_policy(rules_path: str | Path = DEFAULT_RULES_PATH) -> dict[str, Any]:
    """Load gameweek quality policy from domain rules with safe defaults."""

    path = Path(rules_path)
    if not path.exists():
        return {
            "policy_version": "unknown",
            "excluded_gameweeks": [],
            "backfilled_but_untrusted_gameweeks": [],
            "trusted_backfilled_gameweeks": [],
            "eligible_training_gameweeks": [],
            "eligible_holdout_backtest_gameweeks": [],
        }

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    quality = raw.get("gameweek_quality") or {}
    return {
        "policy_version": str(quality.get("policy_version") or raw.get("version") or "unknown"),
        "excluded_gameweeks": _normalise_week_list(quality.get("excluded_gameweeks")),
        "backfilled_but_untrusted_gameweeks": _normalise_week_list(
            quality.get("backfilled_but_untrusted_gameweeks")
        ),
        "trusted_backfilled_gameweeks": _normalise_week_list(
            quality.get("trusted_backfilled_gameweeks")
        ),
        "eligible_training_gameweeks": _normalise_week_list(
            quality.get("eligible_training_gameweeks")
        ),
        "eligible_holdout_backtest_gameweeks": _normalise_week_list(
            quality.get("eligible_holdout_backtest_gameweeks")
        ),
    }


def effective_excluded_gameweeks(policy: dict[str, Any]) -> set[int]:
    """Return gameweeks that should be excluded from evaluation use."""

    excluded = set(int(week) for week in policy.get("excluded_gameweeks") or [])
    untrusted = set(int(week) for week in policy.get("backfilled_but_untrusted_gameweeks") or [])
    trusted = set(int(week) for week in policy.get("trusted_backfilled_gameweeks") or [])
    return excluded | (untrusted - trusted)


def classify_gameweek(gameweek: int, policy: dict[str, Any]) -> dict[str, Any]:
    """Classify a gameweek under the current quality policy."""

    week = int(gameweek)
    if week in set(policy.get("excluded_gameweeks") or []):
        return {"trusted": False, "status": "excluded"}
    if week in effective_excluded_gameweeks(policy):
        return {"trusted": False, "status": "backfilled_untrusted"}
    return {"trusted": True, "status": "trusted"}


def split_train_holdout_by_policy(
    df: pd.DataFrame,
    *,
    holdout_gameweeks: int,
    policy: dict[str, Any],
    target_gameweek_col: str = "target_gameweek_id",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply gameweek quality policy and perform temporal holdout split."""

    if target_gameweek_col not in df.columns:
        raise ValueError(f"Missing required column '{target_gameweek_col}'")

    filtered = df.copy()
    filtered[target_gameweek_col] = filtered[target_gameweek_col].astype(int)

    effective_excluded = effective_excluded_gameweeks(policy)
    if effective_excluded:
        filtered = filtered[~filtered[target_gameweek_col].isin(effective_excluded)].copy()

    if filtered.empty:
        raise ValueError("No rows remain after applying gameweek quality exclusions")

    candidate_holdout_weeks = sorted(filtered[target_gameweek_col].dropna().unique().tolist())
    eligible_holdout = set(policy.get("eligible_holdout_backtest_gameweeks") or [])
    if eligible_holdout:
        candidate_holdout_weeks = [w for w in candidate_holdout_weeks if w in eligible_holdout]

    if len(candidate_holdout_weeks) <= holdout_gameweeks:
        raise ValueError("Not enough eligible gameweeks to create holdout set")

    holdout_weeks = set(candidate_holdout_weeks[-holdout_gameweeks:])

    train_candidates = filtered[~filtered[target_gameweek_col].isin(holdout_weeks)].copy()
    eligible_training = set(policy.get("eligible_training_gameweeks") or [])
    if eligible_training:
        train_candidates = train_candidates[
            train_candidates[target_gameweek_col].isin(eligible_training)
        ].copy()

    holdout_df = filtered[filtered[target_gameweek_col].isin(holdout_weeks)].copy()
    if train_candidates.empty:
        raise ValueError("No training rows remain after applying training gameweek policy")
    if holdout_df.empty:
        raise ValueError("No holdout rows remain after applying holdout gameweek policy")

    split_meta = {
        "policy_version": str(policy.get("policy_version") or "unknown"),
        "effective_excluded_gameweeks": sorted(effective_excluded),
        "holdout_gameweeks": sorted(int(w) for w in holdout_weeks),
        "train_gameweeks": sorted(
            int(w) for w in train_candidates[target_gameweek_col].dropna().unique().tolist()
        ),
        "eligible_training_gameweeks": sorted(int(w) for w in eligible_training),
        "eligible_holdout_backtest_gameweeks": sorted(int(w) for w in eligible_holdout),
    }

    return train_candidates, holdout_df, split_meta
