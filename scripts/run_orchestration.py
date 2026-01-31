"""
Entry point for scheduled full orchestration runs.
"""
from __future__ import annotations

import os
from typing import Optional

from flows.fpl_orchestration import fpl_weekly_orchestration


def parse_bool_env(var_name: str, default: bool) -> bool:
    """
    Parse a boolean environment variable.

    Args:
        var_name: Environment variable name
        default: Default value if unset

    Returns:
        Parsed boolean
    """
    raw_value = os.getenv(var_name)
    if raw_value is None:
        return default

    return raw_value.strip().lower() in {"1", "true", "yes", "y"}


def parse_int_env(var_name: str) -> Optional[int]:
    """
    Parse an integer environment variable.

    Args:
        var_name: Environment variable name

    Returns:
        Parsed integer if present and valid, otherwise None
    """
    raw_value = os.getenv(var_name)
    if raw_value is None:
        return None

    stripped = raw_value.strip()
    return int(stripped) if stripped.isdigit() else None


def run_orchestration() -> dict[str, object]:
    """
    Run the full orchestration pipeline with environment-driven options.
    """
    include_player_details = parse_bool_env("INCLUDE_PLAYER_DETAILS", default=True)
    allow_stale_data = parse_bool_env("ALLOW_STALE_DATA", default=True)
    notify_on_success = parse_bool_env("NOTIFY_ON_SUCCESS", default=False)
    max_players = parse_int_env("MAX_PLAYERS")

    dbt_project_dir = os.getenv("DBT_PROJECT_DIR", "dbt_project")
    dbt_profiles_dir = os.getenv("DBT_PROFILES_DIR")
    slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")

    return fpl_weekly_orchestration(
        include_player_details=include_player_details,
        max_players=max_players,
        dbt_project_dir=dbt_project_dir,
        dbt_profiles_dir=dbt_profiles_dir,
        allow_stale_data=allow_stale_data,
        slack_webhook_url=slack_webhook_url,
        notify_on_success=notify_on_success,
    )


if __name__ == "__main__":
    run_orchestration()
