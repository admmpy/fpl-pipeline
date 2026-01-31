"""
Entry point for scheduled FPL ingestion runs.
"""
from __future__ import annotations

import os
from typing import Optional

from flows.fpl_ingestion import fpl_typed_pipeline


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


def run_ingestion() -> dict[str, object]:
    """
    Run the ingestion pipeline with environment-driven options.
    """
    include_player_details = parse_bool_env("INCLUDE_PLAYER_DETAILS", default=True)
    max_players = parse_int_env("MAX_PLAYERS")

    return fpl_typed_pipeline(
        include_player_details=include_player_details,
        max_players=max_players,
    )


if __name__ == "__main__":
    run_ingestion()
