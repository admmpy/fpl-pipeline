"""
Prefect orchestration for ingestion + dbt transformations.
"""
from __future__ import annotations

from typing import Dict, Any, Optional

from prefect import flow, get_run_logger

from flows.fpl_ingestion import fpl_complete_pipeline
from tasks.dbt_tasks import run_dbt_command


@flow(name="FPL Weekly Orchestration", log_prints=True)
def fpl_weekly_orchestration(
    include_player_details: bool = True,
    include_live_gameweek: bool = True,
    max_players: Optional[int] = None,
    dbt_project_dir: str = "/Users/am/Sync/fpl-workspace/fpl_development",
    dbt_profiles_dir: Optional[str] = None,
    allow_stale_data: bool = True,
) -> Dict[str, Any]:
    """
    Orchestrate FPL ingestion and dbt transformations.

    Args:
        include_player_details: Whether to fetch detailed player stats
        include_live_gameweek: Whether to fetch live gameweek data
        max_players: Optional limit on players (for testing)
        dbt_project_dir: Path to dbt project directory
        dbt_profiles_dir: Optional path to dbt profiles directory
        allow_stale_data: Whether to proceed when dbt fails

    Returns:
        Dictionary with ingestion + dbt results
    """
    logger = get_run_logger()
    logger.info("Starting weekly orchestration...")

    ingestion_results = fpl_complete_pipeline(
        include_player_details=include_player_details,
        include_live_gameweek=include_live_gameweek,
        max_players=max_players,
    )

    dbt_result = run_dbt_command(
        command=["dbt", "build"],
        project_dir=dbt_project_dir,
        profiles_dir=dbt_profiles_dir,
    )

    using_stale_data = False
    if not dbt_result.is_success:
        if allow_stale_data:
            using_stale_data = True
            logger.warning(
                "dbt build failed. Proceeding with stale data by design."
            )
        else:
            logger.error("dbt build failed and stale data is disabled.")

    return {
        "ingestion_results": ingestion_results,
        "dbt": {
            "is_success": dbt_result.is_success,
            "return_code": dbt_result.return_code,
        },
        "using_stale_data": using_stale_data,
    }
