"""
Prefect orchestration for ingestion + dbt transformations.
"""
from __future__ import annotations

from typing import Dict, Any, Optional

from prefect import flow, get_run_logger

from flows.fpl_ingestion import fpl_typed_pipeline  # Use TYPED pipeline with MERGE/UPSERT
from tasks.dbt_tasks import run_dbt_command
from tasks.ml_tasks import fetch_training_data, prepare_inference_data, run_ml_inference
from tasks.optimizer_tasks import optimize_squad_task
from tasks.reporting_tasks import format_squad_summary, send_slack_notification
from tasks.snowflake_tasks import load_typed_records_to_snowflake


@flow(name="FPL Weekly Orchestration", log_prints=True)
def fpl_weekly_orchestration(
    include_player_details: bool = True,
    max_players: Optional[int] = None,
    dbt_project_dir: str = "/Users/am/Sync/fpl-workspace/fpl_development",
    dbt_profiles_dir: Optional[str] = None,
    allow_stale_data: bool = True,
    slack_webhook_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Orchestrate FPL ingestion, dbt transformations, ML inference, and optimization.
    
    Uses HYBRID pipeline:
    - TYPED tables with MERGE/UPSERT: players, teams, gameweeks, fixtures
    - VARIANT with INSERT: raw_element_summary (player history)
    
    Args:
        include_player_details: Whether to fetch raw_element_summary (VARIANT)
        max_players: Optional limit for testing
        dbt_project_dir: Path to dbt project
        dbt_profiles_dir: Optional dbt profiles directory
        allow_stale_data: Continue if dbt fails
        slack_webhook_url: Optional Slack webhook for notifications
    """
    logger = get_run_logger()
    logger.info("Starting weekly orchestration...")

    # 1. Ingestion (HYBRID: typed tables + VARIANT player history)
    ingestion_results = fpl_typed_pipeline(
        include_player_details=include_player_details,
        max_players=max_players
    )

    # 2. Transformations (dbt)
    dbt_result = run_dbt_command(
        command=["dbt", "build"],
        project_dir=dbt_project_dir,
        profiles_dir=dbt_profiles_dir,
    )

    if not dbt_result.is_success and not allow_stale_data:
        logger.error("dbt build failed and stale data is disabled. Stopping.")
        return {"status": "failed", "step": "dbt"}

    # 3. ML Inference
    feature_df = fetch_training_data()
    prepared_df = prepare_inference_data(feature_df)
    predictions = run_ml_inference(prepared_df)

    # 4. Optimization
    recommended_squad = optimize_squad_task(predictions)

    # 5. Load Recommendations to Snowflake
    # Note: User manages the table creation for recommended_squad
    load_result = load_typed_records_to_snowflake(
        table_name="recommended_squad",
        records=recommended_squad
    )

    # 6. Reporting
    summary = format_squad_summary(recommended_squad)
    send_slack_notification(summary, webhook_url=slack_webhook_url)

    return {
        "ingestion_results": ingestion_results,
        "dbt": {
            "is_success": dbt_result.is_success,
            "return_code": dbt_result.return_code,
        },
        "ml_load_result": load_result,
        "status": "success"
    }
