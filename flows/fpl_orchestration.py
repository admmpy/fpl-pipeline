"""
Prefect orchestration for ingestion + dbt transformations.
"""
from __future__ import annotations

from typing import Dict, Any, Optional

from prefect import flow, get_run_logger
from prefect.states import State

from flows.fpl_ingestion import fpl_typed_pipeline  # Use TYPED pipeline with MERGE/UPSERT
from tasks.dbt_tasks import run_dbt_command
from tasks.ml_tasks import fetch_training_data, prepare_inference_data, run_ml_inference
from tasks.optimizer_tasks import optimize_squad_task
from tasks.reporting_tasks import (
    format_squad_summary,
    send_slack_notification,
    notify_pipeline_failure,
    notify_pipeline_success,
)
from tasks.snowflake_tasks import load_typed_records_to_snowflake, ensure_typed_table_exists


def pipeline_failure_hook(flow, flow_run, flow_state: State) -> None:
    """
    Hook called when the pipeline flow fails.
    
    Sends a failure alert to Slack with error details.
    
    Args:
        flow: The flow object
        flow_run: The flow run object
        flow_state: The failed state object
    """
    flow_name = flow_run.name or "FPL Pipeline"
    error_message = str(flow_state.message) if flow_state.message else "Unknown error"
    
    context = {
        "flow_run_id": str(flow_run.id),
        "flow_run_name": flow_run.name,
    }
    
    # Attempt to send failure notification
    try:
        notify_pipeline_failure.fn(
            flow_name=flow_name,
            error_message=error_message,
            context=context
        )
    except Exception as e:
        print(f"Failed to send Slack failure notification: {e}")


def pipeline_success_hook(flow, flow_run, flow_state: State) -> None:
    """
    Hook called when the pipeline flow succeeds.
    
    Optionally sends a success summary to Slack if notify_on_success is True.
    
    Args:
        flow: The flow object
        flow_run: The flow run object
        flow_state: The success state object
    """
    params = flow_run.parameters or {}
    notify_on_success = params.get("notify_on_success", False)
    slack_webhook_url = params.get("slack_webhook_url")
    
    if notify_on_success:
        try:
            # Access metrics from the flow return value
            result = flow_state.result()
            metrics = result.get("metrics", {})
            
            notify_pipeline_success.fn(
                flow_name="FPL Weekly Orchestration",
                metrics=metrics,
                webhook_url=slack_webhook_url
            )
        except Exception as e:
            print(f"Failed to send Slack success notification in hook: {e}")


@flow(
    name="FPL Weekly Orchestration",
    log_prints=True,
    on_failure=[pipeline_failure_hook],
    on_completion=[pipeline_success_hook]
)
def fpl_weekly_orchestration(
    include_player_details: bool = True,
    max_players: Optional[int] = None,
    dbt_project_dir: str = "/Users/am/Sync/fpl-workspace/fpl_development",
    dbt_profiles_dir: Optional[str] = None,
    allow_stale_data: bool = True,
    slack_webhook_url: Optional[str] = None,
    notify_on_success: bool = False,
) -> Dict[str, Any]:
    """
    Orchestrate FPL ingestion, dbt transformations, ML inference, and optimization.
    
    Uses HYBRID pipeline:
    - TYPED tables with MERGE/UPSERT: players, teams, gameweeks, fixtures
    - VARIANT with INSERT: raw_element_summary (player history)
    
    Automatic Slack notifications:
    - Pipeline failures: Always notified via on_failure hook
    - Source failures: Notified if critical sources fail
    - Success: Optional (set notify_on_success=True)
    
    Args:
        include_player_details: Whether to fetch raw_element_summary (VARIANT)
        max_players: Optional limit for testing
        dbt_project_dir: Path to dbt project
        dbt_profiles_dir: Optional dbt profiles directory
        allow_stale_data: Continue if dbt fails
        slack_webhook_url: Optional Slack webhook for notifications
        notify_on_success: Whether to send success notification
    
    Raises:
        Exception: If critical pipeline step fails
    """
    logger = get_run_logger()
    logger.info("Starting weekly orchestration...")
    
    pipeline_metrics = {
        "steps_completed": [],
        "steps_failed": [],
    }

    # 1. Ingestion (HYBRID: typed tables + VARIANT player history)
    try:
        logger.info("Step 1: Running FPL data ingestion...")
        ingestion_results = fpl_typed_pipeline(
            include_player_details=include_player_details,
            max_players=max_players
        )
        
        # Check for critical ingestion failures
        typed_results = ingestion_results.get("typed_tables", {})
        if typed_results.get("players", {}).get("status") == "failed":
            error_msg = typed_results["players"].get("error", "Unknown error")
            notify_pipeline_failure.fn(
                flow_name="FPL Ingestion - Players",
                error_message=f"Critical source failed: {error_msg}",
                context={"step": "ingestion", "source": "bootstrap_static (players)"},
                webhook_url=slack_webhook_url
            )
            raise ValueError(f"Critical ingestion failure: Players table failed - {error_msg}")
        
        pipeline_metrics["steps_completed"].append("ingestion")
        logger.info("Ingestion completed successfully")
        
    except Exception as e:
        pipeline_metrics["steps_failed"].append("ingestion")
        logger.error(f"Ingestion failed: {e}")
        raise

    # 2. Transformations (dbt)
    try:
        logger.info("Step 2: Running dbt transformations...")
        dbt_result = run_dbt_command(
            command=["dbt", "build"],
            project_dir=dbt_project_dir,
            profiles_dir=dbt_profiles_dir,
        )

        if not dbt_result.is_success:
            error_msg = f"dbt build failed with return code {dbt_result.return_code}"
            
            if not allow_stale_data:
                notify_pipeline_failure.fn(
                    flow_name="FPL Orchestration - dbt",
                    error_message=error_msg,
                    context={"step": "dbt", "allow_stale_data": False},
                    webhook_url=slack_webhook_url
                )
                logger.error("dbt build failed and stale data is disabled. Stopping.")
                raise ValueError(error_msg)
            else:
                logger.warning(f"{error_msg} - Continuing with stale data")
                pipeline_metrics["steps_completed"].append("dbt (with errors)")
        else:
            pipeline_metrics["steps_completed"].append("dbt")
            logger.info("dbt transformations completed successfully")
            
    except Exception as e:
        pipeline_metrics["steps_failed"].append("dbt")
        logger.error(f"dbt transformations failed: {e}")
        if not allow_stale_data:
            raise

    # 3. ML Inference
    predictions = []
    try:
        logger.info("Step 3: Running ML inference...")
        feature_df = fetch_training_data()
        prepared_df = prepare_inference_data(feature_df)
        predictions = run_ml_inference(prepared_df)
        pipeline_metrics["steps_completed"].append("ml_inference")
        pipeline_metrics["prediction_count"] = len(predictions)
        logger.info(f"ML inference completed: {len(predictions)} predictions")

    except Exception as e:
        pipeline_metrics["steps_failed"].append("ml_inference")
        logger.error(f"ML inference failed: {e}")
        logger.warning("Skipping optimization and loading — dashboard will use previous data")

    # 4-6: Only run if predictions were generated
    load_result = {}
    recommended_squad = []
    if predictions:
        # 4. Optimization
        try:
            logger.info("Step 4: Running squad optimization...")
            recommended_squad = optimize_squad_task(predictions)
            pipeline_metrics["steps_completed"].append("optimization")
            pipeline_metrics["squad_size"] = len(recommended_squad)
            logger.info(f"Optimization completed: {len(recommended_squad)} players selected")

        except Exception as e:
            pipeline_metrics["steps_failed"].append("optimization")
            logger.error(f"Optimization failed: {e}")
            raise

        # 5. Load Recommendations to Snowflake
        try:
            logger.info("Step 5: Loading recommendations to Snowflake...")
            ensure_typed_table_exists("recommended_squad")
            load_result = load_typed_records_to_snowflake(
                table_name="recommended_squad",
                records=recommended_squad
            )
            pipeline_metrics["steps_completed"].append("load_recommendations")
            pipeline_metrics["records_loaded"] = load_result.get("loaded", 0)
            logger.info(f"Loaded {load_result.get('loaded', 0)} recommendations to Snowflake")

        except Exception as e:
            pipeline_metrics["steps_failed"].append("load_recommendations")
            logger.error(f"Loading recommendations failed: {e}")
            raise

        # 5b. Refresh reporting models after recommendations load
        try:
            logger.info("Step 5b: Refreshing reporting models...")
            dbt_post_result = run_dbt_command(
                command=[
                    "dbt",
                    "build",
                    "--select",
                    "fct_model_analysis rprt_squad_performance_comparison rprt_squad_performance_summary",
                ],
                project_dir=dbt_project_dir,
                profiles_dir=dbt_profiles_dir,
            )
            if not dbt_post_result.is_success:
                logger.warning(
                    f"Post-load dbt build failed with return code {dbt_post_result.return_code}"
                )
                pipeline_metrics["steps_completed"].append("dbt_post_load (with errors)")
            else:
                pipeline_metrics["steps_completed"].append("dbt_post_load")
                logger.info("Reporting models refreshed successfully")
        except Exception as e:
            pipeline_metrics["steps_failed"].append("dbt_post_load")
            logger.warning(f"Post-load dbt build failed (non-critical): {e}")

        # 6. Reporting
        try:
            logger.info("Step 6: Generating reports...")
            summary = format_squad_summary(recommended_squad)
            send_slack_notification(summary, webhook_url=slack_webhook_url)
            pipeline_metrics["steps_completed"].append("reporting")
            logger.info("Reports generated and sent")

        except Exception as e:
            # Reporting failures should not stop the pipeline
            logger.warning(f"Reporting failed (non-critical): {e}")
            pipeline_metrics["steps_completed"].append("reporting (with errors)")
    else:
        logger.warning("No predictions available — skipping optimization, load, and reporting")
        pipeline_metrics["steps_completed"].append("skipped_downstream (no predictions)")

    logger.info("Weekly orchestration completed successfully!")

    return {
        "ingestion_results": ingestion_results,
        "dbt": {
            "is_success": dbt_result.is_success,
            "return_code": dbt_result.return_code,
        },
        "ml_load_result": load_result,
        "metrics": pipeline_metrics,
        "status": "success"
    }
