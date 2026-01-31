"""
Main Prefect flow for FPL data ingestion.
"""
from prefect import flow, get_run_logger
from prefect.states import State
from typing import Dict, Any, Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import STATIC_ENDPOINTS, DYNAMIC_ENDPOINTS, get_snowflake_config
from tasks import (
    fetch_fpl_endpoint,
    extract_player_ids,
    extract_current_gameweek,
    fetch_dynamic_endpoint_batch,
    ensure_table_exists,
    load_to_snowflake,
    load_batch_to_snowflake,
)
from tasks.reporting_tasks import notify_source_failure


def ingestion_failure_hook(flow_run, flow_state: State) -> None:
    """
    Hook called when the ingestion flow fails.
    
    Sends a source failure alert to Slack.
    
    Args:
        flow_run: The flow run object
        flow_state: The failed state object
    """
    flow_name = flow_run.name or "FPL Ingestion"
    error_message = str(flow_state.message) if flow_state.message else "Unknown error"
    
    # Attempt to send failure notification
    try:
        notify_source_failure.fn(
            source_name=flow_name,
            error_message=error_message
        )
    except Exception as e:
        print(f"Failed to send Slack source failure notification: {e}")


@flow(name="FPL Static Data Ingestion", log_prints=True)
def ingest_static_endpoints() -> Dict[str, Any]:
    """
    Ingest all static FPL endpoints (bootstrap-static, fixtures, overall_league).
    
    Returns:
        Dictionary with results for each endpoint
    """
    logger = get_run_logger()
    logger.info("Starting FPL static data ingestion...")
    
    results = {}
    bootstrap_data = None
    
    for endpoint_name, config in STATIC_ENDPOINTS.items():
        try:
            logger.info(f"\nProcessing endpoint: {endpoint_name}")
            
            # Ensure table exists
            ensure_table_exists(config["table"])
            
            # Fetch data
            api_response = fetch_fpl_endpoint(
                url=config["url"],
                endpoint_name=endpoint_name
            )
            
            # Store bootstrap data for later use
            if endpoint_name == "bootstrap_static":
                bootstrap_data = api_response
            
            # Load to Snowflake
            load_success = load_to_snowflake(
                table_name=config["table"],
                api_response=api_response
            )
            
            results[endpoint_name] = {
                "status": "success",
                "loaded_to_snowflake": load_success,
                "data_size": len(str(api_response["data"]))
            }
            
        except Exception as e:
            logger.error(f"Failed to process {endpoint_name}: {e}")
            results[endpoint_name] = {
                "status": "failed",
                "error": str(e)
            }
    
    logger.info(f"\nStatic data ingestion complete!")
    logger.info(f"   Results: {results}")
    
    return {
        "results": results,
        "bootstrap_data": bootstrap_data
    }


@flow(name="FPL Player Details Ingestion", log_prints=True)
def ingest_player_details(
    bootstrap_data: Dict[str, Any],
    max_players: Optional[int] = None
) -> Dict[str, Any]:
    """
    Ingest detailed stats for all players.
    
    Args:
        bootstrap_data: Bootstrap-static data containing player IDs
        max_players: Optional limit on number of players to fetch (for testing)
        
    Returns:
        Dictionary with ingestion results
    """
    logger = get_run_logger()
    logger.info("Starting player details ingestion...")
    
    endpoint_config = DYNAMIC_ENDPOINTS["element_summary"]
    table_name = endpoint_config["table"]
    
    try:
        # Ensure table exists
        ensure_table_exists(table_name)
        
        # Extract player IDs
        player_ids = extract_player_ids(bootstrap_data)
        
        # Limit if specified (useful for testing)
        if max_players:
            player_ids = player_ids[:max_players]
            logger.info(f"Limited to first {max_players} players")
        
        # Fetch all player details
        player_responses = fetch_dynamic_endpoint_batch(
            url_template=endpoint_config["url_template"],
            ids=player_ids,
            endpoint_name="element_summary"
        )
        
        # Load to Snowflake in batch
        load_results = load_batch_to_snowflake(
            table_name=table_name,
            api_responses=player_responses
        )
        
        logger.info(f"Player details ingestion complete!")
        logger.info(f"   Loaded: {load_results['success']}/{load_results['total']} records")
        
        return {
            "status": "success",
            "load_results": load_results
        }
        
    except Exception as e:
        logger.error(f"Player details ingestion failed: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }


@flow(name="FPL Live Gameweek Ingestion", log_prints=True)
def ingest_live_gameweek(bootstrap_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ingest live gameweek data for the current/next gameweek.
    
    Args:
        bootstrap_data: Bootstrap-static data containing gameweek info
        
    Returns:
        Dictionary with ingestion results
    """
    logger = get_run_logger()
    logger.info("Starting live gameweek ingestion...")
    
    endpoint_config = DYNAMIC_ENDPOINTS["live_gameweek"]
    table_name = endpoint_config["table"]
    
    try:
        # Ensure table exists
        ensure_table_exists(table_name)
        
        # Extract current gameweek
        gameweek_id = extract_current_gameweek(bootstrap_data)
        
        if gameweek_id is None:
            logger.warning("No active gameweek found, skipping live data")
            return {
                "status": "skipped",
                "reason": "No active gameweek"
            }
        
        # Fetch live gameweek data
        url = endpoint_config["url_template"].format(gameweek_id=gameweek_id)
        api_response = fetch_fpl_endpoint(
            url=url,
            endpoint_name=f"live_gameweek_{gameweek_id}"
        )
        
        # Load to Snowflake
        load_success = load_to_snowflake(
            table_name=table_name,
            api_response=api_response
        )
        
        logger.info(f"Live gameweek ingestion complete!")
        
        return {
            "status": "success",
            "gameweek_id": gameweek_id,
            "loaded_to_snowflake": load_success
        }
        
    except Exception as e:
        logger.error(f"Live gameweek ingestion failed: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }


@flow(name="FPL Complete Data Pipeline", log_prints=True)
def fpl_complete_pipeline(
    include_player_details: bool = True,
    include_live_gameweek: bool = True,
    max_players: Optional[int] = None
) -> Dict[str, Any]:
    """
    Complete FPL data ingestion pipeline.
    
    This is the main entry point that orchestrates all data ingestion.
    
    Args:
        include_player_details: Whether to fetch detailed player stats
        include_live_gameweek: Whether to fetch live gameweek data
        max_players: Optional limit on players (for testing)
        
    Returns:
        Dictionary with complete pipeline results
    """
    logger = get_run_logger()
    
    print("\n" + "="*60)
    print("FPL DATA PIPELINE")
    print("="*60 + "\n")
    
    # Check Snowflake configuration
    snowflake_config = get_snowflake_config()
    if snowflake_config:
        logger.info("Snowflake configured - data will be loaded to database")
    else:
        logger.info("Snowflake not configured - data will be fetched but not loaded")
    
    pipeline_results = {}
    
    # Step 1: Ingest static endpoints (always run)
    print("\nSTEP 1: Ingesting static endpoints...")
    static_results = ingest_static_endpoints()
    pipeline_results["static_endpoints"] = static_results["results"]
    bootstrap_data = static_results["bootstrap_data"]
    
    # Step 2: Ingest player details (optional)
    if include_player_details and bootstrap_data:
        print("\nSTEP 2: Ingesting player details...")
        player_results = ingest_player_details(
            bootstrap_data=bootstrap_data,
            max_players=max_players
        )
        pipeline_results["player_details"] = player_results
    else:
        logger.info("Skipping player details ingestion")
        pipeline_results["player_details"] = {"status": "skipped"}
    
    # Step 3: Ingest live gameweek (optional)
    if include_live_gameweek and bootstrap_data:
        print("\nSTEP 3: Ingesting live gameweek data...")
        gameweek_results = ingest_live_gameweek(bootstrap_data)
        pipeline_results["live_gameweek"] = gameweek_results
    else:
        logger.info("Skipping live gameweek ingestion")
        pipeline_results["live_gameweek"] = {"status": "skipped"}
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60 + "\n")
    
    return pipeline_results


# =============================================================================
# NEW TYPED FLOWS (Using typed columns instead of VARIANT)
# =============================================================================

from tasks.transformation_tasks import (
    parse_players_from_bootstrap,
    parse_teams_from_bootstrap,
    parse_gameweeks_from_bootstrap,
    parse_fixtures,
)
from tasks.snowflake_tasks import (
    ensure_typed_table_exists,
    load_typed_records_to_snowflake,
    load_typed_records_append_to_snowflake,
)


@flow(name="FPL Typed Static Data Ingestion", log_prints=True)
def ingest_static_endpoints_typed() -> Dict[str, Any]:
    """
    Ingest static FPL endpoints using typed tables (not VARIANT).
    
    This is the NEW approach: JSON -> Parse -> Typed columns
    
    Returns:
        Dictionary with results for each endpoint/table
    """
    logger = get_run_logger()
    logger.info("Starting FPL static data ingestion (TYPED)...")
    
    results = {}
    bootstrap_response = None
    
    # Step 1: Fetch bootstrap-static (contains players, teams, gameweeks)
    logger.info("\nFetching bootstrap-static...")
    try:
        bootstrap_response = fetch_fpl_endpoint(
            url=STATIC_ENDPOINTS["bootstrap_static"]["url"],
            endpoint_name="bootstrap_static"
        )
        
        # Get current gameweek for metadata
        current_gw = extract_current_gameweek(bootstrap_response)
        
        # Parse into separate entity lists
        players = parse_players_from_bootstrap(bootstrap_response, gameweek_id=current_gw)
        teams = parse_teams_from_bootstrap(bootstrap_response)
        gameweeks = parse_gameweeks_from_bootstrap(bootstrap_response)
        
        # Load players (current state - MERGE)
        logger.info("\nLoading players to Snowflake...")
        ensure_typed_table_exists("players")
        player_result = load_typed_records_to_snowflake("players", players)
        results["players"] = {
            "status": "success",
            "records": player_result["loaded"],
            "total": player_result["total"]
        }
        
        # Load players gameweek snapshot (historical - APPEND)
        logger.info("\nAppending players gameweek snapshot to Snowflake...")
        ensure_typed_table_exists("players_gameweek_snapshot")
        snapshot_result = load_typed_records_append_to_snowflake("players_gameweek_snapshot", players)
        results["players_gameweek_snapshot"] = {
            "status": "success",
            "records": snapshot_result["loaded"],
            "total": snapshot_result["total"]
        }
        
        # Load teams
        logger.info("\nLoading teams to Snowflake...")
        ensure_typed_table_exists("teams")
        team_result = load_typed_records_to_snowflake("teams", teams)
        results["teams"] = {
            "status": "success",
            "records": team_result["loaded"],
            "total": team_result["total"]
        }
        
        # Load gameweeks
        logger.info("\nLoading gameweeks to Snowflake...")
        ensure_typed_table_exists("gameweeks")
        gameweek_result = load_typed_records_to_snowflake("gameweeks", gameweeks)
        results["gameweeks"] = {
            "status": "success",
            "records": gameweek_result["loaded"],
            "total": gameweek_result["total"]
        }
        
    except Exception as e:
        logger.error(f"Failed to process bootstrap-static: {e}")
        results["bootstrap_static"] = {"status": "failed", "error": str(e)}
        bootstrap_response = None
    
    # Step 2: Fetch and load fixtures
    logger.info("\nFetching fixtures...")
    try:
        fixtures_response = fetch_fpl_endpoint(
            url=STATIC_ENDPOINTS["fixtures"]["url"],
            endpoint_name="fixtures"
        )
        
        fixtures = parse_fixtures(fixtures_response)
        
        logger.info("\nLoading fixtures to Snowflake...")
        ensure_typed_table_exists("fixtures")
        fixture_result = load_typed_records_to_snowflake("fixtures", fixtures)
        results["fixtures"] = {
            "status": "success",
            "records": fixture_result["loaded"],
            "total": fixture_result["total"]
        }
        
    except Exception as e:
        logger.error(f"Failed to process fixtures: {e}")
        results["fixtures"] = {"status": "failed", "error": str(e)}
    
    logger.info("\nTyped data ingestion complete!")
    logger.info(f"   Results: {results}")
    
    return {
        "results": results,
        "bootstrap_data": bootstrap_response,
    }


@flow(
    name="FPL Complete Typed Pipeline",
    log_prints=True,
    on_failure=[ingestion_failure_hook]
)
def fpl_typed_pipeline(
    include_player_details: bool = True,
    max_players: Optional[int] = None
) -> Dict[str, Any]:
    """
    Complete FPL data ingestion pipeline using HYBRID approach:
    - Typed tables (MERGE/UPSERT): players, teams, gameweeks, fixtures
    - VARIANT (INSERT): raw_element_summary (complex player history)
    
    Args:
        include_player_details: Whether to fetch detailed player history (VARIANT)
        max_players: Optional limit on players (for testing)
    
    Returns:
        Dictionary with complete pipeline results
    """
    logger = get_run_logger()
    
    logger.info("=" * 60)
    logger.info("FPL HYBRID DATA PIPELINE")
    logger.info("=" * 60)
    
    # Check Snowflake configuration
    snowflake_config = get_snowflake_config()
    if snowflake_config:
        logger.info("Snowflake configured - data will be loaded")
        logger.info("  - Typed tables (MERGE): players, teams, gameweeks, fixtures")
        logger.info("  - Typed tables (APPEND): players_gameweek_snapshot")
        logger.info("  - VARIANT (INSERT): raw_element_summary")
    else:
        logger.info("Snowflake not configured - data will be fetched but not loaded")
    
    pipeline_results = {}
    
    # Step 1: Ingest static endpoints to TYPED tables (with MERGE/UPSERT)
    logger.info("STEP 1: Ingesting static endpoints (typed with MERGE)...")
    typed_payload = ingest_static_endpoints_typed()
    typed_results = typed_payload["results"]
    bootstrap_response = typed_payload.get("bootstrap_data")
    pipeline_results["typed_tables"] = typed_results
    
    # Step 2: Ingest player details to VARIANT table (raw_element_summary)
    if include_player_details and bootstrap_response:
        logger.info("STEP 2: Ingesting player details (VARIANT)...")

        player_results = ingest_player_details(
            bootstrap_data=bootstrap_response,
            max_players=max_players
        )
        pipeline_results["player_details"] = player_results
    else:
        if include_player_details:
            logger.warning("Skipping player details ingestion - bootstrap data unavailable")
        else:
            logger.info("Skipping player details ingestion")
        pipeline_results["player_details"] = {"status": "skipped"}
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 60)
    
    # Print summary
    logger.info("Summary - Typed Tables:")
    for table, result in typed_results.items():
        if result["status"] == "success":
            logger.info(
                "  [SUCCESS] %s: %s/%s records",
                table,
                result["records"],
                result["total"],
            )
        else:
            logger.error("  [FAILED] %s: %s", table, result.get("error", "Unknown error"))
    
    if include_player_details:
        logger.info("Summary - VARIANT Tables:")
        pd_result = pipeline_results["player_details"]
        if pd_result["status"] == "success":
            load_res = pd_result["load_results"]
            logger.info(
                "  [SUCCESS] raw_element_summary: %s/%s records",
                load_res["success"],
                load_res["total"],
            )
        else:
            logger.error("  [FAILED] raw_element_summary")
    
    return pipeline_results


if __name__ == "__main__":
    # Run the hybrid pipeline (typed tables + raw_element_summary)
    fpl_typed_pipeline(include_player_details=True, max_players=None)
