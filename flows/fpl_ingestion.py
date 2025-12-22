"""
Main Prefect flow for FPL data ingestion.
"""
from prefect import flow, get_run_logger
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
        
        # Load players
        logger.info("\nLoading players to Snowflake...")
        ensure_typed_table_exists("players")
        player_result = load_typed_records_to_snowflake("players", players)
        results["players"] = {
            "status": "success",
            "records": player_result["loaded"],
            "total": player_result["total"]
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
    
    return results


@flow(name="FPL Complete Typed Pipeline", log_prints=True)
def fpl_typed_pipeline() -> Dict[str, Any]:
    """
    Complete FPL data ingestion pipeline using TYPED tables.
    
    This is the new approach that replaces VARIANT columns with proper types.
    
    Returns:
        Dictionary with complete pipeline results
    """
    logger = get_run_logger()
    
    print("\n" + "="*60)
    print("FPL TYPED DATA PIPELINE")
    print("="*60 + "\n")
    
    # Check Snowflake configuration
    snowflake_config = get_snowflake_config()
    if snowflake_config:
        logger.info("Snowflake configured - data will be loaded to typed tables")
    else:
        logger.info("Snowflake not configured - data will be fetched but not loaded")
    
    # Ingest static endpoints (players, teams, gameweeks, fixtures)
    print("\nSTEP 1: Ingesting static endpoints (typed)...")
    results = ingest_static_endpoints_typed()
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60 + "\n")
    
    # Print summary
    print("Summary:")
    for table, result in results.items():
        if result["status"] == "success":
            print(f"  [SUCCESS] {table}: {result['records']}/{result['total']} records")
        else:
            print(f"  [FAILED] {table}: {result.get('error', 'Unknown error')}")
    
    return results


if __name__ == "__main__":
    # Run the complete pipeline
    # For testing, you can limit player details:
    # fpl_complete_pipeline(max_players=10)
    
    # Run OLD VARIANT approach:
    # fpl_complete_pipeline()
    
    # Run NEW TYPED approach:
    fpl_typed_pipeline()
