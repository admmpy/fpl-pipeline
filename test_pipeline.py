#!/usr/bin/env python3
"""
Test script for FPL pipeline.

This script allows you to test the pipeline without Snowflake configuration.
"""
from flows.fpl_ingestion import fpl_complete_pipeline


def test_api_only():
    """Test API fetching without Snowflake (useful for development)."""
    print("\nTesting API fetching only (no Snowflake required)...\n")
    
    # Run pipeline with limited player details for faster testing
    results = fpl_complete_pipeline(
        include_player_details=True,
        include_live_gameweek=True,
        max_players=5  # Only fetch 5 players for testing
    )
    
    print("\nTest Results:")
    print("-" * 60)
    
    # Static endpoints
    print("\n Static Endpoints:")
    for endpoint, result in results["static_endpoints"].items():
        status = "[SUCCESS]" if result["status"] == "success" else "[FAILED]"
        print(f"   {status} {endpoint}: {result['status']}")
    
    # Player details
    if results["player_details"]["status"] != "skipped":
        print(f"\n Player Details:")
        if results["player_details"]["status"] == "success":
            load_results = results["player_details"]["load_results"]
            print(f"   [SUCCESS] Fetched {load_results['total']} players")
        else:
            print(f"   [FAILED] {results['player_details'].get('error')}")
    
    # Live gameweek
    if results["live_gameweek"]["status"] != "skipped":
        print(f"\n Live Gameweek:")
        if results["live_gameweek"]["status"] == "success":
            gw_id = results["live_gameweek"]["gameweek_id"]
            print(f"   [SUCCESS] Fetched gameweek {gw_id}")
        else:
            print(f"   [FAILED] {results['live_gameweek'].get('error')}")
    
    print("\n" + "-" * 60)
    print("Test complete!\n")


def test_full_pipeline():
    """Test complete pipeline with all players (requires Snowflake)."""
    print("\nTesting full pipeline with Snowflake...\n")
    
    results = fpl_complete_pipeline(
        include_player_details=True,
        include_live_gameweek=True,
        max_players=None  # Fetch all players
    )
    
    print("\nFull pipeline test complete!\n")
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        test_full_pipeline()
    else:
        test_api_only()
