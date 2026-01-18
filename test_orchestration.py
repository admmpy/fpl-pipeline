from flows.fpl_orchestration import fpl_weekly_orchestration
from dotenv import load_dotenv
import os

if __name__ == "__main__":
    load_dotenv()
    # Run a test of the flow with minimal data for speed
    print("Starting test run of FPL Weekly Orchestration...")
    result = fpl_weekly_orchestration(
        include_player_details=True,
        include_live_gameweek=True,
        max_players=10,  # Small sample for testing
        allow_stale_data=True
    )
    print(f"Test run complete. Status: {result.get('status')}")
