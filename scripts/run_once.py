"""
Simple test runner for FPL pipeline - runs once without scheduling.

Use this to test the pipeline before deploying with a schedule.
"""
import sys
import os

# Add parent directory to path to allow imports from flows, tasks, etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flows.fpl_orchestration import fpl_weekly_orchestration

def run_pipeline():
    """
    Execute the FPL pipeline once for testing.
    """
    print("=" * 60)
    print("FPL Pipeline Test Run")
    print("=" * 60)
    
    result = fpl_weekly_orchestration(
        include_player_details=True,  # Fetch raw_element_summary (VARIANT)
        dbt_project_dir="/Users/am/Sync/fpl-workspace/fpl_development",
        allow_stale_data=True
    )
    
    print("\n" + "=" * 60)
    print("Pipeline execution complete!")
    print("=" * 60)
    print(f"Result: {result}")
    
    return result

if __name__ == "__main__":
    run_pipeline()
