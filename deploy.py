"""
Script to deploy the FPL Weekly Orchestration flow (Prefect 3.x).
"""
from flows.fpl_orchestration import fpl_weekly_orchestration

def deploy():
    """
    Deploy the weekly orchestration flow using Prefect 3.x serve() pattern.
    
    This will start a local server that:
    - Runs the flow on schedule (Every Tuesday at 9:00 AM)
    - Keeps the process running to execute scheduled runs
    - Can be stopped with Ctrl+C
    
    For production, consider using `flow.deploy()` with a work pool instead.
    """
    print("Starting FPL Weekly Orchestration deployment...")
    print("Schedule: Every Tuesday at 9:00 AM")
    print("Press Ctrl+C to stop the server")
    print("-" * 60)
    
    fpl_weekly_orchestration.serve(
        name="Weekly FPL Strategy Run",
        cron="0 9 * * 2",  # Every Tuesday at 9:00 AM
        tags=["production", "fpl"],
        parameters={
            "include_player_details": True,  # Fetch raw_element_summary (VARIANT)
            "dbt_project_dir": "/Users/am/Sync/fpl-workspace/fpl_development",
            "allow_stale_data": True
        }
    )

if __name__ == "__main__":
    deploy()
