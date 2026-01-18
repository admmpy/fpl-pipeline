"""
Script to deploy the FPL Weekly Orchestration flow.
"""
from flows.fpl_orchestration import fpl_weekly_orchestration
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

def deploy():
    """
    Create a deployment for the weekly orchestration flow.
    """
    deployment = Deployment.build_from_flow(
        flow=fpl_weekly_orchestration,
        name="Weekly FPL Strategy Run",
        schedule=CronSchedule(cron="0 9 * * 2"), # Every Tuesday at 9:00 AM
        tags=["production", "fpl"],
        work_queue_name="local-mac-worker",
        parameters={
            "allow_stale_data": True,
            "dbt_project_dir": "/Users/am/Sync/fpl-workspace/fpl_development"
        }
    )
    deployment.apply()
    print("Deployment 'Weekly FPL Strategy Run' created successfully.")

if __name__ == "__main__":
    deploy()
