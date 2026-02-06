"""
Simple test runner for FPL pipeline - runs once without scheduling.

Use this to test the pipeline before deploying with a schedule.
"""
import sys
import os

# Add parent directory to path to allow imports from flows, tasks, etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flows.fpl_orchestration import fpl_weekly_orchestration
from tasks.ml_tasks import fetch_training_data, prepare_inference_data, run_ml_inference
from tasks.optimizer_tasks import optimize_squad_task

def run_pipeline():
    """
    Execute the FPL pipeline once for testing.
    """
    fast_mode = os.getenv("RUN_ONCE_FAST", "").lower() in {"1", "true", "yes"}

    print("=" * 60)
    print("FPL Pipeline Test Run")
    print("=" * 60)

    if fast_mode:
        print("FAST MODE: Running ML inference + optimization only")
        feature_df = fetch_training_data.fn()
        prepared_df = prepare_inference_data.fn(feature_df)
        predictions = run_ml_inference.fn(prepared_df)
        recommended_squad = optimize_squad_task.fn(predictions)
        result = {
            "prediction_count": len(predictions),
            "squad_size": len(recommended_squad),
        }
    else:
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
