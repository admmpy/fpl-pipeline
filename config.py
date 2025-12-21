"""
Configuration for FPL data pipeline.
"""

import os
from dotenv import load_dotenv
from typing import Dict, List, Optional

# Load environment variables
load_dotenv()

# FPL API Configuration
FPL_BASE_URL = "https://fantasy.premierleague.com/api" 

# Static endpoints (can be called directly)
STATIC_ENDPOINTS = {
    "bootstrap_static": {
        "url": f"{FPL_BASE_URL}/bootstrap-static/",
        "description": "Core static data (players, teams, gameweeks)",
        "table": "raw_bootstrap_static",
    },
    "fixtures": {
        "url": f"{FPL_BASE_URL}/fixtures/",
        "description": "Match fixtures and results",
        "table": "raw_fixtures",
    },
    "overall_league": {
        "url": f"{FPL_BASE_URL}/leagues-classic/314/standings/",
        "description": "Official FPL overall league standings",
        "table": "raw_overall_league",
    },
}

# Dynamic endpoints (require parameters)
DYNAMIC_ENDPOINTS = {
    "element_summary": {
        "url_template": f"{FPL_BASE_URL}/element-summary/{{player_id}}/",
        "description": "Player detailed stats",
        "table": "raw_element_summary",
        "params": ["player_id"],
        "id_source": "bootstrap_static",  # Where to get player IDs from
    },
    "live_gameweek": {
        "url_template": f"{FPL_BASE_URL}/event/{{gameweek_id}}/live/",
        "description": "Live gameweek data",
        "table": "raw_live_gameweek",
        "params": ["gameweek_id"],
        "id_source": "bootstrap_static",  # Get current gameweek from here
    },
}

# Snowflake Configuration
def get_snowflake_config() -> Optional[Dict[str, str]]:
    """
    Load Snowflake configuration from environment.
    
    Returns:
        Dictionary with Snowflake connection parameters if all are present,
        None if any are missing (allows pipeline to continue without Snowflake)
    """
    config = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "database": os.getenv("SNOWFLAKE_DATABASE"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    }
    
    # Check if all required variables are present
    missing = [k for k, v in config.items() if v is None]
    if missing:
        print(f"WARNING: Missing Snowflake config: {', '.join([m.upper() for m in missing])}")
        print("Pipeline will continue without Snowflake loading.")
        return None
    
    return config

# Pipeline Settings
RATE_LIMIT_DELAY = 0.5  # Seconds between API calls
MAX_RETRIES = 3
RETRY_DELAY = 5  # Seconds
