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

# =============================================================================
# SNOWFLAKE TABLE SCHEMAS (Typed Columns - Direct INSERT Pattern)
# =============================================================================

# Player Schema - Essential fields for FPL analysis
PLAYER_SCHEMA = {
    # Identity & Basic Info
    "player_id": "INTEGER NOT NULL",
    "web_name": "VARCHAR(100)",
    "first_name": "VARCHAR(100)",
    "second_name": "VARCHAR(100)",
    "team_id": "INTEGER",
    "position_id": "INTEGER",  # element_type in API (1=GK, 2=DEF, 3=MID, 4=FWD)
    
    # Cost & Value
    "now_cost": "INTEGER",  # In 0.1m increments (e.g., 105 = £10.5m)
    "selected_by_percent": "FLOAT",
    "form": "FLOAT",
    "value_season": "FLOAT",
    "value_form": "FLOAT",
    
    # Core Stats
    "total_points": "INTEGER",
    "minutes": "INTEGER",
    "goals_scored": "INTEGER",
    "assists": "INTEGER",
    "clean_sheets": "INTEGER",
    "goals_conceded": "INTEGER",
    "bonus": "INTEGER",
    "bps": "INTEGER",
    
    # Status
    "status": "VARCHAR(10)",
    "chance_of_playing_next_round": "INTEGER",
    
    # Metadata
    "ingestion_timestamp": "TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()",
    "gameweek_fetched": "INTEGER",
}

# Team Schema
TEAM_SCHEMA = {
    "team_id": "INTEGER NOT NULL",
    "name": "VARCHAR(100)",
    "short_name": "VARCHAR(10)",
    "strength": "INTEGER",
    "form": "FLOAT",
    "points": "INTEGER",
    "position": "INTEGER",
    "strength_overall_home": "INTEGER",
    "strength_overall_away": "INTEGER",
    "strength_attack_home": "INTEGER",
    "strength_attack_away": "INTEGER",
    "strength_defence_home": "INTEGER",
    "strength_defence_away": "INTEGER",
    "ingestion_timestamp": "TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()",
}

# Gameweek Schema
GAMEWEEK_SCHEMA = {
    "gameweek_id": "INTEGER NOT NULL",
    "name": "VARCHAR(50)",
    "deadline_time": "TIMESTAMP_NTZ",
    "finished": "BOOLEAN",
    "is_current": "BOOLEAN",
    "is_next": "BOOLEAN",
    "is_previous": "BOOLEAN",
    "average_entry_score": "INTEGER",
    "highest_score": "INTEGER",
    "most_selected": "INTEGER",
    "most_transferred_in": "INTEGER",
    "most_captained": "INTEGER",
    "most_vice_captained": "INTEGER",
    "ingestion_timestamp": "TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()",
}

# Fixture Schema
FIXTURE_SCHEMA = {
    "fixture_id": "INTEGER NOT NULL",
    "gameweek_id": "INTEGER",
    "kickoff_time": "TIMESTAMP_NTZ",
    "team_h": "INTEGER",
    "team_a": "INTEGER",
    "team_h_score": "INTEGER",
    "team_a_score": "INTEGER",
    "finished": "BOOLEAN",
    "started": "BOOLEAN",
    "ingestion_timestamp": "TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()",
}

# Map table names to their schemas
TABLE_SCHEMAS = {
    "players": PLAYER_SCHEMA,
    "teams": TEAM_SCHEMA,
    "gameweeks": GAMEWEEK_SCHEMA,
    "fixtures": FIXTURE_SCHEMA,
}


def generate_create_table_sql(table_name: str, schema: Dict[str, str]) -> str:
    """
    Generate CREATE TABLE SQL statement from schema definition.
    
    Args:
        table_name: Name of the table
        schema: Dictionary mapping column names to their types
        
    Returns:
        SQL CREATE TABLE statement
    """
    columns = []
    primary_key = None
    
    for column_name, column_type in schema.items():
        columns.append(f"    {column_name} {column_type}")
        
        # Identify primary key (first NOT NULL field)
        if "NOT NULL" in column_type and primary_key is None:
            primary_key = column_name
    
    # Add primary key constraint if found
    if primary_key:
        columns.append(f"    PRIMARY KEY ({primary_key})")
    
    sql = f"""CREATE TABLE IF NOT EXISTS {table_name} (
{',\n'.join(columns)}
);"""
    
    return sql

# =============================================================================
# OPTIMIZATION CONFIGURATION
# =============================================================================

OPTIMIZATION_CONFIG = {
    "optimization": {
        "budget": 100.0,
        "solver": "ECOS_BB",
        "squad_constraints": {
            "total_players": 15,
            "goalkeeper_count": 2,
            "defender_count": 5,
            "midfielder_count": 5,
            "forward_count": 3,
            "max_per_team": 3,
        },
        "starting_constraints": {
            "total_players": 11,
            "min_goalkeeper": 1,
            "min_defender": 3,
            "min_midfielder": 2,
            "min_forward": 1,
        },
        "transfer_penalty": 4.0,
        "max_transfers": 15,  # Wildcard mode by default
    }
}
