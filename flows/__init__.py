"""
Flows package for FPL pipeline.
"""
from .fpl_ingestion import (
    ingest_static_endpoints,
    ingest_player_details,
    ingest_live_gameweek,
    fpl_complete_pipeline,
)

__all__ = [
    "ingest_static_endpoints",
    "ingest_player_details",
    "ingest_live_gameweek",
    "fpl_complete_pipeline",
    "fpl_weekly_orchestration",
]

