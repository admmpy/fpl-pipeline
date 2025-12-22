"""
Tasks package for FPL pipeline.
"""
from .api_tasks import (
    fetch_fpl_endpoint,
    extract_player_ids,
    extract_current_gameweek,
    fetch_dynamic_endpoint_batch,
)
from .snowflake_tasks import (
    ensure_table_exists,
    load_to_snowflake,
    load_batch_to_snowflake,
)

__all__ = [
    "fetch_fpl_endpoint",
    "extract_player_ids",
    "extract_current_gameweek",
    "fetch_dynamic_endpoint_batch",
    "ensure_table_exists",
    "load_to_snowflake",
    "load_batch_to_snowflake",
]

"""
Tasks package for FPL pipeline.
"""
from .api_tasks import (
    fetch_fpl_endpoint,
    extract_player_ids,
    extract_current_gameweek,
    fetch_dynamic_endpoint_batch,
)
from .snowflake_tasks import (
    ensure_table_exists,
    load_to_snowflake,
    load_batch_to_snowflake,
    ensure_typed_table_exists,
    load_typed_records_to_snowflake,
)

__all__ = [
    "fetch_fpl_endpoint",
    "extract_player_ids",
    "extract_current_gameweek",
    "fetch_dynamic_endpoint_batch",
    "ensure_table_exists",
    "load_to_snowflake",
    "load_batch_to_snowflake",
    "ensure_typed_table_exists",
    "load_typed_records_to_snowflake",
]