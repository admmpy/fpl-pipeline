"""
Utilities package for FPL pipeline.
"""
from .snowflake_client import (
    get_snowflake_connection,
    create_raw_table_if_not_exists,
    insert_raw_data,
    test_connection,
)

__all__ = [
    "get_snowflake_connection",
    "create_raw_table_if_not_exists",
    "insert_raw_data",
    "test_connection",
]

"""
Utilities package for FPL pipeline.
"""
from .snowflake_client import (
    get_snowflake_connection,
    create_raw_table_if_not_exists,
    insert_raw_data,
    test_connection,
    create_typed_table,
    insert_typed_records,
)

__all__ = [
    "get_snowflake_connection",
    "create_raw_table_if_not_exists",
    "insert_raw_data",
    "test_connection",
    "create_typed_table",
    "insert_typed_records",
]