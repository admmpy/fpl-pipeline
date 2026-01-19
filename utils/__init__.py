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

from .slack_client import (
    send_slack_message,
    format_failure_alert,
    format_source_failure_alert,
    format_success_summary,
    test_slack_webhook,
)

__all__ = [
    "get_snowflake_connection",
    "create_raw_table_if_not_exists",
    "insert_raw_data",
    "test_connection",
    "create_typed_table",
    "insert_typed_records",
    "send_slack_message",
    "format_failure_alert",
    "format_source_failure_alert",
    "format_success_summary",
    "test_slack_webhook",
]