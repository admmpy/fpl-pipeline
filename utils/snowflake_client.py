"""
Snowflake connection and utility functions.
"""
import snowflake.connector
from snowflake.connector import DictCursor
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_snowflake_config


@contextmanager
def get_snowflake_connection():
    """
    Context manager for Snowflake database connections.
    
    Yields:
        snowflake.connector.connection: Active Snowflake connection
        
    Raises:
        ValueError: If Snowflake configuration is missing
        snowflake.connector.Error: If connection fails
    """
    config = get_snowflake_config()
    
    if config is None:
        raise ValueError("Snowflake configuration is not available")
    
    conn = None
    try:
        conn = snowflake.connector.connect(
            account=config["account"],
            user=config["user"],
            password=config["password"],
            warehouse=config["warehouse"],
            database=config["database"],
            schema=config["schema"],
        )
        yield conn
    finally:
        if conn:
            conn.close()


def create_raw_table_if_not_exists(
    table_name: str,
    connection: Optional[snowflake.connector.connection] = None
) -> bool:
    """
    Create a raw data table in Snowflake if it doesn't exist.
    
    Table schema:
        - id: Auto-incrementing primary key
        - ingestion_timestamp: When data was loaded
        - data: VARIANT column storing raw JSON
        - source_url: URL the data came from
        - endpoint_name: Name of the endpoint
    
    Args:
        table_name: Name of the table to create
        connection: Optional existing connection (creates new one if not provided)
        
    Returns:
        True if table was created or already exists
    """
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id INTEGER AUTOINCREMENT,
        ingestion_timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
        data VARIANT,
        source_url VARCHAR(500),
        endpoint_name VARCHAR(100),
        PRIMARY KEY (id)
    )
    """
    
    if connection:
        cursor = connection.cursor()
        cursor.execute(create_sql)
        cursor.close()
        return True
    else:
        with get_snowflake_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(create_sql)
            cursor.close()
            return True


def insert_raw_data(
    table_name: str,
    data: Dict[str, Any],
    metadata: Dict[str, Any],
    connection: Optional[snowflake.connector.connection] = None
) -> bool:
    """
    Insert raw JSON data into a Snowflake table.
    
    Args:
        table_name: Target table name
        data: JSON data to insert
        metadata: Metadata about the data (url, endpoint_name, etc.)
        connection: Optional existing connection
        
    Returns:
        True if insert was successful
    """
    import json
    
    insert_sql = f"""
    INSERT INTO {table_name} (data, source_url, endpoint_name)
    SELECT 
        PARSE_JSON(%s),
        %s,
        %s
    """
    
    json_str = json.dumps(data)
    source_url = metadata.get("url", "")
    endpoint_name = metadata.get("endpoint_name", "")
    
    if connection:
        cursor = connection.cursor()
        cursor.execute(insert_sql, (json_str, source_url, endpoint_name))
        cursor.close()
        return True
    else:
        with get_snowflake_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(insert_sql, (json_str, source_url, endpoint_name))
            conn.commit()
            cursor.close()
            return True


def test_connection() -> bool:
    """
    Test Snowflake connection.
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        with get_snowflake_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT CURRENT_VERSION()")
            version = cursor.fetchone()
            cursor.close()
            print(f"Connected to Snowflake version: {version[0]}")
            return True
    except Exception as e:
        print(f"Snowflake connection failed: {e}")
        return False

def create_typed_table(
    table_name: str,
    schema: Dict[str, str],
    connection: Optional[snowflake.connector.connection] = None
) -> bool:
    """
    Create a typed table in Snowflake from schema definition.
    
    Args:
        table_name: Name of the table to create
        schema: Dictionary mapping column names to types
        connection: Optional existing connection
        
    Returns:
        True if table was created or already exists
    """
    from config import generate_create_table_sql
    
    create_sql = generate_create_table_sql(table_name, schema)
    
    if connection:
        cursor = connection.cursor()
        cursor.execute(create_sql)
        cursor.close()
        return True
    else:
        with get_snowflake_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(create_sql)
            cursor.close()
            return True


def insert_typed_records(
    table_name: str,
    records: List[Dict[str, Any]],
    connection: Optional[snowflake.connector.connection] = None
) -> int:
    """
    Insert typed records into Snowflake table.
    
    Args:
        table_name: Target table name
        records: List of dictionaries with typed data
        connection: Optional existing connection
        
    Returns:
        Number of records inserted
    """
    if not records:
        return 0
    
    # Get column names from first record
    columns = list(records[0].keys())
    
    # Build INSERT statement
    columns_str = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))
    insert_sql = f"""
    INSERT INTO {table_name} ({columns_str})
    VALUES ({placeholders})
    """
    
    # Prepare data as tuples
    data_tuples = [
        tuple(record.get(col) for col in columns)
        for record in records
    ]
    
    if connection:
        cursor = connection.cursor()
        cursor.executemany(insert_sql, data_tuples)
        cursor.close()
        return len(data_tuples)
    else:
        with get_snowflake_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(insert_sql, data_tuples)
            conn.commit()
            cursor.close()
            return len(data_tuples)