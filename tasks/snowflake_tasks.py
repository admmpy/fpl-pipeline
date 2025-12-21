"""
Snowflake loading tasks for FPL pipeline.
"""
from prefect import task, get_run_logger
from typing import Dict, Any, List, Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.snowflake_client import (
    get_snowflake_connection,
    create_raw_table_if_not_exists,
    insert_raw_data,
)
from config import get_snowflake_config


@task
def ensure_table_exists(table_name: str) -> bool:
    """
    Ensure a raw data table exists in Snowflake.
    
    Args:
        table_name: Name of the table to create/verify
        
    Returns:
        True if successful, False if Snowflake not configured or connection failed
    """
    logger = get_run_logger()
    
    # Check if Snowflake is configured
    if get_snowflake_config() is None:
        logger.warning(f"Snowflake not configured, skipping table creation for {table_name}")
        return False
    
    try:
        logger.info(f"Ensuring table {table_name} exists...")
        create_raw_table_if_not_exists(table_name)
        logger.info(f"Table {table_name} ready")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create table {table_name}: {e}")
        logger.warning(f"Continuing without Snowflake for {table_name}")
        return False


@task
def load_to_snowflake(
    table_name: str,
    api_response: Dict[str, Any]
) -> bool:
    """
    Load API response data to Snowflake.
    
    Args:
        table_name: Target table name
        api_response: Response from fetch_fpl_endpoint containing data and metadata
        
    Returns:
        True if successful, False if Snowflake not configured or load failed
    """
    logger = get_run_logger()
    
    # Check if Snowflake is configured
    if get_snowflake_config() is None:
        logger.warning(f"Snowflake not configured, skipping load to {table_name}")
        return False
    
    try:
        data = api_response["data"]
        metadata = api_response["metadata"]
        
        logger.info(f"Loading data to {table_name}...")
        
        insert_raw_data(
            table_name=table_name,
            data=data,
            metadata=metadata
        )
        
        logger.info(f"Successfully loaded data to {table_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load data to {table_name}: {e}")
        logger.warning(f"Continuing without Snowflake for {table_name}")
        return False


@task
def load_batch_to_snowflake(
    table_name: str,
    api_responses: List[Dict[str, Any]]
) -> Dict[str, int]:
    """
    Load multiple API responses to Snowflake in batch.
    
    Args:
        table_name: Target table name
        api_responses: List of responses from fetch_fpl_endpoint
        
    Returns:
        Dictionary with success/failure counts
    """
    logger = get_run_logger()
    
    # Check if Snowflake is configured
    if get_snowflake_config() is None:
        logger.warning(f"Snowflake not configured, skipping batch load to {table_name}")
        return {"total": len(api_responses), "success": 0, "failed": len(api_responses)}
    
    logger.info(f"Loading {len(api_responses)} records to {table_name}...")
    
    success_count = 0
    failed_count = 0
    
    try:
        with get_snowflake_connection() as conn:
            for i, response in enumerate(api_responses, 1):
                try:
                    data = response["data"]
                    metadata = response["metadata"]
                    
                    insert_raw_data(
                        table_name=table_name,
                        data=data,
                        metadata=metadata,
                        connection=conn
                    )
                    success_count += 1
                    
                    # Progress logging every 100 records
                    if i % 100 == 0:
                        logger.info(f"   Progress: {i}/{len(api_responses)} records loaded")
                        
                except Exception as e:
                    logger.warning(f"Failed to load record {i}: {e}")
                    failed_count += 1
                    continue
            
            # Commit all inserts
            conn.commit()
        
        logger.info(
            f"Batch load complete: {success_count} success, {failed_count} failed"
        )
        
        return {
            "total": len(api_responses),
            "success": success_count,
            "failed": failed_count
        }
        
    except Exception as e:
        logger.error(f"Batch load failed: {e}")
        raise

