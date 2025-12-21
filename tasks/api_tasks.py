"""
API tasks for fetching FPL data.
"""
import requests
import time
from datetime import datetime
from prefect import task, get_run_logger
from typing import Dict, Any, Optional
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RATE_LIMIT_DELAY, MAX_RETRIES, RETRY_DELAY


@task(retries=MAX_RETRIES, retry_delay_seconds=RETRY_DELAY)
def fetch_fpl_endpoint(
    url: str, 
    endpoint_name: str,
    rate_limit: bool = True
) -> Dict[str, Any]:
    """
    Fetch data from a single FPL API endpoint.
    
    Args:
        url: Full URL to fetch
        endpoint_name: Name for logging purposes
        rate_limit: Whether to apply rate limiting delay
        
    Returns:
        Dictionary containing:
            - data: JSON response from API
            - metadata: Information about the fetch (timestamp, url, etc.)
        
    Raises:
        requests.RequestException: If API call fails after retries
    """
    logger = get_run_logger()
    
    # Apply rate limiting to be respectful to FPL API
    if rate_limit:
        time.sleep(RATE_LIMIT_DELAY)
    
    try:
        logger.info(f"Fetching {endpoint_name} from {url}")
        
        # Make the request with timeout
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Parse JSON
        data = response.json()
        
        # Create metadata
        metadata = {
            "endpoint_name": endpoint_name,
            "url": url,
            "fetch_timestamp": datetime.utcnow().isoformat(),
            "status_code": response.status_code,
            "response_size_bytes": len(response.content),
        }
        
        logger.info(
            f"Successfully fetched {endpoint_name} "
            f"({metadata['response_size_bytes']:,} bytes)"
        )
        
        return {
            "data": data,
            "metadata": metadata,
        }
        
    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching {endpoint_name} from {url}")
        raise
    except requests.exceptions.HTTPError as e:
        logger.error(
            f"HTTP error {e.response.status_code} fetching {endpoint_name}: {e}"
        )
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching {endpoint_name}: {e}")
        raise
    except ValueError as e:
        logger.error(f"Invalid JSON response from {endpoint_name}: {e}")
        raise


@task
def extract_player_ids(bootstrap_data: Dict[str, Any]) -> list[int]:
    """
    Extract all player IDs from bootstrap-static data.
    
    Args:
        bootstrap_data: Response from fetch_fpl_endpoint for bootstrap-static
        
    Returns:
        List of player IDs
    """
    logger = get_run_logger()
    
    try:
        players = bootstrap_data["data"]["elements"]
        player_ids = [player["id"] for player in players]
        
        logger.info(f"Extracted {len(player_ids)} player IDs")
        return player_ids
        
    except KeyError as e:
        logger.error(f"Failed to extract player IDs: missing key {e}")
        raise


@task
def extract_current_gameweek(bootstrap_data: Dict[str, Any]) -> Optional[int]:
    """
    Extract the current/next gameweek ID from bootstrap-static data.
    
    Args:
        bootstrap_data: Response from fetch_fpl_endpoint for bootstrap-static
        
    Returns:
        Current gameweek ID, or None if no active gameweek
    """
    logger = get_run_logger()
    
    try:
        events = bootstrap_data["data"]["events"]
        
        # Find the current or next gameweek
        for event in events:
            if event["is_current"]:
                gw_id = event["id"]
                logger.info(f"Current gameweek: {gw_id}")
                return gw_id
        
        # If no current gameweek, find the next one
        for event in events:
            if event["is_next"]:
                gw_id = event["id"]
                logger.info(f"Next gameweek: {gw_id}")
                return gw_id
        
        logger.warning("No current or next gameweek found")
        return None
        
    except KeyError as e:
        logger.error(f"Failed to extract gameweek: missing key {e}")
        raise


@task
def fetch_dynamic_endpoint_batch(
    url_template: str,
    ids: list[int],
    endpoint_name: str,
    max_items: Optional[int] = None
) -> list[Dict[str, Any]]:
    """
    Fetch data from a dynamic endpoint for multiple IDs.
    
    Args:
        url_template: URL template with {id} placeholder
        ids: List of IDs to fetch
        endpoint_name: Name for logging
        max_items: Optional limit on number of items to fetch
        
    Returns:
        List of results from fetch_fpl_endpoint
    """
    logger = get_run_logger()
    
    # Limit items if specified
    if max_items:
        ids = ids[:max_items]
        logger.info(f"Limiting to first {max_items} items")
    
    logger.info(f"Fetching {len(ids)} items for {endpoint_name}")
    
    results = []
    failed_ids = []
    
    for i, item_id in enumerate(ids, 1):
        try:
            # Format the URL with the ID
            url = url_template.format(player_id=item_id, gameweek_id=item_id)
            
            # Fetch the data
            result = fetch_fpl_endpoint.fn(
                url=url,
                endpoint_name=f"{endpoint_name}_{item_id}",
                rate_limit=True
            )
            results.append(result)
            
            # Progress logging every 50 items
            if i % 50 == 0:
                logger.info(f"   Progress: {i}/{len(ids)} items fetched")
                
        except Exception as e:
            logger.warning(f"Failed to fetch {endpoint_name} for ID {item_id}: {e}")
            failed_ids.append(item_id)
            continue
    
    logger.info(
        f"Fetched {len(results)}/{len(ids)} items for {endpoint_name}"
    )
    
    if failed_ids:
        logger.warning(f"Failed IDs: {failed_ids[:10]}{'...' if len(failed_ids) > 10 else ''}")
    
    return results

