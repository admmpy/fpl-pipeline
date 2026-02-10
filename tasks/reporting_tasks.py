"""
Tasks for reporting and notifications.
"""
from typing import Dict, Any, List, Optional
from prefect import task, get_run_logger
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.slack_client import (
    send_slack_message,
    format_failure_alert,
    format_source_failure_alert,
    format_success_summary,
)


@task
def format_squad_summary(squad_records: List[Dict[str, Any]]) -> str:
    """
    Format the recommended squad into a human-readable summary.
    
    Args:
        squad_records: List of player records in the recommended squad
        
    Returns:
        Formatted string summary for Slack
    """
    logger = get_run_logger()
    
    if not squad_records:
        return "No squad recommendations found."
    
    summary = [":trophy: *FPL Recommended Squad:*", ""]
    
    # Sort by position (1=GK, 2=DEF, 3=MID, 4=FWD)
    sorted_squad = sorted(squad_records, key=lambda x: x.get('position_id', 0))
    
    # Group by position
    position_names = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    current_position = None
    
    for player in sorted_squad:
        position_id = player.get('position_id', 0)
        
        # Add position header when it changes
        if position_id != current_position:
            current_position = position_id
            summary.append(f"\n*{position_names.get(position_id, 'Unknown')}:*")
        
        captain_star = " (C)" if player.get('is_captain') else ""
        vice_star = " (VC)" if player.get('is_vice_captain') else ""
        
        expected_pts = player.get('expected_points_next_gw', 0)
        cost = player.get('now_cost', 0)
        
        line = (
            f"  - {player.get('web_name')}{captain_star}{vice_star} | "
            f"£{cost:.1f}m | Pts: {expected_pts:.1f}"
        )
        summary.append(line)
    
    # Add total squad value
    total_cost = sum(p.get('now_cost', 0) for p in sorted_squad)
    summary.append(f"\n*Total Squad Value:* £{total_cost:.1f}m")
        
    return "\n".join(summary)


@task(retries=2, retry_delay_seconds=5)
def send_slack_notification(
    message: str,
    webhook_url: Optional[str] = None
) -> bool:
    """
    Send a notification to Slack via webhook.
    
    Args:
        message: Message text to send (supports Slack markdown)
        webhook_url: Optional webhook URL (uses env var if not provided)
        
    Returns:
        True if notification was sent successfully
    """
    logger = get_run_logger()
    
    success = send_slack_message(message, webhook_url=webhook_url)
    
    if success:
        logger.info("Slack notification sent successfully")
    else:
        logger.warning("Failed to send Slack notification")
    
    return success


@task
def notify_pipeline_failure(
    flow_name: str,
    error_message: str,
    context: Optional[Dict[str, Any]] = None,
    webhook_url: Optional[str] = None
) -> bool:
    """
    Send a pipeline failure alert to Slack.
    
    Args:
        flow_name: Name of the failed flow
        error_message: Error message or exception details
        context: Optional context dict (e.g., gameweek, step)
        webhook_url: Optional webhook URL
        
    Returns:
        True if alert was sent successfully
    """
    logger = get_run_logger()
    
    alert_message = format_failure_alert(flow_name, error_message, context)
    success = send_slack_message(alert_message, webhook_url=webhook_url)
    
    if success:
        logger.info(f"Failure alert sent for {flow_name}")
    else:
        logger.warning(f"Failed to send failure alert for {flow_name}")
    
    return success


@task
def notify_source_failure(
    source_name: str,
    error_message: str,
    url: Optional[str] = None,
    webhook_url: Optional[str] = None
) -> bool:
    """
    Send a data source failure alert to Slack.
    
    Args:
        source_name: Name of the failed data source
        error_message: Error message
        url: Optional URL that failed
        webhook_url: Optional webhook URL
        
    Returns:
        True if alert was sent successfully
    """
    logger = get_run_logger()
    
    alert_message = format_source_failure_alert(source_name, error_message, url)
    success = send_slack_message(alert_message, webhook_url=webhook_url)
    
    if success:
        logger.info(f"Source failure alert sent for {source_name}")
    else:
        logger.warning(f"Failed to send source failure alert for {source_name}")
    
    return success


@task
def notify_pipeline_success(
    flow_name: str,
    metrics: Dict[str, Any],
    webhook_url: Optional[str] = None
) -> bool:
    """
    Send a pipeline success summary to Slack.
    
    Args:
        flow_name: Name of the completed flow
        metrics: Dictionary of metrics to report
        webhook_url: Optional webhook URL
        
    Returns:
        True if notification was sent successfully
    """
    logger = get_run_logger()
    
    summary_message = format_success_summary(flow_name, metrics)
    success = send_slack_message(summary_message, webhook_url=webhook_url)
    
    if success:
        logger.info(f"Success summary sent for {flow_name}")
    else:
        logger.warning(f"Failed to send success summary for {flow_name}")
    
    return success
