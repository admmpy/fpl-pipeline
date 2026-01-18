"""
Tasks for reporting and notifications.
"""
from typing import Dict, Any, List
from prefect import task, get_run_logger
import os

@task
def format_squad_summary(squad_records: List[Dict[str, Any]]) -> str:
    """
    Format the recommended squad into a human-readable summary.
    """
    logger = get_run_logger()
    
    if not squad_records:
        return "No squad recommendations found."
    
    summary = ["*FPL Recommended Squad:*"]
    
    # Sort by position (roughly)
    # 1=GK, 2=DEF, 3=MID, 4=FWD
    sorted_squad = sorted(squad_records, key=lambda x: x.get('position_id', 0))
    
    for player in sorted_squad:
        captain_star = " (C)" if player.get('is_captain') else ""
        vice_star = " (VC)" if player.get('is_vice_captain') else ""
        
        line = f"- {player.get('web_name')}{captain_star}{vice_star} | Points: {player.get('expected_points_5_gw', 0):.1f}"
        summary.append(line)
        
    return "\n".join(summary)

@task
def send_slack_notification(message: str, webhook_url: str = None) -> bool:
    """
    Send a notification to Slack via Webhook.
    
    Note: In a production environment, use Prefect Blocks or env vars for the URL.
    """
    logger = get_run_logger()
    
    url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
    
    if not url:
        logger.warning("No SLACK_WEBHOOK_URL found. Skipping notification.")
        logger.info(f"Message that would have been sent:\n{message}")
        return False
        
    try:
        import requests
        response = requests.post(url, json={"text": message}, timeout=10)
        response.raise_for_status()
        logger.info("Slack notification sent successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to send Slack notification: {e}")
        return False
