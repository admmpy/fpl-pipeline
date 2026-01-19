"""
Slack webhook client for pipeline notifications.
"""
import requests
from typing import Optional, Dict, Any
from datetime import datetime
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_slack_config


def send_slack_message(
    message: str,
    webhook_url: Optional[str] = None,
    timeout: int = 10
) -> bool:
    """
    Send a message to Slack via webhook.
    
    Args:
        message: Message text to send (supports Slack markdown)
        webhook_url: Optional webhook URL (uses config if not provided)
        timeout: Request timeout in seconds
        
    Returns:
        True if message was sent successfully, False otherwise
    """
    url = webhook_url or get_slack_config()
    
    if not url:
        print("WARNING: No Slack webhook URL configured. Message not sent.")
        print(f"Message content:\n{message}\n")
        return False
    
    try:
        payload = {"text": message}
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        
        print(f"Slack notification sent successfully at {datetime.utcnow().isoformat()}")
        return True
        
    except requests.exceptions.Timeout:
        print(f"ERROR: Slack webhook timeout after {timeout}s")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"ERROR: Slack webhook HTTP error {e.response.status_code}: {e}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Slack webhook request failed: {e}")
        return False
    except Exception as e:
        print(f"ERROR: Unexpected error sending Slack message: {e}")
        return False


def format_failure_alert(
    flow_name: str,
    error_message: str,
    context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Format a failure alert message for Slack.
    
    Args:
        flow_name: Name of the failed flow/task
        error_message: Error message or exception details
        context: Optional context dict (e.g., gameweek, records processed)
        
    Returns:
        Formatted Slack message with failure details
    """
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    lines = [
        ":x: *Pipeline Failure Alert*",
        "",
        f"*Flow:* {flow_name}",
        f"*Time:* {timestamp}",
        f"*Error:* {error_message}",
    ]
    
    if context:
        lines.append("")
        lines.append("*Context:*")
        for key, value in context.items():
            lines.append(f"  - {key}: {value}")
    
    return "\n".join(lines)


def format_source_failure_alert(
    source_name: str,
    error_message: str,
    url: Optional[str] = None
) -> str:
    """
    Format a source/API failure alert message for Slack.
    
    Args:
        source_name: Name of the failed data source/endpoint
        error_message: Error message
        url: Optional URL that failed
        
    Returns:
        Formatted Slack message for source failure
    """
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    lines = [
        ":warning: *Data Source Failure*",
        "",
        f"*Source:* {source_name}",
        f"*Time:* {timestamp}",
        f"*Error:* {error_message}",
    ]
    
    if url:
        lines.append(f"*URL:* {url}")
    
    return "\n".join(lines)


def format_success_summary(
    flow_name: str,
    metrics: Dict[str, Any]
) -> str:
    """
    Format a success summary message for Slack.
    
    Args:
        flow_name: Name of the completed flow
        metrics: Dictionary of metrics to report
        
    Returns:
        Formatted Slack message with success details
    """
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    lines = [
        ":white_check_mark: *Pipeline Success*",
        "",
        f"*Flow:* {flow_name}",
        f"*Completed:* {timestamp}",
        "",
        "*Metrics:*",
    ]
    
    for key, value in metrics.items():
        lines.append(f"  - {key}: {value}")
    
    return "\n".join(lines)


def test_slack_webhook() -> bool:
    """
    Test Slack webhook connection with a simple message.
    
    Returns:
        True if test message was sent successfully
    """
    test_message = (
        ":bell: *FPL Pipeline Slack Integration Test*\n\n"
        f"Connection test successful at {datetime.utcnow().isoformat()}\n"
        "Slack notifications are now configured."
    )
    
    return send_slack_message(test_message)
