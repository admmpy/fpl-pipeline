"""
Test script for Slack webhook integration.

Usage:
    python scripts/test_slack.py
"""
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.slack_client import (
    test_slack_webhook,
    format_failure_alert,
    format_source_failure_alert,
    format_success_summary,
    send_slack_message,
)
from config import get_slack_config


def main():
    """Test Slack webhook integration."""
    print("=" * 60)
    print("Slack Webhook Integration Test")
    print("=" * 60)
    print()
    
    # Check if webhook is configured
    webhook_url = get_slack_config()
    
    if not webhook_url:
        print("ERROR: SLACK_WEBHOOK_URL not configured in .env file")
        print()
        print("To configure:")
        print("1. Go to https://api.slack.com/apps")
        print("2. Create a new app or select existing")
        print("3. Enable 'Incoming Webhooks'")
        print("4. Create a webhook for your channel")
        print("5. Add to .env: SLACK_WEBHOOK_URL=https://hooks.slack.com/...")
        return 1
    
    print(f"Webhook URL configured: {webhook_url[:50]}...")
    print()
    
    # Test 1: Basic connection test
    print("Test 1: Basic Connection Test")
    print("-" * 60)
    if test_slack_webhook():
        print("SUCCESS: Test message sent to Slack")
    else:
        print("FAILED: Could not send test message")
        return 1
    print()
    
    # Test 2: Failure alert format
    print("Test 2: Pipeline Failure Alert")
    print("-" * 60)
    failure_msg = format_failure_alert(
        flow_name="FPL Weekly Orchestration",
        error_message="Timeout fetching bootstrap-static endpoint",
        context={
            "gameweek": 20,
            "step": "ingestion",
            "attempt": 3,
        }
    )
    if send_slack_message(failure_msg):
        print("SUCCESS: Failure alert sent")
    else:
        print("FAILED: Could not send failure alert")
    print()
    
    # Test 3: Source failure alert format
    print("Test 3: Source Failure Alert")
    print("-" * 60)
    source_failure_msg = format_source_failure_alert(
        source_name="FPL API - Bootstrap Static",
        error_message="HTTP 503 Service Unavailable",
        url="https://fantasy.premierleague.com/api/bootstrap-static/"
    )
    if send_slack_message(source_failure_msg):
        print("SUCCESS: Source failure alert sent")
    else:
        print("FAILED: Could not send source failure alert")
    print()
    
    # Test 4: Success summary format
    print("Test 4: Success Summary")
    print("-" * 60)
    success_msg = format_success_summary(
        flow_name="FPL Weekly Orchestration",
        metrics={
            "players_ingested": 692,
            "teams_ingested": 20,
            "gameweeks_ingested": 38,
            "fixtures_ingested": 380,
            "squad_recommendations": 15,
            "duration_seconds": 127.5,
        }
    )
    if send_slack_message(success_msg):
        print("SUCCESS: Success summary sent")
    else:
        print("FAILED: Could not send success summary")
    print()
    
    print("=" * 60)
    print("All tests completed!")
    print("Check your Slack channel for the test messages.")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
