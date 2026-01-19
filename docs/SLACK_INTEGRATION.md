# Slack Webhook Integration

This document describes the Slack webhook integration for pipeline notifications and alerts.

## Overview

The FPL pipeline automatically sends Slack notifications for:

1. **Pipeline Failures**: Automatic alerts when the entire orchestration fails
2. **Source Failures**: Alerts when critical data sources fail to fetch
3. **Success Notifications**: Optional summaries when pipelines complete successfully
4. **Squad Recommendations**: Formatted squad recommendations after optimization

## Setup

### 1. Create Slack Webhook

1. Go to [https://api.slack.com/apps](https://api.slack.com/apps)
2. Click "Create New App" → "From scratch"
3. Name your app (e.g., "FPL Pipeline") and select your workspace
4. Navigate to "Incoming Webhooks" in the sidebar
5. Toggle "Activate Incoming Webhooks" to **On**
6. Click "Add New Webhook to Workspace"
7. Select the channel where notifications should be sent (e.g., `#data-pipeline-alerts`)
8. Copy the webhook URL (looks like `https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXX`)

### 2. Configure Environment Variable

Add the webhook URL to your `.env` file:

```bash
# Slack Configuration
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

**IMPORTANT:** Never commit this URL to version control. The `.env` file is already in `.gitignore`.

### 3. Test Configuration

Run the test script to verify your webhook is working:

```bash
cd pipeline
python scripts/test_slack.py
```

This will send 4 test messages to your Slack channel:
- Connection test
- Pipeline failure alert
- Source failure alert
- Success summary

## Usage

### Automatic Notifications

Pipeline failures are automatically detected and reported via Prefect state hooks:

```python
@flow(
    name="FPL Weekly Orchestration",
    on_failure=[pipeline_failure_hook]  # Automatic failure alerts
)
def fpl_weekly_orchestration():
    # Your pipeline logic
    pass
```

When the flow fails, you'll receive a message like:

```
:x: Pipeline Failure Alert

Flow: FPL Weekly Orchestration
Time: 2026-01-19 14:32:15 UTC
Error: Timeout fetching bootstrap-static endpoint

Context:
  - gameweek: 20
  - step: ingestion
  - flow_run_id: abc123...
```

### Manual Notifications

You can also send manual notifications from within your tasks:

```python
from tasks.reporting_tasks import (
    notify_pipeline_failure,
    notify_source_failure,
    notify_pipeline_success,
)

# Notify about a source failure
notify_source_failure(
    source_name="FPL API - Bootstrap Static",
    error_message="HTTP 503 Service Unavailable",
    url="https://fantasy.premierleague.com/api/bootstrap-static/"
)

# Notify about pipeline success with metrics
notify_pipeline_success(
    flow_name="FPL Weekly Orchestration",
    metrics={
        "players_ingested": 692,
        "duration_seconds": 127.5,
    }
)
```

### Squad Recommendations

After optimization, the pipeline automatically sends formatted squad recommendations:

```python
summary = format_squad_summary(recommended_squad)
send_slack_notification(summary)
```

This produces messages like:

```
:trophy: FPL Recommended Squad:

GK:
  - Raya | £5.5m | Pts: 23.4
  - Martinez | £4.8m | Pts: 19.2

DEF:
  - Saliba | £6.0m | Pts: 28.1
  - Gabriel | £5.8m | Pts: 26.7
  ...

Total Squad Value: £99.2m
```

## Configuration Options

### Flow-Level Configuration

When running the orchestration flow, you can control notification behaviour:

```python
fpl_weekly_orchestration(
    slack_webhook_url="https://hooks.slack.com/...",  # Override .env
    notify_on_success=True,  # Send success notifications (default: False)
)
```

### Notification Retries

Slack notification tasks automatically retry on failure:

```python
@task(retries=2, retry_delay_seconds=5)
def send_slack_notification(message: str):
    # Retries twice with 5-second delays
    pass
```

## Architecture

### Components

1. **`utils/slack_client.py`**: Core webhook client with formatting utilities
2. **`tasks/reporting_tasks.py`**: Prefect tasks for notifications
3. **`flows/fpl_orchestration.py`**: Flow-level failure hooks
4. **`flows/fpl_ingestion.py`**: Source-level failure hooks
5. **`config.py`**: Configuration loader for webhook URL

### Failure Detection

The pipeline uses Prefect state hooks to detect failures at different levels:

```python
# Pipeline-level failures (entire orchestration)
@flow(on_failure=[pipeline_failure_hook])
def fpl_weekly_orchestration():
    pass

# Source-level failures (data ingestion)
@flow(on_failure=[ingestion_failure_hook])
def fpl_typed_pipeline():
    pass
```

### Critical vs Non-Critical Failures

The pipeline distinguishes between:

- **Critical failures**: Stop execution and send alerts
  - Players table ingestion failure
  - ML inference failure
  - Optimization failure

- **Non-critical failures**: Log warnings but continue
  - Reporting task failure
  - Slack notification failure (doesn't stop pipeline)

## Best Practices

### 1. Channel Selection

Create a dedicated channel for pipeline notifications:
- `#data-pipeline-alerts` for failures
- `#data-pipeline-reports` for squad recommendations

### 2. Notification Frequency

Only enable success notifications for:
- Manual runs
- Testing environments
- Weekly summary emails

Disable for scheduled runs to avoid noise.

### 3. Local vs Production

Use different webhook URLs for different environments:

```bash
# Local development
SLACK_WEBHOOK_URL=https://hooks.slack.com/.../dev-channel

# Production
SLACK_WEBHOOK_URL=https://hooks.slack.com/.../prod-alerts
```

### 4. Message Formatting

Use Slack markdown for clarity:
- `:x:` for failures
- `:white_check_mark:` for successes
- `:warning:` for warnings
- `:trophy:` for squad recommendations

## Troubleshooting

### "No SLACK_WEBHOOK_URL found"

**Cause**: Environment variable not set

**Solution**:
1. Check `.env` file exists in `pipeline/` directory
2. Verify `SLACK_WEBHOOK_URL` is set
3. Restart your Python environment to reload `.env`

### "Slack webhook timeout"

**Cause**: Network issues or Slack API unavailable

**Solution**:
- Check your internet connection
- Verify webhook URL is correct
- Check Slack status: [https://status.slack.com](https://status.slack.com)

### "HTTP 404 Not Found"

**Cause**: Invalid webhook URL

**Solution**:
1. Regenerate webhook in Slack app settings
2. Update `.env` file with new URL
3. Run test script to verify

### Messages not appearing

**Cause**: Bot not invited to channel

**Solution**:
1. Go to your Slack channel
2. Click channel name → "Integrations" → "Apps"
3. Add your FPL Pipeline app

## Security Considerations

1. **Never commit webhook URLs**: They're secrets that grant posting access
2. **Rotate webhooks periodically**: Regenerate if exposed
3. **Use channel-specific webhooks**: Different webhooks for different channels
4. **Limit webhook scope**: Only grant permissions needed (post messages)

## Extending Notifications

### Custom Message Formats

Create custom formatters in `utils/slack_client.py`:

```python
def format_dbt_failure_alert(model_name: str, error: str) -> str:
    """Format a dbt model failure alert."""
    return f":x: *dbt Model Failure*\n\nModel: {model_name}\nError: {error}"
```

### Additional Hooks

Add hooks for specific flows:

```python
@flow(on_failure=[dbt_failure_hook])
def run_dbt_transformations():
    pass
```

### Rich Formatting

Use Slack Block Kit for richer messages:

```python
payload = {
    "blocks": [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*Pipeline Failure*"}
        },
        {
            "type": "divider"
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Flow:*\n{flow_name}"},
                {"type": "mrkdwn", "text": f"*Time:*\n{timestamp}"}
            ]
        }
    ]
}
```

## Production Deployment

### Using Prefect Blocks (Recommended)

For production, use Prefect Secret Blocks instead of `.env`:

```python
from prefect.blocks.system import Secret

# Store webhook securely
webhook_block = Secret(value="https://hooks.slack.com/...")
webhook_block.save("slack-webhook-prod")

# Retrieve in flow
webhook_url = Secret.load("slack-webhook-prod").get()
```

### Cloud Scheduler Integration

When using a cloud scheduler (e.g., GitHub Actions, Prefect Cloud):

1. Store webhook URL as a secret in your scheduler
2. Pass as environment variable at runtime
3. Ensure secrets are encrypted at rest

## Monitoring

Track notification delivery:

```python
success = send_slack_notification(message)
if not success:
    logger.error("Slack notification failed - check webhook configuration")
```

Consider implementing:
- Notification delivery logs
- Fallback notification channels (email)
- Dead letter queue for failed notifications

## References

- [Slack Incoming Webhooks](https://api.slack.com/messaging/webhooks)
- [Prefect State Hooks](https://docs.prefect.io/concepts/flows/#flow-hooks)
- [Slack Message Formatting](https://api.slack.com/reference/surfaces/formatting)
