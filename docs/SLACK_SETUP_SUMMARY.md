# Slack Integration Implementation Summary

## What Was Implemented

A complete Slack webhook integration for your FPL pipeline that automatically alerts you when things go wrong and optionally notifies you of successful runs.

## Files Created

### 1. Core Slack Client (`utils/slack_client.py`)
- `send_slack_message()`: Core function to send messages via webhook
- `format_failure_alert()`: Formats pipeline failure messages
- `format_source_failure_alert()`: Formats data source failure messages
- `format_success_summary()`: Formats success notifications
- `test_slack_webhook()`: Tests webhook connection

### 2. Enhanced Reporting Tasks (`tasks/reporting_tasks.py`)
Updated with three new Prefect tasks:
- `notify_pipeline_failure()`: Sends pipeline failure alerts
- `notify_source_failure()`: Sends data source failure alerts
- `notify_pipeline_success()`: Sends success summaries
- Enhanced `format_squad_summary()`: Better formatting with positions and costs

### 3. Configuration Updates (`config.py`)
- Added `get_slack_config()`: Loads webhook URL from environment

### 4. Flow-Level Failure Hooks

#### `flows/fpl_orchestration.py`
- `pipeline_failure_hook()`: Automatic alert when orchestration fails
- Enhanced error handling with context
- Critical vs non-critical failure detection
- Optional success notifications

#### `flows/fpl_ingestion.py`
- `ingestion_failure_hook()`: Automatic alert when data sources fail

### 5. Test Script (`scripts/test_slack.py`)
Complete test suite that sends 4 test messages:
- Connection test
- Pipeline failure example
- Source failure example
- Success summary example

### 6. Documentation

#### `docs/SLACK_INTEGRATION.md`
Complete guide covering:
- Setup instructions
- Usage examples
- Configuration options
- Troubleshooting
- Best practices
- Security considerations

#### `ENV_EXAMPLE.txt`
Template for `.env` file with all configuration options

#### Updated existing docs:
- `README.md`: Added Slack section
- `GETTING_STARTED.md`: Added setup instructions

## How It Works

### Automatic Failure Detection

The pipeline uses Prefect state hooks to automatically detect failures:

```
Flow Execution
    ├─ Success → Continue
    └─ Failure → on_failure hook
                    └─ Send Slack alert
```

**Pipeline-level failures** (entire orchestration):
- Ingestion failures
- dbt transformation failures
- ML inference failures
- Optimization failures

**Source-level failures** (data ingestion):
- API timeout or HTTP errors
- Missing critical data (e.g., players table)

### Alert Levels

1. **Critical Failures** → Stop pipeline + Send alert
   - Players table ingestion fails
   - ML inference fails
   - Optimization fails

2. **Non-Critical Failures** → Continue pipeline + Log warning
   - dbt fails (if `allow_stale_data=True`)
   - Reporting fails
   - Slack notification fails (never stops pipeline)

## Setup Steps

### 1. Create Slack Webhook

1. Visit [https://api.slack.com/apps](https://api.slack.com/apps)
2. Click "Create New App" → "From scratch"
3. Name it "FPL Pipeline" and select your workspace
4. Navigate to "Incoming Webhooks"
5. Toggle "Activate Incoming Webhooks" to **On**
6. Click "Add New Webhook to Workspace"
7. Select channel (e.g., `#data-pipeline-alerts`)
8. Copy the webhook URL

### 2. Add to Environment

Add to your `.env` file:

```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXX
```

### 3. Test Integration

```bash
cd pipeline
python scripts/test_slack.py
```

You should see 4 test messages in your Slack channel.

### 4. Run Pipeline

The next time your pipeline runs, you'll automatically get alerts for any failures:

```bash
python scripts/run_once.py
```

## Usage Examples

### Testing Locally

```bash
# Test the webhook connection
python scripts/test_slack.py

# Run pipeline with Slack notifications
python scripts/run_once.py
```

### Production Deployment

In production, scheduled runs are triggered by GitHub Actions workflows.
The Slack webhook URL is read from `.env` automatically, and failures are reported via the hooks.

### Custom Notifications

Send manual notifications from your code:

```python
from tasks.reporting_tasks import notify_pipeline_failure

notify_pipeline_failure(
    flow_name="Custom Flow",
    error_message="Something went wrong",
    context={"gameweek": 20, "step": "custom_step"}
)
```

## Message Examples

### Pipeline Failure
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

### Source Failure
```
:warning: Data Source Failure

Source: FPL API - Bootstrap Static
Time: 2026-01-19 14:32:15 UTC
Error: HTTP 503 Service Unavailable
URL: https://fantasy.premierleague.com/api/bootstrap-static/
```

### Squad Recommendations
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

## Key Features

### Reliability
- Retries on failure (2 retries with 5s delay)
- Never stops pipeline if Slack fails
- Graceful degradation if webhook not configured

### Observability
- Automatic failure detection via Prefect hooks
- Detailed error context in messages
- Timestamps in UTC

### Security
- Webhook URL stored in `.env` (never committed)
- Uses environment variables for configuration
- Can use Prefect Secret Blocks for production

## Next Steps

### Immediate
1. ✅ Create Slack webhook
2. ✅ Add to `.env` file
3. ✅ Run test script
4. ✅ Verify messages appear in Slack

### Optional Enhancements
- Create separate channels for different alert types
- Use Prefect Secret Blocks for production
- Add success notifications (currently opt-in)
- Implement rich formatting with Block Kit
- Add email fallback notifications

## Troubleshooting

### No messages appearing?

**Check 1:** Is webhook URL configured?
```bash
cat .env | grep SLACK_WEBHOOK_URL
```

**Check 2:** Test the connection
```bash
python scripts/test_slack.py
```

**Check 3:** Check logs for errors
```bash
# Look for "Slack notification sent successfully" or error messages
```

### Messages sent but not visible?

**Check 1:** Is bot added to channel?
- Go to Slack channel
- Click channel name → "Integrations" → "Apps"
- Add your FPL Pipeline app

**Check 2:** Is webhook URL valid?
- URLs expire if regenerated
- Check Slack app settings for current URL

## Production Considerations

### Environment-Specific Webhooks

Use different webhooks for different environments:

```bash
# .env.dev
SLACK_WEBHOOK_URL=https://hooks.slack.com/.../dev-channel

# .env.prod
SLACK_WEBHOOK_URL=https://hooks.slack.com/.../prod-alerts
```

### Prefect Secret Blocks (Recommended)

For production, store webhook as a Prefect Secret:

```python
from prefect.blocks.system import Secret

webhook_block = Secret(value="https://hooks.slack.com/...")
webhook_block.save("slack-webhook-prod")
```

### Rate Limiting

Slack webhooks have a rate limit of 1 message per second. The current implementation respects this with the 10-second timeout.

### Monitoring

Consider tracking:
- Notification delivery rate
- Failed notification count
- Time to alert (from failure to notification)

## Support

For issues or questions:
1. Check [docs/SLACK_INTEGRATION.md](SLACK_INTEGRATION.md) for detailed documentation
2. Review Slack webhook logs in Prefect UI
3. Test with `scripts/test_slack.py`
4. Verify webhook URL is valid in Slack app settings
