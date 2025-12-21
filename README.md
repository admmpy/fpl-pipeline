# FPL Data Pipeline

A production-ready data pipeline for ingesting Fantasy Premier League (FPL) data using Prefect and Snowflake.

## Overview

This pipeline fetches data from the FPL API and optionally loads it into Snowflake. It uses Prefect for orchestration, providing robust error handling, retries, and observability.

## Architecture

```
FPL API → Fetch Tasks → Snowflake Tasks → Snowflake (raw JSON storage)
```

Data is stored as JSON in Snowflake VARIANT columns, ready for downstream transformation.

## Project Structure

```
pipeline/
├── config.py                 # Configuration & endpoints
├── .env                      # Snowflake credentials (not in git)
├── requirements.txt          # Python dependencies
├── flows/
│   └── fpl_ingestion.py     # Main Prefect flow
├── tasks/
│   ├── api_tasks.py         # API fetching tasks
│   └── snowflake_tasks.py   # Snowflake loading tasks
├── utils/
│   └── snowflake_client.py  # Snowflake connection utilities
└── test_pipeline.py         # Test script
```

## Quick Start

### 1. Set Up Environment

```bash
# Activate virtual environment
source venv/bin/activate

# Verify dependencies
pip list | grep prefect
```

### 2. Configure Snowflake (Optional)

Create a `.env` file:

```bash
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_user
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema
```

**Note:** The pipeline works without Snowflake - it will fetch data but skip loading.

### 3. Test the Pipeline

```bash
# Test with limited data (no Snowflake required)
python test_pipeline.py

# Test full pipeline with Snowflake
python test_pipeline.py --full
```

### 4. Run the Pipeline

```bash
# Run directly
python flows/fpl_ingestion.py

# Or import and run
python -c "from flows.fpl_ingestion import fpl_complete_pipeline; fpl_complete_pipeline()"
```

## Data Endpoints

### Static Endpoints (Always Fetched)
- **bootstrap_static**: Core data (players, teams, gameweeks, stats)
- **fixtures**: Match fixtures and results
- **overall_league**: Official FPL global league standings

### Dynamic Endpoints (Conditional)
- **element_summary**: Detailed stats per player (700+ players)
- **live_gameweek**: Real-time data for current gameweek

## Pipeline Options

```python
from flows.fpl_ingestion import fpl_complete_pipeline

# Full pipeline
fpl_complete_pipeline()

# Skip player details (faster)
fpl_complete_pipeline(include_player_details=False)

# Test with limited players
fpl_complete_pipeline(max_players=10)

# Static data only
fpl_complete_pipeline(
    include_player_details=False,
    include_live_gameweek=False
)
```

## Snowflake Schema

Each endpoint creates a table with this structure:

```sql
CREATE TABLE raw_bootstrap_static (
    id INTEGER AUTOINCREMENT,
    ingestion_timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    data VARIANT,              -- Raw JSON from API
    source_url VARCHAR(500),   -- API endpoint URL
    endpoint_name VARCHAR(100), -- Endpoint identifier
    PRIMARY KEY (id)
);
```

The `data` column stores the complete JSON response.

## Configuration

### API Settings (config.py)

```python
RATE_LIMIT_DELAY = 0.5  # Seconds between API calls
MAX_RETRIES = 3         # Number of retries on failure
RETRY_DELAY = 5         # Seconds between retries
```

### Adding New Endpoints

Edit `config.py`:

```python
# For static endpoints (no parameters)
STATIC_ENDPOINTS = {
    "new_endpoint": {
        "url": f"{FPL_BASE_URL}/new-endpoint/",
        "description": "Description",
        "table": "raw_new_endpoint",
    },
}

# For dynamic endpoints (with parameters)
DYNAMIC_ENDPOINTS = {
    "new_dynamic": {
        "url_template": f"{FPL_BASE_URL}/endpoint/{{param_id}}/",
        "description": "Description",
        "table": "raw_new_dynamic",
        "params": ["param_id"],
        "id_source": "bootstrap_static",
    },
}
```

## Scheduling

### Option 1: Prefect Deployments

```bash
# Create a deployment
prefect deployment build flows/fpl_ingestion.py:fpl_complete_pipeline \
    --name "FPL Daily Ingestion" \
    --cron "0 2 * * *"  # Run daily at 2 AM

# Apply the deployment
prefect deployment apply fpl_complete_pipeline-deployment.yaml

# Start a worker
prefect worker start --pool default
```

### Option 2: Cron Job

```bash
# Add to crontab
0 2 * * * cd /path/to/pipeline && source venv/bin/activate && python flows/fpl_ingestion.py
```

## Testing

```bash
# Test API connectivity
python -c "from tasks.api_tasks import fetch_fpl_endpoint; print(fetch_fpl_endpoint.fn('https://fantasy.premierleague.com/api/bootstrap-static/', 'test'))"

# Test Snowflake connection
python -c "from utils.snowflake_client import test_connection; test_connection()"

# Run test suite
python test_pipeline.py
```

## Troubleshooting

### "Snowflake not configured"
- Create `.env` file with credentials
- Or run without Snowflake (data will be fetched but not loaded)

### "Rate limit exceeded"
- Increase `RATE_LIMIT_DELAY` in `config.py`
- Reduce `max_players` parameter

### "Connection timeout"
- Check internet connection
- FPL API may be down (check https://fantasy.premierleague.com)

## Key Features

- **Graceful Error Handling**: Continues if one endpoint fails
- **Automatic Retries**: Configurable retry logic for failed requests
- **Rate Limiting**: Respects FPL API limits
- **Optional Snowflake**: Works with or without database
- **Prefect Observability**: Track all runs in Prefect UI
- **Flexible Execution**: Skip endpoints, limit data for testing

## Files

- **config.py**: All configuration and endpoint definitions
- **flows/fpl_ingestion.py**: Main pipeline orchestration
- **tasks/api_tasks.py**: API fetching logic with retries
- **tasks/snowflake_tasks.py**: Database loading logic
- **utils/snowflake_client.py**: Connection management
- **test_pipeline.py**: Testing script

## License

This project is for educational and personal use.
