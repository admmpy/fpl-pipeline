# Getting Started with Your FPL Pipeline

## What You Have

A production-ready data pipeline that:

1. **Fetches FPL data** from multiple API endpoints
2. **Loads to Snowflake** (optional - works without it too)
3. **Handles errors gracefully** with retries and logging
4. **Uses Prefect** for orchestration and observability

## Project Structure

```
pipeline/
├── config.py               # All configuration in one place
├── flows/
│   └── fpl_ingestion.py   # Main pipeline orchestration
├── tasks/
│   ├── api_tasks.py       # API fetching logic
│   └── snowflake_tasks.py # Database loading logic
├── utils/
│   └── snowflake_client.py # Connection management
└── test_pipeline.py       # Test script
```

## How It Works

### 1. Configuration (config.py)

All endpoints and settings are defined here:

- **STATIC_ENDPOINTS**: Simple URLs (bootstrap-static, fixtures, overall_league)
- **DYNAMIC_ENDPOINTS**: Require parameters (player details, live gameweek)
- **get_snowflake_config()**: Loads credentials from .env

### 2. API Tasks (tasks/api_tasks.py)

- `fetch_fpl_endpoint()`: Fetches a single URL with retries
- `extract_player_ids()`: Gets all player IDs from bootstrap data
- `extract_current_gameweek()`: Finds active gameweek
- `fetch_dynamic_endpoint_batch()`: Fetches multiple endpoints (e.g., all players)

### 3. Snowflake Tasks (tasks/snowflake_tasks.py)

- `ensure_table_exists()`: Creates tables if needed
- `load_to_snowflake()`: Loads single record
- `load_batch_to_snowflake()`: Loads multiple records efficiently

### 4. Main Flow (flows/fpl_ingestion.py)

- `ingest_static_endpoints()`: Fetches bootstrap, fixtures, league
- `ingest_player_details()`: Fetches all player stats
- `ingest_live_gameweek()`: Fetches current gameweek data
- `fpl_complete_pipeline()`: Orchestrates everything

## Running the Pipeline

### Option 1: Test Mode (No Snowflake Required)

```bash
cd /Users/am/Sync/fpl-workspace/pipeline
source venv/bin/activate
python test_pipeline.py
```

This will:
- Fetch 5 players (for speed)
- Skip Snowflake loading
- Show you results

### Option 2: Production Mode (With Snowflake)

1. **Set up .env file:**
```bash
# Create .env with your credentials
nano .env
```

Add:
```
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_user
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema
```

2. **Run full pipeline:**
```bash
python flows/fpl_ingestion.py
```

Or for all players:
```bash
python test_pipeline.py --full
```

## Key Features

### 1. Graceful Degradation
- Works without Snowflake (just fetches data)
- Continues if one endpoint fails
- Retries failed requests automatically

### 2. Rate Limiting
- 0.5 second delay between requests
- Respects FPL API limits
- Configurable in config.py

### 3. Observability
- Prefect UI shows all runs
- Detailed logging for debugging
- Progress indicators for long operations

### 4. Flexible Execution
```python
from flows.fpl_ingestion import fpl_complete_pipeline

# Full pipeline
fpl_complete_pipeline()

# Skip player details (faster)
fpl_complete_pipeline(include_player_details=False)

# Test with 10 players
fpl_complete_pipeline(max_players=10)
```

## Data Flow

```
FPL API → API Tasks → Snowflake Tasks → Snowflake (VARIANT columns)
```

Data is stored as raw JSON in Snowflake, ready for downstream transformation.

## Next Steps

### 1. Test the API Fetching
```bash
python test_pipeline.py
```

### 2. Set Up Snowflake (Optional)
- Create database and schema in Snowflake
- Add credentials to .env
- Run: `python test_pipeline.py --full`

### 3. Schedule the Pipeline
```bash
# Daily at 2 AM
prefect deployment build flows/fpl_ingestion.py:fpl_complete_pipeline \
    --name "FPL Daily" \
    --cron "0 2 * * *"

prefect deployment apply fpl_complete_pipeline-deployment.yaml
prefect worker start --pool default
```

## Troubleshooting

### "Snowflake not configured"
- This is normal if you haven't set up .env
- Pipeline will still fetch data, just won't load it

### "Connection timeout"
- FPL API might be slow or down
- Increase timeout in api_tasks.py (currently 30 seconds)

### "Too many requests"
- Increase RATE_LIMIT_DELAY in config.py
- Default is 0.5 seconds

## Snowflake Tables

Each endpoint creates a table with:
- `id`: Auto-incrementing primary key
- `ingestion_timestamp`: When data was loaded
- `data`: VARIANT column storing raw JSON
- `source_url`: API endpoint URL
- `endpoint_name`: Endpoint identifier

Query example:
```sql
SELECT 
    data:elements[0]:web_name::STRING as player_name,
    data:elements[0]:now_cost::NUMBER as price
FROM raw_bootstrap_static
ORDER BY ingestion_timestamp DESC
LIMIT 1;
```

## Key Files to Customize

1. **config.py**: Add/remove endpoints, adjust rate limits
2. **flows/fpl_ingestion.py**: Change pipeline logic
3. **tasks/api_tasks.py**: Modify API fetching behaviour
4. **test_pipeline.py**: Adjust test parameters

## Resources

### Prefect
- Docs: https://docs.prefect.io
- Run `prefect server start` for UI
- View flows at http://localhost:4200

### FPL API
- Base URL: https://fantasy.premierleague.com/api
- No authentication required
- Unofficial API - use responsibly

### Snowflake
- VARIANT columns: Store JSON without schema
- Parse using `data:field::TYPE` syntax
- Efficient for semi-structured data

That's it! You're ready to start ingesting FPL data.
