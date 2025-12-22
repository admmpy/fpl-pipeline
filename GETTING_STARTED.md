# Getting Started with Your FPL Pipeline

## What You Have

A Python pipeline that fetches Fantasy Premier League data and loads it into Snowflake with proper typed columns.

## Architecture

```
FPL API → Python (parse JSON) → Snowflake (typed tables)
```

One API call (bootstrap-static) is split into three normalized tables (players, teams, gameweeks).

## How It Works

### 1. Configuration (config.py)
- Defines API endpoints
- Defines table schemas
- Pipeline settings (rate limits, retries)

### 2. API Tasks (tasks/api_tasks.py)
- Fetches data from FPL API
- Handles retries and rate limiting
- Returns raw JSON responses

### 3. Transformation (tasks/transformation_tasks.py)
- Parses JSON responses
- Maps API fields to database columns
- Handles type conversions and nulls

### 4. Snowflake Tasks (tasks/snowflake_tasks.py)
- Creates typed tables
- Inserts parsed data
- Handles connection errors

### 5. Flow (flows/fpl_ingestion.py)
- Orchestrates the entire pipeline
- Runs: fetch → parse → load

## Running the Pipeline

```bash
cd /Users/am/Sync/fpl-workspace/pipeline
source venv/bin/activate
python flows/fpl_ingestion.py
```

This will:
1. Fetch bootstrap-static (players, teams, gameweeks)
2. Fetch fixtures
3. Parse all data into typed dictionaries
4. Create Snowflake tables if needed
5. Load all data

## Setting Up Snowflake

Create `.env` file:

```bash
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_user  
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema
```

Without this, the pipeline will fetch data but skip loading to Snowflake.

## Verifying Your Data

After running, check Snowflake:

```sql
-- See your tables
SHOW TABLES;

-- Count records
SELECT COUNT(*) FROM players;    -- Should be ~770
SELECT COUNT(*) FROM teams;      -- Should be 20
SELECT COUNT(*) FROM gameweeks;  -- Should be 38
SELECT COUNT(*) FROM fixtures;   -- Should be 380

-- Sample player data
SELECT * FROM players LIMIT 5;
```

## Key Concepts

### Typed Columns
Data is stored with proper types:
- `INTEGER` for IDs, costs, stats
- `VARCHAR` for names
- `FLOAT` for calculated values
- `BOOLEAN` for flags
- `TIMESTAMP_NTZ` for dates

This means you can query directly without JSON parsing:

```sql
-- Simple and fast
SELECT web_name, total_points FROM players;

-- Not needed anymore:
-- SELECT data:web_name::STRING FROM raw_data;
```

### Normalization
One API response becomes multiple tables:
- **bootstrap-static** → players + teams + gameweeks
- **fixtures** → fixtures

This eliminates data duplication and makes queries cleaner.

### Error Handling
- API calls retry 3 times on failure
- 0.5 second delay between requests
- Pipeline continues even if one endpoint fails
- Detailed logging for debugging

## Scheduling

Run every Thursday morning at 2 AM:

```bash
prefect deployment build flows/fpl_ingestion.py:fpl_typed_pipeline \
    --name "FPL Weekly" \
    --cron "0 2 * * 4"  # Thursday = 4

prefect deployment apply fpl_typed_pipeline-deployment.yaml
prefect worker start --pool default
```

**Cron syntax:** `0 2 * * 4`
- `0` = minute (0)
- `2` = hour (2 AM)
- `*` = every day of month
- `*` = every month
- `4` = Thursday (0=Sunday, 1=Monday, ..., 4=Thursday)

To change the time, modify the hour (e.g., `0 6 * * 4` for 6 AM).

## Troubleshooting

### Import Errors
```bash
# Make sure venv is activated
source venv/bin/activate

# Verify packages
pip list | grep prefect
pip list | grep snowflake
```

### Snowflake Errors
- Check `.env` credentials are correct
- Test connection: `python -c "from utils.snowflake_client import test_connection; test_connection()"`

### API Timeouts
- FPL API might be slow during gameweeks
- Pipeline will retry automatically

## What's Next

1. **Run the pipeline** - Get data flowing
2. **Query your data** - Explore in Snowflake
3. **Schedule it** - Automate with Prefect
4. **Build dashboards** - Connect to your BI tool
5. **Add more endpoints** - Extend the pipeline

## Resources

- **Prefect Docs**: https://docs.prefect.io
- **FPL API**: https://fantasy.premierleague.com/api
- **Snowflake Docs**: https://docs.snowflake.com
