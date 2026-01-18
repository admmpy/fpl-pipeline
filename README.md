# FPL Data Pipeline

A Python pipeline for ingesting Fantasy Premier League data into Snowflake using Prefect orchestration.

## Overview

Fetches data from the FPL API and loads it into Snowflake tables with proper typed columns. Built with Prefect for orchestration, error handling, and observability.

## Architecture

```
FPL API → Fetch → Parse JSON → Load to Snowflake (typed columns)
```

Data is transformed in Python and inserted into properly typed Snowflake tables.

## Project Structure

```
pipeline/
├── config.py                 # Endpoints, schemas, settings
├── .env                      # Snowflake credentials (not in git)
├── requirements.txt          # Python dependencies
├── flows/
│   └── fpl_ingestion.py     # Main Prefect flow
├── tasks/
│   ├── api_tasks.py         # API fetching
│   ├── transformation_tasks.py  # JSON parsing
│   └── snowflake_tasks.py   # Database loading
└── utils/
    └── snowflake_client.py  # Connection management
```

## Quick Start

### 1. Set Up Environment

```bash
# Activate virtual environment
source venv/bin/activate

# Verify dependencies
pip list | grep prefect
```

### 2. Configure Snowflake

Create a `.env` file:

```bash
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_user
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema
```

### 3. Run the Pipeline

```bash
# Run the typed pipeline
python flows/fpl_ingestion.py
```

Or import and run:

```python
from flows.fpl_ingestion import fpl_typed_pipeline
fpl_typed_pipeline()
```

## Data Tables

The pipeline creates four Snowflake tables:

### players
```sql
- player_id (PK)
- web_name, first_name, second_name
- team_id, position_id
- now_cost, form, total_points
- goals_scored, assists, clean_sheets
- status, chance_of_playing_next_round
- ingestion_timestamp, gameweek_fetched
```

### teams
```sql
- team_id (PK)
- name, short_name
- strength, position, points
- strength_overall_home, strength_overall_away
- strength_attack_home, strength_attack_away
- strength_defence_home, strength_defence_away
- ingestion_timestamp
```

### gameweeks
```sql
- gameweek_id (PK)
- name, deadline_time
- finished, is_current, is_next
- average_entry_score, highest_score
- most_selected, most_captained
- ingestion_timestamp
```

### fixtures
```sql
- fixture_id (PK)
- gameweek_id, kickoff_time
- team_h, team_a
- team_h_score, team_a_score
- finished, started
- ingestion_timestamp
```

## Configuration

### API Settings (config.py)

```python
RATE_LIMIT_DELAY = 0.5  # Seconds between API calls
MAX_RETRIES = 3         # Number of retries on failure
RETRY_DELAY = 5         # Seconds between retries
```

### Adding New Endpoints

Edit `config.py` and add to `STATIC_ENDPOINTS` or `DYNAMIC_ENDPOINTS`. Then create corresponding schema in `TABLE_SCHEMAS` and parsing function in `transformation_tasks.py`.

## Scheduling

### Prefect Deployment

```bash
# Create a deployment
prefect deployment build flows/fpl_ingestion.py:fpl_typed_pipeline \
    --name "FPL Weekly Ingestion" \
    --cron "0 2 * * 4"  # Every Thursday at 2 AM

# Apply and start worker
prefect deployment apply fpl_typed_pipeline-deployment.yaml
prefect worker start --pool default
```

### Weekly Orchestration (ML + Optimisation)

The orchestration flow runs ingestion, dbt, ML inference, and squad optimisation:

```bash
python deploy.py
```

Start a local worker to pick up scheduled runs:

```bash
prefect worker start --pool local-mac-worker
```

If you want Slack alerts, add this to your `.env`:

```bash
SLACK_WEBHOOK_URL=your_webhook_url
```

### Cron Job

```bash
# Add to crontab (runs every Thursday at 2 AM)
0 2 * * 4 cd /path/to/pipeline && source venv/bin/activate && python flows/fpl_ingestion.py
```

## Example Queries

```sql
-- Top scoring players
SELECT web_name, total_points, now_cost/10.0 as price
FROM players
ORDER BY total_points DESC
LIMIT 10;

-- Best value players (points per million)
SELECT web_name, total_points, 
       now_cost/10.0 as price,
       (total_points / (now_cost/10.0)) as value
FROM players
WHERE minutes > 500
ORDER BY value DESC
LIMIT 10;

-- Team standings
SELECT name, position, points
FROM teams
ORDER BY position;

-- Upcoming fixtures
SELECT f.fixture_id, 
       h.name as home_team,
       a.name as away_team,
       f.kickoff_time
FROM fixtures f
JOIN teams h ON f.team_h = h.team_id
JOIN teams a ON f.team_a = a.team_id
WHERE f.finished = FALSE
ORDER BY f.kickoff_time;
```

## Troubleshooting

### "Snowflake not configured"
- Create `.env` file with credentials
- Pipeline will fetch data but skip loading without Snowflake

### "Rate limit exceeded"
- Increase `RATE_LIMIT_DELAY` in `config.py`

### "Connection timeout"
- Check internet connection
- FPL API may be temporarily down

## Key Features

- **Typed Columns**: Data loaded with proper types (INTEGER, VARCHAR, FLOAT)
- **Normalized Tables**: Separate tables for players, teams, gameweeks, fixtures
- **Error Handling**: Retries, graceful failures, detailed logging
- **Rate Limiting**: Respects FPL API limits
- **Prefect Orchestration**: Task dependencies, observability, scheduling
- **Query Ready**: Data immediately available for analysis

## Dependencies

- Python 3.12+
- Prefect 3.x (orchestration)
- snowflake-connector-python (database)
- requests (HTTP)
- python-dotenv (configuration)

## License

This project is for educational and personal use.
