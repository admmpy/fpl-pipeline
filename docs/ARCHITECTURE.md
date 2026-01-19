# FPL Pipeline Architecture

## Data Flow Overview

```
┌─────────────┐
│   FPL API   │
└──────┬──────┘
       │
       ├─────────────────────────────────────┬──────────────────────────┐
       │                                     │                          │
       ▼                                     ▼                          ▼
┌──────────────┐                    ┌──────────────┐          ┌─────────────┐
│  bootstrap   │                    │   fixtures   │          │  element    │
│   -static    │                    │              │          │  -summary   │
└──────┬───────┘                    └──────┬───────┘          └──────┬──────┘
       │                                   │                         │
       │ Parse to typed records            │ Parse                   │ Keep as
       │ (Python)                          │ (Python)                │ JSON
       │                                   │                         │
       ▼                                   ▼                         ▼
┌──────────────────────────────────────────────────┐       ┌─────────────────┐
│        Snowflake Typed Tables (MERGE)            │       │ VARIANT Table   │
│  ┌─────────┐ ┌───────┐ ┌────────────┐ ┌────────┤       │  (INSERT)       │
│  │ players │ │ teams │ │ gameweeks  │ │fixtures│       │                 │
│  └─────────┘ └───────┘ └────────────┘ └────────┤       │ raw_element_    │
│  INTEGER    VARCHAR    TIMESTAMP_NTZ   BOOLEAN  │       │ summary         │
└──────────────────────────────────────────────────┘       └─────────────────┘
       │                                                            │
       │                                                            │
       └─────────────────┬──────────────────────────────────────────┘
                         │
                         ▼
                ┌─────────────────┐
                │   dbt Staging   │
                │                 │
                │  stg_players    │
                │  stg_teams      │
                │  stg_fixtures   │
                │  stg_player_    │  ◄─── Parses VARIANT with FLATTEN
                │    history      │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │   dbt Marts     │
                │                 │
                │  dim_players    │
                │  dim_teams      │
                │  fct_players_   │
                │    gameweek     │
                └─────────────────┘
```

## Table Strategy

### Typed Tables (MERGE/UPSERT)

**Purpose:** Reference/dimension data that updates frequently

| Table | Primary Key | Update Frequency | Why Typed? |
|-------|-------------|------------------|------------|
| `players` | `player_id` | Daily | Snapshot data, frequently queried |
| `teams` | `team_id` | Daily | Small table, direct queries common |
| `gameweeks` | `gameweek_id` | Weekly | Metadata lookups |
| `fixtures` | `fixture_id` | Daily | Scores update after matches |

**Benefits:**
- ✅ MERGE prevents duplicate key violations
- ✅ No VARIANT parsing overhead in queries
- ✅ Type safety (INTEGER, VARCHAR, TIMESTAMP)
- ✅ Idempotent pipeline (safe to re-run)

### VARIANT Table (INSERT)

**Purpose:** Complex nested historical data

| Table | Primary Key | Update Frequency | Why VARIANT? |
|-------|-------------|------------------|--------------|
| `raw_element_summary` | Auto-increment | Per player fetch | Nested history array, parsed in dbt |

**Benefits:**
- ✅ Handles variable schema gracefully
- ✅ Stores full API response for audit
- ✅ dbt already optimised for FLATTEN operations
- ✅ Flexible for future API changes

## Pipeline Functions

### 1. `ingest_static_endpoints_typed()`

Fetches and loads to **typed tables**:

```python
fetch_fpl_endpoint("bootstrap-static")
  → parse_players_from_bootstrap(data)
  → MERGE INTO players (player_id) 
  
  → parse_teams_from_bootstrap(data)  
  → MERGE INTO teams (team_id)
  
  → parse_gameweeks_from_bootstrap(data)
  → MERGE INTO gameweeks (gameweek_id)

fetch_fpl_endpoint("fixtures")
  → parse_fixtures(data)
  → MERGE INTO fixtures (fixture_id)
```

### 2. `ingest_player_details()`

Fetches and loads to **VARIANT table**:

```python
for player_id in player_list:
    fetch_fpl_endpoint(f"element-summary/{player_id}")
      → INSERT INTO raw_element_summary (data)
```

### 3. `fpl_typed_pipeline()` (Hybrid)

Combines both approaches:

```python
ingest_static_endpoints_typed()  # MERGE to typed tables
ingest_player_details()          # INSERT to VARIANT
```

## dbt Staging Layer

### Typed Table Sources (Simple SELECT)

```sql
-- stg_players.sql
SELECT 
    player_id,
    web_name,
    now_cost / 10.0 AS now_cost,  -- Simple transformation
    total_points
FROM {{ source('fpl_raw', 'players') }}
QUALIFY ROW_NUMBER() OVER (...) = 1  -- Dedup if needed
```

### VARIANT Source (FLATTEN + Parse)

```sql
-- stg_player_history.sql
WITH flattened AS (
    SELECT
        f.value:element::INTEGER AS player_id,
        f.value:round::INTEGER AS gameweek_id,
        f.value:total_points::INTEGER AS total_points,
        -- ... more fields
    FROM {{ source('fpl_raw', 'raw_element_summary') }},
    LATERAL FLATTEN(input => data:history) AS f
)
SELECT * FROM flattened
```

## When to Use Each Strategy

### Use Typed Tables + MERGE When:
- ✅ Schema is stable and known
- ✅ Data updates frequently (idempotency needed)
- ✅ Table is frequently queried directly
- ✅ You want type safety and validation

### Use VARIANT + INSERT When:
- ✅ Schema is complex or nested
- ✅ Schema might change frequently
- ✅ Data is primarily accessed via dbt transformations
- ✅ You need audit trail of raw responses
- ✅ Performance of FLATTEN in dbt is acceptable

## Migration Path

If you need to migrate from VARIANT to Typed:

1. **Add new typed table** with MERGE logic
2. **Run hybrid pipeline** (both tables load)
3. **Update dbt staging** to point to typed table
4. **Test thoroughly** with fresh data
5. **Deprecate VARIANT table** after validation

## Performance Considerations

### Typed Tables
- **Pros:** Faster queries, smaller storage, no JSON parsing
- **Cons:** Less flexible, requires schema changes for new fields

### VARIANT Tables
- **Pros:** Flexible schema, stores complete response
- **Cons:** Slower queries, FLATTEN overhead, larger storage

### Current Balance
- Static data (80% of queries) → Typed (fast)
- Historical data (20% of queries) → VARIANT (flexible)

## Configuration

All table schemas defined in `config.py`:

```python
TABLE_SCHEMAS = {
    "players": PLAYER_SCHEMA,
    "teams": TEAM_SCHEMA,
    "gameweeks": GAMEWEEK_SCHEMA,
    "fixtures": FIXTURE_SCHEMA,
}

# VARIANT tables created dynamically
```

## Monitoring

### Check Typed Table Freshness

```sql
SELECT 
    'players' AS table_name,
    COUNT(*) AS row_count,
    MAX(ingestion_timestamp) AS last_update
FROM fpl_raw.players

UNION ALL

SELECT 
    'fixtures',
    COUNT(*),
    MAX(ingestion_timestamp)
FROM fpl_raw.fixtures;
```

### Check VARIANT Table Freshness

```sql
SELECT 
    COUNT(DISTINCT data:id) AS unique_players,
    MAX(ingestion_timestamp) AS last_update
FROM fpl_raw.raw_element_summary;
```

---

**Last Updated:** 2026-01-19  
**Architecture:** Hybrid (Typed + VARIANT)  
**Strategy:** Optimise for common queries, keep flexibility for complex data
