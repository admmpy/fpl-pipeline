# Local-First Training Data Workflow

## Purpose

The training/autonomous stack now defaults to local-first snapshot loading to reduce routine Snowflake spend while preserving Snowflake as source-of-truth and controlled fallback.

## Snapshot Layout

All snapshot artefacts are stored under `pipeline/data/training/`:

- `features_YYYY-MM-DDTHHMMSSZ.parquet`
- `features_YYYY-MM-DDTHHMMSSZ.manifest.json`
- `latest.parquet`
- `latest.manifest.json`
- `.refresh.lock` (ephemeral during refresh)

`latest.*` files are updated atomically from a versioned snapshot.

## Environment Variables

- `TRAINING_DATA_SOURCE`:
  - `local` (default)
  - `snowflake`
- `TRAINING_DATA_POLICY`:
  - `LOCAL_ONLY` (default)
  - `LOCAL_THEN_SNOWFLAKE`
- `LOCAL_DATA_DIR`:
  - default `pipeline/data/training`
- `LOCAL_TRAINING_DATA_PATH`:
  - optional explicit snapshot path
- `SNAPSHOT_MAX_AGE_DAYS`:
  - default `8`
- `LOCAL_SNAPSHOT_RETENTION_COUNT`:
  - default `12`

## Refresh Snapshot

Run weekly (or on-demand) from `pipeline/`:

```bash
python scripts/refresh_local_training_snapshot.py
```

Optional flags:

```bash
python scripts/refresh_local_training_snapshot.py \
  --output-dir pipeline/data/training \
  --source-table fct_ml_player_features \
  --retention-count 12 \
  --force
```

## Runtime Modes

### Train with local-only policy

```bash
TRAINING_DATA_SOURCE=local \
TRAINING_DATA_POLICY=LOCAL_ONLY \
python scripts/train_model.py
```

### Autonomous loop with local-only policy

```bash
TRAINING_DATA_SOURCE=local \
TRAINING_DATA_POLICY=LOCAL_ONLY \
python scripts/run_autonomous_loop.py --rules-path config/domain_rules.yaml
```

### Temporary Snowflake override

```bash
TRAINING_DATA_SOURCE=snowflake \
python scripts/run_autonomous_loop.py --rules-path config/domain_rules.yaml
```

## CLI Overrides (Autonomous Loop)

`scripts/run_autonomous_loop.py` now supports:

- `--data-source local|snowflake`
- `--data-policy LOCAL_ONLY|LOCAL_THEN_SNOWFLAKE`
- `--local-snapshot-path <path>`

CLI flags override environment values.

## Validation and Fallback

Local snapshot loading validates:

- allowlisted path under `pipeline/data/training`
- manifest presence and required fields
- snapshot freshness (`SNAPSHOT_MAX_AGE_DAYS`)
- file hash integrity
- contract hash and schema major compatibility

Policy behaviour:

- `LOCAL_ONLY`: fail fast on local validation errors
- `LOCAL_THEN_SNOWFLAKE`: fallback for recoverable local errors (for example missing/stale/corrupt local files)
- schema major mismatch does **not** fallback; it fails fast unless source is explicitly set to `snowflake`

## Observability

Structured source-selection fields are emitted in training/task/autonomous entrypoints:

- `data_source_selected`
- `data_policy`
- `snapshot_path`
- `snapshot_age_days`
- `schema_version`
- `fallback_taken`
- `fallback_reason`
- `manifest_hash`

Autonomous evidence also records:

- `data_source`
- `snapshot_manifest`
- `fallback_event` (when fallback occurs)
