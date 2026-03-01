"""Local-first training data snapshot utilities."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PIPELINE_ROOT.parent
ALLOWLIST_ROOT = PIPELINE_ROOT / "data" / "training"

LATEST_SNAPSHOT_NAME = "latest.parquet"
LATEST_MANIFEST_NAME = "latest.manifest.json"
SNAPSHOT_FILE_PATTERN = re.compile(r"^features_\d{4}-\d{2}-\d{2}T\d{6}Z\.parquet$")

DEFAULT_SOURCE = "local"
DEFAULT_POLICY = "LOCAL_ONLY"
DEFAULT_SNAPSHOT_MAX_AGE_DAYS = 8
DEFAULT_RETENTION_COUNT = 12

MANIFEST_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0.0"

SUPPORTED_SOURCES = {"local", "snowflake"}
SUPPORTED_POLICIES = {"LOCAL_ONLY", "LOCAL_THEN_SNOWFLAKE"}

REQUIRED_COLUMN_DTYPE_FAMILY = {
    "player_id": "integer",
    "gameweek_id": "integer",
    "position_id": "integer",
    "total_points": "number",
    "minutes_played": "number",
    "ict_index": "number",
    "now_cost": "number",
}

REQUIRED_MANIFEST_FIELDS = {
    "manifest_version",
    "schema_version",
    "created_at_utc",
    "source",
    "query_fingerprint",
    "row_count",
    "column_count",
    "columns",
    "dtypes",
    "min_gameweek",
    "max_gameweek",
    "required_columns_present",
    "contract_hash",
    "file_sha256",
    "refresh_run_id",
}

SENSITIVE_KEY_PATTERN = ("KEY", "TOKEN", "SECRET")


class LocalDataError(RuntimeError):
    """Raised when local snapshot loading fails."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "LOCAL_DATA_ERROR",
        allow_fallback: bool = True,
        detail: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.allow_fallback = allow_fallback
        self.detail = detail or {}


@dataclass(frozen=True)
class DataResolution:
    """Resolved runtime mode for training data loading."""

    source: str
    policy: str
    local_dir: Path
    local_snapshot_path: Path
    max_age_days: int


def _contains_sensitive_key(key: str) -> bool:
    upper = key.upper()
    return any(pattern in upper for pattern in SENSITIVE_KEY_PATTERN)


def redact_sensitive(payload: Any) -> Any:
    """Recursively redact keys that match sensitive patterns."""

    if isinstance(payload, dict):
        redacted: dict[str, Any] = {}
        for key, value in payload.items():
            if _contains_sensitive_key(str(key)):
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = redact_sensitive(value)
        return redacted
    if isinstance(payload, list):
        return [redact_sensitive(item) for item in payload]
    return payload


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def hash_payload(payload: dict[str, Any]) -> str:
    """Hash a dictionary deterministically with SHA-256."""

    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def hash_file(path: Path) -> str:
    """Hash file bytes with SHA-256."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def now_utc_iso() -> str:
    """Return ISO-8601 UTC timestamp."""

    return datetime.now(UTC).isoformat()


def _resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "pipeline":
        return (WORKSPACE_ROOT / path).resolve()
    return (PIPELINE_ROOT / path).resolve()


def _is_within(path: Path, root: Path) -> bool:
    resolved_path = path.resolve(strict=False)
    resolved_root = root.resolve(strict=False)
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def ensure_allowed_path(path: Path, *, root: Optional[Path] = None, reason: str = "read") -> Path:
    """Validate a path is constrained to the local data allowlist root."""

    allowlist_root = root or ALLOWLIST_ROOT
    resolved = path.resolve(strict=False)
    if not _is_within(resolved, allowlist_root):
        raise LocalDataError(
            f"Refusing to {reason} outside allowlist root '{allowlist_root}': {resolved}",
            code="ALLOWLIST_VIOLATION",
            allow_fallback=False,
            detail={"path": str(resolved), "root": str(allowlist_root)},
        )
    return resolved


def resolve_source(source: Optional[str] = None) -> str:
    """Resolve data source from override, env, then default."""

    chosen = (source or os.getenv("TRAINING_DATA_SOURCE") or DEFAULT_SOURCE).strip().lower()
    if chosen not in SUPPORTED_SOURCES:
        raise ValueError(
            f"Invalid TRAINING_DATA_SOURCE '{chosen}'. Expected one of {sorted(SUPPORTED_SOURCES)}"
        )
    return chosen


def resolve_policy(policy: Optional[str] = None) -> str:
    """Resolve data policy from override, env, then default."""

    chosen = (policy or os.getenv("TRAINING_DATA_POLICY") or DEFAULT_POLICY).strip().upper()
    if chosen not in SUPPORTED_POLICIES:
        raise ValueError(
            f"Invalid TRAINING_DATA_POLICY '{chosen}'. Expected one of {sorted(SUPPORTED_POLICIES)}"
        )
    return chosen


def resolve_max_age_days(max_age_days: Optional[int] = None) -> int:
    """Resolve max snapshot age in days."""

    if max_age_days is not None:
        return int(max_age_days)
    env_value = os.getenv("SNAPSHOT_MAX_AGE_DAYS")
    return int(env_value) if env_value else DEFAULT_SNAPSHOT_MAX_AGE_DAYS


def resolve_retention_count(retention_count: Optional[int] = None) -> int:
    """Resolve snapshot retention count."""

    if retention_count is not None:
        return int(retention_count)
    env_value = os.getenv("LOCAL_SNAPSHOT_RETENTION_COUNT")
    return int(env_value) if env_value else DEFAULT_RETENTION_COUNT


def resolve_local_data_dir(output_dir: Optional[str | Path] = None) -> Path:
    """Resolve local snapshot directory and enforce allowlist root."""

    raw = str(output_dir or os.getenv("LOCAL_DATA_DIR") or "pipeline/data/training")
    directory = _resolve_project_path(raw)
    return ensure_allowed_path(directory, reason="use local data directory")


def resolve_local_snapshot_path(
    *,
    local_path: Optional[str | Path] = None,
    local_data_dir: Optional[Path] = None,
) -> Path:
    """Resolve snapshot path from override, env, then latest pointer path."""

    override = local_path or os.getenv("LOCAL_TRAINING_DATA_PATH")
    if override:
        snapshot_path = _resolve_project_path(str(override))
        return ensure_allowed_path(snapshot_path, reason="read snapshot")

    data_dir = local_data_dir or resolve_local_data_dir()
    return ensure_allowed_path(data_dir / LATEST_SNAPSHOT_NAME, reason="read latest snapshot")


def resolve_data_resolution(
    *,
    source: Optional[str] = None,
    policy: Optional[str] = None,
    local_path: Optional[str | Path] = None,
    max_age_days: Optional[int] = None,
) -> DataResolution:
    """Resolve source/policy/path defaults in a single object."""

    resolved_source = resolve_source(source)
    resolved_policy = resolve_policy(policy)
    local_dir = resolve_local_data_dir()
    snapshot_path = resolve_local_snapshot_path(local_path=local_path, local_data_dir=local_dir)
    return DataResolution(
        source=resolved_source,
        policy=resolved_policy,
        local_dir=local_dir,
        local_snapshot_path=snapshot_path,
        max_age_days=resolve_max_age_days(max_age_days),
    )


def _dtype_family(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_numeric_dtype(series):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "string"


def _matches_dtype_family(series: pd.Series, expected_family: str) -> bool:
    if expected_family == "integer":
        if pd.api.types.is_integer_dtype(series):
            return True
        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if non_null.empty:
                return True
            return bool(((non_null % 1) == 0).all())
        return False
    if expected_family == "number":
        return pd.api.types.is_numeric_dtype(series)
    if expected_family == "string":
        return pd.api.types.is_string_dtype(series)
    if expected_family == "boolean":
        return pd.api.types.is_bool_dtype(series)
    if expected_family == "datetime":
        return pd.api.types.is_datetime64_any_dtype(series)
    return False


def build_contract_spec(df: pd.DataFrame) -> dict[str, Any]:
    """Build contract spec for required columns and dtype families."""

    missing: list[str] = []
    dtype_families: dict[str, str] = {}
    mismatches: list[dict[str, str]] = []

    for column, expected in REQUIRED_COLUMN_DTYPE_FAMILY.items():
        if column not in df.columns:
            missing.append(column)
            continue
        observed = _dtype_family(df[column])
        dtype_families[column] = observed
        if not _matches_dtype_family(df[column], expected):
            mismatches.append({"column": column, "expected": expected, "observed": observed})

    required_columns_present = not missing and not mismatches
    return {
        "required_columns": list(REQUIRED_COLUMN_DTYPE_FAMILY.keys()),
        "required_dtype_families": dtype_families,
        "required_columns_present": required_columns_present,
        "missing_columns": missing,
        "dtype_mismatches": mismatches,
    }


def build_contract_hash(*, schema_version: str, contract_spec: dict[str, Any]) -> str:
    """Build deterministic contract hash for schema/version governance."""

    payload = {
        "schema_version": schema_version,
        "required_columns": contract_spec.get("required_columns", []),
        "required_dtype_families": contract_spec.get("required_dtype_families", {}),
    }
    return hash_payload(payload)


def _major_version(version: str) -> int:
    part = str(version).split(".")[0]
    try:
        return int(part)
    except ValueError as exc:
        raise LocalDataError(
            f"Invalid semantic version '{version}'",
            code="INVALID_SCHEMA_VERSION",
            allow_fallback=False,
        ) from exc


def resolve_manifest_path(snapshot_path: Path) -> Path:
    """Resolve manifest path for latest pointer or versioned snapshot."""

    if snapshot_path.name == LATEST_SNAPSHOT_NAME:
        return snapshot_path.with_name(LATEST_MANIFEST_NAME)
    if SNAPSHOT_FILE_PATTERN.match(snapshot_path.name):
        return snapshot_path.with_suffix(".manifest.json")
    raise LocalDataError(
        f"Unsupported snapshot filename '{snapshot_path.name}'.",
        code="INVALID_SNAPSHOT_NAME",
    )


def _validate_manifest_shape(manifest: dict[str, Any]) -> None:
    missing_fields = sorted(REQUIRED_MANIFEST_FIELDS - set(manifest))
    if missing_fields:
        raise LocalDataError(
            f"Snapshot manifest missing required fields: {missing_fields}",
            code="MANIFEST_MISSING_FIELDS",
        )


def _parse_created_at_utc(created_at_utc: str) -> datetime:
    raw = created_at_utc.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(raw)
    except ValueError as exc:
        raise LocalDataError(
            f"Invalid created_at_utc in manifest: {created_at_utc}",
            code="MANIFEST_INVALID_TIMESTAMP",
        ) from exc


def _snapshot_age_days(created_at_utc: str) -> float:
    created = _parse_created_at_utc(created_at_utc)
    now = datetime.now(UTC)
    return max((now - created).total_seconds() / 86400.0, 0.0)


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_local_snapshot(
    *,
    snapshot_path: Path,
    max_age_days: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and validate a local snapshot and its manifest."""

    checked_path = ensure_allowed_path(snapshot_path, reason="read snapshot")
    if not checked_path.exists():
        raise LocalDataError(
            f"Local snapshot not found: {checked_path}",
            code="SNAPSHOT_MISSING",
        )

    manifest_path = ensure_allowed_path(resolve_manifest_path(checked_path), reason="read manifest")
    if not manifest_path.exists():
        raise LocalDataError(
            f"Snapshot manifest not found: {manifest_path}",
            code="MANIFEST_MISSING",
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LocalDataError(
            f"Snapshot manifest is not valid JSON: {manifest_path}",
            code="MANIFEST_CORRUPT",
        ) from exc

    _validate_manifest_shape(manifest)

    expected_schema_version = str(SCHEMA_VERSION)
    manifest_schema_version = str(manifest.get("schema_version"))
    if _major_version(manifest_schema_version) != _major_version(expected_schema_version):
        raise LocalDataError(
            (
                "Snapshot schema major version mismatch. "
                f"Expected {expected_schema_version}, got {manifest_schema_version}."
            ),
            code="SCHEMA_MAJOR_MISMATCH",
            allow_fallback=False,
            detail={
                "expected_schema_version": expected_schema_version,
                "manifest_schema_version": manifest_schema_version,
            },
        )

    age_days = _snapshot_age_days(str(manifest["created_at_utc"]))
    if age_days > float(max_age_days):
        raise LocalDataError(
            f"Snapshot is stale ({age_days:.2f} days > {max_age_days} days): {checked_path}",
            code="SNAPSHOT_STALE",
            detail={"snapshot_age_days": age_days, "max_age_days": max_age_days},
        )

    expected_file_sha = str(manifest["file_sha256"])
    actual_file_sha = hash_file(checked_path)
    if expected_file_sha != actual_file_sha:
        raise LocalDataError(
            "Snapshot file hash mismatch",
            code="SNAPSHOT_HASH_MISMATCH",
            detail={"expected_sha256": expected_file_sha, "actual_sha256": actual_file_sha},
        )

    df = _read_parquet(checked_path)
    df.columns = [str(column).lower() for column in df.columns]

    contract_spec = build_contract_spec(df)
    if not contract_spec["required_columns_present"]:
        raise LocalDataError(
            "Snapshot is missing required columns or dtype contract checks failed",
            code="CONTRACT_VALIDATION_FAILED",
            detail=contract_spec,
        )

    expected_contract_hash = str(manifest["contract_hash"])
    actual_contract_hash = build_contract_hash(
        schema_version=manifest_schema_version,
        contract_spec=contract_spec,
    )
    if expected_contract_hash != actual_contract_hash:
        raise LocalDataError(
            "Snapshot contract hash mismatch",
            code="CONTRACT_HASH_MISMATCH",
            detail={"expected_contract_hash": expected_contract_hash, "actual_contract_hash": actual_contract_hash},
        )

    manifest_hash = hash_payload(redact_sensitive(manifest))
    metadata = {
        "data_source_selected": "local",
        "snapshot_path": str(checked_path),
        "snapshot_age_days": float(age_days),
        "schema_version": manifest_schema_version,
        "fallback_taken": False,
        "fallback_reason": None,
        "manifest_hash": manifest_hash,
        "snapshot_manifest": redact_sensitive(manifest),
    }
    return df, metadata


def load_training_dataframe(
    *,
    table_name: str,
    source: Optional[str] = None,
    local_path: Optional[str | Path] = None,
    policy: Optional[str] = None,
    max_age_days: Optional[int] = None,
    snowflake_loader: Optional[Callable[[str], pd.DataFrame]] = None,
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load training data according to local-first source/policy settings."""

    logger = logger or logging.getLogger(__name__)
    resolution = resolve_data_resolution(
        source=source,
        policy=policy,
        local_path=local_path,
        max_age_days=max_age_days,
    )

    if resolution.source == "snowflake":
        if snowflake_loader is None:
            raise ValueError("snowflake_loader is required when source='snowflake'")
        df = snowflake_loader(table_name)
        metadata = {
            "data_source_selected": "snowflake",
            "data_policy": resolution.policy,
            "snapshot_path": None,
            "snapshot_age_days": None,
            "schema_version": None,
            "fallback_taken": False,
            "fallback_reason": None,
            "manifest_hash": None,
            "snapshot_manifest": None,
            "fallback_event": None,
        }
        return df, metadata

    try:
        df, local_metadata = load_local_snapshot(
            snapshot_path=resolution.local_snapshot_path,
            max_age_days=resolution.max_age_days,
        )
        local_metadata["data_policy"] = resolution.policy
        local_metadata["fallback_event"] = None
        return df, local_metadata
    except LocalDataError as exc:
        if resolution.policy == "LOCAL_THEN_SNOWFLAKE" and exc.allow_fallback:
            if snowflake_loader is None:
                raise ValueError("snowflake_loader is required for LOCAL_THEN_SNOWFLAKE fallback") from exc
            logger.warning("Local snapshot unavailable; falling back to Snowflake (%s)", exc)
            df = snowflake_loader(table_name)
            fallback_event = {
                "fallback_taken": True,
                "fallback_reason": str(exc),
                "fallback_code": exc.code,
                "fallback_at_utc": now_utc_iso(),
                "requested_source": resolution.source,
            }
            metadata = {
                "data_source_selected": "snowflake",
                "data_policy": resolution.policy,
                "snapshot_path": str(resolution.local_snapshot_path),
                "snapshot_age_days": None,
                "schema_version": None,
                "fallback_taken": True,
                "fallback_reason": str(exc),
                "manifest_hash": None,
                "snapshot_manifest": None,
                "fallback_event": fallback_event,
            }
            return df, metadata
        raise


def emit_selection_log(
    logger: logging.Logger,
    *,
    context: str,
    metadata: dict[str, Any],
) -> None:
    """Emit structured source/policy selection fields for observability."""

    payload = {
        "context": context,
        "data_source_selected": metadata.get("data_source_selected"),
        "data_policy": metadata.get("data_policy"),
        "snapshot_path": metadata.get("snapshot_path"),
        "snapshot_age_days": metadata.get("snapshot_age_days"),
        "schema_version": metadata.get("schema_version"),
        "fallback_taken": bool(metadata.get("fallback_taken")),
        "fallback_reason": metadata.get("fallback_reason"),
        "manifest_hash": metadata.get("manifest_hash"),
    }
    logger.info("Training data source selection %s", json.dumps(payload, sort_keys=True))


def validate_source_table_name(table_name: str) -> str:
    """Validate source table identifier before embedding into SQL."""

    candidate = table_name.strip()
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_\.]*$", candidate):
        raise ValueError(f"Invalid source table identifier: {table_name}")
    return candidate


def build_training_query(table_name: str, *, apply_training_filters: bool) -> str:
    """Build canonical Snowflake query for feature extraction."""

    safe_table_name = validate_source_table_name(table_name)
    where_clause = ""
    if apply_training_filters:
        where_clause = """
    WHERE f.gameweek_id > 3
        AND f.minutes_played > 0
"""
    return f"""
    SELECT
        f.* EXCLUDE (now_cost),
        COALESCE(f.now_cost, p.current_value) AS now_cost,
        p.position_id
    FROM {safe_table_name} f
    LEFT JOIN dim_players p ON f.player_id = p.player_id
{where_clause}
    """
