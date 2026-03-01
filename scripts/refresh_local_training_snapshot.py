"""Refresh local training snapshot from Snowflake with atomic pointer updates."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# Add pipeline root for script execution compatibility.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_snowflake_config
from utils.local_data import (
    LATEST_MANIFEST_NAME,
    LATEST_SNAPSHOT_NAME,
    MANIFEST_VERSION,
    SCHEMA_VERSION,
    build_contract_hash,
    build_contract_spec,
    build_training_query,
    ensure_allowed_path,
    hash_file,
    hash_payload,
    now_utc_iso,
    redact_sensitive,
    resolve_local_data_dir,
    resolve_max_age_days,
    resolve_retention_count,
)
from utils.snowflake_client import get_snowflake_connection

LOGGER = logging.getLogger(__name__)
LOCK_FILENAME = ".refresh.lock"


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh local-first training snapshot from Snowflake.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--source-table", default="fct_ml_player_features")
    parser.add_argument("--retention-count", type=int, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refresh even when latest snapshot is still fresh.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, path)


def _acquire_lock(lock_path: Path) -> int:
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Refresh lock already held: {lock_path}") from exc

    payload = json.dumps(
        {
            "pid": os.getpid(),
            "acquired_at_utc": now_utc_iso(),
        },
        sort_keys=True,
    ).encode("utf-8")
    os.write(fd, payload)
    return fd


def _release_lock(lock_path: Path, fd: Optional[int]) -> None:
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass
    try:
        if lock_path.exists():
            lock_path.unlink()
    except OSError:
        LOGGER.warning("Failed to remove refresh lock: %s", lock_path)


def _fetch_from_snowflake(source_table: str) -> pd.DataFrame:
    if get_snowflake_config() is None:
        raise RuntimeError("Snowflake configuration is required for snapshot refresh.")
    query = build_training_query(source_table, apply_training_filters=False)
    with get_snowflake_connection() as conn:
        df = pd.read_sql(query, conn)
    df.columns = [str(column).lower() for column in df.columns]
    return df


def _write_parquet_atomic(df: pd.DataFrame, target_path: Path) -> None:
    tmp_path = target_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, target_path)


def _snapshot_age_days(created_at_utc: str) -> float:
    created = datetime.fromisoformat(created_at_utc.replace("Z", "+00:00"))
    return max((datetime.now(UTC) - created).total_seconds() / 86400.0, 0.0)


def _replace_latest_files(
    *,
    latest_parquet: Path,
    latest_manifest: Path,
    new_parquet_tmp: Path,
    new_manifest_tmp: Path,
) -> None:
    parquet_backup = latest_parquet.with_suffix(".parquet.bak")
    manifest_backup = latest_manifest.with_suffix(".manifest.json.bak")

    backups_created: list[Path] = []
    try:
        if latest_parquet.exists():
            shutil.copy2(latest_parquet, parquet_backup)
            backups_created.append(parquet_backup)
        if latest_manifest.exists():
            shutil.copy2(latest_manifest, manifest_backup)
            backups_created.append(manifest_backup)

        os.replace(new_parquet_tmp, latest_parquet)
        os.replace(new_manifest_tmp, latest_manifest)

    except Exception as exc:
        if parquet_backup.exists():
            shutil.copy2(parquet_backup, latest_parquet)
        if manifest_backup.exists():
            shutil.copy2(manifest_backup, latest_manifest)
        raise RuntimeError("Failed to atomically update latest snapshot pointers") from exc
    finally:
        for backup in backups_created:
            try:
                backup.unlink()
            except OSError:
                pass
        for tmp_path in (new_parquet_tmp, new_manifest_tmp):
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass


def _build_manifest(
    *,
    df: pd.DataFrame,
    source_table: str,
    query_fingerprint: str,
    parquet_path: Path,
    refresh_run_id: str,
    created_at_utc: str,
) -> dict[str, Any]:
    contract_spec = build_contract_spec(df)
    if not contract_spec["required_columns_present"]:
        raise RuntimeError(
            "Snapshot refresh contract validation failed: "
            f"{json.dumps(contract_spec, sort_keys=True)}"
        )

    contract_hash = build_contract_hash(
        schema_version=SCHEMA_VERSION,
        contract_spec=contract_spec,
    )

    if "gameweek_id" in df.columns:
        min_gameweek = int(df["gameweek_id"].min()) if not df.empty else None
        max_gameweek = int(df["gameweek_id"].max()) if not df.empty else None
    else:
        min_gameweek = None
        max_gameweek = None

    dtypes = {column: str(dtype) for column, dtype in df.dtypes.items()}
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": created_at_utc,
        "source": "snowflake",
        "source_table": source_table,
        "query_fingerprint": query_fingerprint,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
        "dtypes": dtypes,
        "min_gameweek": min_gameweek,
        "max_gameweek": max_gameweek,
        "required_columns_present": bool(contract_spec["required_columns_present"]),
        "contract_hash": contract_hash,
        "file_sha256": hash_file(parquet_path),
        "refresh_run_id": refresh_run_id,
        "required_dtype_contract": contract_spec["required_dtype_families"],
        "versioned_snapshot": parquet_path.name,
    }
    return redact_sensitive(manifest)


def _prune_retention(output_dir: Path, *, retention_count: int, preserve: set[str]) -> None:
    versioned = sorted(output_dir.glob("features_*.parquet"), key=lambda path: path.name, reverse=True)
    kept = 0
    for parquet_path in versioned:
        if parquet_path.name in preserve:
            kept += 1
            continue
        if kept < retention_count:
            kept += 1
            continue

        manifest_path = parquet_path.with_suffix(".manifest.json")
        LOGGER.info("Pruning old snapshot %s", parquet_path.name)
        try:
            parquet_path.unlink()
        except OSError:
            LOGGER.warning("Failed to prune snapshot %s", parquet_path)
        if manifest_path.exists():
            try:
                manifest_path.unlink()
            except OSError:
                LOGGER.warning("Failed to prune manifest %s", manifest_path)


def refresh_snapshot(
    *,
    output_dir: Optional[str] = None,
    source_table: str = "fct_ml_player_features",
    retention_count: Optional[int] = None,
    force: bool = False,
) -> dict[str, Any]:
    """Refresh snapshot and return run summary."""

    resolved_output_dir = resolve_local_data_dir(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    lock_path = ensure_allowed_path(resolved_output_dir / LOCK_FILENAME, reason="acquire refresh lock")
    latest_parquet = ensure_allowed_path(resolved_output_dir / LATEST_SNAPSHOT_NAME, reason="write latest snapshot")
    latest_manifest = ensure_allowed_path(resolved_output_dir / LATEST_MANIFEST_NAME, reason="write latest manifest")

    resolved_retention = resolve_retention_count(retention_count)
    max_age_days = resolve_max_age_days()
    lock_fd: Optional[int] = None

    try:
        lock_fd = _acquire_lock(lock_path)

        if latest_manifest.exists() and not force:
            try:
                manifest = _read_json(latest_manifest)
                created = str(manifest.get("created_at_utc", ""))
                age_days = _snapshot_age_days(created)
                if age_days <= float(max_age_days):
                    return {
                        "status": "skipped",
                        "reason": "latest_snapshot_is_fresh",
                        "snapshot_age_days": round(age_days, 4),
                        "max_age_days": max_age_days,
                        "latest_snapshot_path": str(latest_parquet),
                    }
            except Exception:
                LOGGER.warning("Unable to evaluate existing latest manifest; continuing refresh")

        refresh_run_id = f"refresh_{uuid.uuid4().hex[:12]}"
        query = build_training_query(source_table, apply_training_filters=False)
        query_fingerprint = hash_payload({"query": query})

        df = _fetch_from_snowflake(source_table)
        created_at_utc = now_utc_iso()

        version_prefix = f"features_{_utc_stamp()}"
        versioned_parquet = ensure_allowed_path(
            resolved_output_dir / f"{version_prefix}.parquet",
            reason="write versioned snapshot",
        )
        versioned_manifest = ensure_allowed_path(
            resolved_output_dir / f"{version_prefix}.manifest.json",
            reason="write versioned manifest",
        )

        _write_parquet_atomic(df, versioned_parquet)
        manifest_payload = _build_manifest(
            df=df,
            source_table=source_table,
            query_fingerprint=query_fingerprint,
            parquet_path=versioned_parquet,
            refresh_run_id=refresh_run_id,
            created_at_utc=created_at_utc,
        )
        _write_json_atomic(versioned_manifest, manifest_payload)

        latest_parquet_tmp = ensure_allowed_path(
            resolved_output_dir / f"{LATEST_SNAPSHOT_NAME}.tmp",
            reason="prepare latest snapshot",
        )
        latest_manifest_tmp = ensure_allowed_path(
            resolved_output_dir / f"{LATEST_MANIFEST_NAME}.tmp",
            reason="prepare latest manifest",
        )
        shutil.copy2(versioned_parquet, latest_parquet_tmp)
        latest_manifest_payload = dict(manifest_payload)
        latest_manifest_payload.update(
            {
                "versioned_snapshot": versioned_parquet.name,
                "versioned_manifest": versioned_manifest.name,
            }
        )
        latest_manifest_tmp.write_text(
            json.dumps(latest_manifest_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _replace_latest_files(
            latest_parquet=latest_parquet,
            latest_manifest=latest_manifest,
            new_parquet_tmp=latest_parquet_tmp,
            new_manifest_tmp=latest_manifest_tmp,
        )

        preserve = {versioned_parquet.name}
        _prune_retention(
            resolved_output_dir,
            retention_count=resolved_retention,
            preserve=preserve,
        )

        summary = {
            "status": "refreshed",
            "refresh_run_id": refresh_run_id,
            "output_dir": str(resolved_output_dir),
            "versioned_snapshot": versioned_parquet.name,
            "versioned_manifest": versioned_manifest.name,
            "latest_snapshot": latest_parquet.name,
            "latest_manifest": latest_manifest.name,
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "manifest_hash": hash_payload(manifest_payload),
            "retention_count": resolved_retention,
        }
        return summary
    finally:
        _release_lock(lock_path, lock_fd)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    try:
        summary = refresh_snapshot(
            output_dir=args.output_dir,
            source_table=args.source_table,
            retention_count=args.retention_count,
            force=bool(args.force),
        )
    except Exception as exc:
        LOGGER.error("Snapshot refresh failed: %s", exc)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
