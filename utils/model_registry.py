"""Atomic model registry helpers for autonomous promotion and rollback."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
from datetime import datetime, UTC
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Optional
import uuid

import numpy as np
import pandas as pd


class RegistryError(RuntimeError):
    """Raised for model registry failures."""


class PromotionError(RegistryError):
    """Raised when promotion fails and rollback is required."""


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOGS_DIR = PIPELINE_ROOT / "logs"
DEFAULT_MODELS_DIR = DEFAULT_LOGS_DIR / "models"
DEFAULT_REGISTRY_PATH = DEFAULT_LOGS_DIR / "model_registry.json"
DEFAULT_ACTIVE_MODEL_PATH = DEFAULT_LOGS_DIR / "model.bin"


def _ensure_allowlisted(path: Path, logs_root: Path) -> Path:
    resolved = path.resolve()
    logs_resolved = logs_root.resolve()
    if resolved != logs_resolved and logs_resolved not in resolved.parents:
        raise RegistryError(f"Path '{resolved}' is outside allowlisted logs root '{logs_resolved}'")
    return resolved


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=True)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def _atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as src_handle, NamedTemporaryFile("wb", dir=dst.parent, delete=False) as tmp:
        shutil.copyfileobj(src_handle, tmp)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, dst)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_registry() -> dict[str, Any]:
    return {
        "active_version": None,
        "active_model_path": None,
        "entries": [],
    }


def _load_registry(registry_path: Path) -> dict[str, Any]:
    if not registry_path.exists():
        return _default_registry()
    with registry_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _to_relative(path: Path, logs_root: Path) -> str:
    return str(path.resolve().relative_to(logs_root.resolve()))


def _entry_for_version(registry: dict[str, Any], version: str) -> Optional[dict[str, Any]]:
    for entry in registry.get("entries", []):
        if entry.get("version") == version:
            return entry
    return None


def _update_entry_statuses(registry: dict[str, Any], new_active_version: str) -> None:
    for entry in registry.get("entries", []):
        if entry.get("version") == new_active_version:
            entry["status"] = "active"
        elif entry.get("status") == "active":
            entry["status"] = "superseded"


def get_active_model(logs_dir: Path | str = DEFAULT_LOGS_DIR) -> Optional[dict[str, Any]]:
    """Return metadata for the currently active model version."""

    logs_root = Path(logs_dir)
    registry_path = _ensure_allowlisted(logs_root / "model_registry.json", logs_root)
    registry = _load_registry(registry_path)

    active_version = registry.get("active_version")
    active_rel_path = registry.get("active_model_path")
    if not active_version or not active_rel_path:
        return None

    active_entry = _entry_for_version(registry, active_version)
    active_path = _ensure_allowlisted(logs_root / active_rel_path, logs_root)
    return {
        "version": active_version,
        "path": str(active_path),
        "checksum": active_entry.get("checksum") if active_entry else None,
        "entry": active_entry,
    }


def register_candidate(
    *,
    version: str,
    artifact_path: str | Path,
    checksum: str,
    metrics: Optional[dict[str, Any]] = None,
    run_id: Optional[str] = None,
    previous_model_version: Optional[str] = None,
    logs_dir: Path | str = DEFAULT_LOGS_DIR,
) -> dict[str, Any]:
    """Append a candidate model row to the registry."""

    logs_root = Path(logs_dir)
    registry_path = _ensure_allowlisted(logs_root / "model_registry.json", logs_root)
    resolved_artifact = _ensure_allowlisted(Path(artifact_path), logs_root)

    registry = _load_registry(registry_path)
    if _entry_for_version(registry, version):
        raise RegistryError(f"Candidate version '{version}' already exists")

    entry = {
        "version": version,
        "artifact_path": _to_relative(resolved_artifact, logs_root),
        "checksum": checksum,
        "metrics": metrics or {},
        "run_id": run_id,
        "status": "candidate",
        "previous_model_version": previous_model_version,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    registry.setdefault("entries", []).append(entry)
    _atomic_write_json(registry_path, registry)
    return entry


def verify_artifact(path: str | Path, checksum: str) -> bool:
    """Verify checksum, deserialisation, and predict capability."""

    artifact_path = Path(path)
    if not artifact_path.exists():
        return False
    if _sha256(artifact_path) != checksum:
        return False

    with artifact_path.open("rb") as handle:
        payload = pickle.load(handle)

    model = payload.get("model") if isinstance(payload, dict) else payload
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    if not isinstance(model, dict) and not hasattr(model, "predict"):
        return False

    feature_cols = metadata.get("feature_cols") or []
    width = max(len(feature_cols), 1)
    frame = pd.DataFrame(np.zeros((1, width)), columns=feature_cols or ["feature_0"])
    if isinstance(model, dict):
        import sys

        scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        import train_model

        _ = train_model.predict_prediction_bundle(
            model,
            frame,
            feature_cols or ["feature_0"],
            use_log_target=bool(metadata.get("use_log_target", False)),
        )
    else:
        _ = model.predict(frame)
    return True


def _switch_active_pointer(
    *,
    version: str,
    artifact_path: Path,
    logs_root: Path,
    previous_model_version: Optional[str],
) -> dict[str, Any]:
    registry_path = _ensure_allowlisted(logs_root / "model_registry.json", logs_root)
    registry = _load_registry(registry_path)

    if not _entry_for_version(registry, version):
        raise RegistryError(f"Cannot set active pointer: unknown version '{version}'")

    registry["active_version"] = version
    registry["active_model_path"] = _to_relative(artifact_path, logs_root)
    registry["previous_model_version"] = previous_model_version
    _update_entry_statuses(registry, version)

    _atomic_write_json(registry_path, registry)
    _atomic_copy(artifact_path, _ensure_allowlisted(logs_root / "model.bin", logs_root))

    active_entry = _entry_for_version(registry, version)
    return {
        "version": version,
        "path": str(artifact_path),
        "checksum": active_entry.get("checksum") if active_entry else None,
        "entry": active_entry,
        "previous_model_version": previous_model_version,
    }


def promote_atomically(
    *,
    model_payload: Any,
    candidate_metrics: Optional[dict[str, Any]] = None,
    run_id: Optional[str] = None,
    logs_dir: Path | str = DEFAULT_LOGS_DIR,
) -> dict[str, Any]:
    """Promote a model payload as active with atomic pointer switching."""

    logs_root = Path(logs_dir)
    models_dir = _ensure_allowlisted(logs_root / "models", logs_root)
    models_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(UTC)
    version = f"model_{now.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid.uuid4().hex[:8]}"
    artifact_path = _ensure_allowlisted(models_dir / f"{version}.bin", logs_root)
    checksum_path = _ensure_allowlisted(models_dir / f"{version}.sha256", logs_root)

    previous_active = get_active_model(logs_root)
    previous_model_version = previous_active["version"] if previous_active else None

    try:
        with NamedTemporaryFile("wb", dir=models_dir, delete=False) as tmp:
            pickle.dump(model_payload, tmp)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, artifact_path)

        checksum = _sha256(artifact_path)
        with NamedTemporaryFile("w", dir=models_dir, delete=False, encoding="utf-8") as tmp_checksum:
            tmp_checksum.write(checksum)
            tmp_checksum.flush()
            os.fsync(tmp_checksum.fileno())
            tmp_checksum_path = Path(tmp_checksum.name)
        os.replace(tmp_checksum_path, checksum_path)

        if not verify_artifact(artifact_path, checksum):
            raise PromotionError("Artifact verification failed")

        register_candidate(
            version=version,
            artifact_path=artifact_path,
            checksum=checksum,
            metrics=candidate_metrics,
            run_id=run_id,
            previous_model_version=previous_model_version,
            logs_dir=logs_root,
        )

        return _switch_active_pointer(
            version=version,
            artifact_path=artifact_path,
            logs_root=logs_root,
            previous_model_version=previous_model_version,
        )

    except Exception as exc:
        if previous_model_version:
            try:
                rollback_to(previous_model_version, logs_dir=logs_root)
            except Exception as rollback_exc:
                raise PromotionError(
                    f"Promotion failed and rollback failed: {exc}; rollback error: {rollback_exc}"
                ) from rollback_exc
        raise PromotionError(f"Promotion failed: {exc}") from exc


def rollback_to(version: str, logs_dir: Path | str = DEFAULT_LOGS_DIR) -> dict[str, Any]:
    """Rollback active pointer to a previous version."""

    logs_root = Path(logs_dir)
    registry_path = _ensure_allowlisted(logs_root / "model_registry.json", logs_root)
    registry = _load_registry(registry_path)
    entry = _entry_for_version(registry, version)
    if entry is None:
        raise RegistryError(f"Rollback target '{version}' not found")

    artifact_path = _ensure_allowlisted(logs_root / entry["artifact_path"], logs_root)
    checksum = entry.get("checksum")
    if not checksum:
        raise RegistryError(f"Rollback target '{version}' is missing checksum")
    if not verify_artifact(artifact_path, checksum):
        raise RegistryError(f"Rollback target '{version}' failed verification")

    registry["active_version"] = version
    registry["active_model_path"] = entry["artifact_path"]
    _update_entry_statuses(registry, version)
    _atomic_write_json(registry_path, registry)
    _atomic_copy(artifact_path, _ensure_allowlisted(logs_root / "model.bin", logs_root))

    return {
        "version": version,
        "path": str(artifact_path),
        "checksum": checksum,
        "entry": entry,
    }
