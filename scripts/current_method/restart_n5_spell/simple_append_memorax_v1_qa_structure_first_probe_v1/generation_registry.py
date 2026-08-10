"""Project-scoped idempotency registry for paid dialogue generation."""

from __future__ import annotations

import fcntl
import hashlib
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from ..core import atomic_json, canonical_json, sha256_bytes, sha256_file


REGISTRY_VERSION = "simple_append_generation_registry_v1"
DEFAULT_GENERATION_REGISTRY = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-state/"
    "simple_append_memorax_v1/generation_registry.json")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class GenerationAlreadyRegistered(RuntimeError):
    def __init__(self, conflicts: Sequence[Mapping[str, Any]]) -> None:
        self.conflicts = [dict(row) for row in conflicts]
        super().__init__("generation_identity_already_registered")


@contextmanager
def _locked(registry_path: Path) -> Iterator[None]:
    lock_path = registry_path.with_name(registry_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read(registry_path: Path) -> dict[str, Any]:
    if not registry_path.exists():
        return {"registry_version": REGISTRY_VERSION, "entries": {}}
    value = json.loads(registry_path.read_text(encoding="utf-8"))
    if (not isinstance(value, Mapping) or
            value.get("registry_version") != REGISTRY_VERSION or
            not isinstance(value.get("entries"), Mapping)):
        raise ValueError("generation_registry_contract_invalid")
    return {"registry_version": REGISTRY_VERSION,
            "entries": dict(value["entries"])}


def files_manifest_sha(paths: Sequence[Path]) -> str:
    rows = [{"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in sorted(paths)]
    return sha256_bytes(canonical_json(rows).encode("utf-8"))


def memorax_namespace_sha(credentials_path: Path) -> str:
    """Fingerprint the configured project without persisting its API key."""
    values: dict[str, str] = {}
    for raw in credentials_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    secret = values.get("MEMORAX_API_KEY", "")
    if not secret:
        raise ValueError("generation_registry_memorax_namespace_missing")
    return hashlib.sha256(
        ("simple-append-project-namespace-v1\0" + secret).encode("utf-8")
    ).hexdigest()


def build_generation_identity(
    *, root_id: str, parent_sha256: str, round_index: int,
    generation_implementation_sha256: str, api_config_sha256: str,
    generation_configuration: Mapping[str, Any],
    memorax_namespace_sha256: str,
) -> dict[str, Any]:
    config_sha = sha256_bytes(
        canonical_json(dict(generation_configuration)).encode("utf-8"))
    slot_sha = sha256_bytes(canonical_json({
        "root_id": str(root_id),
        "parent_dialogue_sha256": str(parent_sha256),
        "round_index": int(round_index),
        "memorax_namespace_sha256": str(memorax_namespace_sha256),
    }).encode("utf-8"))
    value = {
        "identity_version": "simple_append_generation_identity_v1",
        "root_id": str(root_id),
        "parent_dialogue_sha256": str(parent_sha256),
        "round_index": int(round_index),
        "generation_implementation_sha256": str(
            generation_implementation_sha256),
        "generation_configuration_sha256": config_sha,
        "api_config_sha256": str(api_config_sha256),
        "memorax_namespace_sha256": str(memorax_namespace_sha256),
        "generation_slot_sha256": slot_sha,
    }
    value["generation_identity_sha256"] = sha256_bytes(
        canonical_json(value).encode("utf-8"))
    return value


def reserve_generation(
    *, registry_path: Path, identities: Mapping[str, Mapping[str, Any]],
    run_root: Path,
) -> dict[str, Any]:
    """Reserve all roots atomically, before a Planner request can be sent."""
    with _locked(registry_path):
        registry = _read(registry_path)
        entries = dict(registry["entries"])
        conflicts = []
        for root_id, identity in identities.items():
            key = str(identity["generation_identity_sha256"])
            slot = str(identity["generation_slot_sha256"])
            for existing_key, existing in entries.items():
                existing_identity = existing.get("identity") or {}
                existing_slot = str(existing_identity.get(
                    "generation_slot_sha256") or sha256_bytes(
                        canonical_json({
                            "root_id": existing_identity.get("root_id"),
                            "parent_dialogue_sha256": existing_identity.get(
                                "parent_dialogue_sha256"),
                            "round_index": existing_identity.get("round_index"),
                            "memorax_namespace_sha256": existing_identity.get(
                                "memorax_namespace_sha256"),
                        }).encode("utf-8")))
                if existing_key == key or existing_slot == slot:
                    conflicts.append({
                        "root_id": root_id,
                        "generation_identity_sha256": key,
                        "generation_slot_sha256": slot,
                        **dict(existing),
                    })
                    break
        if conflicts:
            raise GenerationAlreadyRegistered(conflicts)
        now = _now()
        for root_id, identity in identities.items():
            key = str(identity["generation_identity_sha256"])
            entries[key] = {
                "root_id": str(root_id),
                "state": "reserved_pre_generation",
                "run_root": str(run_root.resolve()),
                "identity": dict(identity),
                "reserved_at": now,
                "automatic_regeneration_allowed": False,
            }
        atomic_json(registry_path, {
            "registry_version": REGISTRY_VERSION,
            "updated_at": now,
            "entries": entries,
        })
    return {
        "registry_path": str(registry_path.resolve()),
        "reserved_identity_count": len(identities),
        "generation_identity_sha256": {
            root_id: value["generation_identity_sha256"]
            for root_id, value in identities.items()},
    }


def finalize_generation(
    *, registry_path: Path, identities: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    by_root = {str(row.get("root_id")): row for row in rows}
    with _locked(registry_path):
        registry = _read(registry_path)
        entries = dict(registry["entries"])
        for root_id, identity in identities.items():
            key = str(identity["generation_identity_sha256"])
            entry = dict(entries.get(key) or {})
            row = dict(by_root.get(str(root_id)) or {})
            expansion = row.get("expansion") or {}
            generation_ready = bool(expansion.get("generation_ready"))
            memory_pending = bool(expansion.get("pending_memory_writes"))
            if generation_ready and memory_pending:
                state = "generation_complete_memory_pending"
            elif generation_ready:
                state = "generation_complete"
            else:
                state = "generation_failed_or_ambiguous"
            entry.update({
                "state": state,
                "batch_row_status": row.get("status"),
                "expansion_run_root": expansion.get("run_root"),
                "qa_run_root": (row.get("qa") or {}).get("run_root"),
                "memory_write_verified": bool(
                    expansion.get("memory_write_verified")),
                "memory_recovery_required": memory_pending,
                "finalized_at": _now(),
                "automatic_regeneration_allowed": False,
            })
            entries[key] = entry
        atomic_json(registry_path, {
            "registry_version": REGISTRY_VERSION,
            "updated_at": _now(),
            "entries": entries,
        })


def mark_memory_status(
    *, registry_path: Path,
    status_by_root: Mapping[str, Mapping[str, str]],
) -> None:
    """Record read-only provider task confirmation without submitting Add."""
    with _locked(registry_path):
        registry = _read(registry_path)
        entries = dict(registry["entries"])
        for key, raw_entry in entries.items():
            entry = dict(raw_entry)
            root_id = str(entry.get("root_id") or "")
            statuses = status_by_root.get(root_id)
            if statuses is None:
                continue
            values = [str(value) for value in statuses.values()]
            if values and all(value in {"success", "completed"} for value in values):
                entry["state"] = "generation_complete_memory_verified"
                entry["memory_write_verified"] = True
                entry["memory_recovery_required"] = False
                entry["memory_status_confirmed_at"] = _now()
            else:
                entry["memory_status_last_observed"] = dict(statuses)
            entries[key] = entry
        atomic_json(registry_path, {
            "registry_version": REGISTRY_VERSION,
            "updated_at": _now(),
            "entries": entries,
        })
