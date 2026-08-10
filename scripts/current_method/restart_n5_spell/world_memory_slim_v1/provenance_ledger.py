"""Provenance Ledger v1: generated facts cannot be promoted into source truth."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence


PROVENANCE_LEDGER_VERSION = "provenance_ledger_v1"
ORIGINS = frozenset({"source_grounded", "generated_extension"})
VERIFICATION_STATUSES = frozenset({
    "source_grounded", "model_claimed", "semantically_verified", "rejected",
    "uncertain", "dependency_blocked", "atomic_blocked", "scene_blocked",
})
NON_AUTHORITATIVE_STATUSES = frozenset(
    VERIFICATION_STATUSES - {"source_grounded", "semantically_verified"})


def _record_key(row: Mapping[str, Any]) -> str:
    return str(row.get("authoritative_memory_ref") or row.get("record_ref") or "")


def initialize_provenance_ledger(
    *, root_source_id: str, original_source_records: Sequence[Mapping[str, Any]],
    previous_authoritative_memory: Sequence[Mapping[str, Any]] = (),
    previous_provenance_ledger: Sequence[Mapping[str, Any]] = (),
    previous_pending_updates: Sequence[Mapping[str, Any]] = (),
    previous_rejected_audit_log: Sequence[Mapping[str, Any]] = (),
    promoted_child_dialogue: Mapping[str, Any] | None = None,
    promotion_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Initialize physically separated provenance without re-extracting child text."""
    del promoted_child_dialogue  # surface continuity only; never a fact extractor
    promoted_refs = set((promotion_manifest or {}).get("committed_memory_refs") or [])
    ledger: list[dict[str, Any]] = [deepcopy(dict(row)) for row in previous_provenance_ledger]
    by_key = {_record_key(row): row for row in ledger if _record_key(row)}

    authoritative: list[dict[str, Any]] = []
    for record in previous_authoritative_memory:
        row = deepcopy(dict(record))
        ref = str(row.get("memory_ref") or row.get("memory_id") or "")
        prior = by_key.get(ref)
        if prior:
            origin = str(prior.get("origin"))
            verification = str(prior.get("verification_status"))
        else:
            origin = "generated_extension"
            verification = "semantically_verified"
        if origin not in ORIGINS or verification not in VERIFICATION_STATUSES:
            raise ValueError(f"invalid_previous_provenance:{ref}")
        row["origin"] = origin
        row["verification_status"] = verification
        authoritative.append(row)
        if ref and ref not in by_key:
            entry = ledger_entry(
                record_ref=ref, root_source_id=root_source_id, origin=origin,
                verification_status=verification,
                introduced_round=int(row.get("introduced_round", 0)),
                introduced_scene_ref=row.get("introduced_scene_ref"),
                proposal_ref=row.get("proposal_ref"),
                support_turn_refs=row.get("support_turn_refs") or [],
                authoritative_memory_ref=ref,
                parent_record_refs=row.get("parent_record_refs") or [],
            )
            ledger.append(entry)
            by_key[ref] = entry

    for index, record in enumerate(original_source_records, 1):
        ref = str(record.get("record_ref") or f"SOURCE:{index:06d}")
        if ref in by_key:
            continue
        entry = ledger_entry(
            record_ref=ref, root_source_id=root_source_id,
            origin="source_grounded", verification_status="source_grounded",
            introduced_round=0, introduced_scene_ref=None, proposal_ref=None,
            support_turn_refs=record.get("support_turn_refs") or [],
            authoritative_memory_ref=record.get("authoritative_memory_ref"),
            parent_record_refs=[],
        )
        ledger.append(entry)
        by_key[ref] = entry

    # Manifest refs link existing records. They never create or relabel facts.
    missing_promoted_refs = sorted(ref for ref in promoted_refs if ref not in by_key)
    return {
        "provenance_ledger_version": PROVENANCE_LEDGER_VERSION,
        "root_source_id": root_source_id,
        "authoritative_memory": authoritative,
        "pending_updates": [deepcopy(dict(row)) for row in previous_pending_updates],
        "rejected_audit_log": [deepcopy(dict(row)) for row in previous_rejected_audit_log],
        "provenance_ledger": ledger,
        "promoted_linked_memory_refs": sorted(promoted_refs - set(missing_promoted_refs)),
        "missing_promoted_memory_refs": missing_promoted_refs,
    }


def ledger_entry(
    *, record_ref: str, root_source_id: str, origin: str,
    verification_status: str, introduced_round: int,
    introduced_scene_ref: str | None, proposal_ref: str | None,
    support_turn_refs: Iterable[str], authoritative_memory_ref: str | None,
    parent_record_refs: Iterable[str],
) -> dict[str, Any]:
    if origin not in ORIGINS:
        raise ValueError("provenance_origin_invalid")
    if verification_status not in VERIFICATION_STATUSES:
        raise ValueError("provenance_verification_status_invalid")
    if origin == "source_grounded" and verification_status != "source_grounded":
        raise ValueError("source_origin_status_invalid")
    if origin == "generated_extension" and verification_status == "source_grounded":
        raise ValueError("generated_origin_cannot_be_source_grounded")
    return {
        "record_ref": record_ref,
        "root_source_id": root_source_id,
        "origin": origin,
        "verification_status": verification_status,
        "introduced_round": introduced_round,
        "introduced_scene_ref": introduced_scene_ref,
        "proposal_ref": proposal_ref,
        "support_turn_refs": list(support_turn_refs),
        "authoritative_memory_ref": authoritative_memory_ref,
        "parent_record_refs": list(parent_record_refs),
    }

