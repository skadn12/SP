"""Normalize Memorax Buckets into a bounded, non-authoritative story hint."""

from __future__ import annotations

from typing import Any, Mapping

from ..core import canonical_json, sha256_bytes


MEMORY_OVERVIEW_VERSION = "simple_append_memorax_overview_v2_person_events"
BUCKET_SNAPSHOT_VERSION = "simple_append_memorax_bucket_snapshot_v1"
GLOBAL_MEMORY_STATE_VIEW_VERSION = "simple_append_memorax_global_state_view_v2_cited"
STRING_BUCKET_TYPES = ("person", "location", "event", "dreaming")
ALL_BUCKET_TYPES = (*STRING_BUCKET_TYPES, "item")
DEFAULT_BUCKET_LIMITS = {
    "person": 12,
    "location": 8,
    "event": 12,
    "item": 8,
    "dreaming": 8,
}
DEFAULT_CONTENT_LIMIT = 3
PERSON_EVENT_CONTENT_LIMIT = 8


def _nonempty(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _dedupe_strings(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _nonempty(value)
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _source_turn_refs(raw_row: Mapping[str, Any]) -> list[str]:
    """Preserve provider-supplied provenance without inferring it locally."""
    metadata = raw_row.get("metadata")
    candidates = [
        raw_row.get("source_turn_refs"),
        raw_row.get("turn_refs"),
        metadata.get("source_turn_refs") if isinstance(metadata, Mapping) else None,
        metadata.get("turn_refs") if isinstance(metadata, Mapping) else None,
    ]
    for value in candidates:
        refs = _dedupe_strings(value)
        if refs:
            return refs
    return []


def normalize_bucket_snapshot(
    response: Mapping[str, Any], *, root_id: str, user_id: str,
) -> dict[str, Any]:
    if response.get("success") is not True:
        raise ValueError("memorax_bucket_success_required")
    data = response.get("data")
    buckets = data.get("buckets") if isinstance(data, Mapping) else None
    if not isinstance(buckets, Mapping):
        raise ValueError("memorax_bucket_map_required")
    normalized: dict[str, dict[str, Any]] = {}
    for bucket_type in ALL_BUCKET_TYPES:
        source = buckets.get(bucket_type) or {}
        if not isinstance(source, Mapping):
            raise ValueError(f"memorax_bucket_type_invalid:{bucket_type}")
        rows: dict[str, Any] = {}
        for raw_key, raw_row in source.items():
            key = _nonempty(raw_key).casefold()
            if not key or not isinstance(raw_row, Mapping):
                raise ValueError(f"memorax_bucket_row_invalid:{bucket_type}")
            display = _nonempty(raw_row.get("display_key")) or str(raw_key)
            content = raw_row.get("content")
            if bucket_type == "item":
                if not isinstance(content, list):
                    raise ValueError("memorax_item_content_invalid")
                normalized_content: list[dict[str, str]] = []
                for item in content:
                    if not isinstance(item, Mapping):
                        raise ValueError("memorax_item_attribute_invalid")
                    attribute = _nonempty(item.get("attribute"))
                    value = _nonempty(item.get("value"))
                    if attribute and value:
                        normalized_content.append({
                            "attribute": attribute, "value": value})
            else:
                if not isinstance(content, list):
                    raise ValueError(
                        f"memorax_string_content_invalid:{bucket_type}")
                normalized_content = _dedupe_strings(content)
            rows[key] = {
                "canonical_key": key,
                "display_key": display,
                "content": normalized_content,
                # Empty means the backend did not return exact provenance;
                # callers must not reconstruct it from bucket text.
                "source_turn_refs": _source_turn_refs(raw_row),
            }
        normalized[bucket_type] = rows
    result = {
        "bucket_snapshot_version": BUCKET_SNAPSHOT_VERSION,
        "root_id": str(root_id),
        "user_id": str(user_id),
        "buckets": normalized,
        "bucket_counts": {key: len(value)
                          for key, value in normalized.items()},
        "return_raw_text": False,
        "memory_is_authoritative": False,
        "memory_is_qa_fact_source": False,
        "bucket_provenance_version": "provider_source_turn_refs_v1",
        "source_turn_refs_present": all(
            bool(row.get("source_turn_refs"))
            for bucket in normalized.values() for row in bucket.values()
        ) if any(normalized.values()) else False,
    }
    result["snapshot_sha256"] = sha256_bytes(
        canonical_json(result).encode("utf-8"))
    return result


def audit_bucket_delta_visibility(
    before: Mapping[str, Any], after: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two normalized snapshots for deterministic write visibility.

    A successful Child Add is not required to preserve the exact same bucket
    shape: Memorax may add a bucket or update an existing bucket's content.
    The pre/post snapshots are from the same namespace, so an added or changed
    bucket is sufficient evidence that the write became query-visible. This
    helper intentionally does not infer semantic correctness from bucket text.
    """
    errors: list[str] = []
    for label, snapshot in (("before", before), ("after", after)):
        if not isinstance(snapshot, Mapping):
            errors.append(f"{label}_snapshot_invalid")
            continue
        if snapshot.get("bucket_snapshot_version") != BUCKET_SNAPSHOT_VERSION:
            errors.append(f"{label}_snapshot_version_invalid")
        if not isinstance(snapshot.get("buckets"), Mapping):
            errors.append(f"{label}_buckets_invalid")
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        if str(before.get("root_id") or "") != str(after.get("root_id") or ""):
            errors.append("root_id_mismatch")
        if str(before.get("user_id") or "") != str(after.get("user_id") or ""):
            errors.append("user_id_mismatch")

    added: list[str] = []
    updated: list[str] = []
    removed: list[str] = []
    if not errors:
        before_buckets = before["buckets"]
        after_buckets = after["buckets"]
        for bucket_type in ALL_BUCKET_TYPES:
            before_rows = before_buckets.get(bucket_type) or {}
            after_rows = after_buckets.get(bucket_type) or {}
            if not isinstance(before_rows, Mapping):
                errors.append(f"before_bucket_type_invalid:{bucket_type}")
                continue
            if not isinstance(after_rows, Mapping):
                errors.append(f"after_bucket_type_invalid:{bucket_type}")
                continue
            for key in sorted(set(after_rows) - set(before_rows)):
                added.append(f"{bucket_type}:{key}")
            for key in sorted(set(before_rows) - set(after_rows)):
                removed.append(f"{bucket_type}:{key}")
            for key in sorted(set(before_rows) & set(after_rows)):
                if canonical_json(before_rows[key]) != canonical_json(after_rows[key]):
                    updated.append(f"{bucket_type}:{key}")

    return {
        "audit_version": "simple_append_memorax_bucket_delta_audit_v1",
        "valid": not errors,
        "errors": errors,
        "root_id": str(after.get("root_id") or "") if isinstance(after, Mapping) else "",
        "user_id": str(after.get("user_id") or "") if isinstance(after, Mapping) else "",
        "before_snapshot_sha256": str(before.get("snapshot_sha256") or "")
        if isinstance(before, Mapping) else "",
        "after_snapshot_sha256": str(after.get("snapshot_sha256") or "")
        if isinstance(after, Mapping) else "",
        "added_bucket_refs": added,
        "updated_bucket_refs": updated,
        "removed_bucket_refs": removed,
        "changed_bucket_refs": sorted(set(added + updated)),
        "added_bucket_count": len(added),
        "updated_bucket_count": len(updated),
        "removed_bucket_count": len(removed),
        "bucket_visibility_verified": bool(not errors and (added or updated)),
        "semantic_visibility_verified": False,
    }


def compile_global_memory_state_view(
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile the current namespace Memory into a bounded Gate view.

    This is derived from the current Bucket Search response for this root. It
    is not a local history database and never contains Parent transcript turns.
    All bucket contents are retained so a candidate can be compared with state
    stored under another event/person key, not only its selected event.
    """
    if snapshot.get("bucket_snapshot_version") != BUCKET_SNAPSHOT_VERSION:
        raise ValueError("memorax_bucket_snapshot_version_invalid")
    source = snapshot.get("buckets")
    if not isinstance(source, Mapping):
        raise ValueError("memorax_snapshot_buckets_required")
    bucket_codes = {
        "person": "P", "location": "L", "event": "E",
        "item": "I", "dreaming": "D",
    }
    buckets: dict[str, list[dict[str, Any]]] = {}
    for bucket_type in ALL_BUCKET_TYPES:
        rows = source.get(bucket_type) or {}
        if not isinstance(rows, Mapping):
            raise ValueError(f"memorax_snapshot_type_invalid:{bucket_type}")
        materialized: list[dict[str, Any]] = []
        for row_index, key in enumerate(
                sorted(rows, key=lambda value: str(value).casefold()), 1):
            row = rows[key]
            if not isinstance(row, Mapping):
                continue
            content = row.get("content") or []
            bucket_ref = f"GM:{bucket_codes[bucket_type]}{row_index:03d}"
            if bucket_type == "item":
                texts = [
                    f"{str(item.get('attribute') or '').strip()}: "
                    f"{str(item.get('value') or '').strip()}"
                    for item in content if isinstance(item, Mapping)
                    and str(item.get("attribute") or "").strip()
                    and str(item.get("value") or "").strip()
                ]
            else:
                texts = [str(item) for item in content if str(item).strip()]
            memory_facts = [{
                "memory_ref": f"{bucket_ref}:C{content_index:03d}",
                "text": text,
            } for content_index, text in enumerate(texts, 1)]
            materialized.append({
                "memory_bucket_ref": bucket_ref,
                "memory_key": str(row.get("canonical_key") or key),
                "display_key": str(row.get("display_key") or key),
                "memory_facts": memory_facts,
            })
        buckets[bucket_type] = materialized
    result = {
        "global_memory_state_view_version": GLOBAL_MEMORY_STATE_VIEW_VERSION,
        "root_id": str(snapshot.get("root_id") or ""),
        "source_snapshot_sha256": str(snapshot.get("snapshot_sha256") or ""),
        "memory_is_authoritative": False,
        "source_scope": "current_memorax_bucket_search_same_namespace",
        "buckets": buckets,
        "bucket_counts": {key: len(value) for key, value in buckets.items()},
        "allowed_use": "candidate_duplicate_and_current_state_check",
        "forbidden_use": ["qa_gold", "qa_evidence", "parent_transcript_replacement"],
    }
    payload = canonical_json(result)
    result["character_count"] = len(payload)
    result["token_estimate"] = max(1, (len(payload) + 3) // 4)
    result["view_sha256"] = sha256_bytes(
        canonical_json(result).encode("utf-8"))
    return result


def _relevance_key(row: Mapping[str, Any], recent: str) -> tuple[int, str]:
    candidates = [str(row.get("canonical_key") or ""),
                  str(row.get("display_key") or "")]
    overlap = any(value and value.casefold() in recent for value in candidates)
    return (0 if overlap else 1, str(row.get("canonical_key") or ""))


def compile_memory_overview(
    snapshot: Mapping[str, Any], *, recent_dialogue_text: str = "",
    bucket_limits: Mapping[str, int] | None = None,
    content_limit: int = DEFAULT_CONTENT_LIMIT,
    person_event_content_limit: int = PERSON_EVENT_CONTENT_LIMIT,
) -> dict[str, Any]:
    if snapshot.get("bucket_snapshot_version") != BUCKET_SNAPSHOT_VERSION:
        raise ValueError("memorax_bucket_snapshot_version_invalid")
    if int(content_limit) < 1:
        raise ValueError("memorax_content_limit_invalid")
    if int(person_event_content_limit) < 1:
        raise ValueError("memorax_person_event_content_limit_invalid")
    limits = {**DEFAULT_BUCKET_LIMITS, **dict(bucket_limits or {})}
    if any(key not in ALL_BUCKET_TYPES or int(value) < 0
           for key, value in limits.items()):
        raise ValueError("memorax_bucket_limits_invalid")
    recent = _nonempty(recent_dialogue_text).casefold()
    source = snapshot.get("buckets")
    if not isinstance(source, Mapping):
        raise ValueError("memorax_snapshot_buckets_required")
    selected: dict[str, list[dict[str, Any]]] = {}
    for bucket_type in ALL_BUCKET_TYPES:
        rows = source.get(bucket_type) or {}
        if not isinstance(rows, Mapping):
            raise ValueError(f"memorax_snapshot_type_invalid:{bucket_type}")
        ordered = sorted(rows.values(), key=lambda row: _relevance_key(row, recent))
        compact: list[dict[str, Any]] = []
        for row in ordered[:int(limits[bucket_type])]:
            # Person buckets carry the actor-specific ``[事件 @...]`` facts
            # that event buckets often collapse into ``user``. Keep a larger
            # bounded slice for those facts; other bucket types retain the
            # original compact limit.
            content_cap = (int(person_event_content_limit)
                           if bucket_type == "person"
                           else int(content_limit))
            content = list(row.get("content") or [])[:content_cap]
            compact.append({
                "canonical_key": row["canonical_key"],
                "display_key": row["display_key"],
                "content": content,
            })
        selected[bucket_type] = compact
    result = {
        "memory_overview_version": MEMORY_OVERVIEW_VERSION,
        "root_id": str(snapshot.get("root_id") or ""),
        "source_snapshot_sha256": str(snapshot.get("snapshot_sha256") or ""),
        "selection_rule": "recent_key_overlap_then_canonical_v1",
        "people": selected["person"],
        "locations": selected["location"],
        "events": selected["event"],
        "items": selected["item"],
        "themes": selected["dreaming"],
        "memory_is_authoritative": False,
        "memory_is_qa_fact_source": False,
        "allowed_uses": [
            "story_recall_hint",
            "expansion_topic_selection",
            "qa_topic_selection",
        ],
        "forbidden_uses": [
            "qa_gold",
            "qa_evidence",
            "authoritative_turn_replacement",
        ],
    }
    payload = canonical_json(result)
    result["character_count"] = len(payload)
    result["token_estimate"] = max(1, (len(payload) + 3) // 4)
    result["memory_overview_sha256"] = sha256_bytes(
        canonical_json(result).encode("utf-8"))
    return result
