"""Normalize Memorax Buckets into a bounded, non-authoritative story hint."""

from __future__ import annotations

from typing import Any, Mapping

from ..core import canonical_json, sha256_bytes


MEMORY_OVERVIEW_VERSION = "simple_append_memorax_overview_v1"
BUCKET_SNAPSHOT_VERSION = "simple_append_memorax_bucket_snapshot_v1"
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
    }
    result["snapshot_sha256"] = sha256_bytes(
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
) -> dict[str, Any]:
    if snapshot.get("bucket_snapshot_version") != BUCKET_SNAPSHOT_VERSION:
        raise ValueError("memorax_bucket_snapshot_version_invalid")
    if int(content_limit) < 1:
        raise ValueError("memorax_content_limit_invalid")
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
            content = list(row.get("content") or [])[:int(content_limit)]
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
