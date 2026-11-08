"""Read-only Memorax boundary and deterministic R03 dialogue input adapter.

This module intentionally contains no writable Memorax gateway. Search is the
only network operation; all Memory write-shaped entry points fail closed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import httpx


ALLOWED_MEMORAX_ENDPOINTS = frozenset({
    ("POST", "/v1/memory-buckets/search"),
})

ALL_BUCKET_TYPES = ("person", "location", "event", "item", "dreaming")


class MemoraxWriteDisabledError(RuntimeError):
    """Raised before any Memory write or post-generation visibility path."""


class MemoraxEndpointNotAllowedError(RuntimeError):
    """Raised before transport when an endpoint is outside the allowlist."""


class MemoraxReadContractError(RuntimeError):
    """Raised when a Search response is not a usable bucket envelope."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _disabled(operation: str) -> None:
    raise MemoraxWriteDisabledError(f"memorax_write_disabled:{operation}")


def build_add_intent(*args: Any, **kwargs: Any) -> None:
    _disabled("build_add_intent")


def poll_add_status(*args: Any, **kwargs: Any) -> None:
    _disabled("poll_add_status")


def create_child_add_payload(*args: Any, **kwargs: Any) -> None:
    _disabled("create_child_add_payload")


def write_memory_metadata(*args: Any, **kwargs: Any) -> None:
    _disabled("write_memory_metadata")


def post_generation_visibility_check(*args: Any, **kwargs: Any) -> None:
    _disabled("post_generation_visibility_check")


def assert_allowed_endpoint(method: str, path: str) -> None:
    key = (str(method).upper(), str(path))
    if key not in ALLOWED_MEMORAX_ENDPOINTS:
        raise MemoraxEndpointNotAllowedError(
            f"memorax_endpoint_not_allowed:{key[0]}:{key[1]}")


@dataclass(frozen=True)
class ReadOnlyMemoraxConfig:
    base_url: str
    api_key: str
    platform: str = "memory"
    connect_timeout_seconds: float = 15.0
    read_timeout_seconds: float = 180.0


class ReadOnlyMemoraxClient:
    """A Search-only client with an endpoint gate immediately before I/O."""

    def __init__(
        self,
        config: ReadOnlyMemoraxConfig,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not str(config.api_key).strip():
            raise ValueError("memorax_api_key_required")
        if str(config.platform) != "memory":
            raise ValueError("memorax_platform_must_be_memory")
        self.config = config
        self.transport = transport
        self.calls: list[dict[str, Any]] = []

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any],
        request_id: str,
    ) -> dict[str, Any]:
        assert_allowed_endpoint(method, path)
        record: dict[str, Any] = {
            "method": str(method).upper(),
            "path": str(path),
            "request_id": str(request_id),
            "request_attempted": True,
            "http_status": None,
            "authorization_header_recorded": False,
        }
        self.calls.append(record)
        timeout = httpx.Timeout(
            connect=float(self.config.connect_timeout_seconds),
            read=float(self.config.read_timeout_seconds),
            write=float(self.config.read_timeout_seconds),
            pool=float(self.config.connect_timeout_seconds),
        )
        headers = {
            "Authorization": f"Token {self.config.api_key}",
            "X-Platform": self.config.platform,
            "Content-Type": "application/json",
            "X-Request-ID": str(request_id),
        }
        try:
            with httpx.Client(
                base_url=str(self.config.base_url).rstrip("/"),
                headers=headers,
                timeout=timeout,
                transport=self.transport,
            ) as client:
                response = client.request(
                    str(method).upper(), str(path), json=dict(payload))
        except httpx.HTTPError as error:
            record["transport_error"] = type(error).__name__
            raise MemoraxReadContractError(
                f"memorax_search_transport_failed:{type(error).__name__}") from error
        record["http_status"] = int(response.status_code)
        try:
            value = response.json()
        except ValueError as error:
            raise MemoraxReadContractError(
                f"memorax_search_response_not_json:{response.status_code}") from error
        if response.status_code >= 400:
            detail = value.get("detail") if isinstance(value, Mapping) else None
            raise MemoraxReadContractError(
                f"memorax_search_http_error:{response.status_code}:"
                f"{detail or 'unknown'}")
        validate_search_response(value)
        return dict(value)

    def search_buckets(
        self, payload: Mapping[str, Any], *, request_id: str,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/memory-buckets/search",
            payload=payload,
            request_id=request_id,
        )

    def add(self, *args: Any, **kwargs: Any) -> None:
        _disabled("add")

    def add_status(self, *args: Any, **kwargs: Any) -> None:
        _disabled("add_status")

    def delete(self, *args: Any, **kwargs: Any) -> None:
        _disabled("delete")

    def update(self, *args: Any, **kwargs: Any) -> None:
        _disabled("update")

    def clone(self, *args: Any, **kwargs: Any) -> None:
        _disabled("clone")

    def export(self, *args: Any, **kwargs: Any) -> None:
        _disabled("export")

    def import_memory(self, *args: Any, **kwargs: Any) -> None:
        _disabled("import")


def validate_search_response(value: Any) -> None:
    if not isinstance(value, Mapping) or value.get("success") is not True:
        raise MemoraxReadContractError("memorax_search_success_envelope_invalid")
    data = value.get("data")
    buckets = data.get("buckets") if isinstance(data, Mapping) else None
    if not isinstance(buckets, Mapping):
        raise MemoraxReadContractError("memorax_search_bucket_map_required")


def raw_snapshot_from_response(
    response: Mapping[str, Any], *, root_id: str, user_id: str,
) -> dict[str, Any]:
    validate_search_response(response)
    return {
        "snapshot_version": "r03_r02_memory_snapshot_raw_v1",
        "root_id": str(root_id),
        "user_id": str(user_id),
        "buckets": response["data"]["buckets"],
    }


def _content(value: Any) -> Any:
    if isinstance(value, list):
        normalized = [_content(item) for item in value]
        return sorted(normalized, key=canonical_json)
    if isinstance(value, Mapping):
        # Item buckets use small attribute/value records. Retain their semantic
        # fields without copying request-time or backend-only metadata.
        kept = {
            str(key): _content(item)
            for key, item in value.items()
            if str(key) in {"attribute", "value", "text", "content"}
        }
        return kept if kept else canonical_json(value)
    return value


def normalize_bucket_map(
    buckets: Mapping[str, Any], *, root_id: str, user_id: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for bucket_type in ALL_BUCKET_TYPES:
        bucket = buckets.get(bucket_type) or {}
        if not isinstance(bucket, Mapping):
            raise MemoraxReadContractError(
                f"memorax_bucket_type_invalid:{bucket_type}")
        counts[bucket_type] = len(bucket)
        for key, raw in bucket.items():
            if not isinstance(raw, Mapping):
                raise MemoraxReadContractError(
                    f"memorax_bucket_row_invalid:{bucket_type}:{key}")
            ref = (
                raw.get("memory_ref")
                or raw.get("bucket_ref")
                or raw.get("ref")
            )
            rows.append({
                "bucket_type": bucket_type,
                "key": str(raw.get("canonical_key") or key),
                "display_key": str(raw.get("display_key") or key),
                "content": _content(raw.get("content")),
                "ref": str(ref) if ref is not None else None,
            })
    rows.sort(key=lambda row: (
        row["bucket_type"], row["key"].casefold(),
        canonical_json(row["content"]), str(row["ref"] or "")))
    result = {
        "snapshot_version": "r03_r02_memory_snapshot_normalized_v1",
        "root_id": str(root_id),
        "user_id": str(user_id),
        "normalization": [
            "stable_bucket_type_and_key_order",
            "stable_content_order",
            "request_timestamp_excluded",
            "backend_metadata_excluded",
            "provider_refs_retained_when_present",
            "source_turn_refs_not_inferred",
        ],
        "bucket_counts": counts,
        "buckets": rows,
    }
    result["snapshot_sha256"] = sha256_value(result)
    return result


def normalize_search_response(
    response: Mapping[str, Any], *, root_id: str, user_id: str,
) -> dict[str, Any]:
    validate_search_response(response)
    return normalize_bucket_map(
        response["data"]["buckets"], root_id=root_id, user_id=user_id)


def normalize_historical_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    buckets = snapshot.get("buckets")
    if not isinstance(buckets, Mapping):
        raise MemoraxReadContractError("historical_bucket_map_required")
    return normalize_bucket_map(
        buckets,
        root_id=str(snapshot.get("root_id") or ""),
        user_id=str(snapshot.get("user_id") or ""),
    )


def memory_prompt_projection(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    rows = snapshot.get("buckets")
    if not isinstance(rows, list):
        raise ValueError("normalized_memory_rows_required")
    projected = [{
        "bucket_type": row["bucket_type"],
        "key": row["key"],
        "content": row["content"],
    } for row in rows]
    result = {
        "projection_version": "r03_memory_navigation_projection_v1",
        "root_id": snapshot.get("root_id"),
        "rules": [
            "retain_bucket_type_key_content",
            "drop_request_metadata_timestamps_and_refs",
            "no_model_summary",
            "no_source_turn_ref_inference",
        ],
        "buckets": projected,
    }
    result["projection_sha256"] = sha256_value(result)
    return result


def dialogue_as_planner_parent(dialogue: Mapping[str, Any]) -> dict[str, Any]:
    turns = dialogue.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("complete_dialogue_turns_required")
    converted = [{
        "turn_id": str(row["turn_ref"]),
        "speaker": str(row["speaker"]),
        "text": str(row["text"]),
    } for row in turns]
    return {"dialogue": {"turns": converted}}


def render_planner_dry_run(
    *,
    dialogue: Mapping[str, Any],
    memory_projection: Mapping[str, Any],
    feedback_card: Mapping[str, Any],
) -> dict[str, Any]:
    """Render the exact input envelope frozen for the future one-root smoke."""
    from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.prompts import (
        build_planner_prompt,
    )

    parent = dialogue_as_planner_parent(dialogue)
    speakers = sorted({
        str(row["speaker"]) for row in dialogue["turns"]
        if str(row.get("speaker") or "")
    })
    registry = {"speakers": [{
        "speaker_ref": f"SP{index:02d}",
        "canonical_name": name,
    } for index, name in enumerate(speakers, 1)]}
    base = build_planner_prompt(
        parent=parent,
        memory_overview=memory_projection,
        candidates=[],
        verified_tracks=[],
        memory_only_fast=True,
        event_bucket_only=False,
        memory_source_mode="person_bucket",
        speaker_registry=registry,
        strict_speaker_refs=True,
        expose_complete_memory=True,
    )
    dialogue_text = "\n".join(
        f"{row['turn_ref']} | {row['speaker']} | {row['text']}"
        for row in dialogue["turns"]
    )
    wrapper = (
        "\n\nR03 READ-ONLY DIALOGUE ADAPTER CONTRACT\n"
        "The complete cumulative dialogue below is authoritative. Memory is "
        "navigation-only. If they conflict, follow dialogue and mark "
        "MEMORY_DIALOGUE_CONFLICT for the local continuation gate. Continue "
        "append-only; never rewrite an existing turn. The feedback card "
        "specifies the capability gap, not story facts.\n\n"
        "AUTHORITATIVE PARENT+R01+R02-FEEDBACK DIALOGUE\n"
        f"{dialogue_text}\n\n"
        "FROZEN EXPANSION FEEDBACK CARD\n"
        f"{json.dumps(feedback_card, ensure_ascii=False, sort_keys=True, indent=2)}"
    )
    prompt = str(base["prompt"]) + wrapper
    return {
        "prompt_version": "r03_existing_memory_readonly_planner_prompt_v2",
        "core_prompt_version": base["prompt_version"],
        "prompt": prompt,
        "base_prompt": str(base["prompt"]),
        "dialogue_text": dialogue_text,
        "memory_projection_text": canonical_json(memory_projection),
        "feedback_card_text": canonical_json(feedback_card),
        "speaker_registry": registry,
    }


def build_local_r03_turn_refs(
    *, root_id: str, scene_turn_counts: Sequence[int],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for scene_index, count in enumerate(scene_turn_counts, 1):
        for turn_index in range(1, int(count) + 1):
            rows.append({
                "scene_ref": f"R03:S{scene_index:02d}",
                "turn_ref": f"R03:S{scene_index:02d}:T{turn_index:03d}",
                "local_session_ref": (
                    f"{root_id}:r03:s{scene_index:02d}"),
            })
    return rows

