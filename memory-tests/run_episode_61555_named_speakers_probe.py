#!/usr/bin/env python3
"""Live Memorax probe for named speakers in event-bucket extraction."""

from __future__ import annotations

import copy
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


WORKSPACE = Path("/root/autodl-tmp/gxy-selfplay")
PROJECT_ROOT = WORKSPACE / "SP-820-publish"
SOURCE_PATH = (
    WORKSPACE
    / "restart-n5-inputs/conversation_chronicles_v1/roots"
    / "cc-train-episode-61555/unified_root.json"
)
RUN_ROOT = (
    WORKSPACE
    / "restart-n5-runs"
    / "conversation_chronicles_episode_61555_named_speakers_memory_test_v1"
)
CREDENTIALS_PATH = Path("/root/.config/memorax/credentials.env")
SOURCE_ROOT_ID = "cc-train-episode-61555"
ROOT_ID = "cc-train-episode-61555-named-speakers-v1"
USER_NAMESPACE = "memory_speaker_ref_named_v1"
SPEAKER_MAP = {
    "Co-workers A": "Alice",
    "Co-worker A": "Alice",
    "Co-workers B": "Bob",
    "Co-worker B": "Bob",
}
POST_SEARCH_DELAYS_SECONDS = (0, 2, 4, 8, 15, 20, 20, 20, 20, 20)

sys.path.insert(0, str(PROJECT_ROOT))

from scripts.current_method.restart_n5_spell.core import (  # noqa: E402
    atomic_json,
    canonical_json,
    sha256_bytes,
    sha256_file,
)
from scripts.current_method.restart_n5_spell.input_adapter import (  # noqa: E402
    dialogue_content_hash,
)
from scripts.current_method.restart_n5_spell.unified_dialogue_root import (  # noqa: E402
    validate_unified_dialogue_root,
)
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.async_write import (  # noqa: E402
    submit_add_and_wait,
)
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.client import (  # noqa: E402
    MemoraxClient,
    load_memorax_config,
)
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.memory_context import (  # noqa: E402
    normalize_bucket_snapshot,
)
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.payloads import (  # noqa: E402
    build_add_intent,
    build_bucket_search_payload,
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def replace_string(value: str) -> str:
    result = value.replace(SOURCE_ROOT_ID, ROOT_ID)
    for source, target in SPEAKER_MAP.items():
        result = result.replace(source, target)
    return result


def replace_tree(value: Any) -> Any:
    if isinstance(value, str):
        return replace_string(value)
    if isinstance(value, list):
        return [replace_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: replace_tree(item) for key, item in value.items()}
    return value


def clean_root(source: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = replace_tree(copy.deepcopy(dict(source)))
    cleaned["root_id"] = ROOT_ID
    cleaned["sample_id"] = ROOT_ID
    cleaned["lineage_group"] = ROOT_ID
    cleaned["dialogue"]["dialogue_hash"] = dialogue_content_hash(
        cleaned["dialogue"]["turns"]
    )
    return cleaned


def flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        result: list[str] = []
        for key, item in value.items():
            result.extend(flatten_strings(str(key)))
            result.extend(flatten_strings(item))
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(flatten_strings(item))
        return result
    return []


def bucket_rows(snapshot: Mapping[str, Any], bucket_type: str) -> list[dict[str, Any]]:
    rows = (snapshot.get("buckets") or {}).get(bucket_type) or {}
    return [dict(row) for row in rows.values() if isinstance(row, Mapping)]


def semantic_audit(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    placeholder_pattern = re.compile(r"\bco-workers?\s+[ab]\b", re.IGNORECASE)
    person_rows = bucket_rows(snapshot, "person")
    event_rows = bucket_rows(snapshot, "event")
    person_text = "\n".join(flatten_strings(person_rows))
    event_text = "\n".join(flatten_strings(event_rows))
    all_text = "\n".join(flatten_strings(snapshot.get("buckets") or {}))

    target_events = []
    for row in event_rows:
        row_text = "\n".join(flatten_strings(row))
        lowered = row_text.casefold()
        if "dad" in lowered and ("kiss" in lowered or "another woman" in lowered):
            target_events.append(row)
    target_event_text = "\n".join(flatten_strings(target_events))

    checks = {
        "person_bucket_has_alice": "alice" in person_text.casefold(),
        "person_bucket_has_bob": "bob" in person_text.casefold(),
        "event_bucket_has_alice": "alice" in event_text.casefold(),
        "event_bucket_has_bob": "bob" in event_text.casefold(),
        "target_dad_event_found": bool(target_events),
        "target_dad_event_names_alice": "alice" in target_event_text.casefold(),
        "target_dad_event_does_not_use_user": (
            bool(target_events)
            and re.search(r"\buser\b", target_event_text, re.IGNORECASE) is None
        ),
        "event_buckets_do_not_use_user": (
            bool(event_rows)
            and re.search(r"\buser\b", event_text, re.IGNORECASE) is None
        ),
        "no_a_b_placeholders_in_any_bucket": placeholder_pattern.search(all_text) is None,
    }
    return {
        "audit_version": "named_speaker_event_bucket_semantic_audit_v1",
        "root_id": ROOT_ID,
        "user_id": snapshot.get("user_id"),
        "checks": checks,
        "passed": all(checks.values()),
        "person_bucket_count": len(person_rows),
        "event_bucket_count": len(event_rows),
        "target_event_count": len(target_events),
        "target_events": target_events,
        "event_rows": event_rows,
    }


def main() -> int:
    if RUN_ROOT.exists():
        raise RuntimeError(f"run_root_already_exists:{RUN_ROOT}")
    RUN_ROOT.mkdir(parents=True)
    (RUN_ROOT / "memory").mkdir()
    started_at = now_utc()

    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    cleaned = clean_root(source)
    source_file_sha = sha256_file(Path(source["source"]["source_path"]))
    validation = validate_unified_dialogue_root(
        cleaned,
        expected_source_path=Path(source["source"]["source_path"]),
        expected_source_file_sha256=source_file_sha,
        expected_session_count=5,
    )
    serialized_cleaned = canonical_json(cleaned)
    aliases_remaining = sorted(
        set(re.findall(r"\bCo-workers?\s+[AB]\b", serialized_cleaned, re.IGNORECASE))
    )
    speakers = sorted({turn["speaker"] for turn in cleaned["dialogue"]["turns"]})
    cleaning_checks = {
        "unified_root_valid": validation["valid"],
        "root_id_changed": ROOT_ID != SOURCE_ROOT_ID,
        "root_identity_rebound": (
            cleaned["root_id"] == cleaned["sample_id"] == cleaned["lineage_group"] == ROOT_ID
        ),
        "dialogue_hash_changed": (
            cleaned["dialogue"]["dialogue_hash"] != source["dialogue"]["dialogue_hash"]
        ),
        "speaker_set_exact": speakers == ["Alice", "Bob"],
        "a_b_placeholders_removed": not aliases_remaining,
        "same_source_record": cleaned["source"] == source["source"],
        "same_turn_count": (
            cleaned["dialogue"]["turn_count"] == source["dialogue"]["turn_count"] == 60
        ),
    }
    cleaning_manifest = {
        "cleaning_version": "episode_61555_named_speakers_v1",
        "created_at": now_utc(),
        "source_path": str(SOURCE_PATH),
        "source_root_id": SOURCE_ROOT_ID,
        "root_id": ROOT_ID,
        "user_namespace": USER_NAMESPACE,
        "speaker_map": SPEAKER_MAP,
        "source_unified_root_sha256": sha256_file(SOURCE_PATH),
        "cleaned_dialogue_sha256": cleaned["dialogue"]["dialogue_hash"],
        "cleaning_checks": cleaning_checks,
        "validation": validation,
        "aliases_remaining": aliases_remaining,
    }
    atomic_json(RUN_ROOT / "cleaned_unified_root.json", cleaned)
    atomic_json(RUN_ROOT / "cleaning_manifest.json", cleaning_manifest)
    if not all(cleaning_checks.values()):
        raise RuntimeError("cleaning_validation_failed")

    config = load_memorax_config(
        credentials_path=CREDENTIALS_PATH,
        allow_insecure_http=True,
    )
    client = MemoraxClient(config)
    bucket_query = build_bucket_search_payload(
        root_id=ROOT_ID,
        user_namespace=USER_NAMESPACE,
    )
    atomic_json(RUN_ROOT / "memory/bucket_query.json", bucket_query)

    before_response = client.search_buckets(
        bucket_query["payload"], request_id=f"named-speaker-before-{ROOT_ID}"
    )
    before_snapshot = normalize_bucket_snapshot(
        before_response,
        root_id=ROOT_ID,
        user_id=bucket_query["payload"]["user_id"],
    )
    atomic_json(RUN_ROOT / "memory/before_bucket_response.json", before_response)
    atomic_json(RUN_ROOT / "memory/before_bucket_snapshot.json", before_snapshot)
    if any(before_snapshot["bucket_counts"].values()):
        raise RuntimeError("fresh_named_speaker_namespace_not_empty")

    intent = build_add_intent(
        root_id=ROOT_ID,
        round_index=0,
        source_dialogue_sha256=cleaned["dialogue"]["dialogue_hash"],
        turns=cleaned["dialogue"]["turns"],
        async_mode=True,
        user_namespace=USER_NAMESPACE,
    )
    atomic_json(RUN_ROOT / "memory/add_intent.json", intent)

    def persist_state(state: Mapping[str, Any]) -> None:
        atomic_json(RUN_ROOT / "memory/add_write_state.json", dict(state))

    completion = submit_add_and_wait(
        gateway=client,
        intent=intent,
        stage="named_speaker_parent_memory_add",
        on_state_change=persist_state,
    )
    atomic_json(RUN_ROOT / "memory/add_completion.json", completion)

    final_snapshot: dict[str, Any] | None = None
    final_audit: dict[str, Any] | None = None
    search_attempts = []
    for index, delay in enumerate(POST_SEARCH_DELAYS_SECONDS, 1):
        if delay:
            time.sleep(delay)
        request_id = f"named-speaker-after-{ROOT_ID}-{index:02d}"
        response = client.search_buckets(bucket_query["payload"], request_id=request_id)
        snapshot = normalize_bucket_snapshot(
            response,
            root_id=ROOT_ID,
            user_id=bucket_query["payload"]["user_id"],
        )
        audit = semantic_audit(snapshot)
        atomic_json(
            RUN_ROOT / f"memory/after_bucket_response_{index:02d}.json", response
        )
        atomic_json(
            RUN_ROOT / f"memory/after_bucket_snapshot_{index:02d}.json", snapshot
        )
        search_attempts.append({
            "attempt": index,
            "delay_seconds": delay,
            "request_id": request_id,
            "bucket_counts": snapshot["bucket_counts"],
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "semantic_passed": audit["passed"],
        })
        final_snapshot = snapshot
        final_audit = audit
        if audit["passed"]:
            break
        # Once all expected bucket types are materialized, further indexing
        # waits will not repair a semantic speaker-binding error.
        if (
            snapshot["bucket_counts"]["person"] >= 2
            and snapshot["bucket_counts"]["event"] >= 2
            and index >= 2
        ):
            break

    assert final_snapshot is not None and final_audit is not None
    atomic_json(RUN_ROOT / "memory/final_bucket_snapshot.json", final_snapshot)
    atomic_json(RUN_ROOT / "memory/semantic_audit.json", final_audit)
    summary = {
        "test_version": "episode_61555_named_speakers_memorax_live_v1",
        "status": "passed" if final_audit["passed"] else "failed",
        "started_at": started_at,
        "completed_at": now_utc(),
        "source_path": str(SOURCE_PATH),
        "source_root_id": SOURCE_ROOT_ID,
        "root_id": ROOT_ID,
        "user_namespace": USER_NAMESPACE,
        "user_id": bucket_query["payload"]["user_id"],
        "speaker_map": SPEAKER_MAP,
        "cleaning_checks": cleaning_checks,
        "transport_role_policy": intent["payload"]["metadata"]["memorax_role_policy"],
        "all_message_roles": sorted({row["role"] for row in intent["payload"]["messages"]}),
        "add_verified_complete": completion["verified_complete"],
        "add_task_id": completion.get("task_id"),
        "search_attempts": search_attempts,
        "final_bucket_counts": final_snapshot["bucket_counts"],
        "semantic_audit": final_audit,
        "memorax_configuration": config.redacted(),
        "memorax_calls": client.calls,
        "credentials_in_artifacts": False,
    }
    summary["summary_sha256"] = sha256_bytes(canonical_json(summary).encode("utf-8"))
    atomic_json(RUN_ROOT / "MEMORY_NAMED_SPEAKER_TEST_REPORT.json", summary)
    print(json.dumps({
        "status": summary["status"],
        "run_root": str(RUN_ROOT),
        "root_id": ROOT_ID,
        "user_id": summary["user_id"],
        "bucket_counts": summary["final_bucket_counts"],
        "checks": final_audit["checks"],
    }, ensure_ascii=False, indent=2))
    return 0 if final_audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
