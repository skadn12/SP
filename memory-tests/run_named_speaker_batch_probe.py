#!/usr/bin/env python3
"""Independent multi-episode named-speaker Memorax regression probe."""

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
ROOTS_DIR = WORKSPACE / "restart-n5-inputs/conversation_chronicles_v1/roots"
RUN_ROOT = WORKSPACE / "restart-n5-runs/conversation_chronicles_named_speaker_batch_memory_test_v2"
CREDENTIALS_PATH = Path("/root/.config/memorax/credentials.env")
USER_NAMESPACE = "memory_speaker_ref_named_batch_v1"

CASES = (
    {
        "episode": "12596",
        "relationship": "Neighbors",
        "speaker_map": {"Neighbors A": "Maya", "Neighbors B": "Liam"},
    },
    {
        "episode": "146700",
        "relationship": "Co-workers",
        "speaker_map": {"Co-workers A": "Nina", "Co-workers B": "Oscar"},
    },
    {
        "episode": "178392",
        "relationship": "Classmates",
        "speaker_map": {"Classmates A": "Priya", "Classmates B": "Noah"},
    },
    {
        "episode": "41409",
        "relationship": "Parent and Child",
        "speaker_map": {"Parent": "Elena", "Child": "Marcus"},
    },
    {
        "episode": "91716",
        "relationship": "Neighbors",
        "speaker_map": {"Neighbors A": "Sofia", "Neighbors B": "Daniel"},
    },
)

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


def root_id_for(episode: str) -> str:
    return f"cc-train-episode-{episode}-named-batch-v1"


def replace_string(value: str, source_root_id: str, root_id: str, speaker_map: Mapping[str, str]) -> str:
    result = value.replace(source_root_id, root_id)
    for source, target in speaker_map.items():
        result = result.replace(source, target)
        # Session summaries sometimes use the singular form.
        singular = source.replace("Neighbors", "Neighbor").replace(
            "Co-workers", "Co-worker").replace("Classmates", "Classmate")
        result = result.replace(singular, target)
    return result


def replace_tree(value: Any, source_root_id: str, root_id: str, speaker_map: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return replace_string(value, source_root_id, root_id, speaker_map)
    if isinstance(value, list):
        return [replace_tree(item, source_root_id, root_id, speaker_map) for item in value]
    if isinstance(value, dict):
        return {
            key: replace_tree(item, source_root_id, root_id, speaker_map)
            for key, item in value.items()
        }
    return value


def clean_root(source: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    source_root_id = str(source["root_id"])
    root_id = root_id_for(str(case["episode"]))
    cleaned = replace_tree(copy.deepcopy(dict(source)), source_root_id, root_id, case["speaker_map"])
    cleaned["root_id"] = root_id
    cleaned["sample_id"] = root_id
    cleaned["lineage_group"] = root_id
    cleaned["dialogue"]["dialogue_hash"] = dialogue_content_hash(cleaned["dialogue"]["turns"])
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
        result: list[str] = []
        for item in value:
            result.extend(flatten_strings(item))
        return result
    return []


def rows(snapshot: Mapping[str, Any], bucket_type: str) -> list[dict[str, Any]]:
    source = (snapshot.get("buckets") or {}).get(bucket_type) or {}
    return [dict(row) for row in source.values() if isinstance(row, Mapping)]


def semantic_audit(snapshot: Mapping[str, Any], speaker_map: Mapping[str, str], root_id: str) -> dict[str, Any]:
    person_rows = rows(snapshot, "person")
    event_rows = rows(snapshot, "event")
    person_text = "\n".join(flatten_strings(person_rows))
    event_text = "\n".join(flatten_strings(event_rows))
    all_text = "\n".join(flatten_strings(snapshot.get("buckets") or {}))
    names = list(speaker_map.values())
    checks = {
        "person_bucket_present": bool(person_rows),
        "event_bucket_present": bool(event_rows),
        "all_named_people_in_person_bucket": all(name.casefold() in person_text.casefold() for name in names),
        "all_named_people_in_event_bucket": all(name.casefold() in event_text.casefold() for name in names),
        "event_text_has_no_user_subject_placeholder": re.search(r"\buser\b", event_text, re.IGNORECASE) is None,
        "event_text_has_no_a_b_placeholder": re.search(
            r"\b(?:neighbors?|co-workers?|classmates?)\s+[ab]\b", event_text, re.IGNORECASE
        ) is None,
        "all_bucket_text_has_no_a_b_placeholder": re.search(
            r"\b(?:neighbors?|co-workers?|classmates?)\s+[ab]\b", all_text, re.IGNORECASE
        ) is None,
    }
    return {
        "audit_version": "named_speaker_batch_event_bucket_audit_v1",
        "root_id": root_id,
        "speaker_map": dict(speaker_map),
        "checks": checks,
        "passed": all(checks.values()),
        "bucket_counts": snapshot["bucket_counts"],
        "person_rows": person_rows,
        "event_rows": event_rows,
    }


def main() -> int:
    if RUN_ROOT.exists():
        raise RuntimeError(f"run_root_already_exists:{RUN_ROOT}")
    RUN_ROOT.mkdir(parents=True)
    (RUN_ROOT / "cases").mkdir()
    config = load_memorax_config(credentials_path=CREDENTIALS_PATH, allow_insecure_http=True)
    client = MemoraxClient(config)
    case_reports: list[dict[str, Any]] = []

    for case in CASES:
        episode = str(case["episode"])
        root_id = root_id_for(episode)
        source_path = ROOTS_DIR / f"cc-train-episode-{episode}/unified_root.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        cleaned = clean_root(source, case)
        case_dir = RUN_ROOT / "cases" / episode
        case_dir.mkdir()
        source_file_sha = sha256_file(Path(source["source"]["source_path"]))
        validation = validate_unified_dialogue_root(
            cleaned,
            expected_source_path=Path(source["source"]["source_path"]),
            expected_source_file_sha256=source_file_sha,
            expected_session_count=5,
        )
        expected_names = sorted(case["speaker_map"].values())
        actual_speakers = sorted({turn["speaker"] for turn in cleaned["dialogue"]["turns"]})
        cleaning_checks = {
            "unified_root_valid": validation["valid"],
            "root_id_changed": root_id != source["root_id"],
            "speaker_set_exact": actual_speakers == expected_names,
            "same_source_record": cleaned["source"] == source["source"],
            "turn_count_preserved": cleaned["dialogue"]["turn_count"] == source["dialogue"]["turn_count"],
            "no_placeholder_speakers": all(
                turn["speaker"] in expected_names
                for turn in cleaned["dialogue"]["turns"]
            ),
        }
        atomic_json(case_dir / "cleaned_unified_root.json", cleaned)
        atomic_json(case_dir / "cleaning.json", {
            "episode": episode,
            "relationship": case["relationship"],
            "source_path": str(source_path),
            "root_id": root_id,
            "user_namespace": USER_NAMESPACE,
            "speaker_map": case["speaker_map"],
            "source_unified_root_sha256": sha256_file(source_path),
            "cleaned_dialogue_sha256": cleaned["dialogue"]["dialogue_hash"],
            "cleaning_checks": cleaning_checks,
            "validation": validation,
        })
        if not all(cleaning_checks.values()):
            raise RuntimeError(f"cleaning_validation_failed:{episode}")

        query = build_bucket_search_payload(root_id=root_id, user_namespace=USER_NAMESPACE)
        atomic_json(case_dir / "bucket_query.json", query)
        before_response = client.search_buckets(
            query["payload"], request_id=f"named-batch-before-{episode}"
        )
        before_snapshot = normalize_bucket_snapshot(
            before_response, root_id=root_id, user_id=query["payload"]["user_id"]
        )
        atomic_json(case_dir / "before_bucket_snapshot.json", before_snapshot)
        if any(before_snapshot["bucket_counts"].values()):
            raise RuntimeError(f"fresh_namespace_not_empty:{episode}")

        intent = build_add_intent(
            root_id=root_id,
            round_index=0,
            source_dialogue_sha256=cleaned["dialogue"]["dialogue_hash"],
            turns=cleaned["dialogue"]["turns"],
            async_mode=True,
            user_namespace=USER_NAMESPACE,
        )
        atomic_json(case_dir / "add_intent.json", intent)

        def persist_state(state: Mapping[str, Any]) -> None:
            atomic_json(case_dir / "add_write_state.json", dict(state))

        completion = submit_add_and_wait(
            gateway=client,
            intent=intent,
            stage="named_speaker_batch_parent_memory_add",
            on_state_change=persist_state,
        )
        atomic_json(case_dir / "add_completion.json", completion)
        if not completion.get("verified_complete"):
            raise RuntimeError(f"add_not_verified:{episode}")

        final_snapshot = None
        final_audit = None
        search_attempts = []
        for attempt, delay in enumerate((0, 2, 5, 10, 20), 1):
            if delay:
                time.sleep(delay)
            request_id = f"named-batch-after-{episode}-{attempt:02d}"
            response = client.search_buckets(query["payload"], request_id=request_id)
            snapshot = normalize_bucket_snapshot(
                response, root_id=root_id, user_id=query["payload"]["user_id"]
            )
            audit = semantic_audit(snapshot, case["speaker_map"], root_id)
            atomic_json(case_dir / f"after_bucket_snapshot_{attempt:02d}.json", snapshot)
            search_attempts.append({
                "attempt": attempt,
                "delay_seconds": delay,
                "bucket_counts": snapshot["bucket_counts"],
                "semantic_passed": audit["passed"],
            })
            final_snapshot, final_audit = snapshot, audit
            if audit["passed"]:
                break
        assert final_snapshot is not None and final_audit is not None
        atomic_json(case_dir / "semantic_audit.json", final_audit)
        case_reports.append({
            "episode": episode,
            "relationship": case["relationship"],
            "root_id": root_id,
            "user_id": query["payload"]["user_id"],
            "speaker_map": case["speaker_map"],
            "turn_count": cleaned["dialogue"]["turn_count"],
            "cleaning_checks": cleaning_checks,
            "add_verified_complete": completion["verified_complete"],
            "task_id": completion.get("task_id"),
            "search_attempts": search_attempts,
            "semantic_audit": final_audit,
            "status": "passed" if final_audit["passed"] else "failed",
        })
        print(json.dumps({
            "episode": episode,
            "status": case_reports[-1]["status"],
            "bucket_counts": final_snapshot["bucket_counts"],
            "checks": final_audit["checks"],
        }, ensure_ascii=False), flush=True)

    passed = all(row["status"] == "passed" for row in case_reports)
    report = {
        "test_version": "conversation_chronicles_named_speaker_batch_memory_test_v1",
        "status": "passed" if passed else "failed",
        "created_at": now_utc(),
        "source_dataset": "conversation_chronicles/train.jsonl via frozen unified roots",
        "run_root": str(RUN_ROOT),
        "user_namespace": USER_NAMESPACE,
        "case_count": len(case_reports),
        "cases": case_reports,
        "memorax_configuration": config.redacted(),
        "memorax_calls": client.calls,
        "credentials_in_artifacts": False,
        "existing_pipeline_code_modified": False,
    }
    report["report_sha256"] = sha256_bytes(canonical_json(report).encode("utf-8"))
    atomic_json(RUN_ROOT / "BATCH_MEMORY_TEST_REPORT.json", report)
    print(json.dumps({
        "status": report["status"],
        "case_count": report["case_count"],
        "passed_cases": sum(row["status"] == "passed" for row in case_reports),
        "run_root": str(RUN_ROOT),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
