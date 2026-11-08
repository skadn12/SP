"""Frozen cross-root ordering generation; no Solver or training calls."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


HERE = Path(__file__).resolve().parent
PROJECT = Path("/root/autodl-tmp/gxy-selfplay")
WORKTREE = PROJECT / "worktrees/h4-writer-coverage-v1"
sys.path.insert(0, str(WORKTREE))
from scripts.current_method.restart_n5_spell.h4_memory_light_gate_siliconflow_v1.run import (
    MODEL, credential_client, provider_profile,
)
from scripts.current_method.restart_n5_spell.h4_writer_model_comparison_v1.transport import build_payload


VERSION = HERE.name
SOURCE_MANIFEST = PROJECT / "offline-deliveries/h4_latest_expansion_samples_20260922/source_manifest.json"
ROOTS = {
    "cc-train-episode-136202": (149, "61a0127ac31e3837d3475e566c062a1b95e221be66f113b37b978a33ce4b7627"),
    "cc-train-episode-197635": (148, "7b291fcf058d81520af49077c292e990b2e91af32e2356509928b6820e922e74"),
    "cc-train-episode-254852": (151, "8385b541a6d9325ece6aded7cb90ca7ddc2389ebaa86e0d7c24824c47a95f2fd"),
}
OUTPUT_LIMIT = 8192
LABELS = {f"I{i}" for i in range(1, 7)}
STATUSES = {"statement", "report", "decision", "advice", "offer", "promise", "action"}


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, value: object, *, raw: bool = False) -> None:
    """Publish a complete file without replacing an existing artifact."""
    content = (str(value) if raw else json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def profile() -> dict:
    result = provider_profile()
    if MODEL != "Qwen/Qwen3.6-35B-A3B":
        raise ValueError("unexpected_generator_model")
    result.update(profile_id=VERSION, output_limit=OUTPUT_LIMIT, http_retry_limit=0)
    result["request_parameters"].update(
        temperature=0.1, max_tokens=OUTPUT_LIMIT, enable_thinking=False,
        response_format={"type": "json_object"},
    )
    return result


def source_manifest() -> dict[str, dict]:
    rows = read_json(SOURCE_MANIFEST)
    if not isinstance(rows, list):
        raise ValueError("source_manifest_not_array")
    selected = {row["root_id"]: row for row in rows if isinstance(row, dict) and row.get("root_id") in ROOTS}
    if set(selected) != set(ROOTS) or sum(isinstance(row, dict) and row.get("root_id") in ROOTS for row in rows) != 3:
        raise ValueError("source_manifest_roots_changed")
    for root_id, row in selected.items():
        count, expected_hash = ROOTS[root_id]
        if row.get("partial_promotion") is not False or row.get("total_turns") != count or row.get("source_sha256") != expected_hash:
            raise ValueError(f"source_manifest_contract_changed:{root_id}")
    return selected


def checked_source(root_id: str, row: dict, *, frozen: bool) -> tuple[bytes, dict]:
    path = HERE / root_id / "source.json" if frozen else Path(row["source_path"])
    raw = path.read_bytes()
    count, expected_hash = ROOTS[root_id]
    if sha(raw) != expected_hash:
        raise ValueError(f"source_hash_changed:{root_id}")
    data = json.loads(raw)
    turns = data.get("turns")
    if (data.get("root_id") != root_id or data.get("turn_count") != count or
            not isinstance(turns, list) or len(turns) != count or
            data.get("round_count") != 2 or len(data.get("round_boundaries", [])) != 2):
        raise ValueError(f"source_identity_changed:{root_id}")
    refs = [turn.get("turn_ref") for turn in turns if isinstance(turn, dict)]
    if len(refs) != count or any(not isinstance(ref, str) or not ref for ref in refs) or len(set(refs)) != count:
        raise ValueError(f"source_turn_refs_invalid:{root_id}")
    return raw, data


def prompt_for(root_id: str, source: dict, instructions: str) -> str:
    surface = {key: source[key] for key in ("root_id", "turn_count", "round_count", "round_boundaries", "turns")}
    text = instructions.replace("{ROOT_ID}", root_id).replace("{ROOT_SUFFIX}", root_id.rsplit("-", 1)[-1])
    return text + json.dumps(surface, ensure_ascii=False, indent=2) + "\n"


def prepare() -> None:
    if (HERE / "BATCH_PREPARED.json").exists():
        raise FileExistsError("batch_already_prepared_no_overwrite")
    rows = source_manifest()
    instructions = (HERE / "instructions.txt").read_bytes()
    frozen = []
    for root_id in ROOTS:
        row = rows[root_id]
        raw, source = checked_source(root_id, row, frozen=False)
        prompt = prompt_for(root_id, source, instructions.decode("utf-8"))
        payload = build_payload(profile(), prompt)
        directory = HERE / root_id
        if directory.exists():
            raise FileExistsError(f"root_directory_exists_no_overwrite:{root_id}")
        directory.mkdir()
        write_new(directory / "source.json", raw.decode("utf-8"), raw=True)
        write_new(directory / "prompt.txt", prompt, raw=True)
        write_new(directory / "request.json", payload)
        frozen.append({
            "root_id": root_id, "source_path": row["source_path"],
            "source_sha256": sha(raw), "source_turns": len(source["turns"]),
            "source_round_count": source["round_count"], "partial_promotion": False,
            "prompt_sha256": sha(prompt.encode("utf-8")), "payload_sha256": sha(canonical(payload)),
            "requested_model": MODEL, "maximum_candidates": 2, "maximum_http_attempts": 1,
            "status": "PREPARED_NO_API_CALL",
        })
    write_new(HERE / "BATCH_PREPARED.json", {
        "version": VERSION, "requested_model": MODEL, "profile": profile(),
        "source_manifest_path": str(SOURCE_MANIFEST), "source_manifest_sha256": sha(SOURCE_MANIFEST.read_bytes()),
        "instructions_sha256": sha(instructions), "script_sha256": sha(Path(__file__).read_bytes()),
        "roots": frozen, "generator_calls": 0, "solver_calls": 0, "training_started": False,
    })
    print(json.dumps({"status": "PREPARED_NO_API_CALL", "roots": len(frozen), "model": MODEL}))


def frozen_inputs(root_id: str) -> tuple[dict, dict, str, dict]:
    batch = read_json(HERE / "BATCH_PREPARED.json")
    if (batch.get("version") != VERSION or batch.get("requested_model") != MODEL or
            batch.get("source_manifest_sha256") != sha(SOURCE_MANIFEST.read_bytes()) or
            batch.get("instructions_sha256") != sha((HERE / "instructions.txt").read_bytes()) or
            batch.get("script_sha256") != sha(Path(__file__).read_bytes()) or batch.get("profile") != profile()):
        raise ValueError("batch_freeze_changed")
    rows = source_manifest()
    entry = next((row for row in batch["roots"] if row.get("root_id") == root_id), None)
    if entry is None or root_id not in ROOTS or entry.get("partial_promotion") is not False:
        raise ValueError("root_not_prepared")
    original_raw, _ = checked_source(root_id, rows[root_id], frozen=False)
    frozen_raw, source = checked_source(root_id, rows[root_id], frozen=True)
    if frozen_raw != original_raw or entry["source_sha256"] != sha(frozen_raw):
        raise ValueError("source_freeze_changed")
    prompt = (HERE / root_id / "prompt.txt").read_text(encoding="utf-8")
    request = read_json(HERE / root_id / "request.json")
    if (sha(prompt.encode("utf-8")) != entry["prompt_sha256"] or
            prompt != prompt_for(root_id, source, (HERE / "instructions.txt").read_text(encoding="utf-8")) or
            sha(canonical(request)) != entry["payload_sha256"] or request != build_payload(profile(), prompt)):
        raise ValueError("frozen_request_changed")
    return entry, source, prompt, request


def check_candidate(row: object, root_id: str, turns: list[dict]) -> list[str]:
    if not isinstance(row, dict):
        return ["candidate_not_object"]
    errors = []
    suffix = root_id.rsplit("-", 1)[-1]
    allowed_ids = {f"{suffix}-qwen36-order-0{i}" for i in (1, 2)}
    if row.get("candidate_id") not in allowed_ids:
        errors.append("candidate_id_invalid")
    for field in ("topic", "question", "order_reason"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            errors.append(field + "_missing")
    if row.get("root_id") != root_id or row.get("question_type") != "ordering":
        errors.append("root_or_type_mismatch")
    if row.get("chronology_kind") not in {"conversation_order", "within_scene_actions"}:
        errors.append("chronology_kind_invalid")
    items = row.get("items")
    if not isinstance(items, dict) or set(items) != LABELS or any(not isinstance(value, str) or not value.strip() for value in items.values()):
        errors.append("six_items_required")
    gold = row.get("gold")
    if not isinstance(gold, list) or len(gold) != 6 or any(not isinstance(value, str) for value in gold) or set(gold) != LABELS:
        errors.append("gold_not_complete_permutation")
    evidence = row.get("item_evidence")
    if not isinstance(evidence, dict) or set(evidence) != LABELS:
        return errors + ["six_evidence_entries_required"]
    by_ref = {turn["turn_ref"]: turn for turn in turns}
    positions = {turn["turn_ref"]: index for index, turn in enumerate(turns)}
    refs = {}
    for label in sorted(LABELS):
        entry = evidence[label]
        if not isinstance(entry, dict):
            errors.append(label + ":evidence_not_object")
            continue
        ref, quote = entry.get("turn_ref"), entry.get("quote")
        if not isinstance(ref, str) or ref not in by_ref:
            errors.append(label + ":invalid_turn_ref")
            continue
        refs[label] = ref
        if not isinstance(quote, str) or not quote.strip() or quote not in by_ref[ref]["text"]:
            errors.append(label + ":quote_not_verbatim")
        if entry.get("status") not in STATUSES or not isinstance(entry.get("reason"), str) or not entry["reason"].strip():
            errors.append(label + ":invalid_status_or_reason")
    if len(refs) == 6:
        if len(set(refs.values())) != 6:
            errors.append("duplicate_anchor_turns")
        if "gold_not_complete_permutation" not in errors and [positions[refs[label]] for label in gold] != sorted(positions[ref] for ref in refs.values()):
            errors.append("gold_conflicts_with_anchor_order")
        if row.get("chronology_kind") == "within_scene_actions" and len({ref.rsplit(":T", 1)[0] for ref in refs.values()}) != 1:
            errors.append("action_sequence_spans_scenes")
    return errors


def validate(root_id: str) -> dict:
    entry, source, _, _ = frozen_inputs(root_id)
    directory = HERE / root_id
    response = read_json(directory / "RESPONSE.json")
    raw_text, metadata = response["raw_text"], response["metadata"]
    if response.get("source_sha256") != entry["source_sha256"] or response.get("payload_sha256") != entry["payload_sha256"]:
        raise ValueError("response_binding_changed")
    try:
        parsed = json.loads(raw_text)
    except (TypeError, ValueError):
        parsed = None
    questions = parsed.get("questions") if isinstance(parsed, dict) else None
    abstentions = parsed.get("abstentions") if isinstance(parsed, dict) else None
    global_errors = []
    if not isinstance(questions, list):
        global_errors.append("questions_array_missing")
        questions = []
    if len(questions) > 2:
        global_errors.append("candidate_count_exceeds_two")
    if not isinstance(abstentions, list) or any(not isinstance(value, str) for value in abstentions):
        global_errors.append("abstentions_array_invalid")
    if metadata.get("returned_model") != MODEL:
        global_errors.append("returned_model_mismatch")
    checks, seen_ids, seen_events = [], set(), set()
    valid = []
    for index, question in enumerate(questions, 1):
        errors = check_candidate(question, root_id, source["turns"])
        candidate_id = question.get("candidate_id") if isinstance(question, dict) else None
        if candidate_id in seen_ids:
            errors.append("duplicate_candidate_id")
        seen_ids.add(candidate_id)
        evidence = question.get("item_evidence") if isinstance(question, dict) else None
        if isinstance(evidence, dict):
            event_set = tuple(sorted(value.get("turn_ref") for value in evidence.values() if isinstance(value, dict) and isinstance(value.get("turn_ref"), str)))
            if len(event_set) == 6 and event_set in seen_events:
                errors.append("duplicate_event_set")
            seen_events.add(event_set)
        checks.append({"index": index, "candidate_id": candidate_id, "errors": errors})
        if not errors:
            valid.append(question)
    if global_errors:
        valid = []
    provenance = {
        "author": "REMOTE_LARGE_MODEL", "provider": "siliconflow", "requested_model": MODEL,
        "returned_model": metadata.get("returned_model"),
        "response_id": (metadata.get("raw_response") or {}).get("id"),
        "root_id": root_id, "source_sha256": entry["source_sha256"],
        "prompt_sha256": entry["prompt_sha256"], "payload_sha256": entry["payload_sha256"],
        "model_questions_modified_by_codex": False, "solver_evaluated": False,
        "selected_for_training": False, "training_started": False,
    }
    result = {
        "status": "MECHANICAL_CHECKS_ONLY_CONTENT_AUDIT_PENDING" if not global_errors else "MECHANICAL_REJECTED",
        "root_id": root_id, "candidate_count": len(questions), "mechanically_valid_count": len(valid),
        "global_errors": global_errors, "checks": checks, "semantic_gold_verified": False,
        "provenance": provenance,
    }
    write_new(directory / "PARSED_RESPONSE.json", parsed)
    write_new(directory / "VALIDATED_CANDIDATES.json", {"status": result["status"], "provenance": provenance, "questions": valid})
    write_new(directory / "VALIDATION.json", result)
    return result


def execute(root_id: str) -> None:
    entry, _, prompt, payload = frozen_inputs(root_id)
    directory = HERE / root_id
    write_new(directory / "ATTEMPT_RESERVED.json", {
        "status": "RESERVED_ONE_GENERATOR_CALL", "created_at": datetime.now(timezone.utc).isoformat(),
        "requested_model": MODEL, "maximum_http_attempts": 1,
        "payload_sha256": entry["payload_sha256"], "authorization": "USER_REQUEST_CROSSROOT_BATCH_GENERATION",
    })

    def on_attempt(actual: dict) -> None:
        if actual != payload:
            raise ValueError("actual_payload_changed")
        write_new(directory / "HTTP_ATTEMPT.json", {
            "attempt_number": 1, "payload_sha256": sha(canonical(actual)),
            "model": actual["model"], "max_tokens": actual["max_tokens"],
        })

    try:
        response = credential_client(profile(), on_http_attempt=on_attempt).complete(
            prompt=prompt, max_tokens=OUTPUT_LIMIT, json_object=True,
        )
    except Exception as error:
        write_new(directory / "ERROR.json", {
            "status": "ERROR_NO_RETRY", "error_type": type(error).__name__,
            "metadata": getattr(error, "metadata", None),
        })
        print(json.dumps({"root_id": root_id, "status": "ERROR_NO_RETRY", "error_type": type(error).__name__}))
        return
    write_new(directory / "RESPONSE.json", {
        "source_sha256": entry["source_sha256"], "payload_sha256": entry["payload_sha256"],
        "raw_text": response.raw_text, "metadata": response.metadata,
    })
    result = validate(root_id)
    print(json.dumps(result, ensure_ascii=False))


def summary() -> None:
    batch = read_json(HERE / "BATCH_PREPARED.json")
    roots = []
    for entry in batch["roots"]:
        root_id = entry["root_id"]
        path = HERE / root_id / "VALIDATION.json"
        roots.append(read_json(path) if path.exists() else {"root_id": root_id, "status": "NO_VALIDATION"})
    result = {
        "version": VERSION, "requested_model": MODEL,
        "root_count": len(roots), "candidate_count": sum(row.get("candidate_count", 0) for row in roots),
        "mechanically_valid_count": sum(row.get("mechanically_valid_count", 0) for row in roots),
        "roots": roots, "solver_calls": 0, "training_started": False,
    }
    if all(row.get("status") != "NO_VALIDATION" for row in roots):
        write_new(HERE / "BATCH_VALIDATION.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--execute", choices=ROOTS)
    options.add_argument("--validate", choices=ROOTS)
    options.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.execute:
        execute(args.execute)
    elif args.validate:
        print(json.dumps(validate(args.validate), ensure_ascii=False))
    elif args.summary:
        summary()
    else:
        prepare()
