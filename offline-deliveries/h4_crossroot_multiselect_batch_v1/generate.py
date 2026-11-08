"""Frozen cross-root multi-select generation; semantic review is separate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
PROJECT = Path("/root/autodl-tmp/gxy-selfplay")
sys.path.insert(0, str(PROJECT / "offline-deliveries/h4_crossroot_ordering_batch_v2"))
import generate as ordering


VERSION = HERE.name
MODEL = ordering.MODEL
ROOTS = ordering.ROOTS
SOURCE_MANIFEST = ordering.SOURCE_MANIFEST
LABELS = set("ABCDEF")
OUTPUT_LIMIT = ordering.OUTPUT_LIMIT
sha = ordering.sha
canonical = ordering.canonical
read_json = ordering.read_json
write_new = ordering.write_new


def profile() -> dict:
    result = ordering.profile()
    result["profile_id"] = VERSION
    return result


def prompt_for(root_id: str, source: dict, instructions: str) -> str:
    return ordering.prompt_for(root_id, source, instructions)


def prepare() -> None:
    if (HERE / "BATCH_PREPARED.json").exists():
        raise FileExistsError("batch_already_prepared_no_overwrite")
    rows = ordering.source_manifest()
    instructions = (HERE / "instructions.txt").read_bytes()
    planned = []
    for root_id in ROOTS:
        if (HERE / root_id).exists():
            raise FileExistsError(f"root_directory_exists_no_overwrite:{root_id}")
        raw, source = ordering.checked_source(root_id, rows[root_id], frozen=False)
        prompt = prompt_for(root_id, source, instructions.decode("utf-8"))
        payload = ordering.build_payload(profile(), prompt)
        planned.append((root_id, rows[root_id], raw, source, prompt, payload))
    frozen = []
    for root_id, row, raw, source, prompt, payload in planned:
        directory = HERE / root_id
        directory.mkdir()
        write_new(directory / "source.json", raw.decode("utf-8"), raw=True)
        write_new(directory / "prompt.txt", prompt, raw=True)
        write_new(directory / "request.json", payload)
        frozen.append({
            "root_id": root_id, "source_path": row["source_path"],
            "source_sha256": sha(raw), "source_turns": len(source["turns"]),
            "source_round_count": source["round_count"], "partial_promotion": False,
            "prompt_sha256": sha(prompt.encode("utf-8")),
            "payload_sha256": sha(canonical(payload)), "requested_model": MODEL,
            "maximum_candidates": 2, "maximum_http_attempts": 1,
            "status": "PREPARED_NO_API_CALL",
        })
    write_new(HERE / "BATCH_PREPARED.json", {
        "version": VERSION, "requested_model": MODEL, "profile": profile(),
        "source_manifest_path": str(SOURCE_MANIFEST),
        "source_manifest_sha256": sha(SOURCE_MANIFEST.read_bytes()),
        "instructions_sha256": sha(instructions),
        "script_sha256": sha(Path(__file__).read_bytes()),
        "roots": frozen, "generator_calls": 0, "solver_calls": 0,
        "training_started": False,
    })
    print(json.dumps({"status": "PREPARED_NO_API_CALL", "roots": len(frozen), "model": MODEL}))


def frozen_inputs(root_id: str) -> tuple[dict, dict, str, dict]:
    batch = read_json(HERE / "BATCH_PREPARED.json")
    if (batch.get("version") != VERSION or batch.get("requested_model") != MODEL or
            batch.get("source_manifest_sha256") != sha(SOURCE_MANIFEST.read_bytes()) or
            batch.get("instructions_sha256") != sha((HERE / "instructions.txt").read_bytes()) or
            batch.get("script_sha256") != sha(Path(__file__).read_bytes()) or
            batch.get("profile") != profile()):
        raise ValueError("batch_freeze_changed")
    rows = ordering.source_manifest()
    entry = next((row for row in batch["roots"] if row.get("root_id") == root_id), None)
    if root_id not in ROOTS or entry is None or entry.get("partial_promotion") is not False:
        raise ValueError("root_not_prepared")
    original_raw, _ = ordering.checked_source(root_id, rows[root_id], frozen=False)
    frozen_raw = (HERE / root_id / "source.json").read_bytes()
    if frozen_raw != original_raw or sha(frozen_raw) != entry["source_sha256"]:
        raise ValueError("source_freeze_changed")
    source = json.loads(frozen_raw)
    prompt = (HERE / root_id / "prompt.txt").read_text(encoding="utf-8")
    request = read_json(HERE / root_id / "request.json")
    expected = prompt_for(root_id, source, (HERE / "instructions.txt").read_text(encoding="utf-8"))
    if (prompt != expected or sha(prompt.encode("utf-8")) != entry["prompt_sha256"] or
            request != ordering.build_payload(profile(), prompt) or
            sha(canonical(request)) != entry["payload_sha256"]):
        raise ValueError("frozen_request_changed")
    return entry, source, prompt, request


def preflight() -> None:
    rows = []
    for root_id in ROOTS:
        entry, source, prompt, payload = frozen_inputs(root_id)
        directory = HERE / root_id
        rows.append({
            "root_id": root_id, "turns": len(source["turns"]),
            "r02_turns": sum(turn["turn_ref"].startswith("R02:") for turn in source["turns"]),
            "prompt_bytes": len(prompt.encode("utf-8")),
            "payload_sha256": entry["payload_sha256"],
            "request_model": payload["model"],
            "request_count": int((directory / "HTTP_ATTEMPT.json").exists()),
            "response_present": (directory / "RESPONSE.json").exists(),
        })
    print(json.dumps({"status": "FROZEN_INPUTS_VERIFIED_NO_API_CALL", "roots": rows}, ensure_ascii=False))


def check_candidate(row: object, root_id: str, turns: list[dict]) -> list[str]:
    if not isinstance(row, dict):
        return ["candidate_not_object"]
    errors = []
    suffix = root_id.rsplit("-", 1)[-1]
    allowed = {f"{suffix}-qwen36-multi-0{i}" for i in (1, 2)}
    if row.get("candidate_id") not in allowed:
        errors.append("candidate_id_invalid")
    if row.get("root_id") != root_id or row.get("question_type") != "multi_select":
        errors.append("root_or_type_mismatch")
    for field in ("topic", "question", "answer_reason"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            errors.append(f"{field}_missing")
    if isinstance(row.get("question"), str) and "select all that apply" not in row["question"].lower():
        errors.append("select_all_instruction_missing")
    options = row.get("options")
    if (not isinstance(options, dict) or set(options) != LABELS or
            any(not isinstance(value, str) or not value.strip() for value in options.values())):
        errors.append("six_options_required")
    elif len({value.strip().casefold() for value in options.values()}) != 6:
        errors.append("duplicate_option_text")
    gold = row.get("gold")
    valid_gold = (isinstance(gold, list) and len(gold) == 3 and
                  all(isinstance(label, str) for label in gold) and
                  len(set(gold)) == 3 and set(gold) <= LABELS)
    if not valid_gold:
        errors.append("gold_must_have_three_distinct_labels")
    evidence = row.get("option_evidence")
    if not isinstance(evidence, dict) or set(evidence) != LABELS:
        return errors + ["six_evidence_entries_required"]
    by_ref = {turn["turn_ref"]: turn for turn in turns}
    r02_refs = set()
    for label in sorted(LABELS):
        item = evidence[label]
        if not isinstance(item, dict):
            errors.append(f"{label}:evidence_not_object")
            continue
        ref, quote = item.get("turn_ref"), item.get("quote")
        if not isinstance(ref, str) or ref not in by_ref:
            errors.append(f"{label}:invalid_turn_ref")
        elif not isinstance(quote, str) or not quote.strip() or quote not in by_ref[ref]["text"]:
            errors.append(f"{label}:quote_not_verbatim")
        if not isinstance(item.get("reason"), str) or not item["reason"].strip():
            errors.append(f"{label}:reason_missing")
        expected_relation = "supports" if valid_gold and label in gold else "contradicts"
        if item.get("relation") != expected_relation:
            errors.append(f"{label}:relation_mismatch")
        if valid_gold and label in gold and isinstance(ref, str) and ref.startswith("R02:"):
            r02_refs.add(ref)
    if valid_gold and len(r02_refs) < 2:
        errors.append("fewer_than_two_distinct_r02_gold_anchors")
    return errors


def validate(root_id: str) -> dict:
    entry, source, _, _ = frozen_inputs(root_id)
    directory = HERE / root_id
    response = read_json(directory / "RESPONSE.json")
    if (response.get("source_sha256") != entry["source_sha256"] or
            response.get("payload_sha256") != entry["payload_sha256"]):
        raise ValueError("response_binding_changed")
    raw_text, metadata = response["raw_text"], response["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("response_metadata_invalid")
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
    if not isinstance(abstentions, list) or any(not isinstance(v, str) for v in abstentions):
        global_errors.append("abstentions_array_invalid")
    if metadata.get("returned_model") != MODEL:
        global_errors.append("returned_model_mismatch")
    if metadata.get("finish_reason") == "length":
        global_errors.append("response_truncated")
    checks, valid, seen_ids = [], [], set()
    for index, question in enumerate(questions, 1):
        errors = check_candidate(question, root_id, source["turns"])
        candidate_id = question.get("candidate_id") if isinstance(question, dict) else None
        if candidate_id in seen_ids:
            errors.append("duplicate_candidate_id")
        seen_ids.add(candidate_id)
        checks.append({"index": index, "candidate_id": candidate_id, "errors": errors})
        if not errors:
            valid.append(question)
    if global_errors:
        valid = []
    provenance = {
        "author": "REMOTE_LARGE_MODEL", "provider": "siliconflow",
        "requested_model": MODEL, "returned_model": metadata.get("returned_model"),
        "response_id": (metadata.get("raw_response") or {}).get("id"),
        "root_id": root_id, "source_sha256": entry["source_sha256"],
        "prompt_sha256": entry["prompt_sha256"],
        "payload_sha256": entry["payload_sha256"],
        "model_questions_modified_by_codex": False,
        "solver_evaluated": False, "selected_for_training": False,
        "training_started": False,
    }
    result = {
        "status": "MECHANICAL_CHECKS_ONLY_CONTENT_AUDIT_PENDING" if not global_errors else "MECHANICAL_REJECTED",
        "root_id": root_id, "candidate_count": len(questions),
        "mechanically_valid_count": len(valid), "global_errors": global_errors,
        "checks": checks, "semantic_gold_verified": False, "provenance": provenance,
    }
    write_new(directory / "PARSED_RESPONSE.json", parsed)
    write_new(directory / "VALIDATED_CANDIDATES.json", {
        "status": result["status"], "provenance": provenance, "questions": valid,
    })
    write_new(directory / "VALIDATION.json", result)
    return result


def execute(root_id: str) -> None:
    entry, _, prompt, payload = frozen_inputs(root_id)
    directory = HERE / root_id
    write_new(directory / "ATTEMPT_RESERVED.json", {
        "status": "RESERVED_ONE_GENERATOR_CALL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "requested_model": MODEL, "maximum_http_attempts": 1,
        "payload_sha256": entry["payload_sha256"],
        "authorization": "USER_REQUEST_CROSSROOT_MULTISELECT_BATCH_GENERATION",
    })

    def on_attempt(actual: dict) -> None:
        if actual != payload:
            raise ValueError("actual_payload_changed")
        write_new(directory / "HTTP_ATTEMPT.json", {
            "attempt_number": 1, "payload_sha256": sha(canonical(actual)),
            "model": actual["model"], "max_tokens": actual["max_tokens"],
        })

    try:
        response = ordering.credential_client(profile(), on_http_attempt=on_attempt).complete(
            prompt=prompt, max_tokens=OUTPUT_LIMIT, json_object=True,
        )
    except Exception as error:
        write_new(directory / "ERROR.json", {
            "status": "ERROR_NO_RETRY", "error_type": type(error).__name__,
            "metadata": getattr(error, "metadata", None),
        })
        print(json.dumps({"root_id": root_id, "status": "ERROR_NO_RETRY",
                          "error_type": type(error).__name__}))
        return
    write_new(directory / "RESPONSE.json", {
        "source_sha256": entry["source_sha256"],
        "payload_sha256": entry["payload_sha256"],
        "raw_text": response.raw_text, "metadata": response.metadata,
    })
    print(json.dumps(validate(root_id), ensure_ascii=False))


def summary() -> None:
    batch = read_json(HERE / "BATCH_PREPARED.json")
    roots = []
    for entry in batch["roots"]:
        root_id = entry["root_id"]
        path = HERE / root_id / "VALIDATION.json"
        roots.append(read_json(path) if path.exists() else {
            "root_id": root_id, "status": "NO_VALIDATION",
        })
    result = {
        "version": VERSION, "requested_model": MODEL, "root_count": len(roots),
        "candidate_count": sum(row.get("candidate_count", 0) for row in roots),
        "mechanically_valid_count": sum(row.get("mechanically_valid_count", 0) for row in roots),
        "roots": roots, "solver_calls": 0, "training_started": False,
    }
    if all(row.get("status") != "NO_VALIDATION" for row in roots):
        write_new(HERE / "BATCH_VALIDATION.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--execute", choices=ROOTS)
    actions.add_argument("--validate", choices=ROOTS)
    actions.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
    elif args.execute:
        execute(args.execute)
    elif args.validate:
        print(json.dumps(validate(args.validate), ensure_ascii=False))
    elif args.summary:
        summary()
    else:
        prepare()
