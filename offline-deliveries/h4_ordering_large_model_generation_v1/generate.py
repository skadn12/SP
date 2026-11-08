"""One frozen, attributed remote Generator call; no Solver or training calls."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
PROJECT = Path("/root/autodl-tmp/gxy-selfplay")
WORKTREE = PROJECT / "worktrees/h4-writer-coverage-v1"
sys.path.insert(0, str(WORKTREE))
from scripts.current_method.restart_n5_spell.h4_memory_light_gate_siliconflow_v1.run import (
    MODEL, credential_client, provider_profile,
)
from scripts.current_method.restart_n5_spell.h4_writer_model_comparison_v1.transport import build_payload

VERSION = HERE.name
SOURCE = PROJECT / "offline-deliveries/h4_memory_light_gate_siliconflow_qa_v1/prepared_run/frozen/qa_source.json"
SOURCE_SHA = "b163bd9ceed0ec0cf4745e5c414fb5d7615788672c879c6283669aad78395113"
ROOT_ID = "cc-train-episode-259192"
OUTPUT_LIMIT = 8192
LABELS = {f"I{i}" for i in range(1, 7)}
STATUSES = {"statement", "report", "decision", "advice", "offer", "promise", "action"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def write_new(name: str, value: object, *, raw: bool = False) -> None:
    with (HERE / name).open("x", encoding="utf-8") as handle:
        if raw:
            handle.write(str(value))
        else:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")


def read(name: str) -> object:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def source() -> dict:
    data = SOURCE.read_bytes()
    if sha(data) != SOURCE_SHA:
        raise ValueError("source_hash_changed")
    result = json.loads(data)
    turns = result["turns"]
    if result["root_id"] != ROOT_ID or len(turns) != 188:
        raise ValueError("unexpected_source")
    if len({row["turn_ref"] for row in turns}) != len(turns):
        raise ValueError("duplicate_source_refs")
    return result


def profile() -> dict:
    value = provider_profile()
    if MODEL != "Qwen/Qwen3.6-35B-A3B":
        raise ValueError("unexpected_generator_model")
    value.update(profile_id=VERSION, output_limit=OUTPUT_LIMIT, http_retry_limit=0)
    value["request_parameters"].update(
        temperature=0.1, max_tokens=OUTPUT_LIMIT, enable_thinking=False,
        response_format={"type": "json_object"},
    )
    return value


def prepare() -> None:
    value = source()
    surface = {key: value[key] for key in (
        "root_id", "turn_count", "conversation_date_segments",
        "unknown_conversation_date_segments", "turns",
    )}
    instructions = (HERE / "instructions.txt").read_bytes()
    prompt = instructions.decode("utf-8") + json.dumps(surface, ensure_ascii=False, indent=2) + "\n"
    payload = build_payload(profile(), prompt)
    write_new("PREPARED.json", {
        "status": "PREPARED", "version": VERSION, "root_id": ROOT_ID,
        "source_path": str(SOURCE), "source_sha256": SOURCE_SHA, "source_turns": 188,
        "instructions_sha256": sha(instructions), "script_sha256": sha(Path(__file__).read_bytes()),
        "prompt_sha256": sha(prompt.encode("utf-8")), "payload_sha256": sha(canonical(payload)),
        "requested_model": MODEL, "profile": profile(), "target_candidates": 6,
        "maximum_http_attempts": 1, "solver_calls": 0, "training_started": False,
    })
    write_new("prompt.txt", prompt, raw=True)
    write_new("request.json", payload)
    print(json.dumps({"status": "PREPARED", "model": MODEL, "turns": 188, "http_calls": 0}))


def check_candidate(row: object, turns: list[dict]) -> list[str]:
    if not isinstance(row, dict):
        return ["candidate_not_object"]
    errors = []
    for field in ("candidate_id", "topic", "question", "order_reason"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            errors.append(field + "_missing")
    if row.get("root_id") != ROOT_ID or row.get("question_type") != "ordering":
        errors.append("root_or_type_mismatch")
    if row.get("chronology_kind") not in {"conversation_order", "within_scene_actions"}:
        errors.append("chronology_kind_invalid")
    items = row.get("items")
    if not isinstance(items, dict) or set(items) != LABELS or any(
        not isinstance(text, str) or not text.strip() for text in items.values()
    ):
        errors.append("six_items_required")
    gold = row.get("gold")
    if not isinstance(gold, list) or len(gold) != 6 or any(not isinstance(x, str) for x in gold) or set(gold) != LABELS:
        errors.append("gold_not_complete_permutation")
    evidence = row.get("item_evidence")
    if not isinstance(evidence, dict) or set(evidence) != LABELS:
        return errors + ["six_evidence_entries_required"]
    by_ref = {turn["turn_ref"]: turn for turn in turns}
    positions = {turn["turn_ref"]: i for i, turn in enumerate(turns)}
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
        if "gold_not_complete_permutation" not in errors:
            actual = [positions[refs[label]] for label in gold]
            if actual != sorted(actual):
                errors.append("gold_conflicts_with_anchor_order")
        if row.get("chronology_kind") == "within_scene_actions" and len({ref.rsplit(":T", 1)[0] for ref in refs.values()}) != 1:
            errors.append("action_sequence_spans_scenes")
    return errors


def save_candidates(raw_text: str, metadata: dict, turns: list[dict]) -> None:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        write_new("VALIDATION.json", {"status": "INVALID_MODEL_JSON", "candidate_count": 0})
        return
    write_new("parsed_response.json", parsed)
    questions = parsed.get("questions") if isinstance(parsed, dict) else None
    if not isinstance(questions, list):
        write_new("VALIDATION.json", {"status": "MISSING_QUESTIONS_ARRAY", "candidate_count": 0})
        return
    checks, seen_ids, seen_events, validated = [], set(), set(), []
    for index, row in enumerate(questions, 1):
        errors = check_candidate(row, turns)
        candidate_id = row.get("candidate_id") if isinstance(row, dict) else None
        if isinstance(candidate_id, str):
            if candidate_id in seen_ids:
                errors.append("duplicate_candidate_id")
            seen_ids.add(candidate_id)
        if isinstance(row, dict) and isinstance(row.get("item_evidence"), dict):
            event_set = tuple(sorted(str(entry.get("turn_ref")) for entry in row["item_evidence"].values() if isinstance(entry, dict)))
            if len(event_set) == 6 and event_set in seen_events:
                errors.append("duplicate_event_set")
            seen_events.add(event_set)
        checks.append({"index": index, "candidate_id": candidate_id, "errors": errors})
        if not errors:
            validated.append(row)
    provenance = {
        "author": "REMOTE_LARGE_MODEL", "provider": "siliconflow", "requested_model": MODEL,
        "returned_model": metadata.get("returned_model"),
        "response_id": (metadata.get("raw_response") or {}).get("id"),
        "source_path": str(SOURCE), "source_sha256": SOURCE_SHA,
        "model_questions_modified_by_codex": False, "solver_evaluated": False,
        "selected_for_training": False, "training_started": False,
    }
    write_new("CANDIDATES.json", {"status": "GENERATED_NOT_SOLVER_SCREENED", "provenance": provenance,
                                  "questions": questions, "abstentions": parsed.get("abstentions")})
    write_new("VALIDATED_CANDIDATES.json", {"status": "MECHANICAL_CHECKS_ONLY_CONTENT_AUDIT_PENDING",
                                            "provenance": provenance, "questions": validated})
    write_new("VALIDATION.json", {"status": "CHECKED", "candidate_count": len(questions),
                                  "mechanically_valid_count": len(validated), "target_count": 6,
                                  "count_within_requested_limit": len(questions) <= 6, "checks": checks,
                                  "semantic_gold_verified": False})
    lines = ["# Remote Model Generated Ordering Candidates", "", f"- Requested model: `{MODEL}`",
             f"- Returned model: `{metadata.get('returned_model')}`", "- Provider: SiliconFlow",
             f"- Root: `{ROOT_ID}`", "- Complete source: 188 turns, frozen R03 dialogue",
             "- Questions and Gold below are unmodified model output.",
             "- Mechanical validation is recorded separately; semantic audit is pending.",
             "- No Solver evaluation, training selection, SFT export, or training has been performed.", ""]
    for row in questions:
        if not isinstance(row, dict):
            continue
        lines += [f"## {row.get('candidate_id')}: {row.get('topic')}", "", str(row.get("question")), ""]
        if isinstance(row.get("items"), dict):
            lines += [f"- {label}: {text}" for label, text in row["items"].items()]
        lines += ["", "Gold: `" + str(row.get("gold")) + "`", "", str(row.get("order_reason")), ""]
    write_new("QUESTIONS.md", "\n".join(lines), raw=True)


def execute() -> None:
    frozen = read("PREPARED.json")
    value = source()
    prompt = (HERE / "prompt.txt").read_text(encoding="utf-8")
    payload = read("request.json")
    if (sha(prompt.encode("utf-8")) != frozen["prompt_sha256"]
            or sha(canonical(payload)) != frozen["payload_sha256"]
            or build_payload(profile(), prompt) != payload
            or sha((HERE / "instructions.txt").read_bytes()) != frozen["instructions_sha256"]
            or sha(Path(__file__).read_bytes()) != frozen["script_sha256"]):
        raise ValueError("frozen_generation_inputs_changed")
    write_new("ATTEMPT_RESERVED.json", {
        "status": "RESERVED_ONE_GENERATOR_CALL", "created_at": datetime.now(timezone.utc).isoformat(),
        "requested_model": MODEL, "maximum_http_attempts": 1,
        "prompt_sha256": frozen["prompt_sha256"], "authorization": "USER_REQUEST_LARGE_MODEL_QUESTION_GENERATION",
    })

    def on_attempt(actual: dict) -> None:
        if actual != payload:
            raise ValueError("actual_payload_changed")
        write_new("HTTP_ATTEMPT.json", {"attempt_number": 1, "payload_sha256": sha(canonical(actual)),
                                        "model": actual["model"], "max_tokens": actual["max_tokens"]})

    try:
        response = credential_client(profile(), on_http_attempt=on_attempt).complete(
            prompt=prompt, max_tokens=OUTPUT_LIMIT, json_object=True)
    except Exception as error:
        write_new("ERROR.json", {"status": "ERROR_NO_RETRY", "error_type": type(error).__name__,
                                  "metadata": getattr(error, "metadata", None)})
        print(json.dumps({"status": "ERROR_NO_RETRY", "error_type": type(error).__name__}))
        return
    write_new("response.raw.txt", response.raw_text, raw=True)
    write_new("RESPONSE_METADATA.json", response.metadata)
    save_candidates(response.raw_text, response.metadata, value["turns"])
    source()
    print(json.dumps({"status": "RESPONSE_SAVED", "model": response.metadata.get("returned_model"),
                      "validation": read("VALIDATION.json")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    execute() if args.execute else prepare()
