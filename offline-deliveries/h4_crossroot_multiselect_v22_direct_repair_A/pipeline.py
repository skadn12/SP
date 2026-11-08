"""One frozen root-isolated request to repair two v21 claim drafts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DELIVERIES = HERE.parent
ROOT = "cc-train-episode-197635"
MODEL = "Qwen/Qwen3.6-35B-A3B"
LABELS = tuple("ABCDEF")
OLD_IDS = ("197635-qwen36-multi-22", "197635-qwen36-multi-23")
NEW_IDS = ("197635-qwen36-multi-25", "197635-qwen36-multi-26")
V21 = DELIVERIES / "h4_crossroot_multiselect_batch_v21_claim_staged" / ROOT / "batch-01"
V16 = DELIVERIES / "h4_crossroot_multiselect_batch_v16/generate.py"
V16_SHA = "035a3aef0682dd1426bceb61847bac0d1e7c8aec9bb99350a6c86a7043c0b164"
SELECTED_PATHS = (
    "h4_crossroot_multiselect_solver_report_two_roots_v1/SELECTED_QUESTIONS.json",
    "h4_crossroot_multiselect_codex_solver_v15_report/SELECTED_QUESTIONS.json",
    "h4_crossroot_multiselect_v17_254852_solver_selection_v1/SELECTED_QUESTIONS.json",
    "h4_crossroot_multiselect_v20_136202_solver_selection_v1/SELECTED_QUESTIONS.json",
)
INPUT_PATHS = {
    "source": V21 / "source.json",
    "claim_input": V21 / "claim_input.json",
    "draft": V21 / "CLAIMS_DRAFT.json",
    "review": V21 / "CONTENT_REVIEW.json",
    "claim_checks": V21 / "CLAIM_CHECKS.json",
}
OUT = HERE / ROOT / "batch-01"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, value: object, *, raw: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value if raw else json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def dependencies():
    if sha(V16.read_bytes()) != V16_SHA:
        raise ValueError("v16_dependency_changed")
    spec = importlib.util.spec_from_file_location("v22a_frozen_v16", V16)
    assert spec and spec.loader
    v16 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v16)
    _, v8, batch = v16.base()
    return v8, batch, batch.ordering


def materials():
    v8, batch, ordering = dependencies()
    hashes = {str(path.relative_to(DELIVERIES)): sha(path.read_bytes())
              for path in (*INPUT_PATHS.values(), *(DELIVERIES / p for p in SELECTED_PATHS))}
    source, claim_input, draft, review, checks = (read(INPUT_PATHS[key]) for key in INPUT_PATHS)
    if source.get("root_id") != ROOT or claim_input.get("root_id") != ROOT or draft.get("root_id") != ROOT:
        raise ValueError("source_root_mismatch")
    if review.get("draft_sha256") != sha(canonical(draft)) or checks.get("fatal_errors"):
        raise ValueError("v21_review_binding_invalid")
    slots = {row["candidate_id"]: row for row in draft["question_slots"]}
    decisions = {(row["candidate_id"], row["label"]): row for row in review["claim_decisions"]}
    if len(decisions) != len(review["claim_decisions"]):
        raise ValueError("duplicate_review_decision")
    selected = []
    for relative in SELECTED_PATHS:
        data = read(DELIVERIES / relative)
        selected.extend(row for row in data["questions"] if row.get("root_id") == ROOT)
    if len({row["candidate_id"] for row in selected}) != len(selected):
        raise ValueError("selected_candidate_duplicate")
    targets = []
    for old_id, new_id in zip(OLD_IDS, NEW_IDS):
        old = slots[old_id]
        if set(old["claims"]) != set(LABELS):
            raise ValueError("old_slot_incomplete")
        notes = {label: decisions[(old_id, label)] for label in LABELS}
        targets.append({"old_candidate_id": old_id, "new_candidate_id": new_id,
                        "old_claims": old["claims"], "review": notes,
                        "locked_labels": [label for label in LABELS if notes[label]["approved"]]})
    if [target["locked_labels"] for target in targets] != [["C"], ["C", "D"]]:
        raise ValueError("reviewed_lock_set_changed")
    prompt_input = {"root_id": ROOT, "source_scope": "R02",
                    "source_sha256": hashes[str(INPUT_PATHS["source"].relative_to(DELIVERIES))],
                    "selected_questions_avoid_repeating": [
                        {"candidate_id": row["candidate_id"], "topic": row["topic"],
                         "options": row["options"], "option_evidence": row["option_evidence"]}
                        for row in selected],
                    "available_turns": claim_input["available_turns"],
                    "targets": targets}
    instructions = (HERE / "instructions.txt").read_text(encoding="utf-8")
    for key, value in (("{ROOT_ID}", ROOT), ("{ID_ONE}", NEW_IDS[0]), ("{ID_TWO}", NEW_IDS[1])):
        instructions = instructions.replace(key, value)
    prompt = instructions + json.dumps(prompt_input, ensure_ascii=False, indent=2) + "\n"
    return v8, batch, ordering, hashes, source, prompt_input, prompt


def request_data():
    v8, batch, ordering, hashes, source, prompt_input, prompt = materials()
    v16_profile = batch.profile()
    v16_profile["profile_id"] = HERE.name
    v16_profile["output_limit"] = 12288
    v16_profile["request_parameters"]["max_tokens"] = 12288
    request = ordering.build_payload(v16_profile, prompt)
    if request.get("model") != MODEL or request.get("max_tokens") != 12288:
        raise ValueError("model_or_token_limit_changed")
    return v8, ordering, v16_profile, hashes, source, prompt_input, prompt, request


def expected_hashes(hashes, prompt_input, prompt, request):
    return {"input_file_sha256": hashes,
            "instructions_sha256": sha((HERE / "instructions.txt").read_bytes()),
            "script_sha256": sha(Path(__file__).read_bytes()),
            "prompt_input_sha256": sha(canonical(prompt_input)),
            "prompt_sha256": sha(prompt.encode()),
            "payload_sha256": sha(canonical(request))}


def prepare():
    if OUT.exists():
        raise FileExistsError(f"no_overwrite:{OUT}")
    _, _, _, hashes, source, prompt_input, prompt, request = request_data()
    manifest = {"version": HERE.name, "root_id": ROOT, "model": MODEL,
                **expected_hashes(hashes, prompt_input, prompt, request),
                "old_candidate_ids": list(OLD_IDS), "candidate_ids": list(NEW_IDS),
                "maximum_questions": 2, "maximum_http_attempts": 1,
                "status": "PREPARED_NO_API_CALL", "solver_calls": 0, "training_started": False}
    write_new(OUT / "source.json", source)
    write_new(OUT / "repair_input.json", prompt_input)
    write_new(OUT / "PROMPT.txt", prompt, raw=True)
    write_new(OUT / "REQUEST.json", request)
    write_new(OUT / "PREPARED.json", manifest)
    return {"status": manifest["status"], "root_id": ROOT,
            "maximum_questions": 2, "payload_sha256": manifest["payload_sha256"]}


def frozen():
    data = request_data()
    _, _, _, hashes, source, prompt_input, prompt, request = data
    manifest = read(OUT / "PREPARED.json")
    expected = expected_hashes(hashes, prompt_input, prompt, request)
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("frozen_hash_changed")
    if (source != read(OUT / "source.json") or prompt_input != read(OUT / "repair_input.json")
            or prompt != (OUT / "PROMPT.txt").read_text(encoding="utf-8")
            or request != read(OUT / "REQUEST.json")):
        raise ValueError("frozen_input_changed")
    return (*data, manifest)


def validate():
    v8, _, _, _, source, prompt_input, _, _, manifest = frozen()
    response = read(OUT / "RESPONSE.json")
    if response.get("payload_sha256") != manifest["payload_sha256"]:
        raise ValueError("response_binding_changed")
    try:
        parsed = json.loads(response["raw_text"])
    except (KeyError, TypeError, ValueError):
        parsed = None
    fatal = []
    if not isinstance(parsed, dict) or set(parsed) != {"questions", "abstentions"}:
        fatal.append("top_level_schema_invalid")
        parsed = {"questions": [], "abstentions": []}
    rows, abstentions = parsed["questions"], parsed["abstentions"]
    if not isinstance(rows, list) or len(rows) > 2:
        fatal.append("question_count_invalid")
        rows = rows if isinstance(rows, list) else []
    if not isinstance(abstentions, list) or any(not isinstance(x, str) for x in abstentions):
        fatal.append("abstentions_invalid")
    if len(rows) < 2 and not abstentions:
        fatal.append("missing_abstention_reason")
    metadata = response.get("metadata", {})
    if metadata.get("returned_model") != MODEL or metadata.get("finish_reason") == "length":
        fatal.append("returned_model_mismatch_or_truncated")
    available = {turn["turn_ref"]: turn for turn in prompt_input["available_turns"]}
    selected_texts = {x.strip().casefold() for row in prompt_input["selected_questions_avoid_repeating"]
                      for x in row["options"].values()}
    target_map = {target["new_candidate_id"]: target for target in prompt_input["targets"]}
    seen_ids, checks = set(), []
    for row in rows:
        errors = []
        cid = row.get("candidate_id") if isinstance(row, dict) else None
        if cid not in target_map or cid in seen_ids:
            errors.append("unknown_or_duplicate_candidate_id")
        seen_ids.add(cid)
        if isinstance(row, dict):
            errors.extend(v8.check_candidate(dict(row, candidate_id=f"{ROOT[-6:]}-qwen36-multi-01"), ROOT, source["turns"]))
            gold = row.get("gold")
            if isinstance(gold, list) and len(gold) < 6 and gold == list(LABELS[:len(gold)]):
                errors.append("gold_prefix_forbidden")
            evidence = row.get("option_evidence")
            if isinstance(evidence, dict) and set(evidence) == set(LABELS):
                refs = [evidence[label].get("turn_ref") if isinstance(evidence[label], dict) else None
                        for label in LABELS]
                if len(set(refs)) != 6 or any(ref not in available for ref in refs):
                    errors.append("six_distinct_available_r02_turns_required")
            options = row.get("options")
            if isinstance(options, dict) and set(options) == set(LABELS):
                if any(x.strip().casefold() in selected_texts for x in options.values() if isinstance(x, str)):
                    errors.append("selected_option_text_reused")
            target = target_map.get(cid)
            if target and isinstance(options, dict) and isinstance(evidence, dict):
                for label in target["locked_labels"]:
                    claim = target["old_claims"][label]
                    expected = {"turn_ref": claim["turn_ref"], "quote": claim["quote"],
                                "relation": claim["relation"], "reason": claim["rationale"]}
                    if options.get(label) != claim["proposition"] or evidence.get(label) != expected:
                        errors.append(f"{label}:reviewed_claim_changed")
        else:
            errors.append("candidate_not_object")
        checks.append({"candidate_id": cid, "errors": errors, "mechanically_valid": not errors})
    result = {"status": "MECHANICAL_CHECKS_ONLY_CONTENT_REVIEW_PENDING",
              "root_id": ROOT, "model": MODEL, "fatal_errors": fatal, "checks": checks,
              "mechanically_valid_count": 0 if fatal else sum(not row["errors"] for row in checks),
              "semantic_review_required": True, "solver_calls": 0, "training_started": False}
    write_new(OUT / "PARSED_RESPONSE.json", parsed)
    write_new(OUT / "VALIDATION.json", result)
    return result


def execute():
    _, ordering, profile, _, _, _, prompt, request, manifest = frozen()
    write_new(OUT / "ATTEMPT_RESERVED.json", {
        "at": datetime.now(timezone.utc).isoformat(), "payload_sha256": manifest["payload_sha256"],
        "model": MODEL, "maximum_http_attempts": 1})

    def on_attempt(actual):
        if actual != request:
            raise ValueError("actual_payload_changed")
        write_new(OUT / "HTTP_ATTEMPT.json", {"attempt_number": 1,
                  "payload_sha256": sha(canonical(actual)), "model": actual["model"]})

    try:
        result = ordering.credential_client(profile, on_http_attempt=on_attempt).complete(
            prompt=prompt, max_tokens=12288, json_object=True)
    except Exception as error:
        failure = {"status": "ERROR_NO_RETRY", "error_type": type(error).__name__,
                   "metadata": getattr(error, "metadata", None)}
        write_new(OUT / "ERROR.json", failure)
        return failure
    write_new(OUT / "RESPONSE.json", {"payload_sha256": manifest["payload_sha256"],
              "raw_text": result.raw_text, "metadata": result.metadata})
    return validate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "preflight", "execute", "validate"))
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare()
    elif args.action == "preflight":
        *_, manifest = frozen()
        result = {"status": "FROZEN_INPUTS_VERIFIED_NO_API_CALL", "root_id": ROOT,
                  "payload_sha256": manifest["payload_sha256"]}
    elif args.action == "execute":
        result = execute()
    else:
        result = validate()
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
