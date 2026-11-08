"""Independent two-slot per-root generation, with immutable call artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
V8_PATH = PARENT / "h4_crossroot_multiselect_batch_v8/generate.py"
V8_SHA256 = "94db9ed773eae8f1318905f796f9ddbca973bddf225c2c69c7d7041f22026ac2"
V4_PATH = PARENT / "h4_crossroot_multiselect_batch_v4/generate.py"
V4_SHA256 = "a1e65ddacfe877db12d40e131208f7d65bc4bfaaefacbd01a5b84f18f6a94e5e"
SELECTED_PATH = PARENT / "h4_crossroot_multiselect_solver_report_two_roots_v1/SELECTED_QUESTIONS.json"
SELECTED_SHA256 = "08c36536c664bc2394f9e4f96aa3d25bdab6261e96240e4d3b17227ad36cf0e1"
SLOTS = (2, 3)
LABELS = tuple("ABCDEF")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dependency():
    if (sha(V8_PATH.read_bytes()) != V8_SHA256 or sha(V4_PATH.read_bytes()) != V4_SHA256 or
            sha(SELECTED_PATH.read_bytes()) != SELECTED_SHA256):
        raise ValueError("frozen_dependency_changed")
    spec = importlib.util.spec_from_file_location("v12_frozen_v8", V8_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    v4_spec = importlib.util.spec_from_file_location("v12_frozen_v4", V4_PATH)
    assert v4_spec and v4_spec.loader
    v4 = importlib.util.module_from_spec(v4_spec)
    v4_spec.loader.exec_module(v4)
    module.generation_view = v4.generation_view
    return module, module.runner()


def slot_id(root_id: str, slot: int) -> str:
    return f"{root_id.rsplit('-', 1)[-1]}-qwen36-multi-{slot:02d}"


def profile(batch) -> dict:
    result = batch.profile()
    result["profile_id"] = HERE.name
    return result


def selected_questions(root_id: str) -> list[dict]:
    return [row for row in json.loads(SELECTED_PATH.read_text())["questions"]
            if row["root_id"] == root_id]


def prior_questions(root_id: str, slot: int) -> list[dict]:
    questions = selected_questions(root_id)
    if slot == 3:
        path = HERE / root_id / "slot-02/RESPONSE.json"
        if not path.exists():
            raise ValueError("slot_02_response_required_before_slot_03")
        response = json.loads(path.read_text())
        try:
            parsed = json.loads(response["raw_text"])
            additional = parsed.get("questions", [])
        except (TypeError, ValueError, AttributeError):
            additional = []
        if not isinstance(additional, list):
            additional = []
        questions += [row for row in additional if isinstance(row, dict)]
    return questions


def dedup_view(questions: list[dict]) -> list[dict]:
    return [{"candidate_id": row.get("candidate_id"), "gold": row.get("gold"),
             "options": row.get("options"), "anchors": sorted({
                 item.get("turn_ref") for item in (row.get("option_evidence") or {}).values()
                 if isinstance(item, dict) and isinstance(item.get("turn_ref"), str)})}
            for row in questions]


def prompt_for(root_id: str, slot: int, source: dict, instructions: str, prior: list[dict], v8) -> str:
    text = instructions.replace("{ROOT_ID}", root_id).replace(
        "{CANDIDATE_ID}", slot_id(root_id, slot))
    context = {"previous_questions_avoid_repeating": dedup_view(prior),
               "source": v8.generation_view(source)}
    return text + json.dumps(context, ensure_ascii=False, indent=2) + "\n"


def check_candidate(row: object, root_id: str, slot: int, turns: list[dict],
                    prior: list[dict], v8) -> list[str]:
    if not isinstance(row, dict):
        return ["candidate_not_object"]
    test_row = dict(row, candidate_id=slot_id(root_id, 1))
    errors = v8.check_candidate(test_row, root_id, turns)
    if row.get("candidate_id") != slot_id(root_id, slot):
        errors.append("candidate_id_invalid")
    gold = row.get("gold")
    if v8.valid_gold(gold):
        pattern = "".join(sorted(gold))
        if len(gold) < 6 and pattern == "ABCDEF"[:len(gold)]:
            errors.append("gold_prefix_forbidden")
        if any(isinstance(old.get("gold"), list) and len(old["gold"]) == len(gold) and
               set(old["gold"]) == set(gold) for old in prior):
            errors.append("gold_position_pattern_repeated_at_same_cardinality")
    options = row.get("options")
    if isinstance(options, dict) and set(options) == set(LABELS):
        normalized = {str(value).strip().casefold() for value in options.values()}
        for old in prior:
            old_options = old.get("options")
            if isinstance(old_options, dict) and set(old_options) == set(LABELS):
                overlap = len(normalized & {str(value).strip().casefold()
                                            for value in old_options.values()})
                if overlap >= 3:
                    errors.append("three_or_more_prior_options_reused")
                    break
    return errors


def inputs(root_id: str, slot: int):
    v8, batch = dependency()
    ordering = batch.ordering
    if root_id not in batch.ROOTS or slot not in SLOTS:
        raise ValueError("unknown_root_or_slot")
    directory = HERE / root_id / f"slot-{slot:02d}"
    manifest = ordering.read_json(directory / "PREPARED.json")
    if manifest.get("script_sha256") != sha(Path(__file__).read_bytes()) or \
       manifest.get("instructions_sha256") != sha((HERE / "instructions.txt").read_bytes()) or \
       manifest.get("selected_sha256") != SELECTED_SHA256:
        raise ValueError("batch_freeze_changed")
    row = ordering.source_manifest()[root_id]
    raw, source = ordering.checked_source(root_id, row, frozen=False)
    if raw != (directory / "source.json").read_bytes() or manifest["source_sha256"] != sha(raw):
        raise ValueError("source_freeze_changed")
    prior = prior_questions(root_id, slot)
    if sha(ordering.canonical(dedup_view(prior))) != manifest["prior_sha256"]:
        raise ValueError("prior_questions_changed")
    prompt = prompt_for(root_id, slot, source,
                        (HERE / "instructions.txt").read_text(), prior, v8)
    request = ordering.read_json(directory / "request.json")
    if prompt != (directory / "prompt.txt").read_text() or \
       sha(prompt.encode()) != manifest["prompt_sha256"] or \
       request != ordering.build_payload(profile(batch), prompt) or \
       sha(ordering.canonical(request)) != manifest["payload_sha256"]:
        raise ValueError("request_freeze_changed")
    return v8, batch, directory, manifest, source, prior, prompt, request


def prepare(root_id: str, slot: int) -> dict:
    v8, batch = dependency()
    if root_id not in batch.ROOTS or slot not in SLOTS:
        raise ValueError("unknown_root_or_slot")
    ordering = batch.ordering
    prior = prior_questions(root_id, slot)
    row = ordering.source_manifest()[root_id]
    raw, source = ordering.checked_source(root_id, row, frozen=False)
    prompt = prompt_for(root_id, slot, source,
                        (HERE / "instructions.txt").read_text(), prior, v8)
    request = ordering.build_payload(profile(batch), prompt)
    directory = HERE / root_id / f"slot-{slot:02d}"
    if directory.exists():
        raise FileExistsError(f"slot_exists_no_overwrite:{directory}")
    directory.mkdir(parents=True)
    manifest = {"version": HERE.name, "root_id": root_id, "slot": slot,
                "candidate_id": slot_id(root_id, slot), "model": batch.MODEL,
                "source_sha256": sha(raw), "selected_sha256": SELECTED_SHA256,
                "prior_sha256": sha(ordering.canonical(dedup_view(prior))),
                "instructions_sha256": sha((HERE / "instructions.txt").read_bytes()),
                "script_sha256": sha(Path(__file__).read_bytes()),
                "prompt_sha256": sha(prompt.encode()),
                "payload_sha256": sha(ordering.canonical(request)),
                "maximum_candidates": 1, "maximum_http_attempts": 1,
                "status": "PREPARED_NO_API_CALL", "solver_calls": 0, "training_started": False}
    ordering.write_new(directory / "source.json", raw.decode(), raw=True)
    ordering.write_new(directory / "prompt.txt", prompt, raw=True)
    ordering.write_new(directory / "request.json", request)
    ordering.write_new(directory / "PREPARED.json", manifest)
    return manifest


def validate(root_id: str, slot: int) -> dict:
    v8, batch, directory, manifest, source, prior, _, _ = inputs(root_id, slot)
    ordering = batch.ordering
    response = ordering.read_json(directory / "RESPONSE.json")
    if response.get("source_sha256") != manifest["source_sha256"] or \
       response.get("payload_sha256") != manifest["payload_sha256"]:
        raise ValueError("response_binding_changed")
    metadata = response.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("response_metadata_invalid")
    try:
        parsed = json.loads(response["raw_text"])
    except (TypeError, ValueError):
        parsed = None
    questions = parsed.get("questions") if isinstance(parsed, dict) else None
    abstentions = parsed.get("abstentions") if isinstance(parsed, dict) else None
    global_errors = []
    if not isinstance(parsed, dict) or set(parsed) != {"questions", "abstentions"}:
        global_errors.append("top_level_schema_mismatch")
    if not isinstance(questions, list):
        questions = []
        global_errors.append("questions_array_missing")
    if len(questions) > 1:
        global_errors.append("candidate_count_exceeds_one")
    if not isinstance(abstentions, list) or any(not isinstance(x, str) for x in abstentions):
        global_errors.append("abstentions_array_invalid")
    elif not questions and not abstentions:
        global_errors.append("missing_abstention_reason")
    if metadata.get("returned_model") != batch.MODEL:
        global_errors.append("returned_model_mismatch")
    if metadata.get("finish_reason") == "length":
        global_errors.append("response_truncated")
    checks = [{"candidate_id": row.get("candidate_id") if isinstance(row, dict) else None,
               "errors": check_candidate(row, root_id, slot, source["turns"], prior, v8)}
              for row in questions]
    valid = [row for row, check in zip(questions, checks) if not check["errors"]]
    if global_errors:
        valid = []
    status = ("MECHANICAL_REJECTED" if global_errors or (questions and not valid) else
              "MECHANICAL_CHECKS_ONLY_CONTENT_AUDIT_PENDING" if questions else
              "NO_CANDIDATE_ABSTAINED")
    provenance = {"author": "REMOTE_LARGE_MODEL", "provider": "siliconflow",
                  "requested_model": batch.MODEL, "returned_model": metadata.get("returned_model"),
                  "response_id": (metadata.get("raw_response") or {}).get("id"),
                  "root_id": root_id, "slot": slot, "source_sha256": manifest["source_sha256"],
                  "prompt_sha256": manifest["prompt_sha256"],
                  "payload_sha256": manifest["payload_sha256"],
                  "model_questions_modified_by_codex": False,
                  "semantic_gold_verified": False, "solver_evaluated": False,
                  "selected_for_training": False, "training_started": False}
    result = {"status": status, "root_id": root_id, "slot": slot,
              "candidate_count": len(questions), "mechanically_valid_count": len(valid),
              "global_errors": global_errors, "checks": checks, "provenance": provenance}
    ordering.write_new(directory / "PARSED_RESPONSE.json", parsed)
    ordering.write_new(directory / "VALIDATED_CANDIDATES.json", {
        "status": status, "provenance": provenance, "questions": valid})
    ordering.write_new(directory / "VALIDATION.json", result)
    return result


def execute(root_id: str, slot: int) -> dict:
    _, batch, directory, manifest, _, _, prompt, payload = inputs(root_id, slot)
    ordering = batch.ordering
    ordering.write_new(directory / "ATTEMPT_RESERVED.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "model": batch.MODEL,
        "payload_sha256": manifest["payload_sha256"], "maximum_http_attempts": 1})

    def on_attempt(actual: dict) -> None:
        if actual != payload:
            raise ValueError("actual_payload_changed")
        ordering.write_new(directory / "HTTP_ATTEMPT.json", {
            "attempt_number": 1, "payload_sha256": sha(ordering.canonical(actual)),
            "model": actual["model"], "max_tokens": actual["max_tokens"]})

    try:
        response = ordering.credential_client(profile(batch), on_http_attempt=on_attempt).complete(
            prompt=prompt, max_tokens=batch.OUTPUT_LIMIT, json_object=True)
    except Exception as error:
        result = {"status": "ERROR_NO_RETRY", "root_id": root_id, "slot": slot,
                  "error_type": type(error).__name__, "metadata": getattr(error, "metadata", None)}
        ordering.write_new(directory / "ERROR.json", result)
        return result
    ordering.write_new(directory / "RESPONSE.json", {
        "source_sha256": manifest["source_sha256"],
        "payload_sha256": manifest["payload_sha256"],
        "raw_text": response.raw_text, "metadata": response.metadata})
    return validate(root_id, slot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "preflight", "execute", "validate"))
    parser.add_argument("root_id")
    parser.add_argument("slot", type=int, choices=SLOTS)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.root_id, args.slot)
    elif args.action == "preflight":
        _, _, _, manifest, _, prior, prompt, _ = inputs(args.root_id, args.slot)
        result = {"status": "FROZEN_INPUTS_VERIFIED_NO_API_CALL", "root_id": args.root_id,
                  "slot": args.slot, "prior_count": len(prior),
                  "prompt_bytes": len(prompt.encode()),
                  "payload_sha256": manifest["payload_sha256"]}
    elif args.action == "execute":
        result = execute(args.root_id, args.slot)
    else:
        result = validate(args.root_id, args.slot)
    print(json.dumps(result, ensure_ascii=False))
