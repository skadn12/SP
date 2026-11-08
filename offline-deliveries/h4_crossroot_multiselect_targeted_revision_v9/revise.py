"""Frozen two-question model revision; mechanical validation precedes Codex review."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
V8_DIR = HERE.parent / "h4_crossroot_multiselect_batch_v8"
V8_SCRIPT = V8_DIR / "generate.py"
V8_SCRIPT_SHA256 = "94db9ed773eae8f1318905f796f9ddbca973bddf225c2c69c7d7041f22026ac2"
VERSION = HERE.name
MODEL = "Qwen/Qwen3.6-35B-A3B"
OUTPUT_LIMIT = 4096
TARGETS = {
    "cc-train-episode-136202": {
        "label": "E",
        "defect": (
            "Option E names Sarah as the message recipient, but its frozen quote says only "
            "'the message'. Revise E so it makes no claim about who receives it. "
            "Keep E true, and update only its reason and the overall answer_reason."
        ),
    },
    "cc-train-episode-254852": {
        "label": "B",
        "defect": (
            "Option B turns Robin's 'What if I bring coffee' proposal into a definite "
            "commitment. Revise B to preserve the proposal modality, without claiming "
            "she committed or eventually brought coffee. Keep B true, and update "
            "only its reason and the overall answer_reason."
        ),
    },
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def dependency():
    if sha(V8_SCRIPT.read_bytes()) != V8_SCRIPT_SHA256:
        raise ValueError("v8_dependency_changed")
    spec = importlib.util.spec_from_file_location("h4_multiselect_v8_dependency", V8_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError("v8_dependency_unavailable")
    v8 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v8)
    runtime = v8.runner()
    if runtime.MODEL != MODEL:
        raise ValueError("model_dependency_changed")
    return v8, runtime


def profile(runtime) -> dict:
    value = runtime.profile()
    value["profile_id"] = VERSION
    value["output_limit"] = OUTPUT_LIMIT
    value["request_parameters"]["max_tokens"] = OUTPUT_LIMIT
    return value


def source_inputs(root_id: str) -> tuple[bytes, bytes, dict, dict]:
    if root_id not in TARGETS:
        raise ValueError("root_not_targeted")
    batch = read_json(V8_DIR / "BATCH_PREPARED.json")
    entry = next((row for row in batch["roots"] if row["root_id"] == root_id), None)
    if not entry or entry["requested_model"] != MODEL:
        raise ValueError("v8_root_missing_or_model_changed")
    validation = read_json(V8_DIR / root_id / "VALIDATION.json")
    if validation["mechanically_valid_count"] != 1:
        raise ValueError("v8_original_not_mechanically_valid")
    source_raw = (V8_DIR / root_id / "source.json").read_bytes()
    if sha(source_raw) != entry["source_sha256"]:
        raise ValueError("v8_source_hash_changed")
    parsed_raw = (V8_DIR / root_id / "PARSED_RESPONSE.json").read_bytes()
    parsed = json.loads(parsed_raw)
    if set(parsed) != {"questions", "abstentions"} or len(parsed["questions"]) != 1:
        raise ValueError("v8_question_count_changed")
    original = parsed["questions"][0]
    if original["root_id"] != root_id or TARGETS[root_id]["label"] not in original["gold"]:
        raise ValueError("v8_original_identity_changed")
    source = json.loads(source_raw)
    return source_raw, parsed_raw, source, original


def prompt_for(root_id: str, original: dict, source: dict) -> str:
    template = (HERE / "instructions.txt").read_text(encoding="utf-8")
    turns = [turn for turn in source["turns"] if turn["turn_ref"].startswith("R02:")]
    if not turns:
        raise ValueError("r02_turns_missing")
    return (template.replace("{TARGET_LABEL}", TARGETS[root_id]["label"])
            .replace("{DEFECT}", TARGETS[root_id]["defect"])
            .replace("{ORIGINAL_QUESTION}", json.dumps(original, ensure_ascii=False, indent=2))
            .replace("{R02_TURNS}", json.dumps(turns, ensure_ascii=False, indent=2)))


def changed_paths(before: object, after: object, path: str = "") -> set[str]:
    if isinstance(before, dict) and isinstance(after, dict):
        if set(before) != set(after):
            return {path + ".<keys>"}
        return set().union(*(changed_paths(before[key], after[key],
                                           f"{path}.{key}" if path else key)
                             for key in before))
    if before != after:
        return {path}
    return set()


def check_revision(revised: object, original: dict, root_id: str,
                   turns: list[dict], validator) -> list[str]:
    if not isinstance(revised, dict):
        return ["revised_question_not_object"]
    label = TARGETS[root_id]["label"]
    allowed = {f"options.{label}", f"option_evidence.{label}.reason", "answer_reason"}
    actual = changed_paths(original, revised)
    errors = []
    if actual != allowed:
        errors.append("changed_fields_not_exactly_allowed:" + ",".join(sorted(actual)))
    errors.extend(validator(revised, root_id, turns))
    return errors


def prepare() -> None:
    _, runtime = dependency()
    if (HERE / "BATCH_PREPARED.json").exists():
        raise FileExistsError("batch_already_prepared")
    instructions = (HERE / "instructions.txt").read_bytes()
    planned = []
    for root_id in TARGETS:
        if (HERE / root_id).exists():
            raise FileExistsError("root_directory_exists:" + root_id)
        source_raw, parsed_raw, source, original = source_inputs(root_id)
        prompt = prompt_for(root_id, original, source)
        payload = runtime.ordering.build_payload(profile(runtime), prompt)
        planned.append((root_id, source_raw, parsed_raw, original, prompt, payload))
    rows = []
    for root_id, source_raw, parsed_raw, original, prompt, payload in planned:
        directory = HERE / root_id
        directory.mkdir()
        runtime.write_new(directory / "source.json", source_raw.decode("utf-8"), raw=True)
        runtime.write_new(directory / "original_question.json", original)
        runtime.write_new(directory / "prompt.txt", prompt, raw=True)
        runtime.write_new(directory / "request.json", payload)
        rows.append({
            "root_id": root_id, "target_label": TARGETS[root_id]["label"],
            "v8_parsed_sha256": sha(parsed_raw), "source_sha256": sha(source_raw),
            "original_question_sha256": sha(canonical(original)),
            "prompt_sha256": sha(prompt.encode()), "payload_sha256": sha(canonical(payload)),
            "requested_model": MODEL, "maximum_http_attempts": 1,
            "status": "PREPARED_NO_API_CALL",
        })
    runtime.write_new(HERE / "BATCH_PREPARED.json", {
        "version": VERSION, "requested_model": MODEL, "profile": profile(runtime),
        "instructions_sha256": sha(instructions), "script_sha256": sha(Path(__file__).read_bytes()),
        "v8_script_sha256": V8_SCRIPT_SHA256, "roots": rows,
        "model_calls": 0, "solver_calls": 0, "training_started": False,
    })
    print(json.dumps({"status": "PREPARED_NO_API_CALL", "roots": list(TARGETS),
                      "model": MODEL}))


def frozen_inputs(root_id: str):
    _, runtime = dependency()
    batch = read_json(HERE / "BATCH_PREPARED.json")
    if (batch["version"] != VERSION or batch["profile"] != profile(runtime) or
            batch["script_sha256"] != sha(Path(__file__).read_bytes()) or
            batch["instructions_sha256"] != sha((HERE / "instructions.txt").read_bytes())):
        raise ValueError("v9_freeze_changed")
    entry = next((row for row in batch["roots"] if row["root_id"] == root_id), None)
    if entry is None or root_id not in TARGETS:
        raise ValueError("root_not_prepared")
    source_raw, parsed_raw, source, original = source_inputs(root_id)
    directory = HERE / root_id
    if (source_raw != (directory / "source.json").read_bytes() or
            sha(parsed_raw) != entry["v8_parsed_sha256"] or
            read_json(directory / "original_question.json") != original or
            sha(canonical(original)) != entry["original_question_sha256"]):
        raise ValueError("frozen_source_or_question_changed")
    prompt = (directory / "prompt.txt").read_text(encoding="utf-8")
    payload = read_json(directory / "request.json")
    if (prompt != prompt_for(root_id, original, source) or
            sha(prompt.encode()) != entry["prompt_sha256"] or
            payload != runtime.ordering.build_payload(profile(runtime), prompt) or
            sha(canonical(payload)) != entry["payload_sha256"]):
        raise ValueError("frozen_request_changed")
    return runtime, entry, source, original, prompt, payload


def preflight() -> None:
    rows = []
    for root_id in TARGETS:
        _, entry, _, original, prompt, payload = frozen_inputs(root_id)
        directory = HERE / root_id
        rows.append({"root_id": root_id, "target_label": entry["target_label"],
                     "original_gold": original["gold"], "prompt_bytes": len(prompt.encode()),
                     "request_model": payload["model"],
                     "http_attempts": int((directory / "HTTP_ATTEMPT.json").exists()),
                     "response_present": (directory / "RESPONSE.json").exists()})
    print(json.dumps({"status": "FROZEN_INPUTS_VERIFIED_NO_API_CALL", "roots": rows}))


def validate(root_id: str) -> dict:
    runtime, entry, source, original, _, _ = frozen_inputs(root_id)
    directory = HERE / root_id
    response = read_json(directory / "RESPONSE.json")
    if (response.get("source_sha256") != entry["source_sha256"] or
            response.get("original_question_sha256") != entry["original_question_sha256"] or
            response.get("payload_sha256") != entry["payload_sha256"]):
        raise ValueError("response_binding_changed")
    metadata = response.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("response_metadata_invalid")
    try:
        parsed = json.loads(response["raw_text"])
    except (TypeError, ValueError):
        parsed = None
    errors = []
    if not isinstance(parsed, dict) or set(parsed) != {"revised_question", "abstentions"}:
        errors.append("top_level_schema_mismatch")
        parsed = {"revised_question": None, "abstentions": []}
    revised, abstentions = parsed["revised_question"], parsed["abstentions"]
    if not isinstance(abstentions, list) or any(not isinstance(item, str) for item in abstentions):
        errors.append("abstentions_invalid")
    elif revised is None and not abstentions:
        errors.append("abstention_reason_missing")
    elif revised is not None and abstentions:
        errors.append("success_with_abstention_reason")
    if metadata.get("returned_model") != MODEL:
        errors.append("returned_model_mismatch")
    if metadata.get("finish_reason") == "length":
        errors.append("response_truncated")
    if revised is not None:
        errors.extend(check_revision(revised, original, root_id, source["turns"],
                                     runtime.check_candidate))
    valid = revised is not None and not errors
    status = ("MECHANICALLY_VALID_MODEL_REVISION_CONTENT_AUDIT_PENDING" if valid else
              "MODEL_ABSTAINED" if revised is None and not errors else "MECHANICAL_REJECTED")
    result = {
        "status": status, "root_id": root_id, "target_label": entry["target_label"],
        "mechanically_valid_count": int(valid), "errors": errors,
        "abstentions": abstentions if isinstance(abstentions, list) else [],
        "source_author": "Qwen/Qwen3.6-35B-A3B",
        "revising_author": "Qwen/Qwen3.6-35B-A3B",
        "codex_role": "MECHANICAL_VALIDATION_AND_SEPARATE_CONTENT_AUDIT",
        "source_sha256": entry["source_sha256"],
        "original_question_sha256": entry["original_question_sha256"],
        "prompt_sha256": entry["prompt_sha256"],
        "payload_sha256": entry["payload_sha256"],
        "returned_model": metadata.get("returned_model"),
        "response_id": (metadata.get("raw_response") or {}).get("id"),
        "semantic_gold_verified": False, "solver_evaluated": False,
        "training_started": False,
    }
    runtime.write_new(directory / "PARSED_RESPONSE.json", parsed)
    runtime.write_new(directory / "VALIDATION.json", result)
    if valid:
        runtime.write_new(directory / "MECHANICALLY_VALID_REVISION.json", {
            "status": status, "question": revised,
            "source_author": MODEL, "revising_author": MODEL,
            "codex_role": "MECHANICAL_VALIDATION_ONLY",
            "content_audited": False, "ready_for_evaluation": False,
        })
    return result


def execute(root_id: str) -> None:
    runtime, entry, _, _, prompt, payload = frozen_inputs(root_id)
    directory = HERE / root_id
    runtime.write_new(directory / "ATTEMPT_RESERVED.json", {
        "status": "RESERVED_ONE_MODEL_REVISION_CALL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "requested_model": MODEL, "maximum_http_attempts": 1,
        "payload_sha256": entry["payload_sha256"],
    })

    def on_attempt(actual: dict) -> None:
        if actual != payload:
            raise ValueError("actual_payload_changed")
        runtime.write_new(directory / "HTTP_ATTEMPT.json", {
            "attempt_number": 1, "payload_sha256": sha(canonical(actual)),
            "model": actual["model"], "max_tokens": actual["max_tokens"],
        })

    try:
        response = runtime.ordering.credential_client(profile(runtime), on_http_attempt=on_attempt).complete(
            prompt=prompt, max_tokens=OUTPUT_LIMIT, json_object=True,
        )
    except Exception as error:
        runtime.write_new(directory / "ERROR.json", {
            "status": "ERROR_NO_RETRY", "error_type": type(error).__name__,
            "metadata": getattr(error, "metadata", None),
        })
        print(json.dumps({"root_id": root_id, "status": "ERROR_NO_RETRY",
                          "error_type": type(error).__name__}))
        return
    runtime.write_new(directory / "RESPONSE.json", {
        "source_sha256": entry["source_sha256"],
        "original_question_sha256": entry["original_question_sha256"],
        "payload_sha256": entry["payload_sha256"],
        "raw_text": response.raw_text, "metadata": response.metadata,
    })
    print(json.dumps(validate(root_id), ensure_ascii=False))


def summary() -> None:
    _, runtime = dependency()
    rows = []
    for root_id in TARGETS:
        path = HERE / root_id / "VALIDATION.json"
        rows.append(read_json(path) if path.exists() else {
            "root_id": root_id, "status": "NO_VALIDATION"})
    result = {"version": VERSION, "requested_model": MODEL,
              "root_count": len(rows),
              "mechanically_valid_count": sum(row.get("mechanically_valid_count", 0) for row in rows),
              "roots": rows, "solver_calls": 0, "training_started": False}
    if all(row["status"] != "NO_VALIDATION" for row in rows):
        runtime.write_new(HERE / "BATCH_VALIDATION.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--execute", choices=tuple(TARGETS))
    actions.add_argument("--validate", choices=tuple(TARGETS))
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
