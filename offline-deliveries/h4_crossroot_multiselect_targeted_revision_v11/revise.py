"""One-call model revision of option C from the frozen v10 candidate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
V10_DIR = HERE.parent / "h4_crossroot_multiselect_targeted_revision_v10"
V10_SCRIPT = V10_DIR / "recheck.py"
ROOT_ID = "cc-train-episode-136202"
MODEL = "Qwen/Qwen3.6-35B-A3B"
OUTPUT_LIMIT = 4096


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dependency():
    spec = importlib.util.spec_from_file_location("h4_multiselect_v10_dependency", V10_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError("v10_dependency_unavailable")
    v10 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v10)
    summary, pool = v10.build()
    if (summary != read_json(V10_DIR / "BATCH_RECHECK.json") or
            pool != read_json(V10_DIR / "CANDIDATES_PENDING_AUDIT.json") or
            summary["mechanically_valid_count"] != 2):
        raise ValueError("v10_recheck_changed")
    v9, _ = v10.dependency()
    runtime, _, source, _, _, _ = v9.frozen_inputs(ROOT_ID)
    original = next((row for row in pool["questions"] if row["root_id"] == ROOT_ID), None)
    if original is None or original["gold"] != ["A", "C", "E", "F"]:
        raise ValueError("v10_original_identity_changed")
    return v9, runtime, source, original


def profile(runtime) -> dict:
    value = runtime.profile()
    value["profile_id"] = HERE.name
    value["output_limit"] = OUTPUT_LIMIT
    value["request_parameters"]["max_tokens"] = OUTPUT_LIMIT
    return value


def prompt_for(original: dict, source: dict) -> str:
    template = (HERE / "instructions.txt").read_text(encoding="utf-8")
    turns = [turn for turn in source["turns"] if turn["turn_ref"].startswith("R02:")]
    if not turns:
        raise ValueError("r02_turns_missing")
    return (template.replace("{ORIGINAL_QUESTION}", json.dumps(original, ensure_ascii=False, indent=2))
            .replace("{R02_TURNS}", json.dumps(turns, ensure_ascii=False, indent=2)))


def check_revision(revised: object, original: dict, turns: list[dict], validator, v9) -> list[str]:
    if not isinstance(revised, dict):
        return ["revised_question_not_object"]
    changed = v9.changed_paths(original, revised)
    required = {"options.C"}
    allowed = required | {"answer_reason", "option_evidence.C.reason"}
    errors = []
    if not required <= changed or not changed <= allowed:
        errors.append("changed_fields_outside_v11_rule:" + ",".join(sorted(changed)))
    option = revised.get("options", {}).get("C") if isinstance(revised.get("options"), dict) else None
    if isinstance(option, str) and "therapy" in option.casefold():
        errors.append("unsupported_therapy_claim_remains")
    errors.extend(validator(revised, ROOT_ID, turns))
    return errors


def prepare() -> None:
    v9, runtime, source, original = dependency()
    if (HERE / "BATCH_PREPARED.json").exists() or (HERE / ROOT_ID).exists():
        raise FileExistsError("v11_already_prepared")
    prompt = prompt_for(original, source)
    payload = runtime.ordering.build_payload(profile(runtime), prompt)
    directory = HERE / ROOT_ID
    directory.mkdir()
    source_raw = (v9.HERE / ROOT_ID / "source.json").read_bytes()
    runtime.write_new(directory / "source.json", source_raw.decode("utf-8"), raw=True)
    runtime.write_new(directory / "original_question.json", original)
    runtime.write_new(directory / "prompt.txt", prompt, raw=True)
    runtime.write_new(directory / "request.json", payload)
    runtime.write_new(HERE / "BATCH_PREPARED.json", {
        "version": HERE.name, "root_id": ROOT_ID, "target_label": "C",
        "requested_model": MODEL, "profile": profile(runtime),
        "script_sha256": sha(Path(__file__).read_bytes()),
        "instructions_sha256": sha((HERE / "instructions.txt").read_bytes()),
        "v10_script_sha256": sha(V10_SCRIPT.read_bytes()),
        "v10_batch_recheck_sha256": sha((V10_DIR / "BATCH_RECHECK.json").read_bytes()),
        "v10_candidates_sha256": sha((V10_DIR / "CANDIDATES_PENDING_AUDIT.json").read_bytes()),
        "v9_response_sha256": sha((v9.HERE / ROOT_ID / "RESPONSE.json").read_bytes()),
        "source_sha256": sha(source_raw),
        "original_question_sha256": sha(canonical(original)),
        "prompt_sha256": sha(prompt.encode()),
        "payload_sha256": sha(canonical(payload)),
        "maximum_http_attempts": 1, "model_calls": 0,
        "solver_calls": 0, "training_started": False,
        "status": "PREPARED_NO_API_CALL",
    })
    print(json.dumps({"status": "PREPARED_NO_API_CALL", "root_id": ROOT_ID, "model": MODEL}))


def frozen_inputs():
    v9, runtime, source, original = dependency()
    batch = read_json(HERE / "BATCH_PREPARED.json")
    checks = {
        "script_sha256": sha(Path(__file__).read_bytes()),
        "instructions_sha256": sha((HERE / "instructions.txt").read_bytes()),
        "v10_script_sha256": sha(V10_SCRIPT.read_bytes()),
        "v10_batch_recheck_sha256": sha((V10_DIR / "BATCH_RECHECK.json").read_bytes()),
        "v10_candidates_sha256": sha((V10_DIR / "CANDIDATES_PENDING_AUDIT.json").read_bytes()),
        "v9_response_sha256": sha((v9.HERE / ROOT_ID / "RESPONSE.json").read_bytes()),
        "source_sha256": sha((HERE / ROOT_ID / "source.json").read_bytes()),
        "original_question_sha256": sha(canonical(original)),
    }
    if (batch["version"] != HERE.name or batch["root_id"] != ROOT_ID or
            batch["requested_model"] != MODEL or batch["profile"] != profile(runtime) or
            any(batch[key] != value for key, value in checks.items())):
        raise ValueError("v11_freeze_changed")
    directory = HERE / ROOT_ID
    if read_json(directory / "original_question.json") != original:
        raise ValueError("frozen_original_changed")
    prompt = (directory / "prompt.txt").read_text(encoding="utf-8")
    payload = read_json(directory / "request.json")
    if (prompt != prompt_for(original, source) or sha(prompt.encode()) != batch["prompt_sha256"] or
            payload != runtime.ordering.build_payload(profile(runtime), prompt) or
            sha(canonical(payload)) != batch["payload_sha256"]):
        raise ValueError("frozen_request_changed")
    return v9, runtime, batch, source, original, prompt, payload


def validate() -> dict:
    v9, runtime, batch, source, original, _, _ = frozen_inputs()
    directory = HERE / ROOT_ID
    response = read_json(directory / "RESPONSE.json")
    if (response.get("source_sha256") != batch["source_sha256"] or
            response.get("original_question_sha256") != batch["original_question_sha256"] or
            response.get("payload_sha256") != batch["payload_sha256"]):
        raise ValueError("response_binding_changed")
    metadata = response.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("response_metadata_invalid")
    raw = metadata.get("raw_response")
    if not isinstance(raw, dict) or len(raw.get("choices", [])) != 1 or \
            raw["choices"][0].get("message", {}).get("content") != response.get("raw_text"):
        raise ValueError("raw_response_content_mismatch")
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
    if metadata.get("returned_model") != MODEL or raw.get("model") != MODEL:
        errors.append("returned_model_mismatch")
    if metadata.get("finish_reason") != "stop":
        errors.append("response_not_finished")
    if revised is not None:
        errors.extend(check_revision(revised, original, source["turns"],
                                     runtime.check_candidate, v9))
    valid = revised is not None and not errors
    status = ("MECHANICALLY_VALID_CONTENT_AUDIT_PENDING" if valid else
              "MODEL_ABSTAINED" if revised is None and not errors else "MECHANICAL_REJECTED")
    result = {
        "status": status, "root_id": ROOT_ID, "target_label": "C",
        "mechanically_valid_count": int(valid), "errors": errors,
        "abstentions": abstentions if isinstance(abstentions, list) else [],
        "changed_fields": sorted(v9.changed_paths(original, revised)) if isinstance(revised, dict) else [],
        "source_author": MODEL, "revising_author": MODEL,
        "codex_role": "MECHANICAL_VALIDATION_ONLY",
        "source_sha256": batch["source_sha256"],
        "original_question_sha256": batch["original_question_sha256"],
        "prompt_sha256": batch["prompt_sha256"],
        "payload_sha256": batch["payload_sha256"],
        "response_sha256": sha((directory / "RESPONSE.json").read_bytes()),
        "returned_model": metadata.get("returned_model"), "response_id": raw.get("id"),
        "content_audited": False, "ready_for_evaluation": False,
        "solver_calls": 0, "training_started": False,
    }
    runtime.write_new(directory / "PARSED_RESPONSE.json", parsed)
    runtime.write_new(directory / "VALIDATION.json", result)
    if valid:
        runtime.write_new(directory / "CANDIDATE_PENDING_AUDIT.json", {
            "status": status, "question": revised,
            "source_author": MODEL, "revision_author": MODEL,
            "content_audited": False, "ready_for_evaluation": False,
        })
    return result


def execute() -> None:
    v9, runtime, batch, _, _, prompt, payload = frozen_inputs()
    directory = HERE / ROOT_ID
    if (directory / "ATTEMPT_RESERVED.json").exists() or (directory / "RESPONSE.json").exists():
        raise FileExistsError("v11_attempt_already_reserved")
    runtime.write_new(directory / "ATTEMPT_RESERVED.json", {
        "status": "RESERVED_ONE_MODEL_REVISION_CALL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "requested_model": MODEL, "maximum_http_attempts": 1,
        "payload_sha256": batch["payload_sha256"],
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
        print(json.dumps({"status": "ERROR_NO_RETRY", "error_type": type(error).__name__}))
        return
    runtime.write_new(directory / "RESPONSE.json", {
        "source_sha256": batch["source_sha256"],
        "original_question_sha256": batch["original_question_sha256"],
        "payload_sha256": batch["payload_sha256"],
        "raw_text": response.raw_text, "metadata": response.metadata,
    })
    print(json.dumps(validate(), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        _, _, batch, _, original, prompt, payload = frozen_inputs()
        print(json.dumps({"status": "FROZEN_INPUTS_VERIFIED_NO_API_CALL",
                          "root_id": ROOT_ID, "gold": original["gold"],
                          "prompt_bytes": len(prompt.encode()), "request_model": payload["model"],
                          "http_attempts": int((HERE / ROOT_ID / "HTTP_ATTEMPT.json").exists()),
                          "response_present": (HERE / ROOT_ID / "RESPONSE.json").exists()}))
    elif args.execute:
        execute()
    elif args.validate:
        print(json.dumps(validate(), ensure_ascii=False))
    else:
        prepare()
