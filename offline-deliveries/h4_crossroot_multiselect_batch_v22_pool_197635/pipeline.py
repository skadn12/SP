"""Independent claim-pool path for the 197635 two-question comparison."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
DELIVERIES = HERE.parent
V21 = DELIVERIES / "h4_crossroot_multiselect_batch_v21_claim_staged"
V21_SHA256 = "b354a246dfff105908298707c889e32a77033c97e2ca62d7f997fdf0e75e2c0a"
ROOT = "cc-train-episode-197635"
MODEL = "Qwen/Qwen3.6-35B-A3B"
MAX_NEW = 16
LABELS = "ABCDEF"
RUN = HERE / ROOT / "batch-01"
PRIOR = V21 / ROOT / "batch-01"
SELECTED = DELIVERIES / "h4_crossroot_multiselect_codex_solver_v15_report/SELECTED_QUESTIONS.json"
CLAIM_FIELDS = {"claim_id", "scene", "phase", "time_scope", "status", "turn_ref",
                "speaker", "quote", "proposition", "relation", "rationale"}
REUSE_SCOPES = {
    "197635-qwen36-multi-22:C": ("reunion", "before", "today and next week", "plan"),
    "197635-qwen36-multi-23:C": ("workshop", "before", "before learning at the event", "plan"),
    "197635-qwen36-multi-23:D": ("workshop", "after", "after the Buddhist session", "reported_action"),
    "197635-qwen36-multi-24:D": ("reunion", "before", "planning RSVPs", "plan"),
    "197635-qwen36-multi-24:F": ("workshop", "after", "after the event", "suggestion"),
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def encoded(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def read(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, value: object, *, raw: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value if raw else json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def dependency():
    if sha((V21 / "pipeline.py").read_bytes()) != V21_SHA256:
        raise ValueError("v21_dependency_changed")
    spec = importlib.util.spec_from_file_location("frozen_v21_pool_dependency", V21 / "pipeline.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    v16, batch, ordering = module.load_dependencies()
    return v16, batch, ordering


def prior_inputs() -> tuple[dict, bytes, dict[str, str]]:
    paths = [PRIOR / name for name in ("source.json", "claim_input.json", "CLAIMS_DRAFT.json",
                                        "CLAIM_CHECKS.json", "CONTENT_REVIEW.json")]
    paths.append(SELECTED)
    hashes = {str(path.relative_to(DELIVERIES)): sha(path.read_bytes()) for path in paths}
    source_raw = paths[0].read_bytes()
    claim_input, draft, checks, review, selected = [read(path) for path in paths[1:]]
    if sha(source_raw) != read(PRIOR / "PREPARED.json")["source_sha256"]:
        raise ValueError("source_not_v21_frozen")
    if review["draft_sha256"] != sha(encoded(draft)) or checks["fatal_errors"]:
        raise ValueError("v21_review_binding_invalid")
    by_id = {slot["candidate_id"]: slot for slot in draft["question_slots"]}
    decisions = {(item["candidate_id"], item["label"]): item for item in review["claim_decisions"]}
    reused = []
    for claim_id, scope in REUSE_SCOPES.items():
        cid, label = claim_id.split(":")
        if not decisions[cid, label]["approved"]:
            raise ValueError(f"v21_claim_not_approved:{claim_id}")
        claim = by_id[cid]["claims"][label]
        reused.append(dict(claim_id="reuse-" + claim_id, scene=scope[0], phase=scope[1],
                           time_scope=scope[2], status=scope[3], **claim))
    turns = {t["turn_ref"]: t for t in claim_input["available_turns"]}
    for item in reused:
        turn = turns[item["turn_ref"]]
        if item["speaker"] != turn["speaker"] or item["quote"] not in turn["text"]:
            raise ValueError(f"reused_claim_mechanically_invalid:{item['claim_id']}")
    selected_rows = [row for row in selected["questions"] if row["root_id"] == ROOT]
    if len(selected_rows) != 1:
        raise ValueError("selected_question_count_changed")
    prompt_input = {"root_id": ROOT, "scope": "R02", "maximum_new_claims": MAX_NEW,
                    "target_complete_questions": 2,
                    "available_turns": claim_input["available_turns"],
                    "selected_question_options_avoid_semantic_repeat": selected_rows[0]["options"],
                    "reviewed_reusable_claims_candidates_only": reused}
    return prompt_input, source_raw, hashes


def frozen_material():
    v16, batch, ordering = dependency()
    prompt_input, source_raw, hashes = prior_inputs()
    prompt = (HERE / "pool_instructions.txt").read_text(encoding="utf-8") + json.dumps(
        prompt_input, ensure_ascii=False, indent=2) + "\n"
    profile = v16.profile(batch)
    profile["profile_id"] = HERE.name
    request = ordering.build_payload(profile, prompt)
    if request.get("model") != MODEL or request.get("max_tokens") != 12288:
        raise ValueError("request_profile_changed")
    return batch, ordering, prompt_input, source_raw, hashes, prompt, request


def prepare() -> dict:
    if RUN.exists():
        raise FileExistsError(f"no_overwrite:{RUN}")
    _, _, prompt_input, source_raw, hashes, prompt, request = frozen_material()
    manifest = {"version": HERE.name, "root_id": ROOT, "model": MODEL,
                "target_complete_questions": 2, "maximum_new_claims": MAX_NEW,
                "source_sha256": sha(source_raw), "dependencies_sha256": hashes,
                "script_sha256": sha(Path(__file__).read_bytes()),
                "instructions_sha256": sha((HERE / "pool_instructions.txt").read_bytes()),
                "input_sha256": sha(encoded(prompt_input)), "prompt_sha256": sha(prompt.encode()),
                "payload_sha256": sha(encoded(request)), "status": "PREPARED_NO_API_CALL",
                "solver_calls": 0, "training_started": False}
    write_new(RUN / "source.json", source_raw.decode(), raw=True)
    write_new(RUN / "POOL_INPUT.json", prompt_input)
    write_new(RUN / "POOL_PROMPT.txt", prompt, raw=True)
    write_new(RUN / "POOL_REQUEST.json", request)
    write_new(RUN / "PREPARED.json", manifest)
    return {"status": manifest["status"], "root_id": ROOT, "reused_claims": 5,
            "available_turns": len(prompt_input["available_turns"]),
            "payload_sha256": manifest["payload_sha256"]}


def preflight():
    batch, ordering, prompt_input, source_raw, hashes, prompt, request = frozen_material()
    manifest = read(RUN / "PREPARED.json")
    expected = {"source_sha256": sha(source_raw), "dependencies_sha256": hashes,
                "script_sha256": sha(Path(__file__).read_bytes()),
                "instructions_sha256": sha((HERE / "pool_instructions.txt").read_bytes()),
                "input_sha256": sha(encoded(prompt_input)), "prompt_sha256": sha(prompt.encode()),
                "payload_sha256": sha(encoded(request))}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("frozen_hash_changed")
    if ((RUN / "source.json").read_bytes() != source_raw or
            read(RUN / "POOL_INPUT.json") != prompt_input or
            (RUN / "POOL_PROMPT.txt").read_text(encoding="utf-8") != prompt or
            read(RUN / "POOL_REQUEST.json") != request):
        raise ValueError("frozen_input_changed")
    return batch, ordering, prompt_input, manifest, prompt, request


def execute() -> dict:
    batch, ordering, _, manifest, prompt, request = preflight()
    v16, _, _ = dependency()
    profile = v16.profile(batch)
    profile["profile_id"] = HERE.name
    write_new(RUN / "ATTEMPT_RESERVED.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "model": MODEL,
        "payload_sha256": manifest["payload_sha256"], "maximum_http_attempts": 1})

    def on_attempt(actual: dict) -> None:
        if actual != request:
            raise ValueError("actual_payload_changed")
        write_new(RUN / "HTTP_ATTEMPT.json", {"attempt_number": 1,
                  "payload_sha256": sha(encoded(actual)), "model": actual["model"]})

    try:
        response = ordering.credential_client(profile, on_http_attempt=on_attempt).complete(
            prompt=prompt, max_tokens=12288, json_object=True)
    except Exception as error:
        result = {"status": "ERROR_NO_RETRY", "error_type": type(error).__name__,
                  "metadata": getattr(error, "metadata", None)}
        write_new(RUN / "ERROR.json", result)
        return result
    write_new(RUN / "POOL_RESPONSE.json", {"source_sha256": manifest["source_sha256"],
              "payload_sha256": manifest["payload_sha256"], "raw_text": response.raw_text,
              "metadata": response.metadata})
    return validate()


def normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def claim_errors(item: object, turns: dict[str, dict]) -> list[str]:
    if not isinstance(item, dict) or set(item) != CLAIM_FIELDS:
        return ["claim_schema_invalid"]
    errors = []
    ref = item["turn_ref"]
    if not isinstance(ref, str) or ref not in turns:
        errors.append("turn_ref_unavailable")
    else:
        if item["speaker"] != turns[ref]["speaker"]:
            errors.append("speaker_mismatch")
        quote = item["quote"]
        if not isinstance(quote, str) or not quote.strip() or quote not in turns[ref]["text"]:
            errors.append("quote_not_verbatim")
    if not isinstance(item["quote"], str) or len(item["quote"]) > 240:
        errors.append("quote_too_long_or_invalid")
    for key in ("time_scope", "proposition", "rationale"):
        if not isinstance(item[key], str) or not item[key].strip():
            errors.append(f"{key}_missing")
    if item["scene"] not in ("reunion", "workshop"):
        errors.append("scene_invalid")
    if item["phase"] not in ("before", "during", "after", "unspecified"):
        errors.append("phase_invalid")
    if item["status"] not in ("plan", "suggestion", "expectation", "reported_action", "opinion", "other"):
        errors.append("status_invalid")
    if item["relation"] not in ("supports", "contradicts"):
        errors.append("relation_invalid")
    if not isinstance(item["speaker"], str) or item["speaker"].casefold() not in item["proposition"].casefold():
        errors.append("speaker_absent_from_proposition")
    return errors


def validate() -> dict:
    _, _, prompt_input, manifest, _, _ = preflight()
    response = read(RUN / "POOL_RESPONSE.json")
    if (response["source_sha256"] != manifest["source_sha256"] or
            response["payload_sha256"] != manifest["payload_sha256"]):
        raise ValueError("response_binding_invalid")
    try:
        draft = json.loads(response["raw_text"])
    except (KeyError, TypeError, ValueError):
        draft = None
    fatal = []
    if not isinstance(draft, dict) or set(draft) != {"root_id", "claims", "abstentions"}:
        fatal.append("top_level_schema_invalid")
        draft = {"root_id": None, "claims": [], "abstentions": []}
    if draft["root_id"] != ROOT:
        fatal.append("root_mismatch")
    claims = draft["claims"]
    if not isinstance(claims, list) or len(claims) > MAX_NEW:
        fatal.append("claim_count_invalid")
        claims = []
    if not isinstance(draft["abstentions"], list) or any(
            not isinstance(x, str) for x in draft["abstentions"]):
        fatal.append("abstentions_invalid")
    metadata = response.get("metadata", {})
    if metadata.get("returned_model") != MODEL:
        fatal.append("returned_model_mismatch")
    if metadata.get("finish_reason") == "length":
        fatal.append("response_truncated")
    turns = {t["turn_ref"]: t for t in prompt_input["available_turns"]}
    old = prompt_input["reviewed_reusable_claims_candidates_only"]
    known_text = {normalized(x["proposition"]) for x in old}
    selected_text = {normalized(x) for x in prompt_input["selected_question_options_avoid_semantic_repeat"].values()}
    checks, ids = [], set()
    for item in claims:
        cid = item.get("claim_id") if isinstance(item, dict) else None
        errors = claim_errors(item, turns)
        if not isinstance(cid, str) or not re.fullmatch(r"new-(0[1-9]|1[0-6])", cid) or cid in ids:
            errors.append("claim_id_invalid_or_duplicate")
        ids.add(cid)
        if isinstance(item, dict) and isinstance(item.get("proposition"), str):
            proposition = normalized(item["proposition"])
            if proposition in known_text:
                errors.append("exact_batch_proposition_duplicate")
            if proposition in selected_text:
                errors.append("exact_selected_option_duplicate")
            known_text.add(proposition)
        checks.append({"claim_id": cid, "errors": errors, "mechanically_valid": not errors})
    usage = metadata.get("usage", {})
    result = {"status": "MECHANICAL_CHECKS_ONLY_CONTENT_REVIEW_PENDING", "root_id": ROOT,
              "model": MODEL, "fatal_errors": fatal, "new_claim_checks": checks,
              "new_claim_count": len(claims), "new_mechanically_valid_count":
              0 if fatal else sum(not x["errors"] for x in checks),
              "reused_v21_claim_count": len(old), "prompt_tokens": usage.get("prompt_tokens"),
              "completion_tokens": usage.get("completion_tokens"),
              "total_tokens": usage.get("total_tokens"),
              "claim_review_units": len(old) + len(claims),
              "draft_sha256": sha(encoded(draft)), "semantic_review_required": True,
              "solver_calls": 0, "training_started": False}
    write_new(RUN / "POOL_DRAFT.json", draft)
    write_new(RUN / "POOL_CHECKS.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "preflight", "execute", "validate"))
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare()
    elif args.action == "preflight":
        _, _, _, manifest, _, _ = preflight()
        result = {"status": "FROZEN_INPUTS_VERIFIED_NO_API_CALL",
                  "payload_sha256": manifest["payload_sha256"]}
    elif args.action == "execute":
        result = execute()
    else:
        result = validate()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
