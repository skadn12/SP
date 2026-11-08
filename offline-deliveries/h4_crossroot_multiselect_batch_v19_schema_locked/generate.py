"""Root-isolated generation with fixed evidence packs and explicit JSON schema."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
V16 = PARENT / "h4_crossroot_multiselect_batch_v16/generate.py"
V16_SHA = "035a3aef0682dd1426bceb61847bac0d1e7c8aec9bb99350a6c86a7043c0b164"
V17 = PARENT / "h4_crossroot_multiselect_batch_v17_revision"
NEW_SELECTED = PARENT / "h4_crossroot_multiselect_v17_254852_solver_selection_v1/SELECTED_QUESTIONS.json"
ROOTS = ("cc-train-episode-136202", "cc-train-episode-197635", "cc-train-episode-254852")
MODEL = "Qwen/Qwen3.6-35B-A3B"
MAX_TOKENS = 12288
LABELS = tuple("ABCDEF")
PACK_REFS = {
    "136202": (
        ("R02:S01:T002", "R02:S01:T003", "R02:S01:T005", "R02:S01:T010", "R02:S01:T011", "R02:S01:T012"),
        ("R02:S02:T001", "R02:S02:T002", "R02:S02:T007", "R02:S02:T011", "R02:S02:T012", "R02:S02:T013"),
        ("R02:S02:T016", "R02:S02:T018", "R02:S02:T020", "R02:S02:T021", "R02:S02:T022", "R02:S03:T012"),
    ),
    "197635": (
        ("R02:S01:T001", "R02:S01:T002", "R02:S01:T006", "R02:S01:T008", "R02:S01:T013", "R02:S02:T001"),
        ("R02:S02:T002", "R02:S02:T007", "R02:S02:T010", "R02:S02:T011", "R02:S02:T012", "R02:S02:T013"),
        ("R02:S02:T014", "R02:S03:T007", "R02:S03:T012", "R02:S03:T013", "R02:S03:T014", "R02:S03:T022"),
    ),
    "254852": (
        ("R02:S01:T005", "R02:S01:T009", "R02:S01:T010", "R02:S01:T011", "R02:S01:T013", "R02:S01:T014"),
        ("R02:S02:T003", "R02:S02:T004", "R02:S02:T007", "R02:S02:T011", "R02:S02:T013", "R02:S02:T014"),
        ("R02:S01:T003", "R02:S01:T008", "R02:S02:T006", "R02:S02:T010", "R02:S03:T012", "R02:S03:T013"),
    ),
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_v16():
    if sha(V16.read_bytes()) != V16_SHA:
        raise ValueError("v16_dependency_changed")
    spec = importlib.util.spec_from_file_location("v18_frozen_v16", V16)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate_id(root_id: str, slot: int) -> str:
    return f"{root_id.rsplit('-', 1)[-1]}-qwen36-multi-{slot + 14:02d}"


def historic_questions(root_id: str) -> list[dict]:
    rows = []
    for version in ("h4_crossroot_multiselect_batch_v16", "h4_crossroot_multiselect_batch_v17_revision"):
        parsed = json.loads((PARENT / version / root_id / "PARSED_RESPONSE.json").read_text())
        rows.extend(q for q in parsed["questions"] if isinstance(q, dict))
    return rows


def evidence_refs(question: dict) -> set[str]:
    entries = question.get("option_evidence", {})
    if not isinstance(entries, dict):
        return set()
    return {item["turn_ref"] for item in entries.values()
            if isinstance(item, dict) and isinstance(item.get("turn_ref"), str)}


def materials(root_id: str):
    if root_id not in ROOTS:
        raise ValueError("unknown_root")
    v16 = load_v16()
    _, v8, batch = v16.base()
    ordering = batch.ordering
    raw, source = ordering.checked_source(root_id, ordering.source_manifest()[root_id], frozen=False)
    context, prior = v16.prior_context(root_id)
    selected_raw = NEW_SELECTED.read_bytes()
    selected = [q for q in json.loads(selected_raw)["questions"] if q["root_id"] == root_id]
    if selected:
        selected_refs = set().union(*(evidence_refs(q) for q in selected))
        context["source_records"].append({"path": str(NEW_SELECTED), "sha256": sha(selected_raw),
                                          "question_count": len(selected)})
        context["forbidden_turn_refs"] = sorted(set(context["forbidden_turn_refs"]) | selected_refs)
        context["forbidden_themes"] = sorted(set(context["forbidden_themes"]) |
                                             {q["topic"] for q in selected})
        context["prior_selected_gold_patterns"].extend(
            {"candidate_id": q["candidate_id"], "gold": "".join(sorted(q["gold"]))}
            for q in selected)
        prior.extend(selected)
    view = v16.generation_view(source, context)
    available = {turn["turn_ref"]: turn for turn in view["turns"]}
    prior_refs = set(context["forbidden_turn_refs"])
    historic = historic_questions(root_id)
    failed_refs = set().union(*(evidence_refs(q) for q in historic))
    refs = PACK_REFS[root_id[-6:]]
    flattened = [ref for pack in refs for ref in pack]
    if len(refs) != 3 or any(len(pack) != 6 for pack in refs) or len(set(flattened)) != 18:
        raise ValueError("invalid_evidence_packs")
    if any(ref not in available or ref in prior_refs for ref in flattened):
        raise ValueError("pack_contains_forbidden_or_absent_ref")
    evidence_packs = []
    for slot, pack in enumerate(refs, 1):
        anchors = []
        for ref in pack:
            turn = available[ref]
            anchors.append({"turn_ref": ref, "speaker": turn["speaker"], "text": turn["text"],
                            "prior_failed_candidate_anchor": ref in failed_refs})
        evidence_packs.append({"candidate_id": candidate_id(root_id, slot), "anchors": anchors})
    prompt_input = {"root_id": root_id,
                    "prior_selected_themes_avoid": context["forbidden_themes"],
                    "evidence_packs": evidence_packs}
    instructions = (HERE / "instructions.txt").read_text()
    for key, value in (("{ROOT_ID}", root_id), ("{ID_ONE}", candidate_id(root_id, 1)),
                       ("{ID_TWO}", candidate_id(root_id, 2)), ("{ID_THREE}", candidate_id(root_id, 3))):
        instructions = instructions.replace(key, value)
    prompt = instructions + json.dumps(prompt_input, ensure_ascii=False, indent=2) + "\n"
    profile = v16.profile(batch)
    profile["profile_id"] = HERE.name
    request = ordering.build_payload(profile, prompt)
    if request["model"] != MODEL or request["max_tokens"] != MAX_TOKENS:
        raise ValueError("model_or_token_limit_changed")
    return v16, v8, batch, raw, source, context, prior, historic, prompt_input, prompt, profile, request


def prepare(root_id: str) -> dict:
    v16, _, batch, raw, _, context, _, historic, prompt_input, prompt, _, request = materials(root_id)
    ordering = batch.ordering
    directory = HERE / root_id
    if directory.exists():
        raise FileExistsError(f"no_overwrite:{directory}")
    manifest = {"version": HERE.name, "root_id": root_id, "model": MODEL,
                "source_sha256": sha(raw),
                "new_selected_sha256": sha(NEW_SELECTED.read_bytes()),
                "prior_context_sha256": sha(ordering.canonical(context)),
                "failed_candidate_ids": [q.get("candidate_id") for q in historic],
                "reused_failed_anchor_count": sum(a["prior_failed_candidate_anchor"] for p in prompt_input["evidence_packs"] for a in p["anchors"]),
                "instructions_sha256": sha((HERE / "instructions.txt").read_bytes()),
                "script_sha256": sha(Path(__file__).read_bytes()),
                "prompt_input_sha256": sha(ordering.canonical(prompt_input)),
                "prompt_sha256": sha(prompt.encode()), "payload_sha256": sha(ordering.canonical(request)),
                "maximum_candidates": 3, "maximum_http_attempts": 1,
                "status": "PREPARED_NO_API_CALL", "solver_calls": 0, "training_started": False}
    directory.mkdir(parents=True)
    ordering.write_new(directory / "source.json", raw.decode(), raw=True)
    ordering.write_new(directory / "prior_context.json", context)
    ordering.write_new(directory / "evidence_packs.json", prompt_input)
    ordering.write_new(directory / "prompt.txt", prompt, raw=True)
    ordering.write_new(directory / "request.json", request)
    ordering.write_new(directory / "PREPARED.json", manifest)
    return {"root_id": root_id, "status": manifest["status"],
            "fresh_anchor_count": 18 - manifest["reused_failed_anchor_count"],
            "reused_failed_anchor_count": manifest["reused_failed_anchor_count"],
            "prompt_bytes": len(prompt.encode()), "payload_sha256": manifest["payload_sha256"]}


def frozen_inputs(root_id: str):
    data = materials(root_id)
    v16, _, batch, raw, _, context, _, _, prompt_input, prompt, _, request = data
    ordering = batch.ordering
    directory = HERE / root_id
    manifest = ordering.read_json(directory / "PREPARED.json")
    expected = {"source_sha256": sha(raw),
                "new_selected_sha256": sha(NEW_SELECTED.read_bytes()),
                "prior_context_sha256": sha(ordering.canonical(context)),
                "instructions_sha256": sha((HERE / "instructions.txt").read_bytes()),
                "script_sha256": sha(Path(__file__).read_bytes()),
                "prompt_input_sha256": sha(ordering.canonical(prompt_input)),
                "prompt_sha256": sha(prompt.encode()), "payload_sha256": sha(ordering.canonical(request))}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("frozen_hash_changed")
    if (raw != (directory / "source.json").read_bytes() or
            context != ordering.read_json(directory / "prior_context.json") or
            prompt_input != ordering.read_json(directory / "evidence_packs.json") or
            prompt != (directory / "prompt.txt").read_text() or
            request != ordering.read_json(directory / "request.json")):
        raise ValueError("frozen_input_file_changed")
    return (*data, directory, manifest)


def check_questions(v16, v8, root_id, source, context, prior, historic, prompt_input, questions):
    packs = {p["candidate_id"]: {a["turn_ref"] for a in p["anchors"]} for p in prompt_input["evidence_packs"]}
    checks, valid, seen = [], [], set()
    for position, row in enumerate(questions, 1):
        errors, diagnostics = [], []
        candidate = row.get("candidate_id") if isinstance(row, dict) else None
        if candidate not in packs:
            errors.append("unknown_candidate_id")
        elif candidate in seen:
            errors.append("duplicate_candidate_id")
        else:
            seen.add(candidate)
        if isinstance(row, dict) and candidate in packs:
            refs = evidence_refs(row)
            if refs != packs[candidate]:
                errors.append("anchor_set_does_not_match_assigned_pack")
            slot = next(i for i in range(1, 4) if candidate_id(root_id, i) == candidate)
            try:
                shadow = dict(row, candidate_id=v16.candidate_id(root_id, slot))
                candidate_errors, diagnostics = v16.check_candidate(
                    shadow, root_id, slot, source, context, prior + historic + valid, v8)
                errors.extend(candidate_errors)
            except (TypeError, ValueError, KeyError, AttributeError) as error:
                errors.append(f"candidate_checker_exception:{type(error).__name__}")
        if position > 3:
            errors.append("candidate_count_exceeds_three")
        checks.append({"position": position, "candidate_id": candidate,
                       "errors": errors, "diagnostics": diagnostics,
                       "mechanically_valid": not errors})
        if not errors:
            valid.append(row)
    return checks, valid


def validate(root_id: str) -> dict:
    v16, v8, batch, _, source, context, prior, historic, prompt_input, _, _, _, directory, frozen = frozen_inputs(root_id)
    ordering = batch.ordering
    response = ordering.read_json(directory / "RESPONSE.json")
    if (response.get("source_sha256") != frozen["source_sha256"] or
            response.get("payload_sha256") != frozen["payload_sha256"]):
        raise ValueError("response_binding_changed")
    metadata = response.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("response_metadata_invalid")
    try:
        parsed = json.loads(response["raw_text"])
    except (TypeError, ValueError):
        parsed = None
    fatal, diagnostics = [], []
    if not isinstance(parsed, dict):
        fatal.append("response_not_json_object")
        questions, abstentions = [], None
    else:
        if set(parsed) != {"questions", "abstentions"}:
            diagnostics.append("top_level_schema_mismatch")
        questions, abstentions = parsed.get("questions"), parsed.get("abstentions")
        if not isinstance(questions, list):
            fatal.append("questions_array_missing")
            questions = []
    if not isinstance(abstentions, list) or any(not isinstance(x, str) for x in abstentions):
        diagnostics.append("abstentions_array_invalid")
    elif len(questions) < 3 and not abstentions:
        diagnostics.append("missing_abstention_reason")
    if metadata.get("returned_model") != MODEL:
        fatal.append("returned_model_mismatch")
    if metadata.get("finish_reason") == "length":
        fatal.append("response_truncated")
    checks, valid = check_questions(v16, v8, root_id, source, context, prior, historic, prompt_input, questions)
    if fatal:
        valid = []
    provenance = {"author": "REMOTE_LARGE_MODEL", "provider": "siliconflow",
                  "requested_model": MODEL, "returned_model": metadata.get("returned_model"),
                  "response_id": (metadata.get("raw_response") or {}).get("id"),
                  "root_id": root_id, "source_sha256": frozen["source_sha256"],
                  "prompt_sha256": frozen["prompt_sha256"],
                  "payload_sha256": frozen["payload_sha256"],
                  "model_questions_modified_by_codex": False,
                  "semantic_gold_verified": False, "solver_evaluated": False,
                  "selected_for_training": False, "training_started": False}
    status = ("GLOBAL_RESPONSE_INVALID" if fatal else
              "MECHANICAL_CHECKS_ONLY_CONTENT_AUDIT_PENDING" if valid else
              "MECHANICAL_REJECTED" if questions else "NO_CANDIDATE_ABSTAINED")
    result = {"version": HERE.name, "root_id": root_id, "status": status,
              "candidate_count": len(questions), "mechanically_valid_count": len(valid),
              "fatal_response_errors": fatal, "response_diagnostics": diagnostics,
              "checks": checks, "provenance": provenance}
    ordering.write_new(directory / "PARSED_RESPONSE.json", parsed)
    ordering.write_new(directory / "VALIDATED_CANDIDATES.json", {
        "status": status, "provenance": provenance, "questions": valid})
    ordering.write_new(directory / "VALIDATION.json", result)
    return result


def execute(root_id: str) -> dict:
    v16, _, batch, _, _, _, _, _, _, prompt, profile, request, directory, frozen = frozen_inputs(root_id)
    ordering = batch.ordering
    ordering.write_new(directory / "ATTEMPT_RESERVED.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "model": MODEL,
        "payload_sha256": frozen["payload_sha256"], "maximum_http_attempts": 1})

    def on_attempt(actual):
        if actual != request:
            raise ValueError("actual_payload_changed")
        ordering.write_new(directory / "HTTP_ATTEMPT.json", {
            "attempt_number": 1, "payload_sha256": sha(ordering.canonical(actual)),
            "model": actual["model"], "max_tokens": actual["max_tokens"]})

    try:
        response = ordering.credential_client(profile, on_http_attempt=on_attempt).complete(
            prompt=prompt, max_tokens=MAX_TOKENS, json_object=True)
    except Exception as error:
        result = {"status": "ERROR_NO_RETRY", "root_id": root_id,
                  "error_type": type(error).__name__, "metadata": getattr(error, "metadata", None)}
        ordering.write_new(directory / "ERROR.json", result)
        return result
    ordering.write_new(directory / "RESPONSE.json", {
        "source_sha256": frozen["source_sha256"],
        "payload_sha256": frozen["payload_sha256"],
        "raw_text": response.raw_text, "metadata": response.metadata})
    return validate(root_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "preflight", "execute", "validate"))
    parser.add_argument("root_id", choices=ROOTS)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.root_id)
    elif args.action == "preflight":
        *_, directory, frozen = frozen_inputs(args.root_id)
        result = {"status": "FROZEN_INPUTS_VERIFIED_NO_API_CALL", "root_id": args.root_id,
                  "payload_sha256": frozen["payload_sha256"]}
    elif args.action == "execute":
        result = execute(args.root_id)
    else:
        result = validate(args.root_id)
    print(json.dumps(result, ensure_ascii=False))
