"""Two-stage, root-isolated model generation with a reviewed claim gate."""

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
V16 = DELIVERIES / "h4_crossroot_multiselect_batch_v16/generate.py"
V16_SHA256 = "035a3aef0682dd1426bceb61847bac0d1e7c8aec9bb99350a6c86a7043c0b164"
MODEL = "Qwen/Qwen3.6-35B-A3B"
ROOTS = tuple(f"cc-train-episode-{suffix}" for suffix in ("136202", "197635", "254852"))
SELECTED_FILES = (
    "h4_crossroot_multiselect_solver_report_two_roots_v1/SELECTED_QUESTIONS.json",
    "h4_crossroot_multiselect_codex_solver_v15_report/SELECTED_QUESTIONS.json",
    "h4_crossroot_multiselect_v17_254852_solver_selection_v1/SELECTED_QUESTIONS.json",
    "h4_crossroot_multiselect_v20_136202_solver_selection_v1/SELECTED_QUESTIONS.json",
)
HISTORIC_DRAFT_DIRS = (
    "h4_crossroot_multiselect_batch_v16",
    "h4_crossroot_multiselect_batch_v17_revision",
    "h4_crossroot_multiselect_batch_v18_isolated",
    "h4_crossroot_multiselect_batch_v19_schema_locked",
    "h4_crossroot_multiselect_batch_v20_root_repair",
)
LABELS = tuple("ABCDEF")
SLOTS = 4
FIRST_INDEX = 21
CLAIM_FIELDS = {"turn_ref", "speaker", "quote", "proposition", "relation", "rationale"}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def read(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, value: object, *, raw: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value if raw else json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_dependencies():
    if digest(V16.read_bytes()) != V16_SHA256:
        raise ValueError("v16_dependency_changed")
    spec = importlib.util.spec_from_file_location("v21_frozen_v16", V16)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, _, batch = module.base()
    return module, batch, batch.ordering


def candidate_id(root_id: str, slot: int) -> str:
    return f"{root_id[-6:]}-qwen36-multi-{FIRST_INDEX + slot - 1:02d}"


def selected_context(root_id: str) -> tuple[list[dict], dict[str, str]]:
    questions, hashes, seen = [], {}, set()
    for relative in SELECTED_FILES:
        path = DELIVERIES / relative
        raw = path.read_bytes()
        hashes[relative] = digest(raw)
        parsed = json.loads(raw)
        rows = parsed if isinstance(parsed, list) else parsed["questions"]
        for row in rows:
            if row.get("root_id") == root_id and row["candidate_id"] not in seen:
                questions.append(row)
                seen.add(row["candidate_id"])
    return questions, hashes


def selected_refs(questions: list[dict]) -> set[str]:
    return {item["turn_ref"] for q in questions for item in q["option_evidence"].values()}


def attempted_options(root_id: str) -> tuple[list[str], dict[str, str]]:
    texts, hashes = set(), {}
    for version in HISTORIC_DRAFT_DIRS:
        path = DELIVERIES / version / root_id / "PARSED_RESPONSE.json"
        raw = path.read_bytes()
        hashes[str(path.relative_to(DELIVERIES))] = digest(raw)
        parsed = json.loads(raw)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("questions"), list):
            raise ValueError(f"historic_draft_schema_invalid:{path}")
        for row in parsed["questions"]:
            if not isinstance(row, dict) or row.get("root_id") != root_id:
                continue
            options = row.get("options")
            if isinstance(options, dict):
                texts.update(value.strip() for value in options.values()
                             if isinstance(value, str) and value.strip())
    return sorted(texts), hashes


def source_material(root_id: str):
    if root_id not in ROOTS:
        raise ValueError("unknown_root")
    v16, batch, ordering = load_dependencies()
    raw, source = ordering.checked_source(root_id, ordering.source_manifest()[root_id], frozen=False)
    selected, selected_hashes = selected_context(root_id)
    failed_options, draft_hashes = attempted_options(root_id)
    excluded = selected_refs(selected)
    turns = [dict(turn_ref=t["turn_ref"], speaker=t["speaker"], text=t["text"])
             for t in source["turns"] if t["turn_ref"].startswith("R02:")
             and t["turn_ref"] not in excluded]
    prior_options = sorted({option for q in selected for option in q["options"].values()} |
                           set(failed_options))
    claim_input = {"root_id": root_id, "scope": "R02", "excluded_selected_turn_refs": sorted(excluded),
                   "available_turns": turns, "prior_attempted_option_texts_avoid": prior_options,
                   "candidate_ids": [candidate_id(root_id, i) for i in range(1, SLOTS + 1)]}
    return v16, batch, ordering, raw, source, selected_hashes, draft_hashes, claim_input


def claim_prompt(root_id: str, claim_input: dict) -> str:
    instructions = (HERE / "claim_instructions.txt").read_text(encoding="utf-8")
    for key, value in (("{ROOT_ID}", root_id),
                       ("{ID_ONE}", candidate_id(root_id, 1)),
                       ("{ID_TWO}", candidate_id(root_id, 2)),
                       ("{ID_THREE}", candidate_id(root_id, 3)),
                       ("{ID_FOUR}", candidate_id(root_id, 4))):
        instructions = instructions.replace(key, value)
    return instructions + json.dumps(claim_input, ensure_ascii=False, indent=2) + "\n"


def request_for(batch, ordering, prompt: str, stage: str) -> dict:
    profile = load_dependencies()[0].profile(batch)
    profile["profile_id"] = f"{HERE.name}_{stage}"
    request = ordering.build_payload(profile, prompt)
    if request.get("model") != MODEL or request.get("max_tokens") != 12288:
        raise ValueError("model_or_limit_changed")
    return request


def prepare_claims(root_id: str) -> dict:
    _, batch, ordering, raw, _, selected_hashes, draft_hashes, claim_input = source_material(root_id)
    directory = HERE / root_id / "batch-01"
    if directory.exists():
        raise FileExistsError(f"no_overwrite:{directory}")
    prompt = claim_prompt(root_id, claim_input)
    request = request_for(batch, ordering, prompt, "claims")
    manifest = {"version": HERE.name, "root_id": root_id, "stage": "claim_draft",
                "model": MODEL, "source_sha256": digest(raw),
                "selected_file_sha256": selected_hashes,
                "historic_draft_sha256": draft_hashes,
                "script_sha256": digest(Path(__file__).read_bytes()),
                "instructions_sha256": digest((HERE / "claim_instructions.txt").read_bytes()),
                "input_sha256": digest(canonical(claim_input)),
                "prompt_sha256": digest(prompt.encode()),
                "payload_sha256": digest(canonical(request)),
                "requested_question_slots": SLOTS,
                "source_turn_count": len(claim_input["available_turns"]),
                "status": "PREPARED_NO_API_CALL", "solver_calls": 0, "training_started": False}
    write_new(directory / "source.json", raw.decode(), raw=True)
    write_new(directory / "claim_input.json", claim_input)
    write_new(directory / "CLAIM_PROMPT.txt", prompt, raw=True)
    write_new(directory / "CLAIM_REQUEST.json", request)
    write_new(directory / "PREPARED.json", manifest)
    return {"root_id": root_id, "status": manifest["status"],
            "available_r02_turns": len(claim_input["available_turns"]),
            "selected_turn_refs_excluded": len(claim_input["excluded_selected_turn_refs"]),
            "requested_question_slots": SLOTS,
            "payload_sha256": manifest["payload_sha256"]}


def frozen_claims(root_id: str):
    _, batch, ordering, raw, source, selected_hashes, draft_hashes, claim_input = source_material(root_id)
    directory = HERE / root_id / "batch-01"
    manifest = read(directory / "PREPARED.json")
    prompt = claim_prompt(root_id, claim_input)
    request = request_for(batch, ordering, prompt, "claims")
    expected = {"source_sha256": digest(raw), "selected_file_sha256": selected_hashes,
                "historic_draft_sha256": draft_hashes,
                "script_sha256": digest(Path(__file__).read_bytes()),
                "instructions_sha256": digest((HERE / "claim_instructions.txt").read_bytes()),
                "input_sha256": digest(canonical(claim_input)),
                "prompt_sha256": digest(prompt.encode()), "payload_sha256": digest(canonical(request))}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("frozen_hash_changed")
    if (raw != (directory / "source.json").read_bytes() or
            claim_input != read(directory / "claim_input.json") or
            prompt != (directory / "CLAIM_PROMPT.txt").read_text(encoding="utf-8") or
            request != read(directory / "CLAIM_REQUEST.json")):
        raise ValueError("frozen_input_changed")
    return batch, ordering, source, claim_input, directory, manifest


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def inspect_claims(root_id: str, draft: object, claim_input: dict) -> dict:
    fatal = []
    if not isinstance(draft, dict) or set(draft) != {"root_id", "question_slots", "abstentions"}:
        return {"fatal_errors": ["top_level_schema_invalid"], "slots": []}
    if draft["root_id"] != root_id:
        fatal.append("root_mismatch")
    slots, abstentions = draft["question_slots"], draft["abstentions"]
    if not isinstance(slots, list) or len(slots) > SLOTS:
        fatal.append("slot_count_invalid")
        slots = []
    if not isinstance(abstentions, list) or any(not isinstance(x, str) for x in abstentions):
        fatal.append("abstentions_invalid")
    if len(slots) < SLOTS and not abstentions:
        fatal.append("missing_abstention_reason")
    turns = {t["turn_ref"]: t for t in claim_input["available_turns"]}
    prior = {normalize(x) for x in claim_input["prior_attempted_option_texts_avoid"]}
    seen_text, seen_ids, cross_refs, checks = set(), set(), {}, []
    expected_ids = set(claim_input["candidate_ids"])
    for slot in slots:
        errors, diagnostics = [], []
        cid = slot.get("candidate_id") if isinstance(slot, dict) else None
        if not isinstance(slot, dict) or set(slot) != {"candidate_id", "claims"}:
            errors.append("slot_schema_invalid")
        if cid not in expected_ids or cid in seen_ids:
            errors.append("candidate_id_invalid_or_duplicate")
        seen_ids.add(cid)
        claims = slot.get("claims") if isinstance(slot, dict) else None
        if not isinstance(claims, dict) or set(claims) != set(LABELS):
            errors.append("six_labeled_claims_required")
            claims = {}
        refs, positives = [], []
        for label in LABELS:
            claim = claims.get(label)
            if not isinstance(claim, dict) or set(claim) != CLAIM_FIELDS:
                errors.append(f"{label}:claim_schema_invalid")
                continue
            ref, quote, speaker = claim["turn_ref"], claim["quote"], claim["speaker"]
            proposition, relation, rationale = claim["proposition"], claim["relation"], claim["rationale"]
            if not isinstance(ref, str) or ref not in turns:
                errors.append(f"{label}:turn_ref_unavailable")
            else:
                refs.append(ref)
                cross_refs.setdefault(ref, set()).add(cid)
                if speaker != turns[ref]["speaker"]:
                    errors.append(f"{label}:speaker_mismatch")
                if not isinstance(quote, str) or not quote.strip() or quote not in turns[ref]["text"]:
                    errors.append(f"{label}:quote_not_verbatim")
            if isinstance(quote, str) and len(quote) > 240:
                errors.append(f"{label}:quote_too_long")
            if not isinstance(proposition, str) or not proposition.strip():
                errors.append(f"{label}:proposition_missing")
            else:
                norm = normalize(proposition)
                if norm in prior:
                    errors.append(f"{label}:prior_attempted_option_text_reused")
                if norm in seen_text:
                    errors.append(f"{label}:claim_text_repeated_in_batch")
                seen_text.add(norm)
                if isinstance(speaker, str) and speaker.casefold() not in proposition.casefold():
                    diagnostics.append(f"{label}:speaker_name_absent_from_proposition")
            if relation not in ("supports", "contradicts"):
                errors.append(f"{label}:relation_invalid")
            elif relation == "supports":
                positives.append(label)
            if not isinstance(rationale, str) or not rationale.strip():
                errors.append(f"{label}:rationale_missing")
        if len(refs) != len(set(refs)):
            errors.append("turn_ref_repeated_within_question")
        if not 2 <= len(positives) <= 6:
            errors.append("gold_cardinality_invalid")
        if len(positives) < 6 and positives == list(LABELS[:len(positives)]):
            errors.append("gold_prefix_forbidden")
        checks.append({"candidate_id": cid, "errors": errors, "diagnostics": diagnostics,
                       "inferred_gold": positives, "mechanically_valid": not errors})
    if len(checks) + len(abstentions) < SLOTS:
        fatal.append("slot_or_abstention_unaccounted_for")
    reused = {ref: sorted(ids) for ref, ids in cross_refs.items() if len(ids) > 1}
    return {"fatal_errors": fatal, "slots": checks, "cross_question_reused_turn_refs": reused,
            "mechanically_valid_count": sum(not row["errors"] for row in checks),
            "semantic_review_required": True}


def validate_claims(root_id: str) -> dict:
    _, _, _, claim_input, directory, manifest = frozen_claims(root_id)
    response = read(directory / "CLAIM_RESPONSE.json")
    if (response.get("source_sha256") != manifest["source_sha256"] or
            response.get("payload_sha256") != manifest["payload_sha256"]):
        raise ValueError("claim_response_binding_changed")
    metadata = response.get("metadata", {})
    try:
        draft = json.loads(response["raw_text"])
    except (KeyError, TypeError, ValueError):
        draft = None
    result = inspect_claims(root_id, draft, claim_input)
    if metadata.get("returned_model") != MODEL:
        result["fatal_errors"].append("returned_model_mismatch")
    if metadata.get("finish_reason") == "length":
        result["fatal_errors"].append("response_truncated")
    result.update({"status": "MECHANICAL_CHECKS_ONLY_CONTENT_REVIEW_PENDING",
                   "root_id": root_id, "model": MODEL,
                   "response_sha256": digest(canonical(response)),
                   "solver_calls": 0, "training_started": False})
    write_new(directory / "CLAIMS_DRAFT.json", draft)
    write_new(directory / "CLAIM_CHECKS.json", result)
    return result


def approved_slots(root_id: str):
    _, _, _, claim_input, directory, _ = frozen_claims(root_id)
    checks, draft, review = (read(directory / name) for name in
                             ("CLAIM_CHECKS.json", "CLAIMS_DRAFT.json", "CONTENT_REVIEW.json"))
    if checks["fatal_errors"] or review.get("draft_sha256") != digest(canonical(draft)):
        raise ValueError("unreviewable_or_changed_draft")
    decisions = review.get("claim_decisions")
    if not isinstance(decisions, list) or not isinstance(review.get("reviewer"), str) or not review["reviewer"]:
        raise ValueError("content_review_missing")
    keyed = {}
    for item in decisions:
        key = (item.get("candidate_id"), item.get("label"))
        if key in keyed or not isinstance(item.get("approved"), bool) or not isinstance(item.get("reason"), str):
            raise ValueError("review_decision_invalid_or_duplicate")
        keyed[key] = item
    expected = {(slot["candidate_id"], label) for slot in draft["question_slots"] for label in LABELS}
    if set(keyed) != expected:
        raise ValueError("review_does_not_cover_all_claims")
    by_id = {row["candidate_id"]: row for row in checks["slots"]}
    approved = [slot for slot in draft["question_slots"]
                if not by_id[slot["candidate_id"]]["errors"] and
                all(keyed[slot["candidate_id"], label]["approved"] for label in LABELS)]
    if not 1 <= len(approved) <= SLOTS:
        raise ValueError(f"assembly_requires_1_to_4_approved_complete_slots:{len(approved)}")
    return directory, claim_input, approved, review


def prepare_assembly(root_id: str) -> dict:
    directory, _, slots, review = approved_slots(root_id)
    _, batch, ordering, _, _, _, _, _ = source_material(root_id)
    ids = [slot["candidate_id"] for slot in slots]
    prompt_input = {"root_id": root_id, "reviewer": review["reviewer"], "approved_claim_slots": slots}
    prompt = (HERE / "assembly_instructions.txt").read_text(encoding="utf-8") + json.dumps(
        prompt_input, ensure_ascii=False, indent=2) + "\n"
    request = request_for(batch, ordering, prompt, "assembly")
    stage = directory / "assembly"
    if stage.exists():
        raise FileExistsError(f"no_overwrite:{stage}")
    manifest = {"version": HERE.name, "root_id": root_id, "stage": "question_assembly",
                "model": MODEL, "candidate_ids": ids,
                "approved_claims_sha256": digest(canonical(slots)),
                "content_review_sha256": digest(canonical(review)),
                "script_sha256": digest(Path(__file__).read_bytes()),
                "assembly_instructions_sha256": digest((HERE / "assembly_instructions.txt").read_bytes()),
                "prompt_sha256": digest(prompt.encode()), "payload_sha256": digest(canonical(request)),
                "requested_question_slots": SLOTS,
                "approved_question_slots": len(ids),
                "shortfall_from_requested_slots": SLOTS - len(ids),
                "status": "PREPARED_NO_API_CALL", "solver_calls": 0, "training_started": False}
    write_new(stage / "ASSEMBLY_INPUT.json", prompt_input)
    write_new(stage / "ASSEMBLY_PROMPT.txt", prompt, raw=True)
    write_new(stage / "ASSEMBLY_REQUEST.json", request)
    write_new(stage / "PREPARED.json", manifest)
    return {"root_id": root_id, "status": manifest["status"],
            "approved_question_slots": len(ids), "shortfall_from_requested_slots": SLOTS - len(ids),
            "candidate_ids": ids}


def frozen_assembly(root_id: str):
    directory, _, slots, review = approved_slots(root_id)
    _, batch, ordering, _, source, _, _, _ = source_material(root_id)
    stage = directory / "assembly"
    manifest = read(stage / "PREPARED.json")
    prompt_input = {"root_id": root_id, "reviewer": review["reviewer"], "approved_claim_slots": slots}
    prompt = (HERE / "assembly_instructions.txt").read_text(encoding="utf-8") + json.dumps(
        prompt_input, ensure_ascii=False, indent=2) + "\n"
    request = request_for(batch, ordering, prompt, "assembly")
    expected = {"candidate_ids": [slot["candidate_id"] for slot in slots],
                "approved_claims_sha256": digest(canonical(slots)),
                "content_review_sha256": digest(canonical(review)),
                "script_sha256": digest(Path(__file__).read_bytes()),
                "assembly_instructions_sha256": digest((HERE / "assembly_instructions.txt").read_bytes()),
                "prompt_sha256": digest(prompt.encode()), "payload_sha256": digest(canonical(request))}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("assembly_frozen_hash_changed")
    if (prompt_input != read(stage / "ASSEMBLY_INPUT.json") or
            prompt != (stage / "ASSEMBLY_PROMPT.txt").read_text(encoding="utf-8") or
            request != read(stage / "ASSEMBLY_REQUEST.json")):
        raise ValueError("assembly_frozen_input_changed")
    return batch, ordering, source, slots, stage, manifest, prompt, request


def execute(root_id: str, stage_name: str) -> dict:
    if stage_name == "claims":
        batch, ordering, _, _, directory, frozen = frozen_claims(root_id)
        prompt = (directory / "CLAIM_PROMPT.txt").read_text(encoding="utf-8")
        request = read(directory / "CLAIM_REQUEST.json")
        profile_id = "claims"
        response_name = "CLAIM_RESPONSE.json"
    else:
        batch, ordering, _, _, directory, frozen, prompt, request = frozen_assembly(root_id)
        profile_id = "assembly"
        response_name = "ASSEMBLY_RESPONSE.json"
    profile = load_dependencies()[0].profile(batch)
    profile["profile_id"] = f"{HERE.name}_{profile_id}"
    write_new(directory / "ATTEMPT_RESERVED.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "model": MODEL,
        "payload_sha256": frozen["payload_sha256"], "maximum_http_attempts": 1})

    def on_attempt(actual: dict) -> None:
        if actual != request:
            raise ValueError("actual_payload_changed")
        write_new(directory / "HTTP_ATTEMPT.json", {
            "attempt_number": 1, "payload_sha256": digest(canonical(actual)),
            "model": actual["model"], "max_tokens": actual["max_tokens"]})

    try:
        response = ordering.credential_client(profile, on_http_attempt=on_attempt).complete(
            prompt=prompt, max_tokens=12288, json_object=True)
    except Exception as error:
        result = {"status": "ERROR_NO_RETRY", "root_id": root_id,
                  "stage": stage_name, "error_type": type(error).__name__,
                  "metadata": getattr(error, "metadata", None)}
        write_new(directory / "ERROR.json", result)
        return result
    write_new(directory / response_name, {
        "source_sha256": frozen.get("source_sha256", read(HERE / root_id / "batch-01" / "PREPARED.json")["source_sha256"]),
        "payload_sha256": frozen["payload_sha256"],
        "raw_text": response.raw_text, "metadata": response.metadata})
    return validate_claims(root_id) if stage_name == "claims" else validate_assembly(root_id)


def validate_assembly(root_id: str) -> dict:
    _, _, source, slots, stage, frozen, _, _ = frozen_assembly(root_id)
    response = read(stage / "ASSEMBLY_RESPONSE.json")
    claim_manifest = read(HERE / root_id / "batch-01" / "PREPARED.json")
    if (response.get("source_sha256") != claim_manifest["source_sha256"] or
            response.get("payload_sha256") != frozen["payload_sha256"]):
        raise ValueError("assembly_response_binding_changed")
    metadata = response.get("metadata", {})
    try:
        parsed = json.loads(response["raw_text"])
    except (KeyError, TypeError, ValueError):
        parsed = None
    fatal = []
    if not isinstance(parsed, dict) or set(parsed) != {"questions", "abstentions"}:
        fatal.append("top_level_schema_invalid")
        parsed = {"questions": [], "abstentions": []}
    rows = parsed.get("questions")
    if not isinstance(rows, list) or len(rows) != len(slots) or parsed.get("abstentions") != []:
        fatal.append("question_count_or_abstentions_invalid")
        rows = rows if isinstance(rows, list) else []
    if metadata.get("returned_model") != MODEL:
        fatal.append("returned_model_mismatch")
    if metadata.get("finish_reason") == "length":
        fatal.append("response_truncated")
    v16, _, _ = load_dependencies()
    _, v8, _ = v16.base()
    checks = []
    for index, slot in enumerate(slots):
        row = rows[index] if index < len(rows) else None
        errors = []
        cid = slot["candidate_id"]
        if not isinstance(row, dict):
            errors.append("question_missing_or_not_object")
        else:
            shadow = dict(row, candidate_id=f"{root_id[-6:]}-qwen36-multi-01")
            errors.extend(v8.check_candidate(shadow, root_id, source["turns"]))
            if row.get("candidate_id") != cid:
                errors.append("candidate_id_changed")
            claims = slot["claims"]
            expected_options = {label: claims[label]["proposition"] for label in LABELS}
            expected_gold = [label for label in LABELS if claims[label]["relation"] == "supports"]
            expected_evidence = {label: {"turn_ref": claims[label]["turn_ref"],
                                         "quote": claims[label]["quote"],
                                         "relation": claims[label]["relation"],
                                         "reason": claims[label]["rationale"]} for label in LABELS}
            if row.get("options") != expected_options:
                errors.append("approved_options_changed")
            if row.get("gold") != expected_gold:
                errors.append("approved_gold_changed")
            if row.get("option_evidence") != expected_evidence:
                errors.append("approved_evidence_changed")
            refs = [claims[label]["turn_ref"] for label in LABELS]
            if len(set(refs)) != len(LABELS):
                errors.append("turn_ref_repeated_within_question")
        checks.append({"candidate_id": cid, "errors": errors, "mechanically_valid": not errors})
    if len(rows) > len(slots):
        fatal.append("extra_questions_returned")
    result = {"version": HERE.name, "root_id": root_id,
              "status": "ASSEMBLY_MECHANICAL_CHECKS_ONLY_CONTENT_REVIEW_PENDING",
              "fatal_errors": fatal, "checks": checks,
              "mechanically_valid_count": 0 if fatal else sum(not x["errors"] for x in checks),
              "cross_question_turn_reuse_allowed_with_review": True,
              "semantic_question_review_required": True,
              "model": MODEL, "solver_calls": 0, "training_started": False}
    write_new(stage / "ASSEMBLED_QUESTIONS.json", parsed)
    write_new(stage / "ASSEMBLY_CHECKS.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare-claims", "preflight-claims", "execute-claims",
                                           "validate-claims", "prepare-assembly",
                                           "preflight-assembly", "execute-assembly", "validate-assembly"))
    parser.add_argument("root_id", choices=ROOTS)
    args = parser.parse_args()
    if args.action == "prepare-claims":
        result = prepare_claims(args.root_id)
    elif args.action == "preflight-claims":
        *_, manifest = frozen_claims(args.root_id)
        result = {"root_id": args.root_id, "status": "FROZEN_INPUTS_VERIFIED_NO_API_CALL",
                  "payload_sha256": manifest["payload_sha256"]}
    elif args.action == "validate-claims":
        result = validate_claims(args.root_id)
    elif args.action == "execute-claims":
        result = execute(args.root_id, "claims")
    elif args.action == "prepare-assembly":
        result = prepare_assembly(args.root_id)
    elif args.action == "preflight-assembly":
        *_, manifest, _, _ = frozen_assembly(args.root_id)
        result = {"root_id": args.root_id, "status": "FROZEN_ASSEMBLY_VERIFIED_NO_API_CALL",
                  "payload_sha256": manifest["payload_sha256"]}
    elif args.action == "execute-assembly":
        result = execute(args.root_id, "assembly")
    else:
        result = validate_assembly(args.root_id)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
