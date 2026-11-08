"""Two-stage, source-locked QA pilot. Preparing a stage never calls the API."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
WORKTREE = Path("/root/autodl-tmp/gxy-selfplay/worktrees/h4-writer-coverage-v1")
sys.path.insert(0, str(WORKTREE))
from scripts.current_method.restart_n5_spell.h4_memory_light_gate_siliconflow_v1.run import (  # noqa: E402
    MODEL, credential_client, provider_profile,
)

R03 = Path("/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_memory_light_gate_siliconflow_qa_v1/prepared_run/frozen/qa_source.json")
G02 = Path("/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_graph_guided_qa_pilot_v2/realized_dialogue.json")
R03_SHA256 = "b163bd9ceed0ec0cf4745e5c414fb5d7615788672c879c6283669aad78395113"
G02_SHA256 = "be4707100a82f88a2e61535c550fc603981c25b6d2f13613b52c2b87895482f0"
HISTORY_REFS = (
    "R02:S03:T007", "R02:S03:T012", "R02:S03:T013",
    "R03:S01:T010", "R03:S01:T012", "R03:S01:T014",
    "R03:S01:T016", "R03:S01:T019",
    "R03:S03:T001", "R03:S03:T008", "R03:S03:T009",
)
STATUSES = frozenset(("performed_now", "completed_reported", "status_at_checkpoint", "plan", "request", "promise", "hypothetical"))
ORIENTATIONS = frozenset(("as_stated", "opposite"))
TURN_REF_RE = re.compile(r"\b(?:R\d{2}|G\d{2}):S\d{2}:T\d{3}\b")

EXTRACT_RULES = """Extract atomic, independently checkable facts from ONLY the visible turns below. Return JSON only.
Each fact must be stated in one cited utterance. Do not infer an action from a plan, promise, request, later report, transcript position, or silence. Keep object and version exact; an empty version means the utterance does not establish a version. The structured checkpoint must equal the evidence ref. State time scope in claim/opposite_claim using natural event words (such as "when Sarah answered availability"), never raw turn IDs. claim is one source-supported sentence. opposite_claim must be its direct contradiction for the SAME actor, object, version, predicate, and checkpoint, with no added facts. Both wordings will require human source review before question generation. A report of a past action has status completed_reported, never performed_now. Extract nothing unsupported. Use stable unique F001-style IDs. You do not see event keys or an intended answer.
Schema: {"facts":[{"fact_id":"F001","actor":"...","action":"...","object_id":"...","version":"... or empty","checkpoint":"exact turn ref","status":"performed_now|completed_reported|status_at_checkpoint|plan|request|promise|hypothetical","claim":"one directly supported fact","opposite_claim":"direct same-scope contradiction","evidence":{"ref":"exact turn ref","speaker":"speaker of cited turn","quote":"exact nonempty substring of spoken text"}}],"abstain_reason":"empty unless no safe facts"}
VISIBLE TURNS:
"""

COMPOSE_RULES = """Select one two-fact multi-select question from the SOURCE_REVIEWED fact records below. Return JSON only. You may select IDs and orientations; do not write question text, options, gold, or evidence yourself. The program renders those from reviewed records.
Each of five options has exactly two DIFFERENT fact IDs cited by different turns. as_stated uses a reviewed source-supported claim. opposite uses that record's reviewed direct contradiction, preserving its actor, object, version, predicate and checkpoint. Use 2 or 3 fully true options; every other option needs at least one opposite clause. Fully true options must combine R03 with G02, or two different G02 scenes. Choose different fact pairs for all five options. Avoid trivially repeated status sentences. If this cannot be done, return selected_options=[] with a reason. No question or answer is training-certified here.
Schema: {"selected_options":[{"clauses":[{"fact_id":"F001","orientation":"as_stated"},{"fact_id":"F002","orientation":"opposite"}]}],"abstain_reason":"empty for five options; explain otherwise"}
REVIEWED FACTS:
"""


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strict_json(raw: str) -> Any:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate_json_key:{key}")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique_pairs)


def write_new(path: Path, value: Any, *, raw: bool = False) -> None:
    with path.open("x", encoding="utf-8") as handle:
        if raw:
            handle.write(str(value))
        else:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}_missing")
    return value


def _project(turn: Any) -> dict[str, str]:
    if not isinstance(turn, dict):
        raise ValueError("turn_not_object")
    return {key: _nonempty(turn.get(key), f"turn_{key}") for key in ("turn_ref", "speaker", "text")}


def surface() -> dict[str, Any]:
    if sha(R03.read_bytes()) != R03_SHA256 or sha(G02.read_bytes()) != G02_SHA256:
        raise ValueError("source_hash_mismatch")
    old, new = strict_json(R03.read_text(encoding="utf-8")), strict_json(G02.read_text(encoding="utf-8"))
    if old.get("root_id") != new.get("root_id") or new.get("source_r03_sha256") != R03_SHA256:
        raise ValueError("source_root_or_parent_mismatch")
    history = [_project(row) for row in old["turns"]]
    if len(history) != 188 or old.get("turn_count") != 188:
        raise ValueError("history_turn_count_invalid")
    old_by_ref = {row["turn_ref"]: row for row in history}
    if len(old_by_ref) != len(history) or any(ref not in old_by_ref for ref in HISTORY_REFS):
        raise ValueError("history_refs_invalid")
    g02 = [_project(row) for row in new["turns"]]
    if len(g02) != 10 or new.get("turn_count") != 10 or any(not row["turn_ref"].startswith("G02:") for row in g02):
        raise ValueError("g02_turns_invalid")
    rows = [old_by_ref[ref] for ref in HISTORY_REFS] + g02
    if len({row["turn_ref"] for row in rows}) != len(rows):
        raise ValueError("duplicate_visible_ref")
    return {"root_id": old["root_id"], "history_sha256": R03_SHA256, "g02_sha256": G02_SHA256, "turns": rows}


def extraction_prompt(view: dict[str, Any]) -> str:
    return EXTRACT_RULES + json.dumps(view["turns"], ensure_ascii=False, indent=2) + "\n"


def validate_extraction(raw: str, view: dict[str, Any]) -> dict[str, Any]:
    parsed = strict_json(raw)
    if not isinstance(parsed, dict) or set(parsed) != {"facts", "abstain_reason"}:
        raise ValueError("extraction_top_level_invalid")
    facts = parsed["facts"]
    if not isinstance(facts, list) or len(facts) > 60 or not isinstance(parsed["abstain_reason"], str):
        raise ValueError("extraction_shape_invalid")
    if bool(facts) == bool(parsed["abstain_reason"].strip()):
        raise ValueError("abstain_reason_inconsistent")
    turns = {row["turn_ref"]: row for row in view["turns"]}
    seen: set[str] = set()
    clean: list[dict[str, Any]] = []
    required = {"fact_id", "actor", "action", "object_id", "version", "checkpoint", "status", "claim", "opposite_claim", "evidence"}
    for row in facts:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("fact_shape_invalid")
        fact_id = _nonempty(row["fact_id"], "fact_id")
        if fact_id in seen:
            raise ValueError("duplicate_fact_id")
        seen.add(fact_id)
        for field in ("actor", "action", "object_id", "checkpoint", "claim", "opposite_claim"):
            _nonempty(row[field], field)
        if TURN_REF_RE.search(row["claim"]) or TURN_REF_RE.search(row["opposite_claim"]):
            raise ValueError("turn_ref_leaked_into_claim")
        if not isinstance(row["version"], str) or row["status"] not in STATUSES:
            raise ValueError("fact_version_or_status_invalid")
        evidence = row["evidence"]
        if not isinstance(evidence, dict) or set(evidence) != {"ref", "speaker", "quote"}:
            raise ValueError("evidence_shape_invalid")
        ref = _nonempty(evidence["ref"], "evidence_ref")
        if ref not in turns or row["checkpoint"] != ref:
            raise ValueError("evidence_ref_or_checkpoint_invalid")
        if evidence["speaker"] != turns[ref]["speaker"]:
            raise ValueError("evidence_speaker_mismatch")
        quote = _nonempty(evidence["quote"], "evidence_quote")
        if quote not in turns[ref]["text"]:
            raise ValueError("evidence_quote_not_verbatim")
        clean.append(row)
    return {
        "status": "PENDING_HUMAN_SEMANTIC_REVIEW", "root_id": view["root_id"],
        "history_sha256": view["history_sha256"], "g02_sha256": view["g02_sha256"],
        "facts": clean, "abstain_reason": parsed["abstain_reason"], "training_promotion": False,
    }


def prepare_extraction() -> dict[str, Any]:
    view = surface()
    prompt = extraction_prompt(view)
    stage = HERE / "extraction"
    stage.mkdir(exist_ok=False)
    write_new(stage / "prompt.txt", prompt, raw=True)
    manifest = {"stage": "extraction", "model": MODEL, "prompt_sha256": sha(prompt.encode("utf-8")),
                "history_sha256": R03_SHA256, "g02_sha256": G02_SHA256,
                "visible_history_refs": list(HISTORY_REFS), "visible_g02_turn_count": 10,
                "status": "PREPARED_OFFLINE_NOT_EXECUTED", "training_promotion": False}
    write_new(stage / "manifest.json", manifest)
    return manifest


def reviewed_facts(view: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    pending_path = HERE / "extraction" / "pending_ledger.json"
    review_path = HERE / "reviewed_ids.json"
    pending_bytes = pending_path.read_bytes()
    pending = strict_json(pending_bytes.decode("utf-8"))
    review = strict_json(review_path.read_text(encoding="utf-8"))
    if pending.get("status") != "PENDING_HUMAN_SEMANTIC_REVIEW":
        raise ValueError("pending_ledger_missing_or_invalid")
    if any(pending.get(key) != view[key] for key in ("root_id", "history_sha256", "g02_sha256")):
        raise ValueError("pending_source_mismatch")
    if not isinstance(review, dict) or set(review) != {"status", "attestation", "reviewer", "pending_ledger_sha256", "accepted_fact_ids"}:
        raise ValueError("review_manifest_invalid")
    if (review["status"] != "SOURCE_REVIEWED" or review["attestation"] != "SOURCE_AND_OPPOSITE_CLAIM_REVIEWED"
            or not isinstance(review["reviewer"], str) or not review["reviewer"].strip()):
        raise ValueError("review_not_attested")
    if review["pending_ledger_sha256"] != sha(pending_bytes):
        raise ValueError("review_pending_hash_mismatch")
    ids = review["accepted_fact_ids"]
    if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids) or len(ids) != len(set(ids)):
        raise ValueError("accepted_fact_ids_invalid")
    by_id = {row["fact_id"]: row for row in pending["facts"]}
    if len(by_id) != len(pending["facts"]) or any(x not in by_id for x in ids):
        raise ValueError("accepted_fact_id_unknown")
    # The reviewer attests both source-supported and opposite-claim wording.
    return [by_id[x] for x in ids], sha(review_path.read_bytes())


def composition_prompt(facts: list[dict[str, Any]]) -> str:
    return COMPOSE_RULES + json.dumps(facts, ensure_ascii=False, indent=2) + "\n"


def prepare_composition() -> dict[str, Any]:
    view = surface()
    facts, review_sha = reviewed_facts(view)
    prompt = composition_prompt(facts)
    stage = HERE / "composition"
    stage.mkdir(exist_ok=False)
    write_new(stage / "prompt.txt", prompt, raw=True)
    manifest = {"stage": "composition", "model": MODEL, "prompt_sha256": sha(prompt.encode("utf-8")),
                "history_sha256": R03_SHA256, "g02_sha256": G02_SHA256,
                "reviewed_ids_sha256": review_sha, "accepted_fact_count": len(facts),
                "status": "PREPARED_OFFLINE_NOT_EXECUTED", "training_promotion": False}
    write_new(stage / "manifest.json", manifest)
    return manifest


def _true_option_crosses_windows(facts: list[dict[str, Any]]) -> bool:
    refs = [fact["evidence"]["ref"] for fact in facts]
    if any(ref.startswith("G02:") for ref in refs) and any(not ref.startswith("G02:") for ref in refs):
        return True
    return all(ref.startswith("G02:") for ref in refs) and len({ref.rsplit(":T", 1)[0] for ref in refs}) == 2


def validate_composition(raw: str, facts: list[dict[str, Any]], view: dict[str, Any]) -> dict[str, Any]:
    parsed = strict_json(raw)
    if not isinstance(parsed, dict) or set(parsed) != {"selected_options", "abstain_reason"}:
        raise ValueError("composition_top_level_invalid")
    choices = parsed["selected_options"]
    reason = parsed["abstain_reason"]
    if not isinstance(choices, list) or not isinstance(reason, str):
        raise ValueError("composition_shape_invalid")
    if not choices:
        _nonempty(reason, "abstain_reason")
        return {"status": "ABSTAIN", "reason": reason, "questions": [], "training_promotion": False}
    if len(choices) != 5 or reason.strip():
        raise ValueError("five_options_required")
    by_id = {row["fact_id"]: row for row in facts}
    labels = "ABCDE"
    options: dict[str, str] = {}
    evidence: dict[str, Any] = {}
    gold: list[str] = []
    seen_options: set[tuple[tuple[str, str], ...]] = set()
    for label, choice in zip(labels, choices, strict=True):
        if not isinstance(choice, dict) or set(choice) != {"clauses"} or not isinstance(choice["clauses"], list) or len(choice["clauses"]) != 2:
            raise ValueError("two_clauses_required")
        selected: list[dict[str, Any]] = []
        key: list[tuple[str, str]] = []
        clause_rows: list[dict[str, Any]] = []
        for clause in choice["clauses"]:
            if not isinstance(clause, dict) or set(clause) != {"fact_id", "orientation"}:
                raise ValueError("clause_shape_invalid")
            fact_id, orientation = clause["fact_id"], clause["orientation"]
            if not isinstance(fact_id, str) or fact_id not in by_id or orientation not in ORIENTATIONS:
                raise ValueError("unreviewed_fact_or_orientation")
            fact = by_id[fact_id]
            selected.append(fact)
            key.append((fact_id, orientation))
            clause_rows.append({
                "fact_id": fact_id, "verdict": "TRUE" if orientation == "as_stated" else "FALSE",
                "claim": fact["claim"] if orientation == "as_stated" else fact["opposite_claim"],
                "evidence_ref": fact["evidence"]["ref"], "quote": fact["evidence"]["quote"],
                "actor": fact["actor"], "object_id": fact["object_id"],
                "version": fact["version"], "checkpoint": fact["checkpoint"],
            })
        if len({row["fact_id"] for row in clause_rows}) != 2 or len({row["evidence_ref"] for row in clause_rows}) != 2:
            raise ValueError("option_not_two_independent_facts")
        option_key = tuple(sorted(key))
        if option_key in seen_options:
            raise ValueError("duplicate_option")
        seen_options.add(option_key)
        is_true = all(row["verdict"] == "TRUE" for row in clause_rows)
        if is_true and not _true_option_crosses_windows(selected):
            raise ValueError("true_option_not_cross_window")
        if is_true:
            gold.append(label)
        options[label] = " and ".join(row["claim"].rstrip(" .") for row in clause_rows) + "."
        evidence[label] = {"verdict": "TRUE" if is_true else "FALSE", "clauses": clause_rows}
    if len(gold) not in (2, 3):
        raise ValueError("gold_size_invalid")
    if len(set(options.values())) != 5:
        raise ValueError("duplicate_option_text")
    candidate_id = "v3-ms-" + sha(json.dumps(choices, sort_keys=True, separators=(",", ":")).encode("utf-8"))[:12]
    return {
        "status": "UNREVIEWED_CANDIDATE_NOT_GOLD", "candidate_id": candidate_id, "root_id": view["root_id"],
        "question_type": "multi_select", "question": "Which statements are supported by the conversation? Select all that apply.",
        "options": options, "gold": gold, "option_evidence": evidence,
        "history_sha256": R03_SHA256, "g02_sha256": G02_SHA256,
        "training_promotion": False,
    }


def run_stage(stage_name: str) -> dict[str, Any]:
    view = surface()
    stage = HERE / stage_name
    manifest = strict_json((stage / "manifest.json").read_text(encoding="utf-8"))
    prompt_bytes = (stage / "prompt.txt").read_bytes()
    if manifest.get("stage") != stage_name or manifest.get("model") != MODEL or manifest.get("prompt_sha256") != sha(prompt_bytes):
        raise ValueError("frozen_prompt_mismatch")
    if (manifest.get("history_sha256"), manifest.get("g02_sha256")) != (R03_SHA256, G02_SHA256):
        raise ValueError("frozen_source_mismatch")
    facts: list[dict[str, Any]] = []
    if stage_name == "extraction":
        if prompt_bytes.decode("utf-8") != extraction_prompt(view):
            raise ValueError("extraction_prompt_changed")
    elif stage_name == "composition":
        facts, review_sha = reviewed_facts(view)
        if manifest.get("reviewed_ids_sha256") != review_sha or prompt_bytes.decode("utf-8") != composition_prompt(facts):
            raise ValueError("reviewed_ledger_or_prompt_changed")
    else:
        raise ValueError("unknown_stage")
    profile = provider_profile()
    if profile["model"] != MODEL or MODEL != "Qwen/Qwen3.6-35B-A3B":
        raise ValueError("unexpected_model")
    profile["profile_id"] = f"h4_structured_qa_pilot_v3_{stage_name}"
    profile["request_parameters"].update(temperature=0, max_tokens=6144, enable_thinking=False)
    profile["output_limit"] = 6144
    profile["http_retry_limit"] = 0
    write_new(stage / "attempt.reserved.json", {"status": "RESERVED_ONE_ATTEMPT", "prompt_sha256": manifest["prompt_sha256"],
                                                   "model": MODEL, "maximum_http_attempts": 1, "training_promotion": False})

    def on_attempt(payload: dict[str, Any]) -> None:
        write_new(stage / "http_attempt.json", {"attempt_number": 1, "payload_sha256": sha(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))})

    try:
        response = credential_client(profile, on_http_attempt=on_attempt).complete(
            prompt=prompt_bytes.decode("utf-8"), max_tokens=6144, json_object=True)
        write_new(stage / "response_raw.txt", response.raw_text, raw=True)
        write_new(stage / "response_meta.json", {"status": "RESPONSE_RECEIVED", "raw_sha256": sha(response.raw_text.encode("utf-8")),
                                                  "metadata": response.metadata, "training_promotion": False})
    except Exception as error:
        write_new(stage / "failure.json", {"status": "API_FAILED_NO_RETRY", "error_type": type(error).__name__,
                                           "http_attempt_recorded": (stage / "http_attempt.json").exists(), "training_promotion": False})
        raise
    try:
        result = validate_extraction(response.raw_text, view) if stage_name == "extraction" else validate_composition(response.raw_text, facts, view)
        write_new(stage / ("pending_ledger.json" if stage_name == "extraction" else "candidate.json"), result)
        return result
    except (json.JSONDecodeError, TypeError, ValueError, KeyError) as error:
        write_new(stage / "failure.json", {"status": "VALIDATION_FAILED", "error_type": type(error).__name__,
                                           "reason": str(error), "training_promotion": False})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-extraction", "run-extraction", "prepare-composition", "run-composition"))
    args = parser.parse_args()
    if args.command == "prepare-extraction":
        result = prepare_extraction()
    elif args.command == "prepare-composition":
        result = prepare_composition()
    else:
        result = run_stage("extraction" if args.command == "run-extraction" else "composition")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
