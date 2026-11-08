"""Two-call relation-first QA diagnostic. Preparation is offline only."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from typing import Any


HERE = Path(__file__).resolve().parent
V3_DIR = HERE.parent / "h4_structured_qa_pilot_v3"
RELATION_TYPES = frozenset({
    "request_withdrawal", "availability_refinement", "promise_fulfillment",
    "version_decision_change", "revision_approval_identity", "status_persistence",
    "other_two_checkpoint_relation",
})
DECISIONS = frozenset({"supported_nontrivial", "rejected_unsupported", "rejected_ambiguous", "rejected_trivial"})
TURN_REF_RE = re.compile(r"\b(?:R\d{2}|G\d{2}):S\d{2}:T\d{3}\b")
ID_RE = re.compile(r"R\d{3}\Z")

EXTRACT_RULES = """Extract genuine TWO-CHECKPOINT relations from ONLY these spoken turns. Return JSON only.
Each relation is one proposition whose truth requires BOTH exact quoted utterances. Identify the same actor(s), object, version, and natural-language scope across both checkpoints. The supported claim must state the RELATION, such as a prior calendar request being withdrawn, a possible study day being refined to a speaker's availability, or the same v2 outline being revised and approved as an outline for rehearsal. Do NOT join two independently retrievable atomic facts with 'and'. The opposite_claim must be the strict logical contradiction of the SAME relation, with the same actor, object, version, and checkpoints; change only its asserted relational outcome. Do not combine multiple negations, alter the object or version, or turn requests/promises into performed actions.
Two references are evidence checkpoints, NOT proof of chronological occurrence order. Hidden scene metadata, transcript position, and version numbers do not establish when events occurred. Do not claim that v2 was approved AFTER or BECAUSE of revision unless the spoken words establish this. Sarah being free after six does not mean a study time was agreed or an invitation sent. Outline approval does not mean a rehearsal or talk happened. If five distinct defensible relations do not exist, return fewer (possibly zero) and an honest insufficiency reason; do not fabricate to meet a quota. Use unique R001-style IDs.
Schema: {"relations":[{"relation_id":"R001","relation_type":"request_withdrawal|availability_refinement|promise_fulfillment|version_decision_change|revision_approval_identity|status_persistence|other_two_checkpoint_relation","actor":"...","object_id":"...","version":"... or empty","scope":"natural-language checkpoint scope, no raw refs","claim":"one source-supported relational sentence","opposite_claim":"strict same-relation contradiction","checkpoints":[{"ref":"exact turn ref","speaker":"exact speaker","quote":"exact spoken substring"},{"ref":"different exact turn ref","speaker":"exact speaker","quote":"exact spoken substring"}]}],"insufficiency_reason":"empty if enough; otherwise explain"}
VISIBLE TURNS:
"""

COMPOSE_RULES = """Select ONE five-option multi-select question from these source-reviewed TWO-CHECKPOINT relations. Return JSON only. Select five DISTINCT relation IDs, each only once, and one orientation per ID. Choose exactly 2 or 3 as_stated options. opposite uses that record's reviewed direct contradiction of the identical relation. Do not write wording, gold, citations, new claims, or atomic-fact conjunctions. Prefer five substantively different relational judgments; avoid choices answerable from one utterance. If five independent options are impossible, return an empty list and explain. The program renders the unchanged reviewed text and derives gold.
Schema: {"selected_options":[{"relation_id":"R001","orientation":"as_stated|opposite"}],"abstain_reason":"empty for five options; reason otherwise"}
REVIEWED RELATIONS:
"""


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def v3_module():
    spec = importlib.util.spec_from_file_location("h4_structured_qa_pilot_v3_run", V3_DIR / "run.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("v3_module_unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_new(path: Path, value: Any, *, raw: bool = False) -> None:
    with path.open("x", encoding="utf-8") as handle:
        if raw:
            handle.write(value)
        else:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")


def nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}_missing")
    return value


def surface(v3=None) -> dict[str, Any]:
    return (v3 or v3_module()).surface()


def extraction_prompt(view: dict[str, Any]) -> str:
    return EXTRACT_RULES + json.dumps(view["turns"], ensure_ascii=False, indent=2) + "\n"


def relation_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (row["relation_type"], row["actor"].strip().casefold(),
            row["object_id"].strip().casefold(), row["version"].strip().casefold(),
            row["checkpoints"][0]["ref"], row["checkpoints"][1]["ref"])


def validate_extraction(raw: str, view: dict[str, Any], *, strict_json=None) -> dict[str, Any]:
    parsed = (strict_json or v3_module().strict_json)(raw)
    if not isinstance(parsed, dict) or set(parsed) != {"relations", "insufficiency_reason"}:
        raise ValueError("extraction_top_level_invalid")
    relations, reason = parsed["relations"], parsed["insufficiency_reason"]
    if not isinstance(relations, list) or len(relations) > 30 or not isinstance(reason, str):
        raise ValueError("extraction_shape_invalid")
    if len(relations) < 5 and not reason.strip():
        raise ValueError("insufficiency_reason_required")
    if len(relations) >= 5 and reason.strip():
        raise ValueError("insufficiency_reason_inconsistent")
    turns = {turn["turn_ref"]: turn for turn in view["turns"]}
    ids: set[str] = set()
    keys: set[tuple[str, str, str, str, str, str]] = set()
    required = {"relation_id", "relation_type", "actor", "object_id", "version", "scope", "claim", "opposite_claim", "checkpoints"}
    for row in relations:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("relation_shape_invalid")
        rid = nonempty(row["relation_id"], "relation_id")
        if not ID_RE.fullmatch(rid) or rid in ids:
            raise ValueError("relation_id_invalid_or_duplicate")
        ids.add(rid)
        if row["relation_type"] not in RELATION_TYPES:
            raise ValueError("relation_type_invalid")
        for field in ("actor", "object_id", "scope", "claim", "opposite_claim"):
            nonempty(row[field], field)
        if not isinstance(row["version"], str):
            raise ValueError("version_invalid")
        if any(TURN_REF_RE.search(row[field]) for field in ("scope", "claim", "opposite_claim")):
            raise ValueError("turn_ref_leaked_into_proposition")
        if row["claim"].strip() == row["opposite_claim"].strip():
            raise ValueError("opposite_same_as_claim")
        points = row["checkpoints"]
        if not isinstance(points, list) or len(points) != 2:
            raise ValueError("two_checkpoints_required")
        refs: set[str] = set()
        for point in points:
            if not isinstance(point, dict) or set(point) != {"ref", "speaker", "quote"}:
                raise ValueError("checkpoint_shape_invalid")
            ref = nonempty(point["ref"], "checkpoint_ref")
            if ref not in turns or ref in refs:
                raise ValueError("checkpoint_ref_invalid_or_reused")
            refs.add(ref)
            if point["speaker"] != turns[ref]["speaker"] or nonempty(point["quote"], "checkpoint_quote") not in turns[ref]["text"]:
                raise ValueError("checkpoint_speaker_or_quote_mismatch")
        first_ref, second_ref = (point["ref"] for point in points)
        first_g02, second_g02 = first_ref.startswith("G02:"), second_ref.startswith("G02:")
        if not (first_g02 != second_g02 or
                (first_g02 and second_g02 and first_ref.rsplit(":T", 1)[0] != second_ref.rsplit(":T", 1)[0])):
            raise ValueError("relation_not_cross_window")
        key = relation_key(row)
        if key in keys:
            raise ValueError("duplicate_relation_key")
        keys.add(key)
    return {"status": "PENDING_SEMANTIC_REVIEW", "root_id": view["root_id"],
            "history_sha256": view["history_sha256"], "g02_sha256": view["g02_sha256"],
            "relations": relations, "insufficiency_reason": reason, "training_promotion": False}


def prepare_extraction() -> dict[str, Any]:
    v3 = v3_module()
    view = surface(v3)
    prompt = extraction_prompt(view)
    stage = HERE / "extraction"
    stage.mkdir(exist_ok=False)
    write_new(stage / "prompt.txt", prompt, raw=True)
    manifest = {"stage": "extraction", "status": "PREPARED_OFFLINE_NOT_EXECUTED",
                "model": v3.MODEL, "prompt_sha256": sha(prompt.encode("utf-8")),
                "history_sha256": view["history_sha256"], "g02_sha256": view["g02_sha256"],
                "visible_history_refs": list(v3.HISTORY_REFS), "visible_g02_turn_count": 10,
                "maximum_http_attempts": 1, "training_promotion": False}
    write_new(stage / "manifest.json", manifest)
    return manifest


def reviewed_relations(view: dict[str, Any], *, strict_json=None) -> tuple[list[dict[str, Any]], str, str]:
    parse = strict_json or v3_module().strict_json
    stage = HERE / "extraction"
    raw = (stage / "response_raw.txt").read_text(encoding="utf-8")
    pending_bytes = (stage / "pending_relations.json").read_bytes()
    pending = parse(pending_bytes.decode("utf-8"))
    if pending != validate_extraction(raw, view, strict_json=parse):
        raise ValueError("pending_relations_not_raw_extraction")
    review_path = HERE / "reviewed_relations.json"
    review = parse(review_path.read_text(encoding="utf-8"))
    expected = {"status", "attestation", "reviewer", "pending_relations_sha256", "assessments"}
    if not isinstance(review, dict) or set(review) != expected:
        raise ValueError("review_shape_invalid")
    if (review["status"] != "SOURCE_REVIEWED" or
            review["attestation"] != "TWO_SPAN_CLAIM_AND_STRICT_CONTRADICTION_REVIEWED" or
            not isinstance(review["reviewer"], str) or not review["reviewer"].strip() or
            review["pending_relations_sha256"] != sha(pending_bytes)):
        raise ValueError("review_attestation_or_hash_invalid")
    assessments = review["assessments"]
    if not isinstance(assessments, list) or len(assessments) != len(pending["relations"]):
        raise ValueError("review_coverage_invalid")
    by_id = {row["relation_id"]: row for row in pending["relations"]}
    seen: set[str] = set()
    accepted: list[dict[str, Any]] = []
    for assessment in assessments:
        if not isinstance(assessment, dict) or set(assessment) != {"relation_id", "decision", "reason"}:
            raise ValueError("assessment_shape_invalid")
        rid = assessment["relation_id"]
        if not isinstance(rid, str) or rid not in by_id or rid in seen or assessment["decision"] not in DECISIONS:
            raise ValueError("assessment_id_or_decision_invalid")
        nonempty(assessment["reason"], "assessment_reason")
        seen.add(rid)
        if assessment["decision"] == "supported_nontrivial":
            accepted.append(by_id[rid])
    if seen != set(by_id):
        raise ValueError("review_coverage_invalid")
    return accepted, sha(review_path.read_bytes()), sha(pending_bytes)


def composition_prompt(relations: list[dict[str, Any]]) -> str:
    return COMPOSE_RULES + json.dumps(relations, ensure_ascii=False, indent=2) + "\n"


def prepare_composition() -> dict[str, Any]:
    v3 = v3_module()
    view = surface(v3)
    relations, review_sha, pending_sha = reviewed_relations(view, strict_json=v3.strict_json)
    stage = HERE / "composition"
    stage.mkdir(exist_ok=False)
    if len(relations) < 5:
        result = {"status": "ABSTAIN_BEFORE_COMPOSITION", "reason": "fewer_than_five_reviewed_nontrivial_relations",
                  "reviewed_relation_count": len(relations), "questions": [], "training_promotion": False}
        write_new(stage / "abstention.json", result)
        return result
    prompt = composition_prompt(relations)
    write_new(stage / "prompt.txt", prompt, raw=True)
    manifest = {"stage": "composition", "status": "PREPARED_OFFLINE_NOT_EXECUTED",
                "model": v3.MODEL, "prompt_sha256": sha(prompt.encode("utf-8")),
                "history_sha256": view["history_sha256"], "g02_sha256": view["g02_sha256"],
                "reviewed_relations_sha256": review_sha, "pending_relations_sha256": pending_sha,
                "reviewed_relation_count": len(relations), "maximum_http_attempts": 1,
                "training_promotion": False}
    write_new(stage / "manifest.json", manifest)
    return manifest


def materialize(raw: str, relations: list[dict[str, Any]], view: dict[str, Any], *, strict_json=None) -> dict[str, Any]:
    parsed = (strict_json or v3_module().strict_json)(raw)
    if not isinstance(parsed, dict) or set(parsed) != {"selected_options", "abstain_reason"}:
        raise ValueError("selection_top_level_invalid")
    choices, reason = parsed["selected_options"], parsed["abstain_reason"]
    if not isinstance(choices, list) or not isinstance(reason, str):
        raise ValueError("selection_shape_invalid")
    if not choices:
        nonempty(reason, "abstain_reason")
        return {"status": "ABSTAIN", "reason": reason, "questions": [], "training_promotion": False}
    if len(choices) != 5 or reason.strip():
        raise ValueError("five_options_required")
    by_id = {row["relation_id"]: row for row in relations}
    if len(by_id) != len(relations):
        raise ValueError("duplicate_reviewed_relation_id")
    options: dict[str, str] = {}
    evidence: dict[str, Any] = {}
    gold: list[str] = []
    seen_ids: set[str] = set()
    seen_keys: set[tuple[str, str, str, str, str, str]] = set()
    for label, choice in zip("ABCDE", choices, strict=True):
        if not isinstance(choice, dict) or set(choice) != {"relation_id", "orientation"}:
            raise ValueError("selected_option_shape_invalid")
        rid, orientation = choice["relation_id"], choice["orientation"]
        if not isinstance(rid, str) or rid not in by_id or rid in seen_ids:
            raise ValueError("relation_id_unknown_or_reused")
        if orientation not in {"as_stated", "opposite"}:
            raise ValueError("orientation_invalid")
        seen_ids.add(rid)
        relation = by_id[rid]
        key = relation_key(relation)
        if key in seen_keys:
            raise ValueError("duplicate_relation_key")
        seen_keys.add(key)
        statement = relation["claim"] if orientation == "as_stated" else relation["opposite_claim"]
        if TURN_REF_RE.search(statement):
            raise ValueError("turn_ref_leaked_into_option")
        options[label] = statement
        if orientation == "as_stated":
            gold.append(label)
        evidence[label] = {"relation_id": rid, "relation_type": relation["relation_type"],
                           "verdict": "TRUE" if orientation == "as_stated" else "FALSE",
                           "actor": relation["actor"], "object_id": relation["object_id"],
                           "version": relation["version"], "scope": relation["scope"],
                           "checkpoints": relation["checkpoints"]}
    if len(gold) not in (2, 3) or len(set(options.values())) != 5:
        raise ValueError("gold_size_or_duplicate_options")
    selection_sha = sha(json.dumps(choices, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return {"status": "UNREVIEWED_CANDIDATE_NOT_GOLD", "candidate_id": "v5-ms-" + selection_sha[:12],
            "root_id": view["root_id"], "question_type": "multi_select",
            "question": "Which statements are supported by the conversation? Select all that apply.",
            "options": options, "gold": gold, "option_evidence": evidence,
            "history_sha256": view["history_sha256"], "g02_sha256": view["g02_sha256"],
            "training_promotion": False}


def run_stage(stage_name: str) -> dict[str, Any]:
    v3 = v3_module()
    view = surface(v3)
    stage = HERE / stage_name
    manifest = v3.strict_json((stage / "manifest.json").read_text(encoding="utf-8"))
    prompt_bytes = (stage / "prompt.txt").read_bytes()
    if (manifest.get("stage") != stage_name or manifest.get("model") != v3.MODEL or
            manifest.get("prompt_sha256") != sha(prompt_bytes) or
            manifest.get("history_sha256") != view["history_sha256"] or
            manifest.get("g02_sha256") != view["g02_sha256"]):
        raise ValueError("frozen_manifest_mismatch")
    relations: list[dict[str, Any]] = []
    if stage_name == "extraction":
        expected_prompt = extraction_prompt(view)
    elif stage_name == "composition":
        relations, review_sha, pending_sha = reviewed_relations(view, strict_json=v3.strict_json)
        if (len(relations) < 5 or manifest.get("reviewed_relations_sha256") != review_sha or
                manifest.get("pending_relations_sha256") != pending_sha or
                manifest.get("reviewed_relation_count") != len(relations)):
            raise ValueError("reviewed_relations_or_manifest_mismatch")
        expected_prompt = composition_prompt(relations)
    else:
        raise ValueError("unknown_stage")
    if prompt_bytes.decode("utf-8") != expected_prompt:
        raise ValueError("frozen_prompt_mismatch")
    profile = v3.provider_profile()
    if profile["model"] != v3.MODEL or v3.MODEL != "Qwen/Qwen3.6-35B-A3B":
        raise ValueError("unexpected_model")
    profile["profile_id"] = f"h4_relation_qa_pilot_v5_{stage_name}"
    limit = 6144 if stage_name == "extraction" else 2048
    profile["request_parameters"].update(temperature=0, max_tokens=limit, enable_thinking=False)
    profile["output_limit"] = limit
    profile["http_retry_limit"] = 0
    write_new(stage / "attempt.reserved.json", {"status": "RESERVED_ONE_ATTEMPT",
                "prompt_sha256": manifest["prompt_sha256"], "model": v3.MODEL,
                "maximum_http_attempts": 1, "training_promotion": False})

    def on_attempt(payload: dict[str, Any]) -> None:
        write_new(stage / "http_attempt.json", {"attempt_number": 1,
                  "payload_sha256": sha(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))})

    try:
        response = v3.credential_client(profile, on_http_attempt=on_attempt).complete(
            prompt=prompt_bytes.decode("utf-8"), max_tokens=limit, json_object=True)
        write_new(stage / "response_raw.txt", response.raw_text, raw=True)
        write_new(stage / "response_meta.json", {"status": "RESPONSE_RECEIVED",
                  "raw_sha256": sha(response.raw_text.encode("utf-8")), "metadata": response.metadata,
                  "training_promotion": False})
    except Exception as error:
        write_new(stage / "failure.json", {"status": "API_FAILED_NO_RETRY",
                  "error_type": type(error).__name__, "http_attempt_recorded": (stage / "http_attempt.json").exists(),
                  "training_promotion": False})
        raise
    try:
        result = (validate_extraction(response.raw_text, view, strict_json=v3.strict_json)
                  if stage_name == "extraction" else
                  materialize(response.raw_text, relations, view, strict_json=v3.strict_json))
        write_new(stage / ("pending_relations.json" if stage_name == "extraction" else "candidate.json"), result)
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
