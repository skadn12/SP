"""Constrained pair-selection QA probe over V3's frozen reviewed fact ledger."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
V3_DIR = HERE.parent / "h4_structured_qa_pilot_v3"
V3_REVIEW_SHA256 = "5105aa8d372ce25e23e58819a72882ed3b6cbc9ba545e2f5f4009ef3206e1a84"
V3_PENDING_SHA256 = "9e82b0a5a347d8ec026f7d8b50c5fc8bb5850b9abc8b114bb2d2901fe19ac3f1"
V3_RAW_SHA256 = "f9cc90b9d68621f4057a30f88b7ff878e645c6b9869c04892361847fc16e48b3"
V3_FAILURE_REASON = "option_not_two_independent_facts"

# These relations were reviewed for this frozen root, not inferred from object
# labels. In particular, the old draft promise does not establish what v2 did.
PAIR_RELATIONS = {
    ("F002", "F011"): "study_plan_then_possible_day",
    ("F002", "F016"): "study_plan_then_calendar_request_withdrawal",
    ("F007", "F018"): "talk_invitation_then_outline_revision",
    ("F007", "F022"): "talk_invitation_then_outline_approval",
    ("F011", "F016"): "possible_day_then_calendar_request_withdrawal",
    ("F011", "F019"): "possible_day_then_specific_availability",
    ("F012", "F016"): "calendar_request_then_withdrawal",
    ("F012", "F019"): "calendar_request_then_specific_availability",
    ("F015", "F016"): "unsent_invitation_then_request_withdrawal",
    ("F018", "F021"): "outline_revision_then_review",
    ("F018", "F022"): "outline_revision_then_approval",
}

INSTRUCTIONS = """Select one two-fact multi-select item using ONLY the prevalidated pair IDs below. Return JSON only. Each pair has two source-reviewed atomic facts from distinct turns and a controlled same-object or related-object relation. You may select a pair ID and two orientations in its listed fact order. You may not compose a new pair, cite another fact, write option wording, or supply gold. The program renders text and derives gold.
Select five DISTINCT pair IDs. Two or three options must have both orientations as_stated; each other option needs at least one opposite orientation. Both claim and opposite_claim were source-reviewed for the same actor, object, version, predicate, and checkpoint. Prefer meaningful cross-window relations; avoid repeated factual relations. If fewer than five suitable, independent pairs exist, return selected_options=[] and explain why. Do not infer that one fact caused another or that v2 fulfills an unversioned promise.
Schema: {"selected_options":[{"pair_id":"P001","orientations":["as_stated","opposite"]}],"abstain_reason":"empty for five options; explain otherwise"}
REVIEWED FACTS AND PREVALIDATED PAIRS:
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


def load_frozen() -> tuple[Any, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    v3 = v3_module()
    if sha((V3_DIR / "reviewed_ids.json").read_bytes()) != V3_REVIEW_SHA256:
        raise ValueError("v3_review_sha_mismatch")
    if sha((V3_DIR / "extraction" / "pending_ledger.json").read_bytes()) != V3_PENDING_SHA256:
        raise ValueError("v3_pending_sha_mismatch")
    raw = (V3_DIR / "composition" / "response_raw.txt").read_bytes()
    failure = v3.strict_json((V3_DIR / "composition" / "failure.json").read_text(encoding="utf-8"))
    if sha(raw) != V3_RAW_SHA256 or failure.get("status") != "VALIDATION_FAILED" or failure.get("reason") != V3_FAILURE_REASON:
        raise ValueError("v3_failure_baseline_mismatch")
    view = v3.surface()
    facts, review_sha = v3.reviewed_facts(view)
    if review_sha != V3_REVIEW_SHA256:
        raise ValueError("v3_review_binding_mismatch")
    baseline = {"source": "V3_UNTOUCHED_RAW_FAILURE", "raw_sha256": V3_RAW_SHA256,
                "failure_reason": V3_FAILURE_REASON, "shape_valid_questions": 0,
                "source_correct_questions": 0, "training_promotion": False}
    return v3, view, facts, baseline


def cross_window(first_ref: str, second_ref: str) -> bool:
    if first_ref == second_ref:
        return False
    first_new, second_new = first_ref.startswith("G02:"), second_ref.startswith("G02:")
    if first_new != second_new:
        return True
    return first_new and first_ref.rsplit(":T", 1)[0] != second_ref.rsplit(":T", 1)[0]


def relation(first: dict[str, Any], second: dict[str, Any]) -> str | None:
    return PAIR_RELATIONS.get(tuple(sorted((first["fact_id"], second["fact_id"]))))


def enumerate_pairs(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    if len({row["fact_id"] for row in facts}) != len(facts):
        raise ValueError("duplicate_reviewed_fact_id")
    for first, second in itertools.combinations(facts, 2):
        first_ref, second_ref = first["evidence"]["ref"], second["evidence"]["ref"]
        if not cross_window(first_ref, second_ref):
            continue
        relation_name = relation(first, second)
        if relation_name is None:
            continue
        ids = [first["fact_id"], second["fact_id"]]
        pairs.append({"pair_id": f"P{len(pairs) + 1:03d}", "fact_ids": ids,
                      "refs": [first_ref, second_ref], "relation": relation_name})
    return pairs


def prompt_for(facts: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> str:
    compact_facts = [{"fact_id": row["fact_id"], "claim": row["claim"],
                      "opposite_claim": row["opposite_claim"], "actor": row["actor"],
                      "object_id": row["object_id"], "version": row["version"],
                      "checkpoint": row["checkpoint"], "evidence_ref": row["evidence"]["ref"]}
                     for row in facts if any(row["fact_id"] in pair["fact_ids"] for pair in pairs)]
    return INSTRUCTIONS + json.dumps({"facts": compact_facts, "pairs": pairs}, ensure_ascii=False, indent=2) + "\n"


def write_new(path: Path, value: Any, *, raw: bool = False) -> None:
    with path.open("x", encoding="utf-8") as handle:
        if raw:
            handle.write(str(value))
        else:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")


def prepare() -> dict[str, Any]:
    v3, view, facts, baseline = load_frozen()
    pairs = enumerate_pairs(facts)
    prompt = prompt_for(facts, pairs)
    stage = HERE / "prepared"
    stage.mkdir(exist_ok=False)
    write_new(stage / "pair_pool.json", {"status": "PREVALIDATED_STRUCTURE_ONLY", "scope": "manually_curated_after_v3_fact_review_for_this_root_only", "pairs": pairs,
                                         "reviewed_fact_count": len(facts), "training_promotion": False})
    write_new(stage / "prompt.txt", prompt, raw=True)
    write_new(stage / "v3_baseline.json", baseline)
    manifest = {"stage": "constrained_pair_composition", "status": "PREPARED_OFFLINE_NOT_EXECUTED",
                "model": v3.MODEL, "prompt_sha256": sha(prompt.encode("utf-8")),
                "pair_pool_sha256": sha((stage / "pair_pool.json").read_bytes()),
                "v3_reviewed_ids_sha256": V3_REVIEW_SHA256, "v3_pending_ledger_sha256": V3_PENDING_SHA256,
                "v3_failed_raw_sha256": V3_RAW_SHA256, "history_sha256": view["history_sha256"],
                "g02_sha256": view["g02_sha256"], "pair_count": len(pairs),
                "pair_pool_scope": "manually_curated_after_v3_fact_review_for_this_root_only",
                "automatic_pair_discovery_yield_measured": False, "training_promotion": False}
    write_new(stage / "manifest.json", manifest)
    return manifest


def materialize(raw: str, facts: list[dict[str, Any]], pairs: list[dict[str, Any]], view: dict[str, Any], *, strict_json) -> dict[str, Any]:
    parsed = strict_json(raw)
    if not isinstance(parsed, dict) or set(parsed) != {"selected_options", "abstain_reason"}:
        raise ValueError("selection_top_level_invalid")
    selections, reason = parsed["selected_options"], parsed["abstain_reason"]
    if not isinstance(selections, list) or not isinstance(reason, str):
        raise ValueError("selection_shape_invalid")
    if not selections:
        if not reason.strip():
            raise ValueError("abstain_reason_missing")
        return {"status": "ABSTAIN", "reason": reason, "questions": [], "training_promotion": False}
    if len(selections) != 5 or reason.strip():
        raise ValueError("five_options_required")
    by_pair = {row["pair_id"]: row for row in pairs}
    by_fact = {row["fact_id"]: row for row in facts}
    if len(by_pair) != len(pairs) or len(by_fact) != len(facts):
        raise ValueError("duplicate_pool_id")
    options: dict[str, str] = {}
    evidence: dict[str, Any] = {}
    gold: list[str] = []
    seen_pairs: set[str] = set()
    for label, selected in zip("ABCDE", selections, strict=True):
        if not isinstance(selected, dict) or set(selected) != {"pair_id", "orientations"}:
            raise ValueError("selected_option_shape_invalid")
        pair_id, orientations = selected["pair_id"], selected["orientations"]
        if not isinstance(pair_id, str) or pair_id not in by_pair or pair_id in seen_pairs:
            raise ValueError("pair_id_unknown_or_reused")
        if not isinstance(orientations, list) or len(orientations) != 2 or any(value not in {"as_stated", "opposite"} for value in orientations):
            raise ValueError("orientation_invalid")
        seen_pairs.add(pair_id)
        pair = by_pair[pair_id]
        if len(pair["fact_ids"]) != 2 or pair["fact_ids"][0] == pair["fact_ids"][1]:
            raise ValueError("pair_not_two_facts")
        records = [by_fact[fact_id] for fact_id in pair["fact_ids"]]
        refs = [row["evidence"]["ref"] for row in records]
        if refs != pair["refs"] or not cross_window(*refs) or relation(*records) != pair["relation"]:
            raise ValueError("pair_pool_corrupt")
        clauses = [{"fact_id": row["fact_id"], "claim": row["claim"] if orient == "as_stated" else row["opposite_claim"],
                    "verdict": "TRUE" if orient == "as_stated" else "FALSE",
                    "evidence_ref": row["evidence"]["ref"], "quote": row["evidence"]["quote"],
                    "actor": row["actor"], "object_id": row["object_id"],
                    "version": row["version"], "checkpoint": row["checkpoint"]}
                   for row, orient in zip(records, orientations, strict=True)]
        if all(row["verdict"] == "TRUE" for row in clauses):
            gold.append(label)
        options[label] = " and ".join(row["claim"].rstrip(" .") for row in clauses) + "."
        evidence[label] = {"pair_id": pair_id, "relation": pair["relation"],
                           "verdict": "TRUE" if label in gold else "FALSE", "clauses": clauses}
    if len(gold) not in (2, 3) or len(set(options.values())) != 5:
        raise ValueError("gold_size_or_option_duplication_invalid")
    selection_sha = sha(json.dumps(selections, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return {"status": "UNREVIEWED_CANDIDATE_NOT_GOLD", "candidate_id": "v4-ms-" + selection_sha[:12],
            "root_id": view["root_id"], "question_type": "multi_select",
            "question": "Which statements are supported by the conversation? Select all that apply.",
            "options": options, "gold": gold, "option_evidence": evidence,
            "history_sha256": view["history_sha256"], "g02_sha256": view["g02_sha256"],
            "v3_failed_raw_sha256": V3_RAW_SHA256, "training_promotion": False}


def run() -> dict[str, Any]:
    v3, view, facts, _baseline = load_frozen()
    stage = HERE / "prepared"
    manifest = v3.strict_json((stage / "manifest.json").read_text(encoding="utf-8"))
    prompt_bytes = (stage / "prompt.txt").read_bytes()
    pool_bytes = (stage / "pair_pool.json").read_bytes()
    pool = v3.strict_json(pool_bytes.decode("utf-8"))
    pairs = enumerate_pairs(facts)
    if (manifest.get("stage") != "constrained_pair_composition" or manifest.get("model") != v3.MODEL
            or manifest.get("prompt_sha256") != sha(prompt_bytes)
            or manifest.get("pair_pool_sha256") != sha(pool_bytes)
            or pool.get("pairs") != pairs or prompt_bytes.decode("utf-8") != prompt_for(facts, pairs)):
        raise ValueError("frozen_prompt_or_pair_pool_mismatch")
    if (manifest.get("v3_reviewed_ids_sha256"), manifest.get("v3_pending_ledger_sha256"), manifest.get("v3_failed_raw_sha256")) != (V3_REVIEW_SHA256, V3_PENDING_SHA256, V3_RAW_SHA256):
        raise ValueError("frozen_v3_provenance_mismatch")
    if len(pairs) < 5:
        result = {"status": "ABSTAIN", "reason": "fewer_than_five_prevalidated_pairs", "questions": [], "training_promotion": False}
        write_new(stage / "candidate.json", result)
        return result
    profile = v3.provider_profile()
    if profile["model"] != v3.MODEL or v3.MODEL != "Qwen/Qwen3.6-35B-A3B":
        raise ValueError("unexpected_model")
    profile["profile_id"] = "h4_constrained_pair_qa_pilot_v4"
    profile["request_parameters"].update(temperature=0, max_tokens=3072, enable_thinking=False)
    profile["output_limit"] = 3072
    profile["http_retry_limit"] = 0
    write_new(stage / "attempt.reserved.json", {"status": "RESERVED_ONE_ATTEMPT", "prompt_sha256": manifest["prompt_sha256"],
                                                   "model": v3.MODEL, "maximum_http_attempts": 1,
                                                   "v3_baseline_shape_valid_questions": 0, "training_promotion": False})

    def on_attempt(payload: dict[str, Any]) -> None:
        write_new(stage / "http_attempt.json", {"attempt_number": 1,
                                                 "payload_sha256": sha(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))})

    try:
        response = v3.credential_client(profile, on_http_attempt=on_attempt).complete(
            prompt=prompt_bytes.decode("utf-8"), max_tokens=3072, json_object=True)
        write_new(stage / "response_raw.txt", response.raw_text, raw=True)
        write_new(stage / "response_meta.json", {"status": "RESPONSE_RECEIVED",
                                                  "raw_sha256": sha(response.raw_text.encode("utf-8")),
                                                  "metadata": response.metadata, "training_promotion": False})
    except Exception as error:
        write_new(stage / "failure.json", {"status": "API_FAILED_NO_RETRY", "error_type": type(error).__name__,
                                           "http_attempt_recorded": (stage / "http_attempt.json").exists(),
                                           "training_promotion": False})
        raise
    try:
        result = materialize(response.raw_text, facts, pairs, view, strict_json=v3.strict_json)
        write_new(stage / "candidate.json", result)
        return result
    except (json.JSONDecodeError, TypeError, ValueError, KeyError) as error:
        write_new(stage / "failure.json", {"status": "VALIDATION_FAILED", "error_type": type(error).__name__,
                                           "reason": str(error), "training_promotion": False})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    args = parser.parse_args()
    result = prepare() if args.command == "prepare" else run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
