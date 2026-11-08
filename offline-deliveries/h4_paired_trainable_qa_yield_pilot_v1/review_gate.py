"""Blind, source-bound semantic review of untouched paired-pilot candidates."""

from __future__ import annotations

import hashlib
import json
from typing import Any


CHECKS = (
    "shape", "source_supported", "two_independent_turns", "two_distinct_relations",
    "same_actor_object_version_checkpoint", "direct_false_contradiction",
    "visible_occurrence_order", "nontrivial", "no_answer_cue", "deduplicated",
)
REASONS = frozenset({
    "shape", "unsupported_true", "unsupported_false", "status_upgrade",
    "actor_object_swap", "checkpoint_ambiguous", "one_turn_lookup",
    "redundant_clause", "answer_cue", "chronology_unproved", "duplicate", "other",
})


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _text(value: Any, reason: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), reason)
    return value


def _candidate_view(row: dict[str, Any]) -> dict[str, Any]:
    question_type = row.get("question_type")
    _require(question_type in ("multi_select", "ordering"), "question_type_invalid")
    field = "options" if question_type == "multi_select" else "items"
    values = row.get(field)
    gold = row.get("gold")
    _require(isinstance(values, dict), "choices_invalid")
    labels = list(values)
    _require((question_type == "multi_select" and 4 <= len(labels) <= 6 and labels == list("ABCDEF")[:len(labels)])
             or (question_type == "ordering" and labels == [f"I{i}" for i in range(1, 5)]), "choice_labels_invalid")
    _require(all(isinstance(value, str) and value.strip() for value in values.values())
             and len(set(values.values())) == len(values), "choice_text_invalid")
    _require(isinstance(gold, list) and len(gold) == len(set(gold))
             and all(isinstance(label, str) and label in values for label in gold), "gold_invalid")
    _require((question_type == "multi_select" and len(gold) in (2, 3))
             or (question_type == "ordering" and len(gold) == 4 and set(gold) == set(labels)), "gold_size_invalid")
    view = {"question_type": question_type,
            "question": _text(row.get("question"), "question_missing"), field: values}
    if question_type == "ordering":
        view["proposed_order"] = gold
    return view


def _evidence_view(row: dict[str, Any], view: dict[str, Any]) -> dict[str, Any]:
    """Present either arm's citations under the same reviewer-facing keys."""
    question_type = view["question_type"]
    choices: dict[str, list[dict[str, Any]]] = {}
    field = "options" if question_type == "multi_select" else "items"
    for label in view[field]:
        if question_type == "multi_select":
            source = row.get("option_evidence", {}).get(label, {})
            if "clauses" in source:
                choices[label] = [{"claim": clause.get("claim", ""), "refs": clause.get("refs", []),
                                   "quote": clause.get("quote", "")}
                                  for clause in source["clauses"] if isinstance(clause, dict)]
            else:
                choices[label] = [{"claim": view[field][label], "refs": [point.get("ref")],
                                   "quote": point.get("quote", "")}
                                  for point in source.get("checkpoints", []) if isinstance(point, dict)]
        else:
            source = row.get("item_evidence", {}).get(label, {})
            if source:
                choices[label] = [{"claim": view[field][label], "refs": source.get("refs", []),
                                   "quote": source.get("quote", "")}]
            else:
                event = row.get("event_evidence", {}).get(label, {})
                if isinstance(event, dict) and isinstance(event.get("evidence"), dict):
                    point = event["evidence"]
                    choices[label] = [{"claim": view[field][label],
                                       "refs": [point.get("ref")], "quote": point.get("quote", "")}]
                else:
                    choices[label] = []
    edges = []
    if question_type == "ordering":
        raw_edges = row.get("adjacent_edge_evidence", row.get("adjacent_edges", []))
        for index, edge in enumerate(raw_edges if isinstance(raw_edges, list) else []):
            if not isinstance(edge, dict):
                continue
            temporal = edge.get("temporal_evidence", {})
            edges.append({"before_label": view["proposed_order"][index] if index < 3 else "",
                          "after_label": view["proposed_order"][index + 1] if index < 3 else "",
                          "before_refs": edge.get("before_refs", [edge.get("before_ref")]),
                          "after_refs": edge.get("after_refs", [edge.get("after_ref")]),
                          "temporal_refs": edge.get("order_refs", [temporal.get("ref")]),
                          "temporal_quote": temporal.get("quote", "")})
    return {"choices": choices, "edges": edges}


def prepare_review_inputs(artifact_bytes: bytes, surface: dict[str, Any],
                          question_bank_sha256: str, shape_audit: dict[str, Any] | None = None
                          ) -> dict[str, Any]:
    """Return anonymous packets and mechanical skips; arm identity stays with caller."""
    _require(isinstance(artifact_bytes, bytes), "artifact_bytes_required")
    _require(isinstance(question_bank_sha256, str) and len(question_bank_sha256) == 64,
             "question_bank_sha256_invalid")
    _require(isinstance(surface, dict) and isinstance(surface.get("turns"), list)
             and isinstance(surface.get("root_id"), str)
             and isinstance(surface.get("source_sha256"), str), "surface_invalid")
    try:
        payload = json.loads(artifact_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return {"packets": [], "skipped": [{"index": None, "reason": "artifact_json_invalid",
                                             "detail": str(error)[:200]}]}
    if not isinstance(payload, dict) or not isinstance(payload.get("questions"), list):
        detail = (list(shape_audit.get("errors", [])) if shape_audit is not None
                  else ["artifact_questions_invalid"])
        return {"packets": [], "skipped": [{"index": None, "reason": "shape_failed",
                                             "detail": detail or ["artifact_questions_invalid"]}]}
    over_multi_cap = len(payload["questions"]) > 2
    over_order_cap = len(payload["questions"]) > 1 and any(
        isinstance(row, dict) and row.get("question_type") == "ordering"
        for row in payload["questions"])
    if over_multi_cap or over_order_cap:
        return {"packets": [], "skipped": [
            {"index": index, "reason": "candidate_cap_exceeded", "detail": []}
            for index in range(len(payload["questions"]))]}
    audit_rows = shape_audit.get("candidates", []) if shape_audit is not None else None
    if shape_audit is not None:
        _require(isinstance(audit_rows, list) and len(audit_rows) == len(payload["questions"]),
                 "shape_audit_count_mismatch")
        _require(shape_audit.get("source_sha256") == surface["source_sha256"],
                 "shape_audit_source_mismatch")
    packets, skipped = [], []
    for index, row in enumerate(payload["questions"]):
        if shape_audit is not None and (shape_audit.get("errors") or
                                        audit_rows[index].get("status") != "PASS_SHAPE_ONLY_NOT_GOLD"):
            errors = list(shape_audit.get("errors", [])) + list(audit_rows[index].get("errors", []))
            skipped.append({"index": index, "reason": "shape_failed", "detail": errors})
            continue
        try:
            _require(isinstance(row, dict), "candidate_not_object")
            view = _candidate_view(row)
        except ValueError as error:
            skipped.append({"index": index, "reason": "shape_failed", "detail": [str(error)]})
            continue
        candidate_sha = _digest(row)
        anonymous_id = "Q" + _digest([surface["source_sha256"], _sha(artifact_bytes),
                                        index, candidate_sha])[:16]
        packet = {"anonymous_id": anonymous_id,
                  "binding": {"root_id": surface["root_id"],
                              "source_sha256": surface["source_sha256"],
                              "surface_sha256": _digest(surface),
                              "question_bank_sha256": question_bank_sha256,
                              "artifact_sha256": _sha(artifact_bytes),
                              "candidate_index": index,
                              "candidate_sha256": candidate_sha},
                  "candidate": view, "cited_evidence": _evidence_view(row, view),
                  "source_turns": surface["turns"]}
        packets.append(packet)
    return {"packets": packets, "skipped": skipped}


def review_template(packet: dict[str, Any]) -> dict[str, Any]:
    """All blanks require an explicit reviewer judgment; no default passes."""
    candidate = packet["candidate"]
    field = "options" if candidate["question_type"] == "multi_select" else "items"
    edges = []
    if candidate["question_type"] == "ordering":
        edges = [{"before_label": before, "after_label": after, "explicit_order": None,
                  "before_ref": "", "after_ref": "", "temporal_ref": "", "temporal_quote": "",
                  "reason": ""} for before, after in zip(candidate["proposed_order"], candidate["proposed_order"][1:])]
    return {"anonymous_id": packet["anonymous_id"], "packet_sha256": _digest(packet),
            "binding": packet["binding"], "reviewer": "", "review_minutes": None,
            "option_reviews": [{"label": label, "verdict": "", "same_scope": None,
                                "direct_evidence": None, "checked_facts": [], "reason": ""}
                               for label in candidate[field]],
            "edge_reviews": edges, "checks": {key: None for key in CHECKS},
            "dedup": {"tested_relation_keys": [], "temporal_chain_key": "",
                      "prior_bank_result": "", "within_root_result": "",
                      "matched_refs": [], "reason": ""},
            "decision": "", "rejection_reasons": [], "training_promoted": False}


def validate_review(packet: dict[str, Any], review: dict[str, Any],
                    artifact_bytes: bytes, surface: dict[str, Any],
                    question_bank_sha256: str,
                    shape_audit: dict[str, Any] | None = None) -> dict[str, Any]:
    """Recreate the packet from immutable inputs, then validate a complete review."""
    recreated = prepare_review_inputs(artifact_bytes, surface, question_bank_sha256, shape_audit)
    expected = {item["anonymous_id"]: item for item in recreated["packets"]}
    _require(isinstance(packet, dict) and packet.get("anonymous_id") in expected
             and packet == expected[packet["anonymous_id"]], "packet_not_bound_to_raw_candidate")
    template = review_template(packet)
    _require(isinstance(review, dict) and set(review) == set(template), "review_fields_invalid")
    for key in ("anonymous_id", "packet_sha256", "binding", "training_promoted"):
        _require(review[key] == template[key], "review_binding_invalid")
    _text(review["reviewer"], "reviewer_missing")
    _require(type(review["review_minutes"]) in (int, float) and review["review_minutes"] > 0,
             "review_minutes_invalid")
    candidate = packet["candidate"]
    question_type = candidate["question_type"]
    original_rows = json.loads(artifact_bytes)["questions"]
    original_index = packet["binding"]["candidate_index"]
    original = original_rows[original_index]
    gold = original["gold"]
    labels = list(candidate["options" if question_type == "multi_select" else "items"])
    refs = {row["turn_ref"]: row["text"] for row in surface["turns"]}
    _require(isinstance(review["option_reviews"], list)
             and [item.get("label") for item in review["option_reviews"] if isinstance(item, dict)] == labels,
             "option_review_labels_invalid")
    all_fact_keys, all_fact_refs = set(), set()
    option_fact_refs: dict[str, set[str]] = {}
    for item in review["option_reviews"]:
        _require(set(item) == set(template["option_reviews"][0]), "option_review_fields_invalid")
        allowed = {"supported", "contradicted", "ambiguous"} if question_type == "multi_select" else {"observed", "unsupported", "ambiguous"}
        _require(item["verdict"] in allowed and type(item["same_scope"]) is bool
                 and type(item["direct_evidence"]) is bool, "option_review_incomplete")
        _text(item["reason"], "option_review_reason_missing")
        facts = item["checked_facts"]
        _require(isinstance(facts, list) and facts, "checked_facts_missing")
        option_fact_refs[item["label"]] = set()
        for fact in facts:
            _require(isinstance(fact, dict) and set(fact) == {"relation_key", "ref", "quote"},
                     "checked_fact_fields_invalid")
            key = _text(fact["relation_key"], "relation_key_missing")
            ref = _text(fact["ref"], "fact_ref_missing")
            quote = _text(fact["quote"], "fact_quote_missing")
            _require(ref in refs and quote in refs[ref], "checked_fact_not_exact_source")
            all_fact_keys.add(key)
            all_fact_refs.add(ref)
            option_fact_refs[item["label"]].add(ref)
    _require(isinstance(review["edge_reviews"], list)
             and len(review["edge_reviews"]) == len(template["edge_reviews"]),
             "edge_reviews_incomplete")
    for submitted, blank in zip(review["edge_reviews"], template["edge_reviews"]):
        _require(isinstance(submitted, dict) and set(submitted) == set(blank)
                 and (submitted["before_label"], submitted["after_label"]) ==
                 (blank["before_label"], blank["after_label"]), "edge_review_fields_invalid")
        _require(type(submitted["explicit_order"]) is bool, "edge_order_judgment_missing")
        for key in ("before_ref", "after_ref", "temporal_ref"):
            _require(_text(submitted[key], "edge_ref_missing") in refs, "edge_ref_not_in_source")
        _require(_text(submitted["temporal_quote"], "temporal_quote_missing")
                 in refs[submitted["temporal_ref"]], "temporal_quote_not_exact")
        _require(submitted["before_ref"] != submitted["after_ref"]
                 and submitted["before_ref"] in option_fact_refs[submitted["before_label"]]
                 and submitted["after_ref"] in option_fact_refs[submitted["after_label"]],
                 "edge_not_bound_to_reviewed_events")
        _text(submitted["reason"], "edge_review_reason_missing")
    _require(isinstance(review["checks"], dict) and set(review["checks"]) == set(CHECKS)
             and all(type(value) is bool for value in review["checks"].values()),
             "checks_incomplete")
    dedup = review["dedup"]
    _require(isinstance(dedup, dict) and set(dedup) == set(template["dedup"]),
             "dedup_fields_invalid")
    relation_keys = dedup["tested_relation_keys"]
    _require(isinstance(relation_keys, list) and all(isinstance(key, str) and key.strip() for key in relation_keys)
             and len(set(relation_keys)) == len(relation_keys)
             and set(relation_keys).issubset(all_fact_keys), "dedup_relation_keys_invalid")
    _require(isinstance(dedup["temporal_chain_key"], str), "temporal_chain_key_invalid")
    _require(dedup["prior_bank_result"] in {"unique", "duplicate", "ambiguous"}
             and dedup["within_root_result"] in {"unique", "duplicate", "ambiguous"},
             "dedup_judgment_missing")
    _require(isinstance(dedup["matched_refs"], list)
             and all(isinstance(ref, str) and ref.strip() for ref in dedup["matched_refs"]),
             "dedup_matches_invalid")
    _text(dedup["reason"], "dedup_reason_missing")
    _require(review["decision"] in {"pass_to_m0", "reject"}
             and isinstance(review["rejection_reasons"], list)
             and len(set(review["rejection_reasons"])) == len(review["rejection_reasons"])
             and set(review["rejection_reasons"]).issubset(REASONS), "decision_invalid")
    if review["decision"] == "pass_to_m0":
        _require(not review["rejection_reasons"] and all(review["checks"].values()),
                 "passing_checks_inconsistent")
        _require(len(all_fact_refs) >= 2 and len(set(relation_keys)) >= 2,
                 "insufficient_independent_evidence")
        _require(set(relation_keys) == all_fact_keys, "dedup_keys_omit_reviewed_relations")
        _require(dedup["prior_bank_result"] == dedup["within_root_result"] == "unique"
                 and not dedup["matched_refs"], "dedup_pass_inconsistent")
        _require(all(item["same_scope"] and item["direct_evidence"] for item in review["option_reviews"]),
                 "option_evidence_pass_inconsistent")
        if question_type == "multi_select":
            _require(all(len(option_fact_refs[label]) >= 2 for label in labels),
                     "multi_option_requires_two_evidence_turns")
            _require(all(item["verdict"] == ("supported" if item["label"] in gold else "contradicted")
                         for item in review["option_reviews"]), "option_verdict_pass_inconsistent")
        else:
            _require(all(item["verdict"] == "observed" for item in review["option_reviews"])
                     and all(item["explicit_order"] for item in review["edge_reviews"])
                     and bool(dedup["temporal_chain_key"].strip()), "ordering_pass_inconsistent")
    else:
        _require(bool(review["rejection_reasons"]) and not all(review["checks"].values()),
                 "rejection_checks_inconsistent")
    option_counts = {verdict: sum(item["verdict"] == verdict for item in review["option_reviews"])
                     for verdict in ({"supported", "contradicted", "ambiguous"} if question_type == "multi_select"
                                     else {"observed", "unsupported", "ambiguous"})}
    return {"anonymous_id": packet["anonymous_id"], "index": original_index,
            "candidate_id": original.get("candidate_id"), "root_id": surface["root_id"],
            "source_sha256": surface["source_sha256"], "question_type": question_type,
            "decision": review["decision"],
            "status": "PASS_TO_M0" if review["decision"] == "pass_to_m0" else "REJECTED",
            "rejection_reasons": review["rejection_reasons"], "reviewer": review["reviewer"],
            "review_minutes": review["review_minutes"], "review_sha256": _digest(review),
            "question_sha256": packet["binding"]["candidate_sha256"],
            "artifact_sha256": packet["binding"]["artifact_sha256"],
            "check_results": review["checks"],
            "tested_relation_keys": dedup["tested_relation_keys"],
            "temporal_chain_key": dedup["temporal_chain_key"],
            "option_decision_counts": option_counts,
            "edge_decision_counts": {"explicit": sum(item["explicit_order"] for item in review["edge_reviews"]),
                                     "unproved": sum(not item["explicit_order"] for item in review["edge_reviews"])},
            "training_promoted": False}


def summarize_reviews(results: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> dict[str, Any]:
    """Report gate denominators without treating M0 eligibility as training yield."""
    semantic_checks = set(CHECKS) - {"deduplicated"}
    semantic_pass = sum(all(row["check_results"][key] for key in semantic_checks)
                        for row in results)
    dedup_pass = sum(all(row["check_results"].values()) for row in results)
    counts = {"emitted": len(results) + sum(row["index"] is not None for row in skipped),
              "response_shape_failures": sum(row["index"] is None for row in skipped),
              "shape_pass": len(results),
              "semantic_pass": semantic_pass, "dedup_pass": dedup_pass,
              "m0_eligible": sum(row["status"] == "PASS_TO_M0" for row in results),
              "training_promoted": 0}
    reasons: dict[str, int] = {}
    for row in results:
        for reason in row["rejection_reasons"]:
            reasons[reason] = reasons.get(reason, 0) + 1
    for row in skipped:
        reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
    return {"counts": counts, "failure_reasons": reasons,
            "review_minutes": sum(row["review_minutes"] for row in results)}


def aggregate_reviews(reviewed: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> dict[str, Any]:
    """Stage-level artifact consumed by M0 after it revalidates each raw review."""
    _require(all(isinstance(row, dict) and set(row) == {"packet", "review", "result"}
                 for row in reviewed), "reviewed_entry_fields_invalid")
    results = [row["result"] for row in reviewed]
    _require(len({row["index"] for row in results}) == len(results), "duplicate_review_index")
    _require(not set(row["index"] for row in results) & set(row["index"] for row in skipped),
             "review_and_skip_overlap")
    return {"reviews": reviewed, "skipped": skipped, "summary": summarize_reviews(results, skipped),
            "training_promoted": 0}
