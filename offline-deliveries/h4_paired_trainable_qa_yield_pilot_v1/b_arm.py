"""Offline, source-bound relation-first QA arm for the paired pilot.

This module makes no network requests and writes no files. The orchestrator
persists raw model responses, reviews, pools, prompts, and rendered candidates.
"""

from __future__ import annotations

from hashlib import sha256
from itertools import combinations
import json
import re
from typing import Any


KINDS = ("relations", "occurrences", "order_edges")
ID_FIELDS = {"relations": "relation_id", "occurrences": "event_id", "order_edges": "edge_id"}
ID_PATTERNS = {"relations": re.compile(r"R\d{3}\Z"),
               "occurrences": re.compile(r"E\d{3}\Z"),
               "order_edges": re.compile(r"O\d{3}\Z")}
STATES = frozenset({"proposed", "offered", "accepted", "scheduled", "started", "completed", "cancelled"})
ORDERED_STATES = ("proposed", "offered", "accepted", "scheduled", "started", "completed", "cancelled")
RELATION_TYPES = frozenset({"request_withdrawal", "availability_refinement", "promise_fulfillment",
                            "version_decision_change", "revision_approval_identity", "status_persistence",
                            "other_two_checkpoint_relation"})
REF_RE = re.compile(r"(?:[A-Za-z0-9-]+:P\d{2}|R\d{2}:S\d{2}):T\d{3}")

EXTRACTION_RULES = """Extract only source-supported, typed records from the VISIBLE SPOKEN TURNS below. Return strict JSON, no markdown.
Each relation must express ONE nontrivial two-checkpoint change or persistence about the SAME actor, object, version and named scope. Supply typed fields only: the program renders both the supported state-transition claim and a directly contradicted same-checkpoint alternative; do not write option text, a counterclaim or gold. Write scope as a short noun phrase with no temporal preposition. The two checkpoint quotes must directly support the typed transition, including its temporal direction. Do not join two unrelated facts with 'and'. Record proposed, offered, accepted, scheduled, started, completed and cancelled distinctly; an intention, offer, calendar request or approval is not a completed action. Do not infer occurrence order from turn order, scene ID, version number, or metadata. Report time is not exact occurrence time. If a field is unknown, use an empty string; never invent a date. Each quote must be an exact substring of the cited spoken turn. Extract fewer records, including zero, when evidence is insufficient.
An occurrence is an action explicitly reported as having happened; its status must be started or completed. For ordering, include only explicit temporal claims in visible speech, each citing the before and after event IDs and a verbatim temporal quote. Do not create an edge from transcript position or hidden scene metadata. Each edge needs both event refs plus the quoted temporal evidence. A graph with no defensible four-event chain is valid.
Schema: {"relations":[{"relation_id":"R001","relation_type":"request_withdrawal|availability_refinement|promise_fulfillment|version_decision_change|revision_approval_identity|status_persistence|other_two_checkpoint_relation","actor":"...","object_id":"...","version":"... or empty","action":"...","scope":"same natural-language checkpoint scope","state_before":"proposed|offered|accepted|scheduled|started|completed|cancelled","state_after":"proposed|offered|accepted|scheduled|started|completed|cancelled","occurrence_time":"explicit occurrence time or empty","report_time":"explicit report time or empty","checkpoints":[{"ref":"exact turn ref","speaker":"exact speaker","quote":"exact substring"},{"ref":"different exact turn ref","speaker":"exact speaker","quote":"exact substring"}]}],"occurrences":[{"event_id":"E001","actor":"...","object_id":"...","version":"... or empty","action":"event action as a short noun phrase","status":"started|completed","occurrence_time":"explicit occurrence time or empty","report_time":"explicit report time or empty","evidence":{"ref":"exact turn ref","speaker":"exact speaker","quote":"exact substring"}}],"order_edges":[{"edge_id":"O001","before_event_id":"E001","after_event_id":"E002","temporal_evidence":{"ref":"exact turn ref containing explicit before/after language","speaker":"exact speaker","quote":"exact substring"}}],"insufficiency_reason":"empty or explanation of sparse records"}
VISIBLE SPOKEN TURNS:
"""

MULTI_RULES = """Select up to TWO distinct multi-select questions from these reviewed same-scope relation pairs. Return strict JSON only. For each question select 4-6 DISTINCT pair IDs and one orientation per pair; exactly 2-3 must be as_stated. Select relational judgments requiring two evidence turns, with distinct relations. You may choose only pair IDs and orientations. Do not write wording, citations, facts or gold. A pair's contradicted option is the program-rendered alternative directly refuted by its reviewed state transition at the SAME actor, object, version and checkpoints.
Schema: {"questions":[{"selected_options":[{"pair_id":"P001","orientation":"as_stated|contradicted"}]}],"abstain_reason":"empty if two questions, otherwise reason"}
REVIEWED PAIR POOL:
"""

ORDER_RULES = """Select at most ONE four-item ordering question from the reviewed source-visible occurrence-order chains. Return strict JSON only. Select only a chain ID. Do not write item wording, citations, a chronology or gold. The program renders the four event claims and derives order from reviewed explicit temporal edges. If no valid chain exists, abstain.
Schema: {"selected_chain_id":"C001 or empty","abstain_reason":"empty if selected, otherwise reason"}
REVIEWED ORDERING CHAINS:
"""


def _strict_json(raw: str) -> Any:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("duplicate_json_key")
            out[key] = value
        return out

    return json.loads(raw, object_pairs_hook=no_duplicates)


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8")).hexdigest()


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(name + "_invalid")
    return value


def _shape(row: Any, fields: set[str]) -> None:
    if not isinstance(row, dict) or set(row) != fields:
        raise ValueError("record_shape_invalid")


def _quote(point: Any, turns: dict[str, dict[str, str]]) -> None:
    _shape(point, {"ref", "speaker", "quote"})
    ref = _text(point["ref"], "evidence_ref")
    if ref not in turns or point["speaker"] != turns[ref]["speaker"]:
        raise ValueError("evidence_ref_or_speaker_invalid")
    if _text(point["quote"], "evidence_quote") not in turns[ref]["text"]:
        raise ValueError("evidence_quote_not_exact")


def extraction_prompt(surface: dict[str, Any]) -> str:
    turns = [{key: row[key] for key in ("turn_ref", "speaker", "text")}
             for row in surface["turns"]]
    return EXTRACTION_RULES + json.dumps(turns, ensure_ascii=False, indent=2) + "\n"


def _validate_relation(row: Any, turns: dict[str, dict[str, str]]) -> None:
    _shape(row, {"relation_id", "relation_type", "actor", "object_id", "version", "action",
                 "scope", "state_before", "state_after", "occurrence_time", "report_time",
                 "checkpoints"})
    if row["relation_type"] not in RELATION_TYPES:
        raise ValueError("relation_type_invalid")
    for key in ("actor", "object_id", "action", "scope"):
        _text(row[key], key)
    for key in ("version", "occurrence_time", "report_time"):
        _text(row[key], key, empty=True)
    if row["state_before"] not in STATES or row["state_after"] not in STATES:
        raise ValueError("relation_state_invalid")
    if any(REF_RE.search(row[key]) for key in ("actor", "object_id", "version", "action", "scope")):
        raise ValueError("source_ref_leaked_into_question")
    points = row["checkpoints"]
    if not isinstance(points, list) or len(points) != 2:
        raise ValueError("two_checkpoints_required")
    for point in points:
        _quote(point, turns)
    if points[0]["ref"] == points[1]["ref"]:
        raise ValueError("reused_checkpoint_ref")


def _relation_claims(row: dict[str, Any]) -> tuple[str, str]:
    """Render one reviewed transition and a same-checkpoint contradicted alternative."""
    object_name = row["object_id"] + (f" ({row['version']})" if row["version"] else "")
    subject = f"For {row['scope']}, the status of {row['actor']}'s {row['action']} on {object_name}"
    if row["state_before"] == row["state_after"]:
        state = row["state_after"]
        other = ORDERED_STATES[(ORDERED_STATES.index(state) + 1) % len(ORDERED_STATES)]
        return (f"{subject} remained {state} across the two checkpoints.",
                f"{subject} remained {other} across the two checkpoints.")
    else:
        before, after = row["state_before"], row["state_after"]
        return (f"{subject} changed from {before} to {after} across the two checkpoints.",
                f"{subject} changed from {after} to {before} across the two checkpoints.")


def _validate_occurrence(row: Any, turns: dict[str, dict[str, str]]) -> None:
    _shape(row, {"event_id", "actor", "object_id", "version", "action", "status",
                 "occurrence_time", "report_time", "evidence"})
    for key in ("actor", "object_id", "action"):
        _text(row[key], key)
    for key in ("version", "occurrence_time", "report_time"):
        _text(row[key], key, empty=True)
    if row["status"] not in {"started", "completed"}:
        raise ValueError("occurrence_not_observed")
    if any(REF_RE.search(row[key]) for key in ("actor", "object_id", "version", "action")):
        raise ValueError("source_ref_leaked_into_question")
    _quote(row["evidence"], turns)


def _occurrence_text(row: dict[str, Any]) -> str:
    object_name = row["object_id"] + (f" ({row['version']})" if row["version"] else "")
    verb = "began" if row["status"] == "started" else "completed"
    return f"{row['actor']} {verb} {row['action']} for {object_name}."


def _validate_edge(row: Any, turns: dict[str, dict[str, str]]) -> None:
    _shape(row, {"edge_id", "before_event_id", "after_event_id", "temporal_evidence"})
    if (not isinstance(row["before_event_id"], str) or
            not ID_PATTERNS["occurrences"].fullmatch(row["before_event_id"]) or
            not isinstance(row["after_event_id"], str) or
            not ID_PATTERNS["occurrences"].fullmatch(row["after_event_id"]) or
            row["before_event_id"] == row["after_event_id"]):
        raise ValueError("edge_event_ids_invalid")
    _quote(row["temporal_evidence"], turns)


def validate_extraction(raw: str, surface: dict[str, Any]) -> dict[str, Any]:
    """Validate every row independently; bad siblings remain visible in audit."""
    payload = _strict_json(raw)
    if not isinstance(payload, dict) or set(payload) != {*KINDS, "insufficiency_reason"}:
        raise ValueError("extraction_top_level_invalid")
    if not isinstance(payload["insufficiency_reason"], str):
        raise ValueError("insufficiency_reason_invalid")
    if any(not isinstance(payload[kind], list) or len(payload[kind]) > 40 for kind in KINDS):
        raise ValueError("extraction_list_invalid")
    turns = {row["turn_ref"]: row for row in surface["turns"]}
    result: dict[str, Any] = {"root_id": surface["root_id"], "source_sha256": surface["source_sha256"],
                              "raw_sha256": sha256(raw.encode("utf-8")).hexdigest(),
                              "records": {kind: [] for kind in KINDS}, "assessments": [],
                              "rendered_relation_pairs": [], "rendered_occurrences": [],
                              "insufficiency_reason": payload["insufficiency_reason"]}
    checks = {"relations": _validate_relation, "occurrences": _validate_occurrence,
              "order_edges": _validate_edge}
    seen: dict[str, set[str]] = {kind: set() for kind in KINDS}
    for kind in KINDS:
        for index, row in enumerate(payload[kind]):
            record_id = row.get(ID_FIELDS[kind]) if isinstance(row, dict) else None
            try:
                if not isinstance(record_id, str) or not ID_PATTERNS[kind].fullmatch(record_id):
                    raise ValueError("record_id_invalid")
                if record_id in seen[kind]:
                    raise ValueError("duplicate_record_id")
                seen[kind].add(record_id)
                checks[kind](row, turns)
            except (KeyError, TypeError, ValueError) as exc:
                result["assessments"].append({"kind": kind, "index": index, "record_id": record_id,
                                              "status": "mechanically_invalid", "reason": str(exc)})
            else:
                result["records"][kind].append(row)
                if kind == "relations":
                    supported, contradicted = _relation_claims(row)
                    result["rendered_relation_pairs"].append({"relation_id": record_id,
                                                              "supported_claim": supported,
                                                              "contradicted_claim": contradicted})
                elif kind == "occurrences":
                    result["rendered_occurrences"].append({"event_id": record_id,
                                                           "item_text": _occurrence_text(row)})
                result["assessments"].append({"kind": kind, "index": index, "record_id": record_id,
                                              "status": "pending_semantic_review", "reason": ""})
    valid_events = {row["event_id"] for row in result["records"]["occurrences"]}
    valid_edges = {row["edge_id"]: row for row in result["records"]["order_edges"]}
    for assessment in result["assessments"]:
        if assessment["kind"] != "order_edges" or assessment["status"] != "pending_semantic_review":
            continue
        row = valid_edges[assessment["record_id"]]
        if row["before_event_id"] not in valid_events or row["after_event_id"] not in valid_events:
            assessment.update(status="mechanically_invalid", reason="edge_event_missing_or_invalid")
            result["records"]["order_edges"].remove(row)
    return result


def review_template(validated: dict[str, Any]) -> dict[str, Any]:
    return {"root_id": validated["root_id"], "source_sha256": validated["source_sha256"],
            "validated_sha256": _digest(validated), "reviewer": "", "assessments": [
                {"kind": item["kind"], "record_id": item["record_id"], "decision": "",
                 "reason": ""} for item in validated["assessments"]
                if item["status"] == "pending_semantic_review"]}


def select_accepted(validated: dict[str, Any], review: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """A complete review can classify original records but cannot amend them."""
    if (not isinstance(review, dict) or set(review) != {"root_id", "source_sha256", "validated_sha256",
                                                   "reviewer", "assessments"} or
            review["root_id"] != validated["root_id"] or
            review["source_sha256"] != validated["source_sha256"] or
            review["validated_sha256"] != _digest(validated)):
        raise ValueError("review_binding_invalid")
    _text(review["reviewer"], "reviewer")
    expected = {(item["kind"], item["record_id"]) for item in validated["assessments"]
                if item["status"] == "pending_semantic_review"}
    decisions: dict[tuple[str, str], str] = {}
    if not isinstance(review["assessments"], list):
        raise ValueError("review_assessments_invalid")
    for item in review["assessments"]:
        _shape(item, {"kind", "record_id", "decision", "reason"})
        key = (item["kind"], item["record_id"])
        if key not in expected or key in decisions or item["decision"] not in {"supported", "contradicted", "ambiguous"}:
            raise ValueError("review_decision_invalid")
        _text(item["reason"], "review_reason")
        decisions[key] = item["decision"]
    if set(decisions) != expected:
        raise ValueError("review_incomplete")
    return {kind: [row for row in validated["records"][kind]
                   if decisions[(kind, row[ID_FIELDS[kind]])] == "supported"] for kind in KINDS}


def _unique_order(nodes: tuple[str, ...], edges: dict[tuple[str, str], dict[str, Any]]) -> tuple[str, ...] | None:
    remaining = set(nodes)
    order: list[str] = []
    while remaining:
        zero = [node for node in remaining if not any((other, node) in edges for other in remaining)]
        if len(zero) != 1:
            return None
        node = zero[0]
        order.append(node)
        remaining.remove(node)
    if not all((before, after) in edges for before, after in zip(order, order[1:])):
        return None
    return tuple(order)


def _graph_acyclic(nodes: set[str], edges: dict[tuple[str, str], dict[str, Any]]) -> bool:
    remaining = set(nodes)
    while remaining:
        zero = {node for node in remaining if not any((other, node) in edges for other in remaining)}
        if not zero:
            return False
        remaining.difference_update(zero)
    return True


def build_pool(accepted: dict[str, list[dict[str, Any]]], surface: dict[str, Any]) -> dict[str, Any]:
    """Enumerate same-scope claim pairs and uniquely ordered four-event chains."""
    if set(accepted) != set(KINDS):
        raise ValueError("accepted_kinds_invalid")
    turns = {row["turn_ref"]: row for row in surface["turns"]}
    pairs: list[dict[str, Any]] = []
    relation_keys: set[tuple[str, ...]] = set()
    for row in sorted(accepted["relations"], key=lambda item: item["relation_id"]):
        _validate_relation(row, turns)
        key = tuple(row[field].strip().casefold() for field in ("actor", "object_id", "version", "action", "scope"))
        if key in relation_keys:
            continue
        relation_keys.add(key)
        supported, contradicted = _relation_claims(row)
        pairs.append({"pair_id": f"P{len(pairs) + 1:03d}", "relation_id": row["relation_id"],
                      "relation_type": row["relation_type"], "actor": row["actor"],
                      "object_id": row["object_id"], "version": row["version"],
                      "action": row["action"], "scope": row["scope"],
                      "supported_claim": supported,
                      "contradicted_claim": contradicted,
                      "checkpoints": row["checkpoints"]})
    events: dict[str, dict[str, Any]] = {}
    for row in accepted["occurrences"]:
        _validate_occurrence(row, turns)
        if row["event_id"] in events:
            raise ValueError("duplicate_accepted_event_id")
        events[row["event_id"]] = row
    edges: dict[tuple[str, str], dict[str, Any]] = {}
    for row in accepted["order_edges"]:
        _validate_edge(row, turns)
        before, after = row["before_event_id"], row["after_event_id"]
        if before not in events or after not in events:
            continue
        edge_key = (before, after)
        if edge_key in edges or (after, before) in edges:
            continue
        edges[edge_key] = {"edge_id": row["edge_id"], "before_event_id": before,
                           "after_event_id": after,
                           "before_ref": events[before]["evidence"]["ref"],
                           "after_ref": events[after]["evidence"]["ref"],
                           "temporal_evidence": row["temporal_evidence"]}
    chains: list[dict[str, Any]] = []
    if _graph_acyclic(set(events), edges):
        for nodes in combinations(sorted(events), 4):
            order = _unique_order(nodes, edges)
            if order is None:
                continue
            chains.append({"chain_id": f"C{len(chains) + 1:03d}", "event_ids": list(order),
                           "items": [events[event_id] for event_id in order],
                           "adjacent_edges": [edges[(before, after)] for before, after in zip(order, order[1:])]})
    return {"root_id": surface["root_id"], "source_sha256": surface["source_sha256"],
            "pairs": pairs, "ordering_edges": list(edges.values()), "ordering_chains": chains,
            "multi_select_abstain_reason": "fewer_than_four_reviewed_distinct_relations" if len(pairs) < 4 else "",
            "ordering_abstain_reason": "no_four_event_unique_explicit_chain" if not chains else ""}


def composition_prompt(pool: dict[str, Any], question_type: str = "multi_select") -> str:
    if question_type == "multi_select":
        return MULTI_RULES + json.dumps(pool["pairs"], ensure_ascii=False, indent=2) + "\n"
    if question_type == "ordering":
        return ORDER_RULES + json.dumps(pool["ordering_chains"], ensure_ascii=False, indent=2) + "\n"
    raise ValueError("unknown_question_type")


def render_selection(raw: str, pool: dict[str, Any], question_type: str = "multi_select") -> dict[str, Any]:
    payload = _strict_json(raw)
    if question_type == "multi_select":
        return _render_multi(payload, pool)
    if question_type == "ordering":
        return _render_order(payload, pool)
    raise ValueError("unknown_question_type")


def _render_multi(payload: Any, pool: dict[str, Any]) -> dict[str, Any]:
    _shape(payload, {"questions", "abstain_reason"})
    questions, reason = payload["questions"], payload["abstain_reason"]
    if not isinstance(questions, list) or len(questions) > 2 or not isinstance(reason, str):
        raise ValueError("multi_selection_shape_invalid")
    if len(questions) < 2 and not reason.strip():
        raise ValueError("missing_abstain_reason")
    if len(questions) == 2 and reason.strip():
        raise ValueError("unexpected_abstain_reason")
    by_id = {row["pair_id"]: row for row in pool["pairs"]}
    if len(by_id) != len(pool["pairs"]):
        raise ValueError("duplicate_pool_pair_id")
    rendered: list[dict[str, Any]] = []
    seen_question_keys: set[tuple[str, ...]] = set()
    for question in questions:
        _shape(question, {"selected_options"})
        choices = question["selected_options"]
        if not isinstance(choices, list) or len(choices) not in (4, 5, 6):
            raise ValueError("multi_option_count_invalid")
        option_map: dict[str, str] = {}
        evidence: dict[str, Any] = {}
        gold: list[str] = []
        seen_ids: set[str] = set()
        for label, choice in zip("ABCDEF", choices):
            _shape(choice, {"pair_id", "orientation"})
            pair_id, orientation = choice["pair_id"], choice["orientation"]
            if pair_id not in by_id or pair_id in seen_ids or orientation not in {"as_stated", "contradicted"}:
                raise ValueError("pair_or_orientation_invalid")
            seen_ids.add(pair_id)
            pair = by_id[pair_id]
            claim = pair["supported_claim"] if orientation == "as_stated" else pair["contradicted_claim"]
            if REF_RE.search(claim):
                raise ValueError("source_ref_leaked_into_question")
            option_map[label] = claim
            evidence[label] = {"pair_id": pair_id, "relation_id": pair["relation_id"],
                               "verdict": "TRUE" if orientation == "as_stated" else "FALSE",
                               "checkpoints": pair["checkpoints"]}
            if orientation == "as_stated":
                gold.append(label)
        if len(gold) not in (2, 3) or len(set(option_map.values())) != len(option_map):
            raise ValueError("multi_gold_or_duplicate_text_invalid")
        key = tuple(sorted(seen_ids))
        if key in seen_question_keys:
            raise ValueError("duplicate_selected_relations")
        seen_question_keys.add(key)
        rendered.append({"candidate_id": "b-ms-" + _digest(choices)[:12], "question_type": "multi_select",
                         "question": "Which statements are supported by the conversation? Select all that apply.",
                         "options": option_map, "gold": gold, "option_evidence": evidence,
                         "status": "UNREVIEWED_CANDIDATE", "training_promoted": False})
    return {"questions": rendered, "abstain_reason": reason, "training_promoted": False}


def _render_order(payload: Any, pool: dict[str, Any]) -> dict[str, Any]:
    _shape(payload, {"selected_chain_id", "abstain_reason"})
    chain_id = payload["selected_chain_id"]
    reason = payload["abstain_reason"]
    if not isinstance(chain_id, str) or not isinstance(reason, str):
        raise ValueError("ordering_selection_shape_invalid")
    if not chain_id:
        _text(reason, "abstain_reason")
        return {"questions": [], "abstain_reason": reason, "training_promoted": False}
    if reason.strip():
        raise ValueError("unexpected_abstain_reason")
    by_id = {row["chain_id"]: row for row in pool["ordering_chains"]}
    if len(by_id) != len(pool["ordering_chains"]) or chain_id not in by_id:
        raise ValueError("ordering_chain_unknown")
    chain = by_id[chain_id]
    # Keep visible item order independent of the occurrence order in the pool.
    display_events = sorted(chain["items"], key=lambda event: _digest([chain_id, event["event_id"]]))
    if [event["event_id"] for event in display_events] == chain["event_ids"]:
        display_events[0], display_events[1] = display_events[1], display_events[0]
    labels = ["I1", "I2", "I3", "I4"]
    items = {label: _occurrence_text(event) for label, event in zip(labels, display_events)}
    if len(items) != 4 or len(set(items.values())) != 4:
        raise ValueError("ordering_items_not_distinct")
    event_labels = {event["event_id"]: label for label, event in zip(labels, display_events)}
    event_evidence = {
        label: {"event_id": event["event_id"], "actor": event["actor"],
                "object_id": event["object_id"], "version": event["version"],
                "action": event["action"], "status": event["status"],
                "evidence": event["evidence"]}
        for label, event in zip(labels, display_events)
    }
    candidate = {"candidate_id": "b-order-" + _digest(chain["event_ids"])[:12],
                 "question_type": "ordering", "question": "Put these events in their supported occurrence order.",
                 "items": items, "gold": [event_labels[event_id] for event_id in chain["event_ids"]],
                 "display_event_ids": [event["event_id"] for event in display_events],
                 "event_ids": chain["event_ids"],
                 "event_evidence": event_evidence,
                 "adjacent_edge_evidence": chain["adjacent_edges"],
                 "status": "UNREVIEWED_CANDIDATE", "training_promoted": False}
    return {"questions": [candidate], "abstain_reason": "", "training_promoted": False}
