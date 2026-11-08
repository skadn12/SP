"""Historical, transcript-anchored ordering diagnostics (offline only).

These fixtures preserve the original transcript references and deliberately do
not rewrite any historical candidate or claim a model quality result.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from ..core import atomic_json
from .constants import DEFAULT_RUN, SOURCE_WRITER
from .contract import VERSION as CONTRACT_VERSION, derive_ordering


def _expanded(root_id: str):
    path = SOURCE_WRITER / "inputs" / "shared" / root_id / "expanded_dialogue_clean.json"
    if path.is_file():
        return json.loads(path.read_text())
    return {"turns": []}


def _node(item_id, text, actor, state, ref):
    return {"item_id": item_id, "item_text": text, "actor": actor,
            "event_state": state, "evidence_refs": [ref]}


def _edge(before, after, refs, basis):
    return {"before": before, "after": after, "evidence_refs": list(refs),
            "temporal_basis": basis}


def _row(nodes, edges, question):
    refs = [ref for node in nodes for ref in node["evidence_refs"]]
    return {"question_type": "ordering", "ordering_contract_version": CONTRACT_VERSION,
            "question": question, "discriminative_variable": "event chronology",
            "event_nodes": nodes, "temporal_relations": edges,
            "evidence_groups": {"events": refs}, "gold": None}


def exercise(run=DEFAULT_RUN):
    run = Path(run)
    expanded = _expanded("cc-train-episode-122711")
    # Exact references below are from the frozen 122711 transcript used by the
    # completed Writer audit.  Labels describe known diagnostic risks only.
    cases = []
    recall = _row([
        _node("I1", "Emily recalled forming a study group", "Emily", "recalled", "cc-train-episode-122711:P02:T007"),
        _node("I2", "James considered starting a math study group", "James", "considered", "cc-train-episode-122711:P02:T008"),
        _node("I3", "James sought extra help from his teacher", "James", "completed", "cc-train-episode-122711:P03:T008"),
        _node("I4", "James practiced problems every night", "James", "completed", "cc-train-episode-122711:P05:T007"),
    ], [_edge("I1", "I2", ["cc-train-episode-122711:P02:T007", "cc-train-episode-122711:P02:T008"], "same conversation turn order"),
        _edge("I2", "I3", ["cc-train-episode-122711:P02:T008", "cc-train-episode-122711:P03:T008"], "later scene follows earlier scene"),
        _edge("I3", "I4", ["cc-train-episode-122711:P03:T008", "cc-train-episode-122711:P05:T007"], "later scene follows earlier scene")], "Order these math-help events.")
    cases.append(("recall_vs_occurrence", recall, "synthetic_mutation_anchored_real_refs", "Potential recall mistaken as a new occurrence; graph can remain structurally valid."))
    planned = _row([
        _node("I1", "James considered starting a math study group", "James", "considered", "cc-train-episode-122711:P02:T008"),
        _node("I2", "James decided to look into help", "James", "decided", "cc-train-episode-122711:P02:T006"),
        _node("I3", "James sought extra help from his teacher", "James", "completed", "cc-train-episode-122711:P03:T008"),
        _node("I4", "James practiced problems every night", "James", "completed", "cc-train-episode-122711:P05:T007"),
    ], [_edge("I1", "I2", ["cc-train-episode-122711:P02:T008", "cc-train-episode-122711:P02:T006"], "turn order"),
        _edge("I2", "I3", ["cc-train-episode-122711:P02:T006", "cc-train-episode-122711:P03:T008"], "later scene"),
        _edge("I3", "I4", ["cc-train-episode-122711:P03:T008", "cc-train-episode-122711:P05:T007"], "later scene")], "Order intended and completed actions.")
    cases.append(("plan_to_completion", planned, "synthetic_mutation_anchored_real_refs", "Plan/decision may be promoted to completion by semantic error; not blocked mechanically."))
    actor = _row([
        _node("I1", "Emily advised seeking help", "Emily", "suggested", "cc-train-episode-122711:P02:T005"),
        _node("I2", "James considered a math study group", "Emily", "considered", "cc-train-episode-122711:P02:T008"),
        _node("I3", "James sought teacher help", "James", "completed", "cc-train-episode-122711:P03:T008"),
        _node("I4", "James practiced nightly", "James", "completed", "cc-train-episode-122711:P05:T007"),
    ], [_edge("I1", "I2", ["cc-train-episode-122711:P02:T005", "cc-train-episode-122711:P02:T008"], "scene sequence"),
        _edge("I2", "I3", ["cc-train-episode-122711:P02:T008", "cc-train-episode-122711:P03:T008"], "scene sequence"),
        _edge("I3", "I4", ["cc-train-episode-122711:P03:T008", "cc-train-episode-122711:P05:T007"], "scene sequence")], "Order advice-related events.")
    cases.append(("actor_mismatch", actor, "synthetic_mutation_anchored_real_refs", "Intentional actor mismatch remains a schema-valid semantic error."))
    unsupported = _row([
        _node("I1", "Emily mentions a storm", "Emily", "mentioned", "cc-train-episode-122711:P01:T001"),
        _node("I2", "Emily cuts short her trip", "Emily", "decided", "cc-train-episode-122711:P01:T003"),
        _node("I3", "James approves the decision", "James", "commented", "cc-train-episode-122711:P01:T004"),
        _node("I4", "Emily prefers safety", "Emily", "stated", "cc-train-episode-122711:P01:T005"),
    ], [_edge("I1", "I2", ["cc-train-episode-122711:P01:T001"], "usual weather causality"),
        _edge("I2", "I3", ["cc-train-episode-122711:P01:T003"], "adjacent turn order"),
        _edge("I3", "I4", ["cc-train-episode-122711:P01:T004"], "adjacent turn order")], "Order storm-related statements.")
    cases.append(("unsupported_edge", unsupported, "historical_refs", "The graph is unique but one edge basis is an unsupported inference; program must still report semantic NOT_ASSESSED."))
    overlap = copy.deepcopy(unsupported)
    overlap["event_nodes"][2]["event_state"] = "ongoing"
    overlap["temporal_relations"][1]["temporal_basis"] = "No explicit time distinction; overlap intentionally forced."
    cases.append(("overlapping_events", overlap, "synthetic_mutation_anchored_real_refs", "Continuous/overlapping events may be forced into an order; this requires content review."))
    cycle = copy.deepcopy(unsupported)
    cycle["temporal_relations"].append(_edge("I4", "I1", ["cc-train-episode-122711:P01:T005"], "synthetic cycle"))
    cases.append(("cycle_rejection", cycle, "historical_refs", "Cycle must fail before any non-uniqueness label."))
    under = copy.deepcopy(unsupported)
    under["temporal_relations"] = under["temporal_relations"][:1]
    cases.append(("missing_constraints", under, "historical_refs", "Insufficient constraints must fail unique-order derivation."))
    out = []
    for name, row, origin, note in cases:
        result = derive_ordering(row, expanded=expanded, policy={"version": CONTRACT_VERSION, "node_count": 4}, identity={"run_id": "historical_fixture", "arm_id": "offline", "root_id": "cc-train-episode-122711", "intent_id": "fixture", "source": "fixture", "stage": "offline", "request_id": name, "raw_row_index": 1, "candidate_id": "fixture:intent-C-Q01", "raw_sha256": "fixture"})
        out.append({"case": name, "origin": origin, "note": note, "status": result["status"], "errors": result["errors"], "semantic_support_status": result["record"]["semantic_support_status"], "derived_gold": result["record"]["derived_gold"], "refs": [ref for node in row["event_nodes"] for ref in node["evidence_refs"]]})
    report = {"status": "PASS", "external_calls": 0, "content_claim": "NOT_EVALUATED", "cases": out, "source_transcript": "h4_r02_writer_scms_quality_ablation_v1_live_1/inputs/shared/cc-train-episode-122711/expanded_dialogue_clean.json", "note": "These are offline fixtures and anchored mutations, not historical model classifications or replacements."}
    atomic_json(run / "audit/historical_ordering_fixtures.json", report)
    return report
