"""Deterministic, zero-API feasibility audit for the frozen conv-41 outline."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


AUDIT_VERSION = "outline_arc_dependency_feasibility_audit_v1"
ROUTE_INSUFFICIENT = "existing_outline_dependency_support_insufficient"
ROUTE_TYPED = "single_arc_operator_specific_typed_dependency_planner_eligible"


# This is an explicit offline audit, bound by the runner to the authoritative
# outline/anchor hashes.  It is deliberately not a heuristic relation miner.
ARC_AUDIT: tuple[dict[str, Any], ...] = (
    {
        "arc_id": "E1", "feasible": True,
        "candidate_parent_anchor_ids": ["P5"],
        "allowed_operators": ["qualification_requirement"],
        "preferred_operator": "qualification_requirement",
        "parent_state_A": {"subject": "John", "qualification_key": "completed_degree",
            "value": "completed_degree", "source_anchor_id": "P5"},
        "required_child_state_B": {"subject": "city advisory opening",
            "requirement_key": "completed_degree", "required_value": "completed_degree",
            "time_relation": "later", "allowed_stage": "eligibility check"},
        "computed_joint_state_C": {"state_key": "meets_completed_degree_requirement",
            "value": "true", "calculation": "completed_degree == completed_degree"},
        "parent_only_result": "UNKNOWN", "child_only_result": "UNKNOWN",
        "combined_result": "meets_requirement=true",
        "failure_reasons": [],
        "audit_notes": "P5 explicitly says John has a degree; E1 explicitly includes an eligibility check and S2 establishes application requirements. The future requirement can be stated without repeating John's degree."},
    {
        "arc_id": "E2", "feasible": True,
        "candidate_parent_anchor_ids": ["P8"],
        "allowed_operators": ["commitment_outcome"],
        "preferred_operator": "commitment_outcome",
        "parent_state_A": {"subject": "Maria", "action_key":
            "maria_brainstorms_community_ideas_with_john", "state": "planned",
            "source_anchor_id": "P8"},
        "required_child_state_B": {"subject": "Maria", "action_key_ref":
            "maria_brainstorms_community_ideas_with_john", "outcome_state": "completed",
            "time_relation": "later", "allowed_stage": "collection plan"},
        "computed_joint_state_C": {"state_key": "commitment_outcome", "value": "fulfilled",
            "calculation": "same_action_key AND outcome_state=completed"},
        "parent_only_result": "UNKNOWN", "child_only_result": "UNKNOWN",
        "combined_result": "fulfilled",
        "failure_reasons": [],
        "audit_notes": "P8 explicitly proposes that Maria and John brainstorm together; E2's collection-plan stage can realize that same bounded action without mentioning the earlier proposal."},
    {
        "arc_id": "E3", "feasible": False,
        "candidate_parent_anchor_ids": ["P2", "P6"], "allowed_operators": [],
        "preferred_operator": None,
        "parent_state_A": {}, "required_child_state_B": {},
        "computed_joint_state_C": {}, "parent_only_result": "UNKNOWN",
        "child_only_result": "UNKNOWN", "combined_result": "UNKNOWN",
        "failure_reasons": ["no_explicit_parent_claim_prediction_commitment_or_exact_prior_state_for_the_draft_value",
            "the_outdated_assumption_is_created_inside_E3_so_parent_evidence_is_not_required"],
        "audit_notes": "P2 and P6 state broad issue interests/understanding, not the draft proposition that E3 later verifies or revises. Treating those themes as the old state would be retrospective thematic fitting."},
    {
        "arc_id": "E4", "feasible": True,
        "candidate_parent_anchor_ids": ["P3"],
        "allowed_operators": ["commitment_outcome"],
        "preferred_operator": "commitment_outcome",
        "parent_state_A": {"subject": "John", "action_key":
            "john_talks_with_local_leaders_and_organizations_for_support_and_ideas",
            "state": "planned", "source_anchor_id": "P3"},
        "required_child_state_B": {"subject": "John", "action_key_ref":
            "john_talks_with_local_leaders_and_organizations_for_support_and_ideas",
            "outcome_state": "completed", "time_relation": "later",
            "allowed_stage": "questions"},
        "computed_joint_state_C": {"state_key": "commitment_outcome", "value": "fulfilled",
            "calculation": "same_action_key AND outcome_state=completed"},
        "parent_only_result": "UNKNOWN", "child_only_result": "UNKNOWN",
        "combined_result": "fulfilled", "failure_reasons": [],
        "audit_notes": "E4's presentation/questions stage can explicitly realize the complete P3 action by having John speak with local leaders and organization representatives to obtain feedback/support; the child must not mention that this was his earlier plan."},
)


def build_audit(outline: Mapping[str, Any], anchors: Mapping[str, Any],
                parent_turns: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    arcs = outline.get("planned_event_arcs") or []
    if [row.get("event_id") for row in arcs] != ["E1", "E2", "E3", "E4"]:
        raise ValueError("outline_arc_order_invalid")
    snapshot = {row["anchor_id"]: row for row in anchors.get("anchors") or []}
    outline_anchors = {row["anchor_id"]: row for row in outline.get("parent_anchors") or []}
    turns = {row["turn_id"]: row for row in parent_turns}
    if len(parent_turns) != 237 or len(snapshot) != 8 or snapshot != outline_anchors:
        raise ValueError("authoritative_parent_binding_invalid")
    for anchor in snapshot.values():
        turn = turns.get(anchor["turn_id"])
        if not turn or turn.get("speaker") != anchor["speaker"] or turn.get("text") != anchor["text"]:
            raise ValueError(f"parent_anchor_not_verbalized:{anchor['anchor_id']}")
    items = [dict(row) for row in ARC_AUDIT]
    feasible = sum(bool(row["feasible"]) for row in items)
    return {"audit_version": AUDIT_VERSION, "zero_api": True,
        "semantic_graph_used": False, "existing_child_used": False,
        "full_371_turn_dialogue_used": False, "old_qa_used": False,
        "solver_output_used": False, "parent_turn_count": len(parent_turns),
        "items": items, "feasible_arc_count": feasible,
        "typed_planner_threshold": 2,
        "route_conclusion": ROUTE_TYPED if feasible >= 2 else ROUTE_INSUFFICIENT}


def render_review(audit: Mapping[str, Any]) -> str:
    lines = ["# E1–E4 dependency feasibility review", "",
        "This is a zero-API audit. Feasibility requires A-only=UNKNOWN, B-only=UNKNOWN, and a unique program-computable C from A+B.", "",
        "| Arc | Feasible | Parent anchor | Operator | A-only | B-only | Combined |", "|---|---:|---|---|---|---|---|"]
    for row in audit["items"]:
        lines.append(f"| {row['arc_id']} | {'yes' if row['feasible'] else 'no'} | {', '.join(row['candidate_parent_anchor_ids'])} | {', '.join(row['allowed_operators']) or 'none'} | {row['parent_only_result']} | {row['child_only_result']} | {row['combined_result']} |")
    for row in audit["items"]:
        lines += ["", f"## {row['arc_id']}", "", row["audit_notes"]]
        if row["failure_reasons"]:
            lines += ["", "Failure reasons:"] + [f"- `{reason}`" for reason in row["failure_reasons"]]
    lines += ["", f"Feasible: **{audit['feasible_arc_count']}/4**", "",
        f"Route: `{audit['route_conclusion']}`", ""]
    return "\n".join(lines)
