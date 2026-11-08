"""Typed relation contracts for an isolated H4 refinement experiment.

The contract is derived from an already accepted O1 plan.  It is deliberately
kept outside the production Planner/H4A/H4B modules so the experiment changes
one variable: whether the Realizer is required to expose a QA-usable relation.
"""

from __future__ import annotations

from typing import Any, Mapping


RELATION_TYPES = {
    "causal_revision",
    "constraint_replan",
    "new_evidence_revision",
    "proposal_outcome",
}


def _scene_trigger(scene: Mapping[str, Any]) -> Mapping[str, Any]:
    trigger = scene.get("trigger")
    return trigger if isinstance(trigger, Mapping) else {}


def _relation_type(scene: Mapping[str, Any]) -> str:
    trigger_type = str(_scene_trigger(scene).get("type") or "").casefold()
    transition = str(scene.get("transition_type") or "").casefold()
    if trigger_type == "constraint" or "constraint" in transition:
        return "constraint_replan"
    if trigger_type == "new_information":
        return "new_evidence_revision"
    return "causal_revision"


def build_relation_contract(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Build one explicit A -> trigger -> B contract from a three-scene plan."""
    scenes = list(plan.get("scenes") or [])
    if len(scenes) != 3:
        raise ValueError("relation_contract_requires_three_scenes")
    first, middle, final = scenes
    trigger = _scene_trigger(middle)
    relation_type = _relation_type(middle)
    contract = {
        "contract_version": "h4_relation_contract_v1",
        "hypothesis": "H4",
        "execution_scope": "isolated_refinement_probe",
        "relation_id": "RELCHAIN:S01-S02-S03",
        "relation_type": relation_type,
        "state_slot": str(first.get("before_state") or first.get("after_state") or ""),
        "fact_A": {
            "role": "initial_state_or_plan",
            "scene_index": 1,
            "value": str(first.get("after_state") or ""),
            "source_plan_field": "scenes[0].after_state",
        },
        "trigger": {
            "role": "causal_trigger",
            "scene_index": 2,
            "type": str(trigger.get("type") or "new_information"),
            "content": str(trigger.get("content") or middle.get("required_transition") or ""),
            "source_plan_field": "scenes[1].trigger",
        },
        "fact_B": {
            "role": "revised_state_or_plan",
            "scene_index": 2,
            "value": str(middle.get("after_state") or ""),
            "source_plan_field": "scenes[1].after_state",
        },
        "resolution": {
            "role": "later_resolution",
            "scene_index": 3,
            "value": str(final.get("after_state") or ""),
            "source_plan_field": "scenes[2].after_state",
        },
        "typed_edges": [
            {"source": "fact_A", "relation": "constrained_or_revised_by", "target": "trigger"},
            {"source": "trigger", "relation": "causes_revision_to", "target": "fact_B"},
            {"source": "fact_B", "relation": "resolved_or_confirmed_as", "target": "resolution"},
        ],
        "dependency_test": {
            "full_context_requires": ["fact_A", "trigger", "fact_B"],
            "remove_fact_A_should": "leave the original state and direction of change underdetermined",
            "remove_fact_B_should": "leave the revised or final state underdetermined",
            "remove_trigger_should": "leave the reason for revision underdetermined",
        },
        "qa_target": {
            "composite_intent": "explain how the initial state changed, what caused the change, and what revised state followed",
            "required_evidence_groups": ["fact_A", "trigger", "fact_B"],
        },
    }
    if relation_type not in RELATION_TYPES:
        raise ValueError(f"unsupported_relation_type:{relation_type}")
    return contract
