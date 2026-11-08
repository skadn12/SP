"""Scene-local Realizer prompt wrapper for the H4 refinement probe."""

from __future__ import annotations

import json
from typing import Any, Mapping

from scripts.current_method.restart_n5_spell.planner_realizer_temporal_boundary_probe_v1.scene_release_probe_v2 import build_single_scene_prompt_v2


def _contract_block(contract: Mapping[str, Any], index: int) -> str:
    fact_a = contract["fact_A"]
    trigger = contract["trigger"]
    fact_b = contract["fact_B"]
    resolution = contract["resolution"]
    if index == 1:
        stage_rules = (
            "Establish fact_A as the initial state or plan. Keep it provisional. "
            "Do not mention, foreshadow, or speculate about the later trigger or fact_B, "
            "even as a possibility. Do not perform fact_B or the resolution."
        )
    elif index == 2:
        stage_rules = (
            "State the trigger explicitly as new information, constraint, or objection. "
            "Make the causal link explicit (because, so, therefore, after, or an equivalent clear relation), "
            "then state fact_B as a revision of fact_A. Do not treat fact_B as an unrelated new event."
        )
    else:
        stage_rules = (
            "Preserve the A -> trigger -> B relation in the final resolution. "
            "The final scene may resolve fact_B, but must not erase what the initial state was or why it changed."
        )
    # Keep the relation contract itself scene-local.  Giving Scene 1 the
    # future trigger or revised value would leak the answer and invalidate the
    # difficulty experiment.
    payload = {"relation_type": contract["relation_type"],
               "current_scene_rules": stage_rules}
    if index >= 1:
        payload["fact_A"] = fact_a
    if index >= 2:
        payload["trigger"] = trigger
        payload["fact_B"] = fact_b
        payload["typed_edges"] = [edge for edge in contract["typed_edges"] if edge["source"] in {"fact_A", "trigger"}]
    if index >= 3:
        payload["resolution"] = resolution
        payload["typed_edges"] = contract["typed_edges"]
    return (
        "RELATION-CONTRACT REFINEMENT (isolated H4 probe)\n"
        "This contract is a QA-difficulty target, not a request for extra plot. "
        "Use the same named people, object/state slot, and chronological order. "
        "A relation must be expressed in the dialogue, not only in metadata.\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n\nRELATION STAGE INSTRUCTION\n" + stage_rules
    )


def build_relation_scene_prompt(
    *, parent: Mapping[str, Any], prior: list[Mapping[str, Any]],
    current: Mapping[str, Any], index: int, contract: Mapping[str, Any],
) -> dict[str, Any]:
    base = build_single_scene_prompt_v2(
        parent=parent, prior=prior, current=current, index=index)
    relation_block = _contract_block(contract, index)
    prompt = str(base["prompt"]) + "\n\n" + relation_block
    return {
        **base,
        "prompt_version": "h4_relation_contract_refinement_v1_prompt",
        "prompt": prompt,
        "relation_contract_version": contract["contract_version"],
        "relation_id": contract["relation_id"],
        "relation_type": contract["relation_type"],
    }
