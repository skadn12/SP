"""Prompt builders for three separate model roles; no backend is imported."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..core import canonical_json, sha256_bytes
from .schemas import (
    NATURAL_REALIZER_SCHEMA,
    SLIM_PLANNER_SCHEMA,
)


SLIM_PLANNER_PROMPT_VERSION = "slim_world_planner_prompt_v3"
SLIM_PLANNER_RETRY_PROMPT_VERSION = "slim_world_planner_retry_prompt_v3"
NATURAL_REALIZER_PROMPT_VERSION = "natural_realizer_prompt_v2"
NATURAL_REALIZER_CLAIM_ONLY_RETRY_PROMPT_VERSION = \
    "natural_realizer_claim_only_retry_prompt_v1"
EVIDENCE_COMMITTER_PROMPT_VERSION = "evidence_committer_prompt_v3"
EVIDENCE_COMMITTER_RETRY_PROMPT_VERSION = "evidence_committer_retry_prompt_v3"


def planner_action_selection_view(
    action_menu: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the frozen menu to fields the Planner may copy or reason over.

    Program-owned operation and operation-class fields deliberately stay out of
    the model-facing selection view.  The SHA binds this view to the complete,
    immutable action menu used later by the compiler and validator.
    """
    return {
        "affordance_schema_version": action_menu["affordance_schema_version"],
        "root_id": action_menu["root_id"],
        "frozen_action_menu_sha256": sha256_bytes(
            canonical_json(action_menu).encode()),
        "scene_budget": dict(action_menu["scene_budget"]),
        "memory_threads": [{
            "thread_id": row["thread_id"],
            "summary": row["summary"],
        } for row in action_menu["memory_threads"]],
        "actions": [{
            "action_id": row["action_id"],
            "thread_id": row["thread_id"],
            "eligible_participants": list(row["eligible_participants"]),
            "allowed_direction": row["default_post_state"],
            "reason": row["reason"],
            **({"mutually_exclusive_group": row["mutually_exclusive_group"]}
               if row.get("mutually_exclusive_group") else {}),
        } for row in action_menu["actions"]],
    }


def compact_planner_retry_errors(errors: Sequence[str]) -> list[str]:
    """Keep actionable errors without feeding an echoed wrapper back verbatim."""
    result: list[str] = []
    echoed_reference = any(
        error.startswith("planner_program_field:$model.actions[")
        for error in errors)
    if echoed_reference:
        result.extend([
            "retry_reference_object_was_echoed",
            "output_must_not_contain_actions_or_reference_wrapper_fields",
        ])
    if "planner_top_level_schema" in errors:
        result.append("output_must_have_exact_four_top_level_fields")
    for error in errors:
        if (error == "planner_top_level_schema" or
                error.startswith("planner_program_field:$model.actions[")):
            continue
        if error not in result:
            result.append(str(error))
    return result[:12]


def _planner_output_example() -> dict[str, Any]:
    return {
        "planner_schema_version": "slim_world_planner_v2",
        "selected_memory_threads": ["<thread_id for every selected action>"],
        "scene_budget": {"target": 3, "minimum": 2, "maximum": 4},
        "scene_goals": [{
            "goal": "<natural scene objective, not scripted dialogue>",
            "participants": ["<eligible display_name>"],
            "selected_action_ids": ["<listed action_id>"],
        }, {
            "goal": "<second natural scene objective>",
            "participants": ["<eligible display_name>"],
            "selected_action_ids": ["<different listed action_id>"],
        }],
    }


def build_slim_planner_prompt(
    *, parent_turns: Sequence[Mapping[str, Any]],
    action_menu: Mapping[str, Any], role_table: Sequence[Mapping[str, Any]],
) -> str:
    selection_view = planner_action_selection_view(action_menu)
    payload = {
        "root_id": action_menu["root_id"],
        "role_table": [{
            "entity_id": row.get("entity_id"),
            "display_name": row.get("display_name"),
        } for row in role_table],
        "relevant_parent_turns": [{
            "turn_id": row.get("turn_id"),
            "speaker": row.get("speaker"),
            "text": row.get("text"),
        } for row in parent_turns],
        "action_selection_reference": selection_view,
    }
    example = _planner_output_example()
    return "\n".join([
        f"Prompt version: {SLIM_PLANNER_PROMPT_VERSION}",
        "Choose executable actions and plan 2-4 natural scene goals.",
        "The selection reference is read-only context, never an output wrapper.",
        "For each scene, participants must be a nonempty subset of eligible_participants "
        "for every selected action.",
        "selected_memory_threads must include the thread_id of every selected action.",
        "Copy only listed thread IDs, action IDs, and eligible display names.",
        "Do not output scene/delta IDs, operations, memory IDs, state, evidence, dialogue, or handoff IDs.",
        "Each action may be selected once; actions in one mutual-exclusion group may not span scenes.",
        "READ-ONLY SELECTION REFERENCE — DO NOT COPY ITS WRAPPER FIELDS:",
        canonical_json(payload),
        "OUTPUT ONLY ONE JSON OBJECT WITH EXACTLY THESE FOUR TOP-LEVEL FIELDS:",
        "planner_schema_version, selected_memory_threads, scene_budget, scene_goals",
        "Replace every placeholder below with listed values. Do not output the reference object.",
        canonical_json(example),
    ])


def build_slim_planner_retry_prompt(
    *, action_menu: Mapping[str, Any], role_table: Sequence[Mapping[str, Any]],
    errors: Sequence[str],
) -> str:
    """Compact retry: frozen menu plus precise selection errors, no Memory/dialogue."""
    selection_view = planner_action_selection_view(action_menu)
    reference = {
        "root_id": action_menu["root_id"],
        "frozen_action_menu_sha256": selection_view["frozen_action_menu_sha256"],
        "allowed_participants": [str(row.get("display_name")) for row in role_table],
        "scene_budget": selection_view["scene_budget"],
        "memory_threads": selection_view["memory_threads"],
        "actions": selection_view["actions"],
    }
    compact_errors = compact_planner_retry_errors(errors)
    return "\n".join([
        f"Prompt version: {SLIM_PLANNER_RETRY_PROMPT_VERSION}",
        "Repair only the listed selection/schema errors against this frozen action menu.",
        "Do not invent fields, actions, participants, state, memory IDs, or dialogue.",
        "For each scene, participants must be a nonempty subset of eligible_participants "
        "for every selected action.",
        "selected_memory_threads must include the thread_id of every selected action.",
        "Do not copy root_id, frozen_action_menu_sha256, allowed_participants, "
        "memory_threads, actions, previous_errors, or any reference wrapper field.",
        "READ-ONLY FROZEN SELECTION REFERENCE — NEVER RETURN THIS OBJECT:",
        canonical_json(reference),
        "PREVIOUS VALIDATION ERRORS:",
        canonical_json(compact_errors),
        "OUTPUT ONLY ONE JSON OBJECT WITH EXACTLY THESE FOUR TOP-LEVEL FIELDS:",
        "planner_schema_version, selected_memory_threads, scene_budget, scene_goals",
        "Replace every placeholder below with listed values. Do not output reference fields.",
        canonical_json(_planner_output_example()),
    ])


def estimate_prompt_tokens(prompt: str, tokenizer: Any | None = None) -> int:
    if tokenizer is not None:
        return len(tokenizer.encode(prompt, add_special_tokens=False))
    # Stable offline estimate used by unit tests when the local model tokenizer is absent.
    return max(1, (len(prompt.encode("utf-8")) + 3) // 4)


def build_natural_realizer_prompt(
    *, relevant_parent_turns: Sequence[Mapping[str, Any]],
    memory_subset: Mapping[str, Any], scene_goal: Mapping[str, Any],
    accepted_deltas: Sequence[Mapping[str, Any]],
    prior_scenes: Sequence[Mapping[str, Any]],
) -> str:
    payload = {
        "relevant_parent_dialogue": list(relevant_parent_turns),
        "world_memory_subset": dict(memory_subset),
        "scene_goal": dict(scene_goal),
        "accepted_proposed_graph_deltas": list(accepted_deltas),
        "prior_generated_scenes": list(prior_scenes),
    }
    example = {
        "realizer_schema_version": "natural_realizer_v2",
        "scene_key": "<copy input scene_key>",
        "dialogue_turns": [
            {"turn_key": "t01", "speaker": "<allowed participant>",
             "text": "<natural utterance>"},
            {"turn_key": "t02", "speaker": "<allowed participant>",
             "text": "<natural response>"},
        ],
        "claimed_graph_updates": [{
            "proposed_delta_id": "<copy an accepted proposed_delta_id>",
            "claimed_status": "realized",
            "candidate_evidence_turn_keys": ["t01", "t02"],
        }],
        "new_open_thread_claims": [],
    }
    return "\n".join([
        f"Prompt version: {NATURAL_REALIZER_PROMPT_VERSION}",
        "You are the Natural Realizer. Write a coherent scene in natural dialogue.",
        "Use response-local turn_key references only. Do not invent final IDs or indexes.",
        "Use only speakers listed in scene_goal.participants and claim each accepted delta once.",
        "Claim every accepted planned delta exactly once in claimed_graph_updates.",
        "Use claimed_status realized, not_realized, or uncertain; it is never a commit decision.",
        "A planned open_thread belongs only in claimed_graph_updates; do not duplicate it.",
        "new_open_thread_claims is optional emergent-only data (default []); each nonempty "
        "claim must set related_proposed_delta_id to null.",
        "All Realizer claims remain non-authoritative until independent semantic review.",
        "Return one strict JSON object matching this contract:",
        canonical_json(NATURAL_REALIZER_SCHEMA),
        "Shape example (copy actual scene and delta identities from the input):",
        canonical_json(example),
        "Scene input:",
        canonical_json(payload),
    ])


def compact_realizer_claim_retry_errors(errors: Sequence[str]) -> list[str]:
    """Remove invalid key values while retaining actionable claim diagnostics."""
    compact: list[str] = []
    for error in errors:
        parts = str(error).split(":")
        code = parts[0]
        if code == "realizer_claim_evidence_not_in_current_scene":
            item = f"claim_{parts[1]}_candidate_evidence_key_not_in_current_scene"
        elif code == "realizer_claim_realized_evidence_empty":
            item = f"claim_{parts[1]}_realized_requires_nonempty_evidence"
        elif code == "realizer_claim_not_realized_evidence_nonempty":
            item = f"claim_{parts[1]}_not_realized_requires_empty_evidence"
        elif code == "realizer_claim_duplicate_delta":
            item = "same_proposed_delta_must_not_be_claimed_twice"
        elif code == "claimed_update_value":
            continue
        else:
            item = code
        if item not in compact:
            compact.append(item)
    return compact[:12]


def build_realizer_claim_only_retry_prompt(
    *, scene_key: str, frozen_dialogue_turns: Sequence[Mapping[str, Any]],
    proposed_delta_ids: Sequence[str], errors: Sequence[str],
) -> str:
    """Request replacement claims while withholding the rejected response."""
    valid_turn_keys = [str(turn["turn_key"]) for turn in frozen_dialogue_turns]
    dialogue_lines = [
        f"{turn['turn_key']} | {turn['speaker']} | {turn['text']}"
        for turn in frozen_dialogue_turns
    ]
    example_key = valid_turn_keys[0] if valid_turn_keys else "<allowed turn_key>"
    template = {
        "claimed_graph_updates": [{
            "proposed_delta_id": str(delta_id),
            "claimed_status": "realized",
            "candidate_evidence_turn_keys": [example_key],
        } for delta_id in proposed_delta_ids],
        "new_open_thread_claims": [],
    }
    return "\n".join([
        f"Prompt version: {NATURAL_REALIZER_CLAIM_ONLY_RETRY_PROMPT_VERSION}",
        "The dialogue below is valid and frozen.",
        "Do not rewrite or return scene_key or dialogue_turns.",
        f"FROZEN SCENE KEY: {scene_key}",
        "FROZEN DIALOGUE:",
        *dialogue_lines,
        "AVAILABLE PROPOSED DELTA IDS:",
        *[f"- {delta_id}" for delta_id in proposed_delta_ids],
        "ALLOWED EVIDENCE TURN KEYS:",
        *[f"- {key}" for key in valid_turn_keys],
        "PREVIOUS CLAIM ERRORS (the rejected response and invalid keys are omitted):",
        canonical_json(compact_realizer_claim_retry_errors(errors)),
        "RULES:",
        "- Submit every available proposed delta exactly once.",
        "- realized requires one or more exact keys from the allowed list.",
        "- not_realized requires an empty candidate_evidence_turn_keys list.",
        "- uncertain may use an empty list or exact keys from the allowed list.",
        "- Never invent, infer, repair, or fuzzy-match a turn key.",
        "- new_open_thread_claims is emergent-only; related_proposed_delta_id must be null.",
        "- Return exactly two top-level fields and do not return the frozen dialogue.",
        "The template demonstrates shape only; independently choose status and evidence.",
        "OUTPUT ONLY THIS OBJECT:",
        canonical_json(template),
    ])


def build_evidence_committer_prompt(
    *, proposed_delta: Mapping[str, Any], dialogue_turns: Sequence[Mapping[str, Any]],
    realizer_claims: Sequence[Mapping[str, Any]], memory_view: Mapping[str, Any],
) -> str:
    candidate_keys = list(dict.fromkeys(
        str(key)
        for claim in realizer_claims
        for key in claim.get("candidate_evidence_turn_keys") or []))
    candidate_set = set(candidate_keys)
    payload = {
        "proposed_graph_delta": dict(proposed_delta),
        "generated_dialogue_turns": list(dialogue_turns),
        "realizer_claimed_updates": list(realizer_claims),
        "allowed_candidate_evidence_turn_keys": candidate_keys,
        "candidate_evidence_turns": [dict(turn) for turn in dialogue_turns
                                      if str(turn.get("turn_key")) in candidate_set],
        "current_world_memory": dict(memory_view),
    }
    output_template = {
        "evidence_committer_schema_version": "evidence_committer_v1",
        "proposed_delta_id": "<copy the proposed_delta_id>",
        "decision": "<confirmed|rejected|uncertain>",
        "confirmed_post_state": "<nonempty when confirmed; otherwise empty string>",
        "evidence_turn_keys": ["<existing generated turn_key>"],
        "reason": "<brief semantic justification>",
        "semantic_status": "<semantically_verified|rejected|model_claimed>",
    }
    return "\n".join([
        f"Prompt version: {EVIDENCE_COMMITTER_PROMPT_VERSION}",
        "You are an independent Evidence Committer.",
        "Judge whether the generated dialogue semantically supports the proposed change.",
        "The Realizer claim is not proof. You may cite only a subset of the exact keys in "
        "allowed_candidate_evidence_turn_keys; never discover or add another scene turn.",
        "If the Realizer claimed_status is not_realized, confirmed is forbidden.",
        "Use decision confirmed, rejected, or uncertain. Confirmed requires semantic_status "
        "semantically_verified; rejected requires rejected; uncertain must use model_claimed.",
        "confirmed_post_state must always be a JSON string: use a nonempty supported state "
        "for confirmed, and exactly the empty string for rejected or uncertain.",
        "Return exactly seven fields. Do not output decisions, allowed_decisions, schema, "
        "type, metadata, or any wrapper field.",
        "Judge the meaning in context without relying on exact wording or prewritten markers.",
        "READ-ONLY COMMIT REQUEST — DO NOT RETURN THIS WRAPPER:",
        canonical_json(payload),
        "OUTPUT ONLY ONE JSON OBJECT WITH EXACTLY THESE SEVEN FIELDS:",
        "evidence_committer_schema_version, proposed_delta_id, decision, "
        "confirmed_post_state, evidence_turn_keys, reason, semantic_status",
        "Replace every placeholder below; do not add fields:",
        canonical_json(output_template),
    ])


def build_emergent_evidence_committer_prompt(
    *, pending_proposal: Mapping[str, Any],
    dialogue_turns: Sequence[Mapping[str, Any]],
    candidate_evidence_turns: Sequence[Mapping[str, Any]],
    memory_view: Mapping[str, Any],
) -> str:
    """Build an independent semantic request for an emergent open-thread claim."""
    payload = {
        "route_type": "emergent_open_thread",
        "emergent_pending_proposal": dict(pending_proposal),
        "generated_dialogue_turns": list(dialogue_turns),
        "candidate_evidence_turns": list(candidate_evidence_turns),
        "current_world_memory": dict(memory_view),
    }
    output_template = {
        "evidence_committer_schema_version": "evidence_committer_v1",
        "proposed_delta_id": "<copy pending proposal_id; this is not a thread ID>",
        "decision": "<confirmed|rejected|uncertain>",
        "confirmed_post_state": "<nonempty when confirmed; otherwise empty string>",
        "evidence_turn_keys": ["<existing generated turn_key>"],
        "reason": "<brief independent semantic judgment>",
        "semantic_status": "<semantically_verified|rejected|model_claimed>",
    }
    return "\n".join([
        f"Prompt version: {EVIDENCE_COMMITTER_PROMPT_VERSION}",
        "You are an independent Evidence Committer reviewing an emergent open-thread proposal.",
        "The proposal is model_claimed and has no formal thread or handoff ID.",
        "Do not trust the Realizer claim; judge the actual dialogue and cite exact turn_key values.",
        "You may cite only a subset of candidate_evidence_turns; never discover or add "
        "another generated dialogue turn.",
        "Use confirmed, rejected, or uncertain under the standard Evidence Committer contract.",
        "confirmed_post_state must always be a JSON string: use a nonempty supported state "
        "for confirmed, and exactly the empty string for rejected or uncertain.",
        "Confirmed means the dialogue establishes an open follow-up thread, not that it is completed.",
        "Do not generate a final memory, thread, session, turn, or handoff ID.",
        "Return exactly seven fields. Do not output decisions, allowed_decisions, schema, "
        "type, metadata, or any wrapper field.",
        "READ-ONLY COMMIT REQUEST — DO NOT RETURN THIS WRAPPER:",
        canonical_json(payload),
        "OUTPUT ONLY ONE JSON OBJECT WITH EXACTLY THESE SEVEN FIELDS:",
        "evidence_committer_schema_version, proposed_delta_id, decision, "
        "confirmed_post_state, evidence_turn_keys, reason, semantic_status",
        "Replace every placeholder below; do not add fields:",
        canonical_json(output_template),
    ])


def compact_evidence_committer_retry_errors(errors: Sequence[str]) -> list[str]:
    """Translate structural diagnostics into short, non-echoing repair guidance."""
    result: list[str] = []
    if any(error.startswith("evidence_committer_unexpected_field:")
           for error in errors):
        result.append("remove_all_unexpected_fields_including_decisions")
    if any(error.startswith("evidence_committer_missing_field:")
           for error in errors):
        result.append("restore_every_required_output_field")
    if "evidence_committer_schema" in errors:
        result.append("output_must_have_exact_seven_fields")
    if "evidence_committer_values" in errors:
        result.append(
            "confirmed_post_state_must_be_a_string_and_all_references_must_be_valid")
    for error in errors:
        if (error == "evidence_committer_schema" or
                error == "evidence_committer_values" or
                error.startswith("evidence_committer_unexpected_field:") or
                error.startswith("evidence_committer_missing_field:")):
            continue
        if error not in result:
            result.append(str(error))
    return result[:10]


def build_evidence_committer_retry_prompt(
    *, frozen_base_prompt: str, errors: Sequence[str],
) -> str:
    """Retry the same semantic request without appending the rejected response."""
    return "\n".join([
        f"Prompt version: {EVIDENCE_COMMITTER_RETRY_PROMPT_VERSION}",
        "Repair output structure/reference errors only; independently judge the same dialogue.",
        "The previous rejected response is intentionally omitted. Do not recreate extra fields.",
        "RETRY DIAGNOSTICS:",
        canonical_json(compact_evidence_committer_retry_errors(errors)),
        "FROZEN ORIGINAL REQUEST AND EXACT OUTPUT TEMPLATE FOLLOW:",
        frozen_base_prompt,
    ])
