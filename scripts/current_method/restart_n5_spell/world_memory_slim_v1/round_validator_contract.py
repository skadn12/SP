"""Frozen schemas and enums for one-shot Round Expansion Validator v1."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


ROUND_VALIDATOR_SCHEMA_VERSION = (
    "round_expansion_validator_semantic_review_v6_uniform_action_review")
ROUND_VALIDATOR_MODEL_OUTPUT_VERSION = "round_expansion_validator_model_semantic_v1"
ROUND_VALIDATOR_PROMPT_VERSION = (
    "round_expansion_validator_person_event_growth_v6_prompt_v17_uniform_action_review"
)
ROUND_VALIDATOR_RETRY_PROMPT_VERSION = (
    "round_expansion_validator_person_event_growth_v6_retry_v12_uniform_action_review"
)
ROUND_VALIDATOR_MAX_ATTEMPTS = 2
ROUND_VALIDATOR_TERMINAL_CONTRACT_REPAIR_ATTEMPTS = 1
ROUND_VALIDATOR_ABSOLUTE_MAX_ATTEMPTS = (
    ROUND_VALIDATOR_MAX_ATTEMPTS +
    ROUND_VALIDATOR_TERMINAL_CONTRACT_REPAIR_ATTEMPTS
)
ROUND_VALIDATOR_TERMINAL_INSTRUCTION = (
    "NEVER OUTPUT type, properties, required, or additionalProperties."
)

ROUND_DECISIONS = frozenset({"accept", "partial_accept", "reject", "uncertain"})
SCENE_DECISIONS = frozenset({"accept", "reject", "uncertain"})
CONNECTION_STATUSES = frozenset({"valid", "invalid", "uncertain"})
CONTINUITY_STATUSES = frozenset({"valid", "invalid", "uncertain"})
CONSISTENCY_STATUSES = frozenset({"valid", "contradiction", "uncertain"})
DUPLICATE_STATUSES = frozenset({"not_duplicate", "duplicate", "uncertain"})
DEVELOPMENT_STATUSES = frozenset({"valid", "insufficient", "uncertain"})
VERIFICATION_STATUSES = frozenset({"confirmed", "rejected", "uncertain"})
ACTION_REALIZATION_STATUSES = frozenset({
    "realized", "partially_realized", "not_realized", "invalid",
})

ROUND_REASON_CODES = frozenset({
    "supported", "partial_scene_rejection", "historical_disconnect",
    "continuity_break", "future_knowledge_leak", "historical_contradiction",
    "semantic_duplicate", "insufficient_new_development", "dependency_blocked",
    "independent_branch_retained", "no_durable_update", "uncertain_review",
})
SCENE_REASON_CODES = ROUND_REASON_CODES | frozenset({
    "local_gate_passed", "anchor_supported", "branch_supported",
    "update_not_required", "update_required", "primary_action_realized",
    "primary_action_not_realized", "secondary_action_partial",
})

TOP_LEVEL_FIELDS = frozenset({
    "validator_schema_version", "round_decision", "round_reason_codes",
    "scene_reviews", "durable_updates", "semantic_review_observations",
})
MODEL_TOP_LEVEL_FIELDS = TOP_LEVEL_FIELDS - {"validator_schema_version"}
SCENE_REVIEW_FIELDS = frozenset({
    "scene_alias", "decision", "historical_connection_status",
    "continuity_status", "future_knowledge_leak",
    "historical_consistency_status", "semantic_duplicate_status",
    "new_development_status", "reason_codes",
    "historical_supporting_turn_refs", "current_scene_supporting_turn_refs",
    "dependency_context_turn_refs",
    "uses_scene_refs", "scene_requires_update_commit",
    "primary_action_realization", "secondary_action_realization",
})
MODEL_SCENE_REVIEW_FIELDS = SCENE_REVIEW_FIELDS
MODEL_OPTIONAL_SCENE_REVIEW_FIELDS = frozenset({
    "dependency_context_turn_refs",
})
ACTION_REALIZATION_FIELDS = frozenset({
    "action_ref", "action_mode", "continuation_of", "action_type", "status",
    "evidence_turn_refs", "progression_evidence_turn_refs", "reason",
})
MODEL_ACTION_REALIZATION_FIELDS = frozenset({
    "status", "evidence_turn_refs", "progression_evidence_turn_refs", "reason",
})
MODEL_NEW_ACTION_REALIZATION_FIELDS = MODEL_ACTION_REALIZATION_FIELDS
MODEL_CONTINUATION_ACTION_REALIZATION_FIELDS = MODEL_ACTION_REALIZATION_FIELDS
DURABLE_UPDATE_FIELDS = frozenset({
    "candidate_update_key", "source_scene_ref", "source_action_role",
    "evidence_turn_refs", "verification_status", "atomic_group_key",
    "depends_on_candidate_keys",
})
CONNECTED_PERSON_DURABLE_UPDATE_FIELDS = DURABLE_UPDATE_FIELDS | {
    "verified_connection",
}
VERIFIED_CONNECTION_FIELDS = frozenset({
    "verification_status", "connection_kind", "relation_type",
    "evidence_turn_refs", "reason",
})
VERIFIED_CONNECTION_KINDS = frozenset({
    "interaction_link", "explicit_relation", "event_participant",
    "task_participant", "information_source", "thread_participant",
})
OBSERVATION_FIELDS = frozenset({
    "observation_key", "update_type", "source_scene_refs",
    "evidence_turn_refs", "description", "disposition",
})


def round_validator_output_schema() -> dict[str, Any]:
    """Return the model-owned semantic output shape for audit use."""
    return {
        "type": "object", "additionalProperties": False,
        "model_output_version": ROUND_VALIDATOR_MODEL_OUTPUT_VERSION,
        "required": sorted(MODEL_TOP_LEVEL_FIELDS),
        "properties": {
            "round_decision": {"enum": sorted(ROUND_DECISIONS)},
            "round_reason_codes": {"type": "array", "items": {
                "enum": sorted(ROUND_REASON_CODES)}},
            "scene_reviews": {
                "type": "array",
                "item_required": sorted(
                    MODEL_SCENE_REVIEW_FIELDS - MODEL_OPTIONAL_SCENE_REVIEW_FIELDS),
                "item_optional": sorted(MODEL_OPTIONAL_SCENE_REVIEW_FIELDS),
            },
            "durable_updates": {"type": "array", "item_required": sorted(
                DURABLE_UPDATE_FIELDS)},
            "semantic_review_observations": {"type": "array", "item_required": sorted(
                OBSERVATION_FIELDS)},
        },
        "action_realization_shape": {
            "required": sorted(MODEL_ACTION_REALIZATION_FIELDS),
            "new_progression_evidence_rule": "must_be_empty",
            "continuation_progression_evidence_rule": (
                "nonempty_for_realized_or_partially_realized"),
        },
        "verified_connection_shape": {
            "connected_person_item_required": sorted(
                CONNECTED_PERSON_DURABLE_UPDATE_FIELDS),
            "required": sorted(VERIFIED_CONNECTION_FIELDS),
            "connection_kind": sorted(VERIFIED_CONNECTION_KINDS),
            "scope": "introduce_connected_person durable candidates only",
        },
        "program_owned_fields": [
            "validator_schema_version", "action_ref", "action_mode",
            "continuation_of", "action_type",
        ],
    }


def empty_round_validator_output() -> dict[str, Any]:
    return {
        "validator_schema_version": ROUND_VALIDATOR_SCHEMA_VERSION,
        "round_decision": "accept", "round_reason_codes": ["no_durable_update"],
        "scene_reviews": [], "durable_updates": [],
        "semantic_review_observations": [],
    }


def copy_output_schema() -> dict[str, Any]:
    return deepcopy(round_validator_output_schema())
