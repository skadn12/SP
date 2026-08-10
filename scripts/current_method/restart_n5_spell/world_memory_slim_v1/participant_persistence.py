"""Round-level semantic contract for promoting dialogue participants to Memory."""

from __future__ import annotations

import unicodedata
import json
from copy import deepcopy
from typing import Any, Mapping, Sequence


PARTICIPANT_PERSISTENCE_REVIEW_VERSION = (
    "participant_persistence_review_v2_evidence_only"
)
PERSISTENCE_DECISIONS = frozenset({"persist", "do_not_persist", "uncertain", "invalid"})
REVIEW_STATUSES = frozenset({"confirmed", "rejected", "uncertain"})
EQUIVALENCE_STATUSES = frozenset({"distinct", "existing_equivalent", "uncertain"})
CONNECTION_KINDS = frozenset({
    "interaction_link", "explicit_relation", "event_participant",
    "task_participant", "information_source", "thread_participant",
})
PARTICIPANT_REVIEW_FIELDS = frozenset({
    "participant_ref", "persistence_decision",
    "identity_status", "connection_status", "equivalence_status",
    "equivalent_entity_ref", "verified_connection",
    "current_scene_evidence_turn_refs", "dependency_context_turn_refs", "reason",
})
VERIFIED_CONNECTION_FIELDS = frozenset({
    "connection_kind", "target_ref", "relation_type", "description",
    "evidence_turn_refs",
})


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def participant_persistence_review_schema() -> dict[str, Any]:
    return {
        "participant_persistence_review_version": PARTICIPANT_PERSISTENCE_REVIEW_VERSION,
        "item_exact_fields": sorted(PARTICIPANT_REVIEW_FIELDS),
        "persistence_decision_enum": sorted(PERSISTENCE_DECISIONS),
        "identity_and_connection_status_enum": sorted(REVIEW_STATUSES),
        "equivalence_status_enum": sorted(EQUIVALENCE_STATUSES),
        "verified_connection_exact_fields": sorted(VERIFIED_CONNECTION_FIELDS),
        "verified_connection_kind_enum": sorted(CONNECTION_KINDS),
        "authoritative_commit_gate": [
            "persistence_decision=persist",
            "identity_status=confirmed", "connection_status=confirmed",
            "equivalence_status=distinct", "verified evidence refs resolve",
        ],
    }


def build_participant_persistence_prompt_extension(
    round_participant_registry: Mapping[str, Any],
) -> str:
    """Prompt fragment for the existing one-shot Round Validator request."""
    participants = [{key: deepcopy(row.get(key)) for key in (
        "participant_ref", "participant_alias", "display_name",
        "first_scene_alias", "appearance_scene_aliases", "entry_context",
        "anchor_connection")}
        for row in round_participant_registry.get("participants") or []]
    template = [{
        "participant_ref": row["participant_ref"],
        "persistence_decision": None,
        "identity_status": None,
        "connection_status": None,
        "equivalence_status": None,
        "equivalent_entity_ref": None,
        "verified_connection": None,
        "current_scene_evidence_turn_refs": [],
        "dependency_context_turn_refs": [],
        "reason": "",
    } for row in participants]
    return "\n\n".join([
        "DIALOGUE-FIRST PARTICIPANT PERSISTENCE REVIEW",
        "In the SAME one-shot round review response, add participant_reviews. Review every "
        "round participant exactly once. The Planner does not decide persistence. A participant "
        "may persist only when identity and a typed world connection are confirmed from exact dialogue refs "
        "and it is distinct from existing entities. Speaking alone never creates Memory. "
        "A retained scene may have do_not_persist or uncertain participant reviews.",
        "PARTICIPANTS\n" + json.dumps(participants, ensure_ascii=False,
                                      separators=(",", ":")),
        "SCHEMA\n" + json.dumps(participant_persistence_review_schema(),
                                 ensure_ascii=False, separators=(",", ":")),
        "PARTICIPANT_REVIEWS TEMPLATE\n" + json.dumps(
            template, ensure_ascii=False, separators=(",", ":")),
    ])


def build_dialogue_first_round_validator_prompt(
    validator_input: Mapping[str, Any],
    round_participant_registry: Mapping[str, Any],
) -> str:
    """Build one review request for scenes, actions, and participant persistence."""
    from .round_expansion_validator_v1 import build_round_validator_prompt
    from .round_validator_contract import ROUND_VALIDATOR_TERMINAL_INSTRUCTION

    base = build_round_validator_prompt(validator_input)
    terminal = ROUND_VALIDATOR_TERMINAL_INSTRUCTION
    if not base.endswith(terminal):
        raise RuntimeError("round_validator_base_prompt_terminal_drift")
    base = base[:-len(terminal)].rstrip()
    return "\n\n".join([
        base,
        build_participant_persistence_prompt_extension(
            round_participant_registry),
        "OUTPUT TOP LEVEL EXACT KEYS: round_decision, round_reason_codes, "
        "scene_reviews, durable_updates, semantic_review_observations, "
        "participant_reviews. Return every participant review in registry order.",
        terminal,
    ])


def compile_dialogue_first_round_validator_output(
    output: Any, *, realized_scenes: Sequence[Mapping[str, Any]],
    review_world_view: Mapping[str, Any],
    local_gate_results: Mapping[str, Mapping[str, Any]],
    round_participant_registry: Mapping[str, Any],
    source_turn_registry: Mapping[str, Any],
    authoritative_memory: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile one response containing the legacy round review plus participant reviews."""
    from .round_expansion_validator_v1 import (
        compile_round_validator_model_output, validate_round_validator_output,
    )
    from .round_validator_contract import MODEL_TOP_LEVEL_FIELDS

    expected = set(MODEL_TOP_LEVEL_FIELDS) | {"participant_reviews"}
    if not isinstance(output, Mapping) or set(output) != expected:
        return {"valid": False, "classification": "schema_invalid",
                "errors": ["dialogue_first_validator_top_level_schema"],
                "compiled_output": None}
    base = {key: deepcopy(output[key]) for key in MODEL_TOP_LEVEL_FIELDS}
    compilation = compile_round_validator_model_output(
        base, realized_scenes=realized_scenes)
    if not compilation["valid"]:
        return {**compilation, "compiled_output": None}
    compiled_base = compilation["compiled_output"]
    base_validation = validate_round_validator_output(
        compiled_base, realized_scenes=realized_scenes,
        review_world_view=review_world_view,
        local_gate_results=local_gate_results)
    alias_to_ref = {str(row.get("scene_alias") or ""): str(
        row.get("scene_ref") or "") for row in realized_scenes}
    accepted = [alias_to_ref.get(str(row.get("scene_alias") or ""), "")
                for row in compiled_base.get("scene_reviews") or []
                if row.get("decision") == "accept"]
    participant_validation = validate_participant_persistence_reviews(
        output.get("participant_reviews"),
        round_participant_registry=round_participant_registry,
        realized_scenes=realized_scenes,
        source_turn_registry=source_turn_registry,
        authoritative_memory=authoritative_memory,
        accepted_scene_refs=[value for value in accepted if value],
    )
    errors = [*base_validation["errors"], *participant_validation["errors"]]
    compiled = deepcopy(compiled_base)
    compiled["participant_reviews"] = deepcopy(output.get("participant_reviews"))
    compiled["participant_persistence_review_version"] = (
        PARTICIPANT_PERSISTENCE_REVIEW_VERSION)
    return {
        "valid": not errors,
        "classification": ("valid" if not errors else
                           "reference_invalid" if any("ref" in value for value in errors)
                           else "schema_invalid"),
        "errors": list(dict.fromkeys(errors)),
        "compiled_output": compiled,
        "round_review_validation": base_validation,
        "participant_review_validation": participant_validation,
    }


def validate_participant_persistence_reviews(
    reviews: Any, *, round_participant_registry: Mapping[str, Any],
    realized_scenes: Sequence[Mapping[str, Any]],
    source_turn_registry: Mapping[str, Any],
    authoritative_memory: Mapping[str, Any],
    accepted_scene_refs: Sequence[str],
) -> dict[str, Any]:
    """Validate identity/connection decisions without deciding them locally."""
    errors: list[str] = []
    participants = {
        str(row.get("participant_ref") or ""): row
        for row in round_participant_registry.get("participants") or []
    }
    if not isinstance(reviews, list):
        return {"valid": False, "errors": ["participant_reviews_not_list"],
                "persistable_participant_refs": [], "audit_only_participant_refs": []}
    turns = {
        str(row.get("turn_ref") or ""): row
        for row in source_turn_registry.get("turns") or []
    }
    scene_turn_refs: dict[str, set[str]] = {}
    speaker_turn_refs: dict[str, set[str]] = {}
    scene_by_alias: dict[str, str] = {}
    direct: dict[str, set[str]] = {}
    for scene in realized_scenes:
        scene_ref = str(scene.get("scene_ref") or "")
        scene_by_alias[str(scene.get("scene_alias") or "")] = scene_ref
        direct[scene_ref] = {str(value) for value in
                             scene.get("predecessor_scene_refs") or []}
        for turn in scene.get("dialogue_turns") or []:
            ref = str(turn.get("turn_ref") or "")
            scene_turn_refs.setdefault(scene_ref, set()).add(ref)
            speaker_turn_refs.setdefault(str(turn.get("speaker_ref") or ""), set()).add(ref)

    def ancestry(scene_ref: str) -> set[str]:
        result: set[str] = set()
        pending = list(direct.get(scene_ref, set()))
        while pending:
            ref = pending.pop()
            if ref in result:
                continue
            result.add(ref)
            pending.extend(direct.get(ref, set()))
        return result

    accepted = {str(value) for value in accepted_scene_refs}
    entity_refs = {
        str(row.get("entity_ref") or row.get("entity_id") or "")
        for row in authoritative_memory.get("entities") or []
    }
    record_by_ref = {
        str(row.get("memory_ref") or row.get("memory_id") or ""): row
        for row in authoritative_memory.get("authoritative_records") or []
    }
    root_id = str(authoritative_memory.get("root_id") or "")
    existing_names = {
        _normalized(str(row.get("display_name") or "")): str(
            row.get("entity_ref") or row.get("entity_id") or "")
        for row in authoritative_memory.get("entities") or []
        if _text(row.get("display_name"))
    }
    seen: set[str] = set()
    persistable: list[str] = []
    audit_only: list[str] = []
    normalized_reviews: list[dict[str, Any]] = []
    for index, review in enumerate(reviews):
        prefix = f"participant_review:{index}"
        if not isinstance(review, Mapping) or set(review) != PARTICIPANT_REVIEW_FIELDS:
            errors.append(f"{prefix}:schema")
            continue
        ref = str(review.get("participant_ref") or "")
        participant = participants.get(ref)
        if participant is None:
            errors.append(f"{prefix}:participant_ref_unknown:{ref}")
            continue
        if ref in seen:
            errors.append(f"{prefix}:participant_ref_duplicate:{ref}")
        seen.add(ref)
        decision = str(review.get("persistence_decision") or "")
        if decision not in PERSISTENCE_DECISIONS:
            errors.append(f"{prefix}:persistence_decision")
        if review.get("identity_status") not in REVIEW_STATUSES:
            errors.append(f"{prefix}:identity_status")
        if review.get("connection_status") not in REVIEW_STATUSES:
            errors.append(f"{prefix}:connection_status")
        if review.get("equivalence_status") not in EQUIVALENCE_STATUSES:
            errors.append(f"{prefix}:equivalence_status")
        equivalent_ref = review.get("equivalent_entity_ref")
        if review.get("equivalence_status") == "existing_equivalent":
            if not isinstance(equivalent_ref, str) or equivalent_ref not in entity_refs:
                errors.append(f"{prefix}:equivalent_entity_ref")
        elif equivalent_ref is not None:
            errors.append(f"{prefix}:unexpected_equivalent_entity_ref")
        if not _text(review.get("reason")):
            errors.append(f"{prefix}:reason")

        appearances = {
            scene_by_alias.get(str(alias), "")
            for alias in participant.get("appearance_scene_aliases") or []}
        appearances.discard("")
        current_evidence = review.get("current_scene_evidence_turn_refs")
        if (not isinstance(current_evidence, list) or any(
                str(value) not in turns or
                str(value) not in speaker_turn_refs.get(ref, set()) or
                not any(str(value) in scene_turn_refs.get(scene_ref, set())
                        for scene_ref in appearances)
                for value in current_evidence)):
            errors.append(f"{prefix}:current_scene_evidence_turn_refs")
            current_evidence = []
        allowed_dependency_scenes = set().union(
            *(ancestry(scene_ref) for scene_ref in appearances)) if appearances else set()
        dependency_evidence = review.get("dependency_context_turn_refs")
        if (not isinstance(dependency_evidence, list) or any(
                str(value) not in turns or
                not any(str(value) in scene_turn_refs.get(scene_ref, set())
                        for scene_ref in allowed_dependency_scenes)
                for value in dependency_evidence)):
            errors.append(f"{prefix}:dependency_context_turn_refs")
            dependency_evidence = []

        connection = review.get("verified_connection")
        connection_valid = isinstance(connection, Mapping) and set(
            connection) == VERIFIED_CONNECTION_FIELDS
        if decision == "persist" or review.get("connection_status") == "confirmed":
            if not connection_valid:
                errors.append(f"{prefix}:verified_connection_required")
        if connection_valid:
            kind = str(connection.get("connection_kind") or "")
            target = str(connection.get("target_ref") or "")
            evidence = connection.get("evidence_turn_refs")
            if kind not in CONNECTION_KINDS:
                errors.append(f"{prefix}:connection_kind")
            if target not in entity_refs and target not in record_by_ref:
                errors.append(f"{prefix}:connection_target_ref")
            if root_id and target and not target.startswith(root_id + ":"):
                errors.append(f"{prefix}:connection_cross_root")
            if kind == "explicit_relation" and not _text(connection.get("relation_type")):
                errors.append(f"{prefix}:relation_type_required")
            if kind != "explicit_relation" and connection.get("relation_type") is not None:
                errors.append(f"{prefix}:relation_type_forbidden")
            if not _text(connection.get("description")):
                errors.append(f"{prefix}:connection_description")
            allowed_evidence = {str(value) for value in [
                *(current_evidence or []), *(dependency_evidence or [])]}
            if (not isinstance(evidence, list) or not evidence or
                    any(str(value) not in allowed_evidence for value in evidence)):
                errors.append(f"{prefix}:connection_evidence_turn_refs")

        first_scene = scene_by_alias.get(str(participant.get("first_scene_alias") or ""), "")
        persist_gate = (
            decision == "persist" and
            review.get("identity_status") == "confirmed" and
            review.get("connection_status") == "confirmed" and
            review.get("equivalence_status") == "distinct" and
            connection_valid and bool(current_evidence) and first_scene in accepted)
        if decision == "persist" and not persist_gate:
            errors.append(f"{prefix}:persistence_gate_unsatisfied")
        if persist_gate:
            persistable.append(ref)
        else:
            audit_only.append(ref)
        name = _normalized(str(participant.get("display_name") or ""))
        normalized_reviews.append({
            **dict(review),
            "display_name_collision_refs": ([existing_names[name]]
                                            if name in existing_names else []),
            "persistence_gate_passed": persist_gate,
        })
    missing = sorted(set(participants) - seen)
    errors.extend(f"participant_review_missing:{ref}" for ref in missing)
    return {
        "valid": not errors,
        "errors": list(dict.fromkeys(errors)),
        "participant_persistence_review_version": PARTICIPANT_PERSISTENCE_REVIEW_VERSION,
        "persistable_participant_refs": persistable,
        "audit_only_participant_refs": audit_only,
        "normalized_reviews": normalized_reviews,
        "orphan_person_count": 0 if not errors else None,
    }
