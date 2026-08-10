"""Planner Contract v3.3: safe action projection, atomic selection, and novelty."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .operation_registry import OPERATION_REGISTRY, validate_operation_registry
from .relation_registry import (RELATION_TYPE_REGISTRY, relation_contract,
                                validate_relation_registry)


PLANNER_CONTRACT_VERSION = "slim_world_planner_contract_v3_3"
PLANNER_ACTION_VIEW_VERSION = "planner_action_view_v1"
PLANNER_BUNDLE_VERSION = "planner_bundle_selection_v1"
PLANNER_NOVELTY_VERSION = "planner_proposal_novelty_v1"

FROZEN_PLANNER_FIELDS = frozenset({
    "participant_refs", "subject_ref", "subject_refs", "source_ref", "target_ref",
    "knower_refs", "non_knower_refs", "affected_refs", "reporter_ref", "about_refs",
    "scene_presence",
    "allowed_existing_entity_refs", "temporary_entity_refs", "target_memory_ref",
    "operation", "operation_class", "pre_state", "post_state", "authoritative_version",
    "proposal_ref", "scene_ref", "claim_ref", "proposal_dependencies",
    "atomic_group_ref", "final_entity_id", "final_event_id", "final_relation_id",
    "final_thread_id", "final_memory_id", "handoff_id",
})

_ACTION_SUMMARIES = {
    "add_person": "Introduce one neutrally described mentioned person.",
    "add_relation": "Create one canonical relation between program-selected participants.",
    "add_event": "Create a concrete shared event for the connected participants.",
    "create_state": "Create a new explicit state dimension for the selected subject.",
    "add_shared_task": "Create a shared task for the connected participants.",
    "open_thread": "Open one concrete follow-up thread for the selected subject.",
    "add_third_party_report": "Create a third-party report involving the connected participants.",
    "add_information_asymmetry": "Create new information known to only the selected side.",
    "create_conflict": "Create a concrete conflict between the connected participants.",
    "create_misunderstanding": "Create a concrete misunderstanding between participants.",
    "introduce_external_constraint": "Introduce a concrete external constraint.",
    "add_role_handoff": "Create one responsibility handoff between selected participants.",
    "update_state": "Change the value of the selected active state.",
    "change_trust": "Change the selected active trust state.",
    "change_authority": "Change the selected active authority state.",
    "resolve_conflict": "Resolve the selected active conflict.",
    "close_thread": "Close the selected active thread.",
    "supersede": "Replace the selected active record with a verified new version.",
}


def compile_planner_action_view(
    action_menu: Mapping[str, Any], *, role_table: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Project internal actions into a non-editable summary plus writable JSON schema."""
    registry = validate_operation_registry()
    relations = validate_relation_registry()
    if not registry["valid"] or not relations["valid"]:
        raise ValueError("planner_projection_registry_invalid")
    labels = {str(row.get("entity_ref") or row.get("entity_id")):
              str(row.get("display_name") or row.get("display_label") or
                  row.get("entity_ref") or row.get("entity_id"))
              for row in role_table}
    entity_types = {str(row.get("entity_ref") or row.get("entity_id")):
                    str(row.get("entity_type") or "person")
                    for row in role_table}
    selection_policies = _selection_policies(action_menu)
    actions: list[dict[str, Any]] = []
    for internal in action_menu.get("actions") or []:
        operation = str(internal.get("operation") or "")
        if operation not in OPERATION_REGISTRY:
            raise ValueError(f"planner_projection_unknown_operation:{operation}")
        frozen = set((internal.get("frozen_payload") or {}).keys())
        schema = internal.get("payload_schema")
        if not isinstance(schema, Mapping):
            raise ValueError(f"planner_projection_internal_schema_missing:{internal.get('action_id')}")
        writable = {str(key): _json_schema(kind)
                    for key, kind in schema.items() if key not in frozen}
        if "relation_type" in writable:
            endpoints = dict(internal.get("frozen_payload") or {})
            source_ref = str(endpoints.get("source_ref") or
                             (internal.get("allowed_existing_entity_refs") or [""])[0])
            target_values = list(internal.get("allowed_existing_entity_refs") or [])
            target_ref = str(endpoints.get("target_ref") or
                             (target_values[1] if len(target_values) > 1 else ""))
            source_type = entity_types.get(source_ref, "person")
            target_type = entity_types.get(target_ref, "person")
            writable["relation_type"]["enum"] = sorted(
                name for name, contract in RELATION_TYPE_REGISTRY.items()
                if source_type in contract["source_entity_types"] and
                target_type in contract["target_entity_types"])
        overlap = sorted(frozen & set(writable))
        if overlap:
            raise ValueError("planner_projection_frozen_writable_overlap:" + ",".join(overlap))
        required = [str(key) for key, kind in schema.items()
                    if key not in frozen and not _optional(kind)]
        participants = [{"participant_ref": ref,
                         "display_label": labels.get(ref, ref)}
                        for ref in internal.get("allowed_existing_entity_refs") or []]
        policy = selection_policies.get(str(internal["action_id"]), {
            "selection_policy": "single_allowed", "required_bundle_id": None})
        actions.append({
            "action_id": str(internal["action_id"]),
            "action_summary": _ACTION_SUMMARIES[operation],
            "selection_policy": policy["selection_policy"],
            "required_bundle_id": policy["required_bundle_id"],
            "connected_participants": participants,
            "writable_payload_schema": {
                "type": "object", "properties": writable,
                "required": required, "additionalProperties": False,
            },
        })
    result = {
        "planner_action_view_version": PLANNER_ACTION_VIEW_VERSION,
        "planner_contract_version": PLANNER_CONTRACT_VERSION,
        "root_id": str(action_menu.get("root_id") or ""),
        "actions": actions,
    }
    result["planner_action_view_sha256"] = _sha(result)
    return result


def compile_allowed_proposal_bundles(action_menu: Mapping[str, Any]) -> dict[str, Any]:
    """Compile the only currently supported atomic/dependent pair."""
    by_operation = {str(row.get("operation")): row
                    for row in action_menu.get("actions") or []}
    bundles: list[dict[str, Any]] = []
    person = by_operation.get("add_person")
    relation = by_operation.get("add_relation")
    if person and relation:
        existing = list(relation.get("allowed_existing_entity_refs") or [])
        if not existing:
            raise ValueError("planner_bundle_relation_source_missing")
        bundles.append({
            "bundle_id": "B0001",
            "bundle_type": "person_identity_relation",
            "required_action_ids": [str(person["action_id"]), str(relation["action_id"])],
            "bundle_summary": (
                "Introduce a mentioned person and establish one canonical "
                "identity-bearing relation to an existing participant."),
            "frozen_payload_overrides": {
                str(relation["action_id"]): {
                    "source_ref": str(existing[0]),
                    "target_ref": "temp:bundle_person",
                },
            },
            "dependency_edges": [{
                "action_id": str(relation["action_id"]),
                "depends_on_action_ids": [str(person["action_id"])],
            }],
            "atomic_commit_required": True,
            "temporary_entity_binding": {
                "creator_action_id": str(person["action_id"]),
                "consumer_action_id": str(relation["action_id"]),
                "placeholder_ref": "temp:bundle_person",
            },
        })
    result = {"planner_bundle_version": PLANNER_BUNDLE_VERSION,
              "root_id": str(action_menu.get("root_id") or ""),
              "bundles": bundles}
    result["planner_bundle_sha256"] = _sha(result)
    return result


def validate_planner_action_view(
    action_menu: Mapping[str, Any], action_view: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    if action_view.get("planner_action_view_version") != PLANNER_ACTION_VIEW_VERSION:
        errors.append("planner_action_view_version_invalid")
    if action_view.get("root_id") != action_menu.get("root_id"):
        errors.append("planner_action_view_root_mismatch")
    internal_by_id = {str(row.get("action_id")): row
                      for row in action_menu.get("actions") or []}
    projected_by_id = {str(row.get("action_id")): row
                       for row in action_view.get("actions") or []}
    if set(internal_by_id) != set(projected_by_id):
        errors.append("planner_action_view_action_ids_mismatch")
    for action_id in sorted(set(internal_by_id) & set(projected_by_id)):
        internal = internal_by_id[action_id]
        projected = projected_by_id[action_id]
        expected_policy = _selection_policies(action_menu).get(action_id, {
            "selection_policy": "single_allowed", "required_bundle_id": None})
        if projected.get("selection_policy") != expected_policy["selection_policy"]:
            errors.append(f"planner_selection_policy_mismatch:{action_id}")
        if projected.get("required_bundle_id") != expected_policy["required_bundle_id"]:
            errors.append(f"planner_required_bundle_mismatch:{action_id}")
        writable = projected.get("writable_payload_schema")
        if not isinstance(writable, Mapping) or writable.get("additionalProperties") is not False:
            errors.append(f"planner_writable_schema_invalid:{action_id}")
            continue
        properties = writable.get("properties")
        if not isinstance(properties, Mapping):
            errors.append(f"planner_writable_properties_invalid:{action_id}")
            continue
        frozen = set((internal.get("frozen_payload") or {}).keys())
        expected = set((internal.get("payload_schema") or {}).keys()) - frozen
        if set(properties) != expected:
            errors.append(f"planner_writable_fields_mismatch:{action_id}")
        if set(properties) & (FROZEN_PLANNER_FIELDS | frozen):
            errors.append(f"planner_frozen_field_exposed:{action_id}")
        leaked = {"operation", "operation_class", "payload_schema", "frozen_payload",
                  "target_memory_ref", "pre_state", "authoritative_version"} & set(projected)
        if leaked:
            errors.append(f"planner_internal_action_field_exposed:{action_id}")
    return {"valid": not errors, "errors": errors,
            "planner_action_view_version": PLANNER_ACTION_VIEW_VERSION}


def planner_bundle_view(
    bundles: Mapping[str, Any], action_view: Mapping[str, Any],
) -> dict[str, Any]:
    action_by_id = {str(row["action_id"]): row for row in action_view.get("actions") or []}
    values = []
    for bundle in bundles.get("bundles") or []:
        values.append({
            "bundle_id": bundle["bundle_id"],
            "bundle_summary": bundle["bundle_summary"],
            "required_action_ids": list(bundle["required_action_ids"]),
            "complete_selection_required": True,
            "required_action_views": [deepcopy(action_by_id[action_id])
                                      for action_id in bundle["required_action_ids"]],
        })
    return {"planner_bundle_version": PLANNER_BUNDLE_VERSION,
            "bundles": values}


def validate_bundle_selection_registry(
    action_menu: Mapping[str, Any], bundles: Mapping[str, Any],
    action_view: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate frozen bundle membership and bundle-only action policies."""
    errors: list[str] = []
    actions = {str(row.get("action_id")): row
               for row in action_menu.get("actions") or []}
    projected = {str(row.get("action_id")): row
                 for row in action_view.get("actions") or []}
    membership: dict[str, str] = {}
    bundle_ids: set[str] = set()
    for bundle in bundles.get("bundles") or []:
        bundle_id = str(bundle.get("bundle_id") or "")
        required = bundle.get("required_action_ids")
        if not bundle_id or bundle_id in bundle_ids:
            errors.append(f"bundle_definition_duplicate_or_missing_id:{bundle_id}")
        bundle_ids.add(bundle_id)
        if not isinstance(required, list) or not required:
            errors.append(f"bundle_definition_required_actions_missing:{bundle_id}")
            continue
        if len(required) != len(set(str(value) for value in required)):
            errors.append(f"bundle_definition_duplicate_action:{bundle_id}")
        required_ids = {str(value) for value in required}
        overrides = bundle.get("frozen_payload_overrides")
        if not isinstance(overrides, Mapping):
            errors.append(f"bundle_frozen_overrides_missing:{bundle_id}")
            overrides = {}
        for override_action_id, override_payload in overrides.items():
            action_id = str(override_action_id)
            if action_id not in required_ids:
                errors.append(
                    f"bundle_override_action_not_required:{bundle_id}:{action_id}")
            writable = (projected.get(action_id) or {}).get(
                "writable_payload_schema") or {}
            properties = set(writable.get("properties") or {})
            if not isinstance(override_payload, Mapping):
                errors.append(f"bundle_override_payload_invalid:{bundle_id}:{action_id}")
            elif set(override_payload) & properties:
                errors.append(f"bundle_frozen_writable_conflict:{action_id}")
        dependency_edges = bundle.get("dependency_edges")
        if not isinstance(dependency_edges, list):
            errors.append(f"bundle_dependency_metadata_missing:{bundle_id}")
            dependency_edges = []
        for edge in dependency_edges:
            if (not isinstance(edge, Mapping) or
                    set(edge) != {"action_id", "depends_on_action_ids"} or
                    not isinstance(edge.get("depends_on_action_ids"), list)):
                errors.append(f"bundle_dependency_metadata_invalid:{bundle_id}")
                continue
            action_id = str(edge.get("action_id") or "")
            dependencies = [str(value) for value in edge["depends_on_action_ids"]]
            if action_id not in required_ids or any(
                    value not in required_ids for value in dependencies):
                errors.append(f"bundle_dependency_unknown_action:{bundle_id}")
            if action_id in dependencies or len(dependencies) != len(set(dependencies)):
                errors.append(f"bundle_dependency_self_or_duplicate:{bundle_id}:{action_id}")
        if bundle.get("atomic_commit_required") is not True:
            errors.append(f"bundle_atomic_commit_metadata_invalid:{bundle_id}")
        binding = bundle.get("temporary_entity_binding")
        if (not isinstance(binding, Mapping) or
                set(binding) != {"creator_action_id", "consumer_action_id",
                                 "placeholder_ref"} or
                str(binding.get("creator_action_id") or "") not in required_ids or
                str(binding.get("consumer_action_id") or "") not in required_ids or
                not str(binding.get("placeholder_ref") or "").startswith("temp:")):
            errors.append(f"bundle_temporary_binding_invalid:{bundle_id}")
        for raw_action_id in required:
            action_id = str(raw_action_id)
            if action_id not in actions:
                errors.append(f"bundle_definition_unknown_action:{bundle_id}:{action_id}")
                continue
            if action_id in membership and membership[action_id] != bundle_id:
                errors.append(
                    f"bundle_definition_action_in_multiple_bundles:{action_id}")
            membership[action_id] = bundle_id
            writable = (projected.get(action_id) or {}).get(
                "writable_payload_schema")
            if not isinstance(writable, Mapping):
                errors.append(f"bundle_action_writable_schema_missing:{action_id}")
            frozen = set((actions[action_id].get("frozen_payload") or {}).keys())
            properties = set((writable or {}).get("properties") or {})
            if frozen & properties:
                errors.append(f"bundle_frozen_writable_conflict:{action_id}")
    for action_id, row in projected.items():
        policy = row.get("selection_policy")
        required_bundle = row.get("required_bundle_id")
        if policy not in {"single_allowed", "bundle_only"}:
            errors.append(f"planner_selection_policy_invalid:{action_id}")
        if policy == "bundle_only":
            if not required_bundle or required_bundle not in bundle_ids:
                errors.append(f"bundle_only_required_bundle_missing:{action_id}")
            elif membership.get(action_id) != required_bundle:
                errors.append(f"bundle_only_membership_mismatch:{action_id}")
        elif required_bundle is not None:
            errors.append(f"single_allowed_has_required_bundle:{action_id}")
    return {"valid": not errors, "errors": errors,
            "planner_bundle_selection_version": PLANNER_BUNDLE_VERSION}


def effective_action_for_bundle(
    action: Mapping[str, Any], *, bundle_id: str | None,
    bundles: Mapping[str, Any],
) -> dict[str, Any]:
    result = deepcopy(dict(action))
    if bundle_id is None:
        return result
    bundle = next((row for row in bundles.get("bundles") or []
                   if row.get("bundle_id") == bundle_id), None)
    if bundle is None:
        raise ValueError(f"planner_bundle_unknown:{bundle_id}")
    override = (bundle.get("frozen_payload_overrides") or {}).get(
        str(action.get("action_id"))) or {}
    frozen = dict(result.get("frozen_payload") or {})
    frozen.update(deepcopy(dict(override)))
    result["frozen_payload"] = frozen
    return result


def _selection_policies(action_menu: Mapping[str, Any]) -> dict[str, dict[str, str | None]]:
    by_operation = {str(row.get("operation")): row
                    for row in action_menu.get("actions") or []}
    person = by_operation.get("add_person")
    relation = by_operation.get("add_relation")
    if not person or not relation:
        return {}
    return {
        str(person["action_id"]): {
            "selection_policy": "bundle_only", "required_bundle_id": "B0001"},
        str(relation["action_id"]): {
            "selection_policy": "bundle_only", "required_bundle_id": "B0001"},
    }


def validate_writable_content(
    *, action_id: str, content: Any, action_view: Mapping[str, Any],
    scene_index: int,
) -> list[str]:
    action = next((row for row in action_view.get("actions") or []
                   if row.get("action_id") == action_id), None)
    if action is None:
        return [f"unknown_action_id:scene={scene_index}:action={action_id}"]
    if not isinstance(content, Mapping):
        return [f"proposed_content_not_object:scene={scene_index}:action={action_id}"]
    schema = action["writable_payload_schema"]
    properties = schema["properties"]
    errors: list[str] = []
    for key in content:
        if key in FROZEN_PLANNER_FIELDS:
            errors.append(
                f"frozen_field_present:scene={scene_index}:action={action_id}:field={key}")
        elif key not in properties:
            errors.append(
                f"unknown_writable_field:scene={scene_index}:action={action_id}:field={key}")
    for key in schema["required"]:
        if key not in content:
            errors.append(
                f"writable_field_missing:scene={scene_index}:action={action_id}:field={key}")
    for key, value in content.items():
        if key not in properties:
            continue
        definition = properties[key]
        if not _value_matches_schema(value, definition):
            taxonomy = ("invalid_canonical_enum" if "enum" in definition
                        else "writable_field_type")
            errors.append(
                f"{taxonomy}:scene={scene_index}:action={action_id}:field={key}")
    return errors


def validate_compiled_proposal_novelty(
    compiled_plan: Mapping[str, Any], *, authoritative_memory: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    seen: dict[str, str] = {}
    active_refs = set(str(value) for value in
                      (authoritative_memory.get("active_head_index") or {}).values())
    existing_keys: set[str] = set()
    for record in authoritative_memory.get("authoritative_records") or []:
        status = str(record.get("verification_status") or record.get("semantic_status") or "")
        ref = str(record.get("memory_ref") or record.get("memory_id") or "")
        if status not in {"source_grounded", "semantically_verified"}:
            continue
        if active_refs and ref not in active_refs:
            continue
        operation = str(record.get("operation") or "")
        if operation in OPERATION_REGISTRY and OPERATION_REGISTRY[operation]["operation_class"] == "create":
            existing_keys.add(proposal_novelty_key(operation, record.get("payload") or {}))
    for scene in compiled_plan.get("scenes") or []:
        for proposal in scene.get("compiled_proposals") or []:
            ref = str(proposal["proposal_ref"])
            operation = str(proposal["operation"])
            payload = proposal["compiled_payload"]
            key = proposal_novelty_key(operation, payload)
            if key in seen:
                errors.append(
                    f"duplicate_round_proposal:proposal={ref}:first={seen[key]}")
            else:
                seen[key] = ref
            operation_class = str(proposal.get("operation_class") or "")
            if operation_class == "create" and key in existing_keys:
                errors.append(f"existing_fact_collision:proposal={ref}")
            if operation_class in {"update", "transition"}:
                if _normalized(_post_value(operation, payload)) == _normalized(
                        proposal.get("pre_state")):
                    errors.append(f"post_value_unchanged:proposal={ref}")
    return {"novelty_version": PLANNER_NOVELTY_VERSION,
            "valid": not errors, "errors": errors,
            "proposal_count": len(seen), "existing_key_count": len(existing_keys)}


def proposal_novelty_key(operation: str, payload: Mapping[str, Any]) -> str:
    p = deepcopy(dict(payload))
    if operation == "add_event":
        value = [sorted(p.get("participant_refs") or []), _normalized(p.get("description")),
                 _normalized(p.get("narrative_time_expression"))]
    elif operation == "add_information_asymmetry":
        value = [_normalized(p.get("information")), sorted(p.get("knower_refs") or []),
                 sorted(p.get("non_knower_refs") or [])]
    elif operation == "create_state":
        value = [p.get("subject_ref"), _normalized(p.get("state_category")),
                 _normalized(p.get("state_dimension"))]
    elif operation == "add_relation":
        relation_type = str(p.get("relation_type") or "")
        endpoints = [str(p.get("source_ref") or ""), str(p.get("target_ref") or "")]
        if relation_type in RELATION_TYPE_REGISTRY and relation_contract(relation_type)["symmetric"]:
            endpoints.sort()
        value = [relation_type, endpoints]
    elif operation == "open_thread":
        value = [sorted(p.get("subject_refs") or []), _normalized(p.get("description"))]
    elif operation == "add_person":
        value = [_normalized(p.get("display_role")), p.get("scene_presence")]
    elif operation == "add_shared_task":
        value = [sorted(p.get("participant_refs") or []),
                 _normalized(p.get("task_description"))]
    else:
        value = _normalized(p)
    return hashlib.sha256(json.dumps(
        [operation, value], ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def _post_value(operation: str, payload: Mapping[str, Any]) -> Any:
    keys = {
        "update_state": "new_value", "change_trust": "new_value",
        "change_authority": "new_value", "resolve_conflict": "resolution",
        "close_thread": "closure", "supersede": "replacement_payload",
        "add_role_handoff": "handoff_description",
    }
    return payload.get(keys.get(operation, "new_value"))


def _json_schema(kind: Any) -> dict[str, Any]:
    if isinstance(kind, list):
        return {"type": "string", "enum": list(kind)}
    name = str(kind)
    if name == "registered_relation_type":
        return {"type": "string", "enum": sorted(RELATION_TYPE_REGISTRY)}
    if name == "state_category":
        return {"type": "string", "enum": [
            "goal", "belief", "attitude", "commitment", "responsibility",
            "availability", "relationship_state", "emotional_state",
            "knowledge_state", "custom"]}
    if name == "registered_target_type":
        return {"type": "string", "enum": [
            "state", "relation", "event", "thread", "knowledge", "task",
            "responsibility"]}
    if name == "integer":
        return {"type": "integer"}
    if name == "object":
        return {"type": "object"}
    if name == "optional_string":
        return {"type": ["string", "null"]}
    if name.endswith("_list"):
        return {"type": "array", "items": {"type": "string"}, "minItems": 1}
    return {"type": "string"}


def _optional(kind: Any) -> bool:
    return str(kind).startswith("optional_")


def _value_matches_schema(value: Any, schema: Mapping[str, Any]) -> bool:
    allowed = schema.get("type")
    types = allowed if isinstance(allowed, list) else [allowed]
    matches = (("null" in types and value is None) or
               ("string" in types and isinstance(value, str) and bool(value.strip())) or
               ("integer" in types and isinstance(value, int) and not isinstance(value, bool)) or
               ("object" in types and isinstance(value, Mapping)) or
               ("array" in types and isinstance(value, list) and bool(value) and
                all(isinstance(item, str) and item for item in value)))
    if not matches:
        return False
    return "enum" not in schema or value in schema["enum"]


def _normalized(value: Any) -> Any:
    if isinstance(value, str):
        text = unicodedata.normalize("NFKC", value).casefold().strip()
        return re.sub(r"\s+", " ", text)
    if isinstance(value, Mapping):
        return {str(key): _normalized(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalized(item) for item in value]
    return value


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()
