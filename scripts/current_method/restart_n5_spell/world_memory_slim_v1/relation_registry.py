"""Closed relation-type registry and deterministic derived edges."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


RELATION_TYPE_REGISTRY_VERSION = "relation_type_registry_v1"


def _relation(*, directed: bool, inverse: str, identity: bool = True,
              state_updates: bool = False) -> dict[str, Any]:
    return {
        "source_entity_types": ["person"],
        "target_entity_types": ["person"],
        "directed": directed,
        "symmetric": not directed,
        "inverse_relation": inverse,
        "identity_bearing": identity,
        "allows_state_updates": state_updates,
    }


RELATION_TYPE_REGISTRY: dict[str, dict[str, Any]] = {
    "sibling": _relation(directed=False, inverse="sibling"),
    "parent_of": _relation(directed=True, inverse="child_of"),
    "child_of": _relation(directed=True, inverse="parent_of"),
    "spouse": _relation(directed=False, inverse="spouse"),
    "friend": _relation(directed=False, inverse="friend", state_updates=True),
    "coworker": _relation(directed=False, inverse="coworker"),
    "manager_of": _relation(directed=True, inverse="reports_to", state_updates=True),
    "reports_to": _relation(directed=True, inverse="manager_of", state_updates=True),
    "member_of": {
        "source_entity_types": ["person"], "target_entity_types": ["organization"],
        "directed": True, "symmetric": False, "inverse_relation": "has_member",
        "identity_bearing": False, "allows_state_updates": False,
    },
    "collaborates_with": _relation(directed=False, inverse="collaborates_with"),
    "knows": _relation(directed=False, inverse="knows"),
}


def relation_contract(relation_type: str) -> dict[str, Any]:
    if relation_type not in RELATION_TYPE_REGISTRY:
        raise KeyError(f"unknown_relation_type:{relation_type}")
    return deepcopy(RELATION_TYPE_REGISTRY[relation_type])


def derive_relation_edges(proposal_ref: str, payload: Mapping[str, Any]) -> list[dict[str, str]]:
    relation_type = str(payload.get("relation_type") or "")
    contract = relation_contract(relation_type)
    source = str(payload.get("source_ref") or "")
    target = str(payload.get("target_ref") or "")
    if not source or not target or source == target:
        raise ValueError("relation_endpoint_invalid")
    primary = {
        "derived_from_proposal_ref": proposal_ref,
        "source_ref": source, "target_ref": target, "relation_type": relation_type,
    }
    inverse = {
        "derived_from_proposal_ref": proposal_ref,
        "source_ref": target, "target_ref": source,
        "relation_type": str(contract["inverse_relation"]),
    }
    return [primary, inverse]


def validate_relation_registry() -> dict[str, Any]:
    errors: list[str] = []
    required = {"source_entity_types", "target_entity_types", "directed", "symmetric",
                "inverse_relation", "identity_bearing", "allows_state_updates"}
    for name, row in RELATION_TYPE_REGISTRY.items():
        if set(row) != required:
            errors.append(f"relation_contract:{name}")
        inverse = row.get("inverse_relation")
        if inverse != "has_member" and inverse not in RELATION_TYPE_REGISTRY:
            errors.append(f"relation_inverse_missing:{name}")
    return {"valid": not errors, "errors": errors,
            "relation_type_count": len(RELATION_TYPE_REGISTRY)}

