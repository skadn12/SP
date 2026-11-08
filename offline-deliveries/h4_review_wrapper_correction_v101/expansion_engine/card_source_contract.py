"""Bind plan-review source references to a fixed projection of the actual card.

Hashes and paths establish which supplied material was referenced. They do not
certify that a chosen path supports a semantic judgment or that all material
requirements were considered. No history or proposed-plan value enters this
projection.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any


VERSION = "v100_bound_card_source_v1"
PROJECTION_FIELDS = ("card_id", "card_sha256", "guidance_level", "actors", "steps", "dependencies")
BINDING_FIELDS = {"card_id", "card_sha256", "projection_sha256"}
SUBSTANTIVE_FIELDS = {
    "actors": ("role",),
    "steps": ("action", "state"),
    "dependencies": ("relation",),
}
GUIDANCE_ENTRY_FIELDS = {
    "actors": ("actor_id", "role"),
    "steps": ("step_id", "actor_id", "action", "state"),
    "dependencies": ("before", "after", "relation"),
}


def canonical_bytes(value: Any) -> bytes:
    """Canonicalization is fixed here, including preservation of array order."""
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def projection(card: dict[str, Any]) -> dict[str, Any]:
    """Copy only the assigned card's identity and explicit structural guidance."""
    if not isinstance(card, dict):
        raise TypeError("bound_card_source_requires_card_object")
    source = {name: deepcopy(card.get(name))
              for name in ("card_id", "card_sha256", "guidance_level")}
    # Match the pre-existing card-guidance boundary at every nesting level.
    # A raw extraction can retain source quotes or evidence beside an action;
    # those are neither guidance nor historical evidence for this root.
    for name, fields in GUIDANCE_ENTRY_FIELDS.items():
        entries = card.get(name)
        source[name] = [
            {key: deepcopy(entry[key]) for key in fields if key in entry}
            for entry in entries if isinstance(entry, dict)
        ] if isinstance(entries, list) else []
    return source


def build(card: dict[str, Any]) -> dict[str, Any]:
    """Enumerate substantive literal leaves without classifying their meaning."""
    source = projection(card)
    leaves = []
    for collection, names in SUBSTANTIVE_FIELDS.items():
        entries = source.get(collection)
        for index, entry in enumerate(entries if isinstance(entries, list) else []):
            if not isinstance(entry, dict):
                continue
            for name in names:
                value = entry.get(name)
                if isinstance(value, str) and value.strip():
                    leaves.append({"source_path": f"/{collection}/{index}/{name}",
                                   "value": deepcopy(value)})
    binding = {
        "card_id": deepcopy(source["card_id"]),
        "card_sha256": deepcopy(source["card_sha256"]),
        "projection_sha256": hashlib.sha256(canonical_bytes(source)).hexdigest(),
    }
    return {
        "version": VERSION,
        "source_binding": binding,
        "source_projection": source,
        "allowed_source_paths": [leaf["source_path"] for leaf in leaves],
        "source_leaves": leaves,
        "path_semantics": "JSON Pointer into source_projection; zero-based exact literal array positions only",
        "empty_source_paths": "Allowed when this card does not explicitly specify that dimension; explain in function_check.reason",
        "scope": "source identity and literal leaf reference only; not semantic support, completeness, or certification",
    }


def collect_binding_errors(value: Any, card: dict[str, Any], *, path: str) -> list[dict[str, Any]]:
    expected = build(card)["source_binding"]
    errors = []
    if not isinstance(value, dict) or set(value) != BINDING_FIELDS:
        errors.append({"code": "card_source_binding_fields_invalid", "path": path,
                       "actual": deepcopy(value), "allowed": deepcopy(expected)})
    if isinstance(value, dict):
        for name in sorted(BINDING_FIELDS):
            if value.get(name) != expected[name]:
                errors.append({"code": "card_source_binding_mismatch", "path": f"{path}.{name}",
                               "actual": deepcopy(value.get(name)), "allowed": [deepcopy(expected[name])]})
    return errors


def collect_path_errors(value: Any, card: dict[str, Any], *, path: str) -> list[dict[str, Any]]:
    """Reject metadata, containers, wildcards, negative indices and foreign refs.

    Membership in the program's leaf inventory is the whole reference rule.
    The program does not infer relevance from words in a path or leaf value.
    """
    allowed = build(card)["allowed_source_paths"]
    if not isinstance(value, list):
        return [{"code": "card_source_paths_not_array", "path": path,
                 "actual": deepcopy(value), "allowed": {"type": "distinct array", "items": allowed,
                                                            "empty_allowed": True}}]
    errors, seen = [], set()
    for index, pointer in enumerate(value):
        at = f"{path}[{index}]"
        if not isinstance(pointer, str) or pointer not in allowed:
            errors.append({"code": "card_source_path_not_allowed", "path": at,
                           "actual": deepcopy(pointer), "allowed": deepcopy(allowed)})
        elif pointer in seen:
            errors.append({"code": "card_source_path_duplicate", "path": at,
                           "actual": pointer, "allowed": "distinct exact source paths"})
        else:
            seen.add(pointer)
    return errors
