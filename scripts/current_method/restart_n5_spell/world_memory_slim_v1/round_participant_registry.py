"""Deterministic round-local identities for dialogue-first speakers."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from copy import deepcopy
from typing import Any, Mapping, Sequence


ROUND_PARTICIPANT_REGISTRY_VERSION = (
    "round_participant_registry_v3_action_or_speaker_declare_then_reference"
)
EXISTING_SPEAKER_FIELDS = frozenset({"participant_ref", "scene_role", "intents"})
NEW_SPEAKER_DECLARATION_FIELDS = frozenset({
    "participant_alias", "display_name", "scene_role",
    "entry_context", "anchor_connection", "intents",
})
ALIAS_SPEAKER_REFERENCE_FIELDS = frozenset({
    "participant_alias", "scene_role", "intents",
})
# Kept as an import-compatible name for callers that mean a first declaration.
NEW_SPEAKER_FIELDS = NEW_SPEAKER_DECLARATION_FIELDS


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _normalized_name(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _sha(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compile_round_participant_registry(
    *, scenes: Sequence[Mapping[str, Any]],
    authoritative_entities: Sequence[Mapping[str, Any]], round_index: int = 1,
) -> dict[str, Any]:
    """Assign round refs after an action or speaker establishes a narrative alias."""
    authoritative_by_ref = {
        str(row.get("entity_ref") or row.get("entity_id") or ""): row
        for row in authoritative_entities
        if row.get("entity_ref") or row.get("entity_id")
    }
    authoritative_names: dict[str, list[str]] = {}
    for ref, row in authoritative_by_ref.items():
        name = _normalized_name(str(row.get("display_name") or ""))
        if name:
            authoritative_names.setdefault(name, []).append(ref)
    errors: list[str] = []
    warnings: list[str] = []
    aliases: dict[str, dict[str, Any]] = {}
    alias_order: list[str] = []
    next_participant = 1
    scene_bindings: dict[str, list[dict[str, Any]]] = {}
    available_aliases_by_scene: dict[str, list[str]] = {}

    def register_identity(
        *, alias: str, display: str, scene_alias: str, prefix: str,
        declaration_source: str, entry_context: str,
        anchor_connection: str, action_ref: str | None,
    ) -> dict[str, Any] | None:
        nonlocal next_participant
        if re.fullmatch(r"[a-z][a-z0-9_]{0,31}", alias) is None:
            errors.append(f"{prefix}:participant_alias_invalid:{alias}")
            return None
        if not _text(display):
            errors.append(f"{prefix}:display_name_invalid")
            return None
        identity = aliases.get(alias)
        if identity is not None:
            if _normalized_name(str(identity["display_name"])) != _normalized_name(display):
                errors.append(
                    f"{prefix}:participant_alias_identity_conflict:{alias}:"
                    f"{identity['display_name']}:{display}")
                return None
            return identity
        ref = f"R{round_index:02d}:P{next_participant:02d}"
        next_participant += 1
        identity = {
            "participant_alias": alias,
            "participant_ref": ref,
            "display_name": display,
            "entry_context": entry_context,
            "anchor_connection": anchor_connection,
            "declaration_source": declaration_source,
            "declaration_scene_alias": scene_alias,
            "declaration_action_ref": action_ref,
            "speaker_declaration_scene_alias": (
                scene_alias if declaration_source == "speaker" else None),
            "first_scene_alias": scene_alias,
            "appearance_scene_aliases": [],
            "authoritative_entity_ref": None,
            "is_authoritative_memory_entity": False,
        }
        aliases[alias] = identity
        alias_order.append(alias)
        name = _normalized_name(display)
        if name in authoritative_names:
            warnings.append(
                f"{prefix}:display_name_collision_not_merged:{alias}:" +
                ",".join(sorted(authoritative_names[name])))
        return identity

    for scene_index, scene in enumerate(scenes, 1):
        scene_alias = str(scene.get("scene_alias") or f"s{scene_index}")
        bindings: list[dict[str, Any]] = []
        seen_refs: set[str] = set()
        for slot in ("primary_action", "secondary_action"):
            action = scene.get(slot)
            if (not isinstance(action, Mapping) or
                    action.get("action_mode") != "new" or
                    action.get("action_type") != "introduce_connected_person"):
                continue
            arguments = action.get("arguments")
            if not isinstance(arguments, Mapping):
                continue
            alias = str(arguments.get("participant_alias") or "")
            prefix = f"scene:{scene_index}:action:{slot}"
            existing = aliases.get(alias)
            identity = register_identity(
                alias=alias,
                display=str(arguments.get("display_name") or "").strip(),
                scene_alias=scene_alias,
                prefix=prefix,
                declaration_source="introduce_connected_person_action",
                entry_context=str(scene.get("new_development") or
                                  arguments.get("connection_description") or ""),
                anchor_connection=str(arguments.get("connection_description") or ""),
                action_ref=f"S{scene_index:02d}.{slot}",
            )
            if (existing is not None and identity is existing and
                    existing.get("declaration_action_ref") is not None):
                errors.append(f"{prefix}:participant_alias_action_redeclared:{alias}")
            elif identity is not None and identity.get("declaration_action_ref") is None:
                identity["declaration_action_ref"] = f"S{scene_index:02d}.{slot}"
        for speaker_index, raw in enumerate(scene.get("speaker_participants") or [], 1):
            prefix = f"scene:{scene_index}:speaker:{speaker_index}"
            if not isinstance(raw, Mapping):
                errors.append(f"{prefix}:not_object")
                continue
            fields = set(raw)
            if fields == EXISTING_SPEAKER_FIELDS:
                ref = str(raw.get("participant_ref") or "")
                if ref not in authoritative_by_ref:
                    errors.append(f"{prefix}:existing_ref_unknown:{ref}")
                binding = {
                    "participant_ref": ref,
                    "display_name": str(
                        (authoritative_by_ref.get(ref) or {}).get("display_name") or ref),
                    "scene_role": raw.get("scene_role"),
                    "intents": deepcopy(raw.get("intents")),
                    "entry_context": None,
                    "anchor_connection": None,
                    "participant_alias": None,
                }
            elif fields == NEW_SPEAKER_DECLARATION_FIELDS:
                alias = str(raw.get("participant_alias") or "")
                display = str(raw.get("display_name") or "").strip()
                for key in ("display_name", "entry_context", "anchor_connection"):
                    if not _text(raw.get(key)):
                        errors.append(f"{prefix}:{key}_invalid")
                existing = aliases.get(alias)
                identity = register_identity(
                    alias=alias, display=display, scene_alias=scene_alias,
                    prefix=prefix, declaration_source="speaker",
                    entry_context=str(raw.get("entry_context") or ""),
                    anchor_connection=str(raw.get("anchor_connection") or ""),
                    action_ref=None)
                if identity is None:
                    continue
                ref = str(identity["participant_ref"])
                if existing is not None and existing.get(
                        "speaker_declaration_scene_alias") is not None:
                    errors.append(f"{prefix}:participant_alias_redeclared:{alias}")
                elif existing is not None:
                    identity["speaker_declaration_scene_alias"] = scene_alias
                identity["appearance_scene_aliases"].append(scene_alias)
                binding = {
                    "participant_ref": ref,
                    "display_name": display,
                    "scene_role": raw.get("scene_role"),
                    "intents": deepcopy(raw.get("intents")),
                    "entry_context": raw.get("entry_context"),
                    "anchor_connection": raw.get("anchor_connection"),
                    "participant_alias": alias,
                }
            elif fields == ALIAS_SPEAKER_REFERENCE_FIELDS:
                alias = str(raw.get("participant_alias") or "")
                identity = aliases.get(alias)
                if identity is None:
                    errors.append(f"{prefix}:participant_alias_not_declared:{alias}")
                    continue
                identity["appearance_scene_aliases"].append(scene_alias)
                binding = {
                    "participant_ref": str(identity["participant_ref"]),
                    "display_name": str(identity["display_name"]),
                    "scene_role": raw.get("scene_role"),
                    "intents": deepcopy(raw.get("intents")),
                    "entry_context": identity.get("entry_context"),
                    "anchor_connection": identity.get("anchor_connection"),
                    "participant_alias": alias,
                }
            else:
                errors.append(f"{prefix}:speaker_schema")
                continue
            ref = str(binding.get("participant_ref") or "")
            if ref in seen_refs:
                errors.append(f"{prefix}:participant_duplicate:{ref}")
            seen_refs.add(ref)
            if binding.get("scene_role") not in {"primary", "supporting"}:
                errors.append(f"{prefix}:scene_role_invalid")
            intents = binding.get("intents")
            if (not isinstance(intents, list) or not 1 <= len(intents) <= 3 or
                    not all(_text(value) for value in intents)):
                errors.append(f"{prefix}:intents_invalid")
            bindings.append(binding)
        scene_bindings[scene_alias] = bindings
        available_aliases_by_scene[scene_alias] = list(alias_order)

    participants = [deepcopy(aliases[alias]) for alias in alias_order]
    result = {
        "round_participant_registry_version": ROUND_PARTICIPANT_REGISTRY_VERSION,
        "round_index": round_index,
        "participants": participants,
        "scene_bindings": scene_bindings,
        "available_aliases_by_scene": available_aliases_by_scene,
        "alias_to_round_ref": {
            alias: aliases[alias]["participant_ref"] for alias in alias_order},
        "authoritative_entity_refs": sorted(authoritative_by_ref),
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
    }
    result["valid"] = not result["errors"]
    result["round_participant_registry_sha256"] = _sha({
        key: value for key, value in result.items()
        if key != "round_participant_registry_sha256"})
    return result


def legacy_scene_speaker_bindings(
    registry: Mapping[str, Any], scene_alias: str,
) -> list[dict[str, Any]]:
    """Project round-local identities into the existing downstream speaker shape."""
    rows = (registry.get("scene_bindings") or {}).get(scene_alias)
    if not isinstance(rows, list):
        raise ValueError(f"round_participant_scene_unknown:{scene_alias}")
    return [{
        "participant_ref": str(row["participant_ref"]),
        "scene_role": row["scene_role"],
        "intents": deepcopy(row["intents"]),
    } for row in rows]


def validate_round_participant_registry(
    registry: Mapping[str, Any], *, scenes: Sequence[Mapping[str, Any]],
    authoritative_entities: Sequence[Mapping[str, Any]], round_index: int = 1,
) -> dict[str, Any]:
    expected = compile_round_participant_registry(
        scenes=scenes, authoritative_entities=authoritative_entities,
        round_index=round_index)
    errors = list(expected["errors"])
    for field in ("participants", "scene_bindings", "available_aliases_by_scene",
                  "alias_to_round_ref"):
        if registry.get(field) != expected.get(field):
            errors.append(f"round_participant_registry_{field}_mismatch")
    return {
        "valid": not errors,
        "errors": list(dict.fromkeys(errors)),
        "warnings": expected["warnings"],
        "round_participant_registry_version": ROUND_PARTICIPANT_REGISTRY_VERSION,
    }
