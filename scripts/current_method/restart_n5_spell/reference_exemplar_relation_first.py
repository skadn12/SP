"""Alias-only exemplar-guided relation contracts."""

from __future__ import annotations

import inspect
import json
from typing import Any, Mapping, Sequence

from .core import canonical_json, sha256_bytes


CANARY_VERSION = "full_dialogue_unrestricted_exemplar_guided_relation_first_canary_v1"
RELATION_PROMPT_VERSION = (
    "spell_full_dialogue_unrestricted_exemplar_relation_miner_4x_alias_only_v1"
)
RELATION_SCHEMA_VERSION = "unrestricted_exemplar_relation_batch_4x_alias_only_schema_v1"
RELATION_SLOT_COUNT = 4
VAGUE_CONCLUSION_PHRASES = (
    "aligns with", "provide a foundation", "provides a foundation",
    "both emphasize", "are aligned", "is aligned with",
)

VALID_FIELDS = (
    "relation_id", "status", "reasoning_type", "parent_turn_aliases",
    "child_turn_aliases", "parent_unique_fact", "child_unique_fact",
    "combined_conclusion", "missing_from_parent_only", "missing_from_child_only",
)
EMPTY_FIELDS = ("relation_id", "status", "reason")

RELATION_BATCH_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["relations"],
    "properties": {"relations": {"type": "array", "minItems": 4, "maxItems": 4,
        "items": {"oneOf": [
            {"type": "object", "additionalProperties": False,
             "required": list(VALID_FIELDS), "properties": {
                 "relation_id": {"type": "string"}, "status": {"const": "valid"},
                 "reasoning_type": {"type": "string", "minLength": 1},
                 "parent_turn_aliases": {"type": "array", "minItems": 1,
                    "maxItems": 3, "uniqueItems": True,
                    "items": {"type": "string", "minLength": 1}},
                 "child_turn_aliases": {"type": "array", "minItems": 1,
                    "maxItems": 4, "uniqueItems": True,
                    "items": {"type": "string", "minLength": 1}},
                 **{field: {"type": "string", "minLength": 1} for field in (
                     "parent_unique_fact", "child_unique_fact", "combined_conclusion",
                     "missing_from_parent_only", "missing_from_child_only")}}},
            {"type": "object", "additionalProperties": False,
             "required": list(EMPTY_FIELDS), "properties": {
                 "relation_id": {"type": "string"},
                 "status": {"const": "no_valid_relation"},
                 "reason": {"type": "string", "minLength": 1}}},
        ]}}},
}


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def relation_schema_sha256() -> str:
    return sha256_bytes(canonical_json(RELATION_BATCH_SCHEMA).encode())


def build_relation_miner_prompt(
    aliased_turns: Sequence[Mapping[str, Any]], exemplar_reference: str,
) -> str:
    if len(aliased_turns) != 371:
        raise ValueError("full 371-turn dialogue required")
    dialogue = "\n".join(
        f"[{turn['alias']}] partition={turn['source_partition']} | "
        f"session={turn['session_id']} | speaker={turn['speaker']} | {turn['text']}"
        for turn in aliased_turns
    )
    rules = [
        "Return exactly four relation slots R1, R2, R3, R4 in order.",
        "Use status valid or no_valid_relation. Prefer no_valid_relation over a weak or repeated relation.",
        "Search freely across all P001-P237 and C001-C134 turns; no E1/E4 evidence pool restriction applies.",
        "Return aliases only. Do not copy, quote, paraphrase, or summarize dialogue text as an evidence quote; the local program retrieves frozen text by alias.",
        "Each valid relation must use at least one parent alias and one child alias, with at most three parent and four child aliases.",
        "The complete concrete conclusion must be unavailable from parent alone and unavailable from child alone, but uniquely supported when combined.",
        "Reject child-side restatements, single-turn conclusions, repeated evidence, topic-only matches, and unsupported causal joins.",
        "Do not use vague conclusions such as 'aligns with', 'provides a foundation', or 'both emphasize'. State the exact person, requirement, decision, verification, state change, consequence, or selected alternative.",
        "Do not reuse the same parent+child evidence bundle in another slot, even with a different reasoning_type.",
        "Do not generate questions, answers, options, distractors, quotes, or extra fields. Return JSON only.",
    ]
    return "\n\n".join([
        f"PROMPT_VERSION: {RELATION_PROMPT_VERSION}",
        "RELATION MINER RULES\n- " + "\n- ".join(rules),
        "COMPLETE POSITIVE AND NEGATIVE RELATION EXEMPLARS\n" + exemplar_reference,
        "FULL FROZEN DIALOGUE WITH STABLE ALIASES\n" + dialogue,
        "OUTPUT JSON SCHEMA\n" + json.dumps(
            RELATION_BATCH_SCHEMA, ensure_ascii=False, sort_keys=True,
            separators=(",", ":")),
    ])


def relation_prompt_sha256() -> str:
    material = (RELATION_PROMPT_VERSION + RELATION_SCHEMA_VERSION
        + inspect.getsource(build_relation_miner_prompt)
        + canonical_json(RELATION_BATCH_SCHEMA))
    return sha256_bytes(material.encode())


def validate_relation_batch_schema(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {"relations"}:
        return {"decision": "relation_batch_schema_invalid", "reasons": ["root"]}
    items = raw.get("relations")
    if not isinstance(items, list) or len(items) != RELATION_SLOT_COUNT:
        return {"decision": "relation_batch_schema_invalid", "reasons": ["slot_count"]}
    errors = []
    for index, item in enumerate(items, 1):
        relation_id = f"R{index}"
        if not isinstance(item, dict) or item.get("relation_id") != relation_id:
            errors.append(f"{relation_id}:id")
            continue
        if item.get("status") == "valid":
            if set(item) != set(VALID_FIELDS):
                errors.append(f"{relation_id}:fields")
                continue
            for field in ("reasoning_type", "parent_unique_fact", "child_unique_fact",
                          "combined_conclusion", "missing_from_parent_only",
                          "missing_from_child_only"):
                if not _nonempty(item.get(field)):
                    errors.append(f"{relation_id}:{field}")
            for field, maximum in (("parent_turn_aliases", 3),
                                   ("child_turn_aliases", 4)):
                values = item.get(field)
                if (not isinstance(values, list) or not 1 <= len(values) <= maximum
                        or any(not _nonempty(value) for value in values)
                        or len(set(values)) != len(values)):
                    errors.append(f"{relation_id}:{field}")
        elif item.get("status") == "no_valid_relation":
            if set(item) != set(EMPTY_FIELDS) or not _nonempty(item.get("reason")):
                errors.append(f"{relation_id}:empty_shape")
        else:
            errors.append(f"{relation_id}:status")
    if errors:
        return {"decision": "relation_batch_schema_invalid", "reasons": errors}
    return {"decision": "relation_batch_accept", "reasons": [], "raw_batch": raw}


def normalize_relation_batch(
    raw: Mapping[str, Any], aliases: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if validate_relation_batch_schema(raw)["decision"] != "relation_batch_accept":
        raise ValueError("cannot normalize invalid batch")
    rows = []
    seen: set[tuple[str, ...]] = set()
    for item in raw["relations"]:
        row = dict(item)
        if item["status"] == "no_valid_relation":
            row["local_validation"] = "accept_empty"
            rows.append(row)
            continue
        errors = []
        parent, child = [], []
        for partition, field, target in (
            ("parent", "parent_turn_aliases", parent),
            ("child", "child_turn_aliases", child),
        ):
            for alias in item[field]:
                turn = aliases.get(alias)
                if turn is None:
                    errors.append(f"unknown_alias:{alias}")
                elif turn.get("source_partition") != partition:
                    errors.append(f"wrong_partition:{alias}")
                else:
                    target.append(dict(turn))
        support = tuple(sorted(item["parent_turn_aliases"] + item["child_turn_aliases"]))
        if support in seen:
            errors.append("duplicate_relation_support")
        conclusion_folded = item["combined_conclusion"].casefold()
        if any(phrase in conclusion_folded for phrase in VAGUE_CONCLUSION_PHRASES):
            errors.append("vague_combined_conclusion")
        if not errors:
            seen.add(support)
        row.update({
            "parent_quotes": [turn["text"] for turn in parent],
            "child_quotes": [turn["text"] for turn in child],
            "resolved_parent_turns": parent, "resolved_child_turns": child,
            "canonical_support_aliases": list(support),
            "local_validation": "accept" if not errors else "reject",
            "local_validation_errors": errors,
        })
        rows.append(row)
    return {"schema_version": "normalized_exemplar_alias_only_relations_v1",
        "relations": rows,
        "valid_relation_count": sum(row.get("local_validation") == "accept" for row in rows),
        "empty_relation_count": sum(row.get("status") == "no_valid_relation" for row in rows)}
