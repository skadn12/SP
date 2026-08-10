"""Contracts for the unrestricted reference-guided relation-first canary."""

from __future__ import annotations

import inspect
import json
from typing import Any, Mapping, Sequence

from .core import canonical_json, sha256_bytes


CANARY_VERSION = "full_dialogue_unrestricted_reference_guided_relation_first_canary_v1"
RELATION_PROMPT_VERSION = (
    "spell_full_dialogue_unrestricted_reference_guided_relation_miner_8x_v1"
)
RELATION_SCHEMA_VERSION = "unrestricted_reference_relation_batch_8x_schema_v1"
RELATION_AUDIT_VERSION = "offline_joint_relation_audit_v1"
QA_PROMPT_VERSION = "spell_frozen_relation_qa_writer_single_v1"
QA_SCHEMA_VERSION = "frozen_relation_qa_writer_schema_v1"
QA_AUDIT_VERSION = "offline_joint_qa_preservation_audit_v1"
RELATION_SLOT_COUNT = 8
MAX_QA_RELATIONS = 4


VALID_RELATION_FIELDS = (
    "relation_id", "status", "reasoning_type", "parent_turn_aliases",
    "parent_quotes", "child_turn_aliases", "child_quotes",
    "parent_unique_fact", "child_unique_fact", "combined_conclusion",
    "missing_from_parent_only", "missing_from_child_only",
)
EMPTY_RELATION_FIELDS = ("relation_id", "status", "reason")


RELATION_BATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["relations"],
    "properties": {
        "relations": {
            "type": "array",
            "minItems": RELATION_SLOT_COUNT,
            "maxItems": RELATION_SLOT_COUNT,
            "items": {
                "oneOf": [
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": list(VALID_RELATION_FIELDS),
                        "properties": {
                            "relation_id": {"type": "string"},
                            "status": {"const": "valid"},
                            "reasoning_type": {"type": "string", "minLength": 1},
                            "parent_turn_aliases": {
                                "type": "array", "minItems": 1, "maxItems": 3,
                                "uniqueItems": True,
                                "items": {"type": "string", "minLength": 1},
                            },
                            "parent_quotes": {
                                "type": "array", "minItems": 1, "maxItems": 3,
                                "items": {"type": "string", "minLength": 1},
                            },
                            "child_turn_aliases": {
                                "type": "array", "minItems": 1, "maxItems": 4,
                                "uniqueItems": True,
                                "items": {"type": "string", "minLength": 1},
                            },
                            "child_quotes": {
                                "type": "array", "minItems": 1, "maxItems": 4,
                                "items": {"type": "string", "minLength": 1},
                            },
                            **{
                                field: {"type": "string", "minLength": 1}
                                for field in (
                                    "parent_unique_fact", "child_unique_fact",
                                    "combined_conclusion", "missing_from_parent_only",
                                    "missing_from_child_only",
                                )
                            },
                        },
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": list(EMPTY_RELATION_FIELDS),
                        "properties": {
                            "relation_id": {"type": "string"},
                            "status": {"const": "no_valid_relation"},
                            "reason": {"type": "string", "minLength": 1},
                        },
                    },
                ]
            },
        }
    },
}


QA_WRITER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "relation_id", "fixed_parent_turn_aliases", "fixed_child_turn_aliases",
        "question", "answer", "relation_preserved",
    ],
    "properties": {
        "relation_id": {"type": "string", "minLength": 1},
        "fixed_parent_turn_aliases": {
            "type": "array", "minItems": 1, "maxItems": 3,
            "uniqueItems": True, "items": {"type": "string", "minLength": 1},
        },
        "fixed_child_turn_aliases": {
            "type": "array", "minItems": 1, "maxItems": 4,
            "uniqueItems": True, "items": {"type": "string", "minLength": 1},
        },
        "question": {"type": "string", "minLength": 1},
        "answer": {"type": "string", "minLength": 1},
        "relation_preserved": {"const": True},
    },
}


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def relation_schema_sha256() -> str:
    return sha256_bytes(canonical_json(RELATION_BATCH_SCHEMA).encode("utf-8"))


def qa_schema_sha256() -> str:
    return sha256_bytes(canonical_json(QA_WRITER_SCHEMA).encode("utf-8"))


def extract_relation_reference(reference_text: str) -> str:
    """Select only relation discovery/ablation principles from the fixed reference."""

    def section(start: str, end: str) -> str:
        if start not in reference_text or end not in reference_text:
            raise ValueError(f"reference section marker missing: {start!r}/{end!r}")
        return reference_text.split(start, 1)[1].split(end, 1)[0].strip()

    relation_definition = section(
        "## 1. What counts as a genuine joint question?",
        "## 2. Recommended generation process",
    )
    complementary = section("### Step 1: Find complementary evidence", "### Step 2:")
    ablation = section("### Step 3: Perform single-side ablation", "### Step 4:")
    shortcuts = section("### Step 5: Check for shortcuts", "---\n\n## 3.")
    diversity = section("## 5. Diversity requirements", "## 6. Output requirements")
    result = "\n\n".join([
        "RELATION DEFINITION\n" + relation_definition,
        "COMPLEMENTARY EVIDENCE\n" + complementary,
        "SINGLE-SIDE ABLATION\n" + ablation,
        "SHORTCUT AND FORCED-JOIN CHECKS\n" + shortcuts,
        "RELATION-TYPE GUIDANCE\n" + diversity,
    ])
    forbidden = (
        "### Step 4: Write the question", "## 3. Positive examples",
        "## 6. Output requirements", '"question":', '"answer":',
    )
    if any(marker in result for marker in forbidden):
        raise ValueError("question-writing material leaked into relation reference view")
    return result


def build_relation_miner_prompt(
    aliased_turns: Sequence[Mapping[str, Any]], relation_reference: str,
) -> str:
    if len(aliased_turns) != 371:
        raise ValueError("relation miner requires all 371 frozen turns")
    parent = sum(turn.get("source_partition") == "parent" for turn in aliased_turns)
    child = len(aliased_turns) - parent
    if (parent, child) != (237, 134):
        raise ValueError("relation miner parent/child counts are invalid")
    dialogue = "\n".join(
        f"[{turn['alias']}] partition={turn['source_partition']} | "
        f"session={turn['session_id']} | speaker={turn['speaker']} | {turn['text']}"
        for turn in aliased_turns
    )
    rules = [
        "Return exactly one JSON object with exactly eight relation slots R1 through R8 in order.",
        "For each slot, use status valid or no_valid_relation; do not manufacture a relation to fill a slot.",
        "Search freely across every listed parent and child turn; there is no E1/E4 or QA-specific evidence restriction.",
        "A valid relation must cite at least one P alias and one C alias, with no more than three P aliases and four C aliases.",
        "Copy every selected turn's complete text exactly into the parallel quote array; do not paraphrase or extract a substring.",
        "The combined conclusion must be underdetermined from parent alone and underdetermined from child alone, but supported when both sides are combined.",
        "Reject a relation if child repeats or summarizes the parent fact, one turn gives the conclusion, the evidence is merely topic-similar, or the conclusion adds an unsupported causal link.",
        "Do not generate a question, answer, option, distractor, answer letter, or field outside the schema.",
        "Use dialogue text as the only factual source. The reference contains principles only and no target-dialogue facts.",
        "Keep explanations concise so all eight slots fit in the response. Return JSON only.",
    ]
    return "\n\n".join([
        f"PROMPT_VERSION: {RELATION_PROMPT_VERSION}",
        "TASK: full-dialogue unrestricted reference-guided relation mining",
        "RULES\n- " + "\n- ".join(rules),
        "FIXED RELATION REFERENCE PRINCIPLES\n" + relation_reference,
        "FULL FROZEN DIALOGUE WITH STABLE ALIASES\n" + dialogue,
        "OUTPUT JSON SCHEMA\n" + json.dumps(
            RELATION_BATCH_SCHEMA, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ),
    ])


def relation_prompt_sha256() -> str:
    material = (
        RELATION_PROMPT_VERSION + RELATION_SCHEMA_VERSION
        + inspect.getsource(build_relation_miner_prompt)
        + canonical_json(RELATION_BATCH_SCHEMA)
    )
    return sha256_bytes(material.encode("utf-8"))


def validate_relation_batch_schema(raw: Any) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(raw, dict) or set(raw) != {"relations"}:
        return {"decision": "relation_batch_schema_invalid", "reasons": ["root_schema"]}
    relations = raw.get("relations")
    if not isinstance(relations, list) or len(relations) != RELATION_SLOT_COUNT:
        return {"decision": "relation_batch_schema_invalid", "reasons": ["slot_count"]}
    for index, item in enumerate(relations, 1):
        label = f"R{index}"
        if not isinstance(item, dict) or item.get("relation_id") != label:
            errors.append(f"{label}:id_or_object")
            continue
        status = item.get("status")
        if status == "valid":
            if set(item) != set(VALID_RELATION_FIELDS):
                errors.append(f"{label}:fields")
                continue
            for field in (
                "reasoning_type", "parent_unique_fact", "child_unique_fact",
                "combined_conclusion", "missing_from_parent_only",
                "missing_from_child_only",
            ):
                if not _nonempty(item.get(field)):
                    errors.append(f"{label}:{field}")
            for alias_field, quote_field, maximum in (
                ("parent_turn_aliases", "parent_quotes", 3),
                ("child_turn_aliases", "child_quotes", 4),
            ):
                aliases, quotes = item.get(alias_field), item.get(quote_field)
                if (
                    not isinstance(aliases, list) or not 1 <= len(aliases) <= maximum
                    or any(not _nonempty(value) for value in aliases)
                    or len(set(aliases)) != len(aliases)
                ):
                    errors.append(f"{label}:{alias_field}")
                if (
                    not isinstance(quotes, list) or not 1 <= len(quotes) <= maximum
                    or any(not _nonempty(value) for value in quotes)
                    or not isinstance(aliases, list) or len(quotes) != len(aliases)
                ):
                    errors.append(f"{label}:{quote_field}")
        elif status == "no_valid_relation":
            if set(item) != set(EMPTY_RELATION_FIELDS) or not _nonempty(item.get("reason")):
                errors.append(f"{label}:empty_shape")
        else:
            errors.append(f"{label}:status")
    if errors:
        return {"decision": "relation_batch_schema_invalid", "reasons": errors}
    return {"decision": "relation_batch_accept", "reasons": [], "raw_batch": raw}


def normalize_relation_batch(
    raw: Mapping[str, Any], alias_lookup: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    schema = validate_relation_batch_schema(raw)
    if schema["decision"] != "relation_batch_accept":
        raise ValueError("cannot normalize an invalid relation batch")
    normalized: list[dict[str, Any]] = []
    seen_supports: set[tuple[str, ...]] = set()
    for item in raw["relations"]:
        row = dict(item)
        if item["status"] == "no_valid_relation":
            row["local_validation"] = "accept_empty"
            normalized.append(row)
            continue
        errors: list[str] = []
        resolved_parent: list[dict[str, Any]] = []
        resolved_child: list[dict[str, Any]] = []
        for partition, alias_field, quote_field, target in (
            ("parent", "parent_turn_aliases", "parent_quotes", resolved_parent),
            ("child", "child_turn_aliases", "child_quotes", resolved_child),
        ):
            for alias, quote in zip(item[alias_field], item[quote_field]):
                turn = alias_lookup.get(alias)
                if turn is None:
                    errors.append(f"unknown_alias:{alias}")
                elif turn.get("source_partition") != partition:
                    errors.append(f"wrong_partition:{alias}")
                elif turn.get("text") != quote:
                    errors.append(f"quote_mismatch:{alias}")
                else:
                    target.append(dict(turn))
        support = tuple(sorted(item["parent_turn_aliases"] + item["child_turn_aliases"]))
        if support in seen_supports:
            errors.append("duplicate_relation_support")
        if not errors:
            seen_supports.add(support)
        row["resolved_parent_turns"] = resolved_parent
        row["resolved_child_turns"] = resolved_child
        row["canonical_support_aliases"] = list(support)
        row["local_validation"] = "accept" if not errors else "reject"
        row["local_validation_errors"] = errors
        normalized.append(row)
    return {
        "schema_version": "normalized_unrestricted_relation_batch_v1",
        "relations": normalized,
        "valid_relation_count": sum(
            row.get("local_validation") == "accept" for row in normalized
        ),
        "empty_relation_count": sum(
            row.get("status") == "no_valid_relation" for row in normalized
        ),
    }


STAGE1_AUDIT_BOOLEAN_FIELDS = (
    "parent_only_can_derive_combined_conclusion",
    "child_only_can_derive_combined_conclusion",
    "combined_can_derive_combined_conclusion",
    "child_fully_restates_parent",
    "evidence_redundant",
    "natural_relation",
    "unsupported_causal_claim",
)


def derive_relation_audit_decision(item: Mapping[str, Any]) -> str:
    if item["parent_only_can_derive_combined_conclusion"]:
        return "parent_only_sufficient"
    if item["child_only_can_derive_combined_conclusion"]:
        return "child_only_sufficient"
    if not item["combined_can_derive_combined_conclusion"]:
        return "combined_not_supported"
    if item["child_fully_restates_parent"] or item["evidence_redundant"]:
        return "repeated_or_redundant_evidence"
    if not item["natural_relation"]:
        return "unnatural_relation"
    if item["unsupported_causal_claim"]:
        return "unsupported_causal_claim"
    return "lineage_jointly_required"


def validate_stage1_audit(
    audit: Any, normalized: Mapping[str, Any], normalized_sha256: str,
) -> dict[str, Any]:
    accepted = {
        item["relation_id"]: item for item in normalized["relations"]
        if item.get("local_validation") == "accept"
    }
    if (
        not isinstance(audit, dict)
        or audit.get("audit_version") != RELATION_AUDIT_VERSION
        or audit.get("source_normalized_relations_sha256") != normalized_sha256
        or not isinstance(audit.get("items"), list)
    ):
        raise ValueError("stage1 audit wrapper invalid")
    if [item.get("relation_id") for item in audit["items"]] != list(accepted):
        raise ValueError("stage1 audit does not cover locally valid relations in order")
    rows = []
    for raw in audit["items"]:
        if set(raw) != {"relation_id", *STAGE1_AUDIT_BOOLEAN_FIELDS, "notes"}:
            raise ValueError(f"stage1 audit fields invalid:{raw.get('relation_id')}")
        if any(not isinstance(raw[field], bool) for field in STAGE1_AUDIT_BOOLEAN_FIELDS):
            raise ValueError(f"stage1 audit boolean invalid:{raw['relation_id']}")
        if not isinstance(raw["notes"], str):
            raise ValueError(f"stage1 audit notes invalid:{raw['relation_id']}")
        decision = derive_relation_audit_decision(raw)
        rows.append({**dict(raw), "derived_decision": decision})
    return {
        "audit_version": RELATION_AUDIT_VERSION,
        "source_normalized_relations_sha256": normalized_sha256,
        "items": rows,
        "stage1_valid_relations": len(rows),
        "stage1_joint_relations": sum(
            row["derived_decision"] == "lineage_jointly_required" for row in rows
        ),
    }


def select_stage2_relations(
    normalized: Mapping[str, Any], audited: Mapping[str, Any], limit: int = MAX_QA_RELATIONS,
) -> list[dict[str, Any]]:
    passed = {
        row["relation_id"] for row in audited["items"]
        if row["derived_decision"] == "lineage_jointly_required"
    }
    return [
        dict(row) for row in normalized["relations"]
        if row.get("local_validation") == "accept" and row["relation_id"] in passed
    ][:limit]


def freeze_relation(row: Mapping[str, Any], stage1_audit_sha256: str) -> dict[str, Any]:
    fields = (
        "relation_id", "reasoning_type", "parent_turn_aliases", "parent_quotes",
        "child_turn_aliases", "child_quotes", "parent_unique_fact",
        "child_unique_fact", "combined_conclusion", "missing_from_parent_only",
        "missing_from_child_only",
    )
    body = {field: row[field] for field in fields}
    body.update({
        "frozen_relation_version": "audited_joint_relation_v1",
        "stage1_audit_sha256": stage1_audit_sha256,
        "stage1_decision": "lineage_jointly_required",
    })
    return body


def build_qa_writer_prompt(frozen_relation: Mapping[str, Any]) -> str:
    evidence = [
        {"alias": alias, "partition": "parent", "exact_text": quote}
        for alias, quote in zip(
            frozen_relation["parent_turn_aliases"], frozen_relation["parent_quotes"]
        )
    ] + [
        {"alias": alias, "partition": "child", "exact_text": quote}
        for alias, quote in zip(
            frozen_relation["child_turn_aliases"], frozen_relation["child_quotes"]
        )
    ]
    payload = {
        "frozen_relation": {
            key: frozen_relation[key] for key in (
                "relation_id", "reasoning_type", "parent_unique_fact",
                "child_unique_fact", "combined_conclusion",
                "missing_from_parent_only", "missing_from_child_only",
                "stage1_decision", "stage1_audit_sha256",
            )
        },
        "frozen_evidence": evidence,
    }
    rules = [
        "Return exactly one JSON object matching the schema.",
        "Write one natural open-ended question and canonical answer from the frozen combined conclusion.",
        "The question must remain underdetermined from parent evidence alone and from child evidence alone.",
        "Do not reveal an intermediate relation in the question wording.",
        "Do not add facts, causality, specificity, or assumptions absent from the frozen relation.",
        "Do not select, add, remove, replace, or reorder aliases; copy both fixed alias arrays exactly.",
        "Set relation_preserved to true only after verifying the question and answer preserve the frozen relation.",
        "Do not generate distractors, options, quotes, reasoning steps, explanations, or extra fields.",
        "Return JSON only without Markdown or surrounding text.",
    ]
    return "\n\n".join([
        f"PROMPT_VERSION: {QA_PROMPT_VERSION}",
        "FROZEN-RELATION QA WRITER RULES\n- " + "\n- ".join(rules),
        "FROZEN RELATION AND EVIDENCE\n" + json.dumps(
            payload, ensure_ascii=False, sort_keys=True, indent=2
        ),
        "OUTPUT JSON SCHEMA\n" + json.dumps(
            QA_WRITER_SCHEMA, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ),
    ])


def qa_prompt_sha256() -> str:
    material = (
        QA_PROMPT_VERSION + QA_SCHEMA_VERSION + inspect.getsource(build_qa_writer_prompt)
        + canonical_json(QA_WRITER_SCHEMA)
    )
    return sha256_bytes(material.encode("utf-8"))


def validate_qa_writer_object(raw: Any, frozen_relation: Mapping[str, Any]) -> dict[str, Any]:
    required = set(QA_WRITER_SCHEMA["required"])
    if not isinstance(raw, dict) or set(raw) != required:
        return {"decision": "qa_schema_invalid", "reasons": ["qa_schema_invalid"]}
    errors: list[str] = []
    if raw.get("relation_id") != frozen_relation["relation_id"]:
        errors.append("qa_relation_drift")
    if raw.get("fixed_parent_turn_aliases") != frozen_relation["parent_turn_aliases"]:
        errors.append("qa_relation_drift")
    if raw.get("fixed_child_turn_aliases") != frozen_relation["child_turn_aliases"]:
        errors.append("qa_relation_drift")
    if not _nonempty(raw.get("question")):
        errors.append("qa_empty_question")
    if not _nonempty(raw.get("answer")):
        errors.append("qa_empty_answer")
    if raw.get("relation_preserved") is not True:
        errors.append("qa_relation_drift")
    if errors:
        return {"decision": errors[0], "reasons": list(dict.fromkeys(errors))}
    obj = {**dict(raw), "qa_object_version": "frozen_relation_qa_object_v1"}
    return {
        "decision": "qa_accept", "reasons": [], "qa_object": obj,
        "qa_object_sha256": sha256_bytes(canonical_json(obj).encode("utf-8")),
    }


QA_AUDIT_BOOLEAN_FIELDS = (
    "parent_only_can_answer_uniquely", "child_only_can_answer_uniquely",
    "combined_can_answer_uniquely", "single_turn_shortcut",
    "question_leaks_intermediate_relation", "answer_supported_by_frozen_relation",
    "relation_preserved_semantically",
)


def derive_qa_audit_decision(item: Mapping[str, Any]) -> str:
    if item["parent_only_can_answer_uniquely"]:
        return "parent_only_sufficient"
    if item["child_only_can_answer_uniquely"]:
        return "child_only_sufficient"
    if not item["combined_can_answer_uniquely"]:
        return "combined_not_answerable"
    if item["single_turn_shortcut"]:
        return "single_turn_shortcut"
    if item["question_leaks_intermediate_relation"]:
        return "question_leaks_intermediate_relation"
    if not item["answer_supported_by_frozen_relation"]:
        return "answer_not_supported"
    if not item["relation_preserved_semantically"]:
        return "qa_relation_not_preserved"
    return "qa_joint_preserved"


def validate_stage2_audit(
    audit: Any, qa_objects: Sequence[Mapping[str, Any]], qa_summary_sha256: str,
) -> dict[str, Any]:
    expected = [item["relation_id"] for item in qa_objects]
    if (
        not isinstance(audit, dict) or audit.get("audit_version") != QA_AUDIT_VERSION
        or audit.get("source_qa_summary_sha256") != qa_summary_sha256
        or not isinstance(audit.get("items"), list)
        or [item.get("relation_id") for item in audit["items"]] != expected
    ):
        raise ValueError("stage2 audit wrapper or coverage invalid")
    rows = []
    for raw in audit["items"]:
        if set(raw) != {"relation_id", *QA_AUDIT_BOOLEAN_FIELDS, "notes"}:
            raise ValueError(f"stage2 audit fields invalid:{raw.get('relation_id')}")
        if any(not isinstance(raw[field], bool) for field in QA_AUDIT_BOOLEAN_FIELDS):
            raise ValueError(f"stage2 audit boolean invalid:{raw['relation_id']}")
        if not isinstance(raw["notes"], str):
            raise ValueError(f"stage2 audit notes invalid:{raw['relation_id']}")
        rows.append({**dict(raw), "derived_decision": derive_qa_audit_decision(raw)})
    return {
        "audit_version": QA_AUDIT_VERSION,
        "source_qa_summary_sha256": qa_summary_sha256,
        "items": rows,
        "qa_writer_inputs": len(rows),
        "qa_joint_preserved": sum(
            row["derived_decision"] == "qa_joint_preserved" for row in rows
        ),
    }


def route_conclusion(stage1_joint: int, qa_inputs: int, qa_joint: int) -> str:
    preservation = (qa_joint / qa_inputs) if qa_inputs else 0.0
    if stage1_joint <= 1:
        return "reference_guided_relation_miner_not_yet_supported"
    if qa_joint >= 2 and preservation >= 0.75:
        return "reference_guided_relation_first_supported_for_distractor_stage"
    return "relation_miner_supported_qa_writer_bottleneck"
