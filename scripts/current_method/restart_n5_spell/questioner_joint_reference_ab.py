"""Deterministic contracts for the Questioner joint-reference A/B canary."""

from __future__ import annotations

import difflib
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core import canonical_json, sha256_bytes


CANARY_VERSION = "questioner_joint_reference_ab_canary_v1"
BASE_PROMPT_VERSION = "questioner_open_joint_candidate_4x_v1"
REFERENCE_VERSION = "questioner_joint_reference_v1"
SCHEMA_VERSION = "questioner_open_joint_candidate_4x_schema_v1"
FORMAT_RETRY_VERSION = "questioner_open_joint_format_only_retry_v1"
BLIND_REVIEW_VERSION = "questioner_joint_reference_blind_review_v1"
BLIND_SEED = 20260718
EXPECTED_PARENT_TURNS = 237
EXPECTED_CHILD_TURNS = 134
EXPECTED_TOTAL_TURNS = 371
SYSTEM_MESSAGE = (
    "You are a careful reading-comprehension Questioner. Follow the user's "
    "output contract exactly and return only one JSON object."
)
REFERENCE_START = "--- BEGIN JOINT-QUESTION REFERENCE ---"
REFERENCE_END = "--- END JOINT-QUESTION REFERENCE ---"


VALID_FIELDS = (
    "candidate_id",
    "status",
    "reasoning_type",
    "parent_turn_aliases",
    "child_turn_aliases",
    "question",
    "answer",
    "parent_contribution",
    "child_contribution",
    "why_parent_only_is_insufficient",
    "why_child_only_is_insufficient",
    "why_combined_is_sufficient",
    "single_turn_shortcut_check",
    "repeated_evidence_check",
)
EMPTY_FIELDS = ("candidate_id", "status", "reason")
VALID_TEXT_FIELDS = tuple(
    field for field in VALID_FIELDS
    if field not in {"candidate_id", "status", "parent_turn_aliases", "child_turn_aliases"}
)


QUESTIONER_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["candidates"],
    "properties": {
        "candidates": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {
                "oneOf": [
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": list(VALID_FIELDS),
                        "properties": {
                            "candidate_id": {"type": "string"},
                            "status": {"const": "valid"},
                            "reasoning_type": {"type": "string", "minLength": 1},
                            "parent_turn_aliases": {
                                "type": "array", "minItems": 1, "maxItems": 3,
                                "uniqueItems": True,
                                "items": {"type": "string", "minLength": 1},
                            },
                            "child_turn_aliases": {
                                "type": "array", "minItems": 1, "maxItems": 4,
                                "uniqueItems": True,
                                "items": {"type": "string", "minLength": 1},
                            },
                            **{
                                field: {"type": "string", "minLength": 1}
                                for field in VALID_TEXT_FIELDS
                            },
                        },
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": list(EMPTY_FIELDS),
                        "properties": {
                            "candidate_id": {"type": "string"},
                            "status": {"const": "no_valid_candidate"},
                            "reason": {"type": "string", "minLength": 1},
                        },
                    },
                ]
            },
        }
    },
}


FORMAT_ONLY_RETRY_SUFFIX = """

FORMAT-ONLY RETRY
Your previous response did not satisfy the required JSON/schema contract.
Regenerate the complete four-slot response from scratch.
Return exactly one valid JSON object with a `candidates` array containing
exactly Q1, Q2, Q3, and Q4 in that order. Use only the two allowed status
shapes. Fully close every array and object. Do not include Markdown, comments,
explanations outside the JSON object, or any continuation of the previous
response. Do not change the semantic task or invent candidates merely to fill
valid slots; `no_valid_candidate` remains allowed.
""".rstrip()


class JointReferenceCanaryError(RuntimeError):
    pass


class PaidRequestBudget:
    def __init__(self, limit: int = 4):
        self.limit = int(limit)
        self.used = 0

    def reserve(self) -> int:
        if self.used >= self.limit:
            raise JointReferenceCanaryError(
                f"paid API request budget exhausted: {self.used}/{self.limit}"
            )
        self.used += 1
        return self.used


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def schema_sha256() -> str:
    return sha256_bytes(canonical_json(QUESTIONER_OUTPUT_SCHEMA).encode("utf-8"))


def format_retry_suffix_sha256() -> str:
    return sha256_bytes(FORMAT_ONLY_RETRY_SUFFIX.encode("utf-8"))


def load_full_dialogue(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("combined_dialogue"), list):
        raise JointReferenceCanaryError("full dialogue input has an invalid wrapper")
    turns = payload["combined_dialogue"]
    if len(turns) != EXPECTED_TOTAL_TURNS:
        raise JointReferenceCanaryError(
            f"full dialogue count mismatch: {len(turns)}/{EXPECTED_TOTAL_TURNS}"
        )
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise JointReferenceCanaryError(f"turn {index} is not an object")
        for field in ("turn_id", "session_id", "speaker", "text"):
            if not _is_nonempty_string(turn.get(field)):
                raise JointReferenceCanaryError(f"turn {index} has invalid {field}")
    parent = [turn for turn in turns if not str(turn["turn_id"]).startswith("C")]
    child = [turn for turn in turns if str(turn["turn_id"]).startswith("C")]
    if (len(parent), len(child)) != (EXPECTED_PARENT_TURNS, EXPECTED_CHILD_TURNS):
        raise JointReferenceCanaryError(
            f"parent/child mismatch: {len(parent)}/{len(child)}"
        )
    if turns[:EXPECTED_PARENT_TURNS] != parent or turns[EXPECTED_PARENT_TURNS:] != child:
        raise JointReferenceCanaryError("parent/child turns are not in the frozen source order")
    if len({str(turn["turn_id"]) for turn in turns}) != len(turns):
        raise JointReferenceCanaryError("duplicate turn_id in full dialogue")
    return payload, [dict(turn) for turn in turns]


def build_turn_aliases(turns: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    parent_index = child_index = 0
    entries: list[dict[str, Any]] = []
    aliased_turns: list[dict[str, Any]] = []
    for position, raw_turn in enumerate(turns):
        turn = dict(raw_turn)
        partition = "child" if str(turn["turn_id"]).startswith("C") else "parent"
        if partition == "parent":
            parent_index += 1
            alias = f"P{parent_index:03d}"
        else:
            child_index += 1
            alias = f"C{child_index:03d}"
        entry = {
            "alias": alias,
            "source_partition": partition,
            "source_position": position,
            "turn_id": turn["turn_id"],
            "session_id": turn["session_id"],
            "speaker": turn["speaker"],
            "text_sha256": hashlib.sha256(turn["text"].encode("utf-8")).hexdigest(),
        }
        entries.append(entry)
        aliased_turns.append({"alias": alias, "source_partition": partition, **turn})
    if (parent_index, child_index) != (EXPECTED_PARENT_TURNS, EXPECTED_CHILD_TURNS):
        raise JointReferenceCanaryError("alias counts do not match the source dialogue")
    alias_map = {
        "schema_version": "questioner_turn_alias_map_v1",
        "parent_turn_count": parent_index,
        "child_turn_count": child_index,
        "total_turn_count": len(entries),
        "entries": entries,
    }
    aliased = {
        "schema_version": "full_dialogue_with_questioner_aliases_v1",
        "parent_turn_count": parent_index,
        "child_turn_count": child_index,
        "total_turn_count": len(entries),
        "combined_dialogue": aliased_turns,
    }
    return alias_map, aliased


def alias_lookup(alias_map: Mapping[str, Any], aliased_dialogue: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    turns = aliased_dialogue.get("combined_dialogue")
    entries = alias_map.get("entries")
    if not isinstance(turns, list) or not isinstance(entries, list) or len(turns) != len(entries):
        raise JointReferenceCanaryError("alias artifacts are inconsistent")
    result: dict[str, dict[str, Any]] = {}
    for entry, turn in zip(entries, turns):
        if entry.get("alias") != turn.get("alias"):
            raise JointReferenceCanaryError("alias artifact order drift")
        result[str(entry["alias"])] = dict(turn)
    if len(result) != EXPECTED_TOTAL_TURNS:
        raise JointReferenceCanaryError("alias lookup does not cover 371 turns")
    return result


def render_dialogue_for_prompt(aliased_dialogue: Mapping[str, Any]) -> str:
    lines = []
    for turn in aliased_dialogue["combined_dialogue"]:
        lines.append(
            f"[{turn['alias']}] session_id={turn['session_id']} | "
            f"speaker={turn['speaker']} | {turn['text']}"
        )
    return "\n".join(lines)


def build_base_prompt(aliased_dialogue: Mapping[str, Any]) -> str:
    schema_text = json.dumps(QUESTIONER_OUTPUT_SCHEMA, ensure_ascii=False, sort_keys=True)
    dialogue_text = render_dialogue_for_prompt(aliased_dialogue)
    return f"""TASK VERSION: {BASE_PROMPT_VERSION}

Read the complete frozen long dialogue below and produce exactly four candidate slots.
For each slot, either produce a valid open-ended parent-child joint question or report
`no_valid_candidate`. Do not generate multiple-choice options or distractors.

A valid candidate must cite at least one P alias and at least one C alias. It may cite
at most three parent aliases and at most four child aliases. Search freely across all
371 turns. Preserve the supplied aliases exactly.

The requested object describes a relation, an open-ended natural English question,
a canonical open-ended answer, the distinct contribution of both evidence sides,
why either side alone is insufficient, why the combination is sufficient, and checks
for single-turn shortcuts and repeated evidence. If no additional relation survives
the two single-side ablations, use `no_valid_candidate`; do not manufacture a relation
just to fill a slot.

Do not use outside knowledge. Do not output options, distractors, graph facts, hidden
gold, or text outside the JSON object. Candidate IDs must be Q1, Q2, Q3, Q4 in order.

FULL FROZEN DIALOGUE WITH STABLE ALIASES
{dialogue_text}

OUTPUT JSON SCHEMA
{schema_text}"""


def reference_block(reference_text: str) -> str:
    return f"\n\n{REFERENCE_START}\n{reference_text.rstrip()}\n{REFERENCE_END}"


def build_group_prompts(
    aliased_dialogue: Mapping[str, Any], reference_text: str
) -> tuple[str, str, dict[str, Any]]:
    group_a = build_base_prompt(aliased_dialogue)
    block = reference_block(reference_text)
    marker = "\n\nFULL FROZEN DIALOGUE WITH STABLE ALIASES\n"
    if marker not in group_a:
        raise JointReferenceCanaryError("base prompt insertion marker missing")
    group_b = group_a.replace(marker, block + marker, 1)
    restored = group_b.replace(block, "", 1)
    proof = {
        "group_b_without_reference_equals_group_a": restored == group_a,
        "reference_occurrence_count_in_group_b": group_b.count(block),
        "group_a_prompt_sha256": sha256_bytes(group_a.encode("utf-8")),
        "group_b_prompt_sha256": sha256_bytes(group_b.encode("utf-8")),
        "reference_block_sha256": sha256_bytes(block.encode("utf-8")),
    }
    if not proof["group_b_without_reference_equals_group_a"] or proof[
        "reference_occurrence_count_in_group_b"
    ] != 1:
        raise JointReferenceCanaryError("A/B prompt difference is not reference-only")
    return group_a, group_b, proof


def prompt_diff(group_a: str, group_b: str) -> str:
    return "".join(
        difflib.unified_diff(
            group_a.splitlines(keepends=True),
            group_b.splitlines(keepends=True),
            fromfile="group_a_prompt",
            tofile="group_b_prompt",
        )
    )


def _validate_alias_list(
    values: Any, *, minimum: int, maximum: int, field: str, errors: list[str]
) -> None:
    if not isinstance(values, list):
        errors.append(f"{field}:not_array")
        return
    if not minimum <= len(values) <= maximum:
        errors.append(f"{field}:count_{len(values)}")
    if any(not _is_nonempty_string(value) for value in values):
        errors.append(f"{field}:invalid_item")
    if len({str(value) for value in values}) != len(values):
        errors.append(f"{field}:duplicate_item")


def validate_response(raw_text: str, aliases: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return {
            "parse_status": "json_invalid",
            "format_errors": [f"json_invalid:{exc.msg}"],
            "alias_errors": [],
            "retry_eligible": True,
            "payload": None,
        }
    format_errors: list[str] = []
    alias_errors: list[str] = []
    if not isinstance(payload, dict):
        format_errors.append("root:not_object")
        candidates: list[Any] = []
    else:
        if set(payload) != {"candidates"}:
            format_errors.append("root:additional_or_missing_fields")
        candidates_value = payload.get("candidates")
        if not isinstance(candidates_value, list):
            format_errors.append("candidates:not_array")
            candidates = []
        else:
            candidates = candidates_value
            if len(candidates) != 4:
                format_errors.append(f"candidates:count_{len(candidates)}")

    for index, candidate in enumerate(candidates):
        label = f"candidate_{index + 1}"
        if not isinstance(candidate, dict):
            format_errors.append(f"{label}:not_object")
            continue
        expected_id = f"Q{index + 1}"
        if candidate.get("candidate_id") != expected_id:
            format_errors.append(f"{label}:candidate_id_not_{expected_id}")
        status = candidate.get("status")
        if status == "valid":
            if set(candidate) != set(VALID_FIELDS):
                format_errors.append(f"{label}:valid_fields_mismatch")
            for field in VALID_TEXT_FIELDS:
                if not _is_nonempty_string(candidate.get(field)):
                    format_errors.append(f"{label}:{field}_invalid")
            _validate_alias_list(
                candidate.get("parent_turn_aliases"), minimum=1, maximum=3,
                field=f"{label}:parent_turn_aliases", errors=format_errors,
            )
            _validate_alias_list(
                candidate.get("child_turn_aliases"), minimum=1, maximum=4,
                field=f"{label}:child_turn_aliases", errors=format_errors,
            )
            if isinstance(candidate.get("parent_turn_aliases"), list):
                for alias in candidate["parent_turn_aliases"]:
                    turn = aliases.get(str(alias))
                    if turn is None:
                        alias_errors.append(f"{label}:unknown_parent_alias:{alias}")
                    elif turn.get("source_partition") != "parent":
                        alias_errors.append(f"{label}:parent_alias_wrong_partition:{alias}")
            if isinstance(candidate.get("child_turn_aliases"), list):
                for alias in candidate["child_turn_aliases"]:
                    turn = aliases.get(str(alias))
                    if turn is None:
                        alias_errors.append(f"{label}:unknown_child_alias:{alias}")
                    elif turn.get("source_partition") != "child":
                        alias_errors.append(f"{label}:child_alias_wrong_partition:{alias}")
        elif status == "no_valid_candidate":
            if set(candidate) != set(EMPTY_FIELDS):
                format_errors.append(f"{label}:empty_fields_mismatch")
            if not _is_nonempty_string(candidate.get("reason")):
                format_errors.append(f"{label}:reason_invalid")
        else:
            format_errors.append(f"{label}:status_invalid")

    parse_status = (
        "schema_invalid" if format_errors else
        "alias_invalid" if alias_errors else
        "valid"
    )
    return {
        "parse_status": parse_status,
        "format_errors": format_errors,
        "alias_errors": alias_errors,
        "retry_eligible": bool(format_errors),
        "payload": payload,
    }


def normalize_group_result(
    group: str,
    validation: Mapping[str, Any],
    aliases: Mapping[str, Mapping[str, Any]],
    attempt_label: str,
) -> dict[str, Any]:
    payload = validation.get("payload")
    raw_candidates = payload.get("candidates") if isinstance(payload, dict) else None
    candidates: list[dict[str, Any]] = []
    if isinstance(raw_candidates, list) and len(raw_candidates) == 4:
        for index, raw in enumerate(raw_candidates):
            if not isinstance(raw, dict):
                candidates.append({
                    "candidate_id": f"Q{index + 1}",
                    "status": "invalid_local_structure",
                    "local_validation": "reject",
                })
                continue
            candidate = dict(raw)
            candidate_errors = [
                error for error in validation.get("alias_errors", [])
                if error.startswith(f"candidate_{index + 1}:")
            ]
            if raw.get("status") == "valid" and not candidate_errors:
                parent_aliases = raw.get("parent_turn_aliases") or []
                child_aliases = raw.get("child_turn_aliases") or []
                candidate["resolved_parent_turns"] = [
                    dict(aliases[str(alias)]) for alias in parent_aliases if str(alias) in aliases
                ]
                candidate["resolved_child_turns"] = [
                    dict(aliases[str(alias)]) for alias in child_aliases if str(alias) in aliases
                ]
                candidate["local_validation"] = "accept"
            elif raw.get("status") == "no_valid_candidate":
                candidate["local_validation"] = "accept"
            else:
                candidate["local_validation"] = "reject"
                candidate["local_validation_errors"] = candidate_errors
            candidates.append(candidate)
    else:
        candidates = [
            {
                "candidate_id": f"Q{index}",
                "status": "format_invalid_slot",
                "local_validation": "reject",
            }
            for index in range(1, 5)
        ]
    return {
        "group": group,
        "attempt_used": attempt_label,
        "parse_status": validation.get("parse_status"),
        "format_errors": list(validation.get("format_errors") or []),
        "alias_errors": list(validation.get("alias_errors") or []),
        "candidates": candidates,
    }


def candidate_summary(group_result: Mapping[str, Any]) -> dict[str, Any]:
    candidates = group_result.get("candidates") or []
    valid = [
        item for item in candidates
        if item.get("status") == "valid" and item.get("local_validation") == "accept"
    ]
    empty = [item for item in candidates if item.get("status") == "no_valid_candidate"]
    invalid = len(candidates) - len(valid) - len(empty)
    return {
        "valid": len(valid),
        "no_valid_candidate": len(empty),
        "invalid_local_or_format": invalid,
        "reasoning_type_distribution": dict(
            sorted(Counter(str(item["reasoning_type"]) for item in valid).items())
        ),
    }


def build_blind_review(
    group_results: Sequence[Mapping[str, Any]], seed: int = BLIND_SEED
) -> tuple[dict[str, Any], dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    for group_result in group_results:
        group = str(group_result["group"])
        for candidate in group_result["candidates"]:
            slots.append({"group": group, "candidate": candidate})
    if len(slots) != 8:
        raise JointReferenceCanaryError(f"blind review requires 8 slots, found {len(slots)}")
    rng = random.Random(seed)
    rng.shuffle(slots)
    packet_items: list[dict[str, Any]] = []
    key_items: list[dict[str, Any]] = []
    for index, slot in enumerate(slots, 1):
        blind_id = f"B{index:02d}"
        candidate = slot["candidate"]
        status = candidate.get("status")
        accepted_valid = status == "valid" and candidate.get("local_validation") == "accept"
        if accepted_valid:
            packet_item = {
                "blind_id": blind_id,
                "slot_status": "valid",
                "question": candidate["question"],
                "answer": candidate["answer"],
                "parent_evidence": [
                    {
                        key: turn[key] for key in
                        ("alias", "turn_id", "session_id", "speaker", "text")
                    }
                    for turn in candidate["resolved_parent_turns"]
                ],
                "child_evidence": [
                    {
                        key: turn[key] for key in
                        ("alias", "turn_id", "session_id", "speaker", "text")
                    }
                    for turn in candidate["resolved_child_turns"]
                ],
                "reasoning_type": candidate["reasoning_type"],
                "human_audit": {
                    "parent_only_can_answer_uniquely": None,
                    "child_only_can_answer_uniquely": None,
                    "combined_can_answer_uniquely": None,
                    "single_turn_contains_full_answer": None,
                    "evidence_is_repetitive": None,
                    "facts_are_naturally_related": None,
                    "question_is_natural": None,
                    "final_joint_judgment": None,
                    "notes": "",
                },
            }
        elif status == "no_valid_candidate":
            packet_item = {"blind_id": blind_id, "slot_status": "empty_slot"}
        else:
            packet_item = {"blind_id": blind_id, "slot_status": "invalid_slot"}
        packet_items.append(packet_item)
        key_items.append({
            "blind_id": blind_id,
            "group": slot["group"],
            "reference_used": slot["group"] == "B",
            "candidate_id": candidate.get("candidate_id"),
            "source_status": status,
            "local_validation": candidate.get("local_validation"),
        })
    return (
        {
            "blind_review_version": BLIND_REVIEW_VERSION,
            "shuffle_seed": seed,
            "slot_count": len(packet_items),
            "items": packet_items,
        },
        {
            "blind_review_version": BLIND_REVIEW_VERSION,
            "shuffle_seed": seed,
            "items": key_items,
        },
    )


def render_blind_review_markdown(packet: Mapping[str, Any]) -> str:
    lines = [
        "# Questioner Joint-Reference A/B 盲审包",
        "",
        "本文件不显示实验组、reference 使用情况、请求编号或模型自我解释。",
        "所有语义判断字段均为空白，等待人工填写。",
        "",
    ]
    for item in packet["items"]:
        lines.extend([f"## {item['blind_id']}", ""])
        if item["slot_status"] == "empty_slot":
            lines.extend(["该槽位为空（no valid candidate），不进入语义审计。", ""])
            continue
        if item["slot_status"] != "valid":
            lines.extend(["该槽位未通过本地结构/alias 校验，不进入语义审计。", ""])
            continue
        lines.extend([
            f"**Question:** {item['question']}", "",
            f"**Answer:** {item['answer']}", "",
            f"**Reasoning type:** `{item['reasoning_type']}`", "",
            "### Parent evidence", "",
        ])
        for turn in item["parent_evidence"]:
            lines.append(
                f"- `{turn['alias']}` / `{turn['turn_id']}` / `{turn['session_id']}` / "
                f"**{turn['speaker']}**: {turn['text']}"
            )
        lines.extend(["", "### Child evidence", ""])
        for turn in item["child_evidence"]:
            lines.append(
                f"- `{turn['alias']}` / `{turn['turn_id']}` / `{turn['session_id']}` / "
                f"**{turn['speaker']}**: {turn['text']}"
            )
        lines.extend([
            "", "### 空白人工审计表", "",
            "| 字段 | 人工填写 |", "|---|---|",
            "| parent_only_can_answer_uniquely | |",
            "| child_only_can_answer_uniquely | |",
            "| combined_can_answer_uniquely | |",
            "| single_turn_contains_full_answer | |",
            "| evidence_is_repetitive | |",
            "| facts_are_naturally_related | |",
            "| question_is_natural | |",
            "| final_joint_judgment | |",
            "| notes | |", "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def request_bodies_differ_only_by_reference(
    group_a: Mapping[str, Any], group_b: Mapping[str, Any], reference_text: str
) -> bool:
    a = json.loads(json.dumps(group_a))
    b = json.loads(json.dumps(group_b))
    try:
        a_user = a["messages"][1]["content"]
        b_user = b["messages"][1]["content"]
    except (KeyError, IndexError, TypeError):
        return False
    block = reference_block(reference_text)
    b["messages"][1]["content"] = b_user.replace(block, "", 1)
    return a == b and b_user.count(block) == 1 and a_user == b["messages"][1]["content"]

