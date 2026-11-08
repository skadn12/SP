"""Memory-augmented prompt wrappers that preserve simple-append schemas."""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..core import sha256_bytes
from ..simple_append_v1.prompts import (
    build_mcq_batch_prompt,
    build_planner_prompt,
    build_round_realizer_prompt,
)
from .memory_context import MEMORY_OVERVIEW_VERSION


MEMORY_PLANNER_PROMPT_VERSION = "simple_append_memorax_planner_prompt_v1"
MEMORY_REALIZER_PROMPT_VERSION = "simple_append_memorax_realizer_prompt_v1"
MEMORY_MCQ_PROMPT_VERSION = (
    "simple_append_memorax_mcq_prompt_v8_position_first")
SUPPORTED_MEMORY_OVERVIEW_VERSIONS = frozenset({
    "simple_append_memorax_overview_v1",
    "simple_append_memorax_overview_v2_person_events",
})


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _validate_overview(
    memory_overview: Mapping[str, Any], *, root_id: str,
) -> None:
    if memory_overview.get("memory_overview_version") not in \
            SUPPORTED_MEMORY_OVERVIEW_VERSIONS:
        raise ValueError("memorax_memory_overview_version_invalid")
    if str(memory_overview.get("root_id") or "") != str(root_id):
        raise ValueError("memorax_memory_overview_root_mismatch")
    if memory_overview.get("memory_is_qa_fact_source") is not False:
        raise ValueError("memorax_memory_cannot_be_qa_fact_source")


def _result(
    prompt: str, *, version: str, base: Mapping[str, Any],
    overview: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "prompt_version": version,
        "base_prompt_version": base["prompt_version"],
        "prompt": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "character_count": len(prompt),
        "token_estimate": max(1, (len(prompt) + 3) // 4),
        "memory_overview_sha256": overview["memory_overview_sha256"],
        "memory_is_qa_fact_source": False,
    }


def _memory_block(memory_overview: Mapping[str, Any], *, purpose: str) -> str:
    return "\n".join([
        f"NON-AUTHORITATIVE MEMORAX MEMORY FOR {purpose}",
        "Use this compact view to recall earlier people, places, events, items, and themes. "
        "It may be incomplete or paraphrased. The provided dialogue text remains authoritative. "
        "Never invent a fact merely because it appears here, and never use a Memory key as a "
        "turn ID, QA gold, or evidence reference.",
        _json(memory_overview),
    ])


def _qa_memory_topic_index(memory_overview: Mapping[str, Any]) -> dict[str, Any]:
    def labels(key: str) -> list[str]:
        result: list[str] = []
        for row in memory_overview.get(key) or []:
            if not isinstance(row, Mapping):
                continue
            label = str(row.get("display_key") or row.get("canonical_key") or "").strip()
            if label and label not in result:
                result.append(label)
        return result

    return {
        "memory_topic_index_version": "memorax_qa_topic_index_v1",
        "root_id": str(memory_overview.get("root_id") or ""),
        "people": labels("people"),
        "events": labels("events"),
        "locations": labels("locations"),
        "items": labels("items"),
        "themes": labels("themes"),
    }


def _qa_memory_topic_block(memory_overview: Mapping[str, Any]) -> str:
    return "\n".join([
        "NON-AUTHORITATIVE MEMORAX MEMORY FOR QUESTION TOPIC SELECTION",
        "This is a compact index of recalled topic names, not a fact source. Use it only "
        "to notice possible themes. The complete dialogue below is authoritative for every "
        "question, option, answer, and D### evidence reference. Ignore any indexed topic "
        "that cannot be located in the dialogue.",
        _json(_qa_memory_topic_index(memory_overview)),
    ])


def _insert(prompt: str, marker: str, block: str) -> str:
    if marker not in prompt:
        raise ValueError("memorax_base_prompt_marker_missing")
    return prompt.replace(marker, block + "\n\n" + marker, 1)


def build_memory_augmented_planner_prompt(
    tail_context: Mapping[str, Any], memory_overview: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_overview(memory_overview, root_id=str(tail_context["root_id"]))
    base = build_planner_prompt(tail_context)
    prompt = _insert(
        base["prompt"], "OUTPUT EXACTLY THIS SHAPE\n",
        _memory_block(memory_overview, purpose="EXPANSION PLANNING"))
    return _result(
        prompt, version=MEMORY_PLANNER_PROMPT_VERSION,
        base=base, overview=memory_overview)


def build_memory_augmented_realizer_prompt(
    *, compiled_plan: Mapping[str, Any], tail_context: Mapping[str, Any],
    memory_overview: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_overview(memory_overview, root_id=str(tail_context["root_id"]))
    base = build_round_realizer_prompt(
        compiled_plan=compiled_plan, tail_context=tail_context)
    prompt = _insert(
        base["prompt"], "OUTPUT EXACTLY THIS SHAPE WITH EVERY PLANNED SCENE KEY\n",
        _memory_block(memory_overview, purpose="ROUND REALIZATION"))
    return _result(
        prompt, version=MEMORY_REALIZER_PROMPT_VERSION,
        base=base, overview=memory_overview)


def build_memory_augmented_mcq_prompt(
    expanded_dialogue: Mapping[str, Any], memory_overview: Mapping[str, Any],
    *, target_questions: int = 8, maximum_questions: int | None = 12,
    focus_round_index: int | None = None,
) -> dict[str, Any]:
    _validate_overview(
        memory_overview, root_id=str(expanded_dialogue.get("root_id") or ""))
    base = build_mcq_batch_prompt(
        expanded_dialogue, target_questions=target_questions,
        maximum_questions=maximum_questions,
        focus_round_index=focus_round_index,
        compact_source_serialization=True,
        option_contract=True,
        position_first=True)
    block = "\n".join([
        _qa_memory_topic_block(memory_overview),
        "Use Memory only to choose promising cross-round themes. Every question, option, "
        "answer, and evidence alias must still be supported by the authoritative dialogue "
        "partitions below. If a Memory hint cannot be located there, do not ask about it.",
        "TRANSCRIPT-GROUNDED DISTRACTOR CONTRACT",
        "For each wrong option, privately identify a real nearby fact, state, plan, actor, "
        "event, cause, result, or timepoint from the same local story arc. Adapt that real "
        "near-miss to the same answer slot as the gold. The option must be wrong for the "
        "exact question, but its underlying content must occur in the dialogue. Do not "
        "invent a negation, opposite intention, cancellation, denial, or unrelated event "
        "merely to make an answer wrong.",
        "For temporal questions, reuse the same real events and change only their order. "
        "For state or revision questions, use real earlier, proposed, or nearby states. "
        "For actor questions, use other real participants in that local event. For causal "
        "or result questions, use other real causes or results from that local thread. For "
        "plan or action questions, use real proposed, intended, attempted, or executed "
        "alternatives from that thread.",
        "Do not use cancel the event, deny climate change, ignore environmental concerns, "
        "reduce visibility, focus on no environmental goals, blame someone, get fired, get "
        "promoted, hide information, or quit unless that exact competing fact genuinely "
        "appears in the same local story arc. If the dialogue does not provide three "
        "same-dimension local near-misses, omit the candidate instead of inventing them.",
    ])
    prompt = _insert(
        base["prompt"], "AUTHORITATIVE SOURCE PARTITION CONTRACT\n", block)
    result = _result(
        prompt, version=MEMORY_MCQ_PROMPT_VERSION,
        base=base, overview=memory_overview)
    result.update({
        "dialogue_serialization": base["dialogue_serialization"],
        "memory_prompt_view": "memorax_qa_topic_index_v1",
        "memory_overview_full_embedded": False,
    })
    return result
