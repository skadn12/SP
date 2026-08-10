"""Memory-augmented prompt wrappers that preserve simple-append schemas."""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..core import sha256_bytes
from ..simple_append_v1.prompts import build_mcq_batch_prompt
from .durable_prompts import build_planner_prompt, build_round_realizer_prompt
from .memory_context import MEMORY_OVERVIEW_VERSION


MEMORY_PLANNER_PROMPT_VERSION = (
    "simple_append_memorax_fullr1_durable_devdiv_local_span_planner_prompt_v1")
MEMORY_REALIZER_PROMPT_VERSION = (
    "simple_append_memorax_fullr1_durable_devdiv_local_span_realizer_prompt_v1")
MEMORY_MCQ_PROMPT_VERSION = "simple_append_memorax_mcq_prompt_v4"

def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _validate_overview(
    memory_overview: Mapping[str, Any], *, root_id: str,
) -> None:
    if memory_overview.get("memory_overview_version") != \
            MEMORY_OVERVIEW_VERSION:
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


def _insert(prompt: str, marker: str, block: str) -> str:
    if marker not in prompt:
        raise ValueError("memorax_base_prompt_marker_missing")
    return prompt.replace(marker, block + "\n\n" + marker, 1)


def build_memory_augmented_planner_prompt(
    tail_context: Mapping[str, Any], memory_overview: Mapping[str, Any],
    parent_anchor_registry: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_overview(memory_overview, root_id=str(tail_context["root_id"]))
    base = build_planner_prompt(tail_context, parent_anchor_registry)
    prompt = base["prompt"]
    prompt = _insert(
        prompt, "OUTPUT EXACTLY THIS SHAPE\n",
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
        focus_round_index=focus_round_index)
    block = "\n".join([
        _memory_block(memory_overview, purpose="QUESTION TOPIC SELECTION"),
        "Use Memory only to choose promising cross-round themes. Every question, option, "
        "answer, and evidence alias must still be supported by the authoritative dialogue "
        "partitions below. If a Memory hint cannot be located there, do not ask about it.",
    ])
    prompt = _insert(
        base["prompt"], "AUTHORITATIVE SOURCE PARTITION CONTRACT\n", block)
    return _result(
        prompt, version=MEMORY_MCQ_PROMPT_VERSION,
        base=base, overview=memory_overview)
