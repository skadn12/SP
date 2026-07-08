#!/usr/bin/env python3
"""Compatibility subset for the historical v1.3 pilot generator.

The original full script is not present in this cleaned workspace.  The restored
r2/auto graph-expansion path only needs deterministic shuffling, quota rounding,
leak-pattern constants, and SFT chat formatting, so this module keeps those APIs
available without pretending to be the full generator.
"""

from __future__ import annotations

import hashlib
import random
from math import floor
from typing import Any


LEAK_PATTERNS = (
    "the correct answer is",
    "gold answer",
    "gold option",
    "answer key",
    "choose option",
    "correct option",
)


def stable_shuffle(values: list[Any], seed_text: str) -> list[Any]:
    items = list(values)
    seed = int(hashlib.sha1(str(seed_text).encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    rng.shuffle(items)
    return items


def largest_remainder_counts(weights: dict[str, float], n: int) -> dict[str, int]:
    if n <= 0:
        return {key: 0 for key in weights}
    total = sum(max(0.0, float(value)) for value in weights.values())
    if total <= 0:
        base = n // len(weights)
        counts = {key: base for key in weights}
        for key in sorted(weights)[: n - base * len(weights)]:
            counts[key] += 1
        return counts
    raw = {key: n * max(0.0, float(value)) / total for key, value in weights.items()}
    counts = {key: floor(value) for key, value in raw.items()}
    remaining = n - sum(counts.values())
    order = sorted(weights, key=lambda key: (raw[key] - counts[key], key), reverse=True)
    for key in order[:remaining]:
        counts[key] += 1
    return counts


def build_chat_record(sample: dict[str, Any]) -> dict[str, Any]:
    question_type = str(sample["question_type"])
    if question_type == "single_choice":
        system = (
            "You are a long-dialogue memory QA solver. Read carefully, track state updates, "
            "and answer with only the single correct option letter and option text."
        )
        answer = f"{sample['gold_option']}. {sample['gold_answer']}"
    elif question_type == "multi_select":
        system = (
            "You are a long-dialogue memory QA solver. Select all and only correct option letters. "
            "Answer with comma-separated letters."
        )
        answer = ", ".join(str(item) for item in sample["gold_options"])
    elif question_type == "ordering":
        system = (
            "You are a long-dialogue memory QA solver. Order the option letters exactly as requested. "
            "Answer only with letters separated by >."
        )
        answer = " > ".join(str(item) for item in sample["gold_order"])
    else:
        system = (
            "You are a long-dialogue memory QA solver. Reconstruct the requested state exactly. "
            "Answer with only the exact entry text."
        )
        answer = str(sample["gold_answer"])

    sections = ["Dialogue evidence:", str(sample["expanded_dialogue_text"]), "Question:", str(sample["question"])]
    options = sample.get("options")
    if isinstance(options, dict) and options:
        sections.extend(["Options:", "\n".join(f"{label}. {options[label]}" for label in sorted(options))])
    user = "\n\n".join(sections)

    metadata = dict(sample.get("metadata") or {})
    metadata.update(
        {
            "record_id": sample["expanded_graph_id"],
            "sample_name": sample["expanded_dialogue_id"],
            "source_graph_id": sample.get("source_graph_id"),
            "expanded_graph_id": sample.get("expanded_graph_id"),
            "expanded_dialogue_id": sample.get("expanded_dialogue_id"),
            "question_type": question_type,
            "answer_format": sample.get("answer_format"),
            "gold_label": sample.get("gold_option"),
            "gold_options": sample.get("gold_options"),
            "gold_order": sample.get("gold_order"),
            "gold_answer": sample.get("gold_answer"),
            "gold_source": sample.get("gold_source"),
            "attack_type": sample.get("attack_type"),
            "trap_type": sample.get("trap_type"),
            "state_chain_depth": sample.get("state_chain_depth"),
            "added_turn_count": sample.get("added_turn_count"),
            "source_split": sample.get("source_split"),
            "training_use": sample.get("training_use"),
            "difficulty_for_solver": sample.get("difficulty_for_solver") or "frontier_v13",
            "quality_tier": sample.get("quality_tier") or "graph_expansion_v13_auto",
        }
    )
    return {
        "record_id": sample["expanded_graph_id"],
        "sample_name": sample["expanded_dialogue_id"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
        "metadata": metadata,
    }
