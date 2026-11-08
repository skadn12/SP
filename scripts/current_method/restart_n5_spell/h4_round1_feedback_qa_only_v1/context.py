"""Build compact, frozen Round-0 QA feedback context for QA-only generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..core import canonical_json, sha256_file
from . import (
    FEEDBACK_COMPILER_RUN,
    ROUND0_DIFFICULTY_RUN,
    ROUND0_TRANSITION_RUN,
    SLOT_TYPES,
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _unique(rows: list[Mapping[str, Any]], key: str, value: str) -> dict[str, Any]:
    found = [dict(row) for row in rows if str(row.get(key) or "") == value]
    if len(found) != 1:
        raise RuntimeError(f"feedback_context_not_unique:{key}:{value}:{len(found)}")
    return found[0]


def _candidate_detail(row: Mapping[str, Any]) -> dict[str, Any]:
    candidate = row.get("candidate")
    if not isinstance(candidate, Mapping):
        raise RuntimeError("round0_candidate_detail_missing")
    options = candidate.get("options") or candidate.get("items") or {}
    return {
        "question": candidate.get("question"),
        "question_type": candidate.get("question_type"),
        "options_or_items": options,
        "canonical_gold": candidate.get("gold") or [],
        "evidence_refs": candidate.get("evidence_refs") or [],
        "evidence_groups": candidate.get("evidence_groups") or {},
        "generation_source": candidate.get("generation_source"),
        "discriminative_variable": candidate.get("discriminative_variable"),
    }


def load_feedback_packet(root_id: str) -> dict[str, Any]:
    cards = _jsonl(FEEDBACK_COMPILER_RUN / "root_feedback_cards.jsonl")
    seeds = _jsonl(FEEDBACK_COMPILER_RUN / "feedback_seed_manifest.jsonl")
    features = _jsonl(FEEDBACK_COMPILER_RUN / "candidate_feedback_features.jsonl")
    difficulty = _jsonl(ROUND0_DIFFICULTY_RUN / "difficulty_manifest.jsonl")
    transitions = _jsonl(ROUND0_TRANSITION_RUN / "candidate_transition_matrix.jsonl")

    card = _unique(cards, "root_id", root_id)
    seed = _unique(seeds, "root_id", root_id)
    seed_id = str(seed.get("seed_candidate_id") or seed.get("candidate_id") or "")
    if not seed_id:
        raise RuntimeError(f"feedback_seed_id_missing:{root_id}")
    feature = _unique(features, "candidate_id", seed_id)
    difficulty_row = _unique(difficulty, "candidate_id", seed_id)
    transition = _unique(transitions, "candidate_id", seed_id)
    detail = _candidate_detail(difficulty_row)

    # Keep only feedback metadata and the frozen source QA. The complete
    # cumulative dialogue remains the only fact source in the writer prompt.
    return {
        "packet_version": "round1_qa_feedback_only_packet_v1",
        "feedback_source_round": "round0",
        "target_round": "round1",
        "root_id": root_id,
        "source_candidate_id": seed_id,
        "source_candidate_sha256": seed.get("candidate_sha256"),
        "feedback_card": card.get("target") or {},
        "seed_metadata": {
            "gap_type": seed.get("seed_gap_type"),
            "primary_transition": seed.get("seed_primary_transition"),
            "question_type": seed.get("seed_question_type"),
            "training_exposure": seed.get("training_exposure"),
            "evidence_confidence": seed.get("evidence_confidence"),
        },
        "source_qa": detail,
        "transition": {
            "models": transition.get("models") or {},
            "position_mechanism": transition.get("position_mechanism"),
            "primary_transition": transition.get("primary_transition"),
        },
        "canonical_confusion": feature.get("full_context_error_feature") or {},
        "lineage": {
            "feedback_cards_sha256": sha256_file(FEEDBACK_COMPILER_RUN / "root_feedback_cards.jsonl"),
            "feedback_seed_manifest_sha256": sha256_file(FEEDBACK_COMPILER_RUN / "feedback_seed_manifest.jsonl"),
            "candidate_features_sha256": sha256_file(FEEDBACK_COMPILER_RUN / "candidate_feedback_features.jsonl"),
        },
    }


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def render_prompt_context(packet: Mapping[str, Any]) -> dict[str, Any]:
    packet_text = _compact(packet)
    card_text = _compact(packet.get("feedback_card") or {})
    notice = (
        "CURRICULUM FEEDBACK NOTICE: The Round-0 source QA and feedback metadata "
        "are curriculum signals, not facts about the current dialogue. Use "
        "CURRENT_CUMULATIVE_DIALOGUE as the only factual source. Derive every "
        "new question, option, gold answer, and evidence reference from that dialogue."
    )
    intent_suffix = "\n".join([
        "<QA_FEEDBACK_CURRICULUM>",
        notice,
        "FROZEN_ROUND0_FEEDBACK_CARD:", card_text,
        "FROZEN_SOURCE_QA_PACKET:", packet_text,
        "Plan exactly three distinct feedback-conditioned reasoning intents. "
        "Do not output questions or answers.",
        "</QA_FEEDBACK_CURRICULUM>",
    ])
    slot_suffixes = []
    slot_goals = {
        SLOT_TYPES[0]: (
            "Generate a harder successor that exercises the feedback gap "
            "while adding a distinct fact or relation from the current dialogue."),
        SLOT_TYPES[1]: (
            "Target the canonical confusion pattern in the source QA, but "
            "re-derive all facts and evidence from the current dialogue."),
        SLOT_TYPES[2]: (
            "Use the feedback gap as a design constraint while covering a "
            "different part of the current dialogue; do not copy the old QA."),
    }
    for slot_type in SLOT_TYPES:
        slot_suffixes.append("\n".join([
            "<QA_FEEDBACK_CURRICULUM>",
            notice,
            "SLOT_TYPE:", slot_type,
            "SLOT_GOAL:", slot_goals[slot_type],
            "FROZEN_SOURCE_QA_PACKET:", packet_text,
            "The current cumulative dialogue is the only fact source. "
            "Do not mention this packet, feedback, model names, scores, or metadata in output.",
            "</QA_FEEDBACK_CURRICULUM>",
        ]))
    return {
        "intent_suffix": intent_suffix,
        "slot_suffixes": slot_suffixes,
        "packet_sha256": sha256_file(FEEDBACK_COMPILER_RUN / "root_feedback_cards.jsonl"),
    }
