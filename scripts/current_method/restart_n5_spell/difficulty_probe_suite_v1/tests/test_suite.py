from __future__ import annotations

import json
from pathlib import Path

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.semantic_audit import (
    paraphrase_audit, source_audit,
)


def test_paraphrase_freezes_answer_and_evidence() -> None:
    original = {"question": "What changed?", "options": {"A": "x"}, "gold_letter": "A",
                "gold_text": "x", "evidence_turn_refs": ["T1"], "evidence_texts": ["A fact"]}
    variant = {**original, "question": "Which change occurred?"}
    assert paraphrase_audit(original, variant)["valid"]


def test_source_rewrite_keeps_question_and_refs() -> None:
    original = {"question": "What changed?", "options": {"A": "x"}, "gold_letter": "A",
                "gold_text": "x", "evidence_turn_refs": ["T1"], "evidence_texts": ["A fact"]}
    variant = {**original, "evidence_texts": ["A stated fact"]}
    assert source_audit(original, variant)["valid"]

