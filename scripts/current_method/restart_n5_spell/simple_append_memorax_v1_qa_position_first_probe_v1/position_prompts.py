"""Lean semantic Writer prompt using only 1-based dialogue positions."""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..core import sha256_bytes
from .position_contract import POSITION_SCHEMA_VERSION, QUESTION_CAPACITY_MAX


PROMPT_VERSION = "position_distractor_provenance_writer_prompt_v2"


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_position_writer_prompt(
    expanded: Mapping[str, Any], overview: Mapping[str, Any],
) -> dict[str, Any]:
    if expanded.get("root_id") != overview.get("root_id"):
        raise ValueError("position_prompt_root_mismatch")
    turns = [row for row in expanded.get("turns") or [] if isinstance(row, Mapping)]
    dialogue = [
        f"{index} | {row.get('origin')} | {row.get('speaker')} | {row.get('text')}"
        for index, row in enumerate(turns, 1)
    ]
    prompt = "\n".join([
        "ROLE: semantic reasoning-structure and MCQ Writer.",
        "Return one JSON object only. The program, not you, owns all identifiers,",
        "scope derivation, evidence-array synchronization, and answer-letter binding.",
        "The dialogue below is the only authoritative source. The overview is",
        "non-authoritative orientation and can never be evidence.",
        "",
        "YOUR ONLY TASK PER CANDIDATE",
        "1. Find a real local reasoning structure.",
        "2. Select 2-6 minimal sufficient 1-based dialogue positions (hard max 8).",
        "3. Make the semantic direction correct: old->update->new, cause->outcome,",
        "   prerequisite->dependent outcome, or chronological event order.",
        "4. Ensure exactly one option is supported as gold under the wording.",
        "5. Use three plausible near misses from the same local story arc and the",
        "   same answer dimension. Record the 1-based dialogue source positions",
        "   for each wrong option in distractor_sources.",
        "6. Then write the question and options. Skip any candidate whose local",
        "   transcript does not supply three grounded competing alternatives.",
        "",
        "ALLOWED STRUCTURE TYPES",
        "temporal_sequence; state_change; information_revision;",
        "information_visibility; dependency; task_dependency_or_constraint;",
        "causal; competing_result; comparison; cross_turn_comparison;",
        "intended_vs_executed.",
        "Do not generate considered_vs_adopted in this probe.",
        "Choose the type from the actual local structure. For temporal questions,",
        "use the same real events in changed order. For state/revision questions,",
        "use previous/current/nearby real states. For actor questions, use real",
        "people in the same event. For cause/result questions, use other real",
        "causes or results in the same thread. For plan/action questions, use",
        "real previous, proposed, intended, or executed alternatives.",
        "",
        "SEMANTIC REJECTIONS",
        "Do not label actor recall as information visibility, an ordinary decision",
        "as revision, association as dependency, or an outcome as its own cause.",
        "Do not use multiple correct options. Do not manufacture weak opposites such",
        "as ignore, hide, blame, quit forever, get fired, or get promoted unless that",
        "fact really occurs in the same local story arc. A fact elsewhere in the",
        "transcript is not a near miss if it does not compete on the answer dimension.",
        "",
        "LEAN OUTPUT CONTRACT",
        f"position_schema_version={POSITION_SCHEMA_VERSION}.",
        f"questions contains 0-{QUESTION_CAPACITY_MAX} items; {QUESTION_CAPACITY_MAX} is capacity, not a target.",
        "Each question item has exactly: structure_type, evidence_positions,",
        "question, options, gold_text, distractor_sources. options has exactly A,B,C,D.",
        "evidence_positions contains unique 1-based integers from the numbered",
        "dialogue. Never output D###, canonical turn IDs, source_scope,",
        "supporting_turn_ids, evidence_turn_ids, candidate IDs, or answer letters.",
        "gold_text must exactly equal one and only one option value.",
        "distractor_sources is an object whose keys are exactly the three WRONG",
        "option letters. Omit the gold option letter. Each value is a non-empty list",
        "of unique 1-based dialogue positions containing the real fact used for that",
        "wrong option. Do not cite the overview. Do not cite a distant unrelated arc.",
        "abstention_summary has exactly estimated_skipped_count and reason_codes.",
        "An empty questions list requires a positive skipped count and reasons.",
        "There is no minimum question count and no coverage quota.",
        "",
        "FROZEN LOCAL MEMORAX OVERVIEW (NON-AUTHORITATIVE)",
        _compact(overview),
        "",
        "NUMBERED AUTHORITATIVE DIALOGUE",
        *dialogue,
        "",
        "Return only the lean JSON object. Do not add mechanical evidence fields.",
    ])
    return {
        "prompt_version": PROMPT_VERSION,
        "prompt": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "character_count": len(prompt),
        "token_estimate": max(1, (len(prompt) + 3) // 4),
        "dialogue_turn_count": len(turns),
        "question_capacity_max": QUESTION_CAPACITY_MAX,
    }
