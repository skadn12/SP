"""Fixtures for deterministic position materialization."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .position_contract import POSITION_SCHEMA_VERSION


def revision_question(index: int = 1, positions: list[int] | None = None) -> dict[str, Any]:
    old = f"Old plan {index}"
    new = f"Revised plan {index}"
    source_positions = list(positions or [1, 2, 3])
    if len(source_positions) < 2:
        source_positions = [1, 2]
    return {
        "structure_type": "information_revision",
        "evidence_positions": list(positions or [1, 2, 3]),
        "question": f"After the update in case {index}, what was the revised plan?",
        "options": {
            "A": old,
            "B": new,
            "C": f"Nearby alternative {index}",
            "D": f"Earlier unrelated proposal {index}",
        },
        "gold_text": new,
        "distractor_sources": {
            "A": [source_positions[0]],
            "C": [source_positions[1]],
            "D": [source_positions[2] if len(source_positions) > 2 else source_positions[0]],
        },
    }


def envelope(count: int, positions: list[int] | None = None) -> dict[str, Any]:
    return {
        "position_schema_version": POSITION_SCHEMA_VERSION,
        "abstention_summary": {
            "estimated_skipped_count": 1 if count == 0 else 0,
            "reason_codes": ["no_reliable_structure"] if count == 0 else [],
        },
        "questions": [revision_question(index, positions)
                      for index in range(1, count + 1)],
    }


def missing_distractor_provenance() -> dict[str, Any]:
    value = envelope(1)
    del value["questions"][0]["distractor_sources"]["D"]
    return value


def out_of_range_distractor_source() -> dict[str, Any]:
    value = envelope(1)
    value["questions"][0]["distractor_sources"]["D"] = [111]
    return value


def cross_story_distractor_source() -> dict[str, Any]:
    value = envelope(1, [1, 2, 3])
    # Positions 1-3 are in the first parent session; position 50 is elsewhere.
    value["questions"][0]["distractor_sources"]["C"] = [50]
    return value


def invented_distractor_without_source() -> dict[str, Any]:
    value = envelope(1)
    value["questions"][0]["options"]["D"] = "They were promoted for no stated reason."
    value["questions"][0]["distractor_sources"]["D"] = []
    return value


def temporal() -> dict[str, Any]:
    events = ["checked the draft", "reported the error", "revised the workflow"]
    value = envelope(1)
    value["questions"][0] = {
        "structure_type": "temporal_sequence",
        "evidence_positions": [1, 2, 3],
        "question": "Which sequence is correct?",
        "options": {
            "A": ", then ".join(events),
            "B": ", then ".join([events[1], events[0], events[2]]),
            "C": ", then ".join([events[2], events[1], events[0]]),
            "D": ", then ".join([events[0], events[2], events[1]]),
        },
        "gold_text": ", then ".join(events),
        "distractor_sources": {"B": [2], "C": [3], "D": [1]},
    }
    return value


def duplicate_gold() -> dict[str, Any]:
    value = envelope(1)
    row = value["questions"][0]
    row["options"]["C"] = row["gold_text"]
    return value


def copy_fixture(value: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(value)
