"""Adapters for the frozen expanded dialogue and optional planner artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import Turn


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_transcript(path: str | Path) -> tuple[str, list[Turn], dict[str, Any]]:
    """Load canonical transcript turns; evidence always comes from this file."""
    source = Path(path)
    value = read_json(source)
    if not isinstance(value, Mapping) or not isinstance(value.get("turns"), list):
        raise ValueError(f"expanded_dialogue_turns_required:{source}")
    root_id = str(value.get("root_id") or "")
    if not root_id:
        raise ValueError("expanded_dialogue_root_id_required")
    turns: list[Turn] = []
    for index, row in enumerate(value["turns"], 1):
        if not isinstance(row, Mapping):
            raise ValueError(f"expanded_dialogue_turn_invalid:{index}")
        turn_ref = str(row.get("turn_ref") or "")
        speaker = str(row.get("speaker") or "").strip()
        text = str(row.get("text") or "").strip()
        origin = str(row.get("origin") or "")
        if not all((turn_ref, speaker, text)) or origin not in {
                "source_grounded", "generated_extension"}:
            raise ValueError(f"expanded_dialogue_turn_fields_invalid:{index}")
        turns.append(Turn(index, turn_ref, speaker, text, origin, root_id))
    return root_id, turns, dict(value)


def load_structured_sources(paths: list[str | Path]) -> dict[str, Any]:
    """Read planner/promotion JSON for annotations, never as transcript evidence."""
    sources: dict[str, Any] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        value = read_json(path)
        sources[str(path)] = value
    return sources


def _walk(value: Any, path: str = ""):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk(child, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")
    else:
        yield path, value


def structured_texts(sources: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return structured leaf text with stable source paths for audit annotations."""
    result: list[tuple[str, str]] = []
    for source_path, value in sources.items():
        for field_path, leaf in _walk(value):
            if isinstance(leaf, str) and leaf.strip():
                result.append((f"{source_path}:{field_path}", leaf.strip()))
    return result

