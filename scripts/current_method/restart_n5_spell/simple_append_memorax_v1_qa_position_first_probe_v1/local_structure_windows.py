"""Compile deterministic natural Parent-session and Child-scene windows."""

from __future__ import annotations

from typing import Any, Mapping


WINDOW_COMPILER_VERSION = "natural_dialogue_windows_v1"


def compile_natural_windows(expanded: Mapping[str, Any]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    order: list[tuple[str, str]] = []
    errors: list[str] = []
    for position, raw in enumerate(expanded.get("turns") or [], 1):
        if not isinstance(raw, Mapping):
            errors.append(f"turn_not_object:{position}")
            continue
        origin = str(raw.get("origin") or "")
        if origin == "source_grounded":
            kind = "parent_session"
            boundary_ref = str(raw.get("session_ref") or "")
        elif origin == "generated_extension":
            kind = "child_scene"
            boundary_ref = str(raw.get("scene_ref") or "")
        else:
            errors.append(f"unknown_turn_origin:{position}")
            continue
        if not boundary_ref:
            errors.append(f"missing_natural_boundary:{position}")
            continue
        key = (kind, boundary_ref)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append({
            "global_position": position,
            "canonical_turn_id": str(raw.get("turn_ref") or ""),
            "speaker": str(raw.get("speaker") or ""),
            "text": str(raw.get("text") or ""),
            "origin": origin,
        })

    windows: list[dict[str, Any]] = []
    parent_index = 0
    child_index = 0
    for kind, boundary_ref in order:
        turns = grouped[(kind, boundary_ref)]
        if kind == "parent_session":
            parent_index += 1
            window_id = f"PW{parent_index:02d}"
        else:
            child_index += 1
            window_id = f"CW{child_index:02d}"
        positions = [turn["global_position"] for turn in turns]
        windows.append({
            "window_id": window_id,
            "window_kind": kind,
            "boundary_ref": boundary_ref,
            "global_position_start": min(positions),
            "global_position_end": max(positions),
            "turn_count": len(turns),
            "global_positions": positions,
            "turns": turns,
        })
    assigned = [
        position for window in windows for position in window["global_positions"]
    ]
    if len(assigned) != len(set(assigned)):
        errors.append("window_position_overlap")
    expected = list(range(1, len(expanded.get("turns") or []) + 1))
    if sorted(assigned) != expected:
        errors.append("window_position_coverage")
    return {
        "window_compiler_version": WINDOW_COMPILER_VERSION,
        "valid": not errors,
        "errors": sorted(set(errors)),
        "window_count": len(windows),
        "parent_session_count": parent_index,
        "child_scene_count": child_index,
        "assigned_turn_count": len(assigned),
        "windows": windows,
    }
