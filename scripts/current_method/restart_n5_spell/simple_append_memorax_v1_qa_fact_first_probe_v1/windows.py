"""Compile complete Parent-session and Child-scene fact extraction windows."""

from __future__ import annotations

from typing import Any, Mapping


WINDOW_COMPILER_VERSION = "fact_first_natural_windows_v1"


def compile_natural_windows(expanded: Mapping[str, Any]) -> dict[str, Any]:
    windows: list[dict[str, Any]] = []
    errors: list[str] = []
    seen_boundaries: set[tuple[str, str]] = set()
    active_key: tuple[str, str] | None = None
    active: dict[str, Any] | None = None
    parent_count = 0
    child_count = 0

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
        if key != active_key:
            if key in seen_boundaries:
                errors.append(f"noncontiguous_boundary:{kind}:{boundary_ref}")
            seen_boundaries.add(key)
            if kind == "parent_session":
                parent_count += 1
                window_id = f"P{parent_count:02d}"
            else:
                child_count += 1
                window_id = f"C{child_count:02d}"
            active = {
                "window_id": window_id,
                "window_kind": kind,
                "boundary_ref": boundary_ref,
                "origin": origin,
                "window_order": len(windows) + 1,
                "turns": [],
            }
            windows.append(active)
            active_key = key
        assert active is not None
        active["turns"].append({
            "global_position": position,
            "canonical_turn_id": str(raw.get("turn_ref") or ""),
            "speaker": str(raw.get("speaker") or ""),
            "text": str(raw.get("text") or ""),
            "origin": origin,
        })

    for window in windows:
        positions = [turn["global_position"] for turn in window["turns"]]
        window["global_positions"] = positions
        window["global_position_start"] = min(positions)
        window["global_position_end"] = max(positions)
        window["turn_count"] = len(positions)
    assigned = [
        position for window in windows for position in window["global_positions"]
    ]
    expected = list(range(1, len(expanded.get("turns") or []) + 1))
    if len(assigned) != len(set(assigned)):
        errors.append("window_position_overlap")
    if sorted(assigned) != expected:
        errors.append("window_position_coverage")
    return {
        "window_compiler_version": WINDOW_COMPILER_VERSION,
        "valid": not errors,
        "errors": sorted(set(errors)),
        "window_count": len(windows),
        "parent_session_count": parent_count,
        "child_scene_count": child_count,
        "assigned_turn_count": len(assigned),
        "windows": windows,
    }

