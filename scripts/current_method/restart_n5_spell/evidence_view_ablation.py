from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence


EVIDENCE_VIEW_ABLATION_VERSION = "questioner_evidence_view_ablation_v1"
CHILD_ONLY_VIEW = "child_only"
BALANCED_VIEW = "parent_child_balanced"
EVIDENCE_VIEW_MODES = (CHILD_ONLY_VIEW, BALANCED_VIEW)
BALANCED_PARENT_SELECTION = "uniform_endpoint_inclusive_v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def evidence_view_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_turns(turns: Sequence[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for turn in turns:
        if not isinstance(turn, dict):
            raise ValueError(f"{label}_view_invalid_turn")
        turn_id = turn.get("turn_id")
        session_id = turn.get("session_id")
        if (not isinstance(turn_id, str) or not turn_id or turn_id in seen or
                not isinstance(session_id, str) or not session_id or
                not isinstance(turn.get("speaker"), str) or not turn["speaker"] or
                not isinstance(turn.get("text"), str) or not turn["text"]):
            raise ValueError(f"{label}_view_invalid_turn")
        result.append(dict(turn))
        seen.add(turn_id)
    if not result:
        raise ValueError(f"{label}_view_empty")
    return result


def uniform_endpoint_indices(total: int, selected: int) -> tuple[int, ...]:
    """Choose stable, evenly spaced source positions including both endpoints."""
    if total < 1 or selected < 1 or selected > total:
        raise ValueError("balanced_view_selection_invalid")
    if selected == 1:
        return (0,)
    indices = tuple((position * (total - 1)) // (selected - 1)
                    for position in range(selected))
    if len(set(indices)) != selected:
        raise ValueError("balanced_view_selection_not_unique")
    return indices


def build_questioner_evidence_view(*, mode: str,
                                   parent_turns: Sequence[dict[str, Any]],
                                   child_turns: Sequence[dict[str, Any]]) -> dict[str, Any]:
    parent = _validate_turns(parent_turns, "parent")
    child = _validate_turns(child_turns, "child")
    if set(turn["turn_id"] for turn in parent) & set(turn["turn_id"] for turn in child):
        raise ValueError("evidence_view_turn_id_overlap")
    if mode == CHILD_ONLY_VIEW:
        selected_parent: list[dict[str, Any]] = []
        selected_child = child
        selection = {"algorithm":"all_child_turns_v1", "parent_indices":[]}
    elif mode == BALANCED_VIEW:
        count = min(len(parent), len(child))
        indices = uniform_endpoint_indices(len(parent), count)
        selected_parent = [parent[index] for index in indices]
        selected_child = child[:count]
        selection = {"algorithm":BALANCED_PARENT_SELECTION,
                     "parent_indices":list(indices)}
    else:
        raise ValueError("questioner_evidence_view_mode_invalid")
    turns = selected_parent + selected_child
    payload = {"evidence_view_ablation_version":EVIDENCE_VIEW_ABLATION_VERSION,
        "view_mode":mode, "selection":selection,
        "source_parent_turn_count":len(parent), "source_child_turn_count":len(child),
        "selected_parent_turn_count":len(selected_parent),
        "selected_child_turn_count":len(selected_child),
        "selected_turn_count":len(turns), "combined_dialogue":turns}
    payload["view_sha256"] = evidence_view_sha256(payload)
    return payload


def selected_evidence_diagnostics(*, span_ids: Sequence[str],
                                  catalog: dict[str, Any],
                                  scene_contracts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    spans = catalog.get("spans")
    if not isinstance(spans, list):
        raise ValueError("evidence_view_span_catalog_invalid")
    index = {span.get("span_id"):span for span in spans if isinstance(span, dict)}
    if len(index) != len(spans):
        raise ValueError("evidence_view_span_catalog_invalid")
    selected = []
    for span_id in span_ids:
        if span_id not in index:
            raise ValueError("evidence_view_selected_span_missing")
        selected.append(index[span_id])
    child_sessions = {str(span["session_id"]) for span in selected
                      if str(span["session_id"]).startswith("session_")}
    contract_by_session = {
        f"session_{int(contract['scene_index']):02d}":contract
        for contract in scene_contracts
    }
    arcs = {str(arc) for session_id in child_sessions
            for arc in contract_by_session.get(session_id, {}).get(
                "assigned_event_arc_ids", [])}
    stages = {str(stage) for session_id in child_sessions
              for stage in contract_by_session.get(session_id, {}).get(
                  "assigned_event_stage_ids", [])}
    return {"selected_parent_span_count":sum(
                not str(span["session_id"]).startswith("session_") for span in selected),
        "selected_child_span_count":sum(
                str(span["session_id"]).startswith("session_") for span in selected),
        "selected_event_arc_count":len(arcs),
        "selected_event_stage_count":len(stages),
        "selected_session_count":len({span["session_id"] for span in selected}),
        "selected_turn_count":len({span["turn_id"] for span in selected}),
        "selected_event_arc_ids":sorted(arcs),
        "selected_event_stage_ids":sorted(stages)}
