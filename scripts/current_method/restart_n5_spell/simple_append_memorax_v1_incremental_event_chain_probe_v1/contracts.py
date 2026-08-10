"""Mechanical contracts for incremental event-chain Fresh R1."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Mapping, Sequence

from ..simple_append_v1.contracts import (
    PLANNER_SCHEMA_VERSION as BASE_PLANNER_SCHEMA_VERSION,
    compile_round_plan as compile_base_round_plan,
    normalize_round_realizer_output,
    validate_planner_output as validate_base_planner_output,
)


INITIALIZER_SCHEMA_VERSION = "simple_append_local_event_chain_initializer_v2"
CHAIN_SET_VERSION = "simple_append_current_event_chains_v1"
PLANNER_SCHEMA_VERSION = "simple_append_event_chain_planner_v1"
COMPILED_PLAN_VERSION = "simple_append_event_chain_compiled_plan_v1"
DELTA_UPDATER_SCHEMA_VERSION = "simple_append_event_chain_delta_updater_v1"
UPDATED_CHAIN_SET_VERSION = "simple_append_updated_event_chains_v1"

_INITIALIZER_TOP_KEYS = {
    "chain_initializer_schema_version", "session_event_candidates",
    "continuation_links",
}
_SESSION_CANDIDATE_KEYS = {"session_ref", "events"}
_LOCAL_EVENT_KEYS = {
    "event_summary", "event_state", "participants", "evidence_turn_refs",
}
_CONTINUATION_LINK_KEYS = {
    "earlier_session_ref", "earlier_candidate_index",
    "later_session_ref", "later_candidate_index",
}
_PLANNER_TOP_KEYS = {"planner_schema_version", "round_goal", "scenes"}
_PLANNER_SCENE_KEYS = {
    "chain_id", "goal", "participants", "new_development",
    "interaction_beats", "transition_from_previous", "desired_outcome",
    "durable_outcome",
}
_BASE_SCENE_KEYS = {
    "goal", "participants", "new_development", "interaction_beats",
    "transition_from_previous", "desired_outcome",
}
_DELTA_TOP_KEYS = {
    "delta_updater_schema_version", "chain_updates",
    "untracked_new_event_candidates",
}
_DELTA_UPDATE_KEYS = {
    "chain_id", "update_status", "new_current_state", "evidence_turn_refs",
}
_UNTRACKED_KEYS = {"event_summary", "evidence_turn_refs"}
_STOP_WORDS = {
    "a", "an", "the", "to", "and", "or", "of", "in", "on", "at", "for",
    "from", "with", "has", "have", "had", "is", "are", "was", "were", "be",
    "been", "being", "already", "still", "then", "later", "now", "will",
    "would", "should", "could", "their", "his", "her", "they", "them", "it",
    "this", "that",
}
_WORD_NORMALIZATION = {
    "decides": "decide", "decided": "decide", "deciding": "decide",
    "resumes": "resume", "resumed": "resume", "resuming": "resume",
    "starts": "start", "started": "start", "starting": "start",
    "begins": "begin", "began": "begin", "beginning": "begin",
    "plans": "plan", "planned": "plan", "planning": "plan",
    "calls": "call", "called": "call", "calling": "call",
    "agrees": "agree", "agreed": "agree", "agreeing": "agree",
    "accepts": "accept", "accepted": "accept", "accepting": "accept",
    "coverage": "cover", "covered": "cover", "covering": "cover",
}


def _sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _normalized(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def _content_tokens(value: Any) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", _normalized(value))
    return {
        _WORD_NORMALIZATION.get(token, token)
        for token in tokens if token not in _STOP_WORDS
    }


def _base_planner_view(value: Mapping[str, Any]) -> dict[str, Any]:
    scenes = []
    for scene in value.get("scenes") or []:
        if isinstance(scene, Mapping):
            scenes.append({key: deepcopy(scene.get(key)) for key in _BASE_SCENE_KEYS})
        else:
            scenes.append(scene)
    return {
        "planner_schema_version": BASE_PLANNER_SCHEMA_VERSION,
        "round_goal": value.get("round_goal"),
        "scenes": scenes,
    }


def validate_initializer_output(
    value: Any, *, parent_turns: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(value, Mapping) or set(value) != _INITIALIZER_TOP_KEYS:
        return {"valid": False, "errors": ["initializer_top_level_schema"],
                "candidate_count": 0, "link_count": 0}
    if value.get("chain_initializer_schema_version") != INITIALIZER_SCHEMA_VERSION:
        errors.append("initializer_schema_version")
    parent_by_ref = {
        str(row.get("turn_id") or ""): row for row in parent_turns
        if str(row.get("turn_id") or "")
    }
    ordered_sessions: list[str] = []
    refs_by_session: dict[str, set[str]] = {}
    for row in parent_turns:
        session_ref = str(row.get("session_id") or "")
        if session_ref not in refs_by_session:
            ordered_sessions.append(session_ref)
            refs_by_session[session_ref] = set()
        refs_by_session[session_ref].add(str(row.get("turn_id") or ""))
    parent_speakers = {
        str(row.get("speaker") or "") for row in parent_turns
        if str(row.get("speaker") or "")
    }
    windows = value.get("session_event_candidates")
    if not isinstance(windows, list):
        errors.append("initializer_session_candidates_schema")
        windows = []
    output_sessions = [
        str(row.get("session_ref") or "") if isinstance(row, Mapping) else ""
        for row in windows
    ]
    if output_sessions != ordered_sessions:
        errors.append("initializer_session_windows_must_match_parent_order")
    candidate_nodes: dict[tuple[str, int], Mapping[str, Any]] = {}
    candidate_count = 0
    for window_index, window in enumerate(windows, 1):
        prefix = f"session_{window_index:02d}"
        if not isinstance(window, Mapping) or set(window) != _SESSION_CANDIDATE_KEYS:
            errors.append(f"{prefix}:schema")
            continue
        session_ref = str(window.get("session_ref") or "")
        events = window.get("events")
        if not isinstance(events, list) or len(events) > 4:
            errors.append(f"{prefix}:event_count_0_to_4")
            events = []
        summaries: set[str] = set()
        for event_index, event in enumerate(events, 1):
            event_prefix = f"{prefix}:event_{event_index:02d}"
            candidate_count += 1
            if not isinstance(event, Mapping) or set(event) != _LOCAL_EVENT_KEYS:
                errors.append(f"{event_prefix}:schema")
                continue
            for key in ("event_summary", "event_state"):
                if not _text(event.get(key)):
                    errors.append(f"{event_prefix}:{key}")
            summary = _normalized(event.get("event_summary"))
            if summary in summaries:
                errors.append(f"{event_prefix}:duplicate_local_event_summary")
            summaries.add(summary)
            participants = event.get("participants")
            if (not isinstance(participants, list) or
                    not 1 <= len(participants) <= 5 or
                    any(not _text(row) for row in participants) or
                    len(participants) != len(set(participants))):
                errors.append(f"{event_prefix}:participants")
            elif any(str(row) not in parent_speakers for row in participants):
                errors.append(f"{event_prefix}:participant_not_parent_speaker")
            refs = event.get("evidence_turn_refs")
            if (not isinstance(refs, list) or not 1 <= len(refs) <= 8 or
                    any(not _text(row) for row in refs) or
                    len(refs) != len(set(refs))):
                errors.append(f"{event_prefix}:evidence_turn_refs")
            elif any(str(ref) not in parent_by_ref for ref in refs):
                errors.append(f"{event_prefix}:evidence_ref_not_parent")
            elif any(str(ref) not in refs_by_session.get(session_ref, set()) for ref in refs):
                errors.append(f"{event_prefix}:evidence_outside_local_session")
            candidate_nodes[(session_ref, event_index)] = event
    if not 3 <= candidate_count <= 12:
        errors.append("initializer_total_candidate_count_3_to_12")
    links = value.get("continuation_links")
    if not isinstance(links, list) or len(links) > candidate_count:
        errors.append("initializer_continuation_links_schema")
        links = []
    incoming: set[tuple[str, int]] = set()
    outgoing: set[tuple[str, int]] = set()
    session_order = {session: index for index, session in enumerate(ordered_sessions)}
    for link_index, link in enumerate(links, 1):
        prefix = f"link_{link_index:02d}"
        if not isinstance(link, Mapping) or set(link) != _CONTINUATION_LINK_KEYS:
            errors.append(f"{prefix}:schema")
            continue
        earlier = (str(link.get("earlier_session_ref") or ""),
                   link.get("earlier_candidate_index"))
        later = (str(link.get("later_session_ref") or ""),
                 link.get("later_candidate_index"))
        if (not isinstance(earlier[1], int) or isinstance(earlier[1], bool) or
                not isinstance(later[1], int) or isinstance(later[1], bool)):
            errors.append(f"{prefix}:candidate_index")
            continue
        if earlier not in candidate_nodes or later not in candidate_nodes:
            errors.append(f"{prefix}:candidate_ref_unknown")
            continue
        earlier_order = (session_order[earlier[0]], int(earlier[1]))
        later_order = (session_order[later[0]], int(later[1]))
        if earlier_order >= later_order:
            errors.append(f"{prefix}:continuation_must_point_forward")
        if earlier in outgoing:
            errors.append(f"{prefix}:earlier_candidate_has_multiple_successors")
        if later in incoming:
            errors.append(f"{prefix}:later_candidate_has_multiple_predecessors")
        outgoing.add(earlier)
        incoming.add(later)
    return {"valid": not errors, "errors": errors,
            "candidate_count": candidate_count, "link_count": len(links)}


def audit_episode_146700_initialization(chain_set: Mapping[str, Any]) -> dict[str, Any]:
    """Canary-specific semantic checkpoints required before spending Planner tokens."""
    chains = list(chain_set.get("chains") or [])
    checkpoints = {
        "soccer": {
            "required_refs": {
                "cc-train-episode-146700:P01:T006",
                "cc-train-episode-146700:P01:T007",
            },
            "topic_refs": {
                "cc-train-episode-146700:P01:T005",
                "cc-train-episode-146700:P01:T006",
                "cc-train-episode-146700:P01:T007",
                "cc-train-episode-146700:P01:T008",
            },
        },
        "health_coverage": {
            "required_refs": {
                "cc-train-episode-146700:P04:T005",
                "cc-train-episode-146700:P04:T006",
            },
            "topic_refs": {
                f"cc-train-episode-146700:P04:T{index:03d}" for index in range(1, 7)
            },
        },
        "report_mistake": {
            "required_refs": {
                "cc-train-episode-146700:P05:T001",
                "cc-train-episode-146700:P05:T003",
                "cc-train-episode-146700:P05:T009",
            },
            "topic_refs": {
                f"cc-train-episode-146700:P05:T{index:03d}" for index in range(1, 10)
            },
        },
    }
    apology_refs = {
        "cc-train-episode-146700:P03:T003",
        "cc-train-episode-146700:P03:T004",
        "cc-train-episode-146700:P03:T005",
        "cc-train-episode-146700:P04:T010",
        "cc-train-episode-146700:P04:T011",
    }
    cashier_refs = {
        "cc-train-episode-146700:P02:T001",
        "cc-train-episode-146700:P02:T003",
    }
    soccer_earlier_refs = {
        "cc-train-episode-146700:P01:T001",
        "cc-train-episode-146700:P01:T003",
    }
    soccer_later_refs = checkpoints["soccer"]["required_refs"]
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    matched: dict[str, int] = {}
    for index, chain in enumerate(chains, 1):
        source_events = list(chain.get("source_events") or [])
        refs = {
            str(ref) for event in source_events
            for ref in event.get("evidence_turn_refs") or []
        }
        state = _normalized(chain.get("current_state"))
        row = {"chain_index": index, "event_summary": chain.get("event_summary"),
               "matched_checkpoints": []}
        for name, rule in checkpoints.items():
            if refs & rule["topic_refs"]:
                row["matched_checkpoints"].append(name)
                matched[name] = index
                missing = sorted(rule["required_refs"] - refs)
                if missing:
                    errors.append(f"chain_{index:02d}:{name}:missing_checkpoint_evidence")
        if refs & checkpoints["report_mistake"]["topic_refs"] and refs & apology_refs:
            errors.append(f"chain_{index:02d}:apology_report_identity_merged")
        if "soccer" in row["matched_checkpoints"]:
            if not ({"agree", "accept"} & _content_tokens(state)):
                errors.append(f"chain_{index:02d}:soccer_acceptance_missing_from_state")
        if "health_coverage" in row["matched_checkpoints"]:
            if "cover" not in _content_tokens(state):
                errors.append(f"chain_{index:02d}:accepted_coverage_missing_from_state")
        if "report_mistake" in row["matched_checkpoints"]:
            completion_claims = (
                "reported the mistake to the boss", "explained the mistake to the boss",
                "informed the boss", "the boss was informed", "resolved the report mistake",
            )
            if any(claim in state for claim in completion_claims):
                errors.append(f"chain_{index:02d}:report_execution_overstated")
        rows.append(row)
    def matching_chain(required: set[str]) -> int | None:
        for index, chain in enumerate(chains, 1):
            refs = {
                str(ref) for event in chain.get("source_events") or []
                for ref in event.get("evidence_turn_refs") or []
            }
            if refs & required:
                return index
        return None

    soccer_chain = matching_chain(soccer_earlier_refs | soccer_later_refs)
    cashier_chain = matching_chain(cashier_refs)
    apology_chain = matching_chain(apology_refs)
    health_chain = matched.get("health_coverage")
    report_chain = matched.get("report_mistake")
    named_matches = {
        "soccer": soccer_chain, "new_cashier": cashier_chain,
        "friend_apology": apology_chain, "health_coverage": health_chain,
        "report_mistake": report_chain,
    }
    if soccer_chain is not None:
        soccer = chains[soccer_chain - 1]
        soccer_refs = {
            str(ref) for event in soccer.get("source_events") or []
            for ref in event.get("evidence_turn_refs") or []
        }
        if not (soccer_refs & soccer_earlier_refs and soccer_refs & soccer_later_refs):
            errors.append("soccer_miss_and_acceptance_not_in_same_chain")
        if len(soccer.get("source_event_refs") or []) < 2:
            errors.append("soccer_continuation_link_missing")
    for name, index in named_matches.items():
        if index is None:
            errors.append(f"required_concrete_chain_missing:{name}")
    present_indexes = [index for index in named_matches.values() if index is not None]
    if len(present_indexes) != len(set(present_indexes)):
        errors.append("required_concrete_events_not_independent")
    for name in checkpoints:
        if name not in matched:
            errors.append(f"required_checkpoint_chain_missing:{name}")
    if len(matched.values()) != len(set(matched.values())):
        errors.append("required_checkpoint_chains_not_distinct")
    return {
        "valid": not errors,
        "errors": errors,
        "checkpoint_matches": matched,
        "concrete_event_matches": named_matches,
        "chains": rows,
        "scope": "episode_146700_canary_only",
    }


def materialize_event_chains(
    initializer_output: Mapping[str, Any], *, parent_turns: Sequence[Mapping[str, Any]],
    root_id: str, parent_dialogue_sha256: str,
) -> dict[str, Any]:
    validation = validate_initializer_output(
        initializer_output, parent_turns=parent_turns)
    if not validation["valid"]:
        raise ValueError("event_chain_initializer_invalid:" + ",".join(validation["errors"]))
    parent_by_ref = {str(row["turn_id"]): row for row in parent_turns}
    nodes: dict[tuple[str, int], dict[str, Any]] = {}
    node_order: list[tuple[str, int]] = []
    for window_index, window in enumerate(
            initializer_output["session_event_candidates"], 1):
        session_ref = str(window["session_ref"])
        short_session = f"P{window_index:02d}"
        for event_index, raw in enumerate(window["events"], 1):
            key = (session_ref, event_index)
            refs = [str(ref) for ref in raw["evidence_turn_refs"]]
            nodes[key] = {
                "source_event_ref": f"{short_session}:E{event_index:02d}",
                "session_ref": session_ref,
                "event_summary": str(raw["event_summary"]).strip(),
                "event_state": str(raw["event_state"]).strip(),
                "participants": [str(row) for row in raw["participants"]],
                "evidence_turn_refs": refs,
            }
            node_order.append(key)
    outgoing: dict[tuple[str, int], tuple[str, int]] = {}
    incoming: set[tuple[str, int]] = set()
    for link in initializer_output["continuation_links"]:
        earlier = (str(link["earlier_session_ref"]), int(link["earlier_candidate_index"]))
        later = (str(link["later_session_ref"]), int(link["later_candidate_index"]))
        outgoing[earlier] = later
        incoming.add(later)
    paths: list[list[tuple[str, int]]] = []
    visited: set[tuple[str, int]] = set()
    for root in node_order:
        if root in incoming:
            continue
        path = []
        cursor: tuple[str, int] | None = root
        while cursor is not None and cursor not in visited:
            path.append(cursor)
            visited.add(cursor)
            cursor = outgoing.get(cursor)
        paths.append(path)
    for node in node_order:
        if node not in visited:
            paths.append([node])
    chains = []
    for index, path in enumerate(paths, 1):
        source_events = [deepcopy(nodes[key]) for key in path]
        latest = source_events[-1]
        refs = list(latest["evidence_turn_refs"])
        evidence = [{
            "turn_ref": str(parent_by_ref[ref]["turn_id"]),
            "session_ref": str(parent_by_ref[ref]["session_id"]),
            "speaker": str(parent_by_ref[ref]["speaker"]),
            "text": str(parent_by_ref[ref]["text"]),
        } for ref in refs]
        participants = list(dict.fromkeys(
            participant for event in source_events
            for participant in event["participants"]))
        summary = source_events[0]["event_summary"]
        if len(source_events) > 1:
            summary = f"{summary} -> {source_events[-1]['event_summary']}"
        chains.append({
            "chain_id": f"EC{index:04d}",
            "event_summary": summary,
            "participants": participants,
            "current_state": latest["event_state"],
            "current_evidence_turn_refs": refs,
            "current_evidence_source_turns": evidence,
            "source_event_refs": [row["source_event_ref"] for row in source_events],
            "source_events": source_events,
            "history": [{
                "round_id": "PARENT", "session_ref": row["session_ref"],
                "state": row["event_state"],
                "evidence_turn_refs": list(row["evidence_turn_refs"]),
            } for row in source_events],
        })
    result = {
        "chain_set_version": CHAIN_SET_VERSION,
        "root_id": str(root_id),
        "parent_dialogue_sha256": str(parent_dialogue_sha256),
        "chains": chains,
    }
    result["chain_set_sha256"] = _sha(result)
    return result


def validate_planner_output(
    value: Any, *, chain_set: Mapping[str, Any], parent_speakers: Sequence[str],
) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(value, Mapping) or set(value) != _PLANNER_TOP_KEYS:
        return {"valid": False, "errors": ["planner_top_level_schema"],
                "scene_count": 0, "chain_ids": []}
    if value.get("planner_schema_version") != PLANNER_SCHEMA_VERSION:
        errors.append("planner_schema_version")
    scenes = value.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != 3:
        errors.append("planner_scene_count_exactly_3")
        scenes = []
    available = {
        str(row["chain_id"]): row for row in chain_set.get("chains") or []
        if isinstance(row, Mapping) and row.get("chain_id")
    }
    selected: list[str] = []
    for index, scene in enumerate(scenes, 1):
        prefix = f"scene_{index:02d}"
        if not isinstance(scene, Mapping) or set(scene) != _PLANNER_SCENE_KEYS:
            errors.append(f"{prefix}:schema")
            continue
        chain_id = str(scene.get("chain_id") or "")
        selected.append(chain_id)
        if chain_id not in available:
            errors.append(f"{prefix}:chain_id_unknown")
            continue
        if not _text(scene.get("durable_outcome")):
            errors.append(f"{prefix}:durable_outcome")
        participants = scene.get("participants")
        allowed = set(str(row) for row in available[chain_id]["participants"])
        if isinstance(participants, list) and any(str(row) not in allowed for row in participants):
            errors.append(f"{prefix}:participant_outside_chain")
    if len(selected) != len(set(selected)):
        errors.append("planner_chain_ids_not_unique")
    base = validate_base_planner_output(
        _base_planner_view(value), parent_speakers=parent_speakers)
    errors.extend(error for error in base["errors"] if error not in errors)
    return {"valid": not errors, "errors": errors,
            "scene_count": len(scenes), "chain_ids": selected}


def _obvious_repetition(current_state: str, proposed: str) -> bool:
    current = _normalized(current_state)
    candidate = _normalized(proposed)
    if not current or not candidate:
        return False
    if current == candidate or candidate in current:
        return True
    current_tokens = _content_tokens(current_state)
    candidate_tokens = _content_tokens(proposed)
    if len(candidate_tokens) < 4:
        return False
    return len(current_tokens & candidate_tokens) / len(candidate_tokens) >= 0.85


def validate_planner_continuation(
    value: Mapping[str, Any], *, chain_set: Mapping[str, Any],
) -> dict[str, Any]:
    available = {str(row["chain_id"]): row for row in chain_set.get("chains") or []}
    errors: list[str] = []
    checks = []
    for index, scene in enumerate(value.get("scenes") or [], 1):
        if not isinstance(scene, Mapping):
            continue
        chain_id = str(scene.get("chain_id") or "")
        chain = available.get(chain_id)
        if chain is None:
            continue
        repeated = [
            field for field in ("new_development", "durable_outcome")
            if _obvious_repetition(str(chain["current_state"]), str(scene.get(field) or ""))
        ]
        checks.append({
            "scene_index": index, "chain_id": chain_id,
            "current_state": chain["current_state"],
            "new_development": scene.get("new_development"),
            "durable_outcome": scene.get("durable_outcome"),
            "obvious_repetition_fields": repeated,
        })
        errors.extend(
            f"scene_{index:02d}:{field}:obvious_current_state_repetition"
            for field in repeated
        )
    return {"valid": not errors, "errors": errors, "checks": checks,
            "gate_scope": "normalized_exact_and_high_containment_only"}


def compile_round_plan(
    planner_output: Mapping[str, Any], *, chain_set: Mapping[str, Any],
    tail_context: Mapping[str, Any], round_index: int = 1,
) -> dict[str, Any]:
    validation = validate_planner_output(
        planner_output, chain_set=chain_set,
        parent_speakers=tail_context["parent_speakers"])
    if not validation["valid"]:
        raise ValueError("event_chain_planner_invalid:" + ",".join(validation["errors"]))
    continuation = validate_planner_continuation(planner_output, chain_set=chain_set)
    if not continuation["valid"]:
        raise ValueError("event_chain_planner_repetition:" + ",".join(continuation["errors"]))
    compiled = compile_base_round_plan(
        _base_planner_view(planner_output), tail_context=tail_context,
        round_index=round_index)
    available = {str(row["chain_id"]): row for row in chain_set["chains"]}
    for target, source in zip(compiled["scenes"], planner_output["scenes"]):
        chain_id = str(source["chain_id"])
        target["chain_id"] = chain_id
        target["event_chain"] = deepcopy(available[chain_id])
        target["durable_outcome"] = str(source["durable_outcome"]).strip()
    compiled["compiled_plan_version"] = COMPILED_PLAN_VERSION
    compiled["incremental_event_chain_contract"] = {
        "chain_set_sha256": str(chain_set["chain_set_sha256"]),
        "chain_ids_unique": True,
        "current_states_frozen": True,
        "model_generated_chain_ids": False,
        "anchor_contract_used": False,
    }
    compiled.pop("compiled_plan_sha256", None)
    compiled["compiled_plan_sha256"] = _sha(compiled)
    return compiled


def _child_turn_maps(child: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], dict[str, str]]:
    turns: dict[str, Mapping[str, Any]] = {}
    turn_scenes: dict[str, str] = {}
    for scene in child.get("scenes") or []:
        scene_ref = str(scene.get("scene_ref") or "")
        for turn in scene.get("dialogue_turns") or []:
            ref = str(turn.get("turn_ref") or "")
            if ref:
                turns[ref] = turn
                turn_scenes[ref] = scene_ref
    return turns, turn_scenes


def validate_delta_updater_output(
    value: Any, *, chain_set: Mapping[str, Any], compiled_plan: Mapping[str, Any],
    child: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(value, Mapping) or set(value) != _DELTA_TOP_KEYS:
        return {"valid": False, "errors": ["delta_top_level_schema"],
                "update_count": 0, "updated_count": 0}
    if value.get("delta_updater_schema_version") != DELTA_UPDATER_SCHEMA_VERSION:
        errors.append("delta_schema_version")
    available = {str(row["chain_id"]): row for row in chain_set.get("chains") or []}
    scene_by_chain = {
        str(scene["chain_id"]): str(scene["scene_ref"])
        for scene in compiled_plan.get("scenes") or []
    }
    turns, turn_scenes = _child_turn_maps(child)
    updates = value.get("chain_updates")
    if not isinstance(updates, list) or len(updates) != len(scene_by_chain):
        errors.append("delta_exactly_one_update_per_planned_chain")
        updates = []
    seen: set[str] = set()
    updated_count = 0
    for index, update in enumerate(updates, 1):
        prefix = f"update_{index:02d}"
        if not isinstance(update, Mapping) or set(update) != _DELTA_UPDATE_KEYS:
            errors.append(f"{prefix}:schema")
            continue
        chain_id = str(update.get("chain_id") or "")
        if chain_id not in available:
            errors.append(f"{prefix}:chain_id_unknown")
        if chain_id not in scene_by_chain:
            errors.append(f"{prefix}:chain_not_planned")
        if chain_id in seen:
            errors.append(f"{prefix}:duplicate_chain_id")
        seen.add(chain_id)
        status = update.get("update_status")
        refs = update.get("evidence_turn_refs")
        if status not in {"updated", "unchanged"}:
            errors.append(f"{prefix}:update_status")
            continue
        if status == "updated":
            updated_count += 1
            if not _text(update.get("new_current_state")):
                errors.append(f"{prefix}:new_current_state")
            if (not isinstance(refs, list) or not 1 <= len(refs) <= 8 or
                    any(not _text(ref) for ref in refs) or len(refs) != len(set(refs))):
                errors.append(f"{prefix}:evidence_turn_refs")
            else:
                for ref in refs:
                    ref = str(ref)
                    if ref not in turns:
                        errors.append(f"{prefix}:evidence_not_current_child")
                    elif turn_scenes[ref] != scene_by_chain.get(chain_id):
                        errors.append(f"{prefix}:evidence_outside_mapped_scene")
        else:
            if update.get("new_current_state") is not None:
                errors.append(f"{prefix}:unchanged_state_must_be_null")
            if refs != []:
                errors.append(f"{prefix}:unchanged_evidence_must_be_empty")
    if seen != set(scene_by_chain):
        errors.append("delta_planned_chain_coverage")
    candidates = value.get("untracked_new_event_candidates")
    if not isinstance(candidates, list) or len(candidates) > 5:
        errors.append("untracked_candidates_schema")
        candidates = []
    for index, candidate in enumerate(candidates, 1):
        prefix = f"untracked_{index:02d}"
        if not isinstance(candidate, Mapping) or set(candidate) != _UNTRACKED_KEYS:
            errors.append(f"{prefix}:schema")
            continue
        if not _text(candidate.get("event_summary")):
            errors.append(f"{prefix}:event_summary")
        refs = candidate.get("evidence_turn_refs")
        if (not isinstance(refs, list) or not 1 <= len(refs) <= 8 or
                any(str(ref) not in turns for ref in refs) or
                len(refs) != len(set(refs))):
            errors.append(f"{prefix}:evidence_not_current_child")
    return {"valid": not errors, "errors": errors,
            "update_count": len(updates), "updated_count": updated_count}


def apply_delta_updates(
    delta_output: Mapping[str, Any], *, chain_set: Mapping[str, Any],
    compiled_plan: Mapping[str, Any], child: Mapping[str, Any], round_id: str = "R1",
) -> dict[str, Any]:
    validation = validate_delta_updater_output(
        delta_output, chain_set=chain_set, compiled_plan=compiled_plan, child=child)
    if not validation["valid"]:
        raise ValueError("event_chain_delta_invalid:" + ",".join(validation["errors"]))
    updates = {str(row["chain_id"]): row for row in delta_output["chain_updates"]}
    chains = deepcopy(list(chain_set["chains"]))
    for chain in chains:
        update = updates.get(str(chain["chain_id"]))
        if update is None or update["update_status"] == "unchanged":
            continue
        state = str(update["new_current_state"]).strip()
        refs = [str(ref) for ref in update["evidence_turn_refs"]]
        chain["current_state"] = state
        chain["current_evidence_turn_refs"] = refs
        chain["current_evidence_source_turns"] = []
        chain["history"].append({
            "round_id": str(round_id), "state": state,
            "evidence_turn_refs": refs,
        })
    result = {
        "chain_set_version": UPDATED_CHAIN_SET_VERSION,
        "root_id": chain_set["root_id"],
        "parent_dialogue_sha256": chain_set["parent_dialogue_sha256"],
        "previous_chain_set_sha256": chain_set["chain_set_sha256"],
        "chains": chains,
        "untracked_new_event_candidates": deepcopy(
            delta_output["untracked_new_event_candidates"]),
    }
    result["chain_set_sha256"] = _sha(result)
    return result


__all__ = [
    "CHAIN_SET_VERSION", "COMPILED_PLAN_VERSION", "DELTA_UPDATER_SCHEMA_VERSION",
    "INITIALIZER_SCHEMA_VERSION", "PLANNER_SCHEMA_VERSION", "UPDATED_CHAIN_SET_VERSION",
    "apply_delta_updates", "audit_episode_146700_initialization", "compile_round_plan",
    "materialize_event_chains", "normalize_round_realizer_output",
    "validate_delta_updater_output", "validate_initializer_output",
    "validate_planner_continuation", "validate_planner_output",
]
