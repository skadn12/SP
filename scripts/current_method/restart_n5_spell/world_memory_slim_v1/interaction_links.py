"""Deterministic interaction links; co-presence is not a social relation."""

from __future__ import annotations

import hashlib
from collections import defaultdict, deque
from typing import Any, Mapping, Sequence


INTERACTION_LINK_SCHEMA_VERSION = "interaction_links_v1"


def _pair(left: str, right: str) -> tuple[str, str]:
    if not left or not right or left == right:
        raise ValueError("interaction_pair_invalid")
    return tuple(sorted((left, right)))  # type: ignore[return-value]


def _link_id(root_id: str, pair: tuple[str, str]) -> str:
    digest = hashlib.sha256((root_id + "\0" + "\0".join(pair)).encode("utf-8")).hexdigest()[:16]
    return f"{root_id}:IL:{digest}"


def compile_source_interaction_links(
    *, root_id: str, sessions: Sequence[Mapping[str, Any]],
    speaker_to_entity: Mapping[str, str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: {"sessions": set(), "turns": set()})
    if any(not str(ref).startswith(root_id + ":") for ref in speaker_to_entity.values()):
        raise ValueError("interaction_cross_root_entity_ref")
    for session in sessions:
        session_ref = str(session.get("session_id") or session.get("session_ref") or "")
        turns = list(session.get("turns") or [])
        for left, right in zip(turns, turns[1:]):
            left_speaker = str(left.get("speaker") or "")
            right_speaker = str(right.get("speaker") or "")
            if left_speaker == right_speaker:
                continue
            if left_speaker not in speaker_to_entity or right_speaker not in speaker_to_entity:
                raise ValueError("interaction_speaker_entity_missing")
            pair = _pair(speaker_to_entity[left_speaker], speaker_to_entity[right_speaker])
            grouped[pair]["sessions"].add(session_ref)
            grouped[pair]["turns"].update({str(left.get("turn_id") or left.get("turn_ref") or ""),
                                            str(right.get("turn_id") or right.get("turn_ref") or "")})
    return [{
        "interaction_link_id": _link_id(root_id, pair),
        "root_id": root_id, "entity_refs": list(pair),
        "source_session_refs": sorted(value for value in refs["sessions"] if value),
        "source_turn_refs": sorted(value for value in refs["turns"] if value),
        "origin": "source_grounded", "verification_status": "source_grounded",
    } for pair, refs in sorted(grouped.items())]


def compile_generated_interaction_links(
    *, root_id: str, scene_ref: str, speaker_refs: Sequence[str],
    committed: bool, promoted: bool, authoritative_entity_refs: set[str],
) -> list[dict[str, Any]]:
    if not committed or not promoted:
        return []
    if any(not str(ref).startswith(root_id + ":") for ref in speaker_refs):
        raise ValueError("generated_interaction_cross_root_entity_ref")
    if any(ref not in authoritative_entity_refs for ref in speaker_refs):
        raise ValueError("generated_interaction_non_authoritative_speaker")
    pairs = sorted({_pair(left, right) for left, right in zip(speaker_refs, speaker_refs[1:])
                    if left != right})
    return [{
        "interaction_link_id": _link_id(root_id, pair), "root_id": root_id,
        "entity_refs": list(pair), "source_session_refs": [scene_ref],
        "source_turn_refs": [], "origin": "generated_extension",
        "verification_status": "semantically_verified",
    } for pair in pairs]


def validate_scene_connectivity(
    *, primary_refs: Sequence[str], supporting_refs: Sequence[str],
    relation_edges: Sequence[Sequence[str]] = (),
    interaction_links: Sequence[Mapping[str, Any]] = (),
    shared_memory_groups: Sequence[Sequence[str]] = (),
    allows_connectivity_bridge: bool = False,
) -> dict[str, Any]:
    nodes = set(primary_refs) | set(supporting_refs)
    graph: dict[str, set[str]] = defaultdict(set)
    edges = [tuple(edge) for edge in relation_edges]
    edges.extend(tuple(link.get("entity_refs") or []) for link in interaction_links)
    for group in shared_memory_groups:
        group = list(group)
        edges.extend((group[index], group[index + 1]) for index in range(len(group) - 1))
    for edge in edges:
        if len(edge) != 2:
            continue
        left, right = edge
        if left in nodes and right in nodes:
            graph[left].add(right)
            graph[right].add(left)
    if allows_connectivity_bridge:
        return {"valid": True, "errors": [], "bridge_used": True}
    errors: list[str] = []
    if primary_refs:
        seen = {primary_refs[0]}
        queue = deque([primary_refs[0]])
        while queue:
            current = queue.popleft()
            for neighbor in graph[current] - seen:
                seen.add(neighbor)
                queue.append(neighbor)
        if any(ref not in seen for ref in primary_refs):
            errors.append("primary_speakers_disconnected")
        if any(not (graph[ref] & set(primary_refs)) for ref in supporting_refs):
            errors.append("supporting_speaker_not_connected_to_primary")
    return {"valid": not errors, "errors": errors, "bridge_used": False}
