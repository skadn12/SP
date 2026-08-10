"""Deterministic participant-connected bundling for frozen atomic facts."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from itertools import combinations
import re
from typing import Any, Mapping, Sequence


PARTICIPANT_BUNDLER_VERSION = "participant_connected_fact_bundler_v1"
MAX_BUNDLES_EPISODE = 60
MIN_BUNDLE_SIZE = 2
MAX_BUNDLE_SIZE = 4
MAX_FACTS_PER_WINDOW = 2


def known_dialogue_participants(expanded: Mapping[str, Any]) -> list[str]:
    """Return speakers in deterministic first-appearance order."""
    result: list[str] = []
    seen: set[str] = set()
    for turn in expanded.get("turns") or []:
        speaker = str(turn.get("speaker") or "").strip()
        key = speaker.casefold()
        if speaker and key not in seen:
            seen.add(key)
            result.append(speaker)
    return result


def _explicit_name_in_text(name: str, text: str) -> bool:
    pattern = rf"(?<![\w]){re.escape(name)}(?![\w])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def materialize_fact_participants(
    facts: Sequence[Mapping[str, Any]],
    *,
    expanded: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Attach explicit participants without pronoun or semantic inference."""
    known = known_dialogue_participants(expanded)
    turns = list(expanded.get("turns") or [])
    result: list[dict[str, Any]] = []
    for raw in facts:
        row = dict(raw)
        subject = str(row.get("subject") or "").strip()
        source_fragments: list[str] = []
        for value in row.get("source_positions") or []:
            if isinstance(value, int) and 1 <= value <= len(turns):
                turn = turns[value - 1]
                source_fragments.extend([
                    str(turn.get("speaker") or ""),
                    str(turn.get("text") or ""),
                ])
        searchable = "\n".join([
            str(row.get("fact_text") or ""),
            *source_fragments,
        ])
        matched = {
            name for name in known if _explicit_name_in_text(name, searchable)
        }
        if subject:
            matched.add(subject)
        row["participants"] = sorted(matched, key=lambda value: (value.casefold(), value))
        row["participant_extraction"] = {
            "method": "exact_known_name_match_fact_and_source_turn_v1",
            "subject_forced": True,
            "pronoun_resolution": False,
            "semantic_inference": False,
        }
        result.append(row)
    return result


def _participant_set(row: Mapping[str, Any]) -> set[str]:
    return {
        str(value).strip() for value in row.get("participants") or []
        if str(value).strip()
    }


def participant_graph_edges(
    facts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for left, right in combinations(facts, 2):
        shared = sorted(_participant_set(left) & _participant_set(right), key=str.casefold)
        if shared:
            edges.append({
                "left_fact_id": str(left.get("fact_id") or ""),
                "right_fact_id": str(right.get("fact_id") or ""),
                "shared_participants": shared,
            })
    return edges


def participant_graph_connected(facts: Sequence[Mapping[str, Any]]) -> bool:
    if not facts:
        return False
    adjacency: dict[int, set[int]] = {index: set() for index in range(len(facts))}
    for left_index, right_index in combinations(range(len(facts)), 2):
        if _participant_set(facts[left_index]) & _participant_set(facts[right_index]):
            adjacency[left_index].add(right_index)
            adjacency[right_index].add(left_index)
    visited = {0}
    queue = deque([0])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return len(visited) == len(facts)


def validate_participant_bundle(facts: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    if len(facts) < MIN_BUNDLE_SIZE:
        errors.append("bundle_size_below_minimum")
    if len(facts) > MAX_BUNDLE_SIZE:
        errors.append("bundle_size_above_maximum")
    window_counts = Counter(str(row.get("window_id") or "") for row in facts)
    if "" in window_counts or len(window_counts) < 2:
        errors.append("bundle_requires_two_windows")
    if any(count > MAX_FACTS_PER_WINDOW for count in window_counts.values()):
        errors.append("bundle_window_fact_limit_exceeded")
    if any(not _participant_set(row) for row in facts):
        errors.append("fact_participants_empty")
    elif not participant_graph_connected(facts):
        errors.append("participant_graph_disconnected")
    return errors


def _window_kind(row: Mapping[str, Any]) -> str:
    window_id = str(row.get("window_id") or "")
    kind = str(row.get("window_kind") or "")
    if kind == "parent_session" or window_id.startswith("P"):
        return "parent"
    if kind == "child_scene" or window_id.startswith("C"):
        return "child"
    return "unknown"


def _ordered_windows(facts: Sequence[Mapping[str, Any]]) -> list[tuple[int, str]]:
    return sorted({
        (int(row.get("window_order") or 0), str(row.get("window_id") or ""))
        for row in facts
    })


def _bundle_class(facts: Sequence[Mapping[str, Any]]) -> str:
    kinds = {_window_kind(row) for row in facts}
    if "parent" in kinds and "child" in kinds:
        return "parent_child"
    if kinds == {"child"}:
        return "child_child"
    if kinds == {"parent"}:
        return "parent_parent"
    return "unknown"


def _candidate(combo: tuple[Mapping[str, Any], ...]) -> dict[str, Any]:
    windows = _ordered_windows(combo)
    subjects = tuple(sorted({str(row.get("subject") or "") for row in combo}, key=str.casefold))
    participants = tuple(sorted({
        participant for row in combo for participant in _participant_set(row)
    }, key=str.casefold))
    bundle_class = _bundle_class(combo)
    connection_pattern = "same_subject" if len(subjects) == 1 else "cross_subject"
    selection_class = (
        "parent_child_cross_subject"
        if bundle_class == "parent_child" and connection_pattern == "cross_subject"
        else "parent_child_same_subject"
        if bundle_class == "parent_child"
        else bundle_class
    )
    class_rank = {
        "parent_child_cross_subject": 0,
        "parent_child_same_subject": 1,
        "child_child": 2,
        "parent_parent": 3,
    }.get(selection_class, 4)
    fact_ids = tuple(sorted(str(row.get("fact_id") or "") for row in combo))
    window_ids = tuple(window_id for _, window_id in windows)
    coverage_pair = (windows[0][1], windows[-1][1])
    span = windows[-1][0] - windows[0][0]
    return {
        "combo": combo,
        "fact_ids": fact_ids,
        "subjects": subjects,
        "participants": participants,
        "bundle_class": bundle_class,
        "connection_pattern": connection_pattern,
        "selection_class": selection_class,
        "window_ids": window_ids,
        "coverage_pair": coverage_pair,
        "distinct_window_count": len(windows),
        "window_order_span": span,
        "bucket_key": (
            connection_pattern,
            subjects,
            participants,
            coverage_pair,
            window_ids,
        ),
        "rank": (
            class_rank,
            -len(windows),
            -span,
            subjects,
            participants,
            window_ids,
            fact_ids,
        ),
    }


def _bucket_queues(candidates: Sequence[Mapping[str, Any]]) -> dict[str, deque[Mapping[str, Any]]]:
    by_class_bucket: dict[str, dict[tuple[Any, ...], list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for candidate in candidates:
        by_class_bucket[str(candidate["selection_class"])][candidate["bucket_key"]].append(candidate)
    result: dict[str, deque[Mapping[str, Any]]] = {}
    for selection_class, buckets in by_class_bucket.items():
        ordered_buckets: list[deque[Mapping[str, Any]]] = []
        for key in sorted(buckets, key=str):
            rows = sorted(buckets[key], key=lambda row: row["rank"])
            ordered_buckets.append(deque(rows))
        interleaved: deque[Mapping[str, Any]] = deque()
        while any(ordered_buckets):
            for bucket in ordered_buckets:
                if bucket:
                    interleaved.append(bucket.popleft())
        result[selection_class] = interleaved
    return result


def _select(candidates: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    queues = _bucket_queues(candidates)
    # Cross-subject Parent-Child is preferred, but cannot occupy every slot.
    schedule = (
        "parent_child_cross_subject",
        "parent_child_same_subject",
        "parent_child_cross_subject",
        "child_child",
        "parent_parent",
    )
    selected: list[Mapping[str, Any]] = []
    while len(selected) < MAX_BUNDLES_EPISODE:
        progress = False
        for selection_class in schedule:
            queue = queues.get(selection_class)
            if queue:
                selected.append(queue.popleft())
                progress = True
                if len(selected) >= MAX_BUNDLES_EPISODE:
                    break
        if not progress:
            remaining = sorted(
                (row for queue in queues.values() for row in queue),
                key=lambda row: row["rank"],
            )
            selected.extend(remaining[: MAX_BUNDLES_EPISODE - len(selected)])
            break
    return selected


def bundle_participant_connected_facts(
    facts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(facts, key=lambda row: str(row.get("fact_id") or ""))
    candidates: list[dict[str, Any]] = []
    seen_fact_sets: set[tuple[str, ...]] = set()
    exact_duplicate_fact_set_count = 0
    for size in range(MIN_BUNDLE_SIZE, MAX_BUNDLE_SIZE + 1):
        for combo in combinations(ordered, size):
            if validate_participant_bundle(combo):
                continue
            fact_set = tuple(sorted(str(row.get("fact_id") or "") for row in combo))
            if fact_set in seen_fact_sets:
                exact_duplicate_fact_set_count += 1
                continue
            seen_fact_sets.add(fact_set)
            candidates.append(_candidate(combo))
    candidates.sort(key=lambda row: row["rank"])
    selected = _select(candidates)
    all_classes = Counter(str(row["selection_class"]) for row in candidates)
    selected_classes = Counter(str(row["selection_class"]) for row in selected)
    all_bundle_classes = Counter(str(row["bundle_class"]) for row in candidates)
    selected_bundle_classes = Counter(str(row["bundle_class"]) for row in selected)
    all_candidate_window_counts = Counter(
        window_id for row in candidates for window_id in row["window_ids"]
    )
    bundles: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected, 1):
        combo = sorted(candidate["combo"], key=lambda row: str(row.get("fact_id") or ""))
        bundles.append({
            "bundle_id": f"PB{index:04d}",
            "fact_ids": list(candidate["fact_ids"]),
            "subjects": list(candidate["subjects"]),
            "participants": list(candidate["participants"]),
            "connection_pattern": candidate["connection_pattern"],
            "selection_class": candidate["selection_class"],
            "participant_graph_edges": participant_graph_edges(combo),
            "window_ids": list(candidate["window_ids"]),
            "coverage_window_pair": list(candidate["coverage_pair"]),
            "bundle_size": len(combo),
            "distinct_window_count": candidate["distinct_window_count"],
            "window_order_span": candidate["window_order_span"],
            "bundle_class": candidate["bundle_class"],
            "bundler_provenance": PARTICIPANT_BUNDLER_VERSION,
        })
    return {
        "bundler_version": PARTICIPANT_BUNDLER_VERSION,
        "input_fact_count": len(ordered),
        "all_mechanically_eligible_bundle_count": len(candidates),
        "same_subject_eligible_combination_count": sum(
            1 for row in candidates if row["connection_pattern"] == "same_subject"
        ),
        "cross_subject_eligible_combination_count": sum(
            1 for row in candidates if row["connection_pattern"] == "cross_subject"
        ),
        "exact_duplicate_fact_set_count": exact_duplicate_fact_set_count,
        "all_selection_class_counts": dict(sorted(all_classes.items())),
        "materialized_selection_class_counts": dict(sorted(selected_classes.items())),
        "all_bundle_class_counts": dict(sorted(all_bundle_classes.items())),
        "materialized_bundle_class_counts": dict(sorted(selected_bundle_classes.items())),
        "all_candidate_bundle_count_by_window": dict(sorted(all_candidate_window_counts.items())),
        "bundle_count": len(bundles),
        "limits": {
            "max_bundles_episode": MAX_BUNDLES_EPISODE,
            "min_bundle_size": MIN_BUNDLE_SIZE,
            "max_bundle_size": MAX_BUNDLE_SIZE,
            "max_facts_per_window": MAX_FACTS_PER_WINDOW,
            "pre_combination_fact_cap": None,
        },
        "bundles": bundles,
    }
