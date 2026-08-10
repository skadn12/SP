"""Deterministic cross-window fact candidate bundling."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from itertools import combinations
from typing import Any, Mapping, Sequence


BUNDLER_VERSION = "cross_window_fact_bundler_v2"
MAX_BUNDLES_EPISODE = 60
MIN_BUNDLE_SIZE = 2
MAX_BUNDLE_SIZE = 4
MAX_FACTS_PER_WINDOW = 2


def _subject_key(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _window_kind(row: Mapping[str, Any]) -> str:
    kind = str(row.get("window_kind") or "")
    if kind == "parent_session" or str(row.get("window_id") or "").startswith("P"):
        return "parent"
    if kind == "child_scene" or str(row.get("window_id") or "").startswith("C"):
        return "child"
    return "unknown"


def _bundle_class(combo: Sequence[Mapping[str, Any]]) -> str:
    kinds = {_window_kind(row) for row in combo}
    if "parent" in kinds and "child" in kinds:
        return "parent_child"
    if kinds == {"child"}:
        return "child_child"
    if kinds == {"parent"}:
        return "parent_parent"
    return "unknown"


def _ordered_windows(combo: Sequence[Mapping[str, Any]]) -> list[tuple[int, str]]:
    windows = {
        (int(row.get("window_order") or 0), str(row.get("window_id") or ""))
        for row in combo
    }
    return sorted(windows)


def _coverage_pair(combo: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    """Return the farthest window pair represented by a combination."""
    windows = _ordered_windows(combo)
    return windows[0][1], windows[-1][1]


def validate_bundle(
    bundle_facts: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Validate only the mechanical bundle contract."""
    errors: list[str] = []
    size = len(bundle_facts)
    if size < MIN_BUNDLE_SIZE:
        errors.append("bundle_size_below_minimum")
    if size > MAX_BUNDLE_SIZE:
        errors.append("bundle_size_above_maximum")
    subjects = {_subject_key(row.get("subject")) for row in bundle_facts}
    if "" in subjects or len(subjects) != 1:
        errors.append("bundle_subject_mismatch")
    window_counts = Counter(str(row.get("window_id") or "") for row in bundle_facts)
    if "" in window_counts or len(window_counts) < 2:
        errors.append("bundle_requires_two_windows")
    if any(count > MAX_FACTS_PER_WINDOW for count in window_counts.values()):
        errors.append("bundle_window_fact_limit_exceeded")
    return errors


def _candidate(
    combo: tuple[Mapping[str, Any], ...],
    subject_key: str,
) -> dict[str, Any]:
    windows = _ordered_windows(combo)
    bundle_class = _bundle_class(combo)
    class_rank = {"parent_child": 0, "child_child": 1, "parent_parent": 2}.get(
        bundle_class, 3
    )
    fact_ids = tuple(sorted(str(row.get("fact_id")) for row in combo))
    distinct_window_count = len(windows)
    window_span = windows[-1][0] - windows[0][0]
    pair = _coverage_pair(combo)
    return {
        "combo": combo,
        "subject_key": subject_key,
        "coverage_pair": pair,
        "bundle_class": bundle_class,
        "distinct_window_count": distinct_window_count,
        "window_order_span": window_span,
        "fact_ids": fact_ids,
        "rank": (
            class_rank,
            -distinct_window_count,
            -window_span,
            fact_ids,
        ),
    }


def _round_robin_select(
    candidates: Sequence[Mapping[str, Any]],
    limit: int,
) -> list[Mapping[str, Any]]:
    """Rotate subjects and their farthest represented window pairs."""
    by_subject_pair: dict[str, dict[tuple[str, str], deque[Mapping[str, Any]]]] = defaultdict(dict)
    grouped: dict[tuple[str, tuple[str, str]], list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[(candidate["subject_key"], candidate["coverage_pair"])].append(candidate)
    for (subject, pair), rows in grouped.items():
        rows.sort(key=lambda row: row["rank"])
        by_subject_pair[subject][pair] = deque(rows)

    subjects = sorted(by_subject_pair)
    pair_orders: dict[str, list[tuple[str, str]]] = {}
    pair_offsets: dict[str, int] = {}
    for subject in subjects:
        pair_orders[subject] = sorted(
            by_subject_pair[subject],
            key=lambda pair: (by_subject_pair[subject][pair][0]["rank"], pair),
        )
        pair_offsets[subject] = 0

    selected: list[Mapping[str, Any]] = []
    while len(selected) < limit:
        progress = False
        for subject in subjects:
            pairs = pair_orders[subject]
            if not pairs:
                continue
            for _ in range(len(pairs)):
                index = pair_offsets[subject] % len(pairs)
                pair_offsets[subject] += 1
                pair = pairs[index]
                queue = by_subject_pair[subject][pair]
                if queue:
                    selected.append(queue.popleft())
                    progress = True
                    break
            if len(selected) >= limit:
                break
        if not progress:
            break
    return selected


def bundle_facts(facts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for fact in facts:
        key = _subject_key(fact.get("subject"))
        if key:
            groups.setdefault(key, []).append(fact)

    candidates: list[dict[str, Any]] = []
    subject_stats: list[dict[str, Any]] = []
    for subject_key in sorted(groups):
        all_subject_facts = sorted(groups[subject_key], key=lambda row: str(row.get("fact_id")))
        subject_candidates: list[dict[str, Any]] = []
        for size in range(MIN_BUNDLE_SIZE, MAX_BUNDLE_SIZE + 1):
            for combo in combinations(all_subject_facts, size):
                if validate_bundle(combo):
                    continue
                subject_candidates.append(_candidate(combo, subject_key))
        candidates.extend(subject_candidates)
        subject_stats.append({
            "subject": subject_key,
            "input_fact_count": len(all_subject_facts),
            "considered_fact_count": len(all_subject_facts),
            "candidate_bundle_count": len(subject_candidates),
        })

    candidates.sort(key=lambda row: (row["rank"], row["subject_key"], row["coverage_pair"]))
    chosen = _round_robin_select(candidates, MAX_BUNDLES_EPISODE)

    bundles: list[dict[str, Any]] = []
    selected_by_subject = Counter()
    all_class_counts = Counter(row["bundle_class"] for row in candidates)
    selected_class_counts = Counter(row["bundle_class"] for row in chosen)
    for index, candidate in enumerate(chosen, start=1):
        rows = sorted(candidate["combo"], key=lambda row: str(row.get("fact_id")))
        selected_by_subject[candidate["subject_key"]] += 1
        bundles.append({
            "bundle_id": f"B{index:04d}",
            "subject": rows[0].get("subject"),
            "fact_ids": [row["fact_id"] for row in rows],
            "window_ids": [window_id for _, window_id in _ordered_windows(rows)],
            "coverage_window_pair": list(candidate["coverage_pair"]),
            "bundle_size": len(rows),
            "distinct_window_count": candidate["distinct_window_count"],
            "window_order_span": candidate["window_order_span"],
            "bundle_class": candidate["bundle_class"],
            "bundler_provenance": BUNDLER_VERSION,
        })

    for stat in subject_stats:
        stat["emitted_bundle_count"] = selected_by_subject[stat["subject"]]
    return {
        "bundler_version": BUNDLER_VERSION,
        "input_fact_count": len(facts),
        "subject_count": len(groups),
        "all_mechanically_eligible_bundle_count": len(candidates),
        "bundle_count": len(bundles),
        "limits": {
            "max_bundles_episode": MAX_BUNDLES_EPISODE,
            "min_bundle_size": MIN_BUNDLE_SIZE,
            "max_bundle_size": MAX_BUNDLE_SIZE,
            "max_facts_per_window": MAX_FACTS_PER_WINDOW,
            "pre_combination_fact_cap": None,
        },
        "all_candidate_class_counts": dict(sorted(all_class_counts.items())),
        "materialized_class_counts": dict(sorted(selected_class_counts.items())),
        "subject_stats": subject_stats,
        "bundles": bundles,
    }
