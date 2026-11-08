#!/usr/bin/env python3
"""Prepare an adjudication increment, accounting for source exclusions in preflight."""
from __future__ import annotations

import sys

import adapter
import prepare_adjudicated_increment as increment


def read_with_source_exclusions(path):
    value = increment_original_read(path)
    if path.name == "PREFLIGHT.json":
        parent = path.parent.parent
        quality = {row["candidate_id"]: row for row in increment_original_read(parent / "QUALITY_FINAL.json")["items"]}
        source = {row["candidate_id"]: row for row in increment_original_read(parent / "SOURCE_QUALIFICATION.json")["items"]}
        items = increment_original_read(parent / "ITEMS_PRIVATE.json")
        solver_items = increment_original_read(path.parent / "ITEMS_PRIVATE.json")
        expected = {row["candidate_id"] for row in items
                    if quality[row["candidate_id"]]["content_pass"] and source[row["candidate_id"]]["source_qualified"]}
        assert expected == {row["candidate_id"] for row in solver_items}, "PARENT_GATE_SELECTION_MISMATCH"
        assert value["candidate_count"] == len(expected)
        assert value["all_candidate_count"] == len(items)
        value = dict(value)
        value["excluded_candidate_ids"] = sorted({row["candidate_id"] for row in items} - expected)
    return value


if __name__ == "__main__":
    increment_original_read = increment.read
    increment.read = read_with_source_exclusions
    adapter.reserve_budget = increment.reserve_adjudicated_increment
    adapter.main()
