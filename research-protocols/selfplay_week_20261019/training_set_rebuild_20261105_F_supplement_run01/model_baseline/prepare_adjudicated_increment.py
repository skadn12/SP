#!/usr/bin/env python3
"""Reserve and prepare a documented adjudication increment without retrying calls."""
from __future__ import annotations

import fcntl
import json
import math
import os
from pathlib import Path

import adapter


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def reserve_adjudicated_increment(batch_dir: Path) -> None:
    batch_dir = Path(batch_dir).resolve()
    adjudication = read(batch_dir / "ADJUDICATION.json")
    assert adjudication["kind"] == "post_adjudication_increment"
    pool = read(batch_dir / "ITEMS_PRIVATE.json")
    quality = {row["candidate_id"]: row for row in read(batch_dir / "QUALITY_FINAL.json")["items"]}
    source = {row["candidate_id"]: row for row in read(batch_dir / "SOURCE_QUALIFICATION.json")["items"]}
    candidate_ids = [row["candidate_id"] for row in pool]
    root_ids = [row["root_id"] for row in pool]
    assert len(candidate_ids) == len(set(candidate_ids)) == len(root_ids) == len(set(root_ids))
    assert candidate_ids == adjudication["candidate_ids"]
    assert all(quality[key]["content_pass"] is True and source[key]["source_qualified"] is True for key in candidate_ids)
    calls = sum(2 * min(6, math.factorial(len(item["options"]))) * 3 for item in pool)
    assert calls == adjudication["planned_calls"]

    parent_exclusions = {row["parent_batch"]: set(row["candidate_ids"]) for row in adjudication["parent_exclusions"]}
    assert set().union(*parent_exclusions.values()) == set(candidate_ids)
    ledger = adapter.HERE / "BUDGET_RESERVATIONS.jsonl"
    with ledger.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        prior = [json.loads(line) for line in stream if line.strip()]
        assert not any(row["batch"] == str(batch_dir) for row in prior), "INCREMENT_ALREADY_RESERVED"
        prior_by_path = {row["batch"]: row for row in prior}
        assert set(parent_exclusions) <= set(prior_by_path), "PARENT_BATCH_RESERVATION_MISSING"
        for parent_name, excluded_ids in parent_exclusions.items():
            parent = Path(parent_name).resolve()
            parent_reservation = prior_by_path[parent_name]
            assert excluded_ids <= set(parent_reservation["candidate_ids"])
            parent_requests = read(parent / "solver/REQUESTS.json")
            requested = {row["candidate_id"] for row in parent_requests}
            assert not (excluded_ids & requested), "ADJUDICATED_CANDIDATE_ALREADY_EVALUATED"
            preflight = read(parent / "solver/PREFLIGHT.json")
            assert excluded_ids <= set(preflight["excluded_candidate_ids"]), "PARENT_DID_NOT_EXCLUDE_CANDIDATE"
            parent_items = {row["candidate_id"]: row for row in read(parent / "ITEMS_PRIVATE.json")}
            for item in pool:
                if item["candidate_id"] in excluded_ids:
                    assert item["root_id"] == parent_items[item["candidate_id"]]["root_id"]

        prior_candidates = {candidate for row in prior for candidate in row["candidate_ids"]}
        prior_roots = {root for row in prior for root in row["root_ids"]}
        assert set(candidate_ids) <= prior_candidates, "INCREMENT_MUST_REFERENCE_PREVIOUSLY_FROZEN_CANDIDATES"
        assert set(root_ids) <= prior_roots, "INCREMENT_ROOT_OUTSIDE_FROZEN_SCOPE"
        assert sum(row["reserved_requests"] for row in prior) + calls <= 2448, "TOTAL_SOLVER_BUDGET_EXCEEDED"
        assert len(prior_candidates | set(candidate_ids)) <= 68, "TOTAL_DRAFT_BUDGET_EXCEEDED"
        entry = {"batch": str(batch_dir), "candidate_ids": candidate_ids, "root_ids": root_ids,
                 "reserved_requests": calls, "kind": "post_adjudication_increment",
                 "adjudication_id": adjudication["adjudication_id"],
                 "parent_batches": list(parent_exclusions)}
        stream.seek(0, 2)
        stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == "__main__":
    adapter.reserve_budget = reserve_adjudicated_increment
    adapter.main()
