#!/usr/bin/env python3
"""Prepare new root batches after a separately reserved adjudication increment."""
from __future__ import annotations

import fcntl
import json
import math
import os
from pathlib import Path

import adapter


def reserve_new_roots(batch_dir: Path) -> None:
    batch_dir = Path(batch_dir).resolve()
    read = adapter.read
    pool = read(batch_dir / "ITEMS_PRIVATE.json")
    quality = {row["candidate_id"]: row for row in read(batch_dir / "QUALITY_FINAL.json")["items"]}
    source = {row["candidate_id"]: row for row in read(batch_dir / "SOURCE_QUALIFICATION.json")["items"]}
    candidate_ids = [row["candidate_id"] for row in pool]
    root_ids = [row["root_id"] for row in pool]
    assert len(candidate_ids) == len(set(candidate_ids)) == len(root_ids) == len(set(root_ids))
    calls = sum(2 * min(6, math.factorial(len(item["options"]))) * 3 for item in pool
                if quality[item["candidate_id"]]["content_pass"] and source[item["candidate_id"]]["source_qualified"])
    ledger = adapter.HERE / "BUDGET_RESERVATIONS.jsonl"
    with ledger.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        prior = [json.loads(line) for line in stream if line.strip()]
        assert not any(row["batch"] == str(batch_dir) for row in prior), "BATCH_ALREADY_RESERVED_NO_RETRY"
        old_candidates = {candidate for row in prior for candidate in row["candidate_ids"]}
        old_roots = {root for row in prior for root in row["root_ids"]}
        assert not (old_candidates & set(candidate_ids)), "CANDIDATE_ALREADY_RESERVED"
        assert not (old_roots & set(root_ids)), "ROOT_ALREADY_RESERVED"
        assert len(old_candidates | set(candidate_ids)) <= 68, "TOTAL_DRAFT_BUDGET_EXCEEDED"
        assert len(old_roots | set(root_ids)) <= 68, "TOTAL_ROOT_BUDGET_EXCEEDED"
        assert sum(row["reserved_requests"] for row in prior) + calls <= 2448, "TOTAL_SOLVER_BUDGET_EXCEEDED"
        entry = {"batch": str(batch_dir), "candidate_ids": candidate_ids, "root_ids": root_ids,
                 "reserved_requests": calls, "kind": "new_roots_after_adjudication_increment"}
        stream.seek(0, 2)
        stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == "__main__":
    adapter.reserve_budget = reserve_new_roots
    adapter.main()
