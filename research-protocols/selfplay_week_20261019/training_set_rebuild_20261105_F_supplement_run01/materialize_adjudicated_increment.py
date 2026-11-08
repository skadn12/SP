#!/usr/bin/env python3
"""Create a new, source-bound eval batch for candidates excluded before adjudication."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path


RUN = Path(__file__).resolve().parent
EVALUATION = RUN / "evaluation"
OUTPUT = EVALUATION / "adjudicated_increment_01"
SELECTION = {
    "batch_02": ["FS1_75223_01"],
    "batch_03": ["FS1_75241_01", "FS1_87964_01", "FS1_96544_01", "FS1_103866_01"],
}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main() -> None:
    assert not OUTPUT.exists(), "NO_OVERWRITE_OR_RETRY"
    quality_adj_path = RUN / "review_batches/batch_03/QUALITY_ADJUDICATION.json"
    source_adj_path = RUN / "review_batches/batch_02/SOURCE_ADJUDICATION.json"
    quality_adj, source_adj = read(quality_adj_path), read(source_adj_path)
    quality_decisions = {row["candidate_id"]: row for row in quality_adj["items"]}
    assert set(quality_decisions) == set(SELECTION["batch_03"])
    assert source_adj["candidate_id"] == "FS1_75223_01" and source_adj["decision"] == "source_qualified"

    items, quality_rows, source_rows, source_rows_by_root = [], [], [], {}
    bindings: dict[str, str] = {}
    parent_exclusions = []
    calls = 0
    for batch_name, candidate_ids in SELECTION.items():
        parent = EVALUATION / batch_name
        parent_exclusions.append({"parent_batch": str(parent), "candidate_ids": candidate_ids})
        for filename in ("ITEMS_PRIVATE.json", "QUALITY_FINAL.json", "SOURCE_FREEZE.json", "SOURCE_QUALIFICATION.json"):
            path = parent / filename
            bindings[str(path)] = sha(path)
        parent_items = {row["candidate_id"]: row for row in read(parent / "ITEMS_PRIVATE.json")}
        parent_quality = {row["candidate_id"]: row for row in read(parent / "QUALITY_FINAL.json")["items"]}
        parent_sources = read(parent / "SOURCE_FREEZE.json")["sources"]
        parent_source_rows = {row["candidate_id"]: row for row in read(parent / "SOURCE_QUALIFICATION.json")["items"]}
        source_by_root = {row["root_id"]: row for row in parent_sources}
        for candidate_id in candidate_ids:
            item = copy.deepcopy(parent_items[candidate_id])
            qrow = copy.deepcopy(parent_quality[candidate_id])
            srow = copy.deepcopy(parent_source_rows[candidate_id])
            assert item["candidate_id"] == qrow["candidate_id"] == srow["candidate_id"]
            assert item["source_sha256"] == qrow["source_sha256"] == srow["source_sha256"]
            assert item["root_id"] == qrow["root_id"] == srow["root_id"]
            review_path = RUN / "review_batches" / batch_name / "blind_results" / f"{item['review_id']}.json"
            assert sha(review_path) == qrow["review_sha256"]
            bindings[str(review_path)] = sha(review_path)
            source_snapshot = Path(item["source_path"])
            assert sha(source_snapshot) == item["source_sha256"]
            bindings[str(source_snapshot)] = sha(source_snapshot)

            if candidate_id in quality_decisions:
                decision = quality_decisions[candidate_id]
                assert decision["adjudicated_content_pass"] is True
                assert decision["source_sha256"] == item["source_sha256"]
                assert qrow["content_pass"] is False and qrow["status"] == "HOLD"
                qrow["prior_status"] = qrow["status"]
                qrow["prior_hold_reasons"] = list(qrow["hold_reasons"])
                qrow["content_pass"] = True
                qrow["status"] = "PASS_ADJUDICATED"
                qrow["hold_reasons"] = []
                qrow["hard_issues"] = []
                qrow["soft_issues"] = list(qrow.get("soft_issues", [])) + [{
                    "issue": "Question stem does not repeat multi-select instruction",
                    "reason": decision["soft_issue"],
                }]
                qrow["quality_adjudication"] = {
                    "path": str(quality_adj_path), "sha256": sha(quality_adj_path),
                    "adjudication_id": quality_adj["adjudication_id"],
                }
            if candidate_id == source_adj["candidate_id"]:
                assert srow["source_qualified"] is False
                assert srow["source_binding_ok"] is True and not srow["author_quote_origin_errors"]
                assert srow["unknown_reviewer_refs"] == [source_adj["issue"]["invalid_reference"]]
                srow["prior_source_qualification"] = copy.deepcopy(srow)
                srow["source_qualified"] = True
                srow["unknown_reviewer_refs"] = []
                srow["excluded_reviewer_refs"] = [source_adj["issue"]["invalid_reference"]]
                srow["source_adjudication"] = {
                    "path": str(source_adj_path), "sha256": sha(source_adj_path),
                    "adjudication_id": source_adj["adjudication_id"],
                }
            assert qrow["content_pass"] is True and srow["source_qualified"] is True
            qrow["formal_training_approval_created"] = False
            items.append(item)
            quality_rows.append(qrow)
            source_rows.append(srow)
            source_rows_by_root[item["root_id"]] = source_by_root[item["root_id"]]
            calls += 2 * min(6, math.factorial(len(item["options"]))) * 3

    for path in (quality_adj_path, source_adj_path, Path(__file__).resolve(),
                 RUN / "model_baseline/prepare_adjudicated_increment.py"):
        bindings[str(path)] = sha(path)
    parent_batches = [row["parent_batch"] for row in parent_exclusions]
    adjudication = {
        "kind": "post_adjudication_increment",
        "adjudication_id": "supplement_run01_increment_01",
        "candidate_ids": [row["candidate_id"] for row in items],
        "parent_exclusions": parent_exclusions,
        "quality_adjudication": {"path": str(quality_adj_path), "sha256": sha(quality_adj_path)},
        "source_adjudication": {"path": str(source_adj_path), "sha256": sha(source_adj_path)},
        "planned_calls": calls,
        "decision": "Candidates were excluded before any inference, then independently adjudicated. This batch adds only their first planned evaluation calls; it is not a retry.",
        "parent_batches": parent_batches,
        "formal_training_approval_created": False,
    }
    OUTPUT.mkdir(parents=True)
    write(OUTPUT / "ADJUDICATION.json", adjudication)
    write(OUTPUT / "ITEMS_PRIVATE.json", items)
    write(OUTPUT / "QUALITY_FINAL.json", {"items": quality_rows})
    write(OUTPUT / "SOURCE_QUALIFICATION.json", {"items": source_rows})
    write(OUTPUT / "SOURCE_FREEZE.json", {"sources": list(source_rows_by_root.values())})
    write(OUTPUT / "INPUT_BINDINGS.json", {
        "parents": parent_batches,
        "files": [{"path": path, "sha256": digest} for path, digest in sorted(bindings.items())],
    })
    print(json.dumps({"batch": str(OUTPUT), "candidate_count": len(items), "planned_calls": calls,
                      "quality_pass": len(quality_rows), "source_qualified": len(source_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
