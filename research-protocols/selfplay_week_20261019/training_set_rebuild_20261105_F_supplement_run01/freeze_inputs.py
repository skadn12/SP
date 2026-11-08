#!/usr/bin/env python3
"""Verify and snapshot the 68 frozen dialogue inputs for F supplement run 01."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


RUN = Path(__file__).resolve().parent
WEEK = RUN.parent
PREVIOUS = WEEK / "training_set_rebuild_20261104_run01"
ROOT_COVERAGE = PREVIOUS / "ROOT_COVERAGE.json"
SOURCE_FREEZE = PREVIOUS / "SOURCE_FREEZE_ALL68.json"
INPUTS = RUN / "inputs"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_new(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


coverage = read(ROOT_COVERAGE)
source_freeze = read(SOURCE_FREEZE)
coverage_rows = {row["root_id"]: row for row in coverage["roots"]}
sources = {row["root_id"]: row for row in source_freeze["sources"]}
assert len(coverage_rows) == len(sources) == 68
assert set(coverage_rows) == set(sources)
assert coverage["batch_yield_can_stop_remaining_coverage"] is False

entries = []
for root_id in sorted(sources, key=lambda value: int(value.rsplit("-", 1)[1])):
    frozen = sources[root_id]
    source_path = Path(frozen["path"])
    source_bytes = source_path.read_bytes()
    assert digest(source_bytes) == frozen["sha256"], f"source hash changed: {root_id}"
    neutral_path = Path(frozen["neutral_source_info_path"])
    neutral_bytes = neutral_path.read_bytes()
    neutral = json.loads(neutral_bytes)
    assert neutral["root_id"] == root_id, f"neutral source root mismatch: {root_id}"
    assert neutral["canonical_source_binding"]["sha256"] == frozen["sha256"]
    source_doc = json.loads(source_bytes)
    assert len(source_doc["turns"]) == frozen["turn_count"], f"turn count changed: {root_id}"

    dest = INPUTS / root_id
    source_copy = dest / "SOURCE.json"
    neutral_copy = dest / "NEUTRAL_SOURCE_INFO.json"
    dest.mkdir(parents=True, exist_ok=True)
    with source_copy.open("xb") as stream:
        stream.write(source_bytes)
    with neutral_copy.open("xb") as stream:
        stream.write(neutral_bytes)

    old_f = coverage_rows[root_id]["arms"]["F"]
    entries.append({
        "root_id": root_id,
        "batch_index": coverage_rows[root_id]["batch_index"],
        "previous_F_first_drafts": old_f["first_drafts"],
        "previous_F_draft_state": old_f["state"],
        "previous_F_draft_sha256": old_f.get("sha256"),
        "source_path": str(source_path),
        "source_sha256": frozen["sha256"],
        "turn_count": frozen["turn_count"],
        "source_snapshot": str(source_copy.relative_to(RUN)),
        "source_snapshot_sha256": digest(source_copy.read_bytes()),
        "neutral_source_info_path": str(neutral_path),
        "neutral_source_info_snapshot": str(neutral_copy.relative_to(RUN)),
        "neutral_source_info_sha256": digest(neutral_copy.read_bytes()),
        "source_warnings": frozen.get("warnings", []),
    })

root_counts = {
    str(count): sum(row["previous_F_first_drafts"] == count for row in entries)
    for count in (0, 1, 2)
}
assert sum(int(count) * n for count, n in root_counts.items()) == 80

manifest = {
    "schema_version": "f_supplement_input_freeze_v1",
    "run_id": "training_set_rebuild_20261105_F_supplement_run01",
    "frozen_from": {
        "root_coverage_path": str(ROOT_COVERAGE),
        "root_coverage_sha256": digest(ROOT_COVERAGE.read_bytes()),
        "source_freeze_path": str(SOURCE_FREEZE),
        "source_freeze_sha256": digest(SOURCE_FREEZE.read_bytes()),
    },
    "root_count": len(entries),
    "previous_F_draft_count_by_root": root_counts,
    "previous_F_first_drafts_total": 80,
    "coverage_policy": "all 68 roots are required; batch boundaries do not permit early stop",
    "roots": entries,
}
write_new(RUN / "INPUT_FREEZE.json", manifest)

checksums = []
for path in sorted(INPUTS.rglob("*")):
    if path.is_file():
        checksums.append(f"{digest(path.read_bytes())}  {path.relative_to(RUN)}")
for name in ("RUN_PROTOCOL.md", "RUN_CONFIG.json"):
    path = RUN / name
    checksums.append(f"{digest(path.read_bytes())}  {name}")
with (RUN / "INPUT_CHECKSUMS.sha256").open("x", encoding="utf-8") as stream:
    stream.write("\n".join(checksums) + "\n")

print(json.dumps({
    "status": "PASS",
    "roots": len(entries),
    "previous_F_drafts": root_counts,
    "previous_F_total": 80,
    "source_snapshots": len(entries),
    "neutral_source_snapshots": len(entries),
    "sha256_entries": len(checksums),
}, ensure_ascii=False, indent=2))
