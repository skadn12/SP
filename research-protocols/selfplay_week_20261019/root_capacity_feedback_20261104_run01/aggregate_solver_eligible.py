#!/usr/bin/env python3
"""Aggregate strict-PASS items into <=5 item paired-root local Solver batches."""
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
COMPARISON = HERE / "comparison"
OUT = HERE / "solver_eligible"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if OUT.exists():
    raise SystemExit(f"output already exists: {OUT}")

groups = {}
input_rows = []
source_by_root = {}
candidate_ids = set()
for quality_path in sorted(COMPARISON.glob("*/?/batch/QUALITY_FINAL.json")):
    batch = quality_path.parent
    rows = read(quality_path)["items"]
    pool = read(batch / "ITEMS_PRIVATE.json")
    q_by_id = {row["candidate_id"]: row for row in rows}
    if len(q_by_id) != len(rows) or set(q_by_id) != {item["candidate_id"] for item in pool}:
        raise SystemExit(f"candidate/quality mismatch in {batch}")
    freeze = read(batch / "SOURCE_FREEZE.json")["sources"]
    for source in freeze:
        previous = source_by_root.get(source["root_id"])
        if previous is not None and previous != source:
            raise SystemExit(f"inconsistent source freeze: {source['root_id']}")
        source_by_root[source["root_id"]] = source
    source_lookup = {row["root_id"]: row for row in freeze}
    for item in pool:
        cid = item["candidate_id"]
        if cid in candidate_ids:
            raise SystemExit(f"duplicate candidate id: {cid}")
        candidate_ids.add(cid)
        status = q_by_id[cid]
        if status["status"] == "PASS" and status["content_pass"] is True:
            source = source_lookup[item["root_id"]]
            if item["source_sha256"] != source["sha256"] or sha(Path(source["path"])) != source["sha256"]:
                raise SystemExit(f"source hash mismatch: {cid}")
            groups.setdefault(item["root_id"], []).append((item, status, batch))
        input_rows.append({
            "batch": str(batch),
            "quality_path": str(quality_path),
            "quality_sha256": sha(quality_path),
            "items_path": str(batch / "ITEMS_PRIVATE.json"),
            "items_sha256": sha(batch / "ITEMS_PRIVATE.json"),
            "source_freeze_path": str(batch / "SOURCE_FREEZE.json"),
            "source_freeze_sha256": sha(batch / "SOURCE_FREEZE.json"),
        })

if not groups:
    raise SystemExit("no strict PASS candidates are eligible")

for root, members in groups.items():
    arms = [item.get("arm", "C") for item, _, _ in members]
    if len(arms) > 2 or len(arms) != len(set(arms)):
        raise SystemExit(f"duplicate root/arm candidate: {root}")

root_groups = [groups[root] for root in sorted(groups)]
parts = []
current = []
for root_group in root_groups:
    if current and len(current) + len(root_group) > 5:
        parts.append(current)
        current = []
    if len(root_group) > 5:
        raise SystemExit("a root group exceeds Solver batch capacity")
    current.extend(root_group)
if current:
    parts.append(current)

OUT.mkdir(parents=True)
manifest = {
    "schema_version": "solver_eligible_aggregation_v1",
    "selection": "Only QUALITY_FINAL rows with status PASS and content_pass is exactly true.",
    "pairing": "All eligible F/G items for one root remain in the same Solver batch.",
    "solver_max_items_per_batch": 5,
    "author_and_quality_inputs_modified": False,
    "input_bindings": input_rows,
    "source_roots": sorted(source_by_root),
    "eligible_root_arm_groups": {
        root: [item["candidate_id"] for item, _, _ in members]
        for root, members in sorted(groups.items())
    },
    "batches": [],
}
for index, entries in enumerate(parts, 1):
    batch_dir = OUT / f"part_{index:02d}"
    batch_dir.mkdir()
    items = [entry[0] for entry in entries]
    quality = [entry[1] for entry in entries]
    roots = sorted({item["root_id"] for item in items})
    sources = [source_by_root[root] for root in roots]
    write(batch_dir / "ITEMS_PRIVATE.json", items)
    write(batch_dir / "QUALITY_FINAL.json", {"items": quality})
    write(batch_dir / "SOURCE_FREEZE.json", {"sources": sources})
    write(batch_dir / "AGGREGATION_BINDINGS.json", {
        "items": [
            {"candidate_id": item["candidate_id"], "author_path": item["author_path"],
             "author_sha256": item["author_sha256"], "review_path": row["review_path"],
             "review_sha256": row["review_sha256"], "source_sha256": item["source_sha256"]}
            for item, row in zip(items, quality)
        ],
        "all_source_freezes": sources,
    })
    manifest["batches"].append({
        "path": str(batch_dir),
        "candidate_ids": [item["candidate_id"] for item in items],
        "root_ids": roots,
        "candidate_count": len(items),
    })
write(OUT / "AGGREGATION_MANIFEST.json", manifest)
print(json.dumps({"eligible": sum(len(value) for value in groups.values()),
                  "roots": {root: len(entries) for root, entries in sorted(groups.items())},
                  "batches": manifest["batches"]}, ensure_ascii=False))
