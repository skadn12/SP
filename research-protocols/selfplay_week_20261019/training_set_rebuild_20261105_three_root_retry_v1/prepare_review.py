#!/usr/bin/env python3
"""Bind three frozen drafts and prepare the existing Gold-blind review format."""
import ast
import hashlib
import json
import random
from pathlib import Path

RUN = Path(__file__).resolve().parent
OLD = RUN.parent / "training_set_rebuild_20261105_F_supplement_run01"
ROOTS = [f"cc-train-episode-{n}" for n in (189272, 249232, 59126)]
BATCH = RUN / "review_batches" / "batch_01"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


# Compile only the original inspection function, never its top-level batch runner.
old_script = OLD / "prepare_review.py"
tree = ast.parse(old_script.read_text(encoding="utf-8"), filename=str(old_script))
inspect_nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "inspect_candidate"]
assert len(inspect_nodes) == 1
exec(compile(ast.Module(body=inspect_nodes, type_ignores=[]), str(old_script), "exec"))
inventory = {r["root_id"]: r for r in read(OLD / "INPUT_FREEZE.json")["roots"] if r["root_id"] in ROOTS}
assert set(inventory) == set(ROOTS)
assert not BATCH.exists(), "batch already exists"

# Validate every authoring input before constructing or writing any packets.
inputs = []
for root in ROOTS:
    frozen = inventory[root]
    author = RUN / "authoring" / root
    files = [author / name for name in ("EXPLORATION.json", "TARGETS_FROZEN.json", "CANDIDATES_FIRST_DRAFT.json")]
    assert all(p.is_file() for p in files), f"authoring incomplete: {root}"
    exploration, targets, drafts = map(read, files)
    assert exploration["root_id"] == targets["root_id"] == drafts["root_id"] == root
    assert exploration.get("full_source_read") is True
    assert len(drafts["candidates"]) == len(targets["targets"]) <= 1
    assert files[1].stat().st_mtime_ns < files[2].stat().st_mtime_ns, f"target not frozen first: {root}"
    source_path = OLD / "inputs" / root / "SOURCE.json"
    neutral_path = OLD / "inputs" / root / "NEUTRAL_SOURCE_INFO.json"
    assert frozen["source_snapshot"] == str(source_path.relative_to(OLD))
    assert frozen["neutral_source_info_snapshot"] == str(neutral_path.relative_to(OLD))
    assert sha(source_path) == frozen["source_sha256"] == frozen["source_snapshot_sha256"]
    assert sha(neutral_path) == frozen["neutral_source_info_sha256"]
    source, neutral = read(source_path), read(neutral_path)
    assert len(source["turns"]) == frozen["turn_count"] == exploration["turn_count"]
    assert exploration["source_sha256"] == frozen["source_sha256"]
    inputs.append((root, frozen, files, targets, drafts, source_path, neutral_path, source, neutral))

items, mechanical, sources, bindings, zero_roots, packets = [], [], [], [], [], []
for root, frozen, files, targets, drafts, source_path, neutral_path, source, neutral in inputs:
    bindings.append({"root_id": root, "files": [{"path": str(p), "sha256": sha(p), "mtime_ns": p.stat().st_mtime_ns} for p in files]})
    if not drafts["candidates"]:
        zero_roots.append(root)
        continue
    c = drafts["candidates"][0]
    errors = inspect_candidate(c, targets["targets"][0], source, frozen, source_path)
    rid = "R" + root.rsplit("-", 1)[1]
    options = list(c["options"])
    random.Random(int(hashlib.sha256(("supplement-review:" + c["candidate_id"]).encode()).hexdigest(), 16)).shuffle(options)
    labels = [chr(65 + i) for i in range(len(options))] if c["question_type"] == "multi_select" else [f"I{i+1}" for i in range(len(options))]
    mapping = {label: option["option_id"] for label, option in zip(labels, options)}
    reverse = {value: key for key, value in mapping.items()}
    displayed = [{"id": label, "text": option["text"]} for label, option in zip(labels, options)]
    if c["question_type"] == "partial_order":
        gold_labels = [[reverse[a], reverse[b]] for a, b in c["gold_pairs"]]
    else:
        gold_labels = [reverse[oid] for oid in c["gold_option_ids"]]
        if c["question_type"] == "multi_select":
            gold_labels.sort()
    packet = {"review_id": rid, "root_id": root, "question_type": c["question_type"], "question": c["question"], "options": displayed,
              "source": {"path": str(source_path), "sha256": frozen["source_sha256"], "turn_count": frozen["turn_count"],
                         "neutral_source_info_path": str(neutral_path), "warnings": neutral.get("source_risks", [])}}
    if c["question_type"] == "partial_order":
        packet.update(query_scope=c["query_scope"], query_pairs=[[reverse[a], reverse[b]] for a, b in c["query_pairs"]], answer_semantics=c["answer_semantics"])
    items.append({**c, "number": 1, "review_id": rid, "options": displayed, "label_to_option_id": mapping, "gold_labels": gold_labels,
                  "author_path": str(files[2]), "author_sha256": sha(files[2])})
    mechanical.append({"candidate_id": c["candidate_id"], "root_id": root, "mechanical_pass": not errors, "errors": errors})
    sources.append({"root_id": root, "path": str(source_path), "sha256": frozen["source_sha256"], "turn_count": frozen["turn_count"],
                    "neutral_source_info_path": str(neutral_path), "neutral_source_info_sha256": sha(neutral_path)})
    packets.append((rid, packet))

assert len({item["candidate_id"] for item in items}) == len(items)
for rid, packet in packets:
    write(BATCH / "blind_packets" / f"{rid}.json", packet)
write(BATCH / "ITEMS_PRIVATE.json", items)
write(BATCH / "MECHANICAL_REVIEW.json", {"items": mechanical})
write(BATCH / "SOURCE_FREEZE.json", {"sources": sources})
write(BATCH / "BATCH_BINDINGS.json", {"roots": ROOTS, "zero_candidate_roots": zero_roots, "author_files": bindings,
      "old_prepare_review_path": str(old_script), "old_prepare_review_sha256": sha(old_script),
      "input_freeze_path": str(OLD / "INPUT_FREEZE.json"), "input_freeze_sha256": sha(OLD / "INPUT_FREEZE.json")})
print(json.dumps({"batch": str(BATCH), "candidates": len(items), "zero_candidate_roots": zero_roots,
                  "mechanical_failures": [m for m in mechanical if not m["mechanical_pass"]]}, ensure_ascii=False))
