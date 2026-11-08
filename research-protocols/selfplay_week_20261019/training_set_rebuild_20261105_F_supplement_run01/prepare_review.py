#!/usr/bin/env python3
"""Freeze first drafts and create Gold-blind review packets for one batch."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


RUN = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def inspect_candidate(candidate, target, source, frozen, source_path):
    errors = []
    root = frozen["root_id"]
    if candidate.get("root_id") != root or candidate.get("arm") != "F":
        errors.append("root_or_arm_mismatch")
    if candidate.get("source_path") != str(source_path) or candidate.get("source_sha256") != frozen["source_sha256"] or sha(source_path) != frozen["source_sha256"]:
        errors.append("source_binding")
    if candidate.get("source_turn_count") != frozen["turn_count"]:
        errors.append("source_turn_count")
    options = candidate.get("options", [])
    ids = [option["option_id"] for option in options]
    if len(ids) < 2 or len(ids) != len(set(ids)) or len({o["text"] for o in options}) != len(ids):
        errors.append("invalid_options")
    gold = candidate.get("gold_option_ids", [])
    kind = candidate.get("question_type")
    if kind == "multi_select":
        if not (2 <= len(gold) <= 6 and len(gold) < len(ids) and len(set(gold)) == len(gold) and set(gold) <= set(ids)):
            errors.append("invalid_multi_gold")
    elif kind == "ordering":
        if len(gold) != len(ids) or set(gold) != set(ids):
            errors.append("invalid_order_gold")
    elif kind == "partial_order":
        pairs = candidate.get("gold_pairs", [])
        if not pairs or gold != pairs or any(len(p) != 2 or p[0] == p[1] or not set(p) <= set(ids) for p in pairs) or len({tuple(p) for p in pairs}) != len(pairs):
            errors.append("invalid_partial_gold")
        if not candidate.get("query_scope") or not candidate.get("query_pairs") or candidate.get("answer_semantics") != "all_and_only_entailed_pairs_in_query_scope":
            errors.append("undefined_partial_scope")
        closure = {tuple(p) for p in pairs}
        expanded = closure | {(a, d) for a, b in closure for c, d in closure if b == c}
        if expanded != closure or any(a == b for a, b in closure):
            errors.append("partial_gold_not_acyclic_transitive_closure")
    else:
        errors.append("unsupported_question_type")
    if target.get("gold_option_ids", target.get("gold_pairs")) != gold:
        errors.append("target_gold_mismatch")
    facts = target.get("option_facts", [])
    if facts:
        fact_ids = set(facts) if isinstance(facts, dict) else {v.get("option_id") for v in facts if isinstance(v, dict)}
        if fact_ids != set(ids):
            errors.append("target_option_id_mismatch")
    audit = candidate.get("option_audit", [])
    if {row.get("option_id") for row in audit} != set(ids) or len(audit) != len(ids):
        errors.append("incomplete_option_audit")
    for row in audit:
        oid = row.get("option_id")
        if kind == "multi_select" and row.get("correct") != (oid in gold):
            errors.append("truth_gold_mismatch:" + str(oid))
        if kind == "ordering" and oid in gold and row.get("gold_position") != gold.index(oid) + 1:
            errors.append("position_gold_mismatch:" + str(oid))
    turns = {turn["turn_ref"]: turn for turn in source["turns"]}
    evidence = list(candidate.get("evidence", []))
    for row in audit + candidate.get("pair_audit", []):
        evidence.extend(row.get("evidence", []))
    if not evidence:
        errors.append("no_evidence")
    for ev in evidence:
        turn = turns.get(ev.get("turn_ref"))
        if turn is None:
            errors.append("unknown_turn:" + str(ev.get("turn_ref")))
        elif not ev.get("quote") or ev["quote"] not in turn["text"]:
            errors.append("quote_not_exact:" + str(ev.get("turn_ref")))
        elif "origin" in ev and ev["origin"] != turn.get("origin"):
            errors.append("origin_mismatch:" + str(ev.get("turn_ref")))
    return sorted(set(errors))


parser = argparse.ArgumentParser()
parser.add_argument("--batch", required=True, help="New batch name, for example batch_01")
parser.add_argument("--roots", required=True, nargs="+")
args = parser.parse_args()
assert args.batch.startswith("batch_") and "/" not in args.batch and len(args.roots) <= 8
assert len(set(args.roots)) == len(args.roots)
batch = RUN / "review_batches" / args.batch
assert not batch.exists(), "batch already exists"
inventory = {row["root_id"]: row for row in read(RUN / "INPUT_FREEZE.json")["roots"]}
assert set(args.roots) <= set(inventory)
for prior in (RUN / "review_batches").glob("*/BATCH_BINDINGS.json"):
    assert not set(args.roots) & set(read(prior)["roots"]), "root already frozen in another batch"

items, mechanical, sources, bindings, zero_roots = [], [], [], [], []
for root in args.roots:
    frozen = inventory[root]
    author = RUN / "authoring" / root
    files = [author / name for name in ("EXPLORATION.json", "TARGETS_FROZEN.json", "CANDIDATES_FIRST_DRAFT.json")]
    assert all(path.is_file() for path in files), f"authoring incomplete: {root}"
    exploration, targets, drafts = map(read, files)
    assert exploration["root_id"] == targets["root_id"] == drafts["root_id"] == root
    candidates = drafts["candidates"]
    assert len(candidates) <= 1 and len(targets["targets"]) == len(candidates)
    assert files[1].stat().st_mtime_ns <= files[2].stat().st_mtime_ns, f"target not frozen first: {root}"
    assert exploration.get("full_source_read") is True
    source_path = (RUN / frozen["source_snapshot"]).resolve()
    neutral_path = (RUN / frozen["neutral_source_info_snapshot"]).resolve()
    source, neutral = read(source_path), read(neutral_path)
    assert sha(source_path) == frozen["source_sha256"]
    bindings.append({"root_id": root, "files": [{"path": str(p), "sha256": sha(p)} for p in files]})
    if not candidates:
        zero_roots.append(root)
        continue
    c = candidates[0]
    errors = inspect_candidate(c, targets["targets"][0], source, frozen, source_path)
    rid = "R" + root.rsplit("-", 1)[1]
    options = list(c["options"])
    random.Random(int(hashlib.sha256(("supplement-review:" + c["candidate_id"]).encode()).hexdigest(), 16)).shuffle(options)
    labels = [chr(65+i) for i in range(len(options))] if c["question_type"] == "multi_select" else [f"I{i+1}" for i in range(len(options))]
    mapping = {label: option["option_id"] for label, option in zip(labels, options)}
    reverse = {value: key for key, value in mapping.items()}
    displayed = [{"id": label, "text": option["text"]} for label, option in zip(labels, options)]
    if c["question_type"] == "partial_order":
        gold_labels = [[reverse[a], reverse[b]] for a, b in c["gold_pairs"]]
    else:
        gold_labels = [reverse[oid] for oid in c["gold_option_ids"]]
        if c["question_type"] == "multi_select":
            gold_labels.sort()
    packet = {"review_id": rid, "root_id": root, "question_type": c["question_type"],
              "question": c["question"], "options": displayed,
              "source": {"path": str(source_path), "sha256": frozen["source_sha256"], "turn_count": frozen["turn_count"],
                         "neutral_source_info_path": str(neutral_path), "warnings": neutral.get("source_risks", [])}}
    if c["question_type"] == "partial_order":
        packet.update(query_scope=c["query_scope"], query_pairs=[[reverse[a], reverse[b]] for a, b in c["query_pairs"]], answer_semantics=c["answer_semantics"])
    item = {**c, "number": 1, "review_id": rid, "options": displayed, "label_to_option_id": mapping, "gold_labels": gold_labels,
            "author_path": str(files[2]), "author_sha256": sha(files[2])}
    items.append(item)
    mechanical.append({"candidate_id": c["candidate_id"], "root_id": root, "mechanical_pass": not errors, "errors": errors})
    sources.append({"root_id": root, "path": str(source_path), "sha256": frozen["source_sha256"], "turn_count": frozen["turn_count"]})
    write(batch / "blind_packets" / f"{rid}.json", packet)

assert len({item["candidate_id"] for item in items}) == len(items)
write(batch / "ITEMS_PRIVATE.json", items)
write(batch / "MECHANICAL_REVIEW.json", {"items": mechanical})
write(batch / "SOURCE_FREEZE.json", {"sources": sources})
write(batch / "BATCH_BINDINGS.json", {"roots": args.roots, "zero_candidate_roots": zero_roots, "author_files": bindings})
print(json.dumps({"batch": args.batch, "roots": len(args.roots), "candidates": len(items), "zero_candidate_roots": zero_roots,
                  "mechanical_failures": [m for m in mechanical if not m["mechanical_pass"]]}, ensure_ascii=False))
