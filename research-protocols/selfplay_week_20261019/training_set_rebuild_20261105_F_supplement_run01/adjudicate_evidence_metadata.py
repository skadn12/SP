#!/usr/bin/env python3
"""Create an auditable, citation-metadata-only view of frozen first drafts."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


RUN = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def evidence_rows(candidate):
    rows = list(candidate.get("evidence", []))
    for item in candidate.get("option_audit", []) + candidate.get("pair_audit", []):
        rows.extend(item.get("evidence", []))
    return rows


parser = argparse.ArgumentParser()
parser.add_argument("batch")
args = parser.parse_args()
batch = (RUN / "review_batches" / args.batch).resolve()
assert batch.parent == (RUN / "review_batches").resolve()
assert not (batch / "QUALITY_FINAL.json").exists(), "must precede Gold comparison"
items = read(batch / "ITEMS_PRIVATE.json")
mechanical = read(batch / "MECHANICAL_REVIEW.json")["items"]
inventory = {row["root_id"]: row for row in read(RUN / "INPUT_FREEZE.json")["roots"]}
derived = copy.deepcopy(items)
decisions = []
for item, row in zip(derived, mechanical):
    assert item["candidate_id"] == row["candidate_id"]
    errors = row["errors"]
    if not errors:
        continue
    frozen = inventory[item["root_id"]]
    source_path = RUN / frozen["source_snapshot"]
    assert sha(source_path) == frozen["source_sha256"]
    turns = {t["turn_ref"]: t for t in read(source_path)["turns"]}
    fixes = []
    unresolved = []
    before = hashlib.sha256(canonical(item)).hexdigest()
    for error in errors:
        if error == "target_option_id_mismatch":
            target = read(RUN / "authoring" / item["root_id"] / "TARGETS_FROZEN.json")["targets"][0]
            facts = target.get("option_facts", [])
            ids = set(facts) if isinstance(facts, dict) else {x.get("option_id") for x in facts if isinstance(x, dict)}
            if ids == set(item["label_to_option_id"].values()):
                fixes.append({"error": error, "action": "checker_false_positive", "verified_option_ids": sorted(ids)})
            else:
                unresolved.append(error)
            continue
        if ":" not in error:
            unresolved.append(error)
            continue
        kind, old_ref = error.split(":", 1)
        selected = [ev for ev in evidence_rows(item) if ev.get("turn_ref") == old_ref]
        if not selected or kind not in ("unknown_turn", "origin_mismatch"):
            unresolved.append(error)
            continue
        if kind == "unknown_turn":
            prefix = item["root_id"] + ":"
            if not old_ref.startswith(prefix):
                unresolved.append(error)
                continue
            suffix = old_ref[len(prefix):]
            matching = [ref for ref, turn in turns.items()
                        if ref.endswith(":" + suffix) and all(ev.get("quote") and ev["quote"] in turn["text"] for ev in selected)]
            if len(matching) != 1:
                unresolved.append(error)
                continue
            new_ref = matching[0]
            for ev in selected:
                ev["turn_ref"] = new_ref
            fixes.append({"error": error, "old_ref": old_ref, "new_ref": new_ref,
                          "matching_turns": len(matching), "quotes_verified": len(selected)})
        else:
            turn = turns.get(old_ref)
            if not turn or any(not ev.get("quote") or ev["quote"] not in turn["text"] for ev in selected):
                unresolved.append(error)
                continue
            origins = sorted({str(ev.get("origin")) for ev in selected if "origin" in ev})
            if not origins or turn.get("origin") is None:
                unresolved.append(error)
                continue
            for ev in selected:
                if "origin" in ev:
                    ev["origin"] = turn["origin"]
            fixes.append({"error": error, "turn_ref": old_ref, "old_origins": origins,
                          "new_origin": turn["origin"], "quotes_verified": len(selected)})
    after = hashlib.sha256(canonical(item)).hexdigest()
    decisions.append({"candidate_id": item["candidate_id"], "source_sha256": frozen["source_sha256"],
                      "original_item_canonical_sha256": before, "derived_item_canonical_sha256": after,
                      "fixes": fixes, "unresolved_errors": unresolved, "mechanical_pass": not unresolved,
                      "source_qualification_still_required": True})

out = batch / "metadata_adjudication"
out.mkdir(exist_ok=False)
for name, value in (("ITEMS_DERIVED.json", derived), ("DECISIONS.json", {"items": decisions})):
    with (out / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
print(json.dumps({"batch": args.batch, "decisions": decisions}, ensure_ascii=False))
