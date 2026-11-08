#!/usr/bin/env python3
"""Bind quality and source qualification into a separate immutable eval input."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


RUN = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


parser = argparse.ArgumentParser()
parser.add_argument("batch")
args = parser.parse_args()
review_batch = (RUN / "review_batches" / args.batch).resolve()
assert review_batch.parent == (RUN / "review_batches").resolve()
eval_batch = RUN / "evaluation" / args.batch
assert not eval_batch.exists()
derived = review_batch / "metadata_adjudication" / "ITEMS_DERIVED.json"
item_path = derived if derived.exists() else review_batch / "ITEMS_PRIVATE.json"
items = read(item_path)
quality = read(review_batch / "QUALITY_FINAL.json")
sources = read(review_batch / "SOURCE_FREEZE.json")
qualification = []
bindings = [{"path": str(path), "sha256": sha(path)} for path in (item_path, review_batch / "QUALITY_FINAL.json", review_batch / "SOURCE_FREEZE.json")]
for item in items:
    item["number"] = 1
    review_path = review_batch / "blind_results" / f"{item['review_id']}.json"
    review = read(review_path)
    bindings.append({"path": str(review_path), "sha256": sha(review_path)})
    source = read(Path(item["source_path"]))
    turns = {turn["turn_ref"]: turn for turn in source["turns"]}
    exact_errors = []
    evidence = list(item.get("evidence", []))
    for row in item.get("option_audit", []) + item.get("pair_audit", []):
        evidence.extend(row.get("evidence", []))
    for ev in evidence:
        turn = turns.get(ev.get("turn_ref"))
        if turn is None or not ev.get("quote") or ev["quote"] not in turn["text"] or ("origin" in ev and ev["origin"] != turn.get("origin")):
            exact_errors.append(ev)
    required_refs = set(review.get("source_qualification", {}).get("used_turn_refs", []))
    for row in review.get("per_option", []):
        required_refs.update(row.get("turn_refs", []))
    missing_review_refs = sorted(required_refs - set(turns))
    source_binding_ok = sha(Path(item["source_path"])) == item["source_sha256"] == review.get("source_sha256")
    decision = review.get("source_qualification", {})
    assert type(decision.get("source_qualified")) is bool, f"MISSING_SOURCE_QUALIFICATION: {item['candidate_id']}"
    qualified = decision["source_qualified"] and source_binding_ok and not exact_errors and not missing_review_refs
    qualification.append({"candidate_id": item["candidate_id"], "root_id": item["root_id"],
                          "source_qualified": qualified, "source_binding_ok": source_binding_ok,
                          "author_quote_origin_errors": exact_errors, "unknown_reviewer_refs": missing_review_refs,
                          "blind_source_decision": decision, "review_path": str(review_path), "review_sha256": sha(review_path),
                          "source_sha256": item["source_sha256"],
                          "scope": "actual question evidence checked; no whole-root or real-world certification"})

eval_batch.mkdir(parents=True)
write(eval_batch / "ITEMS_PRIVATE.json", items)
write(eval_batch / "QUALITY_FINAL.json", quality)
write(eval_batch / "SOURCE_FREEZE.json", sources)
write(eval_batch / "SOURCE_QUALIFICATION.json", {"items": qualification})
write(eval_batch / "INPUT_BINDINGS.json", {"review_batch": str(review_batch), "files": bindings,
                                         "metadata_only_transport": "number=1 added if omitted; source evidence may use separately adjudicated metadata view"})
print(json.dumps({"batch": args.batch, "items": len(items), "quality_pass": sum(row["content_pass"] for row in quality["items"]),
                  "source_qualified": sum(row["source_qualified"] for row in qualification)}))
