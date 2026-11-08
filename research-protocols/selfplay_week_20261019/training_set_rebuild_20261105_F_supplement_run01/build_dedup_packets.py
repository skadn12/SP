#!/usr/bin/env python3
"""Build score-free, lexical-retrieved comparison packets for semantic review."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


RUN = Path(__file__).resolve().parent
REFERENCE = RUN / "semantic_review"
STOP = set("the a an and or but if then than of in on at to for from by with about as is are was were be been being do did does done have has had what which who when where why how all any every each only same different this that these those it its their her his she he they them we you your i me my our report says said described shows states according listed selected following".split())


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def words(value):
    if isinstance(value, dict):
        value = " ".join(str(k) + " " + words(v) for k, v in value.items())
    elif isinstance(value, list):
        value = " ".join(words(v) for v in value)
    else:
        value = str(value)
    return [w for w in re.findall(r"[a-z0-9]+", value.lower()) if w not in STOP and len(w) > 2]


def text_for(item):
    return " ".join(str(item.get(k, "")) for k in ("question", "event_identity", "cognitive_task", "natural_information_need", "relation_plan", "query_scope", "answer_semantics", "options", "items"))


def refs_for(item):
    out = []
    for path in sorted(REFERENCE.glob("REFERENCE_cc-train-episode-*.json")):
        out.extend(read(path))
    return out


parser = argparse.ArgumentParser()
parser.add_argument("--batch", required=True)
parser.add_argument("--top-k", type=int, default=18)
args = parser.parse_args()
assert 1 <= args.top_k <= 30
batch = (RUN / "review_batches" / args.batch).resolve()
assert batch.parent == (RUN / "review_batches").resolve()
out = RUN / "semantic_review" / "packets" / args.batch
out.mkdir(parents=True, exist_ok=False)
corpus = refs_for(None)
indexed = []
for item in corpus:
    tokens = words(text_for(item))
    indexed.append((item, tokens, Counter(tokens)))

bindings = [{"path": str(p), "sha256": digest(p)} for p in sorted(REFERENCE.glob("REFERENCE_cc-train-episode-*.json"))]
all_candidates = []
for batch_path in sorted((RUN / "review_batches").glob("*/ITEMS_PRIVATE.json")):
    for item in read(batch_path):
        all_candidates.append(item)
index = {item["candidate_id"]: item for item in all_candidates}
assert len(index) == len(all_candidates)

new_items = read(batch / "ITEMS_PRIVATE.json")
for query in new_items:
    tokens = words(text_for(query))
    q = Counter(tokens)
    ranked = []
    for old, old_tokens, old_counts in indexed:
        overlap = sum(min(count, old_counts.get(token, 0)) for token, count in q.items())
        weighted = sum(min(count, old_counts.get(token, 0)) * (1.0 + 3.0 / (1 + old_counts.get(token, 0))) for token, count in q.items())
        denom = sum(q.values()) + sum(old_counts.values()) - overlap
        similarity = overlap / denom if denom else 0
        if old["root_id"] == query["root_id"]:
            similarity += 0.12
        ranked.append((similarity, weighted, old["candidate_id"], old))
    ranked.sort(key=lambda row: (-row[0], -row[1], row[2]))
    # Include the closest lexical matches and all prior same-root candidates.
    selected = {row[2]: row[3] for row in ranked[:args.top_k]}
    for row in ranked:
        if row[3]["root_id"] == query["root_id"]:
            selected[row[2]] = row[3]
    detail = []
    for cid in sorted(selected):
        item = selected[cid]
        detail.append({k: item[k] for k in ("candidate_id", "root_id", "question_type", "question", "options", "items", "event_identity", "cognitive_task", "natural_information_need", "relation_plan", "query_scope", "query_pairs", "answer_semantics") if k in item})
    packet = {"schema_version": "score_free_semantic_family_review_v1", "candidate_id": query["candidate_id"],
              "root_id": query["root_id"], "question_type": query["question_type"], "question": query["question"],
              "options": query.get("options", query.get("items")), "event_identity": query.get("event_identity"),
              "natural_information_need": query.get("natural_information_need"), "relation_plan": query.get("relation_plan"),
              "query_scope": query.get("query_scope"), "answer_semantics": query.get("answer_semantics"),
              "retrieval_policy": "lexical retrieval is only a candidate finder; reviewer can flag a duplicate absent from these matches",
              "reference_count": len(detail), "references": detail}
    path = out / f"{query['candidate_id']}.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(packet, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    bindings.append({"path": str(path), "sha256": digest(path)})

manifest = {"batch": args.batch, "new_candidate_ids": [item["candidate_id"] for item in new_items],
            "historical_candidate_count": len(corpus), "solver_scores_read": False, "gold_answers_in_packet": False,
            "retrieval": {"method": "weighted lexical overlap", "top_k": args.top_k, "all_same_root_matches_added": True},
            "input_bindings": bindings}
with (out / "PACKET_BINDINGS.json").open("x", encoding="utf-8") as stream:
    json.dump(manifest, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({"batch": args.batch, "queries": len(new_items), "historical_records": len(corpus), "max_references_per_query": max((read(p)["reference_count"] for p in out.glob("*.json") if p.name != "PACKET_BINDINGS.json"), default=0)}))
