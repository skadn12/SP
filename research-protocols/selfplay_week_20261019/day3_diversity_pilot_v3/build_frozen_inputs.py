#!/usr/bin/env python3
"""Materialize reviewed Day-3 v3 candidates into solver public/private inputs.

This builder reads author/review records only to bind exact candidate hashes,
reconstruct public questions and collect private gold. It does not call a model.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
WEEK = BASE.parent
FREEZE = BASE / "EXECUTION_FREEZE_CORRECTED.json"
SLOTS = BASE / "SLOT_PLAN.json"
RUNTIME = WEEK / "day3_diversity_crossroot_v1" / "runtime"
OUT = BASE / "frozen_inputs"

def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha(path: Path) -> str:
    return sha_bytes(path.read_bytes())

def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")

def load(rel: str):
    return json.loads((BASE / rel).read_text())

def source_binding(root):
    freeze = load("EXECUTION_FREEZE_CORRECTED.json")
    row = next(x for x in freeze["source_boundary"]["source_bindings"] if x["root_id"] == root)
    path = Path(row["verified_path"])
    assert sha(path) == row["sha256"] and path.stat().st_size == row["bytes"]
    return row, path

def original_id_text_map(rec, typ):
    if typ == "multi_select":
        vals = rec.get("public", {}).get("options", rec.get("options"))
        if isinstance(vals, dict):
            return list(vals.items())
        return [(x["id"], x["text"]) for x in vals]
    vals = rec.get("public", {}).get("items", rec.get("items"))
    if isinstance(vals, dict):
        return list(vals.items())
    return [(x["id"], x["text"]) for x in vals]

def candidate_records():
    out = {}
    # R1: six individual candidate files; MC1 uses reviewed revision 2.
    for slot in (x["slot_id"] for x in load("SLOT_PLAN.json")["slots"] if x["root_id"] == "cc-train-episode-103866"):
        rel = "candidates/revisions/DIVV3-R1-MC1.revision2.json" if slot == "DIVV3-R1-MC1" else f"candidates/{slot}.json"
        out[slot] = (json.loads((BASE / rel).read_text()), rel)
    # R2 uses the revised bundle for MC1 and original content for the other slots.
    original = load("drafts/cc-train-episode-104814.json")
    revised = load("drafts/revisions/cc-train-episode-104814.revision1.json")
    for rec in original["slot_records"]:
        slot = rec["slot_id"]
        if slot == "DIVV3-R2-ORD2":
            continue
        if slot == "DIVV3-R2-MC1":
            chosen = next(x for x in revised["slot_records"] if x["slot_id"] == slot)
            out[slot] = (chosen, "drafts/revisions/cc-train-episode-104814.revision1.json#" + slot)
        else:
            out[slot] = (rec, "drafts/cc-train-episode-104814.json#" + slot)
    # R3 and R4 are per-root bundles. The independent review binds canonical
    # slot-record hashes inside the source bundle.
    for rel in ("drafts/r3/cc-train-episode-109164.json", "drafts/cc-train-episode-112812.json"):
        doc = load(rel)
        records = doc.get("slot_records", doc.get("drafts", []))
        for rec in records:
            out[rec["slot_id"]] = (rec, rel + "#" + rec["slot_id"])
    return out

def review_record_bindings():
    paths = [
        "reviews/DIVV3_R1_R2_INDEPENDENT_REVIEW.json",
        "reviews/DIVV3_R1_R2_REVISION_INDEPENDENT_REVIEW.json",
        "reviews/DIVV3_R3_R4_INDEPENDENT_REVIEW.json",
    ]
    return {p: {"path": str(BASE / p), "sha256": sha(BASE / p), "bytes": (BASE / p).stat().st_size}
            for p in paths}

def main():
    assert not OUT.exists(), f"REFUSE_OVERWRITE:{OUT}"
    freeze = load("EXECUTION_FREEZE_CORRECTED.json")
    slot_plan = load("SLOT_PLAN.json")
    assert sha(BASE / "EXECUTION_FREEZE_CORRECTED.json") == slot_plan["freeze_sha256"]
    assert len(slot_plan["slots"]) == 24
    reviews = review_record_bindings()
    review_docs = {Path(p).name: json.loads(Path(row["path"]).read_text()) for p, row in reviews.items()}
    review_by_slot = {}
    for doc in review_docs.values():
        for row in doc.get("slot_reviews", []):
            if row.get("slot_id"):
                review_by_slot[row["slot_id"]] = row
    initial_review = review_docs["DIVV3_R1_R2_INDEPENDENT_REVIEW.json"]
    revision_review = review_docs["DIVV3_R1_R2_REVISION_INDEPENDENT_REVIEW.json"]
    crossroot_review = review_docs["DIVV3_R3_R4_INDEPENDENT_REVIEW.json"]
    def review_for(rows, sid):
        return next(row for row in rows if row.get("slot_id") == sid)
    # R1/R2 review schemas store records in one review's slot_reviews list;
    # R3/R4 records bind each canonical slot record hash.
    record_by_slot, records = candidate_records(), []
    ledger_slots, public_questions, private_questions = [], [], []
    all_candidates = set(record_by_slot)
    expected_admitted = {x["slot_id"] for x in slot_plan["slots"] if x["slot_id"] != "DIVV3-R2-ORD2"}
    assert all_candidates == expected_admitted, (all_candidates ^ expected_admitted)
    for slot in slot_plan["slots"]:
        sid, root = slot["slot_id"], slot["root_id"]
        sb, srcpath = source_binding(root)
        rreview = review_by_slot.get(sid)
        if sid == "DIVV3-R2-ORD2":
            assert rreview and rreview.get("recommendation") == "PASS_SHORTFALL"
            ledger_slots.append({"slot_id": sid, "root_id": root, "question_type": "ordering",
                "relation_target": slot["relation_target"], "status": "SHORTFALL",
                "candidate_binding": None, "source_sha256": sb["sha256"],
                "review_binding": reviews["reviews/DIVV3_R1_R2_INDEPENDENT_REVIEW.json"],
                "review_finding": rreview.get("finding")})
            continue
        rec, rel = record_by_slot[sid]
        source_hash = (rec.get("source_binding", {}).get("sha256") or rec.get("source_binding_sha256") or
                       rec.get("source_sha256") or rec.get("source_path_sha256") or
                       rec.get("source_binding", {}).get("source_sha256"))
        # R2 draft records carry the root binding at the bundle level.
        assert sb["sha256"] == slot["source_sha256"]
        assert rreview is not None
        status = rreview.get("recommendation", rreview.get("status"))
        assert status == "PASS", (sid, status)
        if sid in ("DIVV3-R3-MC1", "DIVV3-R3-MC2", "DIVV3-R3-MC3", "DIVV3-R3-ORD1", "DIVV3-R3-ORD2", "DIVV3-R3-ORD3",
                   "DIVV3-R4-MC1", "DIVV3-R4-MC2", "DIVV3-R4-MC3", "DIVV3-R4-ORD1", "DIVV3-R4-ORD2", "DIVV3-R4-ORD3"):
            canonical = json.dumps(rec, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            assert sha_bytes(canonical) == rreview["slot_record_canonical_sha256"], sid
        typ = rec["question_type"]
        pairs = original_id_text_map(rec, typ)
        if typ == "multi_select":
            assert [k for k, _ in pairs] == list("ABCDEF")
            question = rec.get("public", {}).get("question") or rec.get("question")
            opts = {k: v for k, v in pairs}
            gold = rec.get("gold")
            assert isinstance(gold, list) and len(gold) == slot["target_gold_count"]
            assert all(k in opts for k in gold)
            public = {"candidate_id": sid, "root_id": root, "question_type": typ,
                      "question": question, "options": opts, "source_path": str(srcpath),
                      "source_sha256": sb["sha256"]}
            private = dict(public, gold=gold)
            item_count = 6
        else:
            assert 3 <= len(pairs) <= 6
            # Convert authored item IDs into the Solver's canonical I1..In
            # while preserving the reviewed authored item order.
            src_items = dict(pairs)
            authored_ids = list(src_items)
            labels = [f"I{i+1}" for i in range(len(authored_ids))]
            remap = dict(zip(authored_ids, labels))
            items = {remap[k]: src_items[k] for k in authored_ids}
            gold_order = rec.get("gold_order", rec.get("gold"))
            if isinstance(gold_order, str):
                gold_order = list(gold_order)
            assert isinstance(gold_order, list) and len(gold_order) == len(authored_ids)
            assert set(gold_order) == set(authored_ids)
            gold = [remap[k] for k in gold_order]
            question = rec.get("public", {}).get("question") or rec.get("question")
            public = {"candidate_id": sid, "root_id": root, "question_type": typ,
                      "question": question, "items": items, "source_path": str(srcpath),
                      "source_sha256": sb["sha256"]}
            private = dict(public, gold=gold)
            item_count = len(items)
        assert question and item_count == slot.get("option_count", item_count) if typ == "multi_select" else question
        assert typ == slot["question_type"]
        assert len(gold) == slot.get("target_gold_count", len(gold)) if typ == "multi_select" else True
        review_path = ("reviews/DIVV3_R1_R2_REVISION_INDEPENDENT_REVIEW.json" if sid in ("DIVV3-R1-MC1", "DIVV3-R2-MC1") else
                       "reviews/DIVV3_R3_R4_INDEPENDENT_REVIEW.json" if sid.startswith(("DIVV3-R3", "DIVV3-R4")) else
                       "reviews/DIVV3_R1_R2_INDEPENDENT_REVIEW.json")
        source_file = BASE / rel.split("#", 1)[0]
        candidate_binding = {"path": str(source_file), "sha256": sha(source_file), "bytes": source_file.stat().st_size,
            "selector": rel.split("#", 1)[1] if "#" in rel else None}
        # Verify the exact authored file/record hash that the independent
        # reviewer inspected. R3/R4 reviews bind canonical per-slot records
        # and the enclosing candidate bundle separately.
        if sid == "DIVV3-R1-MC1":
            expected_hash = review_for(revision_review["slot_reviews"], sid)["candidate_sha256"]
        elif sid == "DIVV3-R2-MC1":
            expected_hash = review_for(revision_review["slot_reviews"], sid)["candidate_sha256"]
        elif sid.startswith("DIVV3-R1-") or sid.startswith("DIVV3-R2-"):
            expected_hash = review_for(initial_review["slot_reviews"], sid)["candidate_sha256"]
        else:
            root_review = next(x for x in crossroot_review["source_bindings"] if x["root_id"] == root)
            expected_hash = root_review["candidate_bundle_sha256"]
        assert candidate_binding["sha256"] == expected_hash, (sid, candidate_binding["sha256"], expected_hash)
        ledger_slots.append({"slot_id": sid, "candidate_id": sid, "root_id": root, "question_type": typ,
            "relation_target": slot["relation_target"], "item_count": item_count,
            "target_gold_count": slot.get("target_gold_count"), "status": "PASS", "candidate_binding": candidate_binding,
            "source_binding": {"path": sb["verified_path"], "sha256": sb["sha256"], "bytes": sb["bytes"]},
            "review_binding": reviews[review_path], "review_status": status,
            "review_finding": rreview.get("finding"),
            "canonical_slot_record_sha256": rreview.get("slot_record_canonical_sha256")})
        public_questions.append(public)
        private_questions.append(private)

    assert len(ledger_slots) == 24 and sum(x["status"] == "PASS" for x in ledger_slots) == 23
    assert len(public_questions) == len(private_questions) == 23
    # Exact duplicate rule, applied to the complete stem and complete answer set.
    signatures = []
    for q in public_questions:
        field = "options" if q["question_type"] == "multi_select" else "items"
        sig = (q["question"].strip(), tuple((k, " ".join(v.casefold().split())) for k, v in q[field].items()))
        assert sig not in signatures, ("EXACT_DUPLICATE", q["candidate_id"])
        signatures.append(sig)

    protocol = load("EXECUTION_FREEZE_CORRECTED.json")
    fixed = json.loads((RUNTIME / "direct_v1/PROTOCOL.json").read_text())["solver"]
    solver = {**{k: fixed[k] for k in ("name", "model", "tokenizer", "MC_mappings", "MC_samples_each_mapping",
        "ordering_mappings", "ordering_samples_each_mapping", "temperature", "top_p", "max_tokens", "enable_thinking",
        "max_model_len", "source_truncation", "automatic_retry")},
        "display_seed": 2026102201, "sampling_seed": 2026102202}
    reqs = 0
    pub_by_id = {q["candidate_id"]: q for q in public_questions}
    for q in public_questions:
        reqs += 18 if q["question_type"] == "multi_select" else min(12, __import__("math").factorial(len(q["items"])))
    planned_slots = []
    for slot in slot_plan["slots"]:
        sid = slot["slot_id"]
        ent = {"slot_id": sid, "candidate_id": sid, "root_id": slot["root_id"],
               "question_type": slot["question_type"], "relation_target": slot["relation_target"],
               "item_count": len(pub_by_id[sid]["options" if slot["question_type"] == "multi_select" else "items"]) if sid in pub_by_id else None,
               "source": {"path": source_binding(slot["root_id"])[1].as_posix(), "sha256": slot["source_sha256"]}}
        planned_slots.append(ent)
    execution_protocol = {"schema_version": "diversity_multiform_pilot_v3_solver_protocol_v1",
        "freeze_path": str(FREEZE), "freeze_sha256": sha(FREEZE), "slot_plan_path": str(SLOTS), "slot_plan_sha256": sha(SLOTS),
        "solver": solver, "budget": {"initial_question_slots": 23, "production_planned_slots": 24, "max_direct_requests": reqs},
        "slots": planned_slots,
        "solver_policy": "Only independently content-PASS candidates enter S0. The one reviewed source shortfall remains in the 24-slot denominator. No performance-based admission or rewrite.",
        "evaluation_scope": "Exploratory direct-answer diagnosis on historically exposed Train roots only; no training, no DEV/TEST, no causal feedback claim.",
        "review_bindings": reviews}
    assert reqs == 336, reqs
    dump(OUT / "PUBLIC_QUESTIONS.json", {"questions": public_questions})
    (OUT / "PUBLIC_QUESTIONS.json").chmod(0o644)
    private_dir = OUT / "private"
    private_dir.mkdir(mode=0o700)
    private_path = private_dir / "PRIVATE_QUESTIONS.json"
    dump(private_path, {"questions": private_questions})
    private_path.chmod(0o600)
    dump(OUT / "EXECUTION_PROTOCOL.json", execution_protocol)
    public_hash, private_hash = sha(OUT / "PUBLIC_QUESTIONS.json"), sha(private_path)
    admission = {"schema_version": "diversity_multiform_solver_admission_v1", "status": "FROZEN",
        "namespace": "train.feedback/diagnostic_only", "freeze_sha256": sha(FREEZE), "slot_plan_sha256": sha(SLOTS),
        "questions": [{"candidate_id": q["candidate_id"]} for q in public_questions],
        "public_questions": {"path": str(OUT / "PUBLIC_QUESTIONS.json"), "sha256": public_hash,
                             "bytes": (OUT / "PUBLIC_QUESTIONS.json").stat().st_size},
        "private_questions": {"path": str(private_path), "sha256": private_hash, "bytes": private_path.stat().st_size},
        "planned_slots": 24, "admitted_count": 23, "candidate_bindings": ledger_slots,
        "review_bindings": reviews,
        "dedup_policy": "Only exact full stem plus full option/item set duplicates are excluded; related tasks remain eligible and tagged.",
        "solver_performance_used_for_admission": False, "dev_test_used": False, "model_calls_used_for_authoring": False}
    dump(OUT / "CONTENT_ADMISSION.json", admission)
    ledger = {"schema_version": "diversity_multiform_admission_ledger_v1", "status": "FROZEN_BEFORE_SOLVER",
        "frozen_at_utc": "2026-10-22T07:00:00Z", "freeze_binding": {"path": str(FREEZE), "sha256": sha(FREEZE)},
        "slot_plan_binding": {"path": str(SLOTS), "sha256": sha(SLOTS)},
        "denominator": {"planned_slots": 24, "planned_roots": 4}, "admitted_count": 23,
        "solver_admitted_candidate_ids": [q["candidate_id"] for q in public_questions], "slots": ledger_slots,
        "public_questions_binding": admission["public_questions"],
        "private_questions_binding": admission["private_questions"],
        "execution_protocol_binding": {"path": str(OUT / "EXECUTION_PROTOCOL.json"), "sha256": sha(OUT / "EXECUTION_PROTOCOL.json"), "bytes": (OUT / "EXECUTION_PROTOCOL.json").stat().st_size},
        "review_bindings": reviews,
        "solver_policy": "No inference until this exact ledger, public/private files, source bindings, reviews and protocol are frozen. Missing/HOLD/SHORTFALL slots remain in the fixed 24-slot denominator."}
    dump(OUT / "ADMISSION_LEDGER_FROZEN.json", ledger)
    print(json.dumps({"status": "FROZEN", "slots": 24, "admitted": 23, "shortfall": 1,
        "roots": 4, "requests": reqs, "public_sha256": public_hash, "private_sha256": private_hash,
        "protocol_sha256": sha(OUT / "EXECUTION_PROTOCOL.json"), "ledger_sha256": sha(OUT / "ADMISSION_LEDGER_FROZEN.json")}, indent=2))

if __name__ == "__main__":
    main()
