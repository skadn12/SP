"""One-shot content admission; never imports Solver responses or scores."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def binding(path):
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def save(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    assert not (HERE / "EXECUTION_FREEZE.json").exists(), "No overwrite of frozen admission"
    generation = read(HERE / "GENERATION_FREEZE.json")
    assert read(HERE / "review/AUTHOR_STRUCTURAL_VALIDATION.json")["status"] == "PASS"
    authors = sorted((HERE / "authors").glob("author_r*.jsonl"))
    rows = [json.loads(line) for path in authors for line in path.read_text().splitlines() if line.strip()]
    candidates = {row["candidate_id"]: row for row in rows}
    assert len(candidates) == len(rows) == 24
    reviews = {}
    review_paths = [HERE / "review/ROOT_CONTENT_REVIEW.json", HERE / "review/INDEPENDENT_CONTENT_REVIEW.json"]
    for path in review_paths:
        payload = read(path)
        reviews[path.stem] = {row["candidate_id"]: row for row in payload["reviews"]}
        assert set(reviews[path.stem]) == set(candidates)
    public, private, ledger, execution_slots = [], [], [], []
    for slot in generation["slots"]:
        qid = slot["slot_id"]
        row = candidates[qid]
        opinions = {name: values[qid]["status"] for name, values in reviews.items()}
        if row["status"] == "SHORTFALL":
            assert set(opinions.values()) == {"SOURCE_SHORTFALL"}
            status = "SOURCE_SHORTFALL"
        elif set(opinions.values()) == {"PASS"}:
            status = "CONTENT_PASS"
        else:
            assert set(opinions.values()) <= {"PASS", "HOLD"}, opinions
            status = "CONTENT_HOLD"
        field = "options" if row["task"] == "multi_select" else "items"
        item_count = len(row.get(field) or {}) or None
        execution_slots.append({"slot_id": qid, "candidate_id": qid, "root_id": row["root_id"],
            "question_type": row["task"], "item_count": item_count,
            "source": {"path": row["source_path"], "sha256": row["source_sha256"]}})
        ledger.append({"slot_id": qid, "candidate_id": qid, "root_id": row["root_id"],
            "status": status, "author_status": row["status"], "review_opinions": opinions,
            "template_id": row["template_id"], "primary_relation": row.get("primary_relation"),
            "answer_format": row["answer_format"], "question_type": row["task"],
            "target_gold_count": slot.get("target_gold_count"),
            "actual_gold_count": len(row["gold"]) if row["status"] == "DRAFT" and field == "options" else None,
            "item_count": item_count, "timeline_kind": row.get("timeline_kind"),
            "fact_family_id": row.get("fact_family_id"), "source_path": row["source_path"],
            "source_sha256": row["source_sha256"], "reason": row.get("reason"),
            "missing_material_feedback": row.get("missing_material_feedback"),
            "parent_review_notes": reviews["ROOT_CONTENT_REVIEW"][qid]["review_notes"],
            "self_contained_operands_or_constraints": reviews["ROOT_CONTENT_REVIEW"][qid]["self_contained_operands_or_constraints"],
            "strong_intrinsic_procedure_cues": reviews["ROOT_CONTENT_REVIEW"][qid]["strong_intrinsic_procedure_cues"],
            "training_admitted": False})
        if status != "CONTENT_PASS":
            continue
        question = {"candidate_id": qid, "root_id": row["root_id"], "question_type": row["task"],
            "question": row["question"], field: row[field],
            "source_path": row["source_path"], "source_sha256": row["source_sha256"]}
        public.append(question)
        private.append(dict(question, gold=row["gold"]))
    assert len(public) > 0
    output = HERE / "exports"
    save(output / "PUBLIC_QUESTIONS.json", {"questions": public})
    save(output / "PRIVATE_QUESTIONS.json", {"questions": private})
    (output / "PRIVATE_QUESTIONS.json").chmod(0o600)
    save(output / "CANDIDATE_LEDGER.json", {"planned_slots": 24, "records": ledger,
        "counts": dict(Counter(row["status"] for row in ledger)), "solver_scores_used_for_admission": False})
    with (output / "ALL_AUTHOR_CANDIDATES.jsonl").open("x") as stream:
        for slot in generation["slots"]:
            stream.write(json.dumps(candidates[slot["slot_id"]], ensure_ascii=False) + "\n")
    calls = sum(18 if q["question_type"] == "multi_select" else min(12, math.factorial(len(q["items"]))) for q in public)
    solver = {"name": "S0_M_clean",
        "model": "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_common_dev_eval_v1_live_3/clean_merged_model",
        "tokenizer": "/root/autodl-tmp/Qwen3-4B", "MC_mappings": 6, "MC_samples_each_mapping": 3,
        "ordering_mappings": 12, "ordering_samples_each_mapping": 1,
        "temperature": 0.8, "top_p": 0.9, "max_tokens": 32, "enable_thinking": False,
        "max_model_len": 16384, "source_truncation": False, "automatic_retry": False,
        "display_seed": 2026102001, "sampling_seed": 2026102002}
    now = datetime.now(timezone.utc).isoformat()
    save(HERE / "CONTENT_ADMISSION.json", {"status": "CONTENT_FROZEN", "namespace": "train.feedback/diversity_crossroot_v1",
        "created_at_utc": now, "planned_slots": 24, "admitted_count": len(public),
        "questions": [{"candidate_id": q["candidate_id"]} for q in public],
        "public_questions": binding(output / "PUBLIC_QUESTIONS.json"),
        "private_questions": binding(output / "PRIVATE_QUESTIONS.json"),
        "reviews": [binding(path) for path in review_paths],
        "all_slot_ledger": binding(output / "CANDIDATE_LEDGER.json"),
        "author_bindings": [binding(path) for path in authors],
        "admission_rule": "Both parent and different-author AI reviews PASS; all shortfalls and target holds retained. R2-MC2 holds under candidate-set nonredundancy, not wrong Gold.",
        "solver_scores_available": False, "training_admission": False, "dev_test_used": False})
    save(HERE / "EXECUTION_FREEZE.json", {"schema_version": "diversity_crossroot_execution_freeze_v1",
        "created_at_utc": now, "phase": "FROZEN_BEFORE_SOLVER", "solver": solver,
        "budget": {"production_planned_slots": 24, "initial_question_slots": len(public), "max_direct_requests": calls},
        "slots": execution_slots, "generation_freeze": binding(HERE / "GENERATION_FREEZE.json"),
        "content_admission": binding(HERE / "CONTENT_ADMISSION.json"),
        "interpretation": "Fixed Train feasibility and diagnosis only. Ordering arity3 uses6 distinct displays, arity4–6 uses12; report separately. No training or DEV/TEST."})
    print(json.dumps({"admitted": len(public), "counts": dict(Counter(row["status"] for row in ledger)),
        "question_types": dict(Counter(q["question_type"] for q in public)), "solver_requests": calls}))


if __name__ == "__main__":
    main()
