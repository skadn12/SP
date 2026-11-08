"""Validate one frozen author attempt and export only independently admitted content."""
from pathlib import Path
import datetime
import hashlib
import json

D = Path(__file__).resolve().parent


def read(p):
    return json.loads(p.read_text())


def bind(p):
    b = p.read_bytes()
    return {"path": str(p), "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}


def save(p, obj):
    with p.open("x") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write(chr(10))


def main():
    freeze = read(D / "GENERATION_FREEZE.json")
    author = D / "authors/CANDIDATE.json"
    review = D / "review/CANDIDATE_CONTENT_REVIEW.json"
    questions = read(author)["questions"]
    assert len(questions) == 1
    q = questions[0]
    slot = freeze["candidate"]
    assert all(q[k] == slot[k] for k in ("candidate_id", "root_id", "question_type", "feedback_id", "family_id"))
    assert len(q["options"]) == 6 and set(q["options"]) == set("ABCDEF")
    assert len(q["gold"]) == len(set(q["gold"])) and set(q["gold"]) <= set(q["options"])
    source = Path(q["source_path"])
    assert bind(source) == slot["source"] and q["source_sha256"] == slot["source"]["sha256"]
    turns = {t["turn_ref"]: t["text"] for t in read(source)["turns"]}
    assert len(turns) == 177
    assert set(q["per_item_evidence"]) == set(q["options"])
    checks = []
    for label, item in q["per_item_evidence"].items():
        for group in ("evidence", "fact_checks"):
            for e in item[group]:
                assert e["turn_ref"] in turns and e["quote"] and e["quote"] in turns[e["turn_ref"]]
                checks.append({"label": label, "group": group, "turn_ref": e["turn_ref"], "exact": True})
    structural = {"status": "PASS", "author_binding": bind(author), "source_binding": bind(source), "source_turns": len(turns), "quote_checks": checks, "unique_gold_labels": True, "semantic_gold_decided_by_independent_review": True}
    save(D / "review/AUTHOR_STRUCTURAL_VALIDATION.json", structural)
    rv = read(review)
    assert rv["reviewed_author_binding"] == bind(author)
    assert rv["technical_blinding"] is False and rv["human_review"] is False
    assert len(rv["questions"]) == 1
    r = rv["questions"][0]
    assert r["candidate_id"] == q["candidate_id"] and r["disposition"] in ("ACCEPT", "HOLD")
    admitted = r["disposition"] == "ACCEPT"
    if admitted:
        assert all(r[k] for k in ("source_gold_unique", "feedback_target_realized", "exact_quotes_valid"))
    keys = ("candidate_id", "root_id", "question_type", "question", "options", "source_path", "source_sha256")
    public = {k: q[k] for k in keys}
    private = dict(public, gold=q["gold"], family_id=q["family_id"], feedback_id=q["feedback_id"])
    pub = D / "exports/PUBLIC_QUESTIONS.json"
    priv = D / "exports/PRIVATE_QUESTIONS.json"
    save(pub, {"questions": [public] if admitted else []})
    save(priv, {"questions": [private] if admitted else []})
    save(D / "exports/CANDIDATE_LEDGER.json", {"planned_slots": 1, "content_accepted": int(admitted), "content_hold": int(not admitted), "training_admitted": 0, "candidate_id": q["candidate_id"], "disposition": r["disposition"], "family_id": q["family_id"], "novelty_class": r["novelty_class"], "old_FBR_R4_MC2_disposition": "HOLD_UNCHANGED"})
    save(D / "CONTENT_ADMISSION.json", {"schema_version": "minimal_repair_content_admission_v1", "status": "PASS", "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "namespace": freeze["namespace"], "planned_slots": 1, "admitted_count": int(admitted), "candidate_ids": [q["candidate_id"]] if admitted else [], "public_questions": bind(pub), "private_questions": bind(priv), "generation_freeze": bind(D / "GENERATION_FREEZE.json"), "author_bindings": [bind(author)], "review_bindings": [bind(review)], "structural_validation": bind(D / "review/AUTHOR_STRUCTURAL_VALIDATION.json"), "admission_before_solver": True, "solver_calls_at_admission": 0, "technical_blinding": False, "human_review": False, "training_admitted": False})
    lines = ["# 新版本证据充分性候选", "", q["question"], ""]
    lines += [k + ". " + v for k, v in q["options"].items()]
    lines += ["", "Gold: " + ", ".join(q["gold"]), "独立内容审核: " + r["disposition"], "同source/family修复，不计为新增独立题家族；旧题保持HOLD。", "来源: " + str(source)]
    (D / "exports/QUESTIONS_WITH_GOLD_ZH.md").write_text(chr(10).join(lines) + chr(10))
    print(json.dumps({"status": "PASS", "accepted": int(admitted), "quote_checks": len(checks), "gold_count": len(q["gold"])}))


if __name__ == "__main__":
    main()
