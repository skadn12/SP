"""Assemble source-bound human semantic audit; never edits generation or selection."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent.parent / "h4_sequential_ab_v55_qa/runs"
SUPPLEMENT = HERE.parent.parent / "h4_sequential_ab_v55_capacity_qa/runs/root135466_c05_capacity4096_01_generation"


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def binding(p):
    return {"path": str(p), "sha256": sha(p)}


def main(export=False):
    audit_files = sorted(HERE.glob("root*_c*_AUDIT.json"))
    audited = {q["candidate_id"]: q for path in audit_files for q in read(path)["questions"]}
    rows, packets, sources = [], [], {}
    pools = sorted(RUNS.glob("root*_collect20_01_c*_generation/MECHANICAL_POOL.json"))
    if (SUPPLEMENT / "MECHANICAL_POOL.json").exists():
        pools.append(SUPPLEMENT / "MECHANICAL_POOL.json")
    for pool in pools:
        ready_path = pool.parent / "SOLVER_POOL.json"
        ready = {q["candidate_id"]: q for q in read(ready_path)["questions"]} if ready_path.exists() else {}
        for mechanical in read(pool)["questions"]:
            q = ready.get(mechanical["candidate_id"], mechanical)
            source = Path(q["source_path"])
            assert sha(source) == q["source_sha256"]
            sources[str(source)] = binding(source)
            field = "options" if q["question_type"] == "multi_select" else "items"
            evidence = q["option_evidence" if field == "options" else "item_evidence"]
            turns = {t["turn_ref"]: t for t in read(source)["turns"]}
            labels = []
            for label, text in q[field].items():
                ev = evidence[label]
                labels.append({"label": label, "text": text, "gold": label in q["gold"], "evidence": ev,
                               "source_turn": turns.get(ev.get("turn_ref")),
                               "final_order_anchor": q.get("order_anchor_spans", {}).get(label)})
            packets.append({"candidate_id": q["candidate_id"], "root_id": q["root_id"],
                "question_type": q["question_type"], "question": q["question"], "gold": q["gold"],
                "ready_for_solver": q["candidate_id"] in ready, "labels": labels,
                "source": binding(source), "mechanical_pool": binding(pool),
                "solver_pool": binding(ready_path) if ready_path.exists() else None})
            audit = audited.get(q["candidate_id"])
            if audit is not None:
                assert audit["source_sha256"] == q["source_sha256"]
                assert audit["mechanical_pool_sha256"] == sha(pool)
                assert audit["core_status"] in {"CORE_USABLE", "KNOWN_SUBSTANTIVE_ISSUE"}
            rows.append({"candidate_id": q["candidate_id"], "root_id": q["root_id"],
                         "question_type": q["question_type"], "ready_for_solver": q["candidate_id"] in ready,
                         "audit": audit, "core_status": audit["core_status"] if audit else "PENDING"})
    report = {"scope": "independent_read_only_source_content_audit_of_actual_new_v55_mechanical_candidates",
              "new_gate": False, "generation_or_gold_edits": False, "solver_scores_used": False,
              "cardinality_rule": "2_to_6_correct_allowed_no_per_question_quota",
              "pronouns_alone_nonblocking": True, "completed_chunks": len(pools),
              "mechanical_questions": len(rows), "audited_questions": sum(r["audit"] is not None for r in rows),
              "status_counts": dict(Counter(r["core_status"] for r in rows)),
              "ready_status_counts": dict(Counter(r["core_status"] for r in rows if r["ready_for_solver"])),
              "by_root": {root: dict(Counter(r["core_status"] for r in rows if r["root_id"] == root))
                          for root in sorted({r["root_id"] for r in rows})}, "questions": rows,
              "bindings": [binding(p) for p in pools + audit_files] + list(sources.values())}
    # Selection is joined only after each source audit was written. It cannot
    # influence the recorded semantic judgments or any writer prompt.
    result_files = sorted(RUNS.glob("root*_collect20_01_c*_evaluation/solver/RESULTS.json"))
    supplement_result = SUPPLEMENT.with_name(SUPPLEMENT.name.replace("_generation", "_evaluation")) / "solver/RESULTS.json"
    if supplement_result.exists():
        result_files.append(supplement_result)
    selected_ids = {qid for p in result_files for qid in read(p)["selected_questions"]}
    evaluated_ids = {q["candidate_id"] for p in result_files for q in read(p)["questions"]}
    report["solver_selection_join"] = {
        "complete_evaluation_chunks": len(result_files), "evaluated_questions": len(evaluated_ids),
        "selected_questions": len(selected_ids),
        "selected_status_counts": dict(Counter(r["core_status"] for r in rows if r["candidate_id"] in selected_ids)),
        "core_usable_selected_ids": [r["candidate_id"] for r in rows if r["candidate_id"] in selected_ids and r["core_status"] == "CORE_USABLE"],
        "known_issue_selected_ids": [r["candidate_id"] for r in rows if r["candidate_id"] in selected_ids and r["core_status"] == "KNOWN_SUBSTANTIVE_ISSUE"],
        "bindings": [binding(p) for p in result_files]}
    (HERE / "REVIEW_PACKETS.json").write_text(json.dumps(packets, ensure_ascii=False, indent=2)+"\n")
    (HERE / "REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    lines = ["# v55 新题独立内容检查", "", "此检查记录已生成题目的事实/Gold 问题，不改出题规则、题目或 Gold，不以答案数量/代词为额外拦截。", "",
             f"已完成 {len(pools)} 个 chunk，{len(rows)} 道机械通过题；内容检查状态 {report['status_counts']}。", ""]
    for row in rows:
        lines += ["## "+row["candidate_id"], "", row["core_status"]+"；进入 Solver："+str(row["ready_for_solver"]), ""]
        audit = row["audit"]
        if audit:
            lines += [audit["summary"], ""]
            lines += ["- "+issue["label"]+"："+issue["reason"] for issue in audit.get("issues", [])]
            lines += [""]
    (HERE / "REPORT.md").write_text("\n".join(lines)+"\n")
    if export:
        assert not any(r["core_status"] == "PENDING" for r in rows), "Unreviewed actual mechanical candidates"
        (HERE.parent / "INDEPENDENT_CONTENT_AUDIT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
        (HERE.parent / "INDEPENDENT_CONTENT_AUDIT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in {"questions", "bindings"}},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true")
    main(parser.parse_args().export)
