"""Offline, write-once finalization of the v16 fixed-source question supplement.

No generation, Solver inference, feedback compilation or training is launched.
Historical artifacts are read-only. The v15 pool format is retained, with an
explicit v16 finalization version and all wrapper/audit dependencies archived.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE.parent
VERSION = "h4_r02_existing_source_yield_v16"
TYPES = ("multi_select", "ordering")
ASSEMBLER_PATH = D / "h4_feedback_training_pool_v15/assemble_pool.py"
spec = importlib.util.spec_from_file_location("v16_frozen_assembler", ASSEMBLER_PATH)
assembler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assembler)
read, require, sha = assembler.read, assembler.require, assembler.sha


def statements(question):
    return question["options" if question["question_type"] == "multi_select" else "items"]


def citations(question):
    """Literal source matching only; it does not establish semantic support."""
    source = read(question["source_path"])
    turns = {row["turn_ref"]: row for row in source["turns"]}
    field = "option_evidence" if question["question_type"] == "multi_select" else "order_anchor_spans"
    evidence = question.get(field, question.get("item_evidence", {}))
    result = []
    for label in statements(question):
        item = evidence.get(label, {})
        if not isinstance(item, dict):
            item = {}
        ref, quote = item.get("turn_ref"), item.get("quote")
        turn = turns.get(ref)
        valid = bool(turn and isinstance(quote, str) and quote and quote in turn["text"]
                     and item.get("speaker") == turn["speaker"])
        spans = []
        if valid:
            start = turn["text"].find(quote)
            while start >= 0:
                spans.append({"char_start": start, "char_end": start + len(quote)})
                start = turn["text"].find(quote, start + 1)
        result.append({"label": label, "turn_ref": ref, "speaker": item.get("speaker"),
            "quote": quote, "literal_quote_and_speaker_match": valid,
            "turn_text_sha256": assembler.digest(turn["text"].encode()) if turn else None,
            "spans": spans, "is_current_r02_branch_reference": bool(isinstance(ref, str)
                and ref.startswith("R02:" + question["expansion_branch"] + ":")),
            "is_valid_current_r02_quote": valid and ref.startswith("R02:" + question["expansion_branch"] + ":")})
    return result


def span_keys(cites):
    return {(row["turn_ref"], row["speaker"], row["turn_text_sha256"], span["char_start"], span["char_end"])
            for row in cites if row["literal_quote_and_speaker_match"] for span in row["spans"]}


def overlap_diagnostics(new_questions, old_questions):
    old = [(q, citations(q)) for q in old_questions]
    output = []
    for question in new_questions:
        refs = citations(question)
        exact = set(statements(question).values())
        normalized = {" ".join(x.casefold().split()) for x in exact}
        qspans = span_keys(refs)
        quotes = {r["quote"] for r in refs if r["literal_quote_and_speaker_match"]}
        pairs = []
        for prior, prior_refs in old:
            if prior["root_id"] != question["root_id"]:
                continue
            old_text = set(statements(prior).values())
            pspans = span_keys(prior_refs)
            span_intersections = [{"new_span": list(a), "old_span": list(b),
                                   "overlap_characters": min(a[4], b[4]) - max(a[3], b[3])}
                                  for a in sorted(qspans) for b in sorted(pspans)
                                  if a[:3] == b[:3] and min(a[4], b[4]) > max(a[3], b[3])]
            shared = sorted(exact & old_text)
            normalized_shared = sorted(normalized & {" ".join(x.casefold().split()) for x in old_text})
            shared_quotes = sorted(quotes & {r["quote"] for r in prior_refs if r["literal_quote_and_speaker_match"]})
            pairs.append({"old_candidate_id": prior["candidate_id"], "old_question_type": prior["question_type"],
                "same_source_context": question["source_sha256"] == prior["source_sha256"],
                "exact_statement_intersection_count": len(shared), "exact_statement_intersection": shared,
                "normalized_statement_intersection_count": len(normalized_shared),
                "exact_valid_quote_intersection_count": len(shared_quotes), "exact_valid_quote_intersection": shared_quotes,
                "identical_source_span_count": len(qspans & pspans), "overlapping_source_spans": span_intersections})
        valid_new = [r for r in refs if r["is_valid_current_r02_quote"]]
        output.append({"candidate_id": question["candidate_id"], "root_id": question["root_id"],
            "question_type": question["question_type"], "branch": question["expansion_branch"],
            "new_branch_reference_diagnostic": question.get("new_branch_reference_diagnostic"),
            "valid_current_r02_cited_option_count": len(valid_new),
            "valid_current_r02_unique_turn_count": len({r["turn_ref"] for r in valid_new}),
            "has_literal_valid_current_r02_quote": bool(valid_new),
            "valid_literal_cited_option_count": sum(r["literal_quote_and_speaker_match"] for r in refs),
            "citations": refs, "same_root_prior_questions_compared": len(pairs), "comparisons": pairs,
            "max_exact_statement_intersection": max((p["exact_statement_intersection_count"] for p in pairs), default=0),
            "max_identical_source_span_count": max((p["identical_source_span_count"] for p in pairs), default=0),
            "diagnostic_only": True, "semantic_novelty_certified": False, "semantic_support_certified": False})
    return output


def generation_counts(run, experiment, supplemental_spec):
    """Count unique assigned raw IDs, final gate output and completed evaluation."""
    rows, inputs = [], []
    for branch in ("A", "B"):
        generation = run / branch / "generation"
        manifest = read(generation / "MANIFEST.json")
        inputs.extend(generation / name for name in ("MANIFEST.json", "REPORT.json", "MECHANICAL_POOL.json", "SOLVER_POOL.json"))
        gen_report = read(generation / "REPORT.json")
        raw_ids = defaultdict(set)
        for proof_path in sorted((generation / "requests").glob("*/PROOF.json")):
            proof = read(proof_path)
            if proof.get("role") != "question_generator":
                continue
            inputs.append(proof_path)
            response = proof_path.with_name("RESPONSE.json")
            if not response.exists():
                continue
            inputs.append(response)
            try:
                raw = json.loads(read(response)["text"])
            except (KeyError, ValueError, TypeError):
                continue
            if not isinstance(raw, dict) or not isinstance(raw.get("questions"), list):
                continue
            assigned = set(proof["candidate_ids"])
            raw_ids[(proof["root_id"], proof["question_type"])].update(
                q["candidate_id"] for q in raw["questions"] if isinstance(q, dict)
                and isinstance(q.get("candidate_id"), str) and q["candidate_id"] in assigned)
        solver_results = run / branch / "solver/RESULTS.json"
        results = read(solver_results)["questions"] if solver_results.exists() else []
        if solver_results.exists():
            inputs.append(solver_results)
        for root in experiment["roots"]:
            active = root in supplemental_spec["active_roots_by_branch"][branch]
            source_row = next(r for r in manifest["source_rows"] if r["root_id"] == root)
            require(bool(source_row["accepted_scenes"]) == active, "source_activity_changed")
            for kind in TYPES:
                outcome = next(r for r in gen_report["roots"] if r["root_id"] == root and r["question_type"] == kind)
                selected = [q for q in results if q["root_id"] == root and q["question_type"] == kind]
                require(len(selected) == outcome["ready_for_solver"], "solver_not_complete_for_ready_questions")
                rows.append({"root_id": root, "branch": branch, "question_type": kind,
                    "active_candidate_slots": outcome["candidate_slots"] if active else 0,
                    "source_unavailable_nominal_slots": 0 if active else outcome["candidate_slots"],
                    "raw_generated_unique_assigned_ids": len(raw_ids[(root, kind)]),
                    "mechanical_pass": outcome["mechanical_pass"], "solver_evaluated": len(selected),
                    "full_solver_calls": sum(q["actual_full_calls"] for q in selected),
                    "selected_before_cross_branch_dedup": sum(q["selected_by_policy"] is True for q in selected),
                    "generator_abstentions": len(outcome["abstentions"]),
                    "mechanical_rejected_final": len(outcome["mechanical_rejected"]),
                    "mechanical_rejection_reasons": dict(Counter(e for q in outcome["mechanical_rejected"] for e in q["errors"])),
                    "ordering_review_rejected": len(outcome["review_rejected"]),
                    "solver_selection_reasons": dict(Counter(q["selection_reason"] for q in selected))})
    require(sum(r["active_candidate_slots"] for r in rows) == 40, "active_slots_not_40")
    return rows, inputs


def results_document(run, output, supplemental_spec, experiment, qa_report, old_questions, baseline, new_questions, pool_report, counts):
    diagnostics = overlap_diagnostics(new_questions, old_questions)
    excluded = {q["excluded_candidate_id"] for q in pool_report["duplicates"] if q["excluded_input_role"] == "new"}
    for row in diagnostics:
        row["added_to_final_pool"] = row["candidate_id"] not in excluded
    per_root = {}
    for root in experiment["roots"]:
        per_root[root] = {}
        for kind in TYPES:
            current = [r for r in counts if r["root_id"] == root and r["question_type"] == kind]
            new = [q for q in new_questions if q["root_id"] == root and q["question_type"] == kind]
            previous = [q for q in baseline if q["root_id"] == root and q["question_type"] == kind]
            per_root[root][kind] = {key: sum(r[key] for r in current) for key in (
                "active_candidate_slots", "raw_generated_unique_assigned_ids", "mechanical_pass", "solver_evaluated", "full_solver_calls")}
            per_root[root][kind].update(previous_same_source_selected=len(previous),
                new_selected_before_pool_dedup=len(new), new_unique_added=sum(q["candidate_id"] not in excluded for q in new),
                combined_same_source_selected=len(previous) + sum(q["candidate_id"] not in excluded for q in new))
    totals = {key: sum(r[key] for r in counts) for key in (
        "active_candidate_slots", "source_unavailable_nominal_slots", "raw_generated_unique_assigned_ids", "mechanical_pass", "solver_evaluated", "full_solver_calls")}
    totals.update(previous_same_source_selected=len(baseline), existing_all_contexts_selected=len(old_questions),
        new_selected_before_pool_dedup=len(new_questions), new_unique_added=pool_report["new_unique_added"],
        combined_same_source_selected=len(baseline) + pool_report["new_unique_added"],
        total_training_pool_unique=pool_report["total_unique"], generation_and_review_http_calls=qa_report["generation_http_attempts"],
        new_selected_with_literal_valid_current_r02_quote=sum(r["has_literal_valid_current_r02_quote"] for r in diagnostics),
        new_selected_without_literal_valid_current_r02_quote=sum(not r["has_literal_valid_current_r02_quote"] for r in diagnostics))
    require(totals["full_solver_calls"] == qa_report["full_solver_calls"], "report_full_call_mismatch")
    return {"version": VERSION, "status": "COMPLETE", "run": str(run), "pool": str(output),
        "source_reuse_only": True, "new_dialogue_turns": 0, "new_dataset_events": 0,
        "active_slot_explanation": "40 active slots = 5 accepted branches x 2 types x 4. The historical QA report's 80 nominal slots include 40 unavailable-branch slots and are not the active denominator.",
        "generator_model": qa_report["generator_model"], "solver_model": qa_report["solver_model"],
        "feedback_round": "previous_training_round_only", "policy_version": qa_report["policy_version"],
        "totals": totals, "per_root_and_type": per_root, "per_branch_and_type": counts,
        "new_selection_overlap_and_r02_citation_diagnostics": diagnostics,
        "baseline_selected_sources": [str(p) for p in supplemental_spec["parent_question_runs"]],
        "diagnostic_limitations": [
            "Overlap and citation coverage never add a selection gate; multiselect citations remain diagnostic only.",
            "Literal quote presence does not prove the proposition or Gold is semantically correct.",
            "Zero exact text/span overlap does not certify different tested facts; semantic novelty is not certified.",
            "P/R01-only questions are possible: this is marginal yield from frozen P+R01+R02 contexts, not a causal R02-versus-R01 experiment.",
            "The same-source baseline is 19; the existing pool's other 6 selected questions come from separate historical R02 contexts."],
        "semantic_gold_certified": False, "semantic_novelty_certified": False,
        "training_started": False, "finalizer_generation_calls": 0, "finalizer_solver_calls": 0}


def markdown(value):
    t = value["totals"]
    lines = ["# 冻结 R02 素材补出题结果", "",
        f"本次没有新增对话或指导事件。40 个有效候选位（不是 QA 模板中的 80 个名义位）中，模型返回 {t['raw_generated_unique_assigned_ids']} 个不同的已分配候选 ID，{t['mechanical_pass']} 题通过现有机械处理，{t['solver_evaluated']} 题完成 Solver，新增入池 **{t['new_unique_added']} 题**。",
        f"同源旧批入选 19 题，本批补出后同源累计 **{t['combined_same_source_selected']} 题**；包含其他历史 R02 素材旧 6 题的完整池由 25 增至 **{t['total_training_pool_unique']} 题**。", "",
        "| Root | 题型 | 有效候选位 | 返回候选ID | 机械通过 | Solver题数 | 同源旧入选 | 本次新增入池 | 同源累计 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for root, by_type in value["per_root_and_type"].items():
        for kind, r in by_type.items():
            label = "多选" if kind == "multi_select" else "排序"
            lines.append(f"| {root.split('-')[-1]} | {label} | {r['active_candidate_slots']} | {r['raw_generated_unique_assigned_ids']} | {r['mechanical_pass']} | {r['solver_evaluated']} | {r['previous_same_source_selected']} | {r['new_unique_added']} | {r['combined_same_source_selected']} |")
    lines += ["", f"出题与排序复核实际 HTTP 调用 {t['generation_and_review_http_calls']} 次，full Solver 回答 {t['full_solver_calls']} 次。生成和排序复核均为 {value['generator_model']}，Solver 为 {value['solver_model']}。继续使用冻结的上一训练轮反馈，未启动训练。", "",
        f"本次筛选通过题中，{t['new_selected_with_literal_valid_current_r02_quote']} 题至少含一个能在当前 R02 分支逐字匹配且人物一致的引文；{t['new_selected_without_literal_valid_current_r02_quote']} 题没有这样的引文。详细记录每题原有 new_branch_reference_diagnostic、有效引文覆盖以及与旧 25 题同 root 的选项原文／引文 span 重叠。", "",
        "上述引用与重叠均只作诊断，未增加准入门槛。逐字匹配不能证明语义支持；没有原文重叠也不证明考查事实不重复。尤其不能把只使用 P/R01 的补题都算成 R02 新内容收益。本实验测量同一冻结上下文再次出题的增量，不能证明 R02 相对 R01 的因果收益。", "",
        "每题保存完整 Gold 和冻结上下文；评估排列、回答次数不扩增独立训练题数。RESULTS.json 含逐 root、题型、分支的完整计数与只读重叠诊断。"]
    return "\n".join(lines) + "\n"


def finalize(run, output):
    run, output = run.resolve(strict=True), output.resolve()
    require(run.is_relative_to(HERE / "runs"), "run_must_be_in_v16_runs")
    require(output.is_relative_to(HERE) and output != HERE and not output.exists(), "new_v16_output_required")
    require(not output.is_relative_to(run), "pool_must_be_outside_frozen_qa_run")
    supplemental_spec = read(run / "SUPPLEMENT_SPEC.json")
    require(supplemental_spec["version"] == VERSION and supplemental_spec["source_reuse_only"] is True
            and supplemental_spec["active_candidate_slots"] == 40
            and supplemental_spec["new_dialogue_turns"] == supplemental_spec["new_dataset_events"] == 0,
            "wrong_supplement_scope")
    experiment, qa_report = read(run / "MANIFEST.json"), read(run / "REPORT.json")
    require(qa_report["status"] == "EVALUATION_COMPLETE", "qa_report_not_complete")
    for name in ("AUDIT.json", "SUPPLEMENT_AUDIT.json"):
        audit = read(run / name)
        require(audit["status"] == "PASS" and Path(audit["run"]).resolve() == run and not audit.get("failures"), "audit_failed_or_wrong_run:" + name)
        if name == "SUPPLEMENT_AUDIT.json":
            require(audit.get("stage") == "complete" and audit.get("audit_script_sha256") == sha(HERE / "audit_supplement.py"),
                    "supplement_audit_incomplete_or_script_changed")
            require(isinstance(audit.get("result_bindings"), list) and audit["result_bindings"], "supplement_result_bindings_missing")
            for binding in audit["result_bindings"]:
                require(sha(binding["path"]) == binding["sha256"], "supplement_audited_result_changed:" + binding["path"])
    existing_pool = Path(supplemental_spec["existing_pool"])
    previous_manifest = read(existing_pool / "MANIFEST.json")
    for binding in previous_manifest["files"]:
        require(sha(existing_pool / binding["path"]) == binding["sha256"], "existing_pool_file_changed")
    old_paths = [Path(p) for p in supplemental_spec["existing_selected_exports"]]
    require(len(old_paths) == len(set(old_paths)) == 4, "four_existing_exports_required")
    old_questions = [q for p in old_paths for q in read(p)["questions"]]
    old_pool_rows = [json.loads(line) for line in (existing_pool / "TRAINING_POOL.jsonl").read_text().splitlines()]
    require(len(old_questions) == len(old_pool_rows) == 25 and
        {q["candidate_id"]: q for q in old_questions} == {q["candidate_id"]: q["original_selected_question"] for q in old_pool_rows},
        "existing_25_exports_do_not_match_frozen_pool")
    baseline = [q for p in supplemental_spec["parent_question_runs"] for q in read(Path(p) / "SELECTED_QUESTIONS.json")["questions"]]
    require(len(baseline) == 19 and len({assembler.statement_key(q) for q in baseline}) == 19, "same_source_baseline_not_19")
    new_path = run / "SELECTED_QUESTIONS.json"
    new_questions = read(new_path)["questions"]
    counts, count_inputs = generation_counts(run, experiment, supplemental_spec)
    additional_bindings = [Path(__file__), HERE / "supplement.py", HERE / "audit_supplement.py", HERE / "POLICY.md",
        run / "SUPPLEMENT_SPEC.json", run / "SUPPLEMENT_AUDIT.json", run / "GENERATION_COMPLETE.json",
        run / "REPORT.json", run / "AUDIT.json", existing_pool / "MANIFEST.json", existing_pool / "REPORT.json",
        existing_pool / "TRAINING_POOL.jsonl", *count_inputs]
    original_inspector, original_write = assembler.Inspector, assembler.write_once
    old_here, historical = assembler.HERE, assembler.HISTORICAL_VERSIONS
    captured = {}

    class SupplementInspector(original_inspector):
        def __init__(self):
            super().__init__()
            for path in additional_bindings:
                self.bind(path)
            # Read-only evidence diagnostics use all historical source contexts.
            for q in old_questions + new_questions:
                self.bind(q["source_path"], q["source_sha256"])

    def augmented_write(path, value):
        if path == output / "MANIFEST.json":
            result = results_document(run, output, supplemental_spec, experiment, qa_report,
                old_questions, baseline, new_questions, value["report"], counts)
            original_write(output / "RESULTS.json", result)
            with (output / "RESULTS.md").open("x", encoding="utf-8") as handle:
                handle.write(markdown(result))
            value["finalization_version"] = VERSION
            value["format_compatibility_version"] = assembler.VERSION
            value["files"] = [{"path": str(p.relative_to(output)), "sha256": sha(p)}
                              for p in sorted(output.rglob("*")) if p.is_file()]
            captured["results"] = result
        original_write(path, value)

    try:
        assembler.HERE = HERE
        assembler.HISTORICAL_VERSIONS = set(historical) | {"h4_feedback_training_pool_v15_qa"}
        assembler.Inspector, assembler.write_once = SupplementInspector, augmented_write
        report = assembler.assemble(old_paths, [new_path], output)
    finally:
        assembler.HERE, assembler.HISTORICAL_VERSIONS = old_here, historical
        assembler.Inspector, assembler.write_once = original_inspector, original_write
    require(report["existing_unique"] == 25, "existing_questions_lost")
    final_manifest = read(output / "MANIFEST.json")
    for binding in final_manifest["files"]:
        require(sha(output / binding["path"]) == binding["sha256"], "final_pool_manifest_mismatch")
    return {"status": "PASS", "output": str(output), **captured["results"]["totals"], "training_started": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="A fresh sibling pool directory under v16; never overwritten.")
    args = parser.parse_args()
    print(json.dumps(finalize(args.run, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
