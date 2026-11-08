"""Export completed v55 collection and historical recovery without changing pools."""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "h4_sequential_ab_v55_qa"
sys.path.insert(0, str(CODE))
import batch
import display_export

INPUTS = set()
OUTPUTS = []
TOTAL_KEYS = ("slots", "initial_drafts", "repair_drafts", "requests_prepared", "http_attempts",
              "local_capacity_blocked_requests", "remote_error_requests", "mechanical_pass", "ready",
              "duplicate_rejected_slots", "other_mechanical_rejected_slots", "abstained_slots",
              "review_rejected_slots", "solver_calls", "selected")


def read(path):
    path = Path(path).resolve()
    INPUTS.add(path)
    return json.loads(path.read_text())


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def verify_bindings(rows):
    for row in rows:
        path = Path(row["path"]).resolve()
        require(path.is_file() and binding(path)["sha256"] == row["sha256"],
                "supplement_recovery_binding_mismatch:" + str(path))
        INPUTS.add(path)


def write(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    OUTPUTS.append(path)


def write_text(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value.rstrip() + "\n")
    OUTPUTS.append(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def distribution(questions):
    return display_export.multi_answer_distribution(questions)


def raw_histogram(rows, repair):
    return dict(sorted(Counter(row["supports_count"] for row in rows
        if row["question_type"] == "multi_select" and row["repair"] is repair
        and row.get("supports_count") is not None).items()))


def score_rows(solver, category, *, expected_questions=None, expected_calls=None):
    solver = Path(solver)
    if not (solver / "RESULTS.json").exists():
        skipped = read(solver / "SKIPPED.json")
        require(skipped["status"] == "EMPTY_POOL_NO_SOLVER" and skipped["candidate_count"] == 0,
                "missing_results_without_empty_pool_marker:" + str(solver))
        require(expected_questions in (None, []) and expected_calls in (None, 0), "unexpected_empty_solver")
        return []
    result = read(solver / "RESULTS.json")
    rows = []
    for score in result["questions"]:
        row = {key: score.get(key) for key in (
            "candidate_id", "root_id", "question_type", "expected_full_calls", "actual_full_calls",
            "full_exact_rate", "full_partial_mean", "selected_by_policy", "selection_reason", "data_errors")}
        row.update(category=category, solver_results_path=str(solver / "RESULTS.json"))
        calls = score["actual_full_calls"]
        exact = Fraction(score["full_exact_rate"]) if score.get("full_exact_rate") is not None else None
        partial = Fraction(score["full_partial_mean"]) if score.get("full_partial_mean") is not None else None
        require(exact is None or (exact * calls).denominator == 1, "nonintegral_exact_solver_count")
        row.update(exact_correct_calls=int(exact * calls) if exact is not None else None,
                   exact_rate_percent=round(float(exact) * 100, 2) if exact is not None else None,
                   partial_score_percent=round(float(partial) * 100, 2) if partial is not None else None)
        rows.append(row)
    if expected_questions is not None:
        require({row["candidate_id"] for row in rows} == {q["candidate_id"] for q in expected_questions},
                "solver_result_question_set_mismatch:" + str(solver))
    if expected_calls is not None:
        require(sum(row["actual_full_calls"] for row in rows) == expected_calls, "solver_result_call_total_mismatch")
    return rows


def request_details(generation, root, chunk, *, category="new_v55"):
    drafts, requests, parse_errors = [], [], []
    for folder in sorted((generation / "requests").glob("*")):
        if not folder.is_dir():
            continue
        proof = read(folder / "PROOF.json")
        has_http = (folder / "HTTP_ATTEMPT.json").exists()
        if has_http:
            read(folder / "HTTP_ATTEMPT.json")
        error = read(folder / "ERROR.json") if (folder / "ERROR.json").exists() else None
        record = {"root_id": root, "chunk": chunk, "category": category,
                  "request_path": str(folder / "REQUEST.json"),
                  "role": proof["role"], "question_type": proof.get("question_type"),
                  "remote_http_attempt": has_http, "error": error,
                  "capacity": proof.get("capacity"),
                  "local_capacity_blocked": bool(error and error.get("error_type") == "LocalCapacityBlocked")}
        INPUTS.add(folder / "REQUEST.json")
        if record["local_capacity_blocked"]:
            require(not has_http, "capacity_block_counted_as_remote")
        requests.append(record)
        if proof["role"] != "question_generator" or not (folder / "RESPONSE.json").exists():
            continue
        response_path = folder / "RESPONSE.json"
        response = read(response_path)
        repair = folder.name.endswith("-generate-2")
        try:
            parsed = json.loads(response["text"])
            questions = parsed.get("questions", [])
            if not isinstance(questions, list):
                raise ValueError("raw_questions_is_not_array")
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError, KeyError) as error:
            parse_errors.append({"root_id": root, "chunk": chunk, "repair": repair,
                                 "response": binding(response_path), "error_type": type(error).__name__,
                                 "error": str(error)})
            continue
        for question in questions:
            row = {"root_id": root, "chunk": chunk, "category": category,
                   "candidate_id": question.get("candidate_id") if isinstance(question, dict) else None,
                   "question_type": proof["question_type"], "repair": repair,
                   "raw_response": binding(response_path), "raw_question": question}
            if proof["question_type"] == "multi_select":
                options = question.get("options") if isinstance(question, dict) else None
                well_formed = isinstance(options, list) and len(options) == 6 and all(
                    isinstance(option, dict) and option.get("relation") in ("supports", "contradicts") for option in options)
                row["supports_count"] = sum(option["relation"] == "supports" for option in options) if well_formed else None
            drafts.append(row)
    return drafts, requests, parse_errors


def collect_chunk(root, index, generation, evaluation, *, category="new_v55", pool_directory=None):
    pool_directory = Path(pool_directory) if pool_directory is not None else evaluation / "training_pool"
    report = read(generation / "REPORT.json")
    pool_report = read(pool_directory / "REPORT.json")
    mechanical = read(generation / "MECHANICAL_POOL.json")["questions"]
    ready = read(generation / "SOLVER_POOL.json")["questions"]
    selected = read(pool_directory / "SELECTED_QUESTIONS.json")["questions"]
    drafts, requests, parse_errors = request_details(generation, root, index, category=category)
    scores = score_rows(evaluation / "solver", category, expected_questions=ready,
                        expected_calls=pool_report["full_solver_calls"])
    duplicates, other_rejects, abstentions, review_rejects = [], [], [], []
    by_type = {}
    for outcome in report["roots"]:
        for rejection in outcome["mechanical_rejected"]:
            target = duplicates if any("duplicate" in error.lower() for error in rejection.get("errors", [])) else other_rejects
            target.append({"question_type": outcome["question_type"], **rejection})
        abstentions += [{"question_type": outcome["question_type"], **row} for row in outcome["abstentions"]]
        review_rejects += [{"question_type": outcome["question_type"], **row} for row in outcome["review_rejected"]]
        by_type[outcome["question_type"]] = outcome
    row = {"root_id": root, "chunk": index, "category": category,
           "generation_run": str(generation), "evaluation_run": str(evaluation), "selected_pool": str(pool_directory),
           "slots": report["candidate_slots"], "initial_drafts": sum(not q["repair"] for q in drafts),
           "repair_drafts": sum(q["repair"] for q in drafts), "requests_prepared": len(requests),
           "http_attempts": sum(q["remote_http_attempt"] for q in requests),
           "local_capacity_blocked_requests": sum(q["local_capacity_blocked"] for q in requests),
           "remote_error_requests": sum(q["remote_http_attempt"] and bool(q["error"]) for q in requests),
           "mechanical_pass": len(mechanical), "ready": len(ready), "selected": len(selected),
           "duplicate_rejected_slots": len(duplicates), "other_mechanical_rejected_slots": len(other_rejects),
           "abstained_slots": len(abstentions), "review_rejected_slots": len(review_rejects),
           "solver_calls": pool_report["full_solver_calls"], "selected_by_type": dict(Counter(q["question_type"] for q in selected)),
           "duplicate_rejections": duplicates, "other_mechanical_rejections": other_rejects,
           "abstentions": abstentions, "review_rejections": review_rejects, "by_type": by_type,
           "raw_parse_errors": parse_errors, "raw_initial_mc_support_count_histogram": raw_histogram(drafts, False),
           "raw_repair_mc_support_count_histogram": raw_histogram(drafts, True),
           "mechanical_distribution": distribution(mechanical), "ready_distribution": distribution(ready),
           "selected_distribution": distribution(selected)}
    require(row["http_attempts"] == report["http_attempts"] and len(mechanical) == report["mechanical_pass"]
            and len(ready) == report["ready_for_solver"] and len(selected) == pool_report["new_unique_added"],
            "chunk_actual_counts_mismatch")
    if "requests_prepared" in report:
        require(len(requests) == report["requests_prepared"], "chunk_request_count_mismatch")
    return row, mechanical, ready, selected, drafts, requests, scores


def collect_capacity_supplement(chunks):
    directory = HERE / "supplement"
    require((directory / "generation.COMPLETE.json").is_file(), "capacity_supplement_generation_not_complete")
    require(not (directory / "generation.ERROR.json").exists(), "capacity_supplement_generation_recorded_error")
    read(directory / "generation.COMPLETE.json")
    manifest = read(directory / "MANIFEST.json")
    evaluation = Path(manifest["evaluation_run"])
    pool_directory, recovery = supplement_finalization(directory, evaluation)
    same_ids = manifest["same_unattempted_candidate_ids"]
    require(manifest["candidate_slots"] == 2 and manifest["additional_distinct_slots"] == 0
            and len(same_ids) == len(set(same_ids)) == 2, "capacity_supplement_scope_mismatch")
    original = [row for row in chunks if row["generation_run"] == manifest["original_generation_run"]]
    require(len(original) == 1 and original[0]["root_id"] == manifest["root_id"],
            "capacity_supplement_original_chunk_not_found")
    blocked = [row for row in original[0]["other_mechanical_rejections"] if row["candidate_id"] in same_ids]
    require({row["candidate_id"] for row in blocked} == set(same_ids)
            and all(row["errors"] == ["LocalCapacityBlocked"] for row in blocked),
            "capacity_supplement_ids_not_original_unattempted_slots")
    value = collect_chunk(manifest["root_id"], "c05_capacity_supplement", Path(manifest["generation_run"]),
                          evaluation, category="new_v55_capacity_supplement", pool_directory=pool_directory)
    row, mechanical, ready, selected, drafts, requests, scores = value
    require(row["slots"] == 2 and {q["candidate_id"] for q in drafts} == set(same_ids),
            "capacity_supplement_actual_slot_ids_mismatch")
    require(not any(q["repair"] for q in drafts), "capacity_supplement_unexpected_repair")
    require(all(q["candidate_id"] in same_ids for q in mechanical + ready + selected),
            "capacity_supplement_unplanned_question")
    if recovery is not None:
        require(row["selected"] == 0 and row["ready"] == recovery["evaluated"]
                and row["solver_calls"] == recovery["full_solver_calls"]
                and all(not q["selected_by_policy"] and not q["data_errors"]
                        and q["actual_full_calls"] == q["expected_full_calls"] for q in scores),
                "capacity_supplement_recovery_result_mismatch")
    row.update(manifest=binding(directory / "MANIFEST.json"),
               same_unattempted_candidate_ids=same_ids, additional_distinct_slots=0,
               superseded_original_capacity_rejected_slots=len(blocked),
               original_generation_run=manifest["original_generation_run"],
               original_blocked_request=manifest["original_blocked_request"],
               original_capacity_rejections_preserved=blocked,
               finalization_recovery=recovery,
               accounting="same_two_planned_slots_with_final_outcomes_replacing_original_capacity_rejections")
    return value


def supplement_finalization(directory, evaluation):
    complete, error = directory / "evaluation.COMPLETE.json", directory / "evaluation.ERROR.json"
    if complete.is_file():
        require(not error.exists(), "capacity_supplement_evaluation_recorded_error")
        read(complete)
        return evaluation / "training_pool", None
    marker_path = directory / "FINALIZATION_RECOVERY.COMPLETE.json"
    require(marker_path.is_file() and error.is_file(), "capacity_supplement_finalization_not_complete")
    marker = read(marker_path)
    require(marker["status"] == "COMPLETE" and marker["recovery_kind"] == "offline_export_only"
            and marker["new_solver_calls"] == marker["new_http_calls"] == 0
            and marker["original_failure_preserved"] is True
            and Path(marker["evaluation_run"]).resolve() == evaluation.resolve() and marker["selected"] == 0
            and marker["models_called_by_recovery"] == 0 and marker["training_started"] is False
            and marker["stock_evaluation_complete_marker_created"] is False,
            "capacity_supplement_finalization_recovery_contract_mismatch")
    verify_bindings([marker["evaluation_error_preserved"], marker["alternate_pool"]]
                    + marker["inputs"] + marker["outputs"])
    require(Path(marker["evaluation_error_preserved"]["path"]).resolve() == error.resolve(),
            "capacity_supplement_recovery_wrong_error_binding")
    pool_manifest_path = Path(marker["alternate_pool"]["path"]).resolve()
    require(pool_manifest_path == (evaluation / "alternate_finalization_empty_pool/MANIFEST.json").resolve(),
            "capacity_supplement_recovery_wrong_pool_binding")
    require(Path(marker["alternate_pool_directory"]).resolve() == pool_manifest_path.parent,
            "capacity_supplement_recovery_directory_mismatch")
    recovery_inputs = {Path(row["path"]).resolve() for row in marker["inputs"]}
    require({(evaluation / "solver/COMPLETE.json").resolve(), (evaluation / "solver/RESULTS.json").resolve(),
             error.resolve()} <= recovery_inputs, "capacity_supplement_recovery_inputs_incomplete")
    pool_manifest = read(pool_manifest_path)
    nested_bindings = [pool_manifest[key] for key in (
        "pipeline_manifest", "generation_manifest", "solver_manifest", "solver_results", "stock_evaluation_error")]
    verify_bindings(nested_bindings + pool_manifest["files"])
    require(Path(pool_manifest["solver_results"]["path"]).resolve() == (evaluation / "solver/RESULTS.json").resolve()
            and pool_manifest["stock_evaluation_error"] == marker["evaluation_error_preserved"]
            and pool_manifest["selection_status"] == "EMPTY_BY_POLICY"
            and pool_manifest["selected_question_count"] == 0
            and pool_manifest["evaluated_question_count"] == marker["evaluated"]
            and pool_manifest["full_solver_calls"] == marker["full_solver_calls"]
            and pool_manifest["alternate_finalization_does_not_replace_stock_error"] is True,
            "capacity_supplement_recovery_bound_pool_mismatch")
    pool_directory = pool_manifest_path.parent
    bound_files = {Path(row["path"]).resolve() for row in pool_manifest["files"]}
    require({pool_directory / "REPORT.json", pool_directory / "SELECTED_QUESTIONS.json"} <= bound_files,
            "capacity_supplement_recovery_pool_outputs_not_bound")
    solver_complete = read(evaluation / "solver/COMPLETE.json")
    require(solver_complete["status"] == "COMPLETE"
            and solver_complete["actual_call_count"] == marker["full_solver_calls"],
            "capacity_supplement_recovery_solver_incomplete")
    failure = read(error)
    require(failure["stage"] == "evaluation" and failure["error_type"] == "CalledProcessError"
            and "'finalize'" in failure["error"], "capacity_supplement_recovery_not_finalization_failure")
    return pool_directory, {**marker, "marker": binding(marker_path),
                            "offline_export_only": True, "original_evaluation_error_retained": True,
                            "original_evaluation_error": binding(error)}


def merge_selected(original, supplement):
    """Apply the existing exact statement-set rule only to the report's combined view."""
    merged = list(original)
    by_key = {batch.pipeline.assembler.statement_key(question): question for question in original}
    duplicates = []
    for question in supplement:
        key = batch.pipeline.assembler.statement_key(question)
        if key in by_key:
            retained = by_key[key]
            duplicates.append({"candidate_id": question["candidate_id"], "retained_candidate_id": retained["candidate_id"],
                               "reason": "SAME_ROOT_TYPE_NORMALIZED_FULL_STATEMENT_SET",
                               "category": "capacity_supplement_combined_report",
                               "gold_content_disagreement": batch.pipeline.assembler.canonical_gold_statements(question)
                                   != batch.pipeline.assembler.canonical_gold_statements(retained)})
        else:
            by_key[key] = question
            merged.append(question)
    return merged, duplicates


def aggregate_scores(rows):
    valid = [r for r in rows if r["exact_correct_calls"] is not None and r["full_partial_mean"] is not None]
    calls = sum(row["actual_full_calls"] for row in valid)
    exact = sum(row["exact_correct_calls"] for row in valid)
    partial = sum((Fraction(row["full_partial_mean"]) * row["actual_full_calls"] for row in valid), Fraction())
    return {"scored_questions": len(valid), "actual_full_calls": calls, "exact_correct_calls": exact,
            "exact_rate": str(Fraction(exact, calls)) if calls else None,
            "weighted_partial_mean": str(partial / calls) if calls else None,
            "exact_rate_percent": round(100 * exact / calls, 2) if calls else None}


def classify_content(mechanical, selected, historical_selected):
    """Report a reviewed subset; leave every generation/evaluation pool intact."""
    path = HERE / "INDEPENDENT_CONTENT_AUDIT.json"
    audit = read(path)
    rows = audit["questions"]
    by_id = {row["candidate_id"]: row for row in rows}
    require(len(by_id) == len(rows), "content_audit_duplicate_candidate_id")
    require(set(by_id) == {q["candidate_id"] for q in mechanical}, "content_audit_incomplete_mechanical_coverage")
    allowed = {"CORE_USABLE", "KNOWN_SUBSTANTIVE_ISSUE"}
    for question in mechanical:
        row = by_id[question["candidate_id"]]
        require(row["core_status"] in allowed and row["root_id"] == question["root_id"]
                and row["question_type"] == question["question_type"], "content_audit_identity_or_status_invalid")
    usable = [q for q in selected if by_id[q["candidate_id"]]["core_status"] == "CORE_USABLE"]
    issues = [q for q in selected if by_id[q["candidate_id"]]["core_status"] == "KNOWN_SUBSTANTIVE_ISSUE"]
    historical_audit_path = HERE / "recovery/CONTENT_AUDIT.json"
    historical_audit = read(historical_audit_path)
    historical_by_id = {row["candidate_id"]: row for row in historical_audit["questions"]}
    for question in historical_selected:
        row = historical_by_id[question["candidate_id"]]
        require(row["recovery_decision"] == "ELIGIBLE_FOR_SOLVER" and not row.get("core_semantic_findings"),
                "historical_selected_not_core_usable")
    result = {"new_machine_selected": len(selected), "new_core_usable_selected": len(usable),
              "new_known_substantive_issue_selected": len(issues),
              "new_core_usable_selected_ids": [q["candidate_id"] for q in usable],
              "new_known_substantive_issue_selected_ids": [q["candidate_id"] for q in issues],
              "new_core_usable_selected_by_type": dict(Counter(q["question_type"] for q in usable)),
              "new_core_usable_selected_by_root": dict(Counter(q["root_id"] for q in usable)),
              "new_known_substantive_issue_selected_by_root": dict(Counter(q["root_id"] for q in issues)),
              "historical_core_usable_selected": len(historical_selected),
              "historical_core_usable_selected_ids": [q["candidate_id"] for q in historical_selected],
              "total_core_usable_selected_new_plus_historical": len(usable) + len(historical_selected),
              "new_mechanical_core_status_counts": dict(Counter(row["core_status"] for row in rows)),
              "new_core_usable_selected_distribution": distribution(usable),
              "content_audit": binding(path), "historical_content_audit": binding(historical_audit_path),
              "scope": "report_subset_of_machine_selected_candidates_after_independent_content_inspection",
              "original_machine_pools_unchanged": True, "original_question_and_gold_unchanged": True,
              "new_pipeline_gate": False, "formal_training_admission": False, "training_allowed": False}
    return result, usable


def histogram_text(value):
    hist = value.get("multi_correct_count_histogram", {})
    return "、".join(f"{key} 项×{count}" for key, count in sorted(hist.items(), key=lambda kv: int(kv[0]))) or "无多选题"


def score_markdown(scores):
    lines = ["# v55 Solver 每题得分", "", "完全正确率与部分得分均来自真实 Solver 调用；历史补评与本轮新题分别列示。", "",
             "| 来源 | 题目 ID | 题型 | 完全正确 | 部分得分均值 | 机器入选 |", "|---|---|---|---:|---:|---|"]
    for row in scores:
        exact = (f"{row['exact_correct_calls']}/{row['actual_full_calls']}（{row['exact_rate_percent']}%）"
                 if row["exact_correct_calls"] is not None else "无有效分数")
        partial = f"{row['full_partial_mean']}（{row['partial_score_percent']}%）" if row["full_partial_mean"] is not None else "无有效分数"
        label = {"historical_v54_recovery": "历史补评", "new_v55_capacity_supplement": "本轮容量补跑"}.get(
            row["category"], "本轮新题")
        lines.append(f"| {label} | {row['candidate_id']} | "
                     f"{'多选' if row['question_type'] == 'multi_select' else '排序'} | {exact} | {partial} | "
                     f"{'是' if row['selected_by_policy'] else '否'} |")
    lines += ["", "机器入选条件沿用既有规则：有效完整调用且部分得分均值严格介于 0 和 1。",
              "多选每题 18 次（6 次随机呈现×3），排序每题 12 次。得分基于冻结 Gold，独立内容检查另列。"]
    return "\n".join(lines)


def report_markdown(summary):
    totals, historical = summary["totals"], summary["historical_recovery"]
    original, supplement = summary["original_collection_totals"], summary["capacity_supplement"]
    effective = summary["effective_yield"]
    lines = ["# v55 出题收集结果", "",
             f"本轮两个 root 共 {totals['slots']} 个出题名额，返回 {totals['initial_drafts']} 道初稿及 "
             f"{totals['repair_drafts']} 道修复稿；{totals['ready']} 道进入 Solver，{totals['selected']} 道机器入选。"
             f"入选新题中，{effective['new_core_usable_selected']} 道内容核心可用，"
             f"{effective['new_known_substantive_issue_selected']} 道有已知实质问题。", "",
             "| root | 名额 | 初稿/修复稿 | 机械通过 | Solver 题数/调用数 | 机器入选（多选/排序） | 核心可用/问题题 | 整题重复拒绝 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary["roots"]:
        lines.append(f"| {row['root_id'].rsplit('-', 1)[-1]} | {row['slots']} | {row['initial_drafts']}/{row['repair_drafts']} | "
                     f"{row['mechanical_pass']} | {row['ready']}/{row['solver_calls']} | {row['selected']}"
                     f"（{row['selected_by_type'].get('multi_select', 0)}/{row['selected_by_type'].get('ordering', 0)}） | "
                     f"{row['core_usable_selected']}/{row['known_substantive_issue_selected']} | "
                     f"{row['duplicate_rejected_slots']} |")
    lines += ["", f"新题多选的正确项数量：机械通过为 {histogram_text(summary['mechanical_distribution'])}；"
              f"进入 Solver 为 {histogram_text(summary['ready_distribution'])}；入选为 {histogram_text(summary['selected_distribution'])}。",
              f"内容核心可用的新入选多选题：{histogram_text(effective['new_core_usable_selected_distribution'])}。",
              "单题允许 2–6 项正确；批次全为六项正确仅做分布提示，不改 Gold、不加固定配额。", "",
              f"实际远程出题/排序审核请求 {totals['http_attempts']} 次；本地容量拦截 {totals['local_capacity_blocked_requests']} 次"
              f"（未发 HTTP），远程错误 {totals['remote_error_requests']} 次。新题 Solver 共 {totals['solver_calls']} 次。",
              f"原始收集得到 {original['ready']} 道 Solver 题；原始 40 个名额中，2 个因本地容量未实际发出。"
              f"这 2 个原名额随后补跑：{supplement['http_attempts']} 次 HTTP、{supplement['initial_drafts']} 道初稿，"
              f"{supplement['duplicate_rejected_slots']} 道整题重复，新增 {supplement['ready']} 道 Solver 题、"
              f"{supplement['selected']} 道机器入选。补跑复用原名额，仍为 40 个不同名额；原始容量拦截记录保留。",
              f"最终拒绝中，整题重复 {totals['duplicate_rejected_slots']} 个名额，其他机械问题 {totals['other_mechanical_rejected_slots']} 个，"
              f"弃答 {totals['abstained_slots']} 个，排序审核拒绝 {totals['review_rejected_slots']} 个。", "",
              f"历史 v54 补评另计：原先因选五被误拦的 2 道均通过更正后的数量规则；"
              f"{historical['evaluated']} 道补跑 {historical['solver_calls']} 次 Solver，入选 {historical['selected']} 道。"
              "另 1 道因把“任何地区”扩大为“任何校园”而保留为已知内容问题，没有改写原文或 Gold。", "",
              f"计入历史补评后，本次核心可用的入选候选共 {effective['total_core_usable_selected_new_plus_historical']} 道"
              f"（新题 {effective['new_core_usable_selected']} 道，历史补评 {effective['historical_core_usable_selected']} 道）。",
              "新题与历史补评都使用冻结的完整对话；本轮没有重新扩充 A/B，没有把本轮 Solver 得分传回出题器。",
              "核心可用子集来自独立内容检查，原机器候选池、题目和 Gold 均保留。该子集仍未正式准入训练，没有新增流水线门槛或启动训练。", "",
              "[全部机械通过题](ALL_MECHANICAL_QUESTIONS.md) · [机器入选题](SELECTED_QUESTIONS.md) · "
              "[核心可用入选题](USABLE_SELECTED_QUESTIONS.md) · "
              "[逐题分数](SOLVER_SCORES.md) · [完整统计](RESULT_SUMMARY.json)"]
    if summary["content_review_artifacts"]:
        lines += ["", "独立内容检查：" + "、".join(f"[{Path(row['path']).name}]({Path(row['path']).name})"
                                                   for row in summary["content_review_artifacts"])]
    if supplement.get("finalization_recovery") is not None:
        lines += ["", "补跑题的 Solver 已完成；原导出程序处理空入选池时出现 ordering 计数字段错误。"
                  "已离线重放既有得分并生成独立的空入选池，新增模型调用为 0；原错误、原得分和部分导出文件均保留。"
                  "[离线导出修复记录](supplement/FINALIZATION_RECOVERY.COMPLETE.json)"]
    return "\n".join(lines)


def main():
    require((HERE / "evaluation.COMPLETE.json").is_file(), "new_collection_evaluation_not_complete")
    require(not (HERE / "evaluation.ERROR.json").exists(), "new_collection_evaluation_recorded_error")
    require((HERE / "recovery/RESULT_SUMMARY.json").is_file(), "historical_recovery_not_finalized")
    require((HERE / "INDEPENDENT_CONTENT_AUDIT.json").is_file(), "independent_content_audit_not_exported")
    read(HERE / "evaluation.COMPLETE.json")
    read(HERE / "generation.COMPLETE.json")
    plan = read(HERE / "EXPERIMENT_PLAN.json")
    roots, chunks, mechanical, ready, selected, drafts, requests, scores = [], [], [], [], [], [], [], []
    for planned in plan["roots"]:
        run = Path(planned["batch_run"])
        batch.verify(run)
        root_chunks, root_mechanical, root_ready, root_drafts, root_scores = [], [], [], [], []
        config = read(run / "CONFIG.json")
        require(config["chunk_count"] == 5 and config["budget"]["candidate_slots"] == 20, "unexpected_collection_size")
        for index in range(1, 6):
            generation, evaluation = batch.chunk_paths(run, index)
            row, qm, qr, qs, raw, req, scored = collect_chunk(planned["root_id"], index, generation, evaluation)
            chunks.append(row); root_chunks.append(row)
            mechanical += qm; root_mechanical += qm
            ready += qr; root_ready += qr
            drafts += raw; root_drafts += raw
            requests += req; scores += scored; root_scores += scored
        pool_report = read(run / "training_pool/REPORT.json")
        chosen = read(run / "training_pool/SELECTED_QUESTIONS.json")["questions"]
        selected += chosen
        root_row = {"root_id": planned["root_id"], "batch_run": str(run),
                    **{key: sum(row[key] for row in root_chunks) for key in TOTAL_KEYS},
                    "selected": len(chosen), "selected_before_combined_dedup": sum(row["selected"] for row in root_chunks),
                    "combined_pool_duplicates": pool_report["duplicates"],
                    "selected_by_type": dict(Counter(q["question_type"] for q in chosen)),
                    "raw_initial_mc_support_count_histogram": raw_histogram(root_drafts, False),
                    "raw_repair_mc_support_count_histogram": raw_histogram(root_drafts, True),
                    "mechanical_distribution": distribution(root_mechanical), "ready_distribution": distribution(root_ready),
                    "selected_distribution": distribution(chosen), "solver_score_aggregate": aggregate_scores(root_scores)}
        require(root_row["selected"] == pool_report["new_unique_added"] and root_row["solver_calls"] == pool_report["full_solver_calls"],
                "combined_root_counts_mismatch")
        roots.append(root_row)
    original_collection_totals = {key: sum(row[key] for row in roots) for key in TOTAL_KEYS}
    original_collection_roots = json.loads(json.dumps(roots))
    supplement, sqm, sqr, sqs, sraw, sreq, sscored = collect_capacity_supplement(chunks)
    require(not ({q["candidate_id"] for q in mechanical} & {q["candidate_id"] for q in sqm}),
            "capacity_supplement_repeats_already_generated_candidate_id")
    mechanical += sqm; ready += sqr; drafts += sraw; requests += sreq; scores += sscored
    selected, supplement_selected_duplicates = merge_selected(selected, sqs)
    supplement["combined_pool_duplicates"] = supplement_selected_duplicates
    supplement["combined_unique_selected_added"] = len(sqs) - len(supplement_selected_duplicates)
    for root_row in roots:
        if root_row["root_id"] != supplement["root_id"]:
            continue
        root_row["original_collection_totals"] = {key: root_row[key] for key in TOTAL_KEYS}
        for key in TOTAL_KEYS:
            if key not in ("slots", "selected"):
                root_row[key] += supplement[key]
        root_row["other_mechanical_rejected_slots"] -= supplement["superseded_original_capacity_rejected_slots"]
        root_row["capacity_supplement_slots"] = supplement["slots"]
        root_row["additional_distinct_slots"] = supplement["additional_distinct_slots"]
        root_row["superseded_original_capacity_rejected_slots"] = supplement["superseded_original_capacity_rejected_slots"]
        root_row["selected_before_combined_dedup"] += supplement["selected"]
        root_row["combined_pool_duplicates"] += supplement_selected_duplicates
        root_mechanical = [q for q in mechanical if q["root_id"] == root_row["root_id"]]
        root_ready = [q for q in ready if q["root_id"] == root_row["root_id"]]
        root_selected = [q for q in selected if q["root_id"] == root_row["root_id"]]
        root_drafts = [q for q in drafts if q["root_id"] == root_row["root_id"]]
        root_scores = [q for q in scores if q["root_id"] == root_row["root_id"]]
        root_row.update(selected=len(root_selected), selected_by_type=dict(Counter(q["question_type"] for q in root_selected)),
                        raw_initial_mc_support_count_histogram=raw_histogram(root_drafts, False),
                        raw_repair_mc_support_count_histogram=raw_histogram(root_drafts, True),
                        mechanical_distribution=distribution(root_mechanical), ready_distribution=distribution(root_ready),
                        selected_distribution=distribution(root_selected), solver_score_aggregate=aggregate_scores(root_scores))
    historical = read(HERE / "recovery/RESULT_SUMMARY.json")
    historical_questions = read(HERE / "recovery/RECOVERY_POOL.json")["questions"]
    historical_selected = read(HERE / "recovery/SELECTED_QUESTIONS.json")["questions"]
    historical_scores = score_rows(HERE / "recovery/solver", "historical_v54_recovery",
                                  expected_questions=historical_questions, expected_calls=historical["solver_calls"])
    scores += historical_scores
    effective_yield, usable_selected = classify_content(mechanical, selected, historical_selected)
    for row in roots:
        row["core_usable_selected"] = effective_yield["new_core_usable_selected_by_root"].get(row["root_id"], 0)
        row["known_substantive_issue_selected"] = effective_yield["new_known_substantive_issue_selected_by_root"].get(row["root_id"], 0)
    content_paths = [HERE / name for name in ("INDEPENDENT_CONTENT_AUDIT.json", "INDEPENDENT_CONTENT_AUDIT.md",
                                               "CONTENT_AUDIT.json", "CONTENT_AUDIT.md") if (HERE / name).is_file()]
    INPUTS.update(content_paths)
    summary = {"status": "LIVE_40_SLOT_COLLECTION_WITH_CAPACITY_SUPPLEMENT_AND_HISTORICAL_RECOVERY_COMPLETE", "roots": roots,
               "totals": {key: sum(row[key] for row in roots) for key in TOTAL_KEYS}, "chunks": chunks,
               "original_collection_totals": original_collection_totals, "original_collection_roots": original_collection_roots,
               "capacity_supplement": supplement,
               "distinct_planned_slots": plan["candidate_slots"], "additional_distinct_slots": 0,
               "final_rejection_counts_replace_two_original_capacity_rejections": True,
               "local_capacity_blocked_request_count_preserves_original_attempt_history": True,
               "raw_initial_mc_support_count_histogram": raw_histogram(drafts, False),
               "raw_repair_mc_support_count_histogram": raw_histogram(drafts, True),
               "mechanical_distribution": distribution(mechanical), "ready_distribution": distribution(ready),
               "selected_distribution": distribution(selected), "historical_recovery": historical,
               "effective_yield": effective_yield,
               "combined_selected_new_plus_historical": len(selected) + len(historical_selected),
               "combined_solver_calls_new_plus_historical": sum(row["solver_calls"] for row in roots) + historical["solver_calls"],
               "solver_scores": scores,
               "new_solver_score_aggregate": aggregate_scores([s for s in scores if s["category"].startswith("new_v55")]),
               "content_review_artifacts": [binding(path) for path in content_paths],
               "content_review_is_separate_from_mechanical_selection": True,
               "machine_selection_is_not_semantic_certification": True, "new_global_semantic_gate": False,
               "historical_recovery_is_not_new_generation": True, "training_calls": 0, "training_allowed": False}
    require(summary["totals"]["slots"] == 40 and len(chunks) == 10, "collection_total_slots_mismatch")
    require(summary["totals"]["mechanical_pass"] == len(mechanical)
            and summary["totals"]["ready"] == len(ready) and summary["totals"]["selected"] == len(selected),
            "combined_collection_and_capacity_supplement_counts_mismatch")
    require(len({q["candidate_id"] for q in mechanical}) == len(mechanical)
            and len({q["candidate_id"] for q in ready}) == len(ready), "combined_pool_duplicate_candidate_id")
    views = [display_export.display_view(q) for q in mechanical]
    new_selected_views = [display_export.display_view(q) for q in selected]
    historical_views = [display_export.display_view(q) for q in historical_selected]
    usable_views = [display_export.display_view(q) for q in usable_selected]
    for questions, displays in ((mechanical, views), (selected, new_selected_views), (historical_selected, historical_views),
                               (usable_selected, usable_views)):
        for question, view in zip(questions, displays):
            display_export.replay_view(question, view)
    summary["mechanical_display"] = display_export.display_stats(views)
    summary["selected_display_new_only"] = display_export.display_stats(new_selected_views)
    summary["selected_display_historical_only"] = display_export.display_stats(historical_views)
    summary["usable_selected_display_new_only"] = display_export.display_stats(usable_views)
    write(HERE / "RESULT_SUMMARY.json", summary)
    write(HERE / "RAW_DRAFTS.json", drafts)
    write(HERE / "REQUEST_COUNTS.json", {"requests": requests,
          "remote_http_attempts": sum(row["remote_http_attempt"] for row in requests),
          "local_capacity_blocked_requests": sum(row["local_capacity_blocked"] for row in requests)})
    write(HERE / "ALL_MECHANICAL_CANDIDATES.json", mechanical)
    write(HERE / "ALL_READY_CANDIDATES.json", ready)
    write(HERE / "SELECTED_CANDIDATES.json", {"new_v55": selected, "historical_v54_recovery": historical_selected})
    write(HERE / "USABLE_SELECTED_CANDIDATES.json", {"new_v55": usable_selected,
          "historical_v54_recovery": historical_selected, "effective_yield": effective_yield,
          "original_question_and_gold_unchanged": True, "original_machine_pools_unchanged": True,
          "formal_training_admission": False, "training_allowed": False})
    write(HERE / "SOLVER_SCORES.json", {"questions": scores})
    write_text(HERE / "ALL_MECHANICAL_QUESTIONS.md", display_export.markdown(views, title="v55 新收集的机械通过题"))
    write_text(HERE / "SELECTED_QUESTIONS.md",
               display_export.markdown(new_selected_views, title="v55 新收集的机器入选题") + "\n\n---\n\n" +
               display_export.markdown(historical_views, title="v54 历史误拦题的补评入选结果（另计）"))
    write_text(HERE / "USABLE_SELECTED_QUESTIONS.md",
               "独立内容检查标记为核心可用的入选候选子集。题目和 Gold 未改写，原机器候选池未改动，仍未正式准入训练。\n\n" +
               display_export.markdown(usable_views, title="v55 内容核心可用的新入选题") + "\n\n---\n\n" +
               display_export.markdown(historical_views, title="v54 历史补评：核心可用入选题（另计）"))
    write_text(HERE / "SOLVER_SCORES.md", score_markdown(scores))
    write_text(HERE / "REPORT.md", report_markdown(summary))
    INPUTS.add(Path(__file__).resolve())
    write(HERE / "REPORT_BINDINGS.json", {"inputs": [binding(path) for path in sorted(INPUTS)],
                                          "outputs": [binding(path) for path in OUTPUTS]})
    print(json.dumps({"new_collection": summary["totals"], "historical_recovery": historical,
                      "selected_distribution": summary["selected_distribution"], "effective_yield": effective_yield}, ensure_ascii=False))


if __name__ == "__main__":
    main()
