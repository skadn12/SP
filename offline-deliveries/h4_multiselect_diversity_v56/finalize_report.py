"""Offline aggregate and display exports for completed v56 collection artifacts."""
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CODE = HERE.parent / 'h4_sequential_ab_v56_qa'
sys.path.insert(0, str(CODE))
import display_export
import evaluate_and_pool as pipeline
import audit_provenance
import run_collection as collection

INPUTS = set()
OUTPUTS = []
TOTAL_KEYS = ('slots', 'initial_drafts', 'repair_drafts', 'requests_prepared', 'http_attempts',
              'local_capacity_blocked_requests', 'remote_error_requests', 'mechanical_pass', 'ready',
              'duplicate_rejected_slots', 'other_mechanical_rejected_slots', 'abstained_slots',
              'review_rejected_slots', 'solver_calls', 'selected')

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

def request_details(generation, root, chunk, *, category="new_v56"):
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

def collect_chunk(root, index, generation, evaluation, *, category="new_v56", pool_directory=None):
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

def aggregate_scores(rows):
    valid = [r for r in rows if r["exact_correct_calls"] is not None and r["full_partial_mean"] is not None]
    calls = sum(row["actual_full_calls"] for row in valid)
    exact = sum(row["exact_correct_calls"] for row in valid)
    partial = sum((Fraction(row["full_partial_mean"]) * row["actual_full_calls"] for row in valid), Fraction())
    return {"scored_questions": len(valid), "actual_full_calls": calls, "exact_correct_calls": exact,
            "exact_rate": str(Fraction(exact, calls)) if calls else None,
            "weighted_partial_mean": str(partial / calls) if calls else None,
            "exact_rate_percent": round(100 * exact / calls, 2) if calls else None}

def histogram_text(value):
    hist = value.get("multi_correct_count_histogram", {})
    return "、".join(f"{key} 项×{count}" for key, count in sorted(hist.items(), key=lambda kv: int(kv[0]))) or "无多选题"


def distribution(questions):
    value = display_export.multi_answer_distribution(questions)
    multi = [q for q in questions if q['question_type'] == 'multi_select']
    correct, total = sum(len(q['gold']) for q in multi), 6 * len(multi)
    all_six = value['multi_all_six']
    value.update(correct_options=correct, total_options=total,
        correct_option_share=str(Fraction(correct, total)) if total else None,
        correct_option_share_percent=round(correct * 100 / total, 2) if total else None,
        all_six_rate=str(Fraction(all_six, len(multi))) if multi else None,
        all_six_rate_percent=round(all_six * 100 / len(multi), 2) if multi else None,
        overall_question_count=len(questions),
        questions_by_type=dict(Counter(q['question_type'] for q in questions)))
    return value


def deduplicate_exact(questions):
    """Use the existing same-root/type/whole-statement-set selected-pool policy."""
    seen, kept, duplicates = {}, [], []
    for question in questions:
        key = pipeline.assembler.statement_key(question)
        if key in seen:
            retained = seen[key]
            duplicates.append({'candidate_id': question['candidate_id'],
                'retained_candidate_id': retained['candidate_id'],
                'reason': 'SAME_ROOT_TYPE_NORMALIZED_FULL_STATEMENT_SET',
                'gold_content_disagreement': pipeline.assembler.canonical_gold_statements(question)
                    != pipeline.assembler.canonical_gold_statements(retained)})
        else:
            seen[key] = question
            kept.append(question)
    return kept, duplicates


def content_subset(mechanical, selected):
    path = HERE / 'INDEPENDENT_CONTENT_AUDIT.json'
    audit = read(path)
    rows = audit['questions']
    by_id = {row['candidate_id']: row for row in rows}
    require(len(by_id) == len(rows) and set(by_id) == {q['candidate_id'] for q in mechanical},
            'content_audit_not_exactly_complete_mechanical_pool')
    statuses = {}
    for question in mechanical:
        row = by_id[question['candidate_id']]
        detail = row.get('audit', row)
        status = row.get('core_status', detail.get('core_status'))
        require(status in ('CORE_USABLE', 'KNOWN_SUBSTANTIVE_ISSUE'), 'unexpected_content_status')
        require(row['root_id'] == question['root_id'] and row['question_type'] == question['question_type'],
                'content_audit_identity_changed')
        if detail.get('source_sha256'):
            require(detail['source_sha256'] == question['source_sha256'], 'content_audit_source_changed')
        require(detail.get('checked_without_solver_scores', audit.get('solver_scores_used') is False),
                'content_audit_used_solver_scores')
        statuses[question['candidate_id']] = status
    usable = [q for q in selected if statuses[q['candidate_id']] == 'CORE_USABLE']
    issues = [q for q in selected if statuses[q['candidate_id']] == 'KNOWN_SUBSTANTIVE_ISSUE']
    return usable, issues, statuses


def score_markdown(scores, statuses):
    lines = ['# v56 Solver 逐题得分', '',
             '| 题目 ID | 正确项数 | 完全正确 | 部分得分均值 | 机器入选 | 内容状态 |',
             '|---|---:|---:|---:|---|---|']
    for row in scores:
        exact = f"{row['exact_correct_calls']}/{row['actual_full_calls']}（{row['exact_rate_percent']}%）"
        partial = f"{row['full_partial_mean']}（{row['partial_score_percent']}%）"
        lines.append(f"| {row['candidate_id']} | {row['correct_option_count']} | {exact} | {partial} | "
                     f"{'是' if row['selected_by_policy'] else '否'} | {statuses[row['candidate_id']]} |")
    lines.extend(['', '每题 18 次完整上下文回答（6 次随机呈现×3）；完全正确率与部分得分分别统计。',
                  '机器入选沿用有效完整调用且 0 < 部分得分均值 < 1；得分基于冻结 Gold。内容可用性单独记录。'])
    return '\n'.join(lines)


def export_questions(stem, questions, title, provenance):
    views = [display_export.display_view(q) for q in questions]
    for question, view in zip(questions, views):
        display_export.replay_view(question, view)
        audit_provenance.independent_display_check(question, view)
    write(HERE / (stem + '_CANDIDATES.json'), {'questions': questions,
        'candidate_count': len(questions), 'original_question_and_gold_unchanged': True,
        'original_machine_pools_unchanged': True, 'training_allowed': False,
        'formal_training_admission': False, 'provenance': provenance})
    write(HERE / (stem + '_DISPLAY.json'), {'questions': views,
        'canonical_file': stem + '_CANDIDATES.json', 'presentation_only': True,
        'gold_blind_permutation_replayed': True})
    write_text(HERE / (stem + '_QUESTIONS.md'), display_export.markdown(views, title=title))
    return display_export.display_stats(views)


def report_markdown(summary):
    totals, before, after = summary['totals'], summary['before_distribution'], summary['merged_distribution']
    new = summary['usable_selected_distribution']
    lines = ['# v56 多选答案分布补充结果', '',
        f"本轮在 root 138343 的冻结 A+B 完整对话上出 {totals['slots']} 个多选名额，"
        f"实际返回 {totals['initial_drafts']} 道初稿和 {totals['repair_drafts']} 道修复稿；"
        f"{totals['ready']} 道进入 Solver，{summary['new_selected_unique']} 道机器入选，"
        f"其中 {summary['new_usable_selected']} 道内容核心可用。", '',
        '| 阶段 | 多选题数 | 正确项数量 | 全选比例 | 正确选项占比 |', '|---|---:|---|---:|---:|']
    for label, value in [('原 v55 可用多选', before), ('本轮新增可用多选', new), ('合并后可用多选', after)]:
        six = '无' if value['all_six_rate_percent'] is None else f"{value['all_six_rate_percent']}%"
        share = '无' if value['correct_option_share_percent'] is None else f"{value['correct_options']}/{value['total_options']}（{value['correct_option_share_percent']}%）"
        lines.append(f"| {label} | {value['multi_select']} | {histogram_text(value)} | {six} | {share} |")
    low = summary['new_usable_two_to_four_count']
    lines.extend(['', f"本轮新增可用题中选二/选三/选四共 {low} 道。"
        + ('本次补充获得了此前缺少的答案数量。' if low else '本次尚未补到有效的选二、选三、选四，答案分布改进目标未达成。'),
        '每题仍允许 2–6 个正确项；软性采样偏好没有改写 Gold，也没有新增数量拒绝规则。', '',
        f"实际 HTTP {totals['http_attempts']} 次、Solver {totals['solver_calls']} 次；"
        f"整题重复拒绝 {totals['duplicate_rejected_slots']} 个名额，其他机械拒绝 {totals['other_mechanical_rejected_slots']}，"
        f"弃答 {totals['abstained_slots']}；机器入选中 {summary['new_known_issue_selected']} 道有已知内容问题，单独保留。",
        f"本轮入选题汇总去除 {len(summary['new_selected_exact_duplicates'])} 道完整集合重复；"
        f"与原 v55 的 19 道可用题合并再去除 {len(summary['merged_exact_duplicates'])} 道，"
        f"最终 {summary['merged_usable_count']} 道（多选 {after['questions_by_type'].get('multi_select',0)}、"
        f"排序 {after['questions_by_type'].get('ordering',0)}）。", '',
        '本轮未重新扩充 A/B，root 138343 是相对 v55 两个 root 的第三个已扩充 root；未宣称新增多个 root。',
        '全部出题先完成，再运行 Solver；本轮 Solver 回答和分数未送回出题器。仍为候选导出，未启动训练。', '',
        '[新增可用题](USABLE_SELECTED_QUESTIONS.md) · [合并可用题](MERGED_USABLE_QUESTIONS.md) · '
        '[逐题得分](SOLVER_SCORES.md) · [全部机械通过题](ALL_MECHANICAL_QUESTIONS.md) · '
        '[机器入选题](SELECTED_QUESTIONS.md) · [来源核验](PROVENANCE_AUDIT.md)'])
    recovery = summary['evaluation_recovery']
    if recovery['mode'] != 'ordinary_completion':
        lines.extend(['', f"评估首个 chunk 完成 {recovery['reused_completed_calls']} 次回答后在清理阶段中断。"
                      f"已离线复用这些回答，并完成其余 {recovery['new_solver_calls']} 次；重复 Solver 调用为 0。"
                      '原中断记录和已完成结果均保留。'])
    lines.extend(['', '内容审阅独立于 Solver 得分；首个 chunk 的 Solver 已在全部 11 题审阅汇总完成前启动，'
                  '因此不能声称全部内容审阅先于首次 Solver。后续五批评分在汇总审阅冻结后继续。'])
    return '\n'.join(lines)


def main():
    plan = collection.verify_plan()
    collection.completion('generation')
    _, evaluation_recovery = audit_provenance.evaluation_completion()
    require((HERE / 'INDEPENDENT_CONTENT_AUDIT.json').is_file(), 'content_audit_not_complete')
    require(not (HERE / 'RESULT_SUMMARY.json').exists(), 'immutable_final_report_already_exists')
    audit = audit_provenance.run_audit('evaluation')
    chunks, mechanical, ready, selected_raw, drafts, requests, scores = [], [], [], [], [], [], []
    for chunk in plan['chunks']:
        generation, evaluation = Path(chunk['generation_run']), Path(chunk['evaluation_run'])
        row, qm, qr, qs, raw, req, scored = collect_chunk(plan['root_id'], chunk['index'], generation, evaluation)
        chunks.append(row); mechanical += qm; ready += qr; selected_raw += qs
        drafts += raw; requests += req; scores += scored
    totals = {key: sum(row[key] for row in chunks) for key in TOTAL_KEYS}
    require(totals['slots'] == 12 and totals['http_attempts'] <= 12 and totals['solver_calls'] <= 216,
            'final_actual_budget_mismatch')
    selected, selected_duplicates = deduplicate_exact(selected_raw)
    usable, issues, statuses = content_subset(mechanical, selected)
    join_path = HERE / 'CONTENT_SELECTION_JOIN.json'
    require(join_path.is_file(), 'independent_content_selection_join_missing')
    join = read(join_path)
    require(join['audit_sha256'] == binding(HERE / 'INDEPENDENT_CONTENT_AUDIT.json')['sha256']
            and join['solver_scores_used_for_independent_content_labels'] is False
            and join['generation_or_gold_edited'] is False and join['new_gate'] is False,
            'content_selection_join_scope_or_binding_invalid')
    require(set(join['selected_and_core_usable_ids']) == {q['candidate_id'] for q in usable}
            and set(join['selected_with_known_issue_ids']) == {q['candidate_id'] for q in issues},
            'content_selection_join_selected_ids_mismatch')
    old_path = HERE.parent / 'h4_question_collection_v55/USABLE_SELECTED_CANDIDATES.json'
    old = read(old_path)
    baseline = old['new_v55'] + old.get('historical_v54_recovery', [])
    before = distribution(baseline)
    require(len(baseline) == 19 and before['multi_select'] == 7
            and before['multi_correct_count_histogram'] == {5: 4, 6: 3}
            and before['correct_options'] == 38 and before['total_options'] == 42,
            'v55_usable_baseline_changed')
    merged, merged_duplicates = deduplicate_exact(baseline + usable)
    score_by_id = {q['candidate_id']: q for q in ready}
    for score in scores:
        score['correct_option_count'] = len(score_by_id[score['candidate_id']]['gold'])
        score['independent_content_status'] = statuses[score['candidate_id']]
    summary = {'version': 'h4_multiselect_diversity_v56_completed_collection_v1',
        'root_id': plan['root_id'], 'totals': totals, 'chunks': chunks,
        'raw_initial_mc_support_count_histogram': raw_histogram(drafts, False),
        'raw_repair_mc_support_count_histogram': raw_histogram(drafts, True),
        'mechanical_distribution': distribution(mechanical), 'ready_distribution': distribution(ready),
        'selected_distribution': distribution(selected), 'usable_selected_distribution': distribution(usable),
        'before_distribution': before, 'merged_distribution': distribution(merged),
        'new_selected_unique': len(selected), 'new_selected_exact_duplicates': selected_duplicates,
        'new_usable_selected': len(usable), 'new_known_issue_selected': len(issues),
        'new_usable_two_to_four_count': sum(2 <= len(q['gold']) <= 4 for q in usable),
        'merged_usable_count': len(merged), 'merged_exact_duplicates': merged_duplicates,
        'new_selected_known_issue_ids': [q['candidate_id'] for q in issues],
        'new_usable_selected_ids': [q['candidate_id'] for q in usable],
        'score_aggregate': aggregate_scores(scores),
        'evaluation_recovery': evaluation_recovery,
        'content_audit': binding(HERE / 'INDEPENDENT_CONTENT_AUDIT.json'),
        'content_selection_join': binding(join_path),
        'baseline_usable_pool': binding(old_path), 'machine_selection_is_not_semantic_certification': True,
        'correct_option_count_is_not_admission_gate': True, 'full_question_records_preserved': True,
        'training_allowed': False, 'training_calls': 0, 'formal_training_admission': False}
    # All input verification and semantic subset accounting finish before outputs.
    if not (HERE / 'PROVENANCE_AUDIT.json').exists():
        audit_provenance.export_audit(audit)
    else:
        require(read(HERE / 'PROVENANCE_AUDIT.json')['status'] == 'PASS', 'prior_provenance_audit_failed')
    for path in ('PROVENANCE_AUDIT.json', 'PROVENANCE_AUDIT.md'):
        OUTPUTS.append(HERE / path)
    provenance = {'experiment_plan': binding(HERE / 'EXPERIMENT_PLAN.json'),
                  'content_audit': summary['content_audit'],
                  'content_selection_join': summary['content_selection_join'],
                  'baseline_usable_pool': summary['baseline_usable_pool']}
    summary['mechanical_display'] = export_questions('ALL_MECHANICAL', mechanical, 'v56 全部机械通过题', provenance)
    summary['selected_display'] = export_questions('SELECTED', selected, 'v56 机器入选题', provenance)
    summary['usable_selected_display'] = export_questions('USABLE_SELECTED', usable, 'v56 新增内容可用题', provenance)
    summary['merged_usable_display'] = export_questions('MERGED_USABLE', merged, 'v55＋v56 合并内容可用题', provenance)
    write(HERE / 'ALL_READY_CANDIDATES.json', {'questions': ready, 'training_allowed': False})
    write(HERE / 'RAW_DRAFTS.json', {'drafts': drafts, 'initial_count': totals['initial_drafts'],
        'repair_count': totals['repair_drafts'], 'raw_question_records_unchanged': True})
    write(HERE / 'REQUEST_COUNTS.json', {'requests': requests, 'totals': totals})
    write(HERE / 'SOLVER_SCORES.json', {'questions': scores, 'aggregate': summary['score_aggregate']})
    write_text(HERE / 'SOLVER_SCORES.md', score_markdown(scores, statuses))
    write(HERE / 'RESULT_SUMMARY.json', summary)
    write_text(HERE / 'REPORT.md', report_markdown(summary))
    verification = {'status': 'PASS', 'v55_bound_files_unchanged': audit['history']['v55_bound_files_unchanged'],
        'new_generation_slots': totals['slots'], 'remote_http_calls': totals['http_attempts'],
        'solver_calls': totals['solver_calls'], 'new_core_usable_selected': len(usable),
        'merged_core_usable': len(merged), 'full_records_and_gold_preserved': True,
        'independent_display_replay': True, 'training_calls': 0,
        'new_usable_two_to_four_count': summary['new_usable_two_to_four_count']}
    write(HERE / 'VERIFICATION.json', verification)
    INPUTS.update((HERE / name for name in ('EXPERIMENT_PLAN.json', 'generation.COMPLETE.json',
                                          'INDEPENDENT_CONTENT_AUDIT.json', 'CONTENT_SELECTION_JOIN.json',
                                          'CONTENT_SELECTION_JOIN.md')))
    INPUTS.add(Path(evaluation_recovery['completion']['path']))
    write(HERE / 'REPORT_BINDINGS.json', {'inputs': [binding(p) for p in sorted(INPUTS)],
        'outputs': [binding(p) for p in sorted(set(OUTPUTS))]})
    paths = sorted({p for directory in (HERE, CODE) for p in directory.rglob('*')
                    if p.is_file() and '__pycache__' not in p.parts and p.name != 'RESULT_BINDINGS.json'})
    write(HERE / 'RESULT_BINDINGS.json', {'scope': 'completed_v56_frozen_code_source_runs_reports_displays',
        'files': [binding(p) for p in paths]})
    print(json.dumps(verification, ensure_ascii=False))


if __name__ == '__main__':
    main()
