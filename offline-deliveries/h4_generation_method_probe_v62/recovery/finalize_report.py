"""Offline aggregate and display exports for completed v62 direct versus staged generation probe."""
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
RECOVERY = Path(__file__).resolve().parent
HERE = RECOVERY.parent
sys.path.insert(0, str(HERE))
CODE = HERE.parent / 'h4_generation_method_probe_v62_qa'
sys.path.insert(0, str(CODE))
import display_export
import evaluate_and_pool as pipeline
import importlib.util
_audit_spec = importlib.util.spec_from_file_location('v62_recovery_audit_for_final_report', RECOVERY / 'audit_provenance.py')
audit_provenance = importlib.util.module_from_spec(_audit_spec)
_audit_spec.loader.exec_module(audit_provenance)
import run_collection as collection
import comparison

INPUTS = set()
OUTPUTS = []
TOTAL_KEYS = ('slots', 'initial_drafts', 'repair_drafts', 'requests_prepared', 'http_attempts',
              'local_capacity_blocked_requests', 'remote_error_requests', 'mechanical_pass', 'ready',
              'duplicate_rejected_slots', 'other_mechanical_rejected_slots', 'abstained_slots',
              'review_rejected_slots', 'solver_calls', 'selected', 'interrupted_http_requests', 'interrupted_unobserved_slots')

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

def request_details(generation, root, chunk, *, category="new_v62"):
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
        response_for_usage = read(folder / "RESPONSE.json") if (folder / "RESPONSE.json").exists() else None
        metadata = response_for_usage.get("metadata", {}) if response_for_usage else {}
        usage = metadata.get("usage") or {}
        record["reported_input_tokens"] = metadata.get("input_tokens", usage.get("prompt_tokens"))
        record["reported_output_tokens"] = metadata.get("output_tokens", usage.get("completion_tokens"))
        record["reported_total_tokens"] = metadata.get("total_tokens", usage.get("total_tokens"))
        record["token_usage_status"] = ("known" if all(isinstance(record[k], int) for k in
            ("reported_input_tokens", "reported_output_tokens")) else "unknown") if has_http else "not_called"
        record["token_usage_source"] = "provider_response_metadata_no_local_estimation"
        interrupted = (generation.name == audit_provenance.recovery.LOST and has_http
            and not (folder / "HTTP_RESULT.json").exists())
        if interrupted:
            audit_provenance.recovery.verify()
            require(not (folder / "RESPONSE.json").exists() and error is None, "interruption_state_changed")
        record["interrupted_unobserved"] = interrupted
        record["remote_outcome"] = "unknown" if interrupted else "observed" if response_for_usage else "not_observed"
        record["remote_error_observed"] = bool(has_http and error)
        record["remote_completion_at_utc"] = None if interrupted else (
            read(folder / "HTTP_RESULT.json").get("http_completed_at_utc") if (folder / "HTTP_RESULT.json").exists() else None)
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

def collect_chunk(root, index, generation, evaluation, *, category="new_v62", pool_directory=None):
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
           "interrupted_http_requests": sum(r["interrupted_unobserved"] for r in requests),
           "interrupted_unobserved_slots": sum(len(o.get("interrupted_slots", [])) for o in report["roots"]),
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
    hist = value.get("multi_correct_count_histogram_two_to_six", value.get("multi_correct_count_histogram", {}))
    return "、".join(f"{key} 项×{count}" for key, count in sorted(hist.items(), key=lambda kv: int(kv[0]))) or "无多选题"


def distribution(questions):
    value = display_export.multi_answer_distribution(questions)
    multi = [q for q in questions if q['question_type'] == 'multi_select']
    correct, total = sum(len(q['gold']) for q in multi), 6 * len(multi)
    all_six = value['multi_all_six']
    counts = Counter(len(q['gold']) for q in multi)
    value.update(blind_all_six_exact_rate=str(Fraction(all_six, len(multi))) if multi else None,
        blind_all_six_partial_mean=str(Fraction(all_six, len(multi))) if multi else None,
        multi_correct_count_histogram_two_to_six={n: counts[n] for n in range(2, 7)},
        five_correct_count=counts[5],
        five_correct_rate_percent=round(100 * counts[5] / len(multi), 2) if multi else None,
        correct_options=correct, total_options=total,
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


def statement_overlap(questions, new_ids):
    """Descriptive shared-statement counts only; this never removes a question."""
    pairs = []
    for index, left in enumerate(questions):
        field = 'options' if left['question_type'] == 'multi_select' else 'items'
        left_texts = {' '.join(text.split()).casefold() for text in left[field].values()}
        for right in questions[index + 1:]:
            if left['root_id'] != right['root_id'] or left['question_type'] != right['question_type']:
                continue
            if not ({left['candidate_id'], right['candidate_id']} & new_ids):
                continue
            shared = left_texts & {' '.join(text.split()).casefold() for text in right[field].values()}
            if shared:
                pairs.append({'candidate_ids': [left['candidate_id'], right['candidate_id']],
                              'root_id': left['root_id'], 'question_type': left['question_type'],
                              'shared_normalized_statement_count': len(shared),
                              'shared_normalized_statements': sorted(shared)})
    return {'diagnostic_only_not_admission_gate': True, 'paraphrase_similarity_not_tested': True,
            'normalization': 'casefold_and_whitespace_only', 'pairs': pairs,
            'pairs_sharing_at_least_four_statements': sum(p['shared_normalized_statement_count'] >= 4 for p in pairs)}


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
    lines = ['# v62 Solver 逐题得分', '',
             '| 题目 ID | 题型 | Gold 项数 | 完全正确 | 部分得分均值 | 机器入选 | 内容状态 |',
             '|---|---|---:|---:|---:|---|---|']
    for row in scores:
        exact = f"{row['exact_correct_calls']}/{row['actual_full_calls']}（{row['exact_rate_percent']}%）"
        partial = f"{row['full_partial_mean']}（{row['partial_score_percent']}%）"
        lines.append(f"| {row['candidate_id']} | {row['question_type']} | {row['correct_option_count']} | {exact} | {partial} | "
                     f"{'是' if row['selected_by_policy'] else '否'} | {statuses[row['candidate_id']]} |")
    lines.extend(['', '本批仅多选，每题18次完整上下文回答（6次随机呈现×3）。完全正确率与部分得分分别统计。',
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


def expansion_summary(plan):
    path = Path(plan["history"]["path"])
    report_path = path.parent / "CANDIDATE_REPORT.json"
    upstream = read(report_path)
    metrics = upstream["metrics"]
    history = read(path)
    formal = metrics["formal_accepted_baseline"]
    routes = {key: metrics["by_route"][key]["retained"] for key in ("A", "B")}
    retained = metrics["retained"]
    cumulative = metrics["provisional_candidate_cumulative"]
    require(upstream["root_id"] == plan["root_id"] and history["root_id"] == plan["root_id"], "expansion_root_mismatch")
    require(history["formal_baseline_turns"] == formal["turns"]
            and len(history["turns"]) == cumulative["turns"]
            and retained["turns"] == sum(routes[k]["turns"] for k in routes)
            and retained["body_tokens"] == sum(routes[k]["body_tokens"] for k in routes)
            and formal["turns"] + retained["turns"] == cumulative["turns"]
            and formal["body_tokens"] + retained["body_tokens"] == cumulative["body_tokens"],
            "expansion_turns_or_body_tokens_mismatch")
    scene_refs = {key: sorted({turn.get("scene_ref") for turn in history["turns"]
                        if turn.get("scene_ref") and turn["scene_ref"].startswith("CANDIDATE:" + key + ":")})
                  for key in ("A", "B")}
    require(all(len(scene_refs[key]) == routes[key]["scenes"] for key in routes), "retained_scene_count_mismatch")
    return {"formal_original_plus_r01": formal, "retained_by_route": routes,
            "retained_scenes_by_route": scene_refs, "retained": retained,
            "full_provisional_cumulative": cumulative,
            "body_token_definition": metrics["body_token_definition"],
            "expansion_report": binding(report_path), "expansion_manifest": binding(path.parent / "MANIFEST.json"),
            "new_expansion_in_this_experiment": False, "history": binding(path)}, history["turns"]


def conversation_markdown(root, turns):
    lines = [f"# {root} 完整对话", "", "原文按 CANDIDATE_HISTORY.json 中的顺序逐轮显示。新增 A/B 仍为待审候选对话。", ""]
    for turn in turns:
        lines.extend([f"### {turn['turn_ref']} · {turn['speaker']}", "", turn["text"], ""])
    return "\n".join(lines)


def cost_summary(requests, usable_count=None):
    actual = [r for r in requests if r["remote_http_attempt"]]
    known = [r for r in actual if r["token_usage_status"] == "known"]
    complete = len(known) == len(actual)
    inputs = sum(r["reported_input_tokens"] for r in known)
    outputs = sum(r["reported_output_tokens"] for r in known)
    value = {"http_attempts": len(actual), "prepared_requests": len(requests),
        "known_usage_http_attempts": len(known), "unknown_usage_http_attempts": len(actual) - len(known),
        "usage_complete": complete, "input_tokens": inputs if complete else "unknown",
        "output_tokens": outputs if complete else "unknown",
        "known_input_token_subtotal": inputs, "known_output_token_subtotal": outputs,
        "money_cost": "unknown_no_frozen_provider_pricing", "planner_included": True}
    if usable_count is not None:
        value["content_usable_questions"] = usable_count
        value["http_per_content_usable_question"] = str(Fraction(len(actual), usable_count)) if usable_count else None
        value["input_tokens_per_content_usable_question"] = str(Fraction(inputs, usable_count)) if complete and usable_count else (None if not usable_count else "unknown")
        value["output_tokens_per_content_usable_question"] = str(Fraction(outputs, usable_count)) if complete and usable_count else (None if not usable_count else "unknown")
    return value


def arm_summary(plan, chunks, drafts, requests, mechanical, ready, selected_raw, statuses, baseline):
    result = {}
    baseline_keys = {pipeline.assembler.statement_key(q) for q in baseline}
    for arm in collection.ARMS:
        indexes = {c["index"] for c in plan["chunks"] if c["arm"] == arm}
        ids = {qid for c in plan["chunks"] if c["arm"] == arm for qid in c["candidate_ids"]}
        rows = [r for r in chunks if r["chunk"] in indexes]
        qm = [q for q in mechanical if q["candidate_id"] in ids]
        qr = [q for q in ready if q["candidate_id"] in ids]
        qs = [q for q in selected_raw if q["candidate_id"] in ids]
        core = [q for q in qm if statuses[q["candidate_id"]] == "CORE_USABLE"]
        usable = [q for q in qs if statuses[q["candidate_id"]] == "CORE_USABLE"]
        unique, duplicates = deduplicate_exact(usable)
        incremental = [q for q in unique if pipeline.assembler.statement_key(q) not in baseline_keys]
        req = [r for r in requests if r["chunk"] in indexes]
        raw = [r for r in drafts if r["chunk"] in indexes]
        result[arm] = {"totals": {key: sum(r[key] for r in rows) for key in TOTAL_KEYS},
            "raw_initial_mc_support_count_histogram": raw_histogram(raw, False),
            "raw_repair_mc_support_count_histogram": raw_histogram(raw, True),
            "mechanical_distribution": distribution(qm), "ready_distribution": distribution(qr),
            "planned_slots": sum(r['slots'] for r in rows),
            "execution_lost_slots": sum(r['interrupted_unobserved_slots'] for r in rows),
            "slots_without_execution_loss": sum(r['slots'] - r['interrupted_unobserved_slots'] for r in rows),
            "content_usable_before_solver": len(core), "content_usable_distribution": distribution(core),
            "difficulty_selected_distribution": distribution(qs),
            "difficulty_selected_and_content_usable": len(usable),
            "usable_selected_distribution": distribution(usable),
            "within_arm_unique_usable_selected": len(unique), "within_arm_exact_duplicates": duplicates,
            "unique_increment_if_this_arm_alone": len(incremental),
            "incremental_ids_if_this_arm_alone": [q["candidate_id"] for q in incremental],
            "cost": cost_summary(req, len(core)),
            "planner_cost": cost_summary([r for r in req if r["role"] == "evidence_planner"]),
            "composition_cost": cost_summary([r for r in req if r["role"] == "question_generator"]),
            "cross_arm_duplicate_retained_for_method_statistics": True}
    return result


def cross_arm_exact_overlap(plan, mechanical):
    arm_by_id = {qid: c["arm"] for c in plan["chunks"] for qid in c["candidate_ids"]}
    grouped = {}
    for q in mechanical:
        grouped.setdefault(pipeline.assembler.statement_key(q), []).append(q)
    return [{"root_id": group[0]["root_id"], "candidate_ids": [q["candidate_id"] for q in group],
             "arms": [arm_by_id[q["candidate_id"]] for q in group],
             "gold_content_disagreement": len({tuple(pipeline.assembler.canonical_gold_statements(q)) for q in group}) > 1}
            for group in grouped.values() if len({arm_by_id[q["candidate_id"]] for q in group}) > 1]


def matched_completed_method_comparison(plan, chunks, drafts, requests, mechanical, ready, selected, statuses, baseline, pairs):
    # c01 lacks a locally observed direct response. Only c02/c03 are matched here.
    kept = [c for c in plan['chunks'] if c['topic_chunk'] in (2, 3)]
    indexes = {c['index'] for c in kept}
    ids = {qid for c in kept for qid in c['candidate_ids']}
    subset_plan = {**plan, 'chunks': kept}
    arms = arm_summary(subset_plan, [r for r in chunks if r['chunk'] in indexes],
        [r for r in drafts if r['chunk'] in indexes], [r for r in requests if r['chunk'] in indexes],
        [q for q in mechanical if q['candidate_id'] in ids], [q for q in ready if q['candidate_id'] in ids],
        [q for q in selected if q['candidate_id'] in ids], statuses, baseline)
    core = [r for r in pairs if r['candidate_id'] in ids and r['core_status'] == 'CORE_USABLE']
    return {'scope': 'matched_topic_chunks_c02_c03_only', 'topic_chunks': [2, 3],
        'planned_slots_per_arm': 4, 'excluded_topic_chunks': [1],
        'exclusion_reason': 'direct_c01_execution_interruption_no_observed_response_not_method_failure',
        'arms': arms,
        'base_trained_by_arm': {arm: comparison.compare([r for r in core if r['arm'] == arm]) for arm in collection.ARMS},
        'base_trained_by_arm_correct_count': {arm: {str(n): comparison.compare([r for r in core if r['arm'] == arm and r['base']['gold_count'] == n]) for n in range(2, 7)} for arm in collection.ARMS},
        'blind_all_six_by_arm': {arm: comparison.blind_all_six([r for r in core if r['arm'] == arm]) for arm in collection.ARMS},
        'interpretation': 'Incomplete one-root exploratory comparison; execution-lost slots do not measure method quality.'}


def report_markdown(summary):
    totals, after = summary["totals"], summary["merged_distribution"]
    cumulative = summary["expansion"]["full_provisional_cumulative"]
    compare = summary["base_trained_comparison"]["primary"]
    base, trained = compare["base"], compare["trained"]
    lines = ["# v62 直接组题与独立证据整理后组题", "",
        "执行恢复说明：direct_c01 的一次请求因用户中断没有保存响应，其远程结果、完成时间和 token usage 均 unknown；两个名额为执行损失，不计内容失败、机械拒绝或提供方错误。未重复请求、未补题。两臂原计划各6名额，本地观测机会为 direct 4、staged 6；全批产量不能据此直接比较方法优劣。", "",
        f"同一 root {summary['root_id']}，完整 {cumulative['turns']} 轮、{cumulative['body_tokens']} 正文 tokens；本次没有新增扩充。",
        "direct 沿用已有证据优先和先反例软提示；staged 只把证据整理拆为一次独立调用，再把冻结的原始证据计划交给同一组题器。两臂沿用相同六个中性主题、全量原文、前轮反馈和预实验公开去重先验，允许每题 2–6 个正确项，不设答案数量配额。",
        "两臂各 3 批×2 个名额，按 direct_c01、staged_c01、staged_c02、direct_c02、direct_c03、staged_c03 预定交错执行。所有批次初始先验相同，当批产出不跨批传播；不是随机因果试验。", "",
        "| 方法 | 计划名额 | 执行损失 | 初稿 | 机械通过 | 评分前内容可用 | 难度入选 | 内容＋难度通过 | 单臂独立新增 | HTTP（含 planner） |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, a in summary['arms'].items():
        t = a['totals']
        lines.append(f"| {arm} | {t['slots']} | {a['execution_lost_slots']} | {t['initial_drafts']} | {t['mechanical_pass']} | {a['content_usable_before_solver']} | {t['selected']} | {a['difficulty_selected_and_content_usable']} | {a['unique_increment_if_this_arm_alone']} | {t['http_attempts']} |")
    lines += ["", "首要方法对比采用评分前冻结的内容可用集；Solver 后难度入选单列。跨臂整套重复保留各臂方法统计，最终合并只收一次。", "",
        "| 方法 | 初稿正确项数 | 内容可用正确项数 | 最终可用正确项数 | input tokens | output tokens | planner HTTP |", "|---|---|---|---|---:|---:|---:|"]
    for arm, a in summary['arms'].items():
        raw = '、'.join(f'{n} 项×{v}' for n, v in a['raw_initial_mc_support_count_histogram'].items()) or '无'
        lines.append(f"| {arm} | {raw} | {histogram_text(a['content_usable_distribution'])} | {histogram_text(a['usable_selected_distribution'])} | {a['cost']['input_tokens']} | {a['cost']['output_tokens']} | {a['planner_cost']['http_attempts']} |")
    lines += ["", "token 成本只取提供方返回的 usage，缺失记 unknown，不按字符数或本地 tokenizer 猜测。staged 成本包括 planner、首次组题和机械修复；细分及每内容可用题成本见 METHOD_COMPARISON.json。",
        f"合并前独立内容＋难度可用 {summary['new_usable_selected_before_exact_dedup']} 道，最终整体精确去重后实际新增 {summary['new_unique_increment']} 道；候选池由 {summary['baseline_question_count']} 增至 {summary['merged_usable_count']} 道（多选 {after['questions_by_type'].get('multi_select', 0)}、排序 {after['questions_by_type'].get('ordering', 0)}）。",
        f"合并多选分布：{histogram_text(after)}。跨臂整套重复组 {len(summary['cross_arm_exact_overlap'])} 组；既有候选全部保留。", "",
        "| 内容可用题主模型比较 | Base | 训练后 M_clean_policy |", "|---|---:|---:|",
        f"| 题数 | {base['questions']} | {trained['questions']} |",
        f"| 完全正确 | {base['exact_correct']}/{base['calls']}（{base['exact_rate_percent']}%） | {trained['exact_correct']}/{trained['calls']}（{trained['exact_rate_percent']}%） |",
        f"| 部分得分均值 | {base['partial_mean']} | {trained['partial_mean']} |", "",
        f"实际生成 HTTP {totals['http_attempts']}/15（含 planner）；训练后 {totals['solver_calls']}/216、Base {summary['base_solver_calls']}/216 次，每题每模型 18 次。远程错误 {totals['remote_error_requests']} 次，本地容量阻断 {totals['local_capacity_blocked_requests']} 次。",
        "本地收集结案且独立内容审冻结后才运行 Solver；被中断请求的远程完成时间仍未知。按方法和正确项数量分层成绩见模型比较；主比较不按模型得分挑题、不改 Gold。这批只有一个 root，重复及相似题不能当独立样本；结果不能证明第二轮训练收益。第二轮训练未开始。", "",
        "[完整对话](FULL_CONVERSATION.md) · [新增可用题](USABLE_SELECTED_QUESTIONS.md) · [合并可用题](MERGED_USABLE_QUESTIONS.md) · [逐题模型比较](BASE_TRAINED_COMPARISON.md) · [来源核验](PROVENANCE_AUDIT.md)"]
    matched = summary['matched_completed_method_comparison']
    insert = ['', '## 同时完成的主题 c02/c03', '',
        '以下每臂4个名额用于主要方法判断，排除无法成对比较的c01。仍是单root小样本，不把执行损失当成方法效果。', '',
        '| 方法 | 名额 | 初稿 | 机械通过 | 评分前内容可用 | 内容可用正确项数 | HTTP | input tokens | output tokens |',
        '|---|---:|---:|---:|---:|---|---:|---:|---:|']
    for arm, a in matched['arms'].items():
        t = a['totals']
        insert.append(f"| {arm} | {a['planned_slots']} | {t['initial_drafts']} | {t['mechanical_pass']} | {a['content_usable_before_solver']} | {histogram_text(a['content_usable_distribution'])} | {t['http_attempts']} | {a['cost']['input_tokens']} | {a['cost']['output_tokens']} |")
    lines[3:3] = insert
    return "\n".join(lines)


def main():
    audit_provenance.recovery.verify()
    adapter_bindings = read(RECOVERY / 'ADAPTER_BINDINGS.json')
    verify_bindings(adapter_bindings['files'])
    plan = collection.verify_plan()
    collection.completion('generation')
    collection.verify_review()
    _, evaluation_recovery = audit_provenance.evaluation_completion()
    require((HERE / 'INDEPENDENT_CONTENT_AUDIT.json').is_file(), 'content_audit_not_complete')
    require(not (HERE / 'RESULT_SUMMARY.json').exists(), 'immutable_final_report_already_exists')
    audit = audit_provenance.run_audit('evaluation')
    chunks, mechanical, ready, selected_raw, drafts, requests, scores = [], [], [], [], [], [], []
    for chunk in plan['chunks']:
        generation, evaluation = Path(chunk['generation_run']), Path(chunk['evaluation_run'])
        row, qm, qr, qs, raw, req, scored = collect_chunk(plan['root_id'], chunk['index'], generation, evaluation)
        row['arm'] = chunk['arm']; row['chunk_name'] = chunk['chunk_name']
        chunks.append(row); mechanical += qm; ready += qr; selected_raw += qs
        drafts += raw; requests += req; scores += scored
    totals = {key: sum(row[key] for row in chunks) for key in TOTAL_KEYS}
    require(totals['slots'] == 12 and totals['http_attempts'] <= 15 and totals['solver_calls'] <= 216,
            'final_actual_budget_mismatch')
    selected, selected_duplicates = deduplicate_exact(selected_raw)
    usable_raw, issues_raw, statuses = content_subset(mechanical, selected_raw)
    usable, usable_duplicates = deduplicate_exact(usable_raw)
    issues = [q for q in selected if statuses[q['candidate_id']] == 'KNOWN_SUBSTANTIVE_ISSUE']
    old_path = Path(collection.CONFIG["baseline_usable_pool"])
    old = read(old_path)
    baseline = old["questions"]
    before = distribution(baseline)
    expected_baseline = collection.CONFIG["expected_baseline_summary"]
    observed_baseline = {"questions": len(baseline), "multi_select": before["multi_select"],
        "ordering": before["questions_by_type"].get("ordering", 0),
        "multi_correct_count_histogram": {str(key): value for key, value in before["multi_correct_count_histogram"].items()},
        "correct_options": before["correct_options"], "total_options": before["total_options"]}
    require(observed_baseline == expected_baseline, "configured_usable_baseline_changed")
    merged, merged_duplicates = deduplicate_exact(baseline + usable)
    score_by_id = {q['candidate_id']: q for q in ready}
    for score in scores:
        score['correct_option_count'] = len(score_by_id[score['candidate_id']]['gold'])
        score['independent_content_status'] = statuses[score['candidate_id']]
    base_comparison, paired_calls = comparison.collect(statuses, {q['candidate_id'] for q in selected_raw})
    arms = arm_summary(plan, chunks, drafts, requests, mechanical, ready, selected_raw, statuses, baseline)
    matched = matched_completed_method_comparison(plan, chunks, drafts, requests, mechanical, ready, selected_raw, statuses, baseline, paired_calls)
    incremental_ids = {q['candidate_id'] for q in merged} - {q['candidate_id'] for q in baseline}
    for arm, value in arms.items():
        value['actually_retained_increment_after_global_dedup'] = len(incremental_ids & set(value['incremental_ids_if_this_arm_alone']))
    expansion, turns = expansion_summary(plan)
    summary = {'version': 'h4_generation_method_probe_v62_completed_collection_v1',
        'root_id': plan['root_id'], 'experiment_scope': plan['experiment_scope'],
        'root_selection': plan['root_selection'], 'baseline_label': collection.CONFIG['baseline_label'],
        'baseline_question_count': len(baseline), 'arms': arms,
        'matched_completed_method_comparison': matched,
        'execution_recovery': binding(RECOVERY / 'RECOVERY_PLAN.json'),
        'recovery_adapter_bindings': binding(RECOVERY / 'ADAPTER_BINDINGS.json'),
        'method_comparison_incomplete_due_to_execution_interruption': True,
        'cross_arm_exact_overlap': cross_arm_exact_overlap(plan, mechanical),
        'generation_cost': cost_summary(requests),
        'new_unique_increment': len(merged) - len(baseline),
        'new_usable_selected_before_exact_dedup': len(usable_raw), 'totals': totals, 'chunks': chunks, 'expansion': expansion,
        'raw_initial_mc_support_count_histogram': raw_histogram(drafts, False),
        'raw_repair_mc_support_count_histogram': raw_histogram(drafts, True),
        'mechanical_distribution': distribution(mechanical), 'ready_distribution': distribution(ready),
        'selected_distribution': distribution(selected), 'usable_selected_distribution': distribution(usable),
        'before_distribution': before, 'merged_distribution': distribution(merged),
        'new_selected_unique': len(selected), 'new_selected_exact_duplicates': selected_duplicates,
        'mechanical_content_status_counts': dict(Counter(statuses.values())),
        'core_usable_mechanical_rate': str(Fraction(sum(status == 'CORE_USABLE' for status in statuses.values()), len(mechanical))) if mechanical else None,
        'new_usable_selected': len(usable), 'new_usable_exact_duplicates': usable_duplicates, 'new_known_issue_selected': len(issues),
        'new_usable_two_to_four_count': sum(q['question_type'] == 'multi_select' and 2 <= len(q['gold']) <= 4 for q in usable),
        'new_usable_two_to_three_count': sum(q['question_type'] == 'multi_select' and 2 <= len(q['gold']) <= 3 for q in usable),
        'new_usable_five_count': sum(q['question_type'] == 'multi_select' and len(q['gold']) == 5 for q in usable),
        'base_trained_comparison': base_comparison,
        'base_solver_calls': base_comparison['actual_calls_per_model'],
        'merged_usable_count': len(merged), 'merged_exact_duplicates': merged_duplicates,
        'statement_overlap': statement_overlap(merged, {q['candidate_id'] for q in usable}),
        'new_selected_known_issue_ids': [q['candidate_id'] for q in issues],
        'new_usable_selected_ids': [q['candidate_id'] for q in usable],
        'score_aggregate': aggregate_scores(scores),
        'evaluation_recovery': evaluation_recovery,
        'content_audit': binding(HERE / 'INDEPENDENT_CONTENT_AUDIT.json'),
        'content_review_completion': binding(HERE / 'CONTENT_REVIEW.COMPLETE.json'),
        'baseline_usable_pool': binding(old_path), 'machine_selection_is_not_semantic_certification': True,
        'correct_option_count_is_not_admission_gate': True, 'full_question_records_preserved': True,
        'cross_root_replication_of_v60': False,
        'direct_vs_evidence_staged_generation': True,
        'within_root_sequential_add_on_not_randomized_control': True,
        'global_question_generation_rules_changed': False, 'source_focus_method_changed_from_v60': False,
        'source_focus_topics_changed_for_current_root': False,
        'all_chunks_same_initial_prior': True, 'cross_chunk_current_feedback': False,
        'historical_public_prior_extended': True,
        'training_allowed': False, 'training_calls': 0, 'formal_training_admission': False}
    # All input verification and semantic subset accounting finish before outputs.
    join = {"audit_sha256": summary["content_audit"]["sha256"],
            "solver_scores_used_for_independent_content_labels": False,
            "generation_or_gold_edited": False, "new_gate": False,
            "selected_and_core_usable_ids": [q["candidate_id"] for q in usable],
            "selected_with_known_issue_ids": [q["candidate_id"] for q in issues],
            "questions": [{"candidate_id": q["candidate_id"], "core_status": statuses[q["candidate_id"]],
                           "selected_by_policy": q["candidate_id"] in {x["candidate_id"] for x in selected}}
                          for q in mechanical]}
    write(HERE / "CONTENT_SELECTION_JOIN.json", join)
    summary["content_selection_join"] = binding(HERE / "CONTENT_SELECTION_JOIN.json")
    if not (HERE / 'PROVENANCE_AUDIT.json').exists():
        audit_provenance.export_audit(audit)
    else:
        require(read(HERE / 'PROVENANCE_AUDIT.json')['status'] == 'PASS', 'prior_provenance_audit_failed')
    for path in ('PROVENANCE_AUDIT.json', 'PROVENANCE_AUDIT.md'):
        OUTPUTS.append(HERE / path)
    provenance = {'experiment_plan': binding(HERE / 'EXPERIMENT_PLAN.json'),
                  'content_audit': summary['content_audit'],
                  'content_review_completion': summary['content_review_completion'],
                  'baseline_usable_pool': summary['baseline_usable_pool']}
    summary['mechanical_display'] = export_questions('ALL_MECHANICAL', mechanical, 'v62 全部机械通过多选题', provenance)
    summary['selected_display'] = export_questions('SELECTED', selected, 'v62 机器入选多选题', provenance)
    summary['usable_selected_display'] = export_questions('USABLE_SELECTED', usable, 'v62 新增内容可用多选题', provenance)
    summary['merged_usable_display'] = export_questions('MERGED_USABLE', merged, '截至v62 合并内容可用题', provenance)
    write_text(HERE / 'FULL_CONVERSATION.md', conversation_markdown(plan['root_id'], turns))
    write(HERE / 'ALL_READY_CANDIDATES.json', {'questions': ready, 'training_allowed': False})
    write(HERE / 'RAW_DRAFTS.json', {'drafts': drafts, 'initial_count': totals['initial_drafts'],
        'repair_count': totals['repair_drafts'], 'raw_question_records_unchanged': True})
    write(HERE / 'REQUEST_COUNTS.json', {'requests': requests, 'totals': totals})
    write(HERE / 'SOLVER_SCORES.json', {'questions': scores, 'aggregate': summary['score_aggregate']})
    write_text(HERE / 'SOLVER_SCORES.md', score_markdown(scores, statuses))
    write(HERE / 'BASE_TRAINED_COMPARISON.json', base_comparison)
    write(HERE / 'STATEMENT_OVERLAP.json', summary['statement_overlap'])
    write(HERE / 'PAIRED_CALLS.json', {'calls': paired_calls})
    write_text(HERE / 'BASE_TRAINED_COMPARISON.md', comparison.markdown(base_comparison))
    write(HERE / 'BLIND_SELECTION_BASELINE.json', {
        'formula': 'TP/k - FP/(6-k), except k=6 has no false-positive term',
        'all_six_partial_by_gold_count': {str(n): 1 if n == 6 else 0 for n in range(2, 7)},
        'before': before, 'merged': summary['merged_distribution'],
        'content_usable_by_arm': {arm: value['content_usable_distribution'] for arm, value in arms.items()},
        'no_model_calls': True})
    write(HERE / 'METHOD_COMPARISON.json', {'arms': arms, 'cost': summary['generation_cost'],
        'primary_method_scope': 'solver_blind_content_usable', 'cross_arm_exact_overlap': summary['cross_arm_exact_overlap'],
        'new_unique_increment': summary['new_unique_increment'], 'same_frozen_prior_source_and_focus': True,
        'matched_completed_topic_comparison': matched, 'execution_loss_is_not_method_failure': True})
    write(HERE / 'MATCHED_TOPIC_COMPARISON.json', matched)
    matched_lines = ['# v62 同时完成主题 c02/c03 的方法与模型比较', '',
        '每臂4个计划名额。c01因direct响应缺失排除出方法配对判断，属于执行中断，不是生成质量失败。各题先独立内容审再评分；未根据Solver结果调整Gold。', '',
        '| 方法 | Gold正确项数 | 内容可用题数 | Base 完全正确 | 训练后完全正确 | Base 部分均值 | 训练后部分均值 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm, strata in matched['base_trained_by_arm_correct_count'].items():
        for count, scoreset in strata.items():
            b, t = scoreset['base'], scoreset['trained']
            matched_lines.append(f"| {arm} | {count} | {b['questions']} | {b['exact_correct']}/{b['calls']} | {t['exact_correct']}/{t['calls']} | {b['partial_mean']} | {t['partial_mean']} |")
    matched_lines.extend(['', '单root、预定非随机顺序、小样本，方法样本不完整。不同问题不是严格相同难度；本轮没有训练，不能把Base差值当第二轮收益。'])
    write_text(HERE / 'MATCHED_TOPIC_COMPARISON.md', '\n'.join(matched_lines))
    note = ['# v62 执行中断与方法判断', '',
        'direct_c01的一次HTTP请求已发送，但用户中断后没有本地响应、HTTP结果或验证记录。远程是否完成、完成时间、返回内容与token usage均unknown。恢复没有重复该请求、补题或伪造服务器回执。',
        '原计划每臂6个名额，direct实际丢失2个名额；该损失单列为执行中断，机械拒绝/内容错误/远程提供方错误均不因此增加。',
        '两臂方法判断优先看MATCHED_TOPIC_COMPARISON（c02/c03，每臂4个名额）。全批汇总仅反映实际候选收集，不能把direct少两题归因于方法劣势。',
        'direct全批token成本含一次unknown usage请求，因此总成本unknown；已观测token小计和匹配主题成本另列，不能当作0成本。中断后只验证从本地恢复观察起至少35秒冷却，不宣称知道远程完成间隔。',
        '所有已观测问题仍用原始原文、公开先验、反馈、Gold、去重和评分。训练仍未开始。']
    write_text(HERE / 'INTERPRETATION_NOTE.md', '\n'.join(note))
    write(HERE / 'RESULT_SUMMARY.json', summary)
    write_text(HERE / 'REPORT.md', report_markdown(summary))
    verification = {'status': 'PASS', 'previous_bound_files_unchanged': audit['history']['previous_bound_files_unchanged'],
        'new_generation_slots': totals['slots'], 'remote_http_calls': totals['http_attempts'],
        'solver_calls': totals['solver_calls'], 'base_solver_calls': summary['base_solver_calls'], 'new_core_usable_selected': len(usable),
        'merged_core_usable': len(merged), 'full_records_and_gold_preserved': True,
        'independent_display_replay': True, 'training_calls': 0,
        'new_usable_two_to_four_count': summary['new_usable_two_to_four_count'],
        'new_usable_two_to_three_count': summary['new_usable_two_to_three_count'],
        'ordering_generated': 0, 'interrupted_http_requests': totals['interrupted_http_requests'],
        'interrupted_unobserved_slots': totals['interrupted_unobserved_slots'],
        'remote_interrupted_outcome': 'unknown', 'execution_loss_is_not_method_failure': True}
    write(HERE / 'VERIFICATION.json', verification)
    INPUTS.update(Path(row['path']) for row in adapter_bindings['files'])
    INPUTS.update((RECOVERY / name for name in ('ADAPTER_BINDINGS.json', 'RECOVERY_PLAN.json', 'ORIGINAL_BINDINGS.json', 'INTERRUPTED_ATTEMPT.json', 'LOCAL_RESUME_COOLDOWN.json')))
    INPUTS.update((HERE / name for name in ('EXPERIMENT_PLAN.json', 'generation.COMPLETE.json',
                                          'INDEPENDENT_CONTENT_AUDIT.json', 'CONTENT_REVIEW.COMPLETE.json')))
    INPUTS.add(Path(evaluation_recovery['completion']['path']))
    write(HERE / 'REPORT_BINDINGS.json', {'inputs': [binding(p) for p in sorted(INPUTS)],
        'outputs': [binding(p) for p in sorted(set(OUTPUTS))]})
    paths = sorted({p for directory in (HERE, CODE) for p in directory.rglob('*')
                    if p.is_file() and '__pycache__' not in p.parts and p.name != 'RESULT_BINDINGS.json'})
    write(HERE / 'RESULT_BINDINGS.json', {'scope': 'completed_v62_mc_focus_code_source_runs_paired_base_reports_displays',
        'files': [binding(p) for p in paths]})
    print(json.dumps(verification, ensure_ascii=False))


if __name__ == '__main__':
    main()
