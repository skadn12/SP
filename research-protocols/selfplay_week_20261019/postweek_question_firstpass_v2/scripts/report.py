"""Report prospective first-pass yield separately from B1 performance."""
from collections import Counter
import csv
import io
from common import BASE, SLOTS, bind, now, read, save


def aggregate(rows):
    n = len(rows)
    ordering = [r for r in rows if r['question_type'] == 'ordering']
    pairs = sum(r['pairwise_total'] or 0 for r in ordering)
    pair_correct = sum(r['pairwise_correct'] or 0 for r in ordering)
    return {'questions': n, 'exact': sum(bool(r['exact']) for r in rows),
            'accuracy': sum(bool(r['exact']) for r in rows) / n if n else None,
            'invalid': sum(not r['valid'] for r in rows),
            'false_positive': sum(r['false_positive'] or 0 for r in rows),
            'false_negative': sum(r['false_negative'] or 0 for r in rows),
            'pairwise_correct': pair_correct, 'pairwise_total': pairs,
            'pairwise_accuracy': pair_correct / pairs if pairs else None}


def main():
    protocol = read(BASE / 'PROTOCOL_FREEZE.json')
    freeze = read(BASE / 'QUESTION_FREEZE.json')
    decisions = freeze['decisions']
    scores = read(BASE / 'solver_runtime/execution/SCORES.json')['records']
    mechanical = read(BASE / 'solver_runtime/execution/ANALYSIS_SUMMARY.json')['visible_reason_mechanical_metrics']
    raw = read(BASE / 'solver_runtime/execution/RAW_RESPONSE_FREEZE.json')
    assert len(decisions) == 56 and len(scores) == 2 * freeze['admitted_questions']
    assert raw['status'] == 'ALL_REQUESTS_TERMINAL_RAW_SEALED'
    pvt = {q['candidate_id']: q for q in read(BASE / 'inputs/PRIVATE_SCORING.json')['questions']}
    for score in scores:
        assert score['candidate_id'] in pvt
    admissions = [dict(root_id=root, planned=14,
                       admitted=sum(d['status'] == 'ADMIT' and d['root_id'] == root for d in decisions))
                  for root in protocol['roots']]
    by_slot = [dict(slot_id=slot, planned=8,
                    admitted=sum(d['status'] == 'ADMIT' and d['slot_id'] == slot for d in decisions))
               for slot in SLOTS]
    review_rows = []
    admission_failures = []
    reviews = list((BASE / 'question_reviews').glob('*/*/REVIEW.json'))
    assert len(reviews) == 8
    for path in reviews:
        review = read(path)
        review_rows += review['questions']
        for row in review['questions']:
            if row['decision'] != 'ADMIT':
                admission_failures.append({
                    'root_id': review['root_id'], 'condition': review['condition'],
                    'slot_id': row['slot_id'], 'decision': row['decision'],
                    'checks': row.get('checks', {}), 'notes': row.get('notes'),
                    'review_binding': bind(path),
                })
    ambiguity = sum(r['decision'] == 'FAIL' and
                    (r.get('submitted_gold_matches') is False or
                     r.get('checks', {}).get('grounding') == 'FAIL' or
                     r.get('checks', {}).get('unique_key') == 'FAIL') for r in review_rows)
    checks = {
        'overall_at_least_90pct': freeze['admitted_questions'] / 56 >= .9,
        'every_root_at_least_85pct': all(r['admitted'] / r['planned'] >= .85 for r in admissions),
        'every_slot_at_least_75pct': all(r['admitted'] / r['planned'] >= .75 for r in by_slot),
        'no_source_or_gold_ambiguity_failure': ambiguity == 0,
    }
    score_cells = []
    condition_cells = []
    gold_cells = []
    root_cells = []
    for branch in ('direct', 'reason'):
        branch_rows = [r for r in scores if r['branch'] == branch]
        groups = {
            'MC2-5': [r for r in branch_rows if r['question_type'] == 'multi_select' and r['gold_count'] < 6],
            'MC6_control': [r for r in branch_rows if r['question_type'] == 'multi_select' and r['gold_count'] == 6],
            'ordering': [r for r in branch_rows if r['question_type'] == 'ordering'],
        }
        score_cells += [{'branch': branch, 'group': group, **aggregate(rows)} for group, rows in groups.items()]
        for condition in protocol['conditions']:
            condition_cells.append({'branch': branch, 'condition': condition,
                                    **aggregate([r for r in branch_rows if r['condition'] == condition])})
        for count in range(2, 7):
            gold_cells.append({'branch': branch, 'gold_count': count,
                               **aggregate([r for r in branch_rows if r['gold_count'] == count])})
        for root in protocol['roots']:
            for condition in protocol['conditions']:
                for slot in SLOTS:
                    rows = [r for r in branch_rows if r['root_id'] == root and r['condition'] == condition
                            and pvt[r['candidate_id']]['slot_id'] == slot]
                    root_cells.append({'root_id': root, 'condition': condition, 'slot_id': slot,
                                       'branch': branch, **aggregate(rows)})
    sample = read(BASE / 'inputs/REASON_SAMPLE.json')['records']
    required = {r['request_id'] for r in sample if r['status'] == 'ADMIT'}
    key_paths = sorted((BASE / 'reason_semantic_reviews').glob('*/*/KEY_FREEZE.json'))
    source_keys = {q['request_id']: q['independently_derived_display_gold']
                   for path in key_paths for q in read(path)['questions']}
    assert len(key_paths) == 4 and set(source_keys) == required
    semantic_paths = list((BASE / 'reason_semantic_reviews').glob('*/*/REVIEW.json'))
    semantic_records = []
    for path in semantic_paths:
        semantic_records += read(path)['records']
    assert {r['request_id'] for r in semantic_records} == required
    assert len(semantic_records) == len(required)
    scored_by_request = {r['request_id']: r for r in scores}
    verdicts = {'PASS', 'HOLD', 'FAIL'}
    for record in semantic_records:
        assert record['decision'] in verdicts and record['final_answer'] in verdicts
        dimensions = [record[k] for k in ('schema_and_completeness', 'final_answer',
                       'answer_reason_relation', 'answer_reason_final_consistency',
                       'answer_reason_source_support', 'answer_reason_actor_attribution')]
        for item in record['item_findings'].values():
            dimensions += [item[k] for k in ('judgment_verdict', 'citation_sufficiency',
                           'citation_precision', 'reason_source_support', 'actor_attribution')]
        assert set(dimensions) <= verdicts | {'NOT_APPLICABLE'}
        expected = 'FAIL' if 'FAIL' in dimensions else 'HOLD' if 'HOLD' in dimensions else 'PASS'
        assert record['decision'] == expected, (record['request_id'], 'STRICT_AGGREGATION_MISMATCH')
        score = scored_by_request[record['request_id']]
        independent_key = record['independently_derived_display_gold']
        frozen_key = source_keys[record['request_id']]
        assert (set(independent_key) == set(frozen_key) if score['question_type'] == 'multi_select'
                else independent_key == frozen_key), (record['request_id'], 'REVIEW_KEY_DRIFT')
        assert record['raw_binding'] == score['raw_binding'], (record['request_id'], 'REVIEW_RAW_BINDING_MISMATCH')
        assert (set(independent_key) == set(score['display_gold']) if score['question_type'] == 'multi_select'
                else independent_key == score['display_gold']), (record['request_id'], 'INDEPENDENT_SOURCE_KEY_MISMATCH')
        if score['valid']:
            assert (record['final_answer'] == 'PASS') == score['exact'], (record['request_id'], 'FINAL_VERDICT_MISMATCH')
    semantic = dict(Counter(r['decision'] for r in semantic_records))
    reason_finals = Counter(r['final_answer'] for r in semantic_records)
    item_dims = {}
    for dimension in ('judgment_verdict', 'citation_sufficiency', 'citation_precision', 'reason_source_support', 'actor_attribution'):
        item_dims[dimension] = dict(Counter(i[dimension] for r in semantic_records for i in r['item_findings'].values()))
    report = {
        'status': 'COMPLETED_PROSPECTIVE_SMALL_BATCH', 'created_at_utc': now(),
        'planned_slots': 56, 'first_attempt_admitted': freeze['admitted_questions'],
        'first_attempt_status_counts': dict(Counter(d['status'] for d in decisions)),
        'admission_by_root': admissions, 'admission_by_slot': by_slot,
        'admission_failure_details': admission_failures,
        'source_or_gold_ambiguity_failures': ambiguity,
        'preregistered_engineering_screen': checks, 'engineering_screen_pass': all(checks.values()),
        'solver_requests': len(scores), 'returned_requests': raw['returned_requests'],
        'solver_wall_seconds': raw['wall_seconds'], 'score_cells': score_cells,
        'condition_cells_descriptive_only': condition_cells, 'score_by_gold_count': gold_cells,
        'root_slot_branch_cells': root_cells,
        'reason_semantic_sample_planned': 16, 'reason_semantic_sample_available': len(required),
        'reason_semantic_sample_verdicts': semantic, 'reason_sample_final_verdicts': dict(reason_finals),
        'reason_item_dimension_counts': item_dims,
        'reason_mechanical_metrics_only': mechanical,
        'semantic_revisions': 0, 'source_extension': False, 'solver_score_selection': False,
        'formal_weakness_claim': False, 'training_admission': False, 'causal_feedback_benefit_claim': False,
        'limits': [
            'Four reused paired Train roots; dependent question observations are not independent population trials.',
            'Reused Train sources are not a held-out generalization evaluation or a before/after training comparison.',
            'QPR1 55/56 was repaired-old-question admission, not a comparable new-author first-pass baseline.',
            'Different questions prevent an answer accuracy change from establishing a causal benefit.',
            'MC6 can be answered by always selecting all; MC5 has only one foil.',
            'Visible explanations are observable output, not verified faithful hidden internal deliberation.',
            'Only the preregistered available 16-slot sample receives semantic reason review; no full-corpus reliability claim.',
            'Imported frozen solver code retains historical internal cohort label FBC1; paths and opaque IDs identify QFP2.',
        ],
        'input_bindings': [bind(BASE / 'PROTOCOL_FREEZE.json'), bind(BASE / 'QUESTION_FREEZE.json'),
                           bind(BASE / 'inputs/EXPORT_FREEZE.json'), bind(BASE / 'inputs/REASON_SAMPLE.json'),
                           bind(BASE / 'solver_runtime/execution/SCORES.json'),
                           bind(BASE / 'solver_runtime/execution/ANALYSIS_SUMMARY.json'),
                           bind(BASE / 'solver_runtime/execution/RAW_RESPONSE_FREEZE.json'),
                           bind(BASE / 'solver_runtime/PREFLIGHT_REVIEW.json'),
                           bind(BASE / 'REASON_REVIEW_PROTOCOL.json'),
                           bind(BASE / 'REASON_REVIEW_INSTRUCTIONS.txt'), bind(__file__)]
                          + [bind(p) for p in reviews + key_paths + semantic_paths],
    }
    save(BASE / 'FINAL_REPORT.json', report)
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(root_cells[0]))
    writer.writeheader(); writer.writerows(root_cells)
    with (BASE / 'ROOT_SLOT_RESULTS.csv').open('x') as f:
        f.write(out.getvalue())
    lines = ['# 新规则首次出题验证结论 QFP2', '',
             f"首次准入 **{freeze['admitted_questions']}/56**；失败与 SHORTFALL 全部保留，未语义重试、未扩充来源、未按 solver 成绩筛选。", '',
             '| Root | 首次准入 |', '|---|---:|']
    lines += [f"| {r['root_id']} | {r['admitted']}/14 |" for r in admissions]
    lines += ['', '| 槽位 | 首次准入 |', '|---|---:|']
    lines += [f"| {r['slot_id']} | {r['admitted']}/8 |" for r in by_slot]
    lines += ['', '| 题组 | direct | reason |', '|---|---:|---:|']
    for group in ('MC2-5', 'MC6_control', 'ordering'):
        cells = [next(c for c in score_cells if c['group'] == group and c['branch'] == branch) for branch in ('direct', 'reason')]
        def text(c):
            return f"{c['exact']}/{c['questions']}（{c['accuracy']:.1%}）" if c['questions'] else '无准入题'
        lines.append(f'| {group} | {text(cells[0])} | {text(cells[1])} |')
    lines += ['', '| 多选 Gold 数量 | direct | reason |', '|---|---:|---:|']
    for count in range(2, 7):
        cells = [next(c for c in gold_cells if c['gold_count'] == count and c['branch'] == branch) for branch in ('direct', 'reason')]
        lines.append(f"| {count}{'（全选控制）' if count == 6 else ''} | {text(cells[0])} | {text(cells[1])} |")
    lines += ['', f"GPU {len(scores)} 次请求，返回 {raw['returned_requests']} 次，耗时 {raw['wall_seconds']:.1f} 秒；所有 raw 封存后评分。",
              f"解释分支最终答案格式有效 {mechanical['final_syntax_valid']}/{mechanical['requests']}；解释结构有效 {mechanical['reason_structure_valid']}/{mechanical['requests']}。机械格式检查不代表引用或解释在语义上可靠。",
              f"预先固定 16 槽解释审核，实际准入可审 {len(required)} 条；严格语义结果 {semantic}。最终答案、引用充分性与理由可靠性分别评价。",
              f"这 {len(required)} 条中，最终答案判定 {dict(reason_finals)}。原始 ANALYSIS_SUMMARY 的语义 PENDING 是评分时状态，本报告仅补充固定样本的语义审核。",
              f"预先冻结的工程筛查：{'通过，可进行下一批不同 root 的分批验证' if all(checks.values()) else '未全部通过，先处理报告中的具体失败类型'}。各检查：{checks}。", '',
              '本轮仅验证新作者首次出题与小样本作答质量，不能将旧题修复 55/56 当对照，也没有证明反馈因果收益、总体稳定性或忠实内部 CoT；没有训练或正式入池。', '',
              '全部题稿见 `questions/`，独立准入依据见 `question_reviews/`，原始作答见 `solver_runtime/execution/raw/`，解释语义审核见 `reason_semantic_reviews/`，逐 root/条件/槽位/作答方式见 `ROOT_SLOT_RESULTS.csv`。', '',
              '| 逐项解释维度 | PASS | HOLD | FAIL | 不适用 |', '|---|---:|---:|---:|---:|']
    for dimension, counts in item_dims.items():
        lines.append(f"| {dimension} | {counts.get('PASS', 0)} | {counts.get('HOLD', 0)} | {counts.get('FAIL', 0)} | {counts.get('NOT_APPLICABLE', 0)} |")
    with (BASE / 'FINAL_REPORT_ZH.md').open('x') as f:
        f.write('\n'.join(lines) + '\n')
    save(BASE / 'FINAL_ARTIFACT_MANIFEST.json', {'created_at_utc': now(),
         'artifacts': [bind(p) for p in (BASE / 'FINAL_REPORT.json', BASE / 'FINAL_REPORT_ZH.md', BASE / 'ROOT_SLOT_RESULTS.csv')],
         'input_bindings': report['input_bindings']})
    print({'firstpass': str(freeze['admitted_questions']) + '/56', 'engineering_screen_pass': all(checks.values()),
           'score_cells': score_cells, 'semantic_sample': semantic})


if __name__ == '__main__':
    main()
