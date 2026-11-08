"""Recalculate existing v2 novelty against the selected formal previous round.

No generation, content re-adjudication, Gold repair, or historical file writes.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
OLD = BASE.parent / 'h4_qa_feedback_constraints_v2'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def artifact(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def unit_cost(tokens, count):
    return {'value': tokens / count, 'denominator': count} if count else {
        'value': None, 'reason': 'ZERO_ELIGIBLE_ITEMS'}


def main():
    lock = read(BASE / 'previous_round_sources.json')
    original_path = OLD / 'reports/FINAL_REPORT.json'
    original = read(original_path)
    roots, all_rows = [], []
    for source, old in zip(lock['roots'], original['roots']):
        root = source['root_id']
        assert old['root_id'] == root
        code = root.rsplit('-', 1)[-1]
        content_path = OLD / 'reports' / f'CONTENT_{code}.json'
        content_rows = read(content_path)['raw_candidate_reviews']
        novelty_path = BASE / 'reports' / f'NOVELTY_{code}.json'
        diagnosis = read(novelty_path)
        novelty_rows = diagnosis.get('candidate_reviews', diagnosis.get('raw_candidate_reviews'))
        novelty = {r['candidate_id']: r for r in novelty_rows}
        assert len(novelty) == len(novelty_rows) == len(content_rows)
        assert set(novelty) == {r['candidate_id'] for r in content_rows}
        strict_path = OLD / 'filtered' / root / 'strict_pass_dedup.json'
        strict = read(strict_path)
        kept = {r['candidate_id'] for r in strict}
        new = [r for r in strict if novelty[r['candidate_id']]['previous_formal_novelty']['verdict'] == 'NEW']
        for row in content_rows:
            review = novelty[row['candidate_id']]
            assert review['in_final_pool'] == row['in_final_pool']
            all_rows.append({'root_id': root, 'candidate_id': row['candidate_id'],
                'question_type': row['question_type'], 'phase': row['phase'],
                'in_final_pool': row['in_final_pool'], 'content_verdict_unchanged': row['verdict'],
                'within_batch_strict_kept_unchanged': row['candidate_id'] in kept,
                'previous_formal_novelty': deepcopy(review['previous_formal_novelty']),
                'eligible_new': row['candidate_id'] in {r['candidate_id'] for r in new},
                'content_diagnosis': str(content_path), 'novelty_diagnosis': str(novelty_path)})
        rows = [r for r in all_rows if r['root_id'] == root]
        final = [r for r in rows if r['in_final_pool']]
        assert len(final) == old['program_accepted']
        assert Counter(r['content_verdict_unchanged'] for r in final) == Counter(old['content_verdicts'])
        assert len(strict) == old['strict_pass_dedup']
        output = BASE / 'filtered' / root / 'new_supported_dedup.json'
        write(output, new)
        record = {'root_id': root, 'previous_formal_questions': len(read(source['question_pool']['path'])),
            'raw_candidates': len(rows), 'program_accepted': len(final),
            'program_accepted_by_type': old['program_accepted_by_type'],
            'content_verdicts_unchanged': old['content_verdicts'],
            'strict_content_pass_within_batch_dedup': len(strict),
            'strict_content_pass_within_batch_dedup_by_type': dict(Counter(r['question_type'] for r in strict)),
            'previous_formal_novelty_raw': dict(Counter(r['previous_formal_novelty']['verdict'] for r in rows)),
            'previous_formal_novelty_final': dict(Counter(r['previous_formal_novelty']['verdict'] for r in final)),
            'previous_formal_novelty_strict': dict(Counter(novelty[r['candidate_id']]['previous_formal_novelty']['verdict'] for r in strict)),
            'new_supported_dedup': len(new),
            'new_supported_dedup_by_type': dict(Counter(r['question_type'] for r in new)),
            'new_supported_candidate_ids': [r['candidate_id'] for r in new],
            'valid_new_ordering': sum(r['question_type'] == 'ordering' for r in new),
            'original_calls': old['calls'], 'original_usage': old['usage'],
            'tokens_per_new_supported': unit_cost(old['usage']['total_tokens'], len(new)),
            'original_error_counts_unchanged': old['errors_final'],
            'sources': [artifact(content_path), artifact(novelty_path), artifact(strict_path), source['question_pool']],
            'corrected_filtered_output': artifact(output)}
        roots.append(record)
    total_new = sum(r['new_supported_dedup'] for r in roots)
    types = Counter()
    for row in roots:
        types.update(row['new_supported_dedup_by_type'])
    assert len(all_rows) == original['raw_candidates'] == 50
    cost = deepcopy(original['cost'])
    cost['tokens_per_new_supported'] = unit_cost(cost['total_tokens'], total_new)
    result = {'status': 'CORRECTED_OFFLINE_STOPPED', 'reviewer': 'Codex offline novelty diagnosis',
        'independent_human_certification': False,
        'previous_formal_question_run_id': lock['previous_question_run_id'],
        'previous_formal_feedback_run_id': lock['previous_feedback_run_id'],
        'scope_lock': artifact(BASE / 'previous_round_sources.json'),
        'actual_generation': 'Existing h4_qa_feedback_constraints_v2 outputs generated with the old mixed recent-test feedback; no corrected-feedback live run.',
        'original_report': artifact(original_path), 'roots': roots,
        'previous_formal_questions': sum(r['previous_formal_questions'] for r in roots),
        'raw_candidates': len(all_rows), 'program_accepted': original['program_accepted'],
        'content_verdicts_unchanged': original['content_verdicts'],
        'strict_content_pass_within_batch_dedup': original['strict_pass_dedup'],
        'new_supported_dedup': total_new, 'new_supported_dedup_by_type': dict(types),
        'valid_new_ordering': types.get('ordering', 0),
        'previous_formal_novelty_raw': dict(Counter(r['previous_formal_novelty']['verdict'] for r in all_rows)),
        'previous_formal_novelty_final': dict(Counter(r['previous_formal_novelty']['verdict'] for r in all_rows if r['in_final_pool'])),
        'original_calls_unchanged': original['calls'], 'original_cost_with_corrected_denominator': cost,
        'new_external_calls': 0, 'gold_modified': False, 'content_verdicts_modified': False,
        'within_batch_dedup_modified': False, 'original_prompts_or_responses_modified': False,
        'corrected_prompt_effectiveness': 'NOT_MEASURED',
        'automatic_next_experiment': False,
        'old_recent_test_novelty': {'status': 'SUPERSEDED_AS_PREVIOUS_FORMAL_ROUND_METRIC',
             'report': str(original_path), 'reason': '42 entries from two current-round tests were incorrectly merged as previous formal history.'}}
    write(BASE / 'reports/CORRECTED_ROWS.json', all_rows)
    write(BASE / 'reports/CORRECTED_REPORT.json', result)
    lines = ['# 上一正式轮反馈与新增量修正', '',
        '已修正：历史题和实例反馈只来自上一正式轮；近期测试不再充当“历史题”。此前“有效新增全为0”不能用于这个口径。', '',
        f'对已经生成的50道原始候选重算后，30道程序接收题中的6道内容通过且批内去重题，有 **{total_new}道有效新增**。内容审核、原Gold、批内去重和原始请求均未修改。', '',
        '| root | 上一正式轮题数 | 程序接收 | 内容通过且批内去重 | 有效新增单选 | 有效新增多选 | 有效新增排序 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in roots:
        t = r['new_supported_dedup_by_type']
        lines.append(f"| {r['root_id'].rsplit('-', 1)[-1]} | {r['previous_formal_questions']} | {r['program_accepted']} | {r['strict_content_pass_within_batch_dedup']} | {t.get('single_choice', 0)} | {t.get('multi_select', 0)} | {t.get('ordering', 0)} |")
    lines += ['',
        '259192的器材准备承诺、具体安排确认及安排状态多选，考察的是R02新增事实，不能因近期测试出过而从本轮新增量中排除。96462的课程地点时间单选、助人行为多选也属于相对正式旧题的新判断；拆分数学作业并设小目标的单选仍与正式旧题重复。112812仍因原有证据绑定／状态问题没有严格合格题。', '',
        '这里的“新增”按考察的事实与判断比较，同一主题本身不等于同题。排序仍为0，不能用本次口径修正声称排序问题已经解决。逐题新颖性属于Codex离线诊断，不是独立人工认证。', '',
        '正式来源已锁定为 `h4_unexpanded122_qa_generation_only_v1` 的三个题池（34/35/35题），反馈依据 `h4_round1_feedback_compiler_v1_live_1` 及其正式评估链。反馈中的M0、clean、inclusive分开记录；Gold一致率不作为语义正确率，训练已见与未见题的混合统计不称为独立测试成绩。', '',
        '新编译反馈只含这104道正式旧题的题面/选项及对应正式反馈，不向模型提供旧Gold。近期试验经验只保留为一般规则，明确要求不要重复所提供的上一轮题目，并保留本轮repair规则。', '',
        '原入口离线prepare/mock已通过：18次历史响应回放，外部HTTP为0；回放仅验证接口、接收和repair路径，不验证新提示的生成效果。当前全文、日期、TEMPORAL合同与既有处理规则保持原样。', '',
        '已完成实验的成本仍为16次HTTP，输入224,357、输出40,465、总264,822 tokens；包含失败/替换primary和repair。按5道有效新增分摊为52,964.4 tokens/题，金额与推理tokens为UNKNOWN；排序有效量为0，不给出有限单题成本。本次修正新增模型调用0。', '',
        '**实际生成时用过的仍是原混合测试清单。上述5道是纠正比较范围后的结果，不是用新反馈重新生成的成绩；新反馈是否更有效尚未测量。**', '',
        '查看路径：', '',
        '- [来源锁定](../previous_round_sources.json)',
        '- [反馈编译清单](../COMPILED_FEEDBACK.json)',
        '- [正式反馈依据](../formal_feedback_observations.md)',
        '- [逐题修正记录](CORRECTED_ROWS.json)',
        '- [离线验证](../OFFLINE_VERIFICATION.md)',
        '- [全部产物与正确入口](../README.md)',
        '- [259192逐题新颖性](NOVELTY_259192.md) · [96462逐题新颖性](NOVELTY_96462.md) · [112812逐题新颖性](NOVELTY_112812.md)', '']
    (BASE / 'reports/CORRECTED_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'status': result['status'], 'valid_new': total_new, 'by_type': dict(types), 'new_http': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
