"""Verify and report the frozen nine-question local Solver evaluation, once."""
from collections import Counter
import csv
import io
import json
from pathlib import Path

import evaluate as e


MODES = ('full_context', 'no_context')
TYPES = ('single_choice', 'multi_select', 'ordering')
TITLES = {'M0': 'M0 基座', 'M_clean_policy': 'Clean', 'M_inclusive_policy': 'Inclusive'}
TYPE_TITLES = {'single_choice': '单选', 'multi_select': '多选', 'ordering': '排序'}
OUTPUT_NAMES = ('RESULTS.json', 'RESULTS.md', 'QUESTION_RESULTS.json',
                'QUESTION_RESULTS.md', 'QUESTION_RESULTS.tsv', 'VERIFICATION.json')


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def indexed(rows, key, name):
    result = {row[key]: row for row in rows}
    require(len(result) == len(rows), 'duplicate_' + name)
    return result


def metrics(rows):
    result = {'candidate_count': len(rows),
              'by_type': dict(Counter(row['question_type'] for row in rows)),
              'difficulty_buckets': dict(Counter(row['difficulty_bucket'] for row in rows))}
    for mode in MODES:
        correct = sum(row[mode + '_correct_count'] for row in rows)
        count = sum(row[mode + '_total_count'] for row in rows)
        result[mode] = {'correct_count': correct, 'response_count': count,
                        'accuracy': correct / count if count else None}
    return result


def cell(metric, mode):
    row = metric[mode]
    return (f"{row['accuracy']:.1%} ({row['correct_count']}/{row['response_count']})"
            if row['response_count'] else '—')


def prediction_hist(requests, mode):
    return dict(Counter(','.join(row['canonical_prediction']) for row in requests
                        if row['context_mode'] == mode and row['scored']))


def summarize():
    output_paths = [e.HERE / name for name in OUTPUT_NAMES] + [e.RUN / 'final_status.json']
    require(not any(path.exists() or path.with_name(path.name + '.sha256').exists()
                    for path in output_paths), 'report_already_exists_do_not_overwrite_or_regenerate')
    pre = e.load_preflight()
    candidates = indexed(e.load_jsonl(e.RUN / 'candidate_manifest.jsonl'), 'candidate_id', 'candidate_id')
    labels = indexed(e.load_jsonl(e.RUN / 'review_labels.jsonl'), 'candidate_id', 'review_label')
    require(set(labels) == set(candidates), 'review_label_scope_mismatch')
    require(len(candidates) == pre['candidate_count'] == 9, 'candidate_count_mismatch')
    by_type = dict(Counter(row['question_type'] for row in candidates.values()))
    require(by_type == pre['by_type'] == {'single_choice': 4, 'multi_select': 3, 'ordering': 2},
            'question_type_scope_mismatch')
    require(set(row['root_id'] for row in candidates.values()) == set(e.ROOTS), 'root_scope_mismatch')
    require(len(e.MODELS) == 3 and set(e.MODELS) == set(TITLES), 'model_scope_mismatch')
    expected_count = sum(e.candidate_request_count(row) for row in candidates.values())
    require(expected_count == pre['request_count_per_model'], 'request_budget_mismatch')
    require(expected_count * len(e.MODELS) == pre['maximum_local_generations'] == 702,
            'total_generation_budget_mismatch')
    require(e.ROLLOUTS == 3, 'rollout_policy_drift')

    questions = {cid: {'candidate': row, 'original_gold': e.contract.canonical_gold(row),
                       'existing_codex_diagnosis_for_provenance_only': labels[cid],
                       'acceptance_policy': ('CODEX_CORRECTED_ORDERING' if row['question_type'] == 'ordering'
                                             else 'PROGRAM_ACCEPTED_WITHOUT_CODEX_SEMANTIC_GATE'),
                       'models': {}} for cid, row in candidates.items()}
    model_results, usages, verified = {}, [], []
    paired_signature = None
    protocol = e.load_json(e.RUN / 'solver_protocol.json')
    for model in e.MODELS:
        md = e.RUN / model
        report = e.load_json(md / 'report.json')
        require(report['status'] == 'COMPLETE' and report['model'] == model, 'model_incomplete:' + model)
        require(not (md / 'STOPPED.json').exists(), 'model_has_stop_record:' + model)
        results = indexed(e.load_jsonl(md / 'candidate_results.jsonl'), 'candidate_id', 'model_candidate')
        summaries = indexed(e.load_jsonl(md / 'summary.jsonl'), 'candidate_id', 'model_summary')
        require(set(results) == set(summaries) == set(candidates), 'candidate_scope_mismatch:' + model)
        ledger = e.Round0RequestLedger(md, solver_contract_sha256=e.sha256_file(md / 'solver_contract.json'))
        counts = ledger.summary()
        for key in ('request_count', 'completed_request_count', 'actual_solver_generation_count'):
            require(counts[key] == expected_count, 'ledger_count_mismatch:' + model + ':' + key)
        for key in ('failed_request_count', 'ambiguous_request_count', 'retried_generation_count',
                    'abandoned_attempt_count', 'interrupted_attempt_count', 'recovered_request_count',
                    'recovered_response_envelope_count'):
            require(not counts.get(key, 0), 'unexpected_failure_retry_or_recovery:' + model + ':' + key)
        states = indexed(ledger.states(), 'request_cache_key', 'ledger_request')
        envelopes = {}
        for cache_key, state in states.items():
            response_path = Path(state['response_envelope_path'])
            require(e.sha256_file(response_path) == state['response_envelope_sha256'], 'response_envelope_drift')
            envelope = e.load_json(response_path)
            require(e.sha256_file(Path(state['prompt_path'])) == state['prompt_sha256'], 'raw_prompt_drift')
            require(e.sha256_bytes(envelope['raw_text'].encode()) == state['raw_response_sha256']
                    == envelope['raw_response_sha256'], 'raw_response_drift')
            require(state['status'] == 'completed' and len(state['attempts']) == 1
                    and state['solver_generation_count'] == 1 and state['decision']['scored'],
                    'request_not_single_scored_attempt')
            require(envelope['response_metadata']['finish_reason'] in ('stop', 'eos'), 'completion_not_explicit')
            envelopes[cache_key] = envelope

        requests = [q for result in results.values() for q in result['requests']]
        request_ids = indexed(requests, 'request_id', 'request_id')
        require(len(requests) == expected_count, 'saved_request_count_mismatch')
        require({q['request_cache_key'] for q in requests} == set(states), 'result_ledger_scope_mismatch')
        signature = {qid: {key: q[key] for key in (
            'candidate_id', 'variant_id', 'context_mode', 'slot_id', 'seed', 'prompt_sha256',
            'display_to_canonical', 'canonical_gold', 'displayed_gold')} for qid, q in request_ids.items()}
        if paired_signature is None:
            paired_signature = signature
        else:
            require(signature == paired_signature, 'cross_model_prompt_seed_or_gold_drift')

        for cid, result in results.items():
            candidate = candidates[cid]
            raw_result_path = md / 'candidate_results' / (e.sha256_bytes(cid.encode()) + '.json')
            require(e.load_json(raw_result_path) == result, 'aggregate_individual_result_drift')
            require(result['root_id'] == candidate['root_id']
                    and result['question_type'] == candidate['question_type'], 'result_identity_drift')
            variants = {v['variant_id']: v for v in e.contract.position_variants(candidate)}
            expected_slots = {(v, mode, slot) for v in variants for mode in MODES for slot in range(e.ROLLOUTS)}
            slots = [(q['variant_id'], q['context_mode'], q['slot_id']) for q in result['requests']]
            require(len(slots) == len(expected_slots) and set(slots) == expected_slots,
                    'position_context_slot_coverage_mismatch')
            trace = []
            for q in result['requests']:
                state, envelope = states[q['request_cache_key']], envelopes[q['request_cache_key']]
                req = state['request_contract']
                for key in ('candidate_id', 'root_id', 'question_type', 'variant_id', 'context_mode',
                            'slot_id', 'seed', 'prompt_sha256', 'display_to_canonical', 'canonical_gold'):
                    require(q[key] == req[key], 'result_request_contract_drift:' + key)
                require(q['candidate_id'] == cid and q['canonical_gold'] == e.contract.canonical_gold(candidate),
                        'original_gold_or_candidate_drift')
                require(all(q[key] == variants[q['variant_id']][key] for key in
                            ('display_to_canonical', 'canonical_gold', 'displayed_gold', 'shift')),
                        'display_permutation_or_gold_drift')
                require(q['raw_output'] == envelope['raw_text']
                        and q['backend_metadata'] == envelope['response_metadata'], 'saved_response_drift')
                require(q['seed'] == q['backend_metadata']['sampling_seed'], 'actual_sampling_seed_drift')
                rescored = e.contract.score_output(q['raw_output'], candidate, variants[q['variant_id']])
                for key in ('correct', 'scored', 'canonical_prediction', 'parse_status'):
                    require(rescored[key] == q[key] == state['decision'][key], 'saved_response_rescoring_mismatch:' + key)
                trace.append({**q, 'raw_prompt_path': state['prompt_path'],
                              'raw_response_envelope_path': state['response_envelope_path'],
                              'raw_response_envelope_sha256': state['response_envelope_sha256'],
                              'actual_model_identifier': envelope['model_identifier']})
            recomputed = e.summarize_candidate_result(result)
            require(all(summaries[cid].get(key) == value for key, value in recomputed.items()), 'saved_summary_drift')
            require(summaries[cid]['model'] == model and summaries[cid]['root_id'] == candidate['root_id']
                    and summaries[cid]['question_type'] == candidate['question_type'], 'summary_identity_drift')
            questions[cid]['models'][model] = {
                'summary': summaries[cid], 'raw_result_path': str(raw_result_path),
                'full_context_prediction_counts': prediction_hist(result['requests'], 'full_context'),
                'no_context_prediction_counts': prediction_hist(result['requests'], 'no_context'), 'requests': trace}

        rows = list(summaries.values())
        require(report['all_original_questions'] == e.metrics(rows), 'model_aggregate_drift')
        model_results[model] = {
            'overall': metrics(rows),
            'by_type': {qt: metrics([r for r in rows if r['question_type'] == qt]) for qt in TYPES},
            'by_root': {root: metrics([r for r in rows if r['root_id'] == root]) for root in e.ROOTS},
            'by_root_and_type': {root: {qt: metrics([r for r in rows if r['root_id'] == root
                                                    and r['question_type'] == qt]) for qt in TYPES} for root in e.ROOTS},
            'raw_report_path': str(md / 'report.json')}
        usage = {'model': model, 'local_generations': len(requests),
                 'input_tokens': sum(q['backend_metadata']['prompt_token_count'] for q in requests),
                 'output_tokens': sum(q['backend_metadata']['completion_token_count'] for q in requests),
                 'finish_reasons': dict(Counter(q['backend_metadata']['finish_reason'] for q in requests)),
                 'actual_model_identifiers': sorted({env['model_identifier'] for env in envelopes.values()}),
                 'elapsed_seconds': report['cost']['elapsed_seconds']}
        require(usage['input_tokens'] == pre['input_token_budget_per_model'], 'input_usage_preflight_mismatch')
        require(all(usage[key] == report['cost'][key] for key in ('local_generations', 'input_tokens', 'output_tokens')),
                'reported_cost_mismatch')
        require(report['cost']['http_requests'] == report['cost']['retries'] == 0, 'http_or_retry_scope_violation')
        usages.append(usage)
        verified.append({'model': model, 'ledger': counts, 'raw_prompt_and_response_hashes_verified': True})

    e.verify_records(pre['sources'])
    comparison = {
        'status': 'COMPLETE_STOPPED', 'completed_at': e.now(), 'source': str(e.SOURCE),
        'root_count': len(e.ROOTS), 'candidate_count': len(candidates), 'question_counts': by_type,
        'scoring': '准确率：按冻结原 Gold 严格判分；多选须集合完全相同，排序须顺序完全相同。',
        'models': model_results, 'usage_by_model': usages,
        'total_local_generations': sum(u['local_generations'] for u in usages),
        'input_tokens': sum(u['input_tokens'] for u in usages), 'output_tokens': sum(u['output_tokens'] for u in usages),
        'external_http_calls': 0, 'retries': 0, 'training_calls': 0, 'paid_api_cost': 0,
        'local_infrastructure_cost': 'UNKNOWN', 'gold_changed': False,
        'choice_semantic_filter_applied': False, 'ordering_codex_corrected_copies': 2,
        'automatic_feedback_activation': False, 'automatic_next_experiment': False,
        'limitations': ['三个已知 Train root 的9道题，不能外推数据集或作为独立泛化评估。',
                       '选择题按程序接收及去重纳入，未按 Codex 语义诊断过滤。',
                       '两道排序为 Codex 离线补齐证据的副本，题干、选项和原 Gold 未改；不是原模型独立成功。',
                       '准确率以冻结 Gold 判分，评估不重新认证题目或 Gold；Codex 诊断不是独立人工认证。',
                       '原文保留冻结日期及 UNKNOWN；本地算力金额未知。',
                       '本批题目和接收规则不同于旧29题评估，不能把准确率差直接归因于反馈。']}
    verification = {'status': 'PASS', 'models': verified, 'source_files_verified_unchanged': len(pre['sources']),
                    'source_and_gold_unchanged': True, 'candidate_count_per_model': len(candidates),
                    'model_count': len(e.MODELS), 'actual_local_generations': comparison['total_local_generations'],
                    'external_api_calls': 0, 'training_calls': 0, 'retries': 0,
                    'cross_model_prompts_seeds_gold_identical': True,
                    'saved_outputs_rescored_identically': True, 'summaries_recomputed_identically': True,
                    'request_slot_coverage_verified': True,
                    'report_script_sha256': e.sha256_file(Path(__file__))}
    markdown, question_md, question_tsv = render(comparison, questions, protocol)
    # All verification and rendering finish before the first output is created.
    e.write_once_json(e.HERE / 'RESULTS.json', comparison)
    e.write_once_json(e.HERE / 'QUESTION_RESULTS.json', list(questions.values()))
    e.write_once_text(e.HERE / 'RESULTS.md', markdown)
    e.write_once_text(e.HERE / 'QUESTION_RESULTS.md', question_md)
    e.write_once_text(e.HERE / 'QUESTION_RESULTS.tsv', question_tsv)
    e.write_once_json(e.HERE / 'VERIFICATION.json', verification)
    e.write_once_json(e.RUN / 'final_status.json', {
        'status': 'COMPLETED_STOPPED', 'completed_at': comparison['completed_at'],
        'models_completed': list(e.MODELS), 'total_local_generations': comparison['total_local_generations'],
        'training_started': False, 'feedback_activated': False, 'automatic_next_experiment': False})
    print(json.dumps({key: comparison[key] for key in
                      ('status', 'candidate_count', 'total_local_generations', 'input_tokens', 'output_tokens')}, ensure_ascii=False))


def render(comparison, questions, protocol):
    lines = ['# 最新9题的本地 Solver 准确率', '',
             '已完成并停止。4单选、3多选、2排序；259192/96462/112812分别为3/4/2题。', '',
             '选择题不按 Codex 语义诊断过滤；排序使用 Codex 补齐证据的副本，原 Gold、题干和选项未改。', '',
             '每个位置3次采样，全文/无原文两条件。单选4位置、多选5位置、排序4位置；每模型234次，三模型共702次本地推理。每条件每模型117次回答，9道不同题，重复回答不等于独立题目。', '',
             f"vLLM：{protocol['generation_config']['vllm_version']}。沿用既有 Solver 采样、提示、置换、parser 和评分；实际参数与原始请求见冻结协议和账本。", '',
             '准确率按冻结原 Gold 严格判分，多选必须选项集合全对，排序必须整个顺序全对。各项以回答次数加权。', '',
             '|模型|全文准确率|无原文准确率|', '|---|---:|---:|']
    for model in e.MODELS:
        m = comparison['models'][model]['overall']
        lines.append(f"|{TITLES[model]}|{cell(m, 'full_context')}|{cell(m, 'no_context')}|")
    for grouping, title in (('by_type', '按题型'), ('by_root', '按 root')):
        lines.extend(['', '## ' + title, '', '|模型|分组|题数|全文准确率|无原文准确率|', '|---|---|---:|---:|---:|'])
        for model in e.MODELS:
            for name, m in comparison['models'][model][grouping].items():
                label = TYPE_TITLES.get(name, name.split('-')[-1])
                lines.append(f"|{TITLES[model]}|{label}|{m['candidate_count']}|{cell(m, 'full_context')}|{cell(m, 'no_context')}|")
    lines.extend(['', '## 按 root × 题型', '', '|模型|root|题型|题数|全文准确率|无原文准确率|', '|---|---|---|---:|---:|---:|'])
    for model in e.MODELS:
        for root, groups in comparison['models'][model]['by_root_and_type'].items():
            for qt, m in groups.items():
                lines.append(f"|{TITLES[model]}|{root.split('-')[-1]}|{TYPE_TITLES[qt]}|{m['candidate_count']}|{cell(m, 'full_context')}|{cell(m, 'no_context')}|")
    lines.extend(['', '## 难度诊断', '', '沿用既有分类规则，仅作诊断，不执行训练准入。', '',
                  '|模型|hard_candidate|全文全对、无原文未全对|两条件均全对|全文更低|全文全错|', '|---|---:|---:|---:|---:|---:|'])
    for model in e.MODELS:
        buckets = comparison['models'][model]['overall']['difficulty_buckets']
        lines.append('|' + TITLES[model] + '|' + '|'.join(str(buckets.get(key, 0)) for key in
                     ('hard_candidate', 'full_context_resolved', 'easy', 'full_context_worse', 'full_zero_review')) + '|')
    lines.extend(['', '## 调用与成本', '', '|模型|本地推理|输入 tokens|输出 tokens|秒|', '|---|---:|---:|---:|---:|'])
    for usage in comparison['usage_by_model']:
        lines.append(f"|{TITLES[usage['model']]}|{usage['local_generations']}|{usage['input_tokens']:,}|{usage['output_tokens']:,}|{usage['elapsed_seconds']}|")
    lines.extend(['', f"总输入 {comparison['input_tokens']:,}、总输出 {comparison['output_tokens']:,} tokens。HTTP/API、重试和训练均为0；可核实付费API费用0，本地算力金额UNKNOWN。", '',
                  '## 解释边界', '', *['- ' + item for item in comparison['limitations']], '',
                  '## 原始记录', '', '- [逐题答案分布、原 Gold、原始请求路径](QUESTION_RESULTS.md)',
                  '- [逐题 JSON（含全部请求和路径）](QUESTION_RESULTS.json)', '- [逐题 TSV](QUESTION_RESULTS.tsv)',
                  '- [完整汇总](RESULTS.json)', '- [执行核验](VERIFICATION.json)',
                  '- [预检预算](run/preflight_report.json)', '- [冻结协议](run/solver_protocol.json)', ''])
    detail = ['# 逐题 Solver 准确率与答案分布', '', '预测已还原为原始选项/节点 ID；准确率按冻结原 Gold 判分。', '']
    table = io.StringIO()
    writer = csv.DictWriter(table, fieldnames=['root_id', 'candidate_id', 'question_type', 'acceptance_policy',
        'model', 'original_gold', 'full_correct', 'full_responses', 'full_accuracy',
        'no_correct', 'no_responses', 'no_accuracy', 'difficulty_bucket', 'raw_result_path'], delimiter='\t')
    writer.writeheader()
    for cid, row in questions.items():
        candidate = row['candidate']
        detail.extend([f'## {cid}', '', f"类型：{TYPE_TITLES[candidate['question_type']]}；接收方式：{row['acceptance_policy']}", '',
                       candidate['question'], '', *[f'- {key}：{value}' for key, value in
                       (candidate.get('options') or candidate['items']).items()], '',
                       '原 Gold：' + ','.join(row['original_gold']), '',
                       '|模型|全文准确率|无原文准确率|全文答案分布|无原文答案分布|', '|---|---:|---:|---|---|'])
        for model in e.MODELS:
            result = row['models'][model]
            summary = result['summary']
            m = metrics([summary])
            detail.append(f"|{TITLES[model]}|{cell(m, 'full_context')}|{cell(m, 'no_context')}|"
                          f"{json.dumps(result['full_context_prediction_counts'], ensure_ascii=False)}|"
                          f"{json.dumps(result['no_context_prediction_counts'], ensure_ascii=False)}|")
            writer.writerow({'root_id': candidate['root_id'], 'candidate_id': cid,
                'question_type': candidate['question_type'], 'acceptance_policy': row['acceptance_policy'],
                'model': model, 'original_gold': ','.join(row['original_gold']),
                'full_correct': summary['full_context_correct_count'], 'full_responses': summary['full_context_total_count'],
                'full_accuracy': summary['full_context_accuracy'], 'no_correct': summary['no_context_correct_count'],
                'no_responses': summary['no_context_total_count'], 'no_accuracy': summary['no_context_accuracy'],
                'difficulty_bucket': summary['difficulty_bucket'], 'raw_result_path': result['raw_result_path']})
        detail.extend(['', '原始请求与输出（全部位置和采样；路径亦保存在逐题 JSON）：', ''])
        for model in e.MODELS:
            result = row['models'][model]
            detail.append(f"- {TITLES[model]}：[该题完整结果]({result['raw_result_path']})")
            for q in result['requests']:
                detail.append(f"  - {q['context_mode']} / {q['variant_id']} / s{q['slot_id']}："
                              f"[请求]({q['raw_prompt_path']}) · [响应]({q['raw_response_envelope_path']})")
        detail.append('')
    return '\n'.join(lines), '\n'.join(detail), table.getvalue()


def main():
    with e.RunLock(e.RUN):
        summarize()


if __name__ == '__main__':
    main()
