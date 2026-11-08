"""Summarize completed local Solver runs without modifying questions or labels."""
from collections import Counter
from pathlib import Path
import csv
import io
import json

import evaluate as e


def cell(metric, mode):
    m = metric[mode]
    if not m['responses']:
        return '—'
    return f"{m['gold_match_rate']:.1%} ({m['gold_matches']}/{m['responses']})"


def prediction_hist(result, mode):
    return dict(Counter(','.join(q['canonical_prediction']) for q in result['requests']
                        if q['context_mode'] == mode and q['scored']))


def main():
    pre = e.load_preflight()
    candidates = {r['candidate_id']: r for r in e.load_jsonl(e.RUN / 'candidate_manifest.jsonl')}
    labels = {r['candidate_id']: r for r in e.load_jsonl(e.RUN / 'review_labels.jsonl')}
    reports, summaries, result_maps = {}, {}, {}
    by_question = {cid: {'candidate': row, 'existing_codex_diagnosis': labels[cid], 'models': {}}
                   for cid, row in candidates.items()}
    usages, verified = [], []
    paired_request_signature = None
    source_protocol = e.load_json(e.RUN / 'solver_protocol.json')
    for model in e.MODELS:
        md = e.RUN / model
        report = e.load_json(md / 'report.json')
        if report['status'] != 'COMPLETE':
            raise RuntimeError('model_incomplete:' + model)
        results = e.load_jsonl(md / 'candidate_results.jsonl')
        summary = e.load_jsonl(md / 'summary.jsonl')
        result_maps[model] = {r['candidate_id']: r for r in results}
        if set(result_maps[model]) != set(candidates):
            raise RuntimeError('candidate_scope_mismatch:' + model)
        reports[model], summaries[model] = report, summary
        ledger = e.Round0RequestLedger(md, solver_contract_sha256=e.sha256_file(md / 'solver_contract.json'))
        counts = ledger.summary()
        if counts['request_count'] != 744 or counts['completed_request_count'] != 744 or counts['actual_solver_generation_count'] != 744:
            raise RuntimeError('request_count_mismatch')
        if counts['failed_request_count'] or counts['ambiguous_request_count'] or counts['retried_generation_count']:
            raise RuntimeError('unexpected_failed_or_retried_request')
        for state in ledger.states():
            envelope = e.load_json(Path(state['response_envelope_path']))
            if e.sha256_file(Path(state['response_envelope_path'])) != state['response_envelope_sha256']:
                raise RuntimeError('response_envelope_drift')
            prompt = Path(state['prompt_path']).read_text()
            if e.sha256_bytes(prompt.encode()) != state['prompt_sha256']:
                raise RuntimeError('raw_prompt_drift')
            if len(state['attempts']) != 1 or not state['decision']['scored']:
                raise RuntimeError('request_not_single_scored_attempt')
            if envelope['response_metadata']['finish_reason'] not in ('stop','eos'):
                raise RuntimeError('completion_not_explicit')
        requests = [q for r in results for q in r['requests']]
        signature = {q['request_id']: {key: q[key] for key in (
            'candidate_id','variant_id','context_mode','slot_id','seed','prompt_sha256',
            'display_to_canonical','canonical_gold','displayed_gold')} for q in requests}
        if paired_request_signature is None:
            paired_request_signature = signature
        elif signature != paired_request_signature:
            raise RuntimeError('cross_model_input_seed_or_gold_drift')
        for result in results:
            candidate = candidates[result['candidate_id']]
            variants = {v['variant_id']: v for v in e.contract.position_variants(candidate)}
            for q in result['requests']:
                rescored = e.contract.score_output(q['raw_output'], candidate, variants[q['variant_id']])
                if any(rescored[k] != q[k] for k in ('correct','scored','canonical_prediction','parse_status')):
                    raise RuntimeError('saved_response_rescoring_mismatch')
        usage = {
            'model': model, 'responses': len(requests),
            'input_tokens': sum(q['backend_metadata']['prompt_token_count'] for q in requests),
            'output_tokens': sum(q['backend_metadata']['completion_token_count'] for q in requests),
            'finish_reasons': dict(Counter(q['backend_metadata']['finish_reason'] for q in requests)),
            'parse_failures': sum(not q['scored'] for q in requests),
            'elapsed_seconds': report['cost']['elapsed_seconds'],
        }
        if usage['input_tokens'] != pre['input_token_budget_per_model']:
            raise RuntimeError('input_usage_preflight_mismatch')
        usages.append(usage)
        summary_map = {r['candidate_id']: r for r in summary}
        for cid, result in result_maps[model].items():
            by_question[cid]['models'][model] = {
                'summary': summary_map[cid],
                'full_context_prediction_counts': prediction_hist(result, 'full_context'),
                'no_context_prediction_counts': prediction_hist(result, 'no_context'),
                'raw_result_path': str(md / 'candidate_results' / (e.sha256_bytes(cid.encode()) + '.json')),
            }
        verified.append({'model': model, 'ledger': counts, 'raw_prompt_and_response_hashes_verified': True})

    comparison = {'status': 'COMPLETE_STOPPED', 'completed_at': e.now(),
                  'source': str(e.SOURCE), 'root_count': 3, 'candidate_count': 29,
                  'question_counts': pre['by_type'], 'models': reports,
                  'usage_by_model': usages,
                  'total_local_generations': sum(u['responses'] for u in usages),
                  'input_tokens': sum(u['input_tokens'] for u in usages),
                  'output_tokens': sum(u['output_tokens'] for u in usages),
                  'external_http_calls': 0, 'retries': 0, 'training_calls': 0,
                  'paid_api_cost': 0, 'local_infrastructure_cost': 'UNKNOWN',
                  'gold_changed': False, 'semantic_filter_applied': False,
                  'automatic_feedback_activation': False, 'automatic_next_experiment': False,
                  'limitations': [
                      'Three selected Train roots, not an independent held-out evaluation.',
                      'All scores compare with original Writer Gold; disagreement is not proof Solver is wrong.',
                      'Codex labels are preexisting offline diagnostics, not independent human certification.',
                      'PASS-deduplicated subset contains 13 SC/MS and zero original strict PASS ordering.',
                      'The three corrected ordering questions are separate artifacts and were not silently substituted.',
                      'Full-context inputs retain synthetic conversation dates and UNKNOWN R02 dates.',
                      'No claim of training readiness or production promotion is made.',
                  ]}
    comparison['supported_ordering_original_gold'] = {
        model: e.metrics([r for r in summaries[model] if r['question_type'] == 'ordering'
                         and r['whole_source_support'] == 'PASS']) for model in e.MODELS}
    comparison['strict_pass_deduplicated_by_root'] = {
        model: {root: e.metrics([r for r in summaries[model] if r['strict_pass_deduplicated']
                                and r['root_id'] == root]) for root in e.ROOTS} for model in e.MODELS}
    comparison['strict_pass_deduplicated_by_type'] = {
        model: {qt: e.metrics([r for r in summaries[model] if r['strict_pass_deduplicated']
                              and r['question_type'] == qt]) for qt in ('single_choice','multi_select','ordering')}
        for model in e.MODELS}
    e.write_once_json(e.HERE / 'RESULTS.json', comparison)
    e.write_once_json(e.HERE / 'QUESTION_RESULTS.json', list(by_question.values()))

    title_names = {'M0': 'M0 基座', 'M_clean_policy': 'Clean', 'M_inclusive_policy': 'Inclusive'}
    text = [
        '# 三个 root 的 Solver 评估', '',
        '已完成并停止。对原程序接收的29道题进行本地评估，未用Codex标签预先过滤，未改题干、选项、证据或Gold。', '',
        '三个root：259192（9题）、96462（10题）、112812（10题）；218519停在Gate候选不足，没有出题，不纳入。', '',
        '## 模型与口径', '',
        '沿用本地Qwen3-4B基座M0，以及第一轮训练后的Clean、Inclusive。每个位置3次采样，完整原文/无原文两条件；单选4位置、多选5位置、排序4位置。每模型744次，共2232次。', '',
        f"实际vLLM版本：{source_protocol['generation_config']['vllm_version']}。SC温度0.7、MS/排序0.8，top_p=0.9，max_tokens=32，thinking=false。", '',
        '完整原文保留Writer实际使用的日期段头与R02 UNKNOWN；复用原Solver题干、系统提示、置换和评分代码。日期渲染适配已明确记录，因此不声称与历史无日期提示逐字相同。', '',
        '**下面全量结果是与原Gold的一致率。已知FAIL题上的高一致率也可能意味着跟随错误Gold，不能当作正确性认证。**', '',
        '## 全部29题', '',
        '|模型|有原文：原Gold一致率|无原文：原Gold一致率|', '|---|---:|---:|',
    ]
    for model in e.MODELS:
        m = reports[model]['all_original_questions']
        text.append(f"|{title_names[model]}|{cell(m,'full_context')}|{cell(m,'no_context')}|")
    text.extend(['', '## 既有Codex PASS且去重的13题', '',
                 '这一子集是10单选＋3多选；没有原样严格通过的排序题。它只是既有离线诊断通过集，不是独立人工真值。', '',
                 '|模型|有原文：原Gold一致率|无原文：原Gold一致率|', '|---|---:|---:|'])
    for model in e.MODELS:
        m = reports[model]['strict_pass_deduplicated']
        text.append(f"|{title_names[model]}|{cell(m,'full_context')}|{cell(m,'no_context')}|")
    text.extend(['', '同一13题按原难度规则分类（不执行训练准入）：', '',
                 '|模型|hard_candidate|有原文全对、无原文未全对|两条件均全对|有原文反而更低|有原文全错|',
                 '|---|---:|---:|---:|---:|---:|'])
    for model in e.MODELS:
        buckets = reports[model]['strict_pass_deduplicated']['difficulty_buckets_against_original_gold']
        text.append('|' + title_names[model] + '|' + '|'.join(str(buckets.get(k,0)) for k in
            ('hard_candidate','full_context_resolved','easy','full_context_worse','full_zero_review')) + '|')
    text.extend(['', '审核PASS且去重集合按题型：', '', '|模型|题型|题数|有原文|无原文|', '|---|---|---:|---:|---:|'])
    for model in e.MODELS:
        for qt in ('single_choice','multi_select'):
            m = comparison['strict_pass_deduplicated_by_type'][model][qt]
            text.append(f"|{title_names[model]}|{qt}|{m['candidate_count']}|{cell(m,'full_context')}|{cell(m,'no_context')}|")
    text.extend(['', '## 全量按题型', '', '|模型|题型|题数|有原文|无原文|', '|---|---|---:|---:|---:|'])
    for model in e.MODELS:
        for qt in ('single_choice','multi_select','ordering'):
            m = reports[model]['by_type'][qt]
            text.append(f"|{title_names[model]}|{qt}|{m['candidate_count']}|{cell(m,'full_context')}|{cell(m,'no_context')}|")
    text.extend(['', '## 全量按root', '', '|模型|root|题数|有原文|无原文|', '|---|---|---:|---:|---:|'])
    for model in e.MODELS:
        for root in e.ROOTS:
            m = reports[model]['by_root'][root]
            text.append(f"|{title_names[model]}|{root.split('-')[-1]}|{m['candidate_count']}|{cell(m,'full_context')}|{cell(m,'no_context')}|")
    text.extend(['', '## 按既有内容诊断标签', '', '|模型|标签|题数|有原文|无原文|', '|---|---|---:|---:|---:|'])
    for model in e.MODELS:
        for label in ('PASS','HOLD','FAIL'):
            m = reports[model]['by_review'][label]
            text.append(f"|{title_names[model]}|{label}|{m['candidate_count']}|{cell(m,'full_context')}|{cell(m,'no_context')}|")
    text.extend(['', '## 全文支持原Gold的3道排序（原始引用绑定仍不完整）', '',
                 '这些题原先为HOLD。这里只分层查看Solver对原Gold的表现，没有把它们自动改为严格PASS，也没有替换为后续Codex修订版本。', '',
                 '|模型|有原文|无原文|', '|---|---:|---:|'])
    for model in e.MODELS:
        m = comparison['supported_ordering_original_gold'][model]
        text.append(f"|{title_names[model]}|{cell(m,'full_context')}|{cell(m,'no_context')}|")
    text.extend(['', '## 调用与消耗', '', '|模型|本地推理次数|输入tokens|输出tokens|运行秒数|', '|---|---:|---:|---:|---:|'])
    for u in usages:
        text.append(f"|{title_names[u['model']]}|{u['responses']}|{u['input_tokens']:,}|{u['output_tokens']:,}|{u['elapsed_seconds']}|")
    text.extend(['', f"总输入 {comparison['input_tokens']:,} tokens，总输出 {comparison['output_tokens']:,} tokens。外部API/HTTP请求0，重试0，训练0；本地算力金额UNKNOWN，不能称本地推理没有成本。", '',
                 '没有启动新的出题、审核模型、Memory、反馈激活或训练。三个root是已知Train案例，结果不能外推全数据集，也不能用于证明第一轮训练的独立泛化收益。', '',
                 '## 查看原始结果', '',
                 '- [逐题题干、选项、原Gold、三个模型答案分布](QUESTION_RESULTS.md)',
                 '- [机器可读逐题结果](QUESTION_RESULTS.json)',
                 '- [完整汇总](RESULTS.json)',
                 '- [预检与预算](run/preflight_report.json)',
                 '- [评估协议](run/solver_protocol.json)',
                 '- [验证结果](VERIFICATION.json)',
                 '- 每模型run目录下的candidate_results.jsonl保留全部响应，raw_prompts/与raw_responses/保留实际输入输出。', ''])
    e.write_once_text(e.HERE / 'RESULTS.md', '\n'.join(text))

    readable = ['# 逐题Solver结果', '', '所有预测均已还原至原始选项/节点ID；Gold和题目保留原样。', '']
    table = io.StringIO()
    writer = csv.DictWriter(table, fieldnames=['root_id','candidate_id','question_type','review','strict_pass_deduplicated',
        'model','original_gold','full_gold_matches','full_responses','no_gold_matches','no_responses','difficulty_bucket'], delimiter='\t')
    writer.writeheader()
    for cid, row in by_question.items():
        c, label = row['candidate'], row['existing_codex_diagnosis']
        readable.extend([f"## {cid}", '', f"类型：{c['question_type']}；原诊断：{label['verdict']}；严格PASS去重集：{label['strict_pass_deduplicated']}", '',
                         c['question'], '', *[f"- {k}：{v}" for k,v in (c.get('options') or c.get('items')).items()], '',
                         f"原Gold：{','.join(e.contract.canonical_gold(c))}", '',
                         '原诊断理由：' + label['reason'], '',
                         '|模型|有原文匹配原Gold|有原文答案分布|无原文答案分布|', '|---|---:|---|---|'])
        for model in e.MODELS:
            detail = row['models'][model]
            s = detail['summary']
            readable.append(f"|{title_names[model]}|{s['full_context_correct_count']}/{s['full_context_total_count']}|{json.dumps(detail['full_context_prediction_counts'],ensure_ascii=False)}|{json.dumps(detail['no_context_prediction_counts'],ensure_ascii=False)}|")
            writer.writerow({'root_id': c['root_id'], 'candidate_id': cid, 'question_type': c['question_type'],
                'review': label['verdict'], 'strict_pass_deduplicated': label['strict_pass_deduplicated'],
                'model': model, 'original_gold': ','.join(e.contract.canonical_gold(c)),
                'full_gold_matches': s['full_context_correct_count'], 'full_responses': s['full_context_total_count'],
                'no_gold_matches': s['no_context_correct_count'], 'no_responses': s['no_context_total_count'],
                'difficulty_bucket': s['difficulty_bucket']})
        readable.append('')
    e.write_once_text(e.HERE / 'QUESTION_RESULTS.md', '\n'.join(readable))
    e.write_once_text(e.HERE / 'QUESTION_RESULTS.tsv', table.getvalue())
    e.verify_records(pre['sources'])
    e.write_once_json(e.HERE / 'VERIFICATION.json', {'status': 'PASS', 'models': verified,
        'source_files_verified_unchanged': len(pre['sources']), 'source_and_gold_unchanged': True,
        'candidate_count_per_model': 29, 'model_count': 3, 'actual_local_generations': 2232,
        'external_api_calls': 0, 'training_calls': 0, 'retries': 0,
        'cross_model_prompts_seeds_gold_identical': True, 'saved_outputs_rescored_identically': True})
    e.write_once_json(e.RUN / 'final_status.json', {
        'status': 'COMPLETED_STOPPED', 'completed_at': comparison['completed_at'],
        'models_completed': list(e.MODELS), 'total_local_generations': 2232,
        'training_started': False, 'feedback_activated': False, 'automatic_next_experiment': False})
    print(json.dumps({k: comparison[k] for k in ('status','total_local_generations','input_tokens','output_tokens')}))


if __name__ == '__main__':
    main()
