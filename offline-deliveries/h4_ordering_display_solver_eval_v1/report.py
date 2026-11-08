"""Read completed local runs, independently verify raw records, then report.

No model/tokenizer loading, generation, GPU, network, or source modification.
Both bounded model runs must be complete; existing reports are never replaced.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
RUN = HERE / 'run'
WORKTREE = Path('/root/autodl-tmp/gxy-selfplay/worktrees/h4-ordering-display-v2')
sys.path.insert(0, str(WORKTREE))

from scripts.current_method.restart_n5_spell.h4_ordering_display_v2.contract import RELABEL_POLICY, permutation_version
from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1 import contract
from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1.io_utils import stable_seed

MODELS = ('M0', 'M_clean_policy')
MODES = ('full_context', 'no_context')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def read(path, *, sidecar=False):
    path = Path(path)
    if sidecar:
        expected = path.with_name(path.name + '.sha256').read_text().strip()
        require(sha(path) == expected, 'sha_sidecar_mismatch:' + str(path))
    return json.loads(path.read_text())


def indexed(rows, key):
    result = {r[key]: r for r in rows}
    require(len(result) == len(rows), 'duplicate_identity:' + key)
    return result


def metric(rows):
    total = len(rows)
    correct = sum(r['correct'] is True for r in rows)
    return {'correct': correct, 'total': total,
            'accuracy': correct / total if total else None,
            'failures': sum(not r['scored'] for r in rows),
            'canonical_prediction_distribution': dict(sorted(Counter(
                ','.join(r['canonical_prediction']) for r in rows).items()))}


def contextual(rows):
    result = {mode: metric([r for r in rows if r['context_mode'] == mode]) for mode in MODES}
    result['full_minus_no_percentage_points'] = 100 * (
        result['full_context']['accuracy'] - result['no_context']['accuracy'])
    return result


def paired(left, right):
    """Paired response counts; never treated as independent question counts."""
    require(set(left) == set(right), 'paired_response_key_mismatch')
    both = left_only = right_only = neither = 0
    for key in left:
        a, b = left[key]['correct'], right[key]['correct']
        both += bool(a and b)
        left_only += bool(a and not b)
        right_only += bool(b and not a)
        neither += bool(not a and not b)
    return {'both_correct': both, 'left_only_correct': left_only,
            'right_only_correct': right_only, 'both_wrong': neither,
            'paired_responses': len(left), 'independent_samples': False}


def request_key(row):
    return (row['candidate_id'], row['variant_id'], row['context_mode'], row['slot_id'])


def fraction(row):
    return f"{row['correct']}/{row['total']} ({100 * row['accuracy']:.1f}%)"


def main():
    output_names = ('RESULTS.json', 'RESULTS.md', 'QUESTION_RESULTS.json',
                    'QUESTION_RESULTS.md', 'VERIFICATION.json')
    require(all(not (HERE / n).exists() and not (HERE / (n + '.sha256')).exists()
                for n in output_names), 'report_exists_do_not_overwrite')
    preflight = read(RUN / 'preflight.json', sidecar=True)
    protocol = read(RUN / 'protocol.json', sidecar=True)
    identities = read(RUN / 'model_identity.json', sidecar=True)
    require(preflight['status'] == 'PASS', 'preflight_not_pass')
    require(preflight['models'] == protocol['models'] == list(MODELS), 'model_scope_changed')
    require(preflight['candidate_count'] == 5 and preflight['per_model_requests'] == 120
            and preflight['maximum_local_generations'] == 240, 'budget_scope_changed')
    require(preflight['protocol_sha256'] == sha(RUN / 'protocol.json'), 'protocol_hash_drift')
    require(protocol['ordering_display_policy'] == RELABEL_POLICY, 'display_policy_changed')
    require(protocol['permutation_contract_version'] == permutation_version(RELABEL_POLICY), 'display_version_changed')
    for source in protocol['sources']:
        require(sha(source['path']) == source['sha256'], 'frozen_source_drift:' + source['path'])
    prepared = Path(protocol['prepared'])
    pool_path = prepared / 'candidate_manifest.json'
    selected = [q for q in read(pool_path) if q['question_type'] == 'ordering']
    require(len(selected) == 5, 'expected_five_ordering_candidates')
    candidates = indexed(selected, 'candidate_id')
    require(list(candidates) == protocol['candidate_order'], 'candidate_order_drift')
    gates = indexed(read(prepared / 'program_gate_results.json'), 'candidate_id')
    prompts = { (r['candidate_id'], r['variant_id'], r['context_mode']): r
               for r in read(prepared / 'prompt_manifest.json') }
    require(len(prompts) == 40, 'prepared_prompt_count')
    for prompt in prompts.values():
        require(sha(prompt['path']) == prompt['sha256'], 'prepared_prompt_sha_drift')

    all_results, model_reports, request_maps, signatures, costs = {}, {}, {}, {}, {}
    verified_rows = 0
    verification_counts = {}
    for model in MODELS:
        md = RUN / model
        model_report = read(md / 'report.json', sidecar=True)
        require(model_report['status'] == 'COMPLETE' and model_report['model'] == model,
                'model_incomplete:' + model)
        require(not (md / 'STOPPED.json').exists(), 'stopped_model_requires_review:' + model)
        frozen = read(md / 'solver_contract.json', sidecar=True)
        contract_sha = sha(md / 'solver_contract.json')
        require(frozen['protocol_sha256'] == sha(RUN / 'protocol.json'), 'model_protocol_mismatch')
        require(frozen['model'] == model, 'model_contract_identity_mismatch')
        model_path = identities['models'][model]['merged_model_path']
        require(frozen['model_path'] == model_path, 'checkpoint_identity_mismatch')
        results = read(md / 'candidate_results.json', sidecar=True)
        result_index = indexed(results, 'candidate_id')
        require(set(result_index) == set(candidates), 'result_candidate_scope_mismatch')
        state_paths = sorted((md / 'request_ledger').glob('*/*.json'))
        state_rows = [read(p) for p in state_paths]
        states = indexed(state_rows, 'request_cache_key')
        require(len(states) == 120, 'ledger_request_count_not_120')
        state_path_index = {r['request_cache_key']: str(p) for p, r in zip(state_paths, state_rows)}
        require(len(list((md / 'raw_responses').glob('*.json'))) == 120, 'raw_response_count_not_120')
        model_requests, signature = {}, {}
        for cid, candidate in candidates.items():
            result = result_index[cid]
            original_result_path = md / 'candidate_results' / (text_sha(cid) + '.json')
            require(read(original_result_path, sidecar=True) == result, 'individual_result_drift')
            require(result['solver_contract_sha256'] == contract_sha, 'result_contract_drift')
            require(result['root_id'] == candidate['root_id'] and result['question_type'] == 'ordering', 'result_identity_drift')
            require(result['candidate_pool_sha256'] == sha(pool_path), 'result_pool_sha_drift')
            dialogue_sha = sha(prepared / 'dialogues' / (candidate['root_id'] + '.json'))
            require(result['dialogue_sha256'] == dialogue_sha, 'result_dialogue_sha_drift')
            require(result['ordering_display_policy'] == RELABEL_POLICY, 'result_policy_drift')
            variants = {v['variant_id']: v for v in contract.position_variants(
                candidate, ordering_display_policy=RELABEL_POLICY)}
            require(contract.audit_rebinding(candidate, list(variants.values()))['pass'], 'variant_audit_failure')
            expected = {(cid, vid, mode, slot) for vid in variants for mode in MODES for slot in range(3)}
            require(len(result['requests']) == result['request_count'] == 24, 'candidate_request_count')
            require({request_key(q) for q in result['requests']} == expected, 'request_coverage_mismatch')
            require(result['technical_failure_count'] == 0, 'unexpected_technical_failure')
            gate_prompts = {(q['variant_id'], q['context_mode']): q for q in gates[cid]['prompt_audit']}
            for q in result['requests']:
                key = request_key(q)
                require(key not in model_requests, 'duplicate_model_request')
                variant = variants[q['variant_id']]
                state = states[q['request_cache_key']]
                req = state['request_contract']
                require(text_sha(json.dumps(req, ensure_ascii=False, sort_keys=True, separators=(',', ':')))
                        == q['request_cache_key'], 'cache_key_not_contract_hash')
                require(state['status'] == 'completed' and state['solver_generation_count'] == 1
                        and len(state['attempts']) == 1 and state['attempts'][0]['status'] == 'completed',
                        'unexpected_failure_or_retry')
                for field in ('candidate_id', 'root_id', 'question_type', 'variant_id', 'context_mode',
                              'slot_id', 'seed', 'prompt_sha256', 'display_to_canonical', 'canonical_gold'):
                    require(q[field] == req[field], 'ledger_identity_drift:' + field)
                require(req['solver_contract_sha256'] == contract_sha, 'ledger_contract_sha_drift')
                require(req['candidate_pool_sha256'] == sha(pool_path) and req['dialogue_sha256'] == dialogue_sha,
                        'ledger_source_sha_drift')
                require(req['ordering_display_policy'] == RELABEL_POLICY, 'ledger_policy_drift')
                require(req['generation_config'] == protocol['generation_config'], 'request_generation_config_drift')
                require(req['backend_identity']['model'] == model_path, 'ledger_model_drift')
                require(req['backend_identity']['tokenizer'] == protocol['tokenizer'], 'ledger_tokenizer_drift')
                prompt = prompts[(cid, q['variant_id'], q['context_mode'])]
                require(q['prompt_sha256'] == prompt['sha256'] == state['prompt_sha256'], 'prompt_sha_mismatch')
                require(sha(state['prompt_path']) == prompt['sha256'], 'raw_prompt_sha_mismatch')
                require(Path(state['prompt_path']).read_bytes() == Path(prompt['path']).read_bytes(), 'raw_prompt_text_mismatch')
                gate_prompt = gate_prompts[(q['variant_id'], q['context_mode'])]
                require(q['prompt_sha256'] == gate_prompt['prompt_sha256'], 'gate_prompt_sha_mismatch')
                require(q['prompt_token_count'] == prompt['input_tokens'] == gate_prompt['prompt_token_count'], 'prompt_token_mismatch')
                for field in ('display_to_canonical', 'displayed_gold', 'canonical_gold', 'shift'):
                    require(q[field] == variant[field], 'saved_variant_drift:' + field)
                require(q['seed'] == stable_seed(candidate['root_id'], cid, q['slot_id']), 'paired_seed_drift')
                envelope_path = Path(state['response_envelope_path'])
                require(envelope_path.resolve().is_relative_to(md.resolve()), 'response_outside_model_directory')
                require(sha(envelope_path) == state['response_envelope_sha256'], 'envelope_sha_mismatch')
                envelope = read(envelope_path)
                require(envelope['request_cache_key'] == q['request_cache_key'] and envelope['attempt_id'] == 1, 'envelope_identity_drift')
                require(envelope['model_identifier'] == state['model_identifier'] == model_path, 'response_model_drift')
                require(envelope['raw_text'] == q['raw_output'], 'raw_output_mismatch')
                require(text_sha(q['raw_output']) == envelope['raw_response_sha256'] == state['raw_response_sha256'], 'raw_output_sha_mismatch')
                metadata = {k: v for k, v in envelope['response_metadata'].items() if k != 'reused'}
                require(metadata == q['backend_metadata'], 'response_metadata_drift')
                require(not metadata.get('mock') and metadata.get('model_inference_performed') is not False, 'mock_response_in_live_results')
                require(metadata['finish_reason'] in ('stop', 'eos'), 'nonexplicit_completion')
                require(metadata['sampling_seed'] == q['seed'], 'actual_seed_drift')
                for field in ('temperature', 'top_p', 'max_tokens', 'structured_output_api', 'structured_output_version'):
                    require(metadata[field] == protocol['generation_config'][field], 'actual_generation_drift:' + field)
                require(metadata['prompt_token_count'] == q['prompt_token_count'], 'actual_input_tokens_drift')
                require(isinstance(metadata['completion_token_count'], int), 'missing_output_usage')
                rescored = contract.score_output(q['raw_output'], candidate, variant)
                for field in ('correct', 'scored', 'canonical_prediction', 'display_prediction', 'parse_status', 'failure_type', 'metrics'):
                    require(q[field] == rescored[field], 'exact_rescore_mismatch:' + field)
                manual_canonical = [variant['display_to_canonical'][label] for label in q['display_prediction']]
                require(manual_canonical == q['canonical_prediction'], 'manual_canonical_mapping_mismatch')
                require(q['correct'] == (manual_canonical == candidate['gold']), 'manual_order_exact_match_mismatch')
                for field in ('correct', 'scored', 'canonical_prediction', 'parse_status', 'failure_type'):
                    require(state['decision'][field] == q[field], 'ledger_decision_drift')
                signature[key] = {field: q[field] for field in
                                  ('seed', 'prompt_sha256', 'display_to_canonical', 'displayed_gold', 'canonical_gold')}
                model_requests[key] = {**q, 'raw_response_path': str(envelope_path),
                                       'raw_prompt_path': state['prompt_path'],
                                       'ledger_path': state_path_index[q['request_cache_key']]}
                verified_rows += 1
        require({q['request_cache_key'] for q in model_requests.values()} == set(states), 'unmatched_ledger_requests')
        requests = list(model_requests.values())
        cost = model_report['cost']
        require(cost['local_generations'] == 120 and cost['http_requests'] == cost['retries'] == 0, 'reported_call_cost_mismatch')
        for target, field in [('input_tokens', 'prompt_token_count'), ('output_tokens', 'completion_token_count')]:
            require(cost[target] == sum(q['backend_metadata'][field] for q in requests), 'token_cost_recomputation_mismatch')
        require(cost['input_tokens'] == preflight['input_tokens_per_model'], 'preflight_input_budget_mismatch')
        for mode in MODES:
            m = metric([q for q in requests if q['context_mode'] == mode])
            require(model_report['accuracy'][mode] == {k: m[k] for k in ('correct', 'total')}, 'saved_accuracy_mismatch')
        for field in ('request_count', 'actual_solver_generation_count', 'completed_request_count'):
            require(model_report['ledger'][field] == 120, 'reported_ledger_count_drift')
        for field in ('failed_request_count', 'retried_generation_count', 'ambiguous_request_count'):
            require(model_report['ledger'][field] == 0, 'reported_ledger_failure_or_retry')
        all_results[model], model_reports[model], request_maps[model], signatures[model], costs[model] = (
            result_index, model_report, model_requests, signature, cost)
        verification_counts[model] = {'verified_raw_requests': len(requests), 'exact_rescores': len(requests),
                                      'single_attempts': 120, 'failed_attempts': 0, 'retries': 0}
    require(signatures[MODELS[0]] == signatures[MODELS[1]], 'cross_model_pair_prompt_seed_mapping_drift')
    roots = sorted({q['root_id'] for q in selected})
    model_stats = {}
    for model in MODELS:
        requests = list(request_maps[model].values())
        modes = {mode: {(q['candidate_id'], q['variant_id'], q['slot_id']): q for q in requests
                        if q['context_mode'] == mode} for mode in MODES}
        model_stats[model] = {
            **contextual(requests), 'independent_question_count': 5,
            'by_root': {root: {**contextual([q for q in requests if q['root_id'] == root]),
                               'question_count': sum(c['root_id'] == root for c in selected)} for root in roots},
            'paired_context_outcomes_left_full_right_no': paired(modes['full_context'], modes['no_context']),
            'cost': costs[model], 'model_path': identities['models'][model]['merged_model_path'],
        }
    paired_models = {}
    for mode in MODES:
        left = {k: q for k, q in request_maps['M0'].items() if q['context_mode'] == mode}
        right = {k: q for k, q in request_maps['M_clean_policy'].items() if q['context_mode'] == mode}
        win = lose = tie = 0
        for cid in candidates:
            a = sum(q['correct'] for q in left.values() if q['candidate_id'] == cid)
            b = sum(q['correct'] for q in right.values() if q['candidate_id'] == cid)
            win += b > a; lose += b < a; tie += b == a
        paired_models[mode] = {
            **paired(left, right), 'left_model': 'M0', 'right_model': 'M_clean_policy',
            'clean_minus_base_percentage_points': 100 * (
                model_stats['M_clean_policy'][mode]['accuracy'] - model_stats['M0'][mode]['accuracy']),
            'question_level_clean_higher': win, 'question_level_clean_lower': lose,
            'question_level_tied': tie,
        }
    details = []
    for number, candidate in enumerate(selected, 1):
        cid = candidate['candidate_id']
        item = {'number': number, 'candidate_id': cid, 'source_key': candidate['display_identity'],
                'root_id': candidate['root_id'], 'question': candidate['question'], 'items': candidate['items'],
                'canonical_gold': candidate['gold'], 'reviewed_presentation': candidate.get('reviewed_presentation'),
                'source_manifest': str(pool_path), 'models': {}}
        for model in MODELS:
            rows = [q for q in request_maps[model].values() if q['candidate_id'] == cid]
            variants = sorted({q['variant_id'] for q in rows})
            modes = {mode: {(q['variant_id'], q['slot_id']): q for q in rows if q['context_mode'] == mode} for mode in MODES}
            item['models'][model] = {
                **contextual(rows), 'result_path': str(RUN / model / 'candidate_results' / (text_sha(cid) + '.json')),
                'paired_context_outcomes_left_full_right_no': paired(modes['full_context'], modes['no_context']),
                'by_variant': {v: contextual([q for q in rows if q['variant_id'] == v]) for v in variants},
                'by_seed_slot': {str(s): {'seed': stable_seed(candidate['root_id'], cid, s),
                    **contextual([q for q in rows if q['slot_id'] == s])} for s in range(3)},
                'request_trace': rows,
            }
        item['clean_minus_base_percentage_points'] = {mode: 100 * (
            item['models']['M_clean_policy'][mode]['accuracy'] - item['models']['M0'][mode]['accuracy']) for mode in MODES}
        item['paired_models'] = {mode: paired(
            {(q['variant_id'], q['slot_id']): q for q in item['models']['M0']['request_trace'] if q['context_mode'] == mode},
            {(q['variant_id'], q['slot_id']): q for q in item['models']['M_clean_policy']['request_trace'] if q['context_mode'] == mode},
        ) for mode in MODES}
        details.append(item)
    total_cost = {k: sum(costs[m][k] for m in MODELS) for k in
                  ('local_generations', 'http_requests', 'retries', 'input_tokens', 'output_tokens')}
    total_cost.update(total_tokens=total_cost['input_tokens'] + total_cost['output_tokens'],
                      local_infrastructure_cost='UNKNOWN', paid_api_cost=0,
                      monetary_cost_per_valid_question='UNKNOWN', technical_failures=0,
                      reasoning_tokens='UNKNOWN_NOT_SEPARATELY_REPORTED')
    findings = [
        'Only five existing ordering tasks were evaluated; each condition has 60 responses per model, not 60 independent questions.',
        'This iteration changed ordering display labels and three reviewed stems; it generated no new questions and left event claims/canonical Gold unchanged.',
        'Full/no-context and two checkpoints are paired on identical prompts, mappings, and seeds; these observations describe known Train examples.',
        'There was no paired old-display arm on these same five tasks. Differences from older runs with other question sets cannot causally estimate the effect of feedback or this display fix.',
        'Gold agreement does not independently certify source support. The retained source-support judgments are Codex offline diagnoses.',
    ]
    report = {'version': 'five_ordering_display_solver_results_v1', 'status': 'COMPLETED_STOPPED',
              'models': model_stats, 'paired_models': paired_models, 'candidate_count': 5,
              'question_type': 'ordering', 'contexts': list(MODES), 'display_variants_per_question': 4,
              'rollouts_per_variant_context': 3, 'cost': total_cost, 'limitations': findings,
              'new_question_generation_calls': 0, 'new_training_calls': 0, 'automatic_next_experiment': False,
              'protocol_path': str(RUN / 'protocol.json'), 'question_results_path': str(HERE / 'QUESTION_RESULTS.json')}
    lines = ['# 五道排序题：修复展示后的 Solver 对照', '',
             '本轮完成 M0 与 Clean 两个本地模型对照，共240次推理。只修复已有题的展示编号和其中3道题干；没有重新出题。', '',
             '|模型|有原文|无原文|有−无（百分点）|', '|---|---|---|---:|']
    for model, row in model_stats.items():
        lines.append(f"|{model}|{fraction(row['full_context'])}|{fraction(row['no_context'])}|{row['full_minus_no_percentage_points']:+.1f}|")
    lines += ['', '**这是5道题的重复测量：每模型每条件60次回答，不是60道独立题。**', '',
              '|条件|Clean−M0（百分点）|两者都对|仅M0对|仅Clean对|两者都错|题级Clean胜/负/平|',
              '|---|---:|---:|---:|---:|---:|---|']
    for mode, row in paired_models.items():
        lines.append(f"|{mode}|{row['clean_minus_base_percentage_points']:+.1f}|{row['both_correct']}|{row['left_only_correct']}|{row['right_only_correct']}|{row['both_wrong']}|{row['question_level_clean_higher']}/{row['question_level_clean_lower']}/{row['question_level_tied']}|")
    lines += ['', '题级胜负按每题12次回答的正确数比较；回答级配对未作独立样本显著性推断。', '',
              '|root|模型|题数|有原文|无原文|有−无（百分点）|', '|---|---|---:|---|---|---:|']
    for root in roots:
        for model in MODELS:
            row = model_stats[model]['by_root'][root]
            lines.append(f"|{root.rsplit('-',1)[-1]}|{model}|{row['question_count']}|{fraction(row['full_context'])}|{fraction(row['no_context'])}|{row['full_minus_no_percentage_points']:+.1f}|")
    lines += ['', '|题号|root|M0有/无|Clean有/无|Clean−M0有/无（百分点）|', '|---|---|---|---|---|']
    for item in details:
        a, b = (item['models'][m] for m in MODELS)
        d = item['clean_minus_base_percentage_points']
        lines.append(f"|Q{item['number']}|{item['root_id'].rsplit('-',1)[-1]}|{a['full_context']['correct']}/12 · {a['no_context']['correct']}/12|{b['full_context']['correct']}/12 · {b['no_context']['correct']}/12|{d['full_context']:+.1f} / {d['no_context']:+.1f}|")
    lines += ['', '|模型|本地推理|输入tokens|输出tokens|运行秒数|', '|---|---:|---:|---:|---:|']
    for model, row in costs.items():
        lines.append(f"|{model}|{row['local_generations']}|{row['input_tokens']:,}|{row['output_tokens']:,}|{row['elapsed_seconds']}|")
    lines += ['', f"合计输入 {total_cost['input_tokens']:,}、输出 {total_cost['output_tokens']:,}，共 {total_cost['total_tokens']:,} tokens。技术失败0、重试0、HTTP0、训练0。付费API金额0；本地算力金额及单题货币成本UNKNOWN。", '',
              '两个模型的提示SHA、展示映射和seed完全配对；240条原始响应均重新解析、回绑canonical ID并按完整顺序复算，同时核对账本、原始响应哈希和实际usage。', '',
              '这5题来自已知Train案例，只能作小样本筛查。没有对同一5题运行旧展示对照，因此不能把旧批次与本批差值归因为反馈或展示修复。高Gold一致率也不构成独立语义认证。', '',
              '[逐题题干、选项、Gold、变体/seed与答案分布](QUESTION_RESULTS.md) · [完整机器结果](RESULTS.json) · [核验记录](VERIFICATION.json)', '',
              '本轮完成后停止，不自动追加题目、Solver采样、反馈激活或训练。']
    qlines = ['# 逐题结果', '', '所有预测均映射回原canonical事件ID。每题每模型有/无原文各12次回答。', '']
    for item in details:
        qlines += [f"## Q{item['number']} · {item['source_key']}", '', item['question'], '',
                   *[f"- {k}：{v}" for k, v in item['items'].items()], '',
                   'Canonical Gold：`' + ','.join(item['canonical_gold']) + '`', '',
                   '|模型|有原文|无原文|有−无（百分点）|有原文canonical预测分布|无原文canonical预测分布|',
                   '|---|---|---|---:|---|---|']
        for model, row in item['models'].items():
            qlines.append(f"|{model}|{fraction(row['full_context'])}|{fraction(row['no_context'])}|{row['full_minus_no_percentage_points']:+.1f}|{json.dumps(row['full_context']['canonical_prediction_distribution'])}|{json.dumps(row['no_context']['canonical_prediction_distribution'])}|")
        qlines += ['', '|模型|展示变体|有原文|无原文|', '|---|---|---|---|']
        for model, row in item['models'].items():
            for variant, values in row['by_variant'].items():
                qlines.append(f"|{model}|{variant}|{fraction(values['full_context'])}|{fraction(values['no_context'])}|")
        qlines += ['', '|模型|seed槽|实际seed|有原文|无原文|', '|---|---|---:|---|---|']
        for model, row in item['models'].items():
            for slot, values in row['by_seed_slot'].items():
                qlines.append(f"|{model}|{slot}|{values['seed']}|{fraction(values['full_context'])}|{fraction(values['no_context'])}|")
        qlines += ['', *[f"[{m}原始逐题结果]({item['models'][m]['result_path']})" for m in MODELS], '',
                   '每变体、每seed的canonical预测分布、配对胜负及全部原始提示/响应/账本路径保留在 `QUESTION_RESULTS.json`。', '']
    verification = {'status': 'PASS', 'verified_requests': verified_rows, 'by_model': verification_counts,
                    'paired_model_prompt_seed_and_mapping_identity': True, 'frozen_source_sha_verified': True,
                    'exact_order_rescoring': True, 'manual_display_to_canonical_check': True,
                    'raw_response_and_envelope_sha_checked': True, 'single_attempt_budget_checked': True,
                    'new_model_calls_in_reporting': 0, 'http_calls_in_reporting': 0,
                    'report_script_sha256': sha(__file__), 'limitations': findings}
    payloads = {'RESULTS.json': report, 'QUESTION_RESULTS.json': details, 'VERIFICATION.json': verification}
    rendered = {name: json.dumps(value, ensure_ascii=False, indent=2) + '\n' for name, value in payloads.items()}
    rendered.update({'RESULTS.md': '\n'.join(lines) + '\n', 'QUESTION_RESULTS.md': '\n'.join(qlines) + '\n'})
    for name, value in rendered.items():
        (HERE / name).write_text(value)
        (HERE / (name + '.sha256')).write_text(text_sha(value) + '\n')
    print(json.dumps({'status': 'PASS', 'verified_requests': verified_rows,
                      'models': {m: {mode: model_stats[m][mode] for mode in MODES} for m in MODELS},
                      'cost': total_cost}, ensure_ascii=False))


if __name__ == '__main__':
    main()
