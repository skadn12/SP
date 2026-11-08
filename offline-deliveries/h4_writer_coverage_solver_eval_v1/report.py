"""Read completed local runs, independently verify raw records, then report.

No model/tokenizer loading, generation, GPU, network, or source modification.
All three bounded runs must be complete; existing reports are never replaced.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
RUN = HERE / 'run'
WORKTREE = Path('/root/autodl-tmp/gxy-selfplay/worktrees/h4-writer-coverage-v1')
sys.dont_write_bytecode = True
sys.path.insert(0, str(WORKTREE))

from scripts.current_method.restart_n5_spell.h4_ordering_display_v2.contract import RELABEL_POLICY, permutation_version
from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1 import contract
from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1.io_utils import stable_seed

MODELS = ('M0', 'M_clean_policy', 'M_inclusive_policy')
EXPECTED_TYPES = {'single_choice': 16, 'multi_select': 7, 'ordering': 5}
LIMIT = 714
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
    return {'correct': correct, 'total': total, 'all_correct': bool(total) and correct == total, 'all_wrong': bool(total) and correct == 0,
            'accuracy': correct / total if total else None,
            'failures': sum(not r['scored'] for r in rows),
            'canonical_prediction_distribution': dict(sorted(Counter(
                ','.join(r['canonical_prediction']) for r in rows).items()))}


def contextual(rows):
    result = {mode: metric([r for r in rows if r['context_mode'] == mode]) for mode in MODES}
    for mode in MODES:
        scoped = [r for r in rows if r['context_mode'] == mode]
        cids = {r['candidate_id'] for r in scoped}
        accuracies = [metric([r for r in scoped if r['candidate_id'] == cid])['accuracy'] for cid in cids]
        result[mode]['question_count'] = len(cids)
        result[mode]['macro_accuracy'] = sum(accuracies) / len(accuracies) if accuracies else None
        result[mode]['questions_all_correct'] = sum(v == 1 for v in accuracies)
        result[mode]['questions_all_wrong'] = sum(v == 0 for v in accuracies)
        result[mode]['questions_partly_correct'] = sum(0 < v < 1 for v in accuracies)
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
    offline = read(HERE / 'OFFLINE_CHECKS.json', sidecar=True)
    require(offline['status'] == 'PASS' and offline['real_model_calls'] == 0
            and offline['mock_responses_scored'] == LIMIT
            and offline['protocol_sha256'] == sha(RUN / 'protocol.json'), 'offline_checks_drift')
    require(preflight['status'] == 'PASS', 'preflight_not_pass')
    require(preflight['models'] == protocol['models'] == list(MODELS), 'model_scope_changed')
    require(preflight['candidate_count'] == 28 and preflight['per_model_requests'] == LIMIT
            and preflight['maximum_local_generations'] == LIMIT * len(MODELS), 'budget_scope_changed')
    require(preflight['protocol_sha256'] == sha(RUN / 'protocol.json'), 'protocol_hash_drift')
    require(protocol['ordering_display_policy'] == RELABEL_POLICY, 'display_policy_changed')
    require(protocol['permutation_contract_version'] == permutation_version(RELABEL_POLICY), 'display_version_changed')
    for source in protocol['sources']:
        require(sha(source['path']) == source['sha256'], 'frozen_source_drift:' + source['path'])
    prepared = Path(protocol['prepared'])
    pool_path = prepared / 'candidate_manifest.json'
    selected = read(pool_path)
    require(len(selected) == 28 and dict(Counter(q['question_type'] for q in selected)) == EXPECTED_TYPES, 'candidate_scope_drift')
    review = read(prepared / 'SELECTION_REVIEW.json')
    require(review['status'] == 'PASS' and review['candidate_count'] == len(selected), 'selection_review_drift')
    for source in review.get('sources', []):
        require(sha(source['path']) == source['sha256'], 'selection_source_drift:' + source['path'])
    candidates = indexed(selected, 'candidate_id')
    require(list(candidates) == protocol['candidate_order'], 'candidate_order_drift')
    gates = indexed(read(prepared / 'program_gate_results.json'), 'candidate_id')
    prompts = { (r['candidate_id'], r['variant_id'], r['context_mode']): r
               for r in read(prepared / 'prompt_manifest.json') }
    require(len(prompts) == preflight['unique_prompts'] == 238, 'prepared_prompt_count')
    require(len({p['prompt_sha256'] for p in prompts.values()}) == 238, 'unique_prompt_hash_count')

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
        require(frozen['generation_config_by_type'] == protocol['generation_config_by_type'], 'model_generation_config_drift')
        checkpoint = identities['verified_checkpoints'][model]
        require(checkpoint['path'] == model_path, 'verified_checkpoint_path_mismatch')
        actual_stats = [{'name': row['name'], 'size': (Path(model_path) / row['name']).stat().st_size,
                         'mtime_ns': (Path(model_path) / row['name']).stat().st_mtime_ns}
                        for row in checkpoint['files']]
        require(actual_stats == identities['checkpoint_stats'][model], 'checkpoint_changed_after_frozen_sha_check')
        results = read(md / 'candidate_results.json', sidecar=True)
        result_index = indexed(results, 'candidate_id')
        require(set(result_index) == set(candidates), 'result_candidate_scope_mismatch')
        state_paths = sorted((md / 'request_ledger').glob('*/*.json'))
        state_rows = [read(p) for p in state_paths]
        states = indexed(state_rows, 'request_cache_key')
        require(len(states) == LIMIT, 'ledger_request_count_mismatch')
        state_path_index = {r['request_cache_key']: str(p) for p, r in zip(state_paths, state_rows)}
        require(len(list((md / 'raw_responses').glob('*.json'))) == LIMIT, 'raw_response_count_mismatch')
        require(len(list((md / 'raw_prompts').glob('*.txt'))) == len(prompts), 'raw_prompt_count_mismatch')
        model_requests, signature = {}, {}
        for cid, candidate in candidates.items():
            result = result_index[cid]
            original_result_path = md / 'candidate_results' / (text_sha(cid) + '.json')
            require(read(original_result_path, sidecar=True) == result, 'individual_result_drift')
            require(result['solver_contract_sha256'] == contract_sha, 'result_contract_drift')
            require(result['root_id'] == candidate['root_id'] and result['question_type'] == candidate['question_type'], 'result_identity_drift')
            require(result['candidate_pool_sha256'] == sha(pool_path), 'result_pool_sha_drift')
            dialogue_sha = sha(prepared / 'dialogues' / (candidate['root_id'] + '.json'))
            require(result['dialogue_sha256'] == dialogue_sha, 'result_dialogue_sha_drift')
            require(result['ordering_display_policy'] == RELABEL_POLICY, 'result_policy_drift')
            variants = {v['variant_id']: v for v in contract.position_variants(
                candidate, ordering_display_policy=RELABEL_POLICY)}
            require(len(variants) == {'single_choice': 4, 'multi_select': 5, 'ordering': 4}[candidate['question_type']],
                    'display_count_by_type_drift')
            require(contract.audit_rebinding(candidate, list(variants.values()))['pass'], 'variant_audit_failure')
            expected = {(cid, vid, mode, slot) for vid in variants for mode in MODES for slot in range(3)}
            require(len(result['requests']) == result['request_count'] == len(variants) * 2 * 3, 'candidate_request_count')
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
                require(not state.get('completed_by_recovery'), 'unexpected_recovery')
                require(Path(state_path_index[q['request_cache_key']]).stem == q['request_cache_key'], 'ledger_path_key_mismatch')
                for field in ('candidate_id', 'root_id', 'question_type', 'variant_id', 'context_mode',
                              'slot_id', 'seed', 'prompt_sha256', 'display_to_canonical', 'canonical_gold'):
                    require(q[field] == req[field], 'ledger_identity_drift:' + field)
                require(req['solver_contract_sha256'] == contract_sha, 'ledger_contract_sha_drift')
                require(req['candidate_pool_sha256'] == sha(pool_path) and req['dialogue_sha256'] == dialogue_sha,
                        'ledger_source_sha_drift')
                require(req['ordering_display_policy'] == RELABEL_POLICY, 'ledger_policy_drift')
                require(req['generation_config'] == protocol['generation_config_by_type'][candidate['question_type']], 'request_generation_config_drift')
                require(req['backend_identity']['model'] == model_path, 'ledger_model_drift')
                require(req['backend_identity']['tokenizer'] == protocol['tokenizer'], 'ledger_tokenizer_drift')
                prompt = prompts[(cid, q['variant_id'], q['context_mode'])]
                require(q['prompt_sha256'] == prompt['prompt_sha256'] == state['prompt_sha256'], 'prompt_sha_mismatch')
                require(sha(state['prompt_path']) == prompt['prompt_sha256'], 'raw_prompt_sha_mismatch')
                require(Path(state['prompt_path']).resolve().is_relative_to(md.resolve()), 'prompt_outside_model_directory')
                gate_prompt = gate_prompts[(q['variant_id'], q['context_mode'])]
                require(q['prompt_sha256'] == gate_prompt['prompt_sha256'], 'gate_prompt_sha_mismatch')
                require(q['prompt_token_count'] == prompt['prompt_token_count'] == gate_prompt['prompt_token_count'], 'prompt_token_mismatch')
                for field in ('display_to_canonical', 'displayed_gold', 'canonical_gold', 'shift'):
                    require(q[field] == variant[field], 'saved_variant_drift:' + field)
                require(q['seed'] == stable_seed(candidate['root_id'], cid, q['slot_id']), 'paired_seed_drift')
                envelope_path = Path(state['response_envelope_path'])
                require(envelope_path.resolve().is_relative_to(md.resolve()), 'response_outside_model_directory')
                require(sha(envelope_path) == state['response_envelope_sha256'], 'envelope_sha_mismatch')
                envelope = read(envelope_path)
                require(envelope['request_cache_key'] == q['request_cache_key'] and envelope['attempt_id'] == 1, 'envelope_identity_drift')
                require(envelope_path.stem == q['request_cache_key'], 'response_path_key_mismatch')
                require(envelope['model_identifier'] == state['model_identifier'] == model_path, 'response_model_drift')
                require(envelope['raw_text'] == q['raw_output'], 'raw_output_mismatch')
                require(text_sha(q['raw_output']) == envelope['raw_response_sha256'] == state['raw_response_sha256'], 'raw_output_sha_mismatch')
                metadata = {k: v for k, v in envelope['response_metadata'].items() if k != 'reused'}
                require(metadata == q['backend_metadata'], 'response_metadata_drift')
                require(metadata == {k: v for k, v in state['response_metadata'].items() if k != 'reused'}, 'ledger_response_metadata_drift')
                require(not metadata.get('mock') and metadata.get('model_inference_performed') is not False, 'mock_response_in_live_results')
                require(metadata['finish_reason'] in ('stop', 'eos'), 'nonexplicit_completion')
                require(metadata['sampling_seed'] == q['seed'], 'actual_seed_drift')
                actual_fields = ['temperature', 'top_p', 'max_tokens']
                if candidate['question_type'] != 'single_choice':
                    actual_fields += ['structured_output_api', 'structured_output_version']
                else:
                    require(metadata['enable_thinking'] is False and metadata['stop_sequences'] == [], 'single_output_mode_drift')
                for field in actual_fields:
                    require(metadata[field] == protocol['generation_config_by_type'][candidate['question_type']][field], 'actual_generation_drift:' + field)
                require(metadata['prompt_token_count'] == q['prompt_token_count'], 'actual_input_tokens_drift')
                require(isinstance(metadata['completion_token_count'], int), 'missing_output_usage')
                require(0 < metadata['completion_token_count'] <= 32, 'output_token_budget_violation')
                rescored = contract.score_output(q['raw_output'], candidate, variant)
                for field in ('correct', 'scored', 'canonical_prediction', 'display_prediction', 'parse_status', 'failure_type', 'metrics'):
                    require(q[field] == rescored[field], 'exact_rescore_mismatch:' + field)
                manual_canonical = [variant['display_to_canonical'][label] for label in q['display_prediction']]
                gold = contract.canonical_gold(candidate)
                if candidate['question_type'] == 'multi_select':
                    manual_canonical = sorted(set(manual_canonical))
                    gold = sorted(set(gold))
                require(manual_canonical == q['canonical_prediction'], 'manual_canonical_mapping_mismatch')
                require(q['correct'] == (manual_canonical == gold), 'manual_exact_match_mismatch')
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
        require(cost['local_generations'] == LIMIT and cost['http_requests'] == cost['retries'] == 0, 'reported_call_cost_mismatch')
        for target, field in [('input_tokens', 'prompt_token_count'), ('output_tokens', 'completion_token_count')]:
            require(cost[target] == sum(q['backend_metadata'][field] for q in requests), 'token_cost_recomputation_mismatch')
        require(cost['input_tokens'] == preflight['input_tokens_per_model'], 'preflight_input_budget_mismatch')
        for mode in MODES:
            m = metric([q for q in requests if q['context_mode'] == mode])
            require(model_report['accuracy'][mode] == {k: m[k] for k in ('correct', 'total')}, 'saved_accuracy_mismatch')
        for field in ('request_count', 'actual_solver_generation_count', 'completed_request_count'):
            require(model_report['ledger'][field] == LIMIT, 'reported_ledger_count_drift')
        for field in ('failed_request_count', 'retried_generation_count', 'ambiguous_request_count'):
            require(model_report['ledger'][field] == 0, 'reported_ledger_failure_or_retry')
        all_results[model], model_reports[model], request_maps[model], signatures[model], costs[model] = (
            result_index, model_report, model_requests, signature, cost)
        verification_counts[model] = {'verified_raw_requests': len(requests), 'exact_rescores': len(requests),
                                      'single_attempts': LIMIT, 'failed_attempts': 0, 'retries': 0}
    require(all(signatures[MODELS[0]] == signatures[m] for m in MODELS[1:]), 'cross_model_pair_prompt_seed_mapping_drift')
    roots = sorted({q['root_id'] for q in selected})
    types = tuple(EXPECTED_TYPES)
    model_stats = {}
    for model in MODELS:
        requests = list(request_maps[model].values())
        modes = {mode: {(q['candidate_id'], q['variant_id'], q['slot_id']): q for q in requests
                        if q['context_mode'] == mode} for mode in MODES}
        model_stats[model] = {
            **contextual(requests), 'unique_question_count': len(selected),
            'by_root': {root: {**contextual([q for q in requests if q['root_id'] == root]),
                               'question_count': sum(c['root_id'] == root for c in selected)} for root in roots},
            'by_type': {kind: {**contextual([q for q in requests if q['question_type'] == kind]),
                               'question_count': EXPECTED_TYPES[kind]} for kind in types},
            'by_root_and_type': {root: {
                kind: {**contextual([q for q in requests if q['root_id'] == root and q['question_type'] == kind]),
                       'question_count': sum(c['root_id'] == root and c['question_type'] == kind for c in selected)}
                for kind in types} for root in roots},
            'paired_context_outcomes_left_full_right_no': paired(modes['full_context'], modes['no_context']),
            'cost': costs[model], 'model_path': identities['models'][model]['merged_model_path'],
        }
    paired_models = {}
    for left_model, right_model in combinations(MODELS, 2):
        pair_name = right_model + '_minus_' + left_model
        paired_models[pair_name] = {}
        for mode in MODES:
            left = {k: q for k, q in request_maps[left_model].items() if q['context_mode'] == mode}
            right = {k: q for k, q in request_maps[right_model].items() if q['context_mode'] == mode}
            wins = losses = ties = 0
            for cid in candidates:
                a = sum(q['correct'] for q in left.values() if q['candidate_id'] == cid)
                b = sum(q['correct'] for q in right.values() if q['candidate_id'] == cid)
                wins += b > a
                losses += b < a
                ties += b == a
            paired_models[pair_name][mode] = {
                **paired(left, right), 'left_model': left_model, 'right_model': right_model,
                'right_minus_left_micro_percentage_points': 100 * (
                    model_stats[right_model][mode]['accuracy'] - model_stats[left_model][mode]['accuracy']),
                'right_minus_left_macro_percentage_points': 100 * (
                    model_stats[right_model][mode]['macro_accuracy'] - model_stats[left_model][mode]['macro_accuracy']),
                'question_level_right_higher': wins, 'question_level_right_lower': losses,
                'question_level_tied': ties,
            }
    details = []
    for number, candidate in enumerate(selected, 1):
        cid = candidate['candidate_id']
        item = {'number': number, 'candidate_id': cid,
                'source_key': candidate.get('display_identity', cid),
                'root_id': candidate['root_id'], 'question_type': candidate['question_type'],
                'question': candidate['question'], 'options': candidate.get('options'),
                'items': candidate.get('items'), 'canonical_gold': contract.canonical_gold(candidate),
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
        item['model_differences'] = {}
        for left, right in combinations(MODELS, 2):
            item['model_differences'][right + '_minus_' + left] = {
                mode: {'percentage_points': 100 * (
                    item['models'][right][mode]['accuracy'] - item['models'][left][mode]['accuracy']),
                    **paired(
                        {(q['variant_id'], q['slot_id']): q for q in item['models'][left]['request_trace'] if q['context_mode'] == mode},
                        {(q['variant_id'], q['slot_id']): q for q in item['models'][right]['request_trace'] if q['context_mode'] == mode})}
                for mode in MODES}
        base_full = item['models']['M0']['full_context']['accuracy']
        base_no = item['models']['M0']['no_context']['accuracy']
        item['training_signal_observations'] = {
            'M0_full_context_partly_correct': 0 < base_full < 1,
            'M0_full_context_all_correct': base_full == 1,
            'M0_full_context_all_wrong': base_full == 0,
            'M0_full_greater_than_no': base_full > base_no,
            'M0_full_positive_no_zero': base_full > 0 and base_no == 0,
            'M0_full_equals_no': base_full == base_no,
            'model_full_context_accuracies_differ': len({item['models'][m]['full_context']['accuracy'] for m in MODELS}) > 1,
            'training_admission_decision': 'NOT_ASSESSED_BY_THIS_EVALUATION',
        }
        details.append(item)
    observation_fields = [key for key, value in details[0]['training_signal_observations'].items() if isinstance(value, bool)]
    signal = {'question_count': len(details),
              **{key: sum(q['training_signal_observations'][key] for q in details) for key in observation_fields},
              'M0_partly_correct_candidate_ids': [q['candidate_id'] for q in details
                    if q['training_signal_observations']['M0_full_context_partly_correct']],
              'by_type': {kind: {key: sum(q['training_signal_observations'][key] for q in details if q['question_type'] == kind)
                                for key in observation_fields} for kind in types},
              'interpretation': 'Descriptions of observed responses only. No automatic training admission, difficulty certification, or independent semantic support judgment.'}
    total_cost = {k: sum(costs[m][k] for m in MODELS) for k in
                  ('local_generations', 'http_requests', 'retries', 'input_tokens', 'output_tokens')}
    total_cost.update(total_tokens=total_cost['input_tokens'] + total_cost['output_tokens'],
                      elapsed_seconds_sum=round(sum(costs[m]['elapsed_seconds'] for m in MODELS), 2),
                      local_infrastructure_cost='UNKNOWN', paid_api_cost=0,
                      monetary_cost_per_valid_question='UNKNOWN', technical_failures=0,
                      reasoning_tokens='UNKNOWN_NOT_SEPARATELY_REPORTED')
    findings = [
        'The 28 questions come from three known Train roots. Each model has 357 repeated responses per context; these are not 357 independent questions.',
        'Micro accuracy divides all correct responses by all responses. Macro accuracy averages each question accuracy with equal question weight. Single-choice/multi-select/ordering have 4/5/4 displays and 12/15/12 responses per context, so micro and macro values can differ.',
        'Within this run all models use identical prompts, display mappings and seeds; full/no-context use the same variants and seeds. No independent-sample significance test is claimed.',
        'Question and context changes between batches prevent causal attribution of historical score differences to Writer feedback, coverage changes, or Gold corrections.',
        'All 23 choice candidates remain as generated; program acceptance is not independent semantic certification. One ordering Gold was corrected in a frozen evaluation copy before inference; see SELECTION_REVIEW.json.',
        'Partial correctness and model disagreement are descriptive signals. They do not automatically admit a question to training or certify difficulty.',
    ]
    report = {'version': 'writer_coverage_three_model_solver_results_v1', 'status': 'COMPLETE',
              'models': model_stats, 'paired_models': paired_models, 'candidate_count': len(selected),
              'by_type': EXPECTED_TYPES, 'contexts': list(MODES), 'unique_prompts': len(prompts),
              'rollouts_per_variant_context': 3, 'responses_per_model_per_context': LIMIT // 2,
              'cost': total_cost, 'limitations': findings, 'training_signal_observations': signal,
              'selection_review': review, 'accuracy_primary': 'micro', 'macro_accuracy_available': True,
              'new_question_generation_calls': 0, 'new_training_calls': 0, 'automatic_next_experiment': False,
              'historical_comparison': 'Reported separately; no historical candidate set is pooled with this run.',
              'protocol_path': str(RUN / 'protocol.json'), 'question_results_path': str(HERE / 'QUESTION_RESULTS.json')}
    lines = ['# Writer coverage：28题、三模型 Solver 评估', '',
             '28题（16单选、7多选、5排序）完成M0、Clean、Inclusive本地配对评估；每模型714次生成，合计2142次。', '',
             '|模型|有原文 micro|无原文 micro|有原文 macro|无原文 macro|有−无 micro（百分点）|',
             '|---|---|---|---:|---:|---:|']
    for model, row in model_stats.items():
        lines.append(f"|{model}|{fraction(row['full_context'])}|{fraction(row['no_context'])}|{100*row['full_context']['macro_accuracy']:.1f}%|{100*row['no_context']['macro_accuracy']:.1f}%|{row['full_minus_no_percentage_points']:+.1f}|")
    lines += ['', 'micro按回答数加权；macro先算每题准确率再对28题等权平均。单选/多选/排序分别为4/5/4种展示×3个seed，每题每种上下文各12/15/12次相关重复；每模型每条件357次回答不等于357道独立题。', '',
              '|右模型−左模型|条件|micro差（百分点）|macro差（百分点）|两者都对|仅左对|仅右对|两者都错|题级右胜/负/平|',
              '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for pair_name, modes in paired_models.items():
        for mode, row in modes.items():
            lines.append(f"|{pair_name}|{mode}|{row['right_minus_left_micro_percentage_points']:+.1f}|{row['right_minus_left_macro_percentage_points']:+.1f}|{row['both_correct']}|{row['left_only_correct']}|{row['right_only_correct']}|{row['both_wrong']}|{row['question_level_right_higher']}/{row['question_level_right_lower']}/{row['question_level_tied']}|")
    for grouping, keys in [('by_root', roots), ('by_type', types)]:
        lines += ['', f"|{'root' if grouping == 'by_root' else '题型'}|模型|题数|有原文 micro|无原文 micro|有原文 macro|无原文 macro|", '|---|---|---:|---|---|---:|---:|']
        for key in keys:
            for model in MODELS:
                row = model_stats[model][grouping][key]
                lines.append(f"|{key}|{model}|{row['question_count']}|{fraction(row['full_context'])}|{fraction(row['no_context'])}|{100*row['full_context']['macro_accuracy']:.1f}%|{100*row['no_context']['macro_accuracy']:.1f}%|")
    lines += ['', '|root|题型|模型|题数|有原文 micro|无原文 micro|有原文 macro|无原文 macro|',
              '|---|---|---|---:|---|---|---:|---:|']
    for root in roots:
        for kind in types:
            for model in MODELS:
                row = model_stats[model]['by_root_and_type'][root][kind]
                lines.append(f"|{root}|{kind}|{model}|{row['question_count']}|{fraction(row['full_context'])}|{fraction(row['no_context'])}|{100*row['full_context']['macro_accuracy']:.1f}%|{100*row['no_context']['macro_accuracy']:.1f}%|")
    lines += ['', '|题号|root|题型|M0有/无|Clean有/无|Inclusive有/无|M0有原文部分正确|', '|---|---|---|---|---|---|---|']
    for item in details:
        values = [f"{item['models'][m]['full_context']['correct']}/{item['models'][m]['full_context']['total']} · {item['models'][m]['no_context']['correct']}/{item['models'][m]['no_context']['total']}" for m in MODELS]
        lines.append(f"|Q{item['number']}|{item['root_id'].rsplit('-',1)[-1]}|{item['question_type']}|" + '|'.join(values) + f"|{'是' if item['training_signal_observations']['M0_full_context_partly_correct'] else '否'}|")
    lines += ['', f"M0有原文部分正确（0<准确率<1）：{signal['M0_full_context_partly_correct']}/28题；全对{signal['M0_full_context_all_correct']}题、全错{signal['M0_full_context_all_wrong']}题。M0有原文高于无原文{signal['M0_full_greater_than_no']}题；三模型有原文准确率存在差异{signal['model_full_context_accuracies_differ']}题。", '',
              '这些是本次回答的训练信号描述，尚未作训练准入判断；全对不自动表示无用，部分正确也不自动表示可训练。', '',
              '|模型|本地生成|输入tokens|输出tokens|运行秒数|', '|---|---:|---:|---:|---:|']
    for model, row in costs.items():
        lines.append(f"|{model}|{row['local_generations']}|{row['input_tokens']:,}|{row['output_tokens']:,}|{row['elapsed_seconds']}|")
    lines += ['', f"合计输入{total_cost['input_tokens']:,}、输出{total_cost['output_tokens']:,}，共{total_cost['total_tokens']:,} tokens。技术失败0、重试0、HTTP0、训练0；本步没有新增Writer生成。付费API金额0；本地基础设施金额UNKNOWN。", '',
              '2142条原始响应均重新解析和计分，并核对真实prompt、response envelope、cache key、账本、seed和展示回绑；三模型逐请求配对一致。', '',
              '选择题23题未修改；1道排序的Gold在生成前经原文审查修正，保留冻结副本与审计。程序通过与本次Gold一致率均不构成独立语义认证。', '',
              '样本来自3个已知Train root，不作普遍能力或独立样本显著性推断。跨批次题集不同，不能把历史准确率差直接归因为Writer覆盖改进。', '',
              '[逐题完整结果](QUESTION_RESULTS.md) · [机器结果](RESULTS.json) · [核验记录](VERIFICATION.json)']
    qlines = ['# 逐题结果', '', '全体预测先映射回canonical ID；单选严格匹配，多选严格集合匹配，排序严格完整顺序匹配。所有“全对”指本轮该上下文下的全部展示×seed回答。', '']
    for item in details:
        qlines += [f"## Q{item['number']} · {item['candidate_id']}", '',
                   f"root：{item['root_id']}；题型：{item['question_type']}。", '', item['question'], '',
                   *[f"- {k}：{v}" for k, v in (item['items'] or item['options']).items()], '',
                   'Canonical Gold：`' + ','.join(item['canonical_gold']) + '`', '',
                   '|模型|有原文|无原文|有/无全对|有−无（百分点）|有原文canonical预测分布|无原文canonical预测分布|',
                   '|---|---|---|---|---:|---|---|']
        for model, row in item['models'].items():
            qlines.append(f"|{model}|{fraction(row['full_context'])}|{fraction(row['no_context'])}|{row['full_context']['all_correct']} / {row['no_context']['all_correct']}|{row['full_minus_no_percentage_points']:+.1f}|{json.dumps(row['full_context']['canonical_prediction_distribution'])}|{json.dumps(row['no_context']['canonical_prediction_distribution'])}|")
        qlines += ['', '|模型对比|有原文差（百分点）|无原文差（百分点）|', '|---|---:|---:|']
        for pair_name, comparison in item['model_differences'].items():
            qlines.append(f"|{pair_name}|{comparison['full_context']['percentage_points']:+.1f}|{comparison['no_context']['percentage_points']:+.1f}|")
        qlines += ['', '|模型|展示变体|有原文|无原文|', '|---|---|---|---|']
        for model, row in item['models'].items():
            for variant, values in row['by_variant'].items():
                qlines.append(f"|{model}|{variant}|{fraction(values['full_context'])}|{fraction(values['no_context'])}|")
        qlines += ['', '|模型|seed槽|实际seed|有原文|无原文|', '|---|---|---:|---|---|']
        for model, row in item['models'].items():
            for slot, values in row['by_seed_slot'].items():
                qlines.append(f"|{model}|{slot}|{values['seed']}|{fraction(values['full_context'])}|{fraction(values['no_context'])}|")
        qlines += ['', *[f"[{m}原始逐题结果]({item['models'][m]['result_path']})" for m in MODELS], '',
                   '全部回答及原始提示/响应/账本路径见 `QUESTION_RESULTS.json`；本报告不作自动训练准入。', '']
    verification = {'status': 'PASS', 'verified_requests': verified_rows, 'by_model': verification_counts,
                    'paired_model_prompt_seed_and_mapping_identity': True, 'frozen_source_sha_verified': True,
                    'selection_source_sha_verified': True, 'all_question_types_exact_rescored': True,
                    'manual_display_to_canonical_check': True, 'raw_prompt_sha_checked': True,
                    'raw_response_and_envelope_sha_checked': True, 'cache_key_contract_hash_checked': True,
                    'single_attempt_budget_checked': True, 'model_identity_stats_unchanged_after_prepare_sha_check': True,
                    'new_model_calls_in_reporting': 0, 'http_calls_in_reporting': 0,
                    'report_script_sha256': sha(__file__), 'protocol_sha256': sha(RUN / 'protocol.json'),
                    'limitations': findings}
    payloads = {'RESULTS.json': report, 'QUESTION_RESULTS.json': details, 'VERIFICATION.json': verification}
    rendered = {name: json.dumps(value, ensure_ascii=False, indent=2) + '\n' for name, value in payloads.items()}
    rendered.update({'RESULTS.md': '\n'.join(lines) + '\n', 'QUESTION_RESULTS.md': '\n'.join(qlines) + '\n'})
    for name, value in rendered.items():
        with (HERE / name).open('x') as stream:
            stream.write(value)
        with (HERE / (name + '.sha256')).open('x') as stream:
            stream.write(text_sha(value) + '\n')
    print(json.dumps({'status': 'PASS', 'verified_requests': verified_rows,
                      'models': {m: {mode: model_stats[m][mode] for mode in MODES} for m in MODELS},
                      'training_signal_observations': signal, 'cost': total_cost}, ensure_ascii=False))


if __name__ == '__main__':
    main()
