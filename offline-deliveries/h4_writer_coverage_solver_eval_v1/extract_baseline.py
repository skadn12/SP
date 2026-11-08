#!/usr/bin/env python3
"""Recompute frozen first-round baselines without loading models or changing sources."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


DELIVERY = Path(__file__).resolve().parent
COMPARISON = Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_qa_solver_first_round_comparison_v1/COMPARISON.json')
COMPARISON_SHA256 = 'd54e0121ab68febfc721111260a87f00e58cfcbbc71c64902d7ec94b04cd9738'
TYPES = ['single_choice', 'multi_select', 'ordering']
MODELS = ['M0', 'M_clean_policy', 'M_inclusive_policy']
CONTEXTS = ['full_context', 'no_context']


def read_source(path, expected=None):
    path = Path(path)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected is not None:
        assert digest == expected, f'source_hash_mismatch:{path}'
    return json.loads(payload), {'path': str(path), 'sha256': digest}


def from_rollout(n, row):
    return {'candidate_count': n, **{
        context: {
            'correct_count': row[f'{context}_correct_count'],
            'total_count': row[f'{context}_total_count'],
            'accuracy': row[f'{context}_accuracy'],
        } for context in CONTEXTS}}


def from_context(n, row):
    return {'candidate_count': n, **{
        context: {key: row[context][key] for key in ['correct_count', 'total_count', 'accuracy']}
        for context in CONTEXTS}}


def pack(rows):
    for row in rows:
        for context in CONTEXTS:
            score = row[context]
            assert score['accuracy'] == score['correct_count'] / score['total_count']
        assert row['full_context']['total_count'] == row['no_context']['total_count']
    for context in CONTEXTS:
        for key in ['correct_count', 'total_count']:
            assert rows[0][context][key] == sum(row[context][key] for row in rows[1:])
    assert rows[0]['candidate_count'] == sum(row['candidate_count'] for row in rows[1:])
    return {context: {
        'correct_counts': [row[context]['correct_count'] for row in rows],
        'accuracies': [row[context]['accuracy'] for row in rows],
    } for context in CONTEXTS}


def check_comparison(scope, rows, comparison_model):
    prefix = {'first_round_train': 'train', 'first_round_test': 'test',
              'same_three_roots_previous_formal': 'same_roots_old'}[scope]
    assert rows[0]['full_context']['accuracy'] == comparison_model[f'{prefix}_full_accuracy']
    if scope != 'same_three_roots_previous_formal':
        assert rows[0]['no_context']['accuracy'] == comparison_model[f'{prefix}_no_accuracy']
    for question_type, row in zip(TYPES, rows[1:]):
        old = comparison_model['types'][question_type]
        if scope == 'first_round_train':
            for context in CONTEXTS:
                for key in ['correct_count', 'total_count', 'accuracy']:
                    assert row[context][key] == old['train_full'][f'{context}_{key}']
        elif scope == 'first_round_test':
            assert row['candidate_count'] == old['test_n']
            for context, short in [('full_context', 'full'), ('no_context', 'no')]:
                assert row[context] == old[f'test_{short}']
        else:
            assert row['full_context'] == old['same_roots_old_full']


def extract():
    comparison, comparison_source = read_source(COMPARISON, COMPARISON_SHA256)
    sources = {'comparison': comparison_source}
    raw = {}
    for name in ['train', 'train_type', 'test', 'same_roots']:
        expected = comparison['sources'][name]
        raw[name], sources[name] = read_source(expected['path'], expected['sha256'])
        sources[name]['sha256_matches_comparison'] = True
    sources['train']['json_pointer'] = '/models/{model}/rollout_weighted'
    sources['train_type']['json_pointer'] = '/groups/{question_type}/models/{model}/rollout_weighted'
    sources['test'].update(overall_json_pointer='/summaries/{model}',
                           by_type_json_pointer='/summaries/{model}/by_question_type/{question_type}')
    sources['same_roots'].update(
        overall_json_pointer='/aggregate_by_model/{model}/{context_mode}',
        by_type_aggregation='Sum correct_count, total_count and candidate_count over /roots/*/aggregate_by_type_and_model/{question_type}/{model}/{context_mode}; then divide summed correct_count by summed total_count')
    out = {
        'status': 'READ_ONLY_VERIFIED',
        'metric': 'SUM_CORRECT_DIVIDED_BY_SUM_SCORED_WITHIN_CONTEXT',
        'metric_order': ['overall', *TYPES],
        'model_display_names': {'M0': 'M0', 'M_clean_policy': 'Clean', 'M_inclusive_policy': 'Inclusive'},
    }
    scope_sources = {'first_round_train': ['train', 'train_type'], 'first_round_test': ['test'],
                     'same_three_roots_previous_formal': ['same_roots']}
    for scope, source_ids in scope_sources.items():
        scoped = {'models': {}}
        for model in MODELS:
            if scope == 'first_round_train':
                rows = [from_rollout(raw['train']['candidate_count'], raw['train']['models'][model]['rollout_weighted'])]
                rows += [from_rollout(raw['train_type']['groups'][t]['candidate_count'],
                                     raw['train_type']['groups'][t]['models'][model]['rollout_weighted']) for t in TYPES]
            elif scope == 'first_round_test':
                summary = raw['test']['summaries'][model]
                rows = [from_context(summary['candidate_count'], summary)]
                rows += [from_context(summary['by_question_type'][t]['candidate_count'], summary['by_question_type'][t]) for t in TYPES]
            else:
                same = raw['same_roots']
                rows = [from_context(same['candidate_count'], same['aggregate_by_model'][model])]
                for question_type in TYPES:
                    blocks = [r['aggregate_by_type_and_model'][question_type][model] for r in same['roots']]
                    row = {'candidate_count': sum(b['full_context']['candidate_count'] for b in blocks)}
                    for context in CONTEXTS:
                        assert all(b[context]['unknown_candidate_count'] == 0 for b in blocks)
                        assert sum(b[context]['candidate_count'] for b in blocks) == row['candidate_count']
                        correct = sum(b[context]['correct_count'] for b in blocks)
                        total = sum(b[context]['total_count'] for b in blocks)
                        row[context] = {'correct_count': correct, 'total_count': total, 'accuracy': correct / total}
                    rows.append(row)
            previous_model = next(m for m in comparison['models'] if m['model'] == model)
            check_comparison(scope, rows, previous_model)
            candidate_counts = [r['candidate_count'] for r in rows]
            scored_counts = [r['full_context']['total_count'] for r in rows]
            if scoped['models']:
                assert scoped['candidate_counts'] == candidate_counts
                assert scoped['scored_counts_per_context'] == scored_counts
            scoped.update(candidate_counts=candidate_counts, scored_counts_per_context=scored_counts)
            scoped['models'][model] = pack(rows)
        scoped['source_ids'] = source_ids
        if scope == 'first_round_train':
            scoped.update(root_count=raw['train']['root_count'], mixed_training_exposure=True)
            assert scoped['candidate_counts'] == [2294, 1153, 572, 569]
        elif scope == 'first_round_test':
            excluded = raw['test']['excluded_after_manual_review']
            assert set(excluded.values()) == {11}
            scoped.update(root_count=raw['test']['test_root_count'], original_candidate_count=raw['test']['test_candidate_count'], excluded_ordering_count=11)
            assert scoped['candidate_counts'] == [496, 256, 125, 115]
            assert scoped['original_candidate_count'] - 11 == scoped['candidate_counts'][0]
        else:
            roots = raw['same_roots']['roots']
            exposure = Counter()
            for root in roots:
                exposure.update(root['training_exposure_counts'])
            scoped.update(root_count=len(roots), root_ids=[r['root_id'] for r in roots],
                          mixed_training_exposure=True, training_exposure_counts=dict(exposure))
            assert scoped['candidate_counts'] == [104, 53, 26, 25]
        out[scope] = scoped
    out['sources'] = sources
    out['comparability'] = compare_current(comparison)
    return out


def compare_current(comparison):
    paths = {
        'current_identity': DELIVERY / 'run/model_identity.json',
        'current_protocol': DELIVERY / 'run/protocol.json',
        'current_selection': DELIVERY / 'prepared/SELECTION_REVIEW.json',
        'current_candidates': DELIVERY / 'prepared/candidate_manifest.json',
        'current_prompts': DELIVERY / 'prepared/prompt_manifest.json',
        'first_round_train_identity': Path(comparison['sources']['train']['path']).parent / 'model_identity_report.json',
        'first_round_test_identity': Path(comparison['sources']['test_identity']['path']),
        'first_round_protocol': Path(comparison['sources']['old_protocol']['path']),
    }
    data, sources = {}, {}
    for name, path in paths.items():
        expected = None
        if name == 'first_round_test_identity':
            expected = comparison['sources']['test_identity']['sha256']
        elif name == 'first_round_protocol':
            expected = comparison['sources']['old_protocol']['sha256']
        data[name], sources[name] = read_source(path, expected)
    current, train, test = [data[k] for k in ['current_identity', 'first_round_train_identity', 'first_round_test_identity']]
    protocol, old = data['current_protocol'], data['first_round_protocol']
    assert train['solver_contract_sha256'] == sources['first_round_protocol']['sha256']
    assert test['solver_contract_sha256'] == sources['first_round_protocol']['sha256']
    identities = {}
    for model in MODELS:
        recorded = current['verified_checkpoints'][model]
        model_meta = current['models'][model]
        model_record = model_meta['base_model_identity' if model == 'M0' else 'merged_model_identity']
        assert recorded == model_record
        if model == 'M0':
            comparisons = {
                'matches_train_base_model_identity': recorded == train['base_model_identity'],
                'matches_first_round_solver_checkpoint_identity': recorded == old['checkpoint_identity'],
                'matches_test_recorded_base_for_clean': recorded == test['models']['M_clean_policy']['base_model_identity'],
                'matches_test_recorded_base_for_inclusive': recorded == test['models']['M_inclusive_policy']['base_model_identity'],
            }
        else:
            comparisons = {}
            for label, historical in [('train', train), ('test', test)]:
                for field in ['merged_model_path', 'merged_model_identity', 'base_model_identity',
                              'adapter_config_sha256', 'adapter_state_sha256', 'adapter_composite_sha256']:
                    comparisons[f'matches_{label}_{field}'] = model_meta[field] == historical['models'][model][field]
        assert all(comparisons.values()), f'checkpoint_metadata_mismatch:{model}'
        identities[model] = {
            'same_checkpoint_identity': True, 'checkpoint_path': recorded['path'],
            'recorded_file_count': len(recorded['files']),
            'recorded_weight_shard_count': sum(f['name'].endswith('.safetensors') for f in recorded['files']),
            'comparisons': comparisons,
        }
    positions = defaultdict(set)
    for prompt in data['current_prompts']:
        positions[(prompt['candidate_id'], prompt['context_mode'])].add(prompt['variant_id'])
    for candidate in data['current_candidates']:
        for context in CONTEXTS:
            assert len(positions[(candidate['candidate_id'], context)]) == old['position_counts'][candidate['question_type']]
    for question_type in TYPES:
        cfg = protocol['generation_config_by_type'][question_type]
        assert cfg['temperature'] == old['temperatures'][question_type]
        for key in ['top_p', 'max_tokens', 'vllm_version']:
            assert cfg[key] == old['generation_config'][key]
        assert cfg['enable_thinking'] == old['enable_thinking']
        if question_type == 'single_choice':
            assert cfg['structured_output_api'] is None and not old['generation_config']['single_choice_structured_output']
        else:
            for key in ['structured_output_api', 'structured_output_version']:
                assert cfg[key] == old['generation_config'][key]
            assert cfg['stop'] == old['generation_config']['structured_stop']
            assert cfg['include_stop_str_in_output'] == old['generation_config']['structured_include_stop_str_in_output']
    assert protocol['rollouts'] == old['rollouts_per_position'] == 3
    assert protocol['max_model_len'] == old['max_model_len']
    assert protocol['tokenizer'] == old['tokenizer']
    selection = data['current_selection']
    return {
        'status': 'SAME_CHECKPOINTS_DIFFERENT_QUESTION_SET_AND_CONTEXT_DISPLAY_PROTOCOL',
        'identity_verification_scope': 'Compared recorded paths, per-file sizes and SHA-256 values in identity metadata; model weights were not read or rehashed by this extraction.',
        'current_identity_scope': current['identity_scope'],
        'adapter_runtime_note': current['adapter_identity'],
        'same_checkpoint_identity_by_model': identities,
        'current_scope': {
            'root_count': len(selection['by_root']), 'candidate_count': selection['candidate_count'],
            'by_question_type': selection['by_type'], 'selection': selection['selection'],
            'ordering_review': selection['ordering_review'],
            'requests_per_model_both_contexts': protocol['request_limit_per_model'],
        },
        'protocol_same_verified': {
            'rollouts_per_position': 3, 'position_counts': old['position_counts'],
            'temperatures': old['temperatures'], 'top_p': old['generation_config']['top_p'],
            'max_tokens': old['generation_config']['max_tokens'], 'max_model_len': old['max_model_len'],
            'tokenizer': old['tokenizer'], 'enable_thinking': old['enable_thinking'],
            'vllm_version': old['generation_config']['vllm_version'],
            'single_choice_structured_output': False,
            'multi_select_and_ordering_structured_output': old['generation_config']['structured_output_api'],
            'seed_recipe': old['seed_schedule'],
            'seed_comparability_note': 'The root_id/candidate_id/slot_id recipe is retained, but different candidate IDs imply different actual seeds across question sets.',
            'scoring': 'Exact agreement after mapping displayed labels to canonical IDs; multi-select uses exact sets and ordering uses exact sequences.',
        },
        'protocol_major_differences': [
            {'field': 'ordering_display',
             'first_round': old['permutation_contract_version'],
             'current': protocol['permutation_contract_version'],
             'meaning': 'First round retains canonical event IDs while cycling display order. Current ordering uses deterministic balanced event-to-display-label rebinding independent of Gold, then maps predictions back to canonical IDs.'},
            {'field': 'full_context_rendering',
             'first_round': 'Shared Solver template with [turn_id] speaker: text dialogue rows.',
             'current': protocol['context_policy'],
             'meaning': 'Current full context uses frozen conversation-date headers and preserves UNKNOWN; the no-context template is retained.'},
            {'field': 'question_set_and_review',
             'first_round': 'Official train 2294, test 496 after 11 ordering exclusions, or the historical same-three-roots subset of 104.',
             'current': f"Latest Writer coverage: {selection['candidate_count']} routed candidates; {selection['by_type']}; ordering review {selection['ordering_review']}.",
             'meaning': 'Questions, topic/type composition and admission/review history differ; these are not paired versions of the same questions.'},
        ],
        'limitations': [
            'Differences between current scores and first-round scores cannot be attributed directly to new training gains: the compared checkpoint identities are unchanged while the questions and context/display protocol differ.',
            'Train 2294 and same-three-roots 104 mix training-exposure groups; the same-three-roots subset belongs to train and is not an independent held-out test.',
            'Accuracy denotes agreement with the recorded canonical Gold. Historical semantic certification of every question and Gold was not performed.',
            'Position/rollout observations are repeated responses to the same questions, not independent unique-question samples.',
            'Overall accuracy is rollout-weighted: single-choice and ordering have 12 responses per question/context, while multi-select has 15.',
            'Only the previous official scopes were extracted. The old comparison current=9 results, deltas and stale current-specific limitations are excluded.',
        ],
        'sources': sources,
    }


if __name__ == '__main__':
    result = extract()
    target = DELIVERY / 'BASELINE.json'
    payload = (json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    target.write_bytes(payload)
    for path in [target, Path(__file__).resolve()]:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_name(path.name + '.sha256').write_text(f'{digest}  {path.name}\n')
    print(json.dumps({'path': str(target), 'sha256': hashlib.sha256(payload).hexdigest(),
                      'baseline_scopes': list(result)[4:7],
                      'same_checkpoint_identity': {m: v['same_checkpoint_identity'] for m, v in result['comparability']['same_checkpoint_identity_by_model'].items()}}, indent=2))
