"""Compile one explicit formal previous round into the existing QA interface.

No latest-directory discovery, cumulative test history, transport, or model calls.
The current transcript source and offline replay fixture are separate from the
previous formal question/evaluation sources.
"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
PROJECT = Path('/root/autodl-tmp/gxy-selfplay')
WORKTREE = PROJECT / 'worktrees/h4-gate-downstream-v1'
sys.path.insert(0, str(WORKTREE))
from scripts.current_method.restart_n5_spell.core import canonical_json, sha256_bytes, sha256_file
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1 import qa_difficulty_feedback_run as runner

VERSION = 'qa_single_previous_formal_round_feedback_v1'
QUESTION_RUN = 'h4_unexpanded122_qa_generation_only_v1'
FEEDBACK_RUN = 'h4_round1_feedback_compiler_v1_live_1'
EVALUATION_RUN = 'h4_round0_trained_solver_train_reeval_v1_live_1'
SHARDS = dict(zip(runner.ROOTS, ('shard-39', 'shard-58', 'shard-02')))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def artifact(path):
    p = Path(path).resolve()
    return {'path': str(p), 'sha256': sha256_file(p)}


def verify_artifact(value):
    require(sha256_file(Path(value['path'])) == value['sha256'], 'previous_formal_source_drift')


def validate_scope(lock):
    runs = PROJECT / 'restart-n5-runs'
    compiler_run = runs / FEEDBACK_RUN
    expected_sources = {
        'formal_feedback_card': compiler_run / 'root_feedback_cards.jsonl',
        'feedback_seed_manifest': compiler_run / 'feedback_seed_manifest.jsonl',
        'formal_preflight': compiler_run / 'preflight_report.json',
    }
    expected_evaluations = {
        'candidate_transition_matrix': runs / EVALUATION_RUN / 'candidate_transition_matrix.jsonl',
        'candidate_training_exposure': runs / EVALUATION_RUN / 'candidate_training_exposure_manifest.jsonl',
        'clean_candidate_results': runs / EVALUATION_RUN / 'clean_train/candidate_results.jsonl',
        'inclusive_candidate_results': runs / EVALUATION_RUN / 'inclusive_train/candidate_results.jsonl',
        'difficulty_manifest': runs / 'h4_round0_b0_k3_difficulty_v1_live_1/difficulty_manifest.jsonl',
        'confirmed_invalid_blacklist': runs / 'h4_round0_b0_matched_training_v1_live_2/confirmed_invalid_blacklist.json',
    }
    require(lock['root_order'] == list(runner.ROOTS), 'formal_root_scope_mismatch')
    require(lock['previous_question_run_id'] == QUESTION_RUN
            and lock['previous_feedback_run_id'] == FEEDBACK_RUN
            and lock['previous_evaluation_run_id'] == EVALUATION_RUN,
            'single_explicit_formal_round_required')
    require(lock['automatic_latest_run_selection'] is False
            and lock['merge_other_question_runs'] is False
            and lock['same_round_tests_are_previous_history'] is False,
            'cumulative_test_history_forbidden')
    require([r['root_id'] for r in lock['roots']] == list(runner.ROOTS), 'formal_root_records_mismatch')
    require(Path(lock['current_transcript_source_run']).resolve()
            == PROJECT / 'offline-deliveries/h4_gate_downstream_v1/prepared_run',
            'current_transcript_source_mismatch')
    for row in lock['roots']:
        expected = PROJECT / 'restart-n5-runs' / QUESTION_RUN / 'shards' / SHARDS[row['root_id']] / 'roots' / row['root_id'] / 'candidate_pool.json'
        require(Path(row['question_pool']['path']).resolve() == expected, 'question_pool_outside_selected_formal_round')
        require(Path(row['formal_feedback_wrapper']['path']).resolve()
                == PROJECT / 'offline-deliveries/h4_planner_recovery_more_roots_v1_inputs' / row['root_id'] / 'feedback_input.json',
                'formal_feedback_wrapper_source_mismatch')
        for field in ('question_pool', 'formal_feedback_wrapper'):
            verify_artifact(row[field])
    for field in ('formal_feedback_card', 'feedback_seed_manifest', 'formal_preflight'):
        require(Path(lock[field]['path']).resolve() == expected_sources[field], 'formal_compiler_source_mismatch')
        verify_artifact(lock[field])
    require(set(lock['evaluation_sources']) == set(expected_evaluations), 'formal_evaluation_source_keys_mismatch')
    for field, record in lock['evaluation_sources'].items():
        require(Path(record['path']).resolve() == expected_evaluations[field], 'formal_evaluation_source_mismatch')
        verify_artifact(record)
    return lock


def question_fields(row):
    return {k: deepcopy(row.get(k, {})) for k in ('question_type', 'question', 'options', 'items')}


def fold_key(row):
    q = question_fields(row)
    return canonical_json([q['question_type'], q['question'], sorted(q['options'].values()), sorted(q['items'].values())])


def main():
    require(not (BASE / 'COMPILED_FEEDBACK.json').exists() and not (BASE / 'feedback').exists(),
            'compiled_feedback_exists_no_overwrite')
    lock = validate_scope(read(BASE / 'previous_round_sources.json'))
    observations_path = BASE / 'formal_feedback_observations.json'
    observations = read(observations_path)
    expected_observation_sources = {k: v for k, v in lock['evaluation_sources'].items()
                                    if k != 'confirmed_invalid_blacklist'}
    expected_observation_sources.update(root_feedback_cards=lock['formal_feedback_card'],
                                        feedback_seed_manifest=lock['feedback_seed_manifest'])
    require(observations['sources'] == expected_observation_sources, 'formal_observation_sources_mismatch')
    by_root = {r['root_id']: r for r in observations['roots']}
    require(set(by_root) == set(runner.ROOTS), 'formal_observations_root_mismatch')
    rules = (BASE / 'writer_constraints.txt').read_text()
    source_run = Path(lock['current_transcript_source_run'])
    pending, records = {}, []

    def output(path, value, text=False):
        require(path not in pending and not path.exists(), 'output_already_exists')
        content = value if text else json.dumps(value, ensure_ascii=False, indent=2) + '\n'
        pending[path] = content
        return {'path': str(path), 'sha256': sha256_bytes(content.encode())}

    for row in lock['roots']:
        root = row['root_id']
        pool = read(row['question_pool']['path'])
        require({k: by_root[root]['candidate_pool'][k] for k in ('path', 'sha256')} == row['question_pool'],
                'formal_observation_pool_mismatch')
        require(pool and all(q['root_id'] == root for q in pool), 'formal_pool_wrong_root')
        require(len({q['candidate_id'] for q in pool}) == len(pool), 'formal_pool_duplicate_id')
        require(row['question_pool']['sha256'] == row['pool_sha256_in_legacy_card_field'], 'legacy_card_pool_identity_mismatch')
        seed = next(q for q in pool if q['candidate_id'] == row['seed_candidate_id'])
        require(sha256_bytes(canonical_json(seed).encode()) == row['seed_candidate_sha256'], 'formal_seed_identity_mismatch')
        require({q['candidate_id'] for q in by_root[root]['candidates']} == {q['candidate_id'] for q in pool},
                'formal_evaluation_candidate_scope_mismatch')
        folded = {}
        for q in pool:
            key = fold_key(q)
            if key not in folded:
                folded[key] = {**question_fields(q), 'previous_candidate_ids': []}
            folded[key]['previous_candidate_ids'].append(q['candidate_id'])
        index = {'root_id': root, 'previous_question_run_id': QUESTION_RUN,
                 'purpose': 'ONLY_THE_SELECTED_PREVIOUS_FORMAL_ROUND_IS_THE_HISTORICAL_EXCLUSION_SCOPE',
                 'claims_status': 'Question and option claims are unverified historical outputs, not transcript evidence or certified answers.',
                 'entries': list(folded.values())}
        # Aggregate observed Gold matches, never recent-test diagnoses or scores.
        formal_feedback = {'root_id': root, 'previous_evaluation_run_id': EVALUATION_RUN,
                           'model_evaluation_sources': {'M0': 'h4_round0_b0_k3_difficulty_v1_live_1',
                                                        'M_clean_policy': EVALUATION_RUN + '/clean_train',
                                                        'M_inclusive_policy': EVALUATION_RUN + '/inclusive_train'},
                           'scope': 'Historical original-Gold agreement on this formal previous-round candidate pool only; no new measurement or semantic certification.',
                           'training_exposure_counts': deepcopy(by_root[root]['training_exposure_counts']),
                           'exposure_note': 'These aggregates mix training-exposed and unexposed questions; they are not held-out generalization scores.',
                           'aggregate_by_model': deepcopy(by_root[root]['aggregate_by_model']),
                           'aggregate_by_type_and_model': deepcopy(by_root[root]['aggregate_by_type_and_model']),
                           'action_instructions': deepcopy(by_root[root]['imperative_feedback']),
                           'use': 'Use instructions to choose a different supported judgment. Do not treat a low historical score as proof of a valid difficult question or copy its Gold. Keep model identities and evaluation conditions separate.'}
        shared = ('\n[PREVIOUS_FORMAL_ROUND_FEEDBACK]\n' + canonical_json(formal_feedback)
                  + '\n[/PREVIOUS_FORMAL_ROUND_FEEDBACK]\n\n[PREVIOUS_FORMAL_ROUND_QUESTIONS]\n'
                  + canonical_json(index) + '\n[/PREVIOUS_FORMAL_ROUND_QUESTIONS]\n')
        miner = ('Use the complete supplied transcript as the only factual source. Select exactly three distinct supported reasoning intents. '
                 'Do not restate the tested judgments in PREVIOUS_FORMAL_ROUND_QUESTIONS or use multiple intent labels for one judgment. '
                 'That list contains the single selected previous formal round; do not extend it with other trials. '
                 'An old event may support a different state, relation or comparison; the same topic alone is not repetition. '
                 'Do not invent temporal order or facts to satisfy the intent contract. Preserve the current output schema.\n')
        context = runner.validate_context({'intent_suffix': miner + shared, 'slot_suffixes': [rules + shared] * 3})
        qa_path = source_run / 'qa/roots' / root / 'qa_source.json'
        qa = read(qa_path)
        directory = BASE / 'feedback' / root
        projection = {'version': VERSION, 'root_id': root, 'previous_formal_question_index': index,
                      'previous_formal_feedback': formal_feedback, 'transcript_complete_sha256': qa['complete_dialogue_sha256'],
                      'source_question_count': len(pool), 'folded_question_count': len(folded),
                      'recent_test_question_count': 0, 'recent_test_score_count': 0,
                      'general_rule_origin': 'Existing generic constraints retained with corrected previous-round scope; no test-specific examples, results, or questions appended.'}
        context_record = output(directory / 'qa_feedback_context.json', context)
        projection_record = output(directory / 'prompt_projection.json', projection)
        preserved_record = output(directory / 'formal_feedback_preserved.json',
                                  Path(row['formal_feedback_wrapper']['path']).read_text(), text=True)
        output(directory / 'miner_feedback.txt', context['intent_suffix'], text=True)
        output(directory / 'writer_feedback.txt', context['slot_suffixes'][0], text=True)
        sources = {'compiler': artifact(__file__), 'source_lock': artifact(BASE / 'previous_round_sources.json'),
                   'writer_rules': artifact(BASE / 'writer_constraints.txt'),
                   'formal_observations': artifact(observations_path), 'formal_question_pool': row['question_pool'],
                   'formal_feedback_wrapper': row['formal_feedback_wrapper'], 'formal_feedback_card': lock['formal_feedback_card'],
                   'formal_seed_manifest': lock['feedback_seed_manifest'], 'formal_preflight': lock['formal_preflight'],
                   'qa_source': artifact(qa_path), **lock['evaluation_sources']}
        records.append({'root_id': root, 'sources': sources, 'context': context_record, 'projection': projection_record,
                        'preserved_correction': preserved_record,
                        'context_object_sha256': runner.digest(context), 'source_question_count': len(pool),
                        'source_question_types': dict(Counter(q['question_type'] for q in pool)),
                        'folded_question_count': len(folded), 'recent_test_questions_in_feedback': 0,
                        'writer_suffix_chars': len(context['slot_suffixes'][0]),
                        'intent_suffix_chars': len(context['intent_suffix'])})
    result = {'version': VERSION, 'status': 'CORRECTED_OFFLINE_NOT_EXECUTED',
              'root_order': list(runner.ROOTS), 'external_calls': 0, 'records': records,
              'previous_question_run_id': QUESTION_RUN, 'previous_feedback_run_id': FEEDBACK_RUN,
              'previous_evaluation_run_id': EVALUATION_RUN, 'automatic_next_experiment': False,
              'planner_automatically_activated': False,
              'total_previous_formal_questions': sum(r['source_question_count'] for r in records),
              'total_folded_questions': sum(r['folded_question_count'] for r in records)}
    output(BASE / 'COMPILED_FEEDBACK.json', result)
    validate_scope(lock)
    for path, content in pending.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            stream.write(content)
    runner._compiled_feedback_files(BASE / 'feedback', source_run)
    print(json.dumps({'status': result['status'], 'formal_questions': result['total_previous_formal_questions'],
                      'folded_questions': result['total_folded_questions'], 'external_calls': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
