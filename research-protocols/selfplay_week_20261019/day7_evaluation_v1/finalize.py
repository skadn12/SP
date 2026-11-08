"""Seal completed Day7 evidence and update the mutable Day7 navigation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
ROLES = ('S0', 'A1', 'B1')


def read(path):
    return json.loads(path.read_text())


def binding(path):
    path = path.resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': digest.hexdigest()}


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    preflight = read(HERE / 'INDEPENDENT_PREFLIGHT.json')
    audit = read(HERE / 'POSTEXECUTION_AUDIT.json')
    paired = read(HERE / 'PAIRED_AGGREGATE.json')
    costs = read(HERE / 'EXECUTION_COSTS.json')
    terminals = read(HERE / 'ENDPOINT_TERMINALS.json')
    assert preflight['status'] == audit['status'] == 'PASS'
    assert paired['status'] == 'COMPLETE'
    assert audit['verified_request_endpoint_pairs'] == costs['total_returned_requests'] == 4338
    assert audit['verified_question_endpoint_pairs'] == 267
    assert terminals['exit_codes'] == dict.fromkeys(ROLES, 0)
    assert terminals['all_roles_raw_sealed'] is True
    assert audit['all_raw_sealed_before_first_scoring'] is True
    assert audit['third_scoring_path_matches_native_requests_and_questions'] is True
    assert paired['independent_recalculation']['status'] == 'PASS'

    for row in preflight['audited_files'] + audit['audit_output_bindings']:
        assert binding(Path(row['path'])) == row, row['path']
    for row in costs['bindings']:
        assert binding(Path(row['path'])) == row, row['path']
    for role in ROLES:
        aggregate = read(HERE / 'models' / role / 'AGGREGATE.json')
        assert aggregate['status'] == 'COMPLETE'
        assert aggregate['denominator_questions'] == 89
        assert aggregate['denominator_requests'] == 1446
        assert aggregate['unknown_usage_requests'] == aggregate['not_submitted_requests'] == 0
    main_key = 'historical115_TEST/HISTORICAL_REGRESSION_PRIMARY/ALL'
    main_difference = paired['comparisons']['B1_minus_A1'][main_key]
    evidence_names = ('TEST_REQUEST_FREEZE.json', 'INDEPENDENT_PREFLIGHT.json',
                      'POSTEXECUTION_AUDIT.json', 'PAIRED_AGGREGATE.json',
                      'REPORT_ZH.md', 'CLOSEOUT_ZH.md', 'EXECUTION_COSTS.json',
                      'REPRODUCTION_ZH.md', 'WEEK_EVIDENCE_INDEX_ZH.md')
    decision = {
        'schema_version': 'day7_audited_single_stage_closeout_v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'D7_EXECUTED_AUDITED_SINGLE_STAGE_CLOSEOUT',
        'execution_complete': True, 'execution_and_scoring_audit': 'PASS',
        'new_TEST_requests': 4338, 'returned_requests': 4338,
        'automatic_retries': 0, 'new_training_runs': 0,
        'questions_per_model_inventory_only': 89,
        'primary_questions': 64, 'primary_historical_roots': 14,
        'primary_main_estimator': 'question_equal_exact',
        'primary_scores': {role: paired['models'][role]['groups'][main_key]['question_equal_exact']
                           for role in ROLES},
        'B1_minus_A1_question_equal_difference': main_difference['question_equal_exact_difference'],
        'B1_minus_A1_root_equal_difference': main_difference['root_equal_exact_difference'],
        'B1_minus_A1_root_descriptive_uncertainty': main_difference['paired_root_descriptive_uncertainty'],
        'feedback_extra_benefit_supported_by_primary_TEST': False,
        'causal_feedback_benefit_established': False,
        'completed_training_stages': 1,
        'second_wave_formal_B_GO': False,
        'missing_models': {'A2': 'NOT_PRODUCED', 'B2': 'NOT_PRODUCED'},
        'old_Round2': 'HISTORICAL_COMPARATOR_NOT_NEWLY_EVALUATED_ON_THIS_CONTRACT',
        'training_pool_full_A1_B1_fit_evaluation': 'NOT_EXECUTED_ONLY_SEPARATE24_QUESTION_PANEL',
        'memory_benefit_established': False, 'fact_memory_adopted': False,
        'test_used_for_checkpoint_selection': False,
        'test_released_to_challenger': False,
        'new_root_generalization_established': False,
        'all_reporting_cohorts_historically_exposed': True,
        'cohort_root_overlap': True, 'single_training_seed': True,
        'full_two_iteration_selfplay_completed': False,
        'content_review_identity': 'INTERNAL_AI_REVIEW_SHARED_ENVIRONMENT_NOT_EXTERNAL_HUMAN_BLIND_REVIEW',
        'end_to_end_generation_monetary_cost_complete': False,
        'new_archive_created': False, 'model_weights_copied': False,
        'evidence': [binding(HERE / name) for name in evidence_names],
        'next_research_action': 'Separate prospective training-only source/citation reliability and cross-root/family reproduction; re-freeze budget and gates before any new feedback production. Do not consume TEST outcomes as Challenger feedback.',
    }
    save(HERE / 'DECISION.json', decision)

    latest = WEEK / 'DAY7_LATEST.json'
    snapshot = HERE / 'PRE_CLOSEOUT_DAY7_NAVIGATION.json'
    with snapshot.open('xb') as stream:
        stream.write(latest.read_bytes())

    paths = sorted(path for path in HERE.rglob('*') if path.is_file()
                   and '__pycache__' not in path.parts
                   and path.name != 'FINAL_MANIFEST.json')
    manifest = {
        'schema_version': 'day7_final_file_manifest_v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': decision['status'], 'scope': str(HERE),
        'excluded': ['FINAL_MANIFEST.json (self)', '__pycache__ (derived cache)',
                     'external model weights (existing model manifests bind them)'],
        'files': [binding(path) for path in paths],
        'external_evidence_closure': {
            'test_inputs': binding(HERE / 'TEST_REQUEST_FREEZE.json'),
            'independent_preflight': binding(HERE / 'INDEPENDENT_PREFLIGHT.json'),
            'postexecution_full_weight_and_file_recheck': binding(HERE / 'POSTEXECUTION_AUDIT.json'),
            'model_manifests': [binding(HERE / 'models' / role / 'EXECUTION_MANIFEST.json')
                                for role in ROLES],
        },
        'archive_created': False, 'base_weights_duplicated': False,
    }
    manifest['file_count'] = len(manifest['files'])
    manifest['total_local_file_bytes'] = sum(row['bytes'] for row in manifest['files'])
    save(HERE / 'FINAL_MANIFEST.json', manifest)
    for row in manifest['files']:
        assert binding(Path(row['path'])) == row, row['path']

    navigation = {
        'schema_version': 'day7_latest_navigation_v3',
        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': decision['status'], 'package': str(HERE),
        'previous_navigation': binding(snapshot),
        'decision': binding(HERE / 'DECISION.json'),
        'report': binding(HERE / 'REPORT_ZH.md'),
        'closeout': binding(HERE / 'CLOSEOUT_ZH.md'),
        'manifest': binding(HERE / 'FINAL_MANIFEST.json'),
        'reproduction': binding(HERE / 'REPRODUCTION_ZH.md'),
        'postexecution_audit': binding(HERE / 'POSTEXECUTION_AUDIT.json'),
        'new_solver_requests': 4338, 'new_training_runs': 0,
        'feedback_causal_benefit_established': False,
        'completed_training_stages': 1,
        'A2_B2': 'NOT_PRODUCED',
        'challenger_release': False,
        'next_action': decision['next_research_action'],
    }
    latest.write_text(json.dumps(navigation, ensure_ascii=False, indent=2) + '\n')
    assert read(latest)['manifest'] == binding(HERE / 'FINAL_MANIFEST.json')
    print(json.dumps({'status': decision['status'], 'manifest_files': manifest['file_count'],
                      'manifest_local_bytes': manifest['total_local_file_bytes'],
                      'decision': binding(HERE / 'DECISION.json'),
                      'manifest': binding(HERE / 'FINAL_MANIFEST.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
