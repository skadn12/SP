"""Seal v98 only after independent accounting and final report review pass."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_rows(rows, expected_count):
    assert len(rows) == expected_count, ('binding_count', len(rows), expected_count)
    for row in rows:
        assert sha(row['path']) == row['sha256'], 'changed:' + row['path']


def verify_nested_bindings(value):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            assert sha(value['path']) == value['sha256'], 'stale_review:' + value['path']
        for item in value.values():
            verify_nested_bindings(item)
    elif isinstance(value, list):
        for item in value:
            verify_nested_bindings(item)


def main():
    destination = BASE / 'COMPLETE_MANIFEST.json'
    assert not destination.exists(), 'already_sealed'
    for name in ('COMPLETE_RESULTS_ZH.md', 'RESULTS.json', 'NEXT_MINFIX_PLAN.md',
                 'evidence/ACTUAL_MINFIX_OBSERVATIONS.json',
                 'evidence/ACTUAL_PLANNER_IDENTITY_PROMPT_OBSERVATIONS.json',
                 'evidence/INDEPENDENT_FINAL_RESULT_AUDIT.json',
                 'evidence/FINAL_REPORT_SECOND_READER.json'):
        assert (BASE / name).is_file(), 'incomplete:' + name
    for name in ('evidence/INDEPENDENT_FINAL_RESULT_AUDIT.json',
                 'evidence/FINAL_REPORT_SECOND_READER.json'):
        review = read(BASE / name)
        assert str(review['status']).startswith('PASS'), 'review_not_passed:' + name
        assert not review.get('failures'), 'review_failures:' + name
        verify_nested_bindings(review)

    protected = read(BASE / 'PRESERVATION_BEFORE.json')['files']
    verify_rows(protected, 2481)
    verify_rows(read(BASE / 'evidence/EXPANSION_MINFIX_RECEIPT.json')['files'], 36)
    verify_rows(read(BASE / 'evidence/ORDERING_MINFIX_RECEIPT.json')['files'], 15)
    verify_rows(read(BASE / 'PRECALL_PLAN.json')['frozen_inputs'], 10)

    results = read(BASE / 'RESULTS.json')
    verify_nested_bindings(results)
    totals = results['totals']
    assert totals['assigned_roots'] == totals['structurally_accepted_cards'] == 4
    assert totals['planner_requests'] == totals['planner_identity_omissions'] == 8
    assert totals['logical_requests'] == totals['http_attempts'] == totals['http_200'] == 16
    for name in ('model_accepted_scenes', 'model_accepted_new_turns',
                 'raw_generated_candidates', 'raw_generated_turns_across_all_attempts',
                 'independent_ready_roots', 'independent_ready_new_turns',
                 'qa_http_calls', 'question_calls', 'solver_calls',
                 'new_multiselect_questions', 'fresh_gold6_controls', 'new_ordering_questions',
                 'content_usable_questions', 'new_selected_candidates', 'training_jobs'):
        assert totals[name] == 0, 'nonzero:' + name
    assert all(count == 0 for count in results['gold_distribution'].values())
    assert all(count == 0 for count in totals['solver_calls_per_model'].values())
    assert results['solver_accuracy'] is None and results['difficulty_comparison'] is None
    assert not results['training_allowed'] and not results['formal_training_admission']
    assert not results['semantic_reliability_certified'] and not results['stable_production_ready']
    assert len(results['roots']) == 4
    for row in results['roots']:
        assert row['independent_status'] == 'NO_ACCEPTED_NEW_SOURCE'
        assert row['ready_for_joint_qa'] is False
        assert row['materialized_source'] is None and row['ready_new_turn_count'] == 0
    ready = read(BASE / 'evidence/READY_SOURCE_DECISIONS.json')
    assert ready['ready_roots'] == [] and ready['ready_new_turns'] == 0
    skip = read(BASE / 'evidence/QA_SKIPPED_NO_READY_SOURCE.json')
    assert skip['status'] == 'SKIPPED' and skip['qa_http_calls'] == skip['solver_calls'] == 0
    assert read(BASE / 'evidence/ACTUAL_QA_INPUT_AUDIT.json')['status'] == 'SKIPPED_NO_READY_SOURCE'
    assert not (BASE / 'shards').exists() and not (BASE / 'qa_engine/runs').exists()
    assert not list((BASE / 'sources').glob('*_SOURCE.json'))

    files = []
    for path in sorted(BASE.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        assert path != destination
        files.append({'path': str(path.resolve()), 'sha256': sha(path),
                      'bytes': path.stat().st_size})
    record = {
        'version': 'v98_complete_manifest_v1', 'status': 'SEALED',
        'sealed_at_utc': datetime.now(timezone.utc).isoformat(),
        'listed_file_count_excluding_this_manifest': len(files),
        'protected_file_count_including_this_manifest': len(files) + 1,
        'self_excluded': True, 'excluded': ['__pycache__/**', '*.pyc'],
        'files': files, 'previous_delivery_protected_files': len(protected),
        'precall_bound_expansion_files': 36, 'precall_bound_qa_files': 15,
        'precall_bound_inputs': 10,
        'assigned_roots': 4, 'logical_requests': 16, 'http_attempts': 16,
        'planner_identity_omissions': 8, 'raw_generated_candidates': 0,
        'model_accepted_scenes': 0, 'model_accepted_new_turns': 0,
        'independent_ready_roots': 0, 'independent_ready_new_turns': 0,
        'qa_http_calls': 0, 'solver_calls': 0, 'new_selected_candidates': 0,
        'training_jobs': 0, 'semantic_generation_reliability_certified': False,
        'reason_for_qa_skip': 'No realized body or independently READY newly expanded source.',
        'next_scope': 'Unify route-specific Planner schema and precise final retry feedback in a new isolated version.'}
    with destination.open('x') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    verify_rows(files, len(files))
    print(json.dumps({'status': 'SEALED_AND_HASH_VERIFIED',
                      'listed_files': len(files), 'protected_files_including_manifest': len(files) + 1,
                      'manifest_sha256': sha(destination)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
