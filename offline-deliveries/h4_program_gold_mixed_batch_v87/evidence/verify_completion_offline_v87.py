"""Test completion in temporary synthetic batches; never close the actual v87 run."""
import ast
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

HOME = Path(__file__).resolve().parents[1]
HELPER = HOME / 'evidence/complete_batch.py'
spec = importlib.util.spec_from_file_location('v87_completion_offline', HELPER)
completion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(completion)
checks = []


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def rejects(label, callback, reason):
    try:
        callback()
    except ValueError as exc:
        assert reason in str(exc), (label, str(exc))
    else:
        raise AssertionError(label + ': accepted invalid fixture')
    checks.append(label)


for path in (HELPER, HOME / 'evidence/write_result_details.py', Path(__file__)):
    ast.parse(path.read_text(), filename=str(path))
checks.append('Both result helpers and this fixture parse without running result generation')
protected = completion.verify_preserved(HOME)
assert len(protected) == 10226
checks.append('Actual 10226 v74-v86 protected files and exact directory inventories remain unchanged')
old = completion.read(completion.PREVIOUS / 'CUMULATIVE_AUDITED_NEW_CANDIDATES.json')
oldtypes, oldgolds = completion.inventory_stats(old, 'candidate_count')
assert dict(oldtypes) == {'multi_select': 36, 'ordering': 6}
assert dict(oldgolds) == {'2': 9, '3': 7, '4': 9, '5': 11}
checks.append('Actual v86 cumulative has42 questions:36 multiselect/6 ordering; Gold2/3/4/5=9/7/9/11')

with tempfile.TemporaryDirectory(prefix='v87_completion_offline_') as directory:
    parent = Path(directory)
    home = parent / 'synthetic_v87'
    evidence = home / 'evidence'
    evidence.mkdir(parents=True)
    previous = parent / 'synthetic_v86'
    previous.mkdir()
    completion.PREVIOUS = previous
    write(previous / 'CUMULATIVE_AUDITED_NEW_CANDIDATES.json', old)
    protected_fixture = {}
    for index in range(10225):
        path = previous / f'protected-{index}.txt'
        path.write_text(f'synthetic preserved artifact {index}\n')
        protected_fixture[str(path.resolve())] = completion.bind(path)['sha256']
    oldpath = previous / 'CUMULATIVE_AUDITED_NEW_CANDIDATES.json'
    protected_fixture[str(oldpath.resolve())] = completion.bind(oldpath)['sha256']
    write(home / 'PRESERVATION_BEFORE.json', {'files': protected_fixture})
    roots = sorted(completion.EXPECTED_ROOTS)
    cfg = {'multiselect_config': {'slots': [{'root_id': root, 'split': 'train'} for root in roots],
                                 'new_child_target_distribution': {str(g): 4 for g in range(2, 6)}},
           'ordering_slots': [{'root_id': root, 'candidate_count': 2} for root in roots],
           'max_candidates': 28, 'max_generation_http': 12, 'max_solver_calls_per_model': 456,
           'training_allowed': False}
    for key, filename in [('ordering_anchor_plan', 'SCENE_PLANS.json'), ('character_registry', 'CHARACTER_REGISTRY.json'), ('actor_reference_policy', 'ACTOR_REFERENCE_POLICY.json')]:
        write(home / filename, {'synthetic_fixture_only': True})
        cfg[key] = completion.bind(home / filename)
    write(home / 'CONFIG.json', cfg)
    # Absence diagnostics have their own schema; real SHA bindings never accept null.
    absent = str((home.parent / 'h4_ready_roots_mixed_batch_v78/multiselect/assets' /
                  'root103866_family/rounds/initial/PUBLIC_ASSETS.json').resolve())
    cfg['multiselect_config']['slots'][0]['slot_id'] = 'root103866_family'
    for index, slot in enumerate(cfg['multiselect_config']['slots'][1:], 1):
        slot['slot_id'] = 'fixture_family_' + str(index)
    write(home / 'CONFIG.json', cfg)
    diagnostic = {'GENERATION_AUDIT': {'repeated_per_fact_replacement_texts': [{
        'root_id': roots[0], 'comparison_status': 'not_available',
        'old_public_assets': {'diagnostic_path': absent, 'availability': 'not_available'},
        'identical_per_fact_replacements': None}]}}
    completion.verify_exact_bindings(diagnostic, home)
    completion.verify_exact_bindings(completion.bind(home / 'CONFIG.json'), home)
    checks.append('Explicit absent comparison diagnostic and real SHA binding each validate')
    rejects('Real SHA null is always rejected',
            lambda: completion.verify_exact_bindings({'path': absent, 'sha256': None}, home),
            'invalid_sha256_binding')
    rejects('Diagnostic at wrong location is rejected',
            lambda: completion.verify_exact_bindings(diagnostic['GENERATION_AUDIT']['repeated_per_fact_replacement_texts'][0]['old_public_assets'], home),
            'unavailable_comparison_outside_diagnostic_location')
    wrong_diagnostic = deepcopy(diagnostic)
    wrong_diagnostic['GENERATION_AUDIT']['repeated_per_fact_replacement_texts'][0]['old_public_assets']['diagnostic_path'] = '/tmp/unexpected-old-assets.json'
    rejects('Diagnostic at wrong historical path is rejected',
            lambda: completion.verify_exact_bindings(wrong_diagnostic, home),
            'invalid_unavailable_comparison_diagnostic')
    wrong_diagnostic = deepcopy(diagnostic)
    wrong_diagnostic['GENERATION_AUDIT']['repeated_per_fact_replacement_texts'][0]['comparison_status'] = 'compared'
    rejects('Diagnostic with false comparison status is rejected',
            lambda: completion.verify_exact_bindings(wrong_diagnostic, home),
            'invalid_unavailable_comparison_diagnostic')
    wrong_binding = completion.bind(home / 'CONFIG.json')
    wrong_binding['sha256'] = 'malformed'
    rejects('Malformed real SHA is rejected',
            lambda: completion.verify_exact_bindings(wrong_binding, home),
            'invalid_sha256_binding')
    wrong_binding = completion.bind(home / 'CONFIG.json')
    wrong_binding['sha256'] = '0' * 64
    rejects('Conflicting real bindings are rejected',
            lambda: completion.verify_exact_bindings([completion.bind(home / 'CONFIG.json'), wrong_binding], home),
            'conflicting_artifact_binding')
    (home / 'fixture_production.py').write_text('# Synthetic immutable production file.\n')
    write(home / 'BATCH_PREPARATION.COMPLETE.json', {'status': 'PREPARED', 'errors': [],
          'files': [completion.bind(p) for p in sorted(home.glob('*')) if p.is_file()]})
    admitted = []
    for root in roots:
        source = home / 'sources' / (root + '_SOURCE.json')
        review = home / 'sources' / (root + '_REVIEW.json')
        write(source, {'synthetic_fixture_only': True})
        write(review, {'synthetic_fixture_only': True})
        admitted.append({'root_id': root, 'turn_count': 1, 'source': completion.bind(source), 'review': completion.bind(review)})
    write(home / 'SOURCE_ADMISSION_SUMMARY.json', {'ready_roots': roots, 'held_roots': [], 'roots': admitted})
    write(home / 'SOURCE_ADMISSION.COMPLETE.json', {'status': 'FROZEN', 'synthetic_fixture_only': True})

    def question(index, kind, gold):
        return {'candidate_id': 'synthetic-' + str(index), 'root_id': roots[index % len(roots)],
                'question_type': kind, 'gold': list('ABCDEF')[:gold] if kind == 'multi_select' else ['I' + str(i) for i in range(1, 7)]}

    selected = [question(0, 'multi_select', 2), question(1, 'multi_select', 3), question(2, 'ordering', 6)]
    control = question(3, 'multi_select', 6)
    rejected = question(4, 'multi_select', 4)
    pool = selected + [control]

    def inventory(questions, count_key):
        types = dict(Counter(q['question_type'] for q in questions))
        golds = {str(g): sum(q['question_type'] == 'multi_select' and len(q['gold']) == g for q in questions) for g in range(2, 6)}
        return {'questions': questions, count_key: len(questions), 'type_distribution': types,
                'gold_distribution': golds, 'multiselect_gold_distribution': golds,
                'training_allowed': False, 'training_started': False, 'formal_training_admission': False, 'merged_into_baseline': False}

    new = inventory(selected, 'new_question_count')
    cumulative = inventory(old['questions'] + selected, 'candidate_count')
    write(home / 'AUDITED_NEW_CANDIDATES.json', new)
    write(home / 'CUMULATIVE_AUDITED_NEW_CANDIDATES.json', cumulative)
    write(home / 'EVALUATION_POOL.json', {'questions': pool})
    write(home / 'PAIRED_CALLS.json', {'calls': [], 'synthetic_fixture_only': True})
    report = {'status': 'PASS', 'evaluation_status': 'COMPLETED', 'selected_new_candidate_count': 3,
              'calls_per_model': 66, 'training_started': False, 'formal_training_admission': False,
              'by_type': {'multi_select': {'mechanically_assembled': 4, 'content_excluded': 1},
                          'ordering': {'mechanically_assembled': 1}}}
    write(home / 'MIXED_EVALUATION_REPORT.json', report)
    write(home / 'multiselect/ALLOCATION.json', {'assembled_distribution': {'2': 1, '3': 1, '4': 1},
          'assembled_child_count': 3, 'control_count': 1})
    multi_rows = [{'candidate_id': q['candidate_id'], 'root_id': q['root_id'],
                   'core_status': 'CORE_USABLE' if q != rejected else 'CORE_UNUSABLE',
                   'reason': 'Synthetic fixture only.'} for q in [selected[0], selected[1], control, rejected]]
    write(home / 'multiselect/FINAL_CONTENT_REVIEW.json', {'questions': multi_rows})
    usable_ids = [q['candidate_id'] for q in multi_rows if q['core_status'] == 'CORE_USABLE']
    write(home / 'multiselect/FINAL_CONTENT_REVIEW.COMPLETE.json', {'status': 'FROZEN', 'before_any_solver': True,
          'solver_scores_used': False, 'candidate_count': 4, 'usable_count': 3,
          'usable_candidate_ids': usable_ids, 'statuses': {q['candidate_id']: q['core_status'] for q in multi_rows},
          'review': completion.bind(home / 'multiselect/FINAL_CONTENT_REVIEW.json')})
    write(home / 'BATCH_REVIEW.COMPLETE.json', {'status': 'FROZEN', 'before_any_solver': True, 'solver_scores_used': False})
    write(home / 'ordering/CONTENT_REVIEW.json', {'accepted_count': 1, 'rejected_count': 0})
    write(home / 'ordering/SOURCE_JUDGMENTS.json', {'questions': [{'candidate_id': selected[2]['candidate_id'],
          'items': {'I' + str(i): {'actor_ok': True} for i in range(1, 7)}}]})

    def metric(calls):
        return {'calls': calls, 'exact_correct': calls // 2, 'exact_percent': 50.0 if calls else None,
                'partial_mean': '1/2' if calls else None}

    roles = {'new_multiselect': 36, 'new_ordering': 12, 'new_candidates': 48, 'historical_control': 18, 'all_questions': 66}
    details = {'new_selected_count': 3, 'content_usable_new_by_multi_gold': {'2': 1, '3': 1},
               'metrics': {side: {key: metric(value) for key, value in roles.items()} for side in ('base', 'trained')},
               'questions': [{**q, 'role': 'historical_control' if q == control else 'new_ordering' if q['question_type'] == 'ordering' else 'new_child',
                              'models': {side: {'selected_by_policy': True} for side in ('base', 'trained')}} for q in pool],
               'prior_comparisons': {'fixture': {'new_questions_by_type': {kind: {'question_count': 1, 'models': {side: metric(12) for side in ('base', 'trained')}} for kind in ('multi_select', 'ordering')}}},
               'ordering_generation_failures': [], 'ordering_content_failures': [],
               'multiselect_unfilled_slots': [], 'multiselect_asset_round_reviews': [],
               'multiselect_content_failures': [multi_rows[-1]]}
    write(evidence / 'RESULT_DETAILS.json', details)
    for name in completion.AUDIT_NAMES:
        audit = {'status': 'PASS', 'failed_checks': [], 'checks': ['synthetic fixture check'], 'check_count': 1,
                 'totals': {'actual_generation_http_calls': 8}, 'expected_calls_per_model': 66,
                 'previous_cumulative_count': 42, 'cumulative_count': 45, 'current_new_count': 3, 'selected_new_count': 3}
        write(evidence / completion.AUDIT_FILES[name], audit)
    for side in ('base', 'trained'):
        for filename in ('manifest.json', 'RESULTS.json', 'COMPLETE.json', 'solver_identity.json', 'responses/call0.json', 'prompts/call0.json'):
            write(home / 'solver' / side / filename, {'synthetic_fixture_only': True})
    state = completion.load_state(home)
    assert state['cumgolds']['6'] == 0 and state['cumtypes']['ordering'] == 7
    checks.append('Full synthetic load validates all dependencies,42+3 cumulative,66 calls and typed Gold accounting')
    pure = deepcopy(state)
    pure['planned_ordering'] = pure['review']['accepted_count']
    assert completion.has_generation_or_content_shortfall(pure)
    pure['details']['multiselect_content_failures'] = []
    assert not completion.has_generation_or_content_shortfall(pure)
    checks.append('Multiselect whole-question rejection alone produces content-shortfall status')
    wrong = deepcopy(new)
    wrong['gold_distribution']['6'] = 1
    rejects('Ordering six items cannot enter multiselect Gold counts', lambda: completion.inventory_stats(wrong, 'new_question_count'), 'export_multiselect_gold_distribution_mismatch')
    with_control = inventory(selected + [control], 'new_question_count')
    with_control['gold_distribution']['6'] = 1
    with_control['multiselect_gold_distribution']['6'] = 1
    rejects('Historical Gold6 controls cannot enter new exports', lambda: completion.inventory_stats(with_control, 'new_question_count'), 'historical_control_in_new_export')
    bad_details = deepcopy(details)
    bad_details['multiselect_content_failures'] = []
    write(evidence / 'RESULT_DETAILS.json', bad_details)
    rejects('Missing multiselect content rejection is caught', lambda: completion.load_state(home), 'multiselect_content_failure_detail_mismatch')
    write(evidence / 'RESULT_DETAILS.json', details)
    completion.finalize(home)
    record = completion.read(evidence / 'RUN_COMPLETE.json')
    assert record['new_selected_count'] == 3 and record['cumulative_candidate_count'] == 45
    assert record['source_root_count'] == 4 and record['independent_test_sources'] is False
    assert record['status'] == 'COMPLETED_WITH_GENERATION_OR_CONTENT_SHORTFALLS'
    assert record['multiselect_content_rejected_count'] == 1
    assert record['training_started'] is False
    index = (evidence / 'ARTIFACT_INDEX.md').read_text()
    assert 'FINAL_CONTENT_REVIEW.json' in index and 'solver/base/responses' in index
    text = (evidence / 'RESULTS.md').read_text()
    assert '多选整题内容拒绝' in text and 'Synthetic fixture only.' in text
    checks.append('Synthetic finalization renders actual fixture counts, content failures and real artifact naming')
    completion.verify_completion(home)
    rejects('Completed output cannot be overwritten', lambda: completion.finalize(home), 'closeout_output_already_exists')
    bound = evidence / 'RESULTS.md'
    original = bound.read_bytes()
    bound.write_bytes(original + b'changed\n')
    rejects('Changed bound artifact is rejected', lambda: completion.verify_completion(home), 'frozen_file_changed')
    bound.write_bytes(original)
    extra = evidence / 'unexpected.txt'
    extra.write_text('unexpected file\n')
    rejects('Additional batch artifact after closeout is rejected', lambda: completion.verify_completion(home), 'completion_file_coverage_changed')
    extra.unlink()
    bound.unlink()
    rejects('Missing bound artifact is rejected', lambda: completion.verify_completion(home), 'completion_file_coverage_changed')
    bound.write_bytes(original)
    previous_extra = previous / 'unexpected.txt'
    previous_extra.write_text('unexpected old-directory file\n')
    rejects('Additional previous-directory artifact is rejected', lambda: completion.verify_completion(home), 'protected_directory_added_or_removed_file')
    previous_extra.unlink()
    restored = completion.verify_completion(home)
    assert restored['status'] == 'PASS'
    checks.append('Restored synthetic fixture replays complete file coverage and all preservation hashes')

assert not list(HOME.rglob('__pycache__'))
record = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS',
          'scope': 'AST checks, read-only actual v74-v86 preservation, and explicit synthetic completion fixtures in a temporary directory. No real v87 completion, audit result, source certification, generation, evaluation or training is fabricated.',
          'checks': checks, 'check_count': len(checks), 'new_model_calls': 0, 'new_http_calls': 0,
          'real_batch_finalized': False, 'production_root_files_modified': False,
          'files': [completion.bind(p) for p in (HELPER, HOME / 'evidence/write_result_details.py', Path(__file__))]}
completion.write(HOME / 'evidence/COMPLETION_HELPER_OFFLINE_VALIDATION_final.json', record)
print({'status': 'PASS', 'check_count': len(checks), 'real_batch_finalized': False})
