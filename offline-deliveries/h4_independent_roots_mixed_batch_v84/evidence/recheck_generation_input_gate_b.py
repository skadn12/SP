"""Independent post-freeze consistency review; no production imports or calls."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def bind(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def dt(value):
    result = datetime.fromisoformat(value)
    if result.utcoffset() is None:
        raise ValueError('naive_timestamp')
    return result


def stamp(path):
    stat = Path(path).stat()
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'ctime_ns': stat.st_ctime_ns,
            'inode': stat.st_ino, 'device': stat.st_dev}


def main():
    evidence = HOME / 'evidence'
    output = evidence / 'GATE_B_GENERATION_INPUT_REVIEW.json'
    if output.exists():
        raise SystemExit('review_exists_refuse_overwrite')
    if (evidence / 'RUN_COMPLETE.json').exists():
        raise SystemExit('no_new_artifacts_after_final_completion')
    files = set(); checks = []

    def load(path):
        files.add(Path(path).resolve())
        return read(path)

    def check(name, condition, detail=None):
        row = {'check': name, 'status': 'PASS' if condition else 'FAIL'}
        if detail is not None:
            row['detail'] = detail
        checks.append(row)

    def verify_rows(name, rows):
        mismatched = [r['path'] for r in rows if bind(r['path']) != r]
        check(name, not mismatched, mismatched)
        files.update(Path(r['path']).resolve() for r in rows)

    config = load(HOME / 'CONFIG.json')
    generation = load(evidence / 'GENERATION_AUDIT.json')
    inputs = load(evidence / 'INPUT_AUDIT.json')
    diagnosis = load(evidence / 'ORDERING_SHORTFALL_DIAGNOSIS.json')
    for audit, name, allowed_status in ((generation, 'generation', ('PASS', 'PASS_WITH_GENERATION_AND_CONTENT_SHORTFALLS')),
                                       (inputs, 'inputs', ('PASS',)), (diagnosis, 'ordering_diagnosis', ('PASS',))):
        check(name + '_all_declared_checks_pass', audit['status'] in allowed_status
              and audit['failed_checks'] == [] and audit['check_count'] == len(audit['checks'])
              and all(row['status'] == 'PASS' for row in audit['checks']))
    verify_rows('generation_auditor_unchanged', [generation['auditor']])
    verify_rows('input_audit_all_bound_artifacts_unchanged', inputs['files'])
    verify_rows('ordering_diagnosis_all_bound_inputs_unchanged', diagnosis['inputs'])
    check('input_audit_preexecution_independent_contract', inputs['phase'] == 'inputs_only'
          and inputs['independent_implementation'] is True
          and inputs['native_or_mixed_score_functions_imported'] is False
          and inputs['new_model_or_http_calls'] == 0 and inputs['training_performed'] is False
          and inputs['frozen_artifacts_modified'] is False)
    check('generation_review_without_solver_outputs_or_calls', generation['disclosures']['audit_generation_http_calls'] == 0
          and generation['disclosures']['audit_solver_calls'] == 0
          and generation['disclosures']['current_solver_outputs_read'] is False
          and generation['disclosures']['frozen_code_modified'] is False)
    weights = inputs['checkpoint_verification']
    check('formal_input_checkpoint_full_byte_hash_recorded', bool(weights)
          and all(row['verification'] == 'full_sha256_recomputed_from_file_bytes'
                  and row['stat_before'] == row['stat_after'] for row in weights))
    check('checkpoint_stat_unchanged_since_full_input_hash', all(stamp(row['path']) == row['stat_after'] for row in weights))

    old = load(HOME / 'PRESERVATION_BEFORE.json')['files']
    old_mismatches = [path for path, digest in old.items() if bind(path)['sha256'] != digest]
    protected_roots = {HOME.parent / Path(path).relative_to(HOME.parent).parts[0] for path in old}
    present = {str(path.resolve()) for root in protected_roots for path in root.rglob('*') if path.is_file()}
    check('old_5797_file_sha_values_unchanged', len(old) == 5797 and not old_mismatches, old_mismatches)
    check('old_protected_directory_inventory_unchanged', present == set(old),
          {'added_count': len(present - set(old)), 'missing_count': len(set(old) - present)})

    frozen = load(HOME / 'BATCH_REVIEW.COMPLETE.json')
    preparation = load(HOME / 'EVALUATION_PREPARATION.COMPLETE.json')
    verify_rows('batch_review_frozen_artifacts_unchanged', frozen['files'])
    verify_rows('evaluation_preparation_frozen_artifacts_unchanged', preparation['files'])
    check('freeze_before_evaluation_preparation', frozen['status'] == 'FROZEN'
          and frozen['before_any_solver'] is True and frozen['solver_scores_used'] is False
          and frozen['training_allowed'] is False
          and dt(frozen['completed_at_utc']) <= dt(preparation['prepared_at_utc']))
    production = load(HOME / 'BATCH_PREPARATION.COMPLETE.json')
    production_bindings = {r['path']: r['sha256'] for r in production['files']}
    current_production = [*HOME.glob('*.py'), *HOME.glob('*.md'), HOME / 'CONFIG.json']
    check('all_frozen_current_production_code_and_config_unchanged', all(
        production_bindings.get(str(p.resolve())) == bind(p)['sha256'] for p in current_production))
    for name, field in (('SCENE_PLANS.json', 'ordering_anchor_plan'), ('CHARACTER_REGISTRY.json', 'character_registry'),
                        ('ACTOR_REFERENCE_POLICY.json', 'actor_reference_policy')):
        check('frozen_policy:' + name, bind(HOME / name) == config[field])
        files.add(HOME / name)

    assembled = load(HOME / 'multiselect/POOL.json')['questions']
    review = load(HOME / 'multiselect/FINAL_CONTENT_REVIEW.json')
    multi_complete = load(HOME / 'multiselect/FINAL_CONTENT_REVIEW.COMPLETE.json')
    groups = config['multiselect_config']['final_review_groups']
    check('actual_four_final_groups_registered', groups == ['a', 'b', 'c', 'd'])
    directory = HOME / 'multiselect/final_review'
    check('exact_four_final_group_file_sets', set(directory.glob('*_SOURCE_JUDGMENTS.json')) == {
        directory / f'group_{g}_SOURCE_JUDGMENTS.json' for g in groups}
        and set(directory.glob('*_AUDIT.json')) == {directory / f'group_{g}_AUDIT.json' for g in groups})
    covered = []; group_counts = {}; private_times = []
    expected_judgment_bindings = [bind(directory / f'group_{g}_SOURCE_JUDGMENTS.json') for g in groups]
    for group in groups:
        judgment_path = directory / f'group_{group}_SOURCE_JUDGMENTS.json'
        judgments = load(judgment_path); audit = load(directory / f'group_{group}_AUDIT.json')
        check('group_disclosure_and_binding:' + group, judgments['group'] == audit['group'] == group
              and audit['source_judgments_sha256'] == bind(judgment_path)['sha256']
              and judgments['child_gold_read_before_save'] is False
              and judgments['current_pool_read_before_save'] is False
              and judgments['current_solver_scores_read'] is False
              and judgments['solver_scores_used'] is False
              and judgments['parent_all_true_provenance_known'] is True
              and judgments['parent_gold_known_before_review'] is True
              and judgments['fresh_blind_parent_review'] is False
              and audit['current_solver_scores_read'] is False and audit['solver_scores_used'] is False
              and audit['source_first_for_current_children'] is True
              and audit['source_judgments_bytes_changed_after_private_read'] is False
              and audit['question_or_gold_edits'] == [])
        saved_bindings = audit['both_source_judgments_saved_bindings']
        check('all_four_judgments_saved_before_private_read:' + group,
              {r['path']: r['sha256'] for r in saved_bindings} == {r['path']: r['sha256'] for r in expected_judgment_bindings}
              and all(dt(load(Path(r['path']))['created_at_utc']) <= dt(audit['first_private_pool_read_at_utc'])
                      for r in saved_bindings)
              and dt(audit['first_private_pool_read_at_utc']) <= dt(audit['created_at_utc'])
              <= dt(multi_complete['completed_at_utc']))
        covered.extend(q['candidate_id'] for q in judgments['questions'])
        group_counts[group] = len(judgments['questions'])
        private_times.append(audit['first_private_pool_read_at_utc'])
    assembled_ids = {q['candidate_id'] for q in assembled}
    check('four_groups_exactly_cover_all_ten_assembled_questions', len(covered) == len(set(covered)) == len(assembled) == 10
          and set(covered) == assembled_ids)
    statuses = {q['candidate_id']: q['core_status'] for q in review['questions']}
    usable_multi = {qid for qid, status in statuses.items() if status == 'CORE_USABLE'}
    rejected_multi = set(statuses) - usable_multi
    check('one_actual_multiselect_content_rejection_not_hidden', rejected_multi == {
        'cc-train-episode-112812-v84-parent-from-reviewed-six'} and len(usable_multi) == 9)
    check('actual_review_statuses_match_frozen_complete', multi_complete['statuses'] == statuses
          and set(multi_complete['usable_candidate_ids']) == usable_multi
          and multi_complete['candidate_count'] == 10 and multi_complete['usable_count'] == 9)
    verify_rows('all_multiselect_frozen_review_records_unchanged', multi_complete['review_records'])
    for path in expected_judgment_bindings:
        check('source_group_in_frozen_review_records:' + Path(path['path']).name, path in multi_complete['review_records'])

    pool = load(HOME / 'EVALUATION_POOL.json')['questions']
    ordering_pool = load(HOME / 'ordering/EVALUATION_POOL.json')['questions']
    check('actual_pool_matches_frozen_multiselect_usable_subset_plus_ordering', pool == [q for q in assembled
        if q['candidate_id'] in usable_multi] + ordering_pool)
    controls = {s['parent_candidate_id'] for s in config['multiselect_config']['slots']}
    counts = {'new_multiselect': sum(q['question_type'] == 'multi_select' and q['candidate_id'] not in controls for q in pool),
              'historical_controls': sum(q['candidate_id'] in controls for q in pool),
              'new_ordering': sum(q['question_type'] == 'ordering' for q in pool)}
    calls = sum(18 if q['question_type'] == 'multi_select' else 12 for q in pool)
    golds = {str(g): sum(q['question_type'] == 'multi_select' and len(q['gold']) == g for q in pool) for g in range(2, 7)}
    check('actual_ten_candidates_six_new_multi_three_controls_one_ordering', len(pool) == 10
          and counts == {'new_multiselect': 6, 'historical_controls': 3, 'new_ordering': 1})
    check('actual_174_calls_per_model_and_gold_distribution', calls == 174 == preparation['calls_per_model']
          == inputs['expected_calls_per_model'] and inputs['candidate_count'] == len(pool)
          and inputs['ordering_candidate_count'] == counts['new_ordering']
          and inputs['content_gold_distribution'] == golds == {'2': 1, '3': 1, '4': 2, '5': 2, '6': 3})
    check('maximum_budgets_not_targets', len(pool) <= config['max_candidates']
          and calls <= config['max_solver_calls_per_model']
          and generation['totals']['actual_generation_http_calls'] <= config['max_generation_http'])
    expected_totals = {'actual_generation_http_calls': 10, 'initial_batch_snapshot_http_calls': 8,
                       'explicit_bounded_supplement_http_calls': 2, 'new_multiselect_six_option_combinations': 6,
                       'historical_controls': 4, 'multiselect_assembled': 10, 'multiselect_content_usable': 9,
                       'ordering_slot_count': 8, 'ordering_mechanical_accepted': 1, 'ordering_content_usable': 1,
                       'ordering_content_rejected': 0, 'protected_previous_file_count': 5797}
    check('generation_totals_match_actual_admitted_and_rejected_records', generation['totals'] == expected_totals)
    check('ordering_diagnosis_counts_match_generation_audit', diagnosis['actual_counts'] == {
        'planned_drafts': 8, 'returned_drafts': 8, 'http_attempts': 4, 'mechanically_accepted': 1,
        'source_content_usable': 1, 'formally_rejected': 7, 'mechanical_and_content_yield': .125})
    rejection_rows = [row for slot in config['ordering_slots'] for row in
                      load(HOME / 'ordering' / slot['slot_id'] / 'REJECTIONS.json')]
    check('diagnosis_preserves_exact_seven_original_formal_reasons', {r['candidate_id']: r['reasons'] for r in rejection_rows}
          == {r['candidate_id']: r['original_formal_reasons'] for r in diagnosis['drafts'] if r['original_formal_reasons']})
    check('generation_shortfall_status_is_honest', generation['status'] == 'PASS_WITH_GENERATION_AND_CONTENT_SHORTFALLS'
          and len(rejection_rows) == 7 and len(rejected_multi) == 1)
    manifests = {side: load(HOME / 'solver' / side / 'manifest.json') for side in ('trained', 'base')}
    strip = lambda value: {k: v for k, v in value.items() if k not in ('prepared_at_utc', 'model', 'solver_model', 'solver_identity')}
    check('paired_actual_manifest_contract_identical', strip(manifests['trained']) == strip(manifests['base'])
          and all(m['candidate_count'] == 10 and m['request_count'] == len(m['requests']) == 174 for m in manifests.values()))
    trained_stage = HOME / 'TRAINED_EVALUATION.STARTED.json'
    if trained_stage.exists():
        start = load(trained_stage)
        check('input_audit_completed_before_actual_solver_stage', dt(inputs['completed_at_utc']) < dt(start['time']))
    check('no_batch_bytecode_cache', not list(HOME.rglob('__pycache__')))
    failed = [row for row in checks if row['status'] == 'FAIL']
    record = {'version': 'v84_gate_b_generation_input_consistency_review_v1',
              'created_at_utc': datetime.now(timezone.utc).isoformat(), 'reviewer': 'v84_source_gate_b',
              'status': 'FAIL' if failed else 'PASS', 'check_count': len(checks), 'checks': checks, 'failed_checks': failed,
              'actual_pool_counts': counts, 'actual_calls_per_model': calls, 'actual_multiselect_gold_distribution': golds,
              'actual_final_review_group_counts': group_counts, 'generation_totals': generation['totals'],
              'protected_previous_file_count': len(old),
              'checkpoint_review': 'Formal INPUT_AUDIT recomputed full byte SHA values for all 27 checkpoint files. This independent consistency review verifies their recorded hashes/unchanged stat identities and does not claim a second full-weight hash.',
              'limits': ['This is record and frozen-input consistency review, not a second independent semantic certification of all answers.',
                         'The four roots are historical training roots; this review does not establish held-out accuracy or stable large-batch generation yield.'],
              'new_model_or_http_calls': 0, 'source_or_production_edits': 0, 'current_solver_outputs_read': False,
              'inputs': [bind(path) for path in sorted(files | {Path(__file__).resolve()})]}
    if failed:
        print(json.dumps({'status': 'FAIL', 'failed_checks': failed}, ensure_ascii=False))
        raise SystemExit(1)
    with output.open('x') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps({'status': 'PASS', 'path': str(output), 'check_count': len(checks),
                      'actual_pool_counts': counts, 'actual_calls_per_model': calls}, ensure_ascii=False))


if __name__ == '__main__':
    main()
