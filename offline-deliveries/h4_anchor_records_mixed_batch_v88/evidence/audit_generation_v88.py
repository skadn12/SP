"""Independent generation/assembly/dedup evidence audit; never invoke a model.

Production validators are replayed, then request receipts and public whole-set
composition are reconstructed separately. This is not blind semantic review.
"""
from collections import Counter
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import hashlib
import json
import os
import re
import sys
import traceback

HOME = Path(__file__).resolve().parents[1]
OUT = HOME / 'evidence/GENERATION_AUDIT.json'
OPENED = set()
CHECKS = []
ROOTS = tuple('cc-train-episode-' + r for r in ('103866', '109164', '112812', '135466'))
sys.dont_write_bytecode = True
sys.path.insert(0, str(HOME))


def guard(event, args):
    if event.startswith('socket.') or event in {'subprocess.Popen', 'os.system', 'os.posix_spawn'}:
        raise RuntimeError('audit_forbids_external:' + event)
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1:3]
        writing = ((isinstance(mode, str) and any(c in mode for c in 'wax+'))
                   or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
        if writing and path not in {OUT, Path('/dev/null')} and not path.is_relative_to(Path('/tmp')):
            raise RuntimeError('audit_forbids_write:' + str(path))
        if path.is_relative_to(HOME):
            if path.is_relative_to(HOME / 'solver'):
                raise RuntimeError('audit_forbids_solver:' + str(path))
            if not writing:
                OPENED.add(str(path))


sys.addaudithook(guard)
import common
import ordering_generation as ordering
import ordering_review
a = common.load_multi('asset_generation')
c = common.load_multi('compose')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def require(condition, name, detail=None):
    record = {'check': name, 'status': 'PASS' if condition else 'FAIL'}
    if detail is not None:
        record['detail'] = detail
    CHECKS.append(record)
    if not condition:
        raise AssertionError(name)


def check_binding(row):
    require(sha(row['path']) == row['sha256'], 'binding:' + row['path'])


def normalized_set(q):
    field = 'options' if q['question_type'] == 'multi_select' else 'items'
    return (q['root_id'], q['question_type'], tuple(sorted(' '.join(t.split()).casefold() for t in q[field].values())))


PRONOUNS = ('he','him','his','himself','she','her','hers','herself')


def independent_name_errors(items):
    return ['ordering_actor_reference_policy_forbidden_pronoun:' + label + ':' + word
            for label, text in items.items()
            for word in sorted(set(re.findall(r'\b(?:he|him|his|himself|she|her|hers|herself)\b', text.casefold())))]


def independent_ordering_prior_context(prior):
    """Reconstruct the public index; full six-item bans remain once in TASK."""
    return {
        'whole_set_text_location': 'DISALLOWED PRIOR WHOLE SETS in the task tail; every distinct six-item set is listed there once.',
        'question_index': [{key: question[key] for key in
            ('candidate_id', 'root_id', 'question_type')}
            for question in prior],
    }


def independent_parent_family_usable(public, judgment):
    """Unknown or invalid saved parent facts cannot form reliable children."""
    expected = {r['fact_id']: r['proposition'] for r in public['parent_facts']}
    parents = judgment['parents']
    return (judgment['stem_valid'] is True and len(parents) == len(expected) == 6
        and {r['fact_id'] for r in parents} == set(expected)
        and all(r['proposition'] == expected[r['fact_id']] and r['relation'] == 'supports'
            and r['scope_valid'] is True and r['fact_distinct'] is True for r in parents))


def independent_scope(q, source):
    """Recreate the old documented public scene boundary, without Gold input."""
    scenes = list(dict.fromkeys(t['turn_ref'].rsplit(':', 1)[0] for t in source['turns']))
    match = re.search(r'\bAPPEAR IN THE SUPPLIED DIALOGUE (?P<restriction>within the (?P<scenes>.+?) scenes?\.?)\s*$', q['question'], re.I)
    selected, fragment = [], None
    if match:
        entries = re.split(r'\s*,\s*(?:and\s+)?|\s+and\s+', match.group('scenes'))
        if entries and all(entry in scenes for entry in entries):
            selected = [scene for scene in scenes if scene in entries]
            fragment = match.group('restriction')
    return {'contract': 'explicit_public_stem_scene_scope_v52', 'candidate_id': q['candidate_id'],
            'kind': 'explicit_scene_union' if selected else 'full_dialogue',
            'public_stem_restriction': fragment, 'scene_refs': selected,
            'matching_turn_refs': [t['turn_ref'] for t in source['turns'] if not selected or t['turn_ref'].rsplit(':', 1)[0] in selected],
            'context_turn_refs': [t['turn_ref'] for t in source['turns']]}



def registry_contract_errors(registry, slots, sources):
    """Independently bind the new registry to exact full sources and unknowns."""
    errors = []
    root_ids = [s['root_id'] for s in slots]
    bindings = registry.get('source_bindings', [])
    characters = registry.get('characters', [])
    if len(root_ids) != len(set(root_ids)) or set(root_ids) != set(ROOTS):
        errors.append('selected_four_root_coverage')
    if len(bindings) != len(root_ids) or {r.get('root_id') for r in bindings} != set(root_ids):
        errors.append('registry_source_root_coverage')
    expected_people = set()
    for slot in slots:
        root = slot['root_id']; source = sources[root]['source']
        names = sorted({t['speaker'] for t in source['turns']})
        expected_people.update((root, name) for name in names)
        rows = [r for r in bindings if r.get('root_id') == root]
        if len(rows) != 1:
            continue
        row = rows[0]
        if not (row.get('path') == slot['source_path'] == sources[root]['path']
                and row.get('sha256') == slot['source_sha256'] == sources[root]['sha256']
                and row.get('full_turn_count') == len(source['turns'])
                and row.get('speaker_names') == names and row.get('all_turns_reviewed') is True):
            errors.append('registry_source_binding:' + root)
    actual_people = [(r.get('root_id'), r.get('canonical_name')) for r in characters]
    if len(actual_people) != len(set(actual_people)) or set(actual_people) != expected_people:
        errors.append('registry_exact_source_speaker_coverage')
    for row in characters:
        label = str(row.get('root_id')) + ':' + str(row.get('canonical_name'))
        for field in ('gender', 'third_person_pronouns'):
            value = row.get(field, {})
            if not (value.get('value', 'MISSING') is None
                    and value.get('status') == 'unknown_no_explicit_source_evidence'
                    and value.get('evidence') == []):
                errors.append('registry_unknown_field:' + label + ':' + field)
        reference = row.get('permitted_question_reference', {})
        if not (reference.get('kind') == 'canonical_name'
                and reference.get('value') == row.get('canonical_name')
                and reference.get('allow_possessive_name') is True
                and reference.get('inferred_gendered_pronouns_allowed') is False):
            errors.append('registry_permitted_name_reference:' + label)
    policy = registry.get('policy', {})
    if not (policy.get('source_unchanged') is True and policy.get('registry_is_planning_aid') is True
            and policy.get('registry_is_extra_QA_evidence') is False
            and policy.get('solver_context_unchanged') is True):
        errors.append('registry_no_extra_qa_facts_policy')
    return errors


def v88_parent_judgments_supported(judgment, public):
    """Validate actual v88 item schema; optional summaries never override rows."""
    if not isinstance(judgment, dict) or judgment.get('version') != 'v88_parent_source_judgments_v1':
        return False
    if 'content_status' in judgment and judgment['content_status'] != 'CORE_USABLE':
        return False
    options = public.get('options')
    rows = judgment.get('judgments')
    if not isinstance(options, dict) or set(options) != set('ABCDEF') or not isinstance(rows, list) or len(rows) != 6:
        return False
    if not all(isinstance(row, dict) for row in rows):
        return False
    labels = [row.get('option') for row in rows]
    if len(set(labels)) != 6 or set(labels) != set(options):
        return False
    for row in rows:
        evidence = row.get('required_evidence')
        if not (row.get('proposition') == options[row['option']]
                and row.get('source_judgment') == 'supported' and row.get('entailed') is True
                and isinstance(row.get('reason'), str) and bool(row['reason'].strip())
                and isinstance(evidence, list) and bool(evidence)
                and all(isinstance(item, dict) and all(isinstance(item.get(key), str) and bool(item[key].strip())
                    for key in ('turn_ref', 'speaker', 'quote')) for item in evidence)):
            return False
    if 'coverage' in judgment:
        coverage = judgment['coverage']
        if not isinstance(coverage, dict):
            return False
        declared = coverage.get('options_reviewed')
        if not isinstance(declared, list) or len(declared) != 6 or set(declared) != set(options):
            return False
        for key, expected in {'option_count': 6, 'supported_count': 6, 'contradicted_count': 0, 'unknown_count': 0}.items():
            if type(coverage.get(key)) is not int or coverage[key] != expected:
                return False
    return True


def validate_admission_contract(config, home, read_fn, bind_fn, check_fn):
    """Reconstruct READY and fresh current-parent bindings without reusing Gold."""
    admission = read_fn(home / 'SOURCE_ADMISSION.COMPLETE.json')
    summary = read_fn(home / 'SOURCE_ADMISSION_SUMMARY.json')
    consistency = config['source_consistency']
    check_fn(consistency['upstream_complete'] == bind_fn(home / 'SOURCE_ADMISSION.COMPLETE.json'),
             'source_admission_config_binding')
    check_fn(admission['status'] == 'FROZEN' and admission['training_allowed'] is False
             and admission['ready_roots'] == summary['ready_roots']
             and admission['held_roots'] == summary['held_roots'], 'source_admission_frozen_summary')
    for row in admission['files']:
        check_fn(bind_fn(row['path']) == row, 'source_admission_file_binding:' + row['path'])
    rows = summary['roots']; by_root = {r['root_id']: r for r in rows}
    check_fn(len(rows) == len(ROOTS) and set(by_root) == set(ROOTS)
             and set(summary['ready_roots']) == set(ROOTS) and summary['held_roots'] == [],
             'all_four_current_sources_ready')
    check_fn(consistency['selected_roots'] == list(ROOTS)
             and consistency['source_review_is_question_answer_certification'] is False
             and set(summary['historical_held_roots_unchanged']) <= set(consistency['excluded_held_roots']),
             'source_ready_selection_preserves_historical_holds')
    candidate_rows = summary.get('candidate_reviews', [])
    check_fn(summary.get('candidate_held_roots') == ['cc-train-episode-210046']
             and len(candidate_rows) == 1 and candidate_rows[0]['root_id'] == 'cc-train-episode-210046'
             and candidate_rows[0]['decision'] == 'HOLD'
             and set(summary['candidate_held_roots']) <= set(consistency['excluded_held_roots'])
             and not (set(summary['candidate_held_roots']) & set(ROOTS)),
             'historical_candidate_hold_reused_and_excluded_from_selected_sources')
    for candidate in candidate_rows:
        candidate_source = read_fn(candidate['source']['path'])
        candidate_review = read_fn(candidate['review']['path'])
        check_fn(bind_fn(candidate['source']['path']) == candidate['source']
                 and bind_fn(candidate['review']['path']) == candidate['review']
                 and candidate['source'] in admission['files'] and candidate['review'] in admission['files']
                 and candidate_review['source'] == candidate['source']
                 and candidate_review['decision'] == 'HOLD' and candidate_review['full_source_read'] is True
                 and candidate_review['turn_refs_read'] == [t['turn_ref'] for t in candidate_source['turns']]
                 and candidate_review['current_question_gold_read'] is False
                 and candidate_review['current_solver_scores_read'] is False
                 and any(f['severity'] == 'BLOCKING' for f in candidate_review['findings']),
                 'historical_candidate_full_source_hold_record_rebound_before_generation:' + candidate['root_id'])
    slots = config['multiselect_config']['slots']
    check_fn(tuple(s['root_id'] for s in slots) == ROOTS
             and tuple(s['root_id'] for s in config['ordering_slots']) == ROOTS,
             'four_root_both_lane_config_coverage')
    preparation_started = read_fn(home / 'BATCH_PREPARATION.STARTED.json')['started_at_utc']
    sources = {}
    for slot in slots:
        root = slot['root_id']; row = by_root[root]
        source_binding = {'path': slot['source_path'], 'sha256': slot['source_sha256']}
        ordering_slot = next(s for s in config['ordering_slots'] if s['root_id'] == root)
        check_fn(all(ordering_slot[k] == slot[k] for k in ('source_path', 'source_sha256', 'history_path', 'history_sha256'))
                 and ordering_slot['candidate_count'] == 2, 'both_lane_same_complete_source:' + root)
        check_fn(row['decision'] == 'READY' and row['source'] == source_binding
                 and row['blocking_findings'] == [], 'selected_ready_source_binding:' + root)
        source = read_fn(slot['source_path']); current_binding = bind_fn(slot['source_path'])
        check_fn(current_binding == source_binding and source['root_id'] == root,
                 'full_original_source_binding:' + root)
        source_review = read_fn(row['review']['path'])
        check_fn(bind_fn(row['review']['path']) == row['review']
                 and source_review['decision'] == 'READY'
                 and source_review['source'] == source_binding
                 and source_review['full_source_read'] is True
                 and source_review['turn_refs_read'] == [t['turn_ref'] for t in source['turns']],
                 'complete_source_ready_review_binding:' + root)
        directory = home / 'parent_review'; suffix = root.split('-')[-1]
        public_path = directory / ('root' + suffix + '_PUBLIC.json')
        parent_path = directory / ('root' + suffix + '_PARENT.json')
        public = read_fn(public_path); parent = read_fn(parent_path)
        judgment = read_fn(slot['parent_review_path']); provenance = read_fn(slot['parent_provenance_path'])
        check_fn(bind_fn(slot['parent_review_path'])['sha256'] == slot['parent_review_sha256']
                 and bind_fn(slot['parent_provenance_path'])['sha256'] == slot['parent_provenance_sha256']
                 and judgment['public_question'] == public
                 and judgment['public_question_path'] == str(public_path)
                 and judgment['public_question_sha256'] == bind_fn(public_path)['sha256'],
                 'fresh_parent_public_and_review_bindings:' + root)
        check_fn(judgment['candidate_id'] == parent['candidate_id'] == slot['parent_candidate_id']
                 and judgment['root_id'] == parent['root_id'] == public['root_id'] == root
                 and judgment['source_path'] == public['source_path'] == slot['source_path']
                 and judgment['source_sha256'] == public['source_sha256'] == slot['source_sha256']
                 and judgment['full_source_read'] is True
                 and judgment['full_source_turn_count'] == len(source['turns'])
                 and v88_parent_judgments_supported(judgment, public), 'fresh_parent_full_source_identity:' + root)
        check_fn(judgment['known_old_parent_gold'] is True
                 and judgment['current_parent_private_gold_read_before_judgments_saved'] is False
                 and judgment['current_parent_private_evidence_read_before_judgments_saved'] is False
                 and provenance['historical_parent_gold_known'] is True
                 and provenance['fresh_blind_parent_review'] is False
                 and provenance['full_source_reread_this_batch'] is True
                 and provenance['source_bytes_unchanged'] is True
                 and provenance['current_solver_scores_read'] is False
                 and provenance['training_allowed'] is False, 
                 'fresh_parent_historical_exposure_and_current_private_disclosure:' + root)
        check_fn(provenance['source'] == source_binding and provenance['public_question'] == bind_fn(public_path)
                 and provenance['parent'] == bind_fn(parent_path)
                 and provenance['parent_review_path'] == slot['parent_review_path']
                 and provenance['parent_review_sha256'] == slot['parent_review_sha256']
                 and provenance['source_judgments_saved_at_utc'] == judgment['created_at_utc']
                 and datetime.fromisoformat(judgment['created_at_utc']).utcoffset() is not None
                 and datetime.fromisoformat(provenance['current_private_parent_construction_at_utc']).utcoffset() is not None
                 and datetime.fromisoformat(judgment['created_at_utc']) <= datetime.fromisoformat(provenance['current_private_parent_construction_at_utc'])
                 <= datetime.fromisoformat(preparation_started),
                 'fresh_parent_provenance_saved_before_current_private:' + root)
        judgments = judgment['judgments']; lookup = {t['turn_ref']: t for t in source['turns']}
        check_fn(len(judgments) == 6 and {j['option'] for j in judgments} == set('ABCDEF')
                 and parent['gold'] == list('ABCDEF')
                 and parent['options'] == public['options'], 'fresh_supported_parent_six_option_identity:' + root)
        for item in judgments:
            label = item['option']
            check_fn(item['proposition'] == public['options'][label]
                     and item['source_judgment'] == 'supported' and item['entailed'] is True
                     and bool(item['required_evidence']) and bool(item['reason']),
                     'fresh_parent_supported_option:' + root + ':' + label)
            for evidence in item['required_evidence']:
                turn = lookup[evidence['turn_ref']]
                check_fn(evidence['speaker'] == turn['speaker'] and isinstance(evidence['quote'], str)
                         and bool(evidence['quote']) and evidence['quote'] in turn['text'],
                         'fresh_parent_exact_source_evidence:' + root + ':' + label)
        sources[root] = {'path': slot['source_path'], 'sha256': slot['source_sha256'], 'source': source}
    registry = read_fn(config['character_registry']['path'])
    check_fn(bind_fn(config['character_registry']['path']) == config['character_registry'], 'new_registry_frozen_binding')
    actor_policy = read_fn(config['actor_reference_policy']['path'])
    check_fn(bind_fn(config['actor_reference_policy']['path']) == config['actor_reference_policy']
             and actor_policy['registry'] == config['character_registry']
             and actor_policy['output_style'] == 'use_source_names_or_explicit_roles_for_people'
             and actor_policy['forbidden_gendered_pronouns'] == ['he','him','his','himself','she','her','hers','herself']
             and actor_policy['new_source_facts'] is False and actor_policy['automatic_repair'] is False,
             'bound_actor_name_policy_adds_no_facts_or_repairs')
    errors = registry_contract_errors(registry, slots, sources)
    check_fn(not errors, 'new_registry_exact_sources_names_and_unknown_fields', errors)
    return {'source_admission': bind_fn(home / 'SOURCE_ADMISSION.COMPLETE.json'),
            'ready_roots': list(ROOTS), 'fresh_current_parent_review_count': len(slots),
            'fresh_blind_parent_review': False, 'all_roots_first_round_train_sources': True,
            'independent_test_claimed': False,
            'candidate_hold_review_reused_from_previous_batch': True,
            'candidate_source_gate_rereview_this_batch': False}


def failed_multiselect_contract(completion, error, reservation, generation, public, assets, judgment, review, receipt, version_map, source, manifest, request_names, model, check):
    """Verify an HTTP failure without converting it into an empty writer answer."""
    check(set(completion) == {'http_completed_at_utc','http_status','outcome'}
          and type(completion['http_status']) is int and 400 <= completion['http_status'] <= 599
          and completion['outcome'] == 'error', 'failed_transport_http_error_receipt')
    check(set(error) == {'error_type','http_attempted','metadata','no_automatic_retry','raw_exception_text_saved'}
          and error['error_type'] == 'ComparisonTransportError' and error['http_attempted'] is True
          and error['no_automatic_retry'] is True and error['raw_exception_text_saved'] is False,
          'failed_transport_typed_error_and_no_retry')
    meta=error['metadata']
    check(meta['requested_model'] == model and meta['http_status'] == completion['http_status']
          and all(meta[key] is None for key in ('actual_model','returned_model','finish_reason','usage',
                    'input_tokens','output_tokens','reasoning_tokens','total_tokens'))
          and meta['timeout'] is False and meta['completion_status'] == 'UNKNOWN',
          'failed_transport_no_success_or_model_output_claim')
    check(all(type(meta[k]) is int and meta[k] == v for k,v in {
          'transport_attempt_count':1,'http_attempt_count':1,'transport_retry_count':0,'rate_limit_retry_count':0}.items())
          and reservation['maximum'] == reservation['request_number'] == 1 and reservation['automatic_retries'] == 0,
          'failed_transport_one_attempt_zero_retries')
    check(isinstance(meta['raw_response_text'],str) and bool(meta['raw_response_text'].strip())
          and 'raw_response' not in meta, 'failed_transport_service_error_body_preserved_without_response')
    expected_files={'ERROR.json','HTTP_ATTEMPT.json','HTTP_RESULT.json','PROFILE.json','PROOF.json',
                    'REQUEST.json','RESERVATION.json','THROTTLE.json'}
    check(set(request_names) == expected_files, 'failed_transport_exact_request_inventory_no_response_or_validation')
    check(generation['status'] == 'NO_VALID_ASSETS' and generation['fact_count'] == 6
          and generation['counterversion_count'] == 0 and generation['requests_prepared'] == generation['http_attempts'] == 1
          and all(generation[k] == 0 for k in ('semantic_retries','mechanical_repairs','transport_retries'))
          and generation['root_id'] == manifest['root_id'] and generation['round_id'] == manifest['round_id']
          and generation['target_fact_ids'] == manifest['target_fact_ids'] and generation['training_allowed'] is False,
          'failed_transport_round_has_zero_generated_versions_no_repair')
    facts=public['parent_facts']; identities={f['fact_id']:f['proposition'] for f in facts}
    check(len(facts)==len(identities)==len(assets['facts'])==6
          and {f['fact_id']:f['supported']['proposition'] for f in assets['facts']} == identities
          and all(f['contradicted'] is None and f['anchor'] is None for f in assets['facts'])
          and public['versions'] == version_map['versions'] == judgment['versions'] == review['versions'] == []
          and receipt['usable_contradicts'] == 0, 'failed_transport_no_generated_candidate_or_invented_review')
    check(len(judgment['parents'])==len(review['parents'])==6
          and {r['fact_id']:r['proposition'] for r in judgment['parents']}==identities
          and {r['fact_id'] for r in review['parents']}==set(identities)
          and judgment['full_source_read'] is True and judgment['source_first'] is True
          and judgment['complete_source_turns_read']==len(source['turns'])
          and judgment['new_variant_private_artifacts_read_before_save'] is False
          and judgment['solver_scores_used'] is False and judgment['current_solver_scores_read'] is False
          and review['source_first'] is True and review['full_source_read'] is True
          and receipt['source_first'] is True and receipt['source_judgments_bytes_changed_after_private_read'] is False,
          'failed_transport_empty_public_projection_has_complete_source_review')
    stamps=[reservation['reserved_at'],completion['http_completed_at_utc'],generation['completed_at_utc'],
            judgment['created_at_utc'],receipt['private_crosscheck_started_at_utc'],receipt['review_completed_at_utc']]
    parsed=[datetime.fromisoformat(t) for t in stamps]
    check(all(t.utcoffset() is not None for t in parsed) and parsed==sorted(parsed)
          and receipt['source_judgments_saved_at_utc']==judgment['created_at_utc'],
          'failed_transport_receipts_precede_saved_empty_review')
    return {'outcome':'TRANSPORT_FAILURE','http_status':completion['http_status'],
            'error_type':error['error_type'],'service_error_body':meta['raw_response_text'],
            'generated_candidate_count':0,'mechanical_validation_performed':False,
            'semantic_rejection_count':0,'public_variant_count':0,'retry_count':0,
            'empty_source_review_complete':True}


def audit():
    config = read(HOME / 'CONFIG.json')
    prep = read(HOME / 'BATCH_PREPARATION.COMPLETE.json')
    require(config['max_generation_http']==12 and config['max_solver_calls_per_model']==456
            and config['max_candidates']==28, 'four_root_mixed_generation_budgets')
    require(config['multiselect_engine'] == str(a.HERE), 'same_v79_multiselect_engine')
    check_binding(config['actor_reference_policy'])
    check_binding(config['character_registry'])
    actor_policy=read(config['actor_reference_policy']['path'])
    registry=read(config['character_registry']['path'])
    require(actor_policy['registry']==config['character_registry']
            and actor_policy['forbidden_gendered_pronouns']==list(PRONOUNS)
            and actor_policy['output_style']=='use_source_names_or_explicit_roles_for_people'
            and actor_policy['new_source_facts'] is False and actor_policy['automatic_repair'] is False,
            'frozen_actor_style_contract_adds_no_qa_facts_or_repairs')
    import source_admission, source_subset
    source_admission.verify()
    source_subset.validate(config)
    admission_findings = validate_admission_contract(config, HOME, read, bind, require)
    for row in prep['files']:
        check_binding(row)
    for row in read(a.HERE / 'OFFLINE_TEST_RECEIPT.json')['files']:
        check_binding(row)
    protected = read(HOME / 'PRESERVATION_BEFORE.json')
    require(len(protected['files'])==11763, 'all_11763_previous_files_protected')
    for path, digest in protected['files'].items():
        require(sha(path) == digest, 'protected_previous_file:' + path)
    for slot in config['multiselect_config']['slots']:
        a.verify(HOME / 'multiselect/assets' / slot['slot_id'], complete=True)
    ordering_result = ordering.verify(HOME)
    ordering_review.verify_review(HOME)
    assembled = c.verify_assembly(HOME / 'multiselect')
    final_review = common.load_multi('evaluate').verify_review(HOME / 'multiselect')
    require(final_review['usable_count'] <= len(assembled['questions']), 'final_review_usable_subset')
    _, _, annotate, capacity, dev = a.base.transport()
    tokenizer = dev.tokenizer_from(str(a.base.TOKENIZER))
    calls, multi_mechanics, order_mechanics = [], [], []
    request_dirs = sorted(HOME.glob('multiselect/assets/*/rounds/*/requests/*')) + sorted(HOME.glob('ordering/*/requests/*'))
    require(len(request_dirs) <= config['max_generation_http'], 'total_generation_request_budget')
    for request in request_dirs:
        is_multi = request.is_relative_to(HOME / 'multiselect')
        round_folder = request.parents[1]
        run = round_folder.parents[1] if is_multi else round_folder
        manifest = read(round_folder / 'MANIFEST.json')
        payload, proof = read(request / 'REQUEST.json'), read(request / 'PROOF.json')
        frozen_payload = read(round_folder / ('INITIAL_REQUEST.json' if is_multi else 'REQUEST.json'))
        frozen_proof = read(round_folder / ('INITIAL_PROOF.json' if is_multi else 'PROOF.json'))
        require(payload == frozen_payload and {k:v for k,v in proof.items() if k != 'capacity'} == frozen_proof,
                'request_frozen_payload_and_proof:' + str(request))
        require(proof['payload_sha256'] == a.base.digest(payload), 'request_payload_sha:' + str(request))
        source = read(run / 'SOURCE.json')
        content = payload['messages'][0]['content']
        context, _ = json.JSONDecoder().raw_decode(content.split('INPUT_JSON\n', 1)[1])
        require(context['source_turns'] == source['turns'] and context['source_metadata'] == {k:v for k,v in source.items() if k != 'turns'},
                'complete_source_context:' + str(request))
        original = Path(manifest['source_history']['path'] if is_multi else manifest['original_source']['path'])
        require((run / 'SOURCE.json').read_bytes() == original.read_bytes(), 'full_source_bytes:' + str(request))
        attempt = read(request / 'HTTP_ATTEMPT.json')
        reservation, completion = read(request / 'RESERVATION.json'), read(request / 'HTTP_RESULT.json')
        require(attempt == {'actual_payload_equals_frozen':True, 'payload_sha256':a.base.digest(payload),
            'requested_model':a.legacy.MODEL, 'http_attempt_number':1, 'transport_retries':0}, 'one_real_http_attempt:' + str(request))
        require(reservation['automatic_retries'] == 0 and reservation['request_number'] == 1, 'reservation_no_retry:' + str(request))
        transport_failed = completion.get('outcome') == 'error'
        if transport_failed:
            require(is_multi, 'failed_transport_supported_multiselect_lane:' + str(request))
            error = read(request / 'ERROR.json'); metadata = error['metadata']; response = None
        else:
            require(completion['http_status'] == 200 and completion['outcome'] == 'response' and not (request / 'ERROR.json').exists(), 'successful_response_receipt:' + str(request))
            response = read(request / 'RESPONSE.json'); metadata = response['metadata']
            require(json.loads(metadata['raw_response_text']) == metadata['raw_response'] and response['text'] == metadata['raw_response']['choices'][0]['message']['content'],
                    'raw_service_text_preserved:' + str(request))
            require(all(metadata[k] == a.legacy.MODEL for k in ['requested_model','actual_model','returned_model'])
                    and metadata['http_status'] == 200 and metadata['finish_reason'] == 'stop' and metadata['timeout'] is False,
                    'model_and_finish_reason:' + str(request))
            require(metadata['transport_attempt_count'] == metadata['http_attempt_count'] == 1
                    and metadata['transport_retry_count'] == metadata['rate_limit_retry_count'] == 0,
                    'transport_zero_retries:' + str(request))
        profile = read(request / 'PROFILE.json')
        require(payload == {'model':profile['model'], 'messages':payload['messages'],
                           **{k:v for k,v in profile['request_parameters'].items() if v is not None}}
                and profile['http_retry_limit'] == 0 and profile['parameter_fallback'] is False, 'profile_exact_request:' + str(request))
        cap = annotate(capacity(profile, tokenizer, payload['messages'], payload['max_tokens']))
        require(cap == proof['capacity'] and cap['status'] == 'PASS' and cap['truncation'] is False, 'capacity_no_truncation:' + str(request))
        entry = {'lane':'multi_select' if is_multi else 'ordering', 'root_id':manifest['root_id'],
            'round_id':manifest.get('round_id'), 'source_turn_count':len(source['turns']),
            'request':bind(request / 'REQUEST.json'), 'proof':bind(request / 'PROOF.json'),
            'http_attempt':bind(request / 'HTTP_ATTEMPT.json'), 'http_result':bind(request / 'HTTP_RESULT.json'),
            'profile':bind(request / 'PROFILE.json'), 'source':bind(run / 'SOURCE.json'),
            'capacity':cap, 'service_reported_usage':metadata['usage'], 'retry_count':0,
            'outcome':'TRANSPORT_FAILURE' if transport_failed else 'RESPONSE', 'http_status':completion['http_status']}
        entry['error' if transport_failed else 'response']=bind(request / ('ERROR.json' if transport_failed else 'RESPONSE.json'))
        if is_multi:
            targets = manifest['target_fact_ids']; tail = a.render_task_tail(manifest, targets)
            require(content.endswith(tail) and context['target_fact_ids'] == targets and proof['target_fact_ids'] == targets,
                    'rendered_current_target_ids:' + str(request))
            skeleton, _ = json.JSONDecoder().raw_decode(tail.split('OUTPUT_JSON_SKELETON\n',1)[1])
            require([r['fact_id'] for r in skeleton['replacements']] == targets, 'exact_target_skeleton:' + str(request))
            review = read(round_folder / 'SOURCE_JUDGMENTS.json'); public = read(round_folder / 'PUBLIC_ASSETS.json')
            if transport_failed:
                generation_complete=read(round_folder/'GENERATION.COMPLETE.json')
                assets=read(round_folder/'ASSETS.json'); private_review=read(round_folder/'ASSET_REVIEW.json')
                receipt=read(round_folder/'REVIEW_RECEIPT.json'); version_map=read(round_folder/'VERSION_MAP.json')
                failure=failed_multiselect_contract(completion,error,reservation,generation_complete,public,assets,
                    review,private_review,receipt,version_map,source,manifest,
                    [p.name for p in request.iterdir() if p.is_file()],a.legacy.MODEL,
                    lambda value,label:require(value,label+':'+str(request)))
                for recorded in generation_complete['files']+receipt['files']:
                    check_binding(recorded)
                require(all(bind(request/name) in generation_complete['files'] for name in
                    ('ERROR.json','HTTP_RESULT.json','HTTP_ATTEMPT.json','REQUEST.json','PROOF.json')),
                    'failed_transport_error_and_request_bound_in_round_completion:'+str(request))
                require(public['source_sha256']==sha(run/'SOURCE.json')
                    and public['assets_sha256']==version_map['assets_sha256']==sha(round_folder/'ASSETS.json')
                    and review['public_assets_sha256']==sha(round_folder/'PUBLIC_ASSETS.json')
                    and bind(round_folder/'PUBLIC_ASSETS.json') in receipt['files']
                    and receipt['source_judgments_sha256']==sha(round_folder/'SOURCE_JUDGMENTS.json'),
                    'failed_transport_empty_projection_review_bindings:'+str(request))
                entry.update(failure)
                entry['empty_round_review']={key:bind(round_folder/name) for key,name in (
                    ('generation_complete','GENERATION.COMPLETE.json'),('public_assets','PUBLIC_ASSETS.json'),
                    ('source_judgments','SOURCE_JUDGMENTS.json'),('asset_review','ASSET_REVIEW.json'),('review_receipt','REVIEW_RECEIPT.json'))}
                errors=[]
            else:
                data, errors = a.validate_assets(response['text'], manifest, source, read(run / 'PARENT.json'), targets)
                require(read(request / 'VALIDATION.json') == a.validation(data, errors), 'multiselect_mechanical_replay:' + str(request))
                try:
                    raw = a.strict_json(response['text'])
                    raw_by_fact = {r['fact_id']:r['replacement_proposition'] for r in raw['replacements']}
                    require(all(v['proposition'] == raw_by_fact[v['fact_id']] for v in public['versions']), 'public_asset_text_unedited:' + str(request))
                except (ValueError, TypeError, KeyError):
                    require(not public['versions'] and bool(errors), 'malformed_writer_output_remains_excluded:' + str(request))
            multi_mechanics.append({'root_id':manifest['root_id'], 'round_id':manifest['round_id'],
                'target_fact_ids':targets, 'mechanical_errors':errors, 'public_version_count':len(public['versions']),
                'generation_outcome':entry['outcome'], 'mechanical_validation_performed':not transport_failed,
                'transport_failure':failure if transport_failed else None,
                'source_review_parent_relations':dict(Counter(r['relation'] for r in review['parents'])),
                'source_review_version_relations':dict(Counter(r['relation'] for r in review['versions'])),
                'source_review':bind(round_folder / 'SOURCE_JUDGMENTS.json'), 'per_version_source_findings':review['versions']})
        else:
            prior = read(run / 'PRIOR.json')
            slot = next(s for s in config['ordering_slots'] if s['slot_id'] == manifest['slot_id'])
            public_fields = ('candidate_id', 'root_id', 'question_type', 'chronology_kind',
                             'question', 'items', 'source_path', 'source_sha256')
            expected_prior = [{key:q[key] for key in public_fields}
                for path in slot.get('prior_pool_paths', [slot['prior_pool_path']])
                for q in read(path)['questions']
                if q['root_id'] == manifest['root_id'] and q['question_type'] == 'ordering']
            require(prior == expected_prior, 'ordering_public_prior_reconstructed_from_config:' + str(request))
            require(len({normalized_set(q) for q in prior}) == len(prior),
                    'ordering_prior_contains_distinct_whole_sets:' + str(request))
            require(context['previous_public_questions_dedup_only'] == independent_ordering_prior_context(prior),
                    'ordering_compact_prior_index_preserves_public_identity_metadata:' + str(request))
            accepted, rejected = ordering.validate_raw(response['text'],manifest,source,run/'SOURCE.json',prior)
            require(accepted == read(run/'POOL.json')['questions'] and rejected == read(run/'REJECTIONS.json'), 'ordering_mechanical_replay:' + str(request))
            tail = content.split('[ORDERING_BATCH_V88]',1)[1].split('[/ORDERING_BATCH_V88]',1)[0]
            policy=read(run/'ACTOR_REFERENCE_POLICY.json')
            require(manifest['actor_reference_policy']==proof['actor_reference_policy']==bind(run/'ACTOR_REFERENCE_POLICY.json'),
                    'actor_policy_bound_to_manifest_and_executed_request:' + str(request))
            characters=[r for r in registry['characters'] if r['root_id']==manifest['root_id']]
            names=sorted({turn['speaker'] for turn in source['turns']})
            require(sorted(r['canonical_name'] for r in characters)==names
                    and all(r['gender']['value'] is None and r['third_person_pronouns']['value'] is None for r in characters),
                    'actor_registry_root_names_and_unknown_pronouns:' + str(request))
            registry_source=next(r for r in registry['source_bindings'] if r['root_id']==manifest['root_id'])
            require(registry_source['sha256']==manifest['source_sha256']==sha(run/'SOURCE.json'),
                    'actor_registry_source_version:' + str(request))
            expected_policy={'version':actor_policy['version'],'root_id':manifest['root_id'],
                'canonical_source_speaker_names':names,'pronoun_status':'unknown_for_all_source_speakers',
                'output_style':'use_source_names_or_explicit_roles_for_people','forbidden_gendered_pronouns':list(PRONOUNS),
                'scope':'generated item propositions only; use source names or explicit roles for any person',
                'source_evidence_unchanged':True,'new_source_facts':False,'automatic_repair':False}
            listed_policy,_=json.JSONDecoder().raw_decode(tail.split('ACTOR_REFERENCE_POLICY:\n',1)[1])
            require(policy==expected_policy==listed_policy
                    and context['actor_reference_policy']=={'location':'ACTOR_REFERENCE_POLICY in the task tail; complete frozen policy appears there once.'},
                    'independent_actor_policy_projection_once_in_tail_with_explicit_context_pointer:' + str(request))
            entry['actor_reference_policy']=bind(run/'ACTOR_REFERENCE_POLICY.json')
            scenes = list(dict.fromkeys(t['turn_ref'].rsplit(':', 1)[0] for t in source['turns']))
            scene_listing, _ = json.JSONDecoder().raw_decode(tail.split('available_scene_ids: ',1)[1])
            prior_listing, _ = json.JSONDecoder().raw_decode(tail.split('DISALLOWED PRIOR WHOLE SETS (public text, deduplication only):\n',1)[1])
            require(scene_listing == scenes and prior_listing['disallowed_prior_six_item_sets'] == [sorted(q['items'].values()) for q in prior], 'scene_and_public_dedup_payload:' + str(request))
            require('"gold"' not in json.dumps(prior_listing) and '"item_evidence"' not in json.dumps(prior_listing), 'ordering_prior_has_no_private_answers:' + str(request))
            plan = read(run/'ANCHOR_PLAN.json')
            require(manifest['anchor_plan'] == bind(run/'ANCHOR_PLAN.json'), 'frozen_anchor_plan_manifest_and_request_binding:' + str(request))
            check_binding(config['ordering_anchor_plan'])
            selected_plans = sorted([p for p in read(config['ordering_anchor_plan']['path'])['plans'] if p['root_id']==manifest['root_id']],key=lambda p:p['slot_number'])
            require(len(selected_plans)==len(manifest['candidate_ids'])==2 and [p['slot_number'] for p in selected_plans]==[1,2], 'independent_anchor_plan_two_slots:' + str(request))
            source_lookup = {t['turn_ref']: t for t in source['turns']}
            source_positions = {t['turn_ref']: i for i,t in enumerate(source['turns'])}
            expected_plans=[]
            for qid,p in zip(manifest['candidate_ids'], selected_plans):
                anchors=p['anchors']; scene=p['scene_id']
                require(len(anchors)==len({v['turn_ref'] for v in anchors})==6 and scene in scenes, 'six_distinct_planned_source_turns:' + qid)
                for anchor in anchors:
                    turn=source_lookup[anchor['turn_ref']]
                    require(anchor['turn_ref'].rsplit(':',1)[0]==scene and anchor['speaker']==turn['speaker']
                        and isinstance(anchor['quote'],str) and bool(anchor['quote'].strip()) and turn['text'].count(anchor['quote'])==1,
                        'independent_exact_single_scene_anchor:' + qid + ':' + anchor['turn_ref'])
                permutation=sorted(anchors,key=lambda v:hashlib.sha256((qid+'\0'+v['turn_ref']+'\0'+v['quote']).encode()).hexdigest())
                expected_plans.append({'candidate_id':qid,'root_id':manifest['root_id'],'scene_id':scene,
                    'question':'Order the statements, using left-to-right clause order within a turn, as they APPEAR IN THE SUPPLIED DIALOGUE within the '+scene+' scene.',
                    'anchors':{label:{k:v[k] for k in ('turn_ref','speaker','quote')} for label,v in zip(('I1','I2','I3','I4','I5','I6'),permutation)}})
            require(plan=={'version':'v88_private_fixed_single_scene_anchors_v1','plan_source':config['ordering_anchor_plan'],
                'root_id':manifest['root_id'],'source_sha256':manifest['source_sha256'],'questions':expected_plans,
                'semantic_certification':False,'gold_supplied_to_writer':False}, 'independent_plan_reconstruction:' + str(request))
            gold_plan = read(run / 'GOLD_PLAN.json')
            require(manifest['gold_plan'] == bind(run / 'GOLD_PLAN.json')
                    and manifest['gold_plan'] in manifest['files'], 'private_gold_plan_frozen_manifest_binding:' + str(request))
            expected_gold_rows = []
            for planned in expected_plans:
                positions = {}
                for label, anchor in planned['anchors'].items():
                    text = source_lookup[anchor['turn_ref']]['text']
                    begin = text.index(anchor['quote'])
                    positions[label] = {'turn_ref': anchor['turn_ref'],
                        'turn_index': source_positions[anchor['turn_ref']],
                        'quote_start': begin, 'quote_end': begin + len(anchor['quote'])}
                chronological = sorted(positions, key=lambda label: (
                    positions[label]['turn_index'], positions[label]['quote_start'], positions[label]['quote_end']))
                expected_gold_rows.append({'candidate_id': planned['candidate_id'], 'root_id': manifest['root_id'],
                    'gold': chronological, 'anchor_positions': positions})
            require(gold_plan == {'version': 'v88_program_gold_plan_v1',
                    'created_at_utc': gold_plan['created_at_utc'], 'source': bind(run / 'SOURCE.json'),
                    'anchor_plan': bind(run / 'ANCHOR_PLAN.json'),
                    'gold_origin': 'source_appearance_order_frozen_before_writer',
                    'semantic_certification': False, 'gold_supplied_to_writer': False,
                    'questions': expected_gold_rows}, 'independent_prepared_gold_plan_exact_source_position_reconstruction:' + str(request))
            require(manifest['gold_origin'] == gold_plan['gold_origin']
                    and datetime.fromisoformat(gold_plan['created_at_utc']).utcoffset() is not None
                    and datetime.fromisoformat(gold_plan['created_at_utc']) <= datetime.fromisoformat(manifest['prepared_at_utc'])
                    <= datetime.fromisoformat(read(HOME / 'ordering/PREPARATION.COMPLETE.json')['completed_at_utc'])
                    <= datetime.fromisoformat(read(run / 'GENERATION.STARTED.json')['started_at_utc']),
                    'gold_prepared_and_frozen_before_writer_generation:' + str(request))
            require('gold_plan' not in context and 'GOLD_PLAN.json' not in content
                    and 'anchor_positions' not in context and 'gold' not in context
                    and all(json.dumps(row['anchor_positions'], ensure_ascii=False) not in content for row in expected_gold_rows),
                    'private_gold_plan_and_answers_not_sent_to_writer:' + str(request))
            expected_gold_by_id = {row['candidate_id']: row for row in expected_gold_rows}
            entry['gold_plan'] = bind(run / 'GOLD_PLAN.json')
            expected_writer_questions = []
            for planned in expected_plans:
                ordered_anchors = sorted(planned['anchors'].values(), key=lambda anchor: source_positions[anchor['turn_ref']])
                expected_writer_questions.append({**{key: planned[key] for key in ('candidate_id','root_id','scene_id','question')},
                    'anchor_records': [{'anchor_id': anchor['turn_ref'], 'speaker': anchor['speaker'],
                        'quote': anchor['quote'], 'statement_prefix': anchor['speaker'] + ' '}
                        for anchor in ordered_anchors]})
            expected_writer_plan = {'version': 'v88_source_id_predicate_writer_plan_v1',
                'root_id': manifest['root_id'], 'source_sha256': manifest['source_sha256'],
                'questions': expected_writer_questions, 'display_labels_supplied': False,
                'gold_supplied_to_writer': False, 'semantic_certification': False}
            writer_plan = read(run / 'WRITER_PLAN.json')
            require(writer_plan == expected_writer_plan, 'independent_source_order_writer_plan_reconstruction:' + str(request))
            require(manifest['writer_plan'] == proof['writer_plan'] == bind(run / 'WRITER_PLAN.json')
                    and manifest['writer_plan'] in manifest['files'] and 'anchor_plan' not in proof,
                    'writer_plan_frozen_manifest_and_request_binding:' + str(request))
            require('anchor_plan' not in context and 'ANCHOR_PLAN.json' not in content
                    and 'GOLD_PLAN.json' not in content and 'anchor_positions' not in content
                    and all(json.dumps(p['anchors'], ensure_ascii=False) not in content for p in expected_plans)
                    and not re.search(r'"I[1-6]"', json.dumps(expected_writer_questions)),
                    'private_label_mapping_and_gold_absent_from_writer_request:' + str(request))
            listed_plans,_=json.JSONDecoder().raw_decode(tail.split('SOURCE_ID_ANCHOR_RECORDS:\n',1)[1])
            require(listed_plans==expected_writer_questions
                    and context['source_id_anchor_records']=={'location':'SOURCE_ID_ANCHOR_RECORDS in the task tail; complete immutable records appear there once.'},
                    'writer_anchor_records_once_in_complete_task_tail_with_context_pointer:' + str(request))
            expected_tail = ('\n' + ordering.TASK + '\nSOURCE_ID_ANCHOR_RECORDS:\n'
                + json.dumps(expected_writer_questions, ensure_ascii=False, separators=(',', ':'))
                + '\navailable_scene_ids: ' + json.dumps(scenes, ensure_ascii=False, separators=(',', ':'))
                + '\nDISALLOWED PRIOR WHOLE SETS (public text, deduplication only):\n'
                + json.dumps({'disallowed_prior_six_item_sets': [sorted(q['items'].values()) for q in prior]}, ensure_ascii=False, separators=(',', ':'))
                + '\nACTOR_REFERENCE_POLICY:\n' + json.dumps(expected_policy, ensure_ascii=False, separators=(',', ':')) + '\n')
            require(tail == expected_tail, 'complete_task_tail_matches_frozen_anchor_records_prompt_and_independent_inputs:' + str(request))
            require(content.count(json.dumps(expected_writer_questions,ensure_ascii=False,separators=(',',':')))==1
                    and content.count(json.dumps(expected_policy,ensure_ascii=False,separators=(',',':')))==1,
                    'complete_writer_plan_and_policy_objects_each_serialized_once:' + str(request))
            entry['anchor_plan']=bind(run/'ANCHOR_PLAN.json')
            entry['writer_plan']=bind(run/'WRITER_PLAN.json')
            try:
                raw = a.strict_json(response['text']); raw_rows = raw['questions']
            except (ValueError, TypeError, KeyError):
                raw_rows = []
            accepted_by_id = {q['candidate_id']:q for q in accepted}; rejected_by_id = {q['candidate_id']:q for q in rejected}
            for qid, row,planned in zip(manifest['candidate_ids'],raw_rows,expected_plans):
                findings = {'candidate_id':qid, 'writer_null':row is None, 'mechanically_accepted':qid in accepted_by_id,
                    'rejection_reasons':rejected_by_id.get(qid,{}).get('reasons',[]), 'gold_or_items_rewritten':False, 'gold_origin':'source_appearance_order_frozen_before_writer',
                    'public_scene_and_exact_evidence_bound_before_writer':True,'planned_scene_id':planned['scene_id']}
                if isinstance(row,dict):
                    findings['program_bound_question'] = planned['question']
                    findings['writer_has_forbidden_gold_field'] = 'gold' in row
                    findings['writer_has_forbidden_item_label_field'] = 'items' in row
                    findings['prepared_program_gold'] = expected_gold_by_id[qid]['gold']
                    rows = row.get('records')
                    expected_ids = {anchor['turn_ref'] for anchor in planned['anchors'].values()}
                    schema_ok = (set(row) == {'candidate_id','root_id','records'}
                        and row.get('candidate_id') == qid and row.get('root_id') == manifest['root_id']
                        and isinstance(rows,list) and len(rows)==6
                        and all(isinstance(r,dict) and set(r)=={'anchor_id','predicate'}
                            and isinstance(r['anchor_id'],str) and isinstance(r['predicate'],str)
                            and bool(r['predicate']) and r['predicate']==r['predicate'].strip() for r in rows))
                    if schema_ok:
                        schema_ok = (len({r['anchor_id'] for r in rows}) == 6
                            and {r['anchor_id'] for r in rows} == expected_ids)
                    findings['independent_writer_record_schema_valid'] = schema_ok
                    if not schema_ok:
                        require(qid not in accepted_by_id and bool(rejected_by_id.get(qid,{}).get('reasons')),
                                'independent_invalid_records_rejected_without_repair:' + qid)
                    else:
                        predicates = {r['anchor_id']:r['predicate'] for r in rows}
                        expected_items = {label:anchor['speaker']+' '+predicates[anchor['turn_ref']]
                            for label,anchor in planned['anchors'].items()}
                        findings['program_constructed_items'] = expected_items
                        findings['verbatim_writer_predicates'] = predicates
                        findings['prior_whole_set_matches'] = [q['candidate_id'] for q in prior if
                            ordering.whole_set(q)==ordering.whole_set({'root_id':manifest['root_id'],'items':expected_items})]
                        style_errors=independent_name_errors(expected_items)
                        findings['name_style_contract_errors']=style_errors
                        if style_errors:
                            require(qid not in accepted_by_id and bool(rejected_by_id[qid]['reasons']),
                                    'independent_name_style_failure_is_rejected_without_edits:' + qid)
                        if qid in accepted_by_id:
                            q = accepted_by_id[qid]
                            require(expected_gold_by_id[qid]['gold'] == q['gold'] == q['original_gold']
                                and expected_items == q['items'] and planned['question']==q['question'],
                                'ordering_verbatim_predicate_prefix_and_prefrozen_private_label_gold:' + qid)
                            require(not style_errors, 'accepted_ordering_items_follow_name_style_contract:' + qid)
                            require(q['item_evidence']=={label:{**anchor,'status':'statement',
                                'reason':'Source-ID-bound predicate with a frozen source-speaker prefix. Independent full-source review must certify the assembled statement.'}
                                for label,anchor in planned['anchors'].items()}, 'independent_program_bound_exact_evidence:' + qid)
                            chronological=sorted(planned['anchors'],key=lambda label:source_positions[planned['anchors'][label]['turn_ref']])
                            require(q['gold']==chronological, 'independent_source_position_gold_constructed_before_writer_not_repair:' + qid)
                            scope = independent_scope(q,source)
                            require(scope == common.load_ordering_gate().ordering_match_scope(q,source), 'independent_public_scope_parser:' + qid)
                            require(scope['kind']=='explicit_scene_union' and scope['scene_refs']==[planned['scene_id']]
                                    and all(v['turn_ref'] in scope['matching_turn_refs'] for v in q['generated_order_anchor_spans'].values()), 'generated_anchors_in_fixed_public_single_scene:' + qid)
                            findings['ordering_match_scope'] = scope
                order_mechanics.append(findings)
        calls.append(entry)

    multi = HOME / 'multiselect'; cfg = read(multi/'CONFIG.json')
    prior_paths = {Path(cfg['baseline_pool'])} | {Path(p) for s in cfg['slots'] for p in s.get('prior_pool_paths',[])}
    prior_index = {}
    for path in sorted(prior_paths):
        for original in c._rows(read(path)):
            q = original.get('original_selected_question',original)
            if q.get('question_type') == 'multi_select':
                prior_index.setdefault(normalized_set(q),[]).append({'candidate_id':q['candidate_id'],'prior':bind(path)})
    current_public = read(multi/'PUBLIC_QUESTIONS.json')['questions']
    members = read(multi/'FAMILY_SIDECAR.json')['members']; roles = {r['candidate_id']:r for r in members}
    require(len(current_public) == len(members) and set(roles) == {q['candidate_id'] for q in current_public}, 'public_family_member_identity')
    require(all(r['new_candidate'] == (r['role']=='child') for r in members), 'control_newness_roles')
    dedup, novel, repeated_facts = [], [], []
    for slot in cfg['slots']:
        run = multi/'assets'/slot['slot_id']; public = read(run/'PUBLIC_ASSETS.json')
        source_judgment = read(run/'SOURCE_JUDGMENTS.json')
        parents_usable = independent_parent_family_usable(public, source_judgment)
        variants = {p['fact_id']:[(p['proposition'],True)] for p in public['parent_facts']}
        for v in public['versions']:
            variants[v['fact_id']].append((v['proposition'],False))
        available = {k:[] for k in slot['target_counts']}
        for combination in (product(*(variants[fid] for fid in sorted(variants))) if parents_usable else ()):
            count = sum(supported for _,supported in combination)
            if count not in available:
                continue
            key = (slot['root_id'],'multi_select',tuple(sorted(' '.join(text.split()).casefold() for text,_ in combination)))
            if len(set(key[2])) != 6:
                continue
            if key in prior_index:
                dedup.append({'root_id':slot['root_id'],'gold_count_from_reviewed_supports':count,
                    'whole_set_digest':c.digest(key),'normalized_statement_set':list(key[2]),
                    'matched_prior_questions':prior_index[key],'excluded_from_new_count':True})
            else:
                available[count].append(key)
        children = [q for q in current_public if q['root_id']==slot['root_id'] and roles[q['candidate_id']]['role']=='child']
        require(parents_usable or not children,
                'unreliable_parent_family_has_no_new_children:' + slot['root_id'])
        for count, keys in available.items():
            matching = [q for q in children if roles[q['candidate_id']]['gold_count']==count]
            require(len(matching) == bool(keys), 'one_novel_combination_per_available_gold:' + slot['root_id'] + ':' + str(count))
            if not keys:
                continue
            chosen = min(keys,key=c.digest); q=matching[0]
            require(normalized_set(q)==chosen and chosen not in prior_index, 'independent_dedup_min_digest_composition:' + q['candidate_id'])
            require(q['candidate_id']==slot['root_id']+'-v79-'+slot['family_id']+'-'+c.digest(chosen)[:12], 'reused_engine_candidate_identity:' + q['candidate_id'])
            novel.append({'candidate_id':q['candidate_id'],'root_id':q['root_id'],'gold_count_from_reviewed_supports':count,
                'whole_set_digest':c.digest(chosen),'new_exact_six_option_combination':True,'new_independent_facts':False,
                'same_parent_and_source_family':slot['family_id']})
        new = read(run/'rounds/initial/PUBLIC_ASSETS.json')
        for old_batch in ('h4_ready_roots_mixed_batch_v78','h4_output_contract_mixed_batch_v79','h4_scoped_ordering_mixed_batch_v80','h4_bound_scene_ordering_mixed_batch_v81'):
            old_path = HOME.parent/old_batch/'multiselect/assets'/slot['slot_id']/'rounds/initial/PUBLIC_ASSETS.json'
            if not old_path.exists():
                repeated_facts.append({'root_id':slot['root_id'], 'comparison_status':'not_available',
                    'old_public_assets':{'diagnostic_path':str(old_path.resolve()),'availability':'not_available'},
                    'new_public_assets':bind(run/'rounds/initial/PUBLIC_ASSETS.json'),
                    'identical_per_fact_replacements':None,
                    'limitation':'No historical asset file exists for this new root/slot; no repetition conclusion is drawn.'})
                continue
            old_by_fact = {r['fact_id']:r['proposition'] for r in read(old_path)['versions']}
            repeated_facts.append({'root_id':slot['root_id'],'comparison_status':'available','old_public_assets':bind(old_path),
                'new_public_assets':bind(run/'rounds/initial/PUBLIC_ASSETS.json'),
                'identical_per_fact_replacements':[v['fact_id'] for v in new['versions'] if old_by_fact.get(v['fact_id'])==v['proposition']]})
    review = read(HOME/'ordering/CONTENT_REVIEW.json'); review_sources = read(HOME/'ordering/SOURCE_JUDGMENTS.json')
    require(review['accepted_count']+review['rejected_count']==ordering_result['candidate_count'] and review['question_or_gold_edits']==[], 'ordering_content_counts_and_no_edits')
    review_findings = []
    for q in review['questions']:
        source_row = next(r for r in review_sources['questions'] if r['candidate_id']==q['candidate_id'])
        private = next(row for row in read(HOME/'ordering/POOL.json')['questions'] if row['candidate_id'] == q['candidate_id'])
        source = read(private['source_path']); lookup = {turn['turn_ref']: turn for turn in source['turns']}
        for label, item in source_row['items'].items():
            primary = item['order_anchor']; fixed = private['item_evidence'][label]
            primary_turn = lookup.get(primary['turn_ref']); fixed_turn = lookup[fixed['turn_ref']]
            primary_start = primary_turn['text'].find(primary['quote']) if primary_turn else -1
            fixed_start = fixed_turn['text'].index(fixed['quote'])
            contained = (primary_turn is not None and primary_start >= 0
                and primary['turn_ref'] == fixed['turn_ref']
                and primary_turn['speaker'] == fixed['speaker']
                and fixed_start <= primary_start
                and primary_start + len(primary['quote']) <= fixed_start + len(fixed['quote']))
            if q['status'] == 'USABLE':
                require(contained, 'usable_ordering_independent_semantic_anchor_within_fixed_label_quote:' + q['candidate_id'] + ':' + label)
            elif not contained and label in q['order_anchor_spans']:
                require(label + ':fixed_anchor_disagrees_with_independent_semantic_anchor' in q['reasons'],
                        'fixed_label_semantic_mismatch_is_retained_as_exclusion:' + q['candidate_id'] + ':' + label)
        review_findings.append({'candidate_id':q['candidate_id'],'status':q['status'],'reasons':q['reasons'],
            'complete_source_turns_read':source_row['complete_source_turns_read'], 'source_review_reason':source_row.get('reason'),
            'alternative_orders':source_row['alternative_orders'], 'items':source_row['items']})
    initial_count=read(HOME/'BATCH_GENERATION.COMPLETE.json')['http_attempts']
    expected_initial=len(config['multiselect_config']['slots'])+len(config['ordering_slots'])
    multi_calls=read(multi/'GENERATION.COMPLETE.json')['http_attempts']
    ordering_calls=read(HOME/'ordering/GENERATION.COMPLETE.json')['http_attempts']
    supplement_calls=sum(row['round_id']=='supplement' for row in calls)
    require(initial_count == expected_initial and multi_calls+ordering_calls==len(calls) and len(calls)-initial_count==supplement_calls,
            'initial_snapshot_plus_explicit_supplements')
    for slot in cfg['slots']:
        require(sum(r['root_id']==slot['root_id'] and r['round_id']=='supplement' for r in calls)<=1, 'supplement_family_budget:' + slot['root_id'])
    controls=sum(r['role']=='control6' for r in members)
    transport_failures=[row for row in calls if row['outcome']=='TRANSPORT_FAILURE']
    shortfalls=bool(transport_failures or len(novel)<sum(len(s['target_counts']) for s in cfg['slots']) or ordering_result['candidate_count']<sum(s['candidate_count'] for s in config['ordering_slots']) or review['rejected_count'] or final_review['usable_count']<len(assembled['questions']))
    return {'status':'PASS_WITH_GENERATION_AND_CONTENT_SHORTFALLS' if shortfalls else 'PASS',
        'totals':{'actual_generation_http_calls':len(calls),
            'successful_generation_http_calls':len(calls)-len(transport_failures),
            'failed_generation_http_calls':len(transport_failures),
            'generation_http_status_distribution':dict(Counter(str(row['http_status']) for row in calls)),
            'initial_batch_snapshot_http_calls':initial_count,
            'explicit_bounded_supplement_http_calls':supplement_calls,'new_multiselect_six_option_combinations':len(novel),
            'historical_controls':controls,'multiselect_assembled':len(assembled['questions']),'multiselect_content_usable':final_review['usable_count'],
            'ordering_slot_count':sum(s['candidate_count'] for s in config['ordering_slots']),
            'ordering_mechanical_accepted':ordering_result['candidate_count'],'ordering_content_usable':review['accepted_count'],
            'ordering_content_rejected':review['rejected_count'],'protected_previous_file_count':len(protected['files'])},
        'source_admission_and_parent_findings':admission_findings,
        'calls':calls,'transport_failures':transport_failures,'multiselect_rounds':multi_mechanics,'ordering_mechanical_findings':order_mechanics,
        'ordering_source_review_findings':review_findings,'deduplicated_combinations':dedup,'new_combinations':novel,
        'repeated_per_fact_replacement_texts':repeated_facts,
        'evidence':[bind(multi/'PUBLIC_QUESTIONS.json'),bind(multi/'ALLOCATION.json'),bind(multi/'FAMILY_SIDECAR.json'),
            bind(HOME/'ordering/CONTENT_REVIEW.json'),bind(HOME/'ordering/SOURCE_JUDGMENTS.json'),bind(HOME/'BATCH_PREPARATION.COMPLETE.json')],
        'limitations':['This audit replays records and exact-set composition, not source answer semantics or semantic novelty.',
            'The pronoun token rule is a generation style contract; passing it does not establish correct actor attribution, facts or order.',
            'New multiselect combinations retain parent/source dependencies; the audit does not certify independent facts.',
            'Fixed single-scene anchors prevent writer binding edits, but do not certify proposition support or unique semantic order.',
            'Explicit scene scope affects only matching occurrences; complete source remains in every generation request.',
            'Capacity uses the frozen local proxy; service token usage is reported separately.']}


def main():
    if OUT.exists():
        raise RuntimeError('refuse_overwrite:' + str(OUT))
    report = {'version':'v88_independent_generation_audit_v2_transport_failure','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'disclosures':{'audit_generation_http_calls':0,'audit_solver_calls':0,'network_and_subprocess_blocked':True,
            'generation_private_and_raw_read':True,'current_final_multiselect_private_pool_read':True,
            'current_solver_outputs_read':False,'frozen_code_modified':False,'blind_semantic_review_claimed':False}}
    try:
        report.update(audit())
    except BaseException as error:
        report['status']='FAIL'
        CHECKS.append({'check':'audit_execution_exception','status':'FAIL','detail':{'error':repr(error),'traceback':traceback.format_exc()}})
    report['checks']=CHECKS
    report['check_count']=len(CHECKS)
    report['failed_checks']=[row for row in CHECKS if row['status']=='FAIL']
    report['opened_current_batch_files']=sorted(OPENED)
    report['auditor']=bind(__file__)
    with OUT.open('x') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({'path':str(OUT),'status':report['status'],'checks':len(CHECKS),'failed_checks':report['failed_checks']},ensure_ascii=False))
    if report['status']=='FAIL':
        raise SystemExit(1)


if __name__=='__main__':
    main()
