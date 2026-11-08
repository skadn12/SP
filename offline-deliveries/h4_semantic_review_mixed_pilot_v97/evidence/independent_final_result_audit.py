#!/usr/bin/env python3
"""Read-only stdlib independent final accounting; does not import project code.

An exclusive output is deliberate: every completed audit, including a failure,
must remain visible. This certifies saved counts and bindings, not semantics.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


BASE = Path(__file__).resolve().parent.parent
RUN = BASE / 'expansion_engine/runs/four_roots_B_scene01'
OUT = BASE / 'evidence/INDEPENDENT_FINAL_RESULT_AUDIT.json'
ROOTS = ['cc-train-episode-19053', 'cc-train-episode-9557',
         'cc-train-episode-104814', 'cc-train-episode-194259']
SOURCES = ['soda:train:6028079', 'soda:train:6028171',
           'soda:train:6028073', 'soda:train:6027861']
checks, failures, binding_records = [], [], []


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def canonical_sha(value):
    return text_sha(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               separators=(',', ':')))


def check(name, condition, actual=None, expected=None):
    row = {'check': name, 'status': 'PASS' if condition else 'FAIL'}
    if not condition:
        row.update(actual=actual, expected=expected)
        failures.append(row)
    checks.append(row)
    return condition


def eq(name, actual, expected):
    return check(name, actual == expected, actual, expected)


def binding(item, name):
    path = Path(item['path'])
    observed = sha(path) if path.is_file() else None
    eq(name, observed, item['sha256'])
    binding_records.append({'path': str(path), 'sha256': observed})


def nested_bindings(value, label):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            binding(value, label)
        for key, child in value.items():
            if isinstance(child, (list, dict)):
                nested_bindings(child, label + '/' + key)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            nested_bindings(child, label + '/' + str(i))


def payload_with_user(request, text):
    result = dict(request)
    result['messages'] = [dict(message) for message in request['messages']]
    result['messages'][-1]['content'] = text
    return result


def identity(use):
    return (use['dataset'], use['split'], str(use['item_id']))


def main():
    final = read(BASE / 'RESULTS.json')
    stage = read(BASE / 'evidence/STAGE_RESULTS.json')
    report = read(RUN / 'REPORT.json')
    manifest = read(RUN / 'MANIFEST.json')
    precall = read(BASE / 'PRECALL_PLAN.json')
    decisions = read(BASE / 'evidence/READY_SOURCE_DECISIONS.json')
    feedback_audit = read(BASE / 'evidence/TRANSPORT_FEEDBACK_AUDIT.json')
    eq('fixed final root assignments', final['fixed_roots'], ROOTS)
    eq('fixed prepared root assignments', manifest['roots'], ROOTS)
    eq('fixed precall root assignments', precall['candidate_roots'], ROOTS)
    eq('fixed precall sources', precall['source_ids'], SOURCES)
    eq('fixed decisions root assignments', decisions['fixed_roots'], ROOTS)
    eq('route did not change', manifest['selected_routes'], ['dataset_new_event'])
    eq('no replacement after failure', final['fixed_scope_replaced_after_failure'], False)
    eq('precall no fallback', precall['historical_source_fallback'], False)
    eq('precall no failure replacement', precall['failure_replacement'], False)

    preservation = read(BASE / 'PRESERVATION_BEFORE.json')
    eq('protected old file count declaration', preservation['protected_count'], 1863)
    eq('protected old file count actual', len(preservation['files']), 1863)
    for entry in preservation['files']:
        binding(entry, 'old protected file:' + entry['path'])
    code_counts = {}
    for name, expected in [('EXPANSION_MINFIX_RECEIPT.json', 142),
                           ('ORDERING_MINFIX_RECEIPT.json', 15)]:
        receipt = read(BASE / 'evidence' / name)
        eq(name + ' file count', len(receipt['files']), expected)
        code_counts[name] = len(receipt['files'])
        for entry in receipt['files']:
            binding(entry, 'frozen code:' + entry['path'])
    snapshot = read(RUN / 'EXECUTION_CODE_SNAPSHOT.json')
    for entry in snapshot['files']:
        binding({'path': entry['original_path'], 'sha256': entry['sha256']},
                'execution original:' + entry['original_path'])
        binding({'path': entry['snapshot_path'], 'sha256': entry['sha256']},
                'execution snapshot:' + entry['snapshot_path'])
    for entry in manifest['files']:
        binding(entry, 'prepared manifest:' + entry['path'])

    root_data = {}
    for root in ROOTS:
        branch = RUN / 'roots' / root / 'B'
        old = read(branch / 'INPUT.json')['turns']
        cumulative = read(branch / 'cumulative_dialogue.json')['turns']
        eq(root + ' frozen full input digest', canonical_sha(old), read(branch / 'INPUT.json')['branch_base_sha256'])
        eq(root + ' cumulative prefix', cumulative[:len(old)], old)
        result = read(branch / 'RESULT.json')
        eq(root + ' result vs report route', result, next(r for r in report['routes'] if r['root_id'] == root))
        root_data[root] = {'root_id': root, 'source_ids': [SOURCES[ROOTS.index(root)]],
                           'request_roles': Counter(), 'input_turns': len(old),
                           'mechanically_valid_plans': 0, 'plans_cleared_for_realization': 0,
                           'realizer_responses': 0, 'raw_generated_turns': 0,
                           'model_accepted_scenes': 0, 'model_accepted_new_turns': 0,
                           'cumulative_turns': len(cumulative), 'raw_candidates': []}
    eq('historical counts in fixed order', [root_data[r]['input_turns'] for r in ROOTS], [80, 92, 94, 149])

    roles, finish, statuses, max_tokens = Counter(), Counter(), Counter(), Counter()
    usage = Counter()
    prompt_tokens, output_tokens, http_attempt_numbers, raw_candidates = [], [], [], []
    feedback_chains = []
    extraction_accepted, extraction_rejected = [], []
    request_rows, retry_rows = [], []
    directories = sorted(p for p in (RUN / 'requests').iterdir() if p.is_dir())
    eq('logical request sequence', [int(p.name.split('-', 1)[0]) for p in directories], list(range(1, 22)))
    for directory in directories:
        proof = read(directory / 'PROOF.json')
        request = read(directory / 'REQUEST.json')
        response = read(directory / 'RESPONSE.json')
        role, root = proof['role'], proof.get('root_id')
        roles[role] += 1
        if root:
            root_data[root]['request_roles'][role] += 1
        digest = canonical_sha(request)
        eq(directory.name + ' frozen payload hash', digest, proof['payload_sha256'])
        max_tokens[str(request['max_tokens'])] += 1
        attempts = sorted(directory.glob('HTTP_ATTEMPT*.json'))
        results = sorted(directory.glob('HTTP_RESULT_*.json'))
        eq(directory.name + ' attempt/result pair count', len(attempts), len(results))
        for attempt_path, result_path in zip(attempts, results):
            attempt, result = read(attempt_path), read(result_path)
            http_attempt_numbers.append(attempt['http_attempt_number'])
            eq(attempt_path.name + '/' + directory.name + ' payload', attempt['payload_sha256'], digest)
            eq(attempt_path.name + '/' + directory.name + ' frozen equality', attempt['actual_payload_equals_frozen'], True)
            statuses[str(result['http_status'])] += 1
            if len(attempts) > 1:
                retry_rows.append({'request': directory.name, 'attempt': attempt, 'result': result})
        raw_http = sorted(directory.glob('HTTP_RESPONSE_*.json'))
        eq(directory.name + ' one successful saved HTTP response', len(raw_http), 1)
        actual_response = read(raw_http[0])
        eq(directory.name + ' logical response text equals actual attempt', response['text'], actual_response['text'])
        provider = json.loads(actual_response['metadata']['raw_response_text'])
        choice = provider['choices'][0]
        eq(directory.name + ' actual provider text', choice['message']['content'], response['text'])
        finish[choice['finish_reason']] += 1
        eq(directory.name + ' finish binding', response['metadata']['finish_reason'], choice['finish_reason'])
        for name in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            usage[name] += provider['usage'][name]
            eq(directory.name + ' usage:' + name, response['metadata']['usage'][name], provider['usage'][name])
        prompt_tokens.append(provider['usage']['prompt_tokens'])
        output_tokens.append(provider['usage']['completion_tokens'])
        eq(directory.name + ' usage sums', provider['usage']['total_tokens'],
           provider['usage']['prompt_tokens'] + provider['usage']['completion_tokens'])
        raw = json.loads(response['text'])
        if role == 'dataset_extractor':
            validation = read(directory / 'VALIDATION.json')
            extraction_accepted.extend(validation['accepted_cards'])
            extraction_rejected.extend(validation['rejected_records'])
            eq(directory.name + ' candidate accounting', len(raw['cards']), len(validation['candidate_results']))
        if role == 'planner':
            if read(directory / 'VALIDATION.json')['status'] == 'PASS':
                root_data[root]['mechanically_valid_plans'] += len(raw['scenes'])
            decision_path = directory / 'EVENT_STATE_DECISION.json'
            if decision_path.exists() and read(decision_path)['status'] == 'PASS':
                root_data[root]['plans_cleared_for_realization'] += len(raw['scenes'])
        if role == 'realizer':
            turns = raw['turns']
            accepted = read(directory / 'ACCEPTANCE.json')
            row = {'request': str(directory), 'turns': len(turns), 'acceptance': accepted}
            raw_candidates.append(row)
            root_data[root]['raw_candidates'].append(row)
            root_data[root]['realizer_responses'] += 1
            root_data[root]['raw_generated_turns'] += len(turns)
            if accepted['status'] == 'PASS':
                root_data[root]['model_accepted_scenes'] += 1
                root_data[root]['model_accepted_new_turns'] += len(turns)
                saved = read(RUN / 'roots' / root / 'B/S01.json')
                eq(directory.name + ' accepted saved exact raw body', saved, raw)
        if role in ('planner', 'realizer'):
            text = request['messages'][-1]['content']
            frozen_suffix = (RUN / 'feedback' / root / ('dataset_new_event__' + role + '.txt')).read_text(encoding='utf-8')
            receipt = proof['feedback_receipt']
            tail_receipt = proof['planner_tail_receipt'] if role == 'planner' else proof['realizer_tail_receipt']
            suffix_position = text.rfind(frozen_suffix)
            check(directory.name + ' literal frozen feedback exists', suffix_position >= 0)
            before_feedback = text[:suffix_position]
            with_feedback = text[:suffix_position + len(frozen_suffix)]
            tail = text[suffix_position + len(frozen_suffix):]
            eq(directory.name + ' suffix hash', text_sha(frozen_suffix), receipt['suffix_sha256'])
            eq(directory.name + ' suffix length', len(frozen_suffix), receipt['suffix_chars'])
            eq(directory.name + ' pre feedback text', text_sha(before_feedback), receipt['before_user_text_sha256'])
            eq(directory.name + ' with feedback text', text_sha(with_feedback), receipt['request_user_text_sha256'])
            eq(directory.name + ' with feedback payload', canonical_sha(payload_with_user(request, with_feedback)), receipt['request_payload_sha256'])
            eq(directory.name + ' tail before text', tail_receipt['before_text_sha256'], receipt['request_user_text_sha256'])
            eq(directory.name + ' tail before payload', tail_receipt['before_payload_sha256'], receipt['request_payload_sha256'])
            eq(directory.name + ' tail bytes', text_sha(tail), tail_receipt['tail_sha256'])
            eq(directory.name + ' tail final text', text_sha(text), tail_receipt['final_text_sha256'])
            eq(directory.name + ' tail final payload', digest, tail_receipt['final_payload_sha256'])
            eq(directory.name + ' prior training source lock', sha(RUN / 'feedback/source_lock.json'), receipt['source_lock_sha256'])
            eq(directory.name + ' no current feedback', receipt['current_generation_feedback_used'], False)
            eq(directory.name + ' target model', receipt['target_solver_model'], 'M_clean_policy')
            eq(directory.name + ' no other branch', proof['other_branch_in_generator_context'], False)
            eq(directory.name + ' no truncation', proof['capacity']['truncation'], False)
            input_value, _ = json.JSONDecoder().raw_decode(before_feedback.split('INPUT_JSON\n', 1)[1])
            full_history = read(RUN / 'roots' / root / 'B/INPUT.json')['turns']
            eq(directory.name + ' actual entire history', input_value['cumulative_history'], full_history)
            eq(directory.name + ' history digest', canonical_sha(full_history), proof['history_sha256'])
            eq(directory.name + ' history turn count', len(full_history), proof['history_turns'])
            feedback_chains.append({'request': directory.name, 'role': role,
                                    'history_turns': len(full_history), 'suffix_sha256': text_sha(frozen_suffix)})
        request_rows.append({'request': directory.name, 'root_id': root, 'role': role,
                             'http_attempts': len(attempts), 'finish_reason': choice['finish_reason'],
                             'provider_usage': provider['usage']})

    eq('global actual HTTP attempt sequence', sorted(http_attempt_numbers), list(range(1, 23)))
    eq('actual roles', dict(roles), dict(stage['roles']))
    eq('actual HTTP statuses', dict(statuses), final['transport']['http_statuses'])
    eq('actual finish reasons', dict(finish), final['transport']['finish_reasons'])
    eq('all request caps', dict(max_tokens), {'4096': 21})
    eq('actual provider usage', dict(usage), final['transport']['provider_usage'])
    eq('actual prompt token range', [min(prompt_tokens), max(prompt_tokens)], final['transport']['prompt_token_range'])
    eq('actual completion token range', [min(output_tokens), max(output_tokens)], final['transport']['completion_token_range'])
    eq('length responses', finish.get('length', 0), final['transport']['length_response_count'])
    eq('output cap hits', sum(n == 4096 for n in output_tokens), 0)
    eq('terminal saved transport errors', len(list((RUN / 'requests').glob('*/ERROR.json'))), final['transport']['terminal_transport_failures'])
    eq('retry has two attempt records', len(retry_rows), 2)
    eq('retry request fixed', {r['request'] for r in retry_rows}, {'018-cc-train-episode-104814-B-S01-event-state-1-1'})
    eq('retry statuses', [r['result']['http_status'] for r in retry_rows], [503, 200])
    eq('retry keeps exact payload', len({r['attempt']['payload_sha256'] for r in retry_rows}), 1)
    eq('retry second flag', retry_rows[1]['attempt']['same_payload_retry'], True)
    eq('actual feedback chain count', len(feedback_chains), 7)
    eq('feedback audit exact chain count', feedback_audit['summary']['feedback_exact_chain_verified'], len(feedback_chains))
    eq('feedback audit full history count', feedback_audit['summary']['actual_full_history_writer_requests'], len(feedback_chains))
    eq('feedback final exact chain count', final['feedback']['exact_previous_training_feedback_chains_verified'], len(feedback_chains))
    eq('feedback final history count', final['feedback']['full_history_writer_requests_verified'], len(feedback_chains))
    eq('feedback final planner count', final['feedback']['actual_planner_requests'], roles['planner'])
    eq('feedback final realizer count', final['feedback']['actual_realizer_requests'], roles['realizer'])
    eq('feedback execution not inferred', final['feedback']['presence_proves_model_followed_feedback'], False)

    cards = read(RUN / 'dataset_cards_result.json')
    eq('accepted cards equal per-attempt validation', cards['accepted_cards'], extraction_accepted)
    eq('accepted card count', len(extraction_accepted), 3)
    eq('actual per-attempt rejected candidate records', len(extraction_rejected), 2)
    eq('aggregate rejected candidates', [r for r in cards['rejected_records'] if r['scope'] == 'candidate'], extraction_rejected)
    eq('aggregate missing-source record separate', len([r for r in cards['rejected_records'] if r['scope'] == 'source']), 1)
    eq('raw realizer candidate turn counts', [r['turns'] for r in raw_candidates], [14, 5, 14])
    for root in ROOTS:
        actual = root_data[root]
        actual['request_roles'] = dict(actual['request_roles'])
        row = next(r for r in final['roots'] if r['root_id'] == root)
        eq(root + ' explicit extraction request attribution', row['source_card_extraction_requests'], [1, 1, 1, 2][ROOTS.index(root)])
        stage_row = next(r for r in stage['roots'] if r['root_id'] == root)
        for key in ('source_ids', 'request_roles', 'input_turns', 'mechanically_valid_plans',
                    'plans_cleared_for_realization', 'realizer_responses', 'raw_generated_turns',
                    'model_accepted_scenes', 'model_accepted_new_turns'):
            eq(root + ' final:' + key, row[key], actual[key])
            eq(root + ' collector:' + key, stage_row[key], actual[key])
        route = next(r for r in report['routes'] if r['root_id'] == root)
        eq(root + ' report accepted scenes', route['accepted_scenes'], actual['model_accepted_scenes'])
        eq(root + ' report added turns', route['added_turns'], actual['model_accepted_new_turns'])
        eq(root + ' cumulative count', actual['cumulative_turns'], actual['input_turns'] + actual['model_accepted_new_turns'])

    ready_roots = [r['root_id'] for r in decisions['roots'] if r['ready_for_joint_qa']]
    eq('READY roots computed from individual decisions', ready_roots, decisions['ready_roots'])
    eq('READY roots actually zero', ready_roots, [])
    eq('READY new turns from individuals', sum(r['ready_new_turn_count'] for r in decisions['roots']), 0)
    eq('READY new turns aggregate', decisions['ready_new_turns'], 0)
    for row in decisions['roots']:
        root = row['root_id']
        eq(root + ' decisions accepted turns', row['model_accepted_new_turns'], root_data[root]['model_accepted_new_turns'])
        final_row = next(r for r in final['roots'] if r['root_id'] == root)
        eq(root + ' final independent status', final_row['independent_status'], row['independent_status'])
        eq(root + ' final READY turns', final_row['independent_ready_new_turns'], row['ready_new_turn_count'])
        if root == 'cc-train-episode-9557':
            source_path = Path(row['materialized_source']['path'])
            eq('materialized sole accepted source byte copy', source_path.read_bytes(), (RUN / 'roots' / root / 'B/cumulative_dialogue.json').read_bytes())
            review = read(BASE / 'source_reviews/root9557_REVIEW.json')
            eq('formal source review exact binding', review['source'], row['materialized_source'])
            eq('formal source review full turn count', review['complete_turn_count'], 106)
            eq('formal source review full read', review['full_source_read'], True)
            eq('formal source review nonblind disclosure', review['fresh_blind_review'], False)
            eq('sole materialized decision HOLD', row['independent_status'], 'HOLD')
        else:
            eq(root + ' no fabricated materialized source', row['materialized_source'], None)
            eq(root + ' no fabricated source review', (BASE / 'source_reviews' / ('root' + root.rsplit('-', 1)[1] + '_REVIEW.json')).exists(), False)
    eq('source count exactly one', len(list((BASE / 'sources').glob('*.json'))), 1)
    eq('formal source review count exactly one', len(list((BASE / 'source_reviews').glob('*.json'))), 1)
    nested_bindings(decisions, 'READY decision binding')

    spec = read(BASE / 'inputs/SHARD_SPEC.json')
    plans = read(BASE / 'inputs/SCENE_PLANS.json')
    skip = read(BASE / 'evidence/QA_SKIPPED_NO_READY_SOURCE.json')
    qa_audit = read(BASE / 'evidence/ACTUAL_QA_INPUT_AUDIT.json')
    eq('empty actual shard specification', spec['roots'], [])
    eq('empty actual scene plans', plans['plans'], [])
    eq('skip status', skip['status'], 'SKIPPED')
    eq('skip QA calls zero', skip['qa_http_calls'], 0)
    eq('skip Solver calls zero', skip['solver_calls'], 0)
    eq('skip training prohibited', skip['training_allowed'], False)
    eq('actual QA audit status', qa_audit['status'], 'SKIPPED_NO_READY_SOURCE')
    eq('actual QA audit calls', qa_audit['audit_http_calls'], 0)
    nested_bindings(spec, 'shard specification binding')
    nested_bindings(skip, 'QA skip binding')
    nested_bindings(qa_audit, 'actual QA audit binding')
    forbidden_paths = [BASE / p for p in ('shards', 'solver', 'training', 'qa_engine/runs', 'qa_engine/requests')]
    eq('no QA shard Solver training output locations', [str(p) for p in forbidden_paths if p.exists()], [])
    qa_artifacts = [str(p.relative_to(BASE)) for p in (BASE / 'qa_engine').rglob('*') if p.is_file() and p.suffix != '.py']
    eq('QA engine contains source only', qa_artifacts, [])
    all_requests = sorted(str(p.parent.relative_to(BASE)) for p in BASE.rglob('REQUEST.json') if 'execution_code' not in p.parts)
    eq('all delivery actual REQUEST files belong to expansion run', all_requests,
       sorted(str(p.relative_to(BASE)) for p in directories))
    eq('actual expansion question calls', report['question_calls'], 0)
    eq('actual expansion Solver calls', report['solver_calls'], 0)
    eq('actual expansion training', report['training_started'], False)

    before = read(RUN / 'INVENTORY_BEFORE.json')
    reservations = read(RUN / 'SOURCE_RESERVATIONS.json')['reservations']
    before_keys = {identity(u) for u in before['accepted_uses'] + before['reserved_uses']}
    new_keys = {identity(u) for u in reservations}
    inventory = {'occupied_identities_before': len(before_keys),
                 'occupied_identities_after': len(before_keys | new_keys),
                 'newly_permanently_reserved_identities': len(new_keys - before_keys),
                 'new_assignment_overlap': len(new_keys & before_keys),
                 'historical_conflicting_identities_preserved': len(before['historical_conflicts'])}
    eq('new reservations exact assignment', [(r['root_id'], r['source_id']) for r in reservations], list(zip(ROOTS, SOURCES)))
    for key, value in inventory.items():
        eq('final inventory:' + key, final['source_inventory'][key], value)
    eq('historical global reuse not waived', final['source_inventory']['historical_global_uniqueness_status'], 'FAIL_HISTORICAL_REUSE_CONFIRMED_NOT_WAIVED')
    totals = {'assigned_roots': len(ROOTS), 'card_extraction_requests': roles['dataset_extractor'],
              'usable_cards': len(extraction_accepted),
              'roots_with_planner_requests': sum(bool(r['request_roles'].get('planner')) for r in root_data.values()),
              'roots_with_plans_cleared_for_realization': sum(bool(r['plans_cleared_for_realization']) for r in root_data.values()),
              'roots_with_realizer_requests': sum(bool(r['realizer_responses']) for r in root_data.values()),
              'roots_with_generated_turns': sum(bool(r['raw_generated_turns']) for r in root_data.values()),
              'raw_generated_candidates': len(raw_candidates),
              'raw_generated_turns_across_all_attempts': sum(r['turns'] for r in raw_candidates),
              'model_accepted_scenes': sum(r['model_accepted_scenes'] for r in root_data.values()),
              'model_accepted_new_turns': sum(r['model_accepted_new_turns'] for r in root_data.values()),
              'http_attempts': len(http_attempt_numbers), 'http_max': manifest['http_max'],
              'question_calls': 0, 'solver_calls': 0}
    for key, value in totals.items():
        eq('final total:' + key, final['totals'][key], value)
        eq('collector total:' + key, stage['totals'][key], value)
    eq('final complete historical count', final['historical_source_turns'], sum(r['input_turns'] for r in root_data.values()))
    eq('final logical request count', final['transport']['logical_requests'], len(directories))
    eq('report logical request count', report['http_request_reservations'], len(directories))
    eq('report HTTP attempt count', report['http_attempts'], len(http_attempt_numbers))
    for key in ('independent_ready_roots', 'independent_ready_scenes', 'independent_ready_new_turns',
                'generated_multiselect_questions', 'generated_ordering_questions', 'qa_http_calls',
                'new_selected_candidates', 'fresh_gold6_controls', 'training_jobs'):
        eq('final zero:' + key, final['totals'][key], 0)
    eq('final Gold all zero', final['totals']['ms_gold_distribution'], {str(n): 0 for n in range(2, 7)})
    eq('final both Solver models zero', final['totals']['solver_calls_per_model'], {'M0': 0, 'M_clean_policy': 0})
    eq('final formal training prohibited', final['totals']['formal_training_admission'], False)
    for key in ('new_event_semantic_reliability_certified', 'ordering_live_generation_tested',
                'solver_comparison_available', 'production_stability_certified'):
        eq('honest final limitation:' + key, final['validation'][key], False)
    nested_bindings(final, 'final result evidence binding')
    eq('final run status', final['status'], report['status'])
    eq('final semantic certification absent', report['semantic_certified'], False)

    return {'status': 'PASS_ACCOUNTING_AND_BINDINGS_ONLY' if not failures else 'FAIL',
            'directly_recomputed_totals': totals, 'historical_source_turns': 415,
            'roots': list(root_data.values()), 'roles': dict(roles),
            'transport': {'logical_requests': len(directories), 'http_attempts': len(http_attempt_numbers),
                          'http_statuses': dict(statuses), 'finish_reasons': dict(finish),
                          'provider_usage': dict(usage), 'prompt_token_range': [min(prompt_tokens), max(prompt_tokens)],
                          'completion_token_range': [min(output_tokens), max(output_tokens)],
                          'same_payload_retry': retry_rows},
            'card_validation': {'accepted_cards': len(extraction_accepted),
                                'per_attempt_rejected_candidate_records': len(extraction_rejected),
                                'aggregate_rejected_records': len(cards['rejected_records']),
                                'aggregate_extra_record_is_missing_source_accounting': True},
            'feedback_direct_exact_chains': feedback_chains,
            'inventory_saved_artifact_recount': inventory,
            'protected_old_files_rehashed': len(preservation['files']),
            'frozen_code_binding_counts': code_counts,
            'execution_snapshot_files_rehashed': len(snapshot['files']),
            'prepared_manifest_files_rehashed': len(manifest['files']),
            'independent_ready_roots': ready_roots, 'qa_calls': 0, 'solver_calls': 0,
            'training_started': False, 'requests': request_rows}


if __name__ == '__main__':
    result = {'schema_version': 'h4.v97.independent_final_result_audit.v1',
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'method': 'Independent stdlib-only raw artifact recomputation; no project modules imported.',
              'model_calls': 0, 'production_edits': False,
              'semantic_certification': False,
              'limits': ['PASS certifies saved accounting and exact file bindings only; independent semantic decisions are bound, not mechanically proven.',
                         'No current-round QA private Gold or Solver output was read. Initial auditor incorrectly read historical compiled candidate metadata while assuming it was dialogue; initial failure and script are preserved. Independent earlier source decisions predate that read. Protected v95/v96 files were only SHA256 hashed.',
                         'Zero downstream work is established by the saved counters, empty actual specification, explicit skip chain and absence of local run artifacts; unrecorded external activity cannot be certified.',
                         'The complete Chinese report is bound at audit time; semantic prose checking belongs to the separate second-reader review.']}
    try:
        result.update(main())
    except Exception as exc:
        result['status'] = 'FAIL_AUDITOR_EXCEPTION'
        failures.append({'check': 'auditor completed', 'status': 'FAIL',
                         'exception_type': type(exc).__name__, 'message': str(exc)})
    result['initial_auditor_failure'] = {'path': str(BASE / 'evidence/INDEPENDENT_FINAL_RESULT_AUDIT_INITIAL_FAILURE.json'), 'sha256': sha(BASE / 'evidence/INDEPENDENT_FINAL_RESULT_AUDIT_INITIAL_FAILURE.json'), 'cause': 'Auditor compared dialogue to historical candidate metadata and used compiled source_lock instead of actual frozen run feedback/source_lock; these are auditor errors, not production failures.', 'initial_script': {'path': str(BASE / 'evidence/independent_final_result_audit_INITIAL_FAILURE.py'), 'sha256': sha(BASE / 'evidence/independent_final_result_audit_INITIAL_FAILURE.py')}}
    result.update(check_count=len(checks), failure_count=len(failures), failures=failures,
                  checks=checks, file_bindings_verified=binding_records,
                  audit_script={'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)},
                  final_result={'path': str(BASE / 'RESULTS.json'), 'sha256': sha(BASE / 'RESULTS.json')},
                  final_chinese_report={'path': str(BASE / 'COMPLETE_RESULTS_ZH.md'), 'sha256': sha(BASE / 'COMPLETE_RESULTS_ZH.md')})
    with OUT.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({'status': result['status'], 'checks': len(checks),
                      'failure_names': [f['check'] for f in failures], 'output': str(OUT), 'sha256': sha(OUT)}, ensure_ascii=False))
    raise SystemExit(0 if not failures else 1)
