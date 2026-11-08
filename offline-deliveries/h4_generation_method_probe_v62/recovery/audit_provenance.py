"""Offline provenance audit for the frozen direct versus staged twelve-slot v62 probe."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys

sys.dont_write_bytecode = True
RECOVERY = Path(__file__).resolve().parent
HERE = RECOVERY.parent
sys.path.insert(0, str(HERE))
import importlib.util
_spec = importlib.util.spec_from_file_location('v62_recovery_runner_for_audit', RECOVERY / 'recover_collection.py')
recovery = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recovery)
CODE = HERE.parent / 'h4_generation_method_probe_v62_qa'
sys.path.insert(0, str(CODE))
import generate as gen
import evaluate_and_pool as pipeline
import run_collection as collection

def read(path):
    return json.loads(Path(path).read_text())

def sha(value):
    data = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(data).hexdigest()

def bind(path):
    path = Path(path)
    return {'path': str(path), 'sha256': sha(path.read_bytes())}

def check(value, why):
    if not value:
        raise AssertionError(why)

def project_prior(rows):
    output = {}
    for row in rows:
        facts = row.get('facts') or []
        if not facts:
            continue
        key = row.get('question_type'), tuple(sorted(' '.join(t.casefold().split()) for t in facts))
        public = {'question_type': row.get('question_type'), 'facts': facts,
                  'dedup_action': 'avoid_exact_six_statement_set_only_paraphrases_allowed' if row.get('hard_dedup') else 'warning_only'}
        if key not in output or row.get('hard_dedup'):
            output[key] = public
    return list(output.values())

def stamp(text):
    return datetime.fromisoformat(text).timestamp()

def ordinary_generation_audit(plan):
    roots, requests, completions, count = [], [], [], Counter()
    for row in plan['roots']:
        run, root = Path(row['generation_run']), row['root_id']
        check(bind(row['history']['path']) == row['history'], 'history changed')
        check(bind(row['prior']['path']) == row['prior'], 'explicit prior changed')
        gen.verify(run)
        check(all(q['question_type'] in gen.GENERATION_TYPES for q in read(run / 'SOLVER_POOL.json')['questions']), 'unexpected question type')
        m = read(run / 'MANIFEST.json')
        src = read(run / 'sources' / (root + '.json'))
        history = read(row['history']['path'])
        check(src['turns'] == history['turns'], 'source truncated or changed')
        prior = read(run / 'prior' / (root + '.json'))
        frozen = read(run / 'feedback/source_lock.json')
        check(frozen['status'] == 'EXPLICIT_PREVIOUS_TRAINING_ROUND_ONLY' and frozen['merge_current_generation_feedback'] is False, 'feedback not frozen')
        original_feedback = HERE.parent / 'h4_groundless_random_feedback_v1/feedback/compiled_v4'
        for suffix in ['source_lock.json', root + '/writer__M_clean_policy_feedback.txt']:
            check((run / 'feedback' / suffix).read_bytes() == (original_feedback / suffix).read_bytes(), 'feedback source changed')
        per_root = Counter()
        accepted = {kind: [] for kind in gen.TYPES}
        last_validation = {}
        raw_support_histogram = Counter()
        for folder in sorted((run / 'requests').iterdir()):
            payload, proof = read(folder / 'REQUEST.json'), read(folder / 'PROOF.json')
            check(proof['role'] in ('question_generator', 'evidence_planner') and proof['question_type'] == 'multi_select', 'unexpected request role or generator type')
            check(proof['role'] != 'evidence_planner' or m['generation_method'] == 'staged', 'planner_in_direct_arm')
            check(0 < payload['max_tokens'] <= 4096, 'output token cap exceeded')
            attempt = read(folder / 'HTTP_ATTEMPT.json') if (folder / 'HTTP_ATTEMPT.json').exists() else None
            check(proof['payload_sha256'] == sha(payload), 'prepared payload binding failed')
            if attempt is not None:
                check(proof['payload_sha256'] == attempt['payload_sha256'], 'actual HTTP binding failed')
                check(attempt['actual_payload_equals_frozen'] and attempt['transport_retries'] == 0, 'actual HTTP changed or retried')
            gen.verify_prompt_request(run, payload, proof)
            text = payload['messages'][0]['content']
            marker = '\nINPUT_JSON\n'
            context = json.JSONDecoder().raw_decode(text.split(marker, 1)[1])[0]
            check(context['source_turns'] == src['turns'], 'actual HTTP full source mismatch')
            check(proof['source_sha256'] == sha((run / 'sources' / (root + '.json')).read_bytes()), 'source receipt mismatch')
            response = read(folder / 'RESPONSE.json') if (folder / 'RESPONSE.json').exists() else None
            result = read(folder / 'HTTP_RESULT.json') if attempt is not None else None
            throttle = read(folder / 'THROTTLE.json') if attempt is not None else None
            if attempt is not None:
                previous = throttle['previous_completion']
                gap = stamp(throttle['http_started_at_utc']) - previous['completed_at_epoch'] if previous else None
                check(gap is None or gap >= 35, 'completion-to-next-start gap below 35s')
                completions.append(stamp(result['http_completed_at_utc']))
                per_root['http_requests'] += 1
                if result['outcome'] == 'response':
                    check(response is not None and response['metadata']['finish_reason'] == 'stop', 'complete response missing')
                else:
                    check((folder / 'ERROR.json').exists(), 'HTTP failure lacks explicit error')
                    per_root['http_errors'] += 1
            else:
                gap = None
                error = read(folder / 'ERROR.json')
                check(error['error_type'] == 'LocalCapacityBlocked' and error['http_attempts'] == 0,
                      'request without HTTP must be explicit local capacity block')
                check(proof['capacity']['status'] != 'PASS', 'local capacity block has passing capacity')
                per_root['local_capacity_blocks'] += 1
            success = attempt is not None and result['outcome'] == 'response'
            entry = {'request': bind(folder / 'REQUEST.json'), 'role': proof['role'], 'source_turns': len(src['turns']),
                     'full_source_equal': True, 'feedback_receipt_verified': True,
                     'actual_http_attempted': attempt is not None, 'actual_http_payload_verified': attempt is not None,
                     'completion_to_start_gap_seconds': gap,
                     'completed_at_utc': result['http_completed_at_utc'] if result else None,
                     'capacity_status': proof['capacity']['status'],
                     'required_context_tokens': proof['capacity']['required_context_tokens']}
            if proof['role'] == 'question_generator':
                kind = proof['question_type']
                expected_prior = project_prior(prior + accepted[kind])
                check(context['previous_candidates_dedup_only'] == expected_prior, 'public prior not exact frozen-only / accepted-same-type projection')
                check(all(set(x) == {'question_type', 'facts', 'dedup_action'} for x in expected_prior), 'private metadata in public prior')
                expected_context = gen.initial_context(run, root, kind)
                is_repair = 'mechanical_repair' in context
                if is_repair:
                    prev = last_validation[kind]
                    expected_context['candidate_ids'] = [x['candidate_id'] for x in prev['rejected']]
                    repair = {'instruction': 'Repair only listed mechanical/source-binding errors; preserve accepted slots.',
                              'failures': gen.legacy.repair_failures(prev['rejected'])}
                    if prev['envelope_error']:
                        repair['envelope_error'] = prev['envelope_error']
                    expected_context.update(mechanical_repair=repair, previous_candidates_dedup_only=expected_prior)
                    expected_context = gen.focus_context(m, expected_context)
                compared_context = dict(context)
                if m['generation_method'] == 'staged':
                    frozen_plan = gen.verified_evidence_plan(run, root)
                    check(compared_context.pop('frozen_evidence_plan_raw_json') == frozen_plan['raw_text'], 'staged_plan_context_changed')
                    check(compared_context.pop('evidence_plan_use_instruction') == gen.PLAN_USE_INSTRUCTION, 'staged_plan_instruction_changed')
                check(compared_context == expected_context, 'unrecognized generator context or cross-type live feedback')
                expected_payload, _ = gen.make_payload(run, root, kind, expected_context)
                check(payload == expected_payload, 'actual generator text is not exact authorized reconstruction')
                if success:
                    validation = read(folder / 'VALIDATION.json')
                    try:
                        raw = json.loads(response['text'])
                    except (ValueError, TypeError):
                        check(bool(validation['envelope_error']), 'unparseable response lacks validation error')
                        raw = {}
                    raw_questions = raw.get('questions', []) if isinstance(raw, dict) else []
                    if not isinstance(raw_questions, list):
                        check(bool(validation['envelope_error']), 'malformed question list lacks validation error')
                        raw_questions = []
                    if kind == 'multi_select':
                        for draft in raw_questions:
                            if isinstance(draft, dict) and isinstance(draft.get('options'), list):
                                raw_support_histogram[sum(x.get('relation') == 'supports' for x in draft['options'] if isinstance(x, dict))] += 1
                    per_root['repair_requests' if is_repair else 'initial_requests'] += 1
                    per_root['repair_drafts' if is_repair else 'initial_drafts'] += len(raw_questions)
                    per_root['repair_accepted' if is_repair else 'initial_accepted'] += len(validation['accepted_ids'])
                    per_root['abstentions'] += len(validation['abstentions'])
                    entry.update(question_type=kind, repair=is_repair, candidate_slots=len(context['candidate_ids']),
                                 emitted_drafts=len(raw_questions), accepted_ids=validation['accepted_ids'],
                                 rejected=[{'candidate_id': x['candidate_id'], 'errors': x['errors']} for x in validation['rejected']],
                                 abstentions=validation['abstentions'], public_prior_rows=len(expected_prior), public_prior_has_no_gold_or_solver=True,
                                 same_slot_own_draft_gold_in_repair=bool(is_repair and any('gold' in x.get('draft', {}) for x in context['mechanical_repair']['failures'])))
                    accepted_ids = set(validation['accepted_ids'])
                    for q in raw_questions:
                        if isinstance(q, dict) and q.get('candidate_id') in accepted_ids:
                            values = q.get('options') or q.get('items')
                            if isinstance(values, list):
                                values = [x.get('proposition', '') if isinstance(x, dict) else str(x) for x in values]
                            accepted[kind].append({'candidate_id': q['candidate_id'], 'question_type': kind, 'question': q['question'],
                                                   'facts': list(values.values()) if isinstance(values, dict) else list(values)})
                    last_validation[kind] = validation
                else:
                    entry.update(question_type=kind, repair=is_repair, candidate_slots=len(context['candidate_ids']),
                                 public_prior_rows=len(expected_prior), public_prior_has_no_gold_or_solver=True,
                                 error=read(folder / 'ERROR.json')['error_type'])

            else:
                expected_context = gen.initial_context(run, root, 'multi_select')
                check(context == expected_context, 'planner_initial_context_changed')
                expected_payload, _ = gen.make_planner_payload(run, root)
                check(payload == expected_payload, 'planner_payload_not_exact_reconstruction')
                per_root['evidence_plan_requests'] += int(attempt is not None)
                entry['planner_full_context_and_frozen_prior_verified'] = True
                # No question validation or Solver data belongs to the evidence planner.
                per_root['prepared_requests'] += 1
                requests.append(entry)
                continue
            per_root['prepared_requests'] += 1
            requests.append(entry)
        report = read(run / 'REPORT.json')
        mechanical = read(run / 'MECHANICAL_POOL.json')['questions']
        pool = read(run / 'SOLVER_POOL.json')['questions']
        check(all(2 <= len(q['gold']) <= 6 for q in mechanical if q['question_type'] == 'multi_select'),
              'accepted MC violates user 2-to-6 contract')
        check(per_root['http_requests'] == report['http_attempts'] and len(mechanical) == report['mechanical_pass'] and len(pool) == report['ready_for_solver'], 'generation count mismatch')
        check(per_root['http_requests'] <= (2 if m['generation_method'] == 'direct' else 3), 'per-arm chunk HTTP limit exceeded')
        check(per_root['evidence_plan_requests'] <= int(m['generation_method'] == 'staged'), 'planner count limit exceeded')
        per_root.update(candidate_slots=report['candidate_slots'], mechanical_pass=len(mechanical), ready_for_solver=len(pool))
        final_rejections = [z for outcome in report['roots'] for z in outcome['mechanical_rejected']]
        per_root['final_mechanical_or_dedup_rejected'] = len(final_rejections)
        per_root['exact_statement_set_duplicates'] = sum(any('duplicate_' in error for error in z['errors']) for z in final_rejections)
        count.update(per_root)
        roots.append({'root_id': root, 'chunk': row['chunk'], 'arm': m['generation_method'], 'generation_run': str(run),
                      'raw_mc_support_count_histogram': dict(sorted(raw_support_histogram.items())),
                      'source_turns': len(src['turns']), 'counts': dict(per_root),
                      'final_failures': [{'question_type': x['question_type'],
                                          'mechanical_rejected': [{'candidate_id': z['candidate_id'], 'errors': z['errors']} for z in x['mechanical_rejected']],
                                          'review_rejected': x['review_rejected'], 'abstentions': x['abstentions']} for x in report['roots']],
                      'mechanical_mc_correct_counts': [len(q['gold']) for q in mechanical if q['question_type'] == 'multi_select'],
                      'mechanical_order_stems': [q['question'] for q in mechanical if q['question_type'] == 'ordering']})
    return {'status': 'PASS', 'counts': dict(count), 'roots': roots, 'requests': requests,
            'minimum_completion_to_start_gap_seconds': min((x['completion_to_start_gap_seconds'] for x in requests if x['completion_to_start_gap_seconds'] is not None), default=None),
            'last_generation_http_completion_epoch': max(completions, default=0),
            'repair_note': 'Same rejected slot may include its own generator draft Gold for mechanical repair; this is not Solver feedback. All public priors and independent reviewer projections exclude Gold/evidence/Solver data.'}

def interrupted_generation_audit(row):
    frozen_plan, receipt = recovery.verify()
    first, run, folder = recovery.interrupted_identity(frozen_plan)
    check(row['generation_run'] == str(run) and row['chunk'] == 'direct_c01', 'wrong interrupted chunk')
    observation = read(RECOVERY / 'INTERRUPTED_ATTEMPT.json')
    check(observation['remote_outcome'] == 'unknown' and observation['remote_completion_at_utc'] is None
          and observation['remote_error_observed'] is False, 'unknown outcome improperly resolved')
    check(observation['replayed_http_calls'] == observation['replacement_slots'] == 0, 'lost request replayed')
    root = row['root_id']
    m = gen.verify(run)
    source = read(run / 'sources' / (root + '.json'))
    history = read(row['history']['path'])
    check(bind(row['history']['path']) == row['history'] and bind(row['prior']['path']) == row['prior'], 'lost source/prior changed')
    check(source['turns'] == history['turns'], 'lost request source changed')
    lock = read(run / 'feedback/source_lock.json')
    check(lock['status'] == 'EXPLICIT_PREVIOUS_TRAINING_ROUND_ONLY' and lock['merge_current_generation_feedback'] is False, 'lost feedback not frozen')
    original_feedback = HERE.parent / 'h4_groundless_random_feedback_v1/feedback/compiled_v4'
    for suffix in ['source_lock.json', root + '/writer__M_clean_policy_feedback.txt']:
        check((run / 'feedback' / suffix).read_bytes() == (original_feedback / suffix).read_bytes(), 'lost feedback changed')
    payload, proof = read(folder / 'REQUEST.json'), read(folder / 'PROOF.json')
    attempt, throttle = read(folder / 'HTTP_ATTEMPT.json'), read(folder / 'THROTTLE.json')
    check(proof['role'] == 'question_generator' and proof['question_type'] == 'multi_select', 'lost request role changed')
    check(0 < payload['max_tokens'] <= 4096, 'lost output cap changed')
    check(proof['payload_sha256'] == attempt['payload_sha256'] == sha(payload), 'lost request binding changed')
    check(attempt['actual_payload_equals_frozen'] is True and attempt['transport_retries'] == 0, 'lost request replayed or changed')
    gen.verify_prompt_request(run, payload, proof)
    context = json.JSONDecoder().raw_decode(payload['messages'][0]['content'].split('\nINPUT_JSON\n', 1)[1])[0]
    expected = gen.initial_context(run, root, 'multi_select')
    check(context == expected and context['source_turns'] == source['turns'], 'lost initial context changed')
    check(proof['source_sha256'] == sha((run / 'sources' / (root + '.json')).read_bytes()), 'lost source hash changed')
    check(context['previous_candidates_dedup_only'] == project_prior(read(run / 'prior' / (root + '.json'))), 'lost public prior changed')
    previous = throttle['previous_completion']
    gap = stamp(throttle['http_started_at_utc']) - previous['completed_at_epoch'] if previous else None
    check(gap is None or gap >= 35, 'lost request preceding known gap failed')
    report = read(run / 'REPORT.json')
    check(report['remote_error_requests'] == 0 and report['unobserved_response_requests'] == 1, 'interruption mislabeled provider failure')
    check(report['request_costs'] == gen.request_costs(run), 'lost unknown usage changed')
    outcome = report['roots'][0]
    check(outcome['mechanical_rejected'] == outcome['abstentions'] == outcome['review_rejected'] == [], 'interruption mislabeled content outcome')
    check([x['candidate_id'] for x in outcome['interrupted_slots']] == first['candidate_ids'], 'lost slot ids changed')
    check(all(x['mechanical_validation_performed'] is False and x['observed_draft'] is False for x in outcome['interrupted_slots']), 'lost slots claim observed draft')
    check(pipeline.verify_generation(run)[1] == [], 'lost chunk has candidates')
    for name in recovery.MISSING: check(not (folder / name).exists(), 'fabricated lost request result:' + name)
    counts = {'http_requests': 1, 'prepared_requests': 1, 'candidate_slots': 2,
              'mechanical_pass': 0, 'ready_for_solver': 0, 'interrupted_http_requests': 1,
              'interrupted_unobserved_slots': 2, 'observed_response_slots': 0,
              'http_errors': 0, 'initial_requests': 1, 'initial_drafts': 0,
              'repair_requests': 0, 'repair_drafts': 0, 'evidence_plan_requests': 0}
    entry = {'request': bind(folder / 'REQUEST.json'), 'role': proof['role'], 'question_type': 'multi_select',
             'source_turns': len(source['turns']), 'full_source_equal': True,
             'feedback_receipt_verified': True, 'actual_http_attempted': True,
             'actual_http_payload_verified': True, 'completion_to_start_gap_seconds': gap,
             'completed_at_utc': None, 'remote_outcome': 'unknown', 'remote_error_observed': False,
             'interruption': bind(RECOVERY / 'INTERRUPTED_ATTEMPT.json'),
             'capacity_status': proof['capacity']['status'], 'required_context_tokens': proof['capacity']['required_context_tokens'],
             'token_usage': 'unknown', 'emitted_drafts': None, 'locally_saved_drafts': 0,
             'public_prior_rows': len(context['previous_candidates_dedup_only']), 'public_prior_has_no_gold_or_solver': True}
    root_record = {'root_id': root, 'chunk': 'direct_c01', 'arm': 'direct', 'generation_run': str(run),
             'source_turns': len(source['turns']), 'counts': counts, 'raw_mc_support_count_histogram': {},
             'final_failures': [], 'interrupted_unobserved_slots': outcome['interrupted_slots'],
             'mechanical_mc_correct_counts': [], 'mechanical_order_stems': []}
    return counts, entry, root_record


def generation_audit(plan):
    lost = [r for r in plan['roots'] if r['chunk'] == 'direct_c01']
    intact = [r for r in plan['roots'] if r['chunk'] != 'direct_c01']
    check(len(lost) == 1 and len(intact) == 5, 'recovery audit requires original six chunks')
    result = ordinary_generation_audit({**plan, 'roots': intact})
    counts, request, root = interrupted_generation_audit(lost[0])
    combined = Counter(result['counts']); combined.update(counts)
    result['counts'] = dict(combined)
    result['requests'].insert(0, request); result['roots'].insert(0, root)
    result['minimum_completion_to_start_gap_seconds'] = min((x['completion_to_start_gap_seconds'] for x in result['requests'] if x['completion_to_start_gap_seconds'] is not None), default=None)
    cooldown = read(RECOVERY / 'LOCAL_RESUME_COOLDOWN.json')
    check(cooldown['basis'] == 'local_recovery_observation_only' and cooldown['remote_completion_gap_verified'] is False, 'recovery cooldown misrepresented')
    check(cooldown['waited_seconds'] >= 35 or (cooldown['observed_interval_seconds'] is not None and cooldown['observed_interval_seconds'] >= 35), 'local recovery cooldown insufficient')
    result.update(execution_recovery=bind(RECOVERY / 'RECOVERY_PLAN.json'),
        interrupted_request_remote_completion_unknown=True,
        completion_gap_after_interrupted_request_verifiable=False,
        local_recovery_cooldown=bind(RECOVERY / 'LOCAL_RESUME_COOLDOWN.json'),
        ordinary_completion_markers_used_only_for_observed_requests=True,
        semantic_quality_not_inferred_from_execution_loss=True)
    return result


def independent_display_check(q, view):
    kind = q['question_type']; field = 'options' if kind == 'multi_select' else 'items'
    labels = list('ABCDEF') if kind == 'multi_select' else ['I' + str(i) for i in range(1, 7)]
    public = {'candidate_id': q['candidate_id'], 'question_type': kind, 'question': q['question'], 'statements': sorted(q[field].values()), 'seed': view['display_seed']}
    texts = sorted(q[field].values()); random.Random(int(sha(public), 16)).shuffle(texts)
    expected = dict(zip(labels, texts)); check(view[field] == expected, 'display is not replayable Gold-blind permutation')
    original_to_display = {old: next(k for k, v in expected.items() if v == text) for old, text in q[field].items()}
    mapped = [original_to_display[k] for k in q['gold']]
    if kind == 'multi_select':
        mapped.sort(key=labels.index)
    check(view['gold'] == mapped and view['original_to_display'] == original_to_display, 'display changed proposition truth / order')
    check(view['canonical_question_sha256'] == sha(q), 'canonical question hash mismatch')

def evaluation_audit(plan, generation):
    complete = read(HERE / 'generation.COMPLETE.json')
    completion_time = stamp(complete['completed_at_utc'])
    review = collection.verify_review()
    review_time = stamp(review['completed_at_utc'])
    check(review_time > completion_time, 'content review freeze predates generation completion')
    supplement_complete = HERE / 'supplement/generation.COMPLETE.json'
    if (HERE / 'supplement/MANIFEST.json').exists():
        check(supplement_complete.is_file(), 'capacity supplement generation not complete')
        completion_time = max(completion_time, stamp(read(supplement_complete)['completed_at_utc']))
    roots = []
    for row in plan['roots']:
        run = Path(row['evaluation_run']); solver = run / 'solver'
        questions = {q['candidate_id']: q for q in read(Path(row['generation_run']) / 'SOLVER_POOL.json')['questions']}
        if not questions:
            skipped = read(solver / 'SKIPPED.json')
            check(skipped['status'] == 'EMPTY_POOL_NO_SOLVER' and skipped['candidate_count'] == 0,
                  'empty Solver pool must be explicitly skipped')
            check(not (solver / 'STARTED.json').exists() and not list((solver / 'responses').glob('*.json')),
                  'empty pool unexpectedly called Solver')
            pool = run / 'training_pool'
            check(not (pool / 'TRAINING_POOL.jsonl').read_text().strip()
                  and not (pool / 'DISPLAY_CANDIDATES.jsonl').read_text().strip(), 'empty pool exported questions')
            roots.append({'root_id': row['root_id'], 'chunk': row['chunk'], 'full_solver_calls': 0, 'selected': 0,
                          'solver_status': 'EMPTY_POOL_NO_SOLVER', 'all_generation_before_any_solver': 'vacuously_true_no_solver',
                          'selected_display_permutations_and_gold_replayed': 'vacuously_true_empty',
                          'selected_mc_correct_counts': [], 'selected_order_display_gold': [],
                          'selected_all_six': 0, 'selected_order_identity': 0})
            continue
        started = stamp(read(solver / 'STARTED.json')['started_at_utc'])
        check(started > review_time > completion_time > generation['last_generation_http_completion_epoch'], 'Solver started before generation or independent content review completed')
        pipeline.verify(run)
        manifest, done = read(solver / 'manifest.json'), read(solver / 'COMPLETE.json')
        check(manifest['training_started'] is False and manifest['training_promotion'] is False, 'training started')
        records = list((solver / 'responses').glob('*.json'))
        pipeline.verify_scores(list(questions.values()), manifest,
            [read(path) for path in records], read(solver / 'RESULTS.json'))
        check(len(records) == done['actual_call_count'] == len(manifest['requests']), 'Solver incomplete')
        prompts = []
        identity_orders = 0
        gold = read(solver / 'gold.json')
        canonical = {q['candidate_id']: q for q in read(solver / 'canonical_bindings.json')['candidates']}
        for request in manifest['requests']:
            q = questions[request['candidate_id']]; binding = canonical[q['candidate_id']]
            check(gold[q['candidate_id']] == [binding['content_id_by_label'][label] for label in q['gold']], 'Solver Gold content changed')
            prompt = solver / 'prompts' / (request['request_id'] + '.txt')
            prompt_text = prompt.read_text()
            check(sha(prompt.read_bytes()) == request['prompt_sha256'], 'actual saved Solver prompt changed')
            source = read(Path(q['source_path']))
            expected_dialogue = '\n'.join('[' + turn['turn_ref'] + '] ' + turn['speaker'] + ': ' + turn['text'] for turn in source['turns'])
            check('Dialogue:\n' + expected_dialogue in prompt_text, 'Solver prompt does not contain full unchanged dialogue')
            check(request['mode'] == 'full_context', 'unexpected groundless Solver call')
            field = 'options' if q['question_type'] == 'multi_select' else 'items'
            variant = request['variant']
            check(all(variant[field][label] == variant['content_to_text'][cid] for label, cid in variant['display_to_content'].items()), 'Solver display content mapping invalid')
            check(set(variant['content_to_text'].values()) == set(q[field].values()), 'Solver proposition text changed')
            if q['question_type'] == 'ordering':
                inverse = {cid: label for label, cid in variant['display_to_content'].items()}
                identity_orders += [inverse[cid] for cid in gold[q['candidate_id']]] == list(variant[field])
            prompts.append(bind(prompt))
        pool = run / 'training_pool'
        canon = [json.loads(x) for x in (pool / 'TRAINING_POOL.jsonl').read_text().splitlines() if x.strip()]
        views = [json.loads(x) for x in (pool / 'DISPLAY_CANDIDATES.jsonl').read_text().splitlines() if x.strip()]
        check(len(canon) == len(views), 'selected display count mismatch')
        for record, view in zip(canon, views):
            independent_display_check(record['original_selected_question'], view)
        roots.append({'root_id': row['root_id'], 'chunk': row['chunk'], 'full_solver_calls': len(records), 'selected': len(canon),
                      'solver_prompt_hashes_verified': len(prompts), 'solver_order_identity_display_calls': identity_orders,
                      'solver_gold_content_and_display_mapping_preserved': True, 'selected_display_permutations_and_gold_replayed': True,
                      'solver_started_after_local_collection_completion': True,
                      'interrupted_remote_completion_known': False,
                      'selected_mc_correct_counts': [len(q['gold']) for q in views if q['question_type'] == 'multi_select'],
                      'selected_order_display_gold': [q['gold'] for q in views if q['question_type'] == 'ordering'],
                      'selected_all_six': sum(q['question_type'] == 'multi_select' and len(q['gold']) == 6 for q in views),
                      'selected_order_identity': sum(q['question_type'] == 'ordering' and q['gold'] == list(q['items']) for q in views)})
    return {'status': 'PASS', 'roots': roots, 'all_generation_before_any_solver': None,
            'all_observed_generation_before_any_solver': True,
            'all_local_collection_accounting_before_any_solver': True,
            'interrupted_request_remote_completion_known': False,
            'latest_collection_or_supplement_generation_complete_epoch': completion_time,
            'content_review_complete_epoch': review_time, 'all_content_review_before_any_solver': True,
            'full_solver_calls': sum(x['full_solver_calls'] for x in roots), 'selected': sum(x['selected'] for x in roots)}

def verify_file_binding(row):
    path = Path(row['path'])
    check(path.is_file() and sha(path.read_bytes()) == row['sha256'], 'frozen file changed: ' + str(path))
    if 'size_bytes' in row:
        check(path.stat().st_size == row['size_bytes'], 'frozen size changed: ' + str(path))


def historical_audit():
    previous = Path(collection.CONFIG["previous_result_bindings"])
    rows = read(previous)["files"]
    for row in rows:
        verify_file_binding(row)
    baseline = read(CODE / "BASELINE_FILES.json")["files"]
    for row in baseline:
        verify_file_binding(row)
    snapshot = collection.verify_copy_snapshot()
    return {"status": "PASS", "previous_result_bindings": bind(previous),
            "previous_bound_files_unchanged": len(rows),
            "original_code_files_unchanged": len(baseline), "copy_snapshot": snapshot}


def public_inventory(path, root):
    inventory = read(path)
    expected, seen, count = [], set(), 0
    for bound in inventory['source_files']:
        verify_file_binding(bound)
        value = read(bound['path'])
        rows = value if isinstance(value, list) else value.get('questions', value.get('candidates', []))
        for question in rows:
            if question.get('root_id') != root:
                continue
            count += 1
            public = collection.public_question(question)
            key = collection.whole_set_key(public)
            if key not in seen:
                seen.add(key)
                expected.append(public)
    check(inventory['questions'] == expected, 'public inventory text/Gold projection changed: ' + str(path))
    check(inventory['original_rows'] == count and inventory['unique_whole_sets'] == len(expected),
          'public inventory counts changed: ' + str(path))
    for row in expected:
        field = 'options' if row['question_type'] == 'multi_select' else 'items'
        check(set(row) == {'candidate_id', 'root_id', 'question_type', 'question', field},
              'private metadata leaked into public inventory')
    return {'path': str(path), 'original_rows': count, 'unique_whole_sets': len(expected),
            'source_files': inventory['source_files'], 'only_unchanged_public_text': True}


def flatten_plan(plan):
    root = plan['root_id']
    rows, audits = [], [public_inventory(Path(plan['historical_prior']['path']), root)]
    for chunk in plan['chunks']:
        index = chunk['index']
        prior = collection.cumulative_prior_path(index)
        check(bind(prior) == plan['historical_prior'], 'all arms must share one frozen preexperiment prior')
        manifest = gen.verify(Path(chunk['generation_run']))
        check(manifest['slot_focus_by_candidate_id'] == chunk['slot_focus_by_candidate_id'], 'frozen focus changed')
        check(manifest['generation_method'] == chunk['arm'], 'frozen generation method changed')
        check(manifest['question_types'] == ['multi_select'] and manifest['budget']['candidate_slots'] == 2,
              'two-multiselect-only chunk contract changed')
        rows.append({'root_id': root, 'chunk': chunk['chunk_name'], 'arm': chunk['arm'],
                     'history': plan['history'], 'prior': bind(prior),
                     'generation_run': chunk['generation_run'], 'evaluation_run': chunk['evaluation_run']})
    return {**plan, 'roots': rows}, {'status': 'PASS', 'inventories': audits}


def evaluation_completion():
    value = collection.completion("evaluation")
    return value, {"mode": "ordinary_completion", "reused_completed_calls": 0,
                   "repeated_solver_calls": 0, "completion": bind(HERE / "evaluation.COMPLETE.json")}


def run_audit(phase='evaluation'):
    check(phase in ('generation', 'evaluation'), 'unknown audit phase')
    adapter = read(RECOVERY / 'ADAPTER_BINDINGS.json')
    for row in adapter['files']: verify_file_binding(row)
    recovery.verify()
    plan = collection.verify_plan()
    check(plan['candidate_slots'] == 12 and plan['remote_http_max'] == 15
          and plan['solver_full_calls_max'] == 216 and len(plan['chunks']) == 6, 'authorized budget changed')
    completed = collection.completion('generation')
    verify_file_binding(completed['experiment_plan'])
    flat, priors = flatten_plan(plan)
    result = {'audited_at_utc': datetime.now(timezone.utc).isoformat(), 'audit_script': bind(__file__),
              'phase': phase, 'history': historical_audit(), 'public_priors': priors,
              'recovery_adapter_bindings': bind(RECOVERY / 'ADAPTER_BINDINGS.json'),
              'models_called_by_audit': 0, 'old_artifact_mutations': 0,
              'semantic_correctness_certified': False, 'experiment_plan': bind(HERE / 'EXPERIMENT_PLAN.json')}
    result['generation'] = generation_audit(flat)
    counts = result['generation']['counts']
    check(counts['candidate_slots'] == 12 and counts['http_requests'] <= 15
          and counts['http_requests'] == completed['http_attempts'], 'generation actual/budget mismatch')
    check(counts.get('ordering_review_requests', 0) == 0, 'ordering reviewer unexpectedly called')
    if phase == 'evaluation':
        evaluation, evaluation_recovery = evaluation_completion()
        verify_file_binding(evaluation['experiment_plan'])
        verify_file_binding(evaluation['generation_completion'])
        verify_file_binding(evaluation['content_review_completion'])
        result['evaluation'] = evaluation_audit(flat, result['generation'])
        result['evaluation_recovery'] = evaluation_recovery
        check(result['evaluation']['full_solver_calls'] <= 216
              and result['evaluation']['full_solver_calls'] == evaluation['full_solver_calls'], 'Solver actual/budget mismatch')
    check(completed.get('execution_recovery') == bind(RECOVERY / 'RECOVERY_PLAN.json'), 'collection recovery binding missing')
    check(completed.get('interrupted_HTTP_attempts') == 1 and completed.get('unresolved_interrupted_slots') == 2, 'collection interruption accounting changed')
    result['execution_recovery'] = bind(RECOVERY / 'RECOVERY_PLAN.json')
    result['status'] = 'PASS'
    return result


def export_audit(result):
    stem = "PROVENANCE_AUDIT" if result["phase"] == "evaluation" else "PROVENANCE_GENERATION_AUDIT"
    with (HERE / (stem + ".json")).open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    counts = result["generation"]["counts"]
    plan = read(HERE / "EXPERIMENT_PLAN.json")
    lines = ["# v62 生成方式对照来源与调用核验", "", "状态：PASS；离线重放，模型调用为 0。首批一请求因执行中断没有本地响应，远程结果/完成时间/usage 均 unknown；两个名额属于执行损失，不是内容或方法失败。", "",
        f"- root {plan['root_id']} 的完整 {plan['source_turn_count']} 轮对话、上一训练轮反馈和公开历史题目保持原样。",
        "- 两臂和所有批次使用同一预实验公开题目先验，不含 Gold、证据、Solver 回答或得分；当批产出不跨批传播；机械修复可含同一失败题原始草稿。",
        "- 本次仅生成多选题，没有排序审核请求；既有排序题与展示规则不变。",
        f"- 前轮 {result['history']['previous_bound_files_unchanged']} 个绑定文件保持不变。", "",
        f"出题 HTTP {counts['http_requests']} 次；最短前次完成至下次发起间隔 {result['generation']['minimum_completion_to_start_gap_seconds']} 秒。"]
    if "evaluation" in result:
        lines.extend(["", f"Solver {result['evaluation']['full_solver_calls']} 次；均晚于本地收集结案和独立内容审阅冻结完成；被中断请求的远程完成时间仍未知。",
                      "完整上下文、逐次评分、Gold 内容映射及随机展示已重放。"])
    lines.extend(["", "此核验确认来源、调用和映射；题目内容判断另见独立内容检查。中断请求之后的远程完成间隔不可验证，仅验证从真实本地恢复观察起至少35秒冷却。未启动训练。"])
    with (HERE / (stem + ".md")).open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    return HERE / (stem + ".json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('generation', 'evaluation'), default='evaluation')
    args = parser.parse_args()
    result = run_audit(args.phase)
    path = export_audit(result)
    print(json.dumps({'status': result['status'], 'phase': args.phase, 'report': str(path)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
