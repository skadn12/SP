"""Read-only v55 collection audit. Model calls and input mutations are forbidden."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CODE = HERE.parent / 'h4_sequential_ab_v55_qa'
sys.path.insert(0, str(CODE))
import generate as gen
import batch
import evaluate_and_pool as pipeline


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


def generation_audit(plan):
    roots, requests, completions, count = [], [], [], Counter()
    for row in plan['roots']:
        run, root = Path(row['generation_run']), row['root_id']
        check(bind(row['history']['path']) == row['history'], 'history changed')
        check(bind(row['prior']['path']) == row['prior'], 'explicit prior changed')
        gen.verify(run)
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
            attempt = read(folder / 'HTTP_ATTEMPT.json') if (folder / 'HTTP_ATTEMPT.json').exists() else None
            check(proof['payload_sha256'] == sha(payload), 'prepared payload binding failed')
            if attempt is not None:
                check(proof['payload_sha256'] == attempt['payload_sha256'], 'actual HTTP binding failed')
                check(attempt['actual_payload_equals_frozen'] and attempt['transport_retries'] == 0, 'actual HTTP changed or retried')
            gen.verify_prompt_request(run, payload, proof)
            text = payload['messages'][0]['content']
            marker = '\nINPUT_JSON\n' if proof['role'] == 'question_generator' else '\nDATA:\n'
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
                check(context == expected_context, 'unrecognized generator context or cross-type live feedback')
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
                check(set(context) == {'candidates', 'root_id', 'source_turns'}, 'unexpected review context')
                allowed = {'candidate_id', 'chronology_kind', 'items', 'question', 'question_type', 'root_id'}
                check(all(set(q) == allowed for q in context['candidates']), 'review sees Gold, evidence or judgments')
                if success:
                    review = read(folder / 'REVIEW.json')
                    per_root['ordering_review_requests'] += 1
                    per_root['ordering_review_accepted'] += len(review['accepted'])
                    per_root['ordering_review_rejected'] += len(review['rejected'])
                entry['review_public_only_no_gold_evidence_or_prior_judgment'] = True
            per_root['prepared_requests'] += 1
            requests.append(entry)
        report = read(run / 'REPORT.json')
        mechanical = read(run / 'MECHANICAL_POOL.json')['questions']
        pool = read(run / 'SOLVER_POOL.json')['questions']
        check(all(2 <= len(q['gold']) <= 6 for q in mechanical if q['question_type'] == 'multi_select'),
              'accepted MC violates user 2-to-6 contract')
        check(per_root['http_requests'] == report['http_attempts'] and len(mechanical) == report['mechanical_pass'] and len(pool) == report['ready_for_solver'], 'generation count mismatch')
        per_root.update(candidate_slots=report['candidate_slots'], mechanical_pass=len(mechanical), ready_for_solver=len(pool))
        final_rejections = [z for outcome in report['roots'] for z in outcome['mechanical_rejected']]
        per_root['final_mechanical_or_dedup_rejected'] = len(final_rejections)
        per_root['exact_statement_set_duplicates'] = sum(any('duplicate_' in error for error in z['errors']) for z in final_rejections)
        count.update(per_root)
        roots.append({'root_id': root, 'chunk': row['chunk'], 'generation_run': str(run),
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
        check(started > completion_time > generation['last_generation_http_completion_epoch'], 'Solver started before all generation completed')
        pipeline.verify(run)
        manifest, done = read(solver / 'manifest.json'), read(solver / 'COMPLETE.json')
        check(manifest['training_started'] is False and manifest['training_promotion'] is False, 'training started')
        records = list((solver / 'responses').glob('*.json'))
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
                      'solver_started_after_all_generation': True,
                      'selected_mc_correct_counts': [len(q['gold']) for q in views if q['question_type'] == 'multi_select'],
                      'selected_order_display_gold': [q['gold'] for q in views if q['question_type'] == 'ordering'],
                      'selected_all_six': sum(q['question_type'] == 'multi_select' and len(q['gold']) == 6 for q in views),
                      'selected_order_identity': sum(q['question_type'] == 'ordering' and q['gold'] == list(q['items']) for q in views)})
    return {'status': 'PASS', 'roots': roots, 'all_generation_before_any_solver': True,
            'latest_collection_or_supplement_generation_complete_epoch': completion_time,
            'full_solver_calls': sum(x['full_solver_calls'] for x in roots), 'selected': sum(x['selected'] for x in roots)}


def verify_file_binding(row):
    path = Path(row['path'])
    check(path.is_file() and sha(path.read_bytes()) == row['sha256'], 'frozen file changed: ' + str(path))
    if 'size_bytes' in row:
        check(path.stat().st_size == row['size_bytes'], 'frozen size changed: ' + str(path))


def historical_audit():
    correction = HERE.parent / 'h4_cardinality_correction_v55/RESULT_BINDINGS.json'
    old = HERE.parent / 'h4_question_rule_validation_v54/RESULT_BINDINGS.json'
    corrected_bindings = read(correction)['files']
    old_bound = next(row for row in corrected_bindings if Path(row['path']) == old)
    verify_file_binding(old_bound)
    for row in corrected_bindings:
        verify_file_binding(row)
    previous = read(old)
    check(previous['file_count'] == len(previous['files']) == 520, 'expected v54 historical520 binding set changed')
    for row in previous['files']:
        verify_file_binding(row)
    return {'status': 'PASS', 'correction_bindings': bind(correction),
            'correction_files_unchanged': len(corrected_bindings),
            'v54_bindings': bind(old), 'v54_files_unchanged': len(previous['files'])}


def prior_audit(plan):
    rows = []
    for row in plan['roots']:
        verify_file_binding(row['prior'])
        expected, seen, originals = [], set(), 0
        inventory = read(row['prior']['path'])

        def add(question):
            nonlocal originals
            originals += 1
            check(question['root_id'] == row['root_id'], 'wrong root in public prior')
            field = 'options' if question['question_type'] == 'multi_select' else 'items'
            key = question['question_type'], tuple(sorted(' '.join(x.split()).casefold() for x in question[field].values()))
            if key not in seen:
                seen.add(key)
                expected.append({k: question[k] for k in ('candidate_id', 'root_id', 'question_type', 'question', field)})

        for bound in inventory['source_files']:
            verify_file_binding(bound)
            for question in read(bound['path'])['questions']:
                add(question)
        for bound in inventory['historical_rejected_response_files']:
            verify_file_binding(bound)
            response = json.loads(read(bound['path'])['text'])
            check(response['root_id'] == row['root_id'], 'recovered public prior root changed')
            for draft in response['questions']:
                check(len(draft['options']) == 6 and sum(x['relation'] == 'supports' for x in draft['options']) == 5,
                      'historical rejected5-correct draft changed')
                add({'candidate_id': draft['candidate_id'], 'root_id': row['root_id'],
                     'question_type': 'multi_select', 'question': draft['question'],
                     'options': dict(zip('ABCDEF', [x['proposition'] for x in draft['options']]))})
        check(inventory['questions'] == expected, 'public historical projection mismatch')
        check(inventory['original_rows'] == originals and inventory['unique_whole_sets'] == len(expected)
              and row['public_prior_rows'] == len(expected), 'public prior counts mismatch')
        expected_files = inventory['source_files'] + inventory['historical_rejected_response_files']
        check(row['prior_source_files'] == expected_files, 'prior source bindings mismatch')
        for q in expected:
            field = 'options' if q['question_type'] == 'multi_select' else 'items'
            check(set(q) == {'candidate_id', 'root_id', 'question_type', 'question', field}, 'prior includes private metadata')
        rows.append({'root_id': row['root_id'], 'source_files': len(expected_files),
                     'original_rows': originals, 'unique_whole_sets': len(expected),
                     'only_public_unchanged_text_no_gold_evidence_or_solver': True})
    return {'status': 'PASS', 'roots': rows}


def flatten_plan(plan):
    chunks = []
    for row in plan['roots']:
        run = Path(row['batch_run'])
        batch.verify(run)
        config = read(run / 'CONFIG.json')
        check(config['budget']['candidate_slots'] == 20 and config['chunk_count'] == 5,
              'bounded20-slot collection configuration changed')
        for i in range(1, 6):
            generation_run, evaluation_run = batch.chunk_paths(run, i)
            batch.verify_chunk_generation(run, i)
            chunks.append({**row, 'chunk': i, 'generation_run': str(generation_run),
                           'evaluation_run': str(evaluation_run)})
    return {**plan, 'roots': chunks}


def combined_audit(plan, evaluation):
    roots = []
    total_before = 0
    for row in plan['roots']:
        run = Path(row['batch_run'])
        batch.verify(run)
        pool = run / 'training_pool'
        report = read(pool / 'REPORT.json')
        records = [json.loads(line) for line in (pool / 'TRAINING_POOL.jsonl').read_text().splitlines() if line.strip()]
        views = [json.loads(line) for line in (pool / 'DISPLAY_CANDIDATES.jsonl').read_text().splitlines() if line.strip()]
        check(len(records) == len(views) == report['new_unique_added'], 'combined selected count mismatch')
        for record, view in zip(records, views):
            independent_display_check(record['original_selected_question'], view)
            check(record['training_allowed'] is False and record['training_started'] is False,
                  'combined pool promoted to training')
        check(all(2 <= len(view['gold']) <= 6 for view in views if view['question_type'] == 'multi_select'),
              'combined selected MC outside2-6')
        expected = sum(chunk['selected'] for chunk in evaluation['roots'] if chunk['root_id'] == row['root_id'])
        check(expected == report['selected_before_exact_dedup'], 'chunk-to-combined selected count mismatch')
        check(expected == len(records) + len(report['duplicates']), 'combined dedup accounting mismatch')
        total_before += expected
        roots.append({'root_id': row['root_id'], 'selected': len(records), 'selected_before_dedup': expected,
                      'exact_set_duplicates': len(report['duplicates']), 'full_solver_calls': report['full_solver_calls'],
                      'display_stats': report['display_export']['stats'], 'combined_display_gold_replayed': True})
    check(total_before == evaluation['selected'], 'total evaluation selected mismatch')
    return {'status': 'PASS', 'roots': roots, 'selected': sum(row['selected'] for row in roots),
            'full_solver_calls': sum(row['full_solver_calls'] for row in roots),
            'individual_all_six_allowed': True, 'fixed_cardinality_distribution_required': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['generation', 'evaluation'], required=True)
    args = parser.parse_args()
    plan = read(HERE / 'EXPERIMENT_PLAN.json')
    prepared = read(HERE / 'preparation.COMPLETE.json')
    verify_file_binding(prepared['experiment_plan'])
    for key in ('generation_code', 'orchestrator', 'capacity_check'):
        verify_file_binding(plan[key])
    check(plan['candidate_slots'] == 40 and plan['remote_http_max'] == 60 and plan['solver_full_calls_max'] == 600,
          'authorized40-slot run budget changed')
    generation_complete = read(HERE / 'generation.COMPLETE.json')
    verify_file_binding(generation_complete['experiment_plan'])
    for row in generation_complete['roots']:
        verify_file_binding(row['completion'])
        for log in row['logs']:
            verify_file_binding(log)
    result = {'audited_at_utc': datetime.now(timezone.utc).isoformat(), 'audit_script': bind(__file__),
              'phase': args.phase, 'history': historical_audit(), 'public_priors': prior_audit(plan),
              'models_called_by_audit': 0, 'old_artifact_mutations': 0,
              'semantic_correctness_certified': False}
    flat = flatten_plan(plan)
    result['generation'] = generation_audit(flat)
    counts = result['generation']['counts']
    check(counts['candidate_slots'] == 40 and counts['http_requests'] <= 60, 'generation budget accounting mismatch')
    check(counts['http_requests'] == sum(row['qwen_http_attempts'] for row in generation_complete['roots']),
          'root generation HTTP totals mismatch')
    if args.phase == 'evaluation':
        evaluation_complete = read(HERE / 'evaluation.COMPLETE.json')
        verify_file_binding(evaluation_complete['experiment_plan'])
        verify_file_binding(evaluation_complete['generation_completion'])
        for row in evaluation_complete['roots']:
            for key in ('completion', 'pool_report'):
                verify_file_binding(row[key])
            for log in row['logs']:
                verify_file_binding(log)
        result['evaluation'] = evaluation_audit(flat, result['generation'])
        result['combined'] = combined_audit(plan, result['evaluation'])
        check(result['evaluation']['full_solver_calls'] <= 600, 'Solver call budget exceeded')
    name = 'PROVENANCE_AUDIT' if args.phase == 'evaluation' else 'PROVENANCE_GENERATION_AUDIT'
    result['status'] = 'PASS'
    (HERE / (name + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# v55 collection provenance audit', '',
             'Status: PASS. Audit makes no model calls and changes no input artifacts.', '',
             '- All actual HTTP requests preserve the entire frozen source and exact previous-round feedback.',
             '- Historical and within-batch public priors contain unchanged statement text and no Gold, evidence, or Solver results.',
             '- Mechanical repairs may include only the same failed slot’s own draft Gold; this is not Solver feedback.',
             '- Ordering reviewers receive public questions and full dialogue without writer Gold or evidence.',
             '- Accepted multi-select questions allow 2–6 correct options. Individual all-six questions remain allowed.',
             '- Actual calls and local capacity blocks are counted separately; spacing is verified on actual calls.',
             '- All520 frozen v54 files and all correction bindings are unchanged.', '',
             'Generation counts: ' + json.dumps(counts, ensure_ascii=False),
             'Minimum completion-to-start gap: ' + str(result['generation']['minimum_completion_to_start_gap_seconds']) + ' seconds.']
    if 'evaluation' in result:
        lines += ['', 'Every collection Solver call began after both roots finished generation. Full dialogue, prompt hashes, content-Gold mappings, and selected display shuffles replayed.',
                  'Solver calls: ' + str(result['evaluation']['full_solver_calls']) + '; final selected after combined dedup: ' + str(result['combined']['selected']) + '.']
    lines += ['', 'These checks establish provenance, counts, and presentation; they do not certify semantic correctness. No training is started by this workflow.']
    (HERE / (name + '.md')).write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': result['status'], 'phase': args.phase, 'counts': counts,
                      'selected': result.get('combined', {}).get('selected'),
                      'report': str(HERE / (name + '.json'))}, ensure_ascii=False))


if __name__ == '__main__':
    main()
