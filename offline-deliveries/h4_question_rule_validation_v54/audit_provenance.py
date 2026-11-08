"""Read-only audit of this bounded live validation; never invokes a model."""
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
CODE = HERE.parent / 'h4_sequential_ab_v54_qa'
sys.path.insert(0, str(CODE))
import generate as gen


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
        public_rows, used_sets = [], set()
        for source_file in row['prior_source_files']:
            check(bind(source_file['path']) == source_file, 'public prior source file changed')
            for q in read(source_file['path'])['questions']:
                field = 'options' if q['question_type'] == 'multi_select' else 'items'
                key = q['question_type'], tuple(sorted(' '.join(x.split()).casefold() for x in q[field].values()))
                if key in used_sets:
                    continue
                used_sets.add(key)
                public_rows.append({k: q[k] for k in ('candidate_id', 'root_id', 'question_type', 'question', field)})
        check(read(row['prior']['path'])['questions'] == public_rows, 'public historical pool projection changed')
        check(row['public_prior_rows'] == len(public_rows), 'public historical prior count mismatch')
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
        for folder in sorted((run / 'requests').iterdir()):
            payload, proof = read(folder / 'REQUEST.json'), read(folder / 'PROOF.json')
            attempt = read(folder / 'HTTP_ATTEMPT.json')
            check(proof['payload_sha256'] == attempt['payload_sha256'] == sha(payload), 'actual HTTP binding failed')
            check(attempt['actual_payload_equals_frozen'] and attempt['transport_retries'] == 0, 'actual HTTP changed or retried')
            gen.verify_prompt_request(run, payload, proof)
            text = payload['messages'][0]['content']
            marker = '\nINPUT_JSON\n' if proof['role'] == 'question_generator' else '\nDATA:\n'
            context = json.JSONDecoder().raw_decode(text.split(marker, 1)[1])[0]
            check(context['source_turns'] == src['turns'], 'actual HTTP full source mismatch')
            check(proof['source_sha256'] == sha((run / 'sources' / (root + '.json')).read_bytes()), 'source receipt mismatch')
            response, result, throttle = read(folder / 'RESPONSE.json'), read(folder / 'HTTP_RESULT.json'), read(folder / 'THROTTLE.json')
            check(result['outcome'] == 'response' and response['metadata']['finish_reason'] == 'stop', 'incomplete HTTP response')
            previous = throttle['previous_completion']
            gap = stamp(throttle['http_started_at_utc']) - previous['completed_at_epoch'] if previous else None
            check(gap is None or gap >= 35, 'completion-to-next-start gap below 35s')
            entry = {'request': bind(folder / 'REQUEST.json'), 'role': proof['role'], 'source_turns': len(src['turns']),
                     'full_source_equal': True, 'feedback_receipt_verified': True, 'actual_http_payload_verified': True,
                     'completion_to_start_gap_seconds': gap, 'completed_at_utc': result['http_completed_at_utc']}
            completions.append(stamp(result['http_completed_at_utc']))
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
                validation = read(folder / 'VALIDATION.json')
                raw = json.loads(response['text'])
                raw_questions = raw.get('questions', [])
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
                    if q['candidate_id'] in accepted_ids:
                        values = q.get('options') or q.get('items')
                        if isinstance(values, list):
                            values = [x.get('proposition', '') if isinstance(x, dict) else str(x) for x in values]
                        accepted[kind].append({'candidate_id': q['candidate_id'], 'question_type': kind, 'question': q['question'],
                                               'facts': list(values.values()) if isinstance(values, dict) else list(values)})
                last_validation[kind] = validation
            else:
                check(set(context) == {'candidates', 'root_id', 'source_turns'}, 'unexpected review context')
                allowed = {'candidate_id', 'chronology_kind', 'items', 'question', 'question_type', 'root_id'}
                check(all(set(q) == allowed for q in context['candidates']), 'review sees Gold, evidence or judgments')
                review = read(folder / 'REVIEW.json')
                per_root['ordering_review_requests'] += 1
                per_root['ordering_review_accepted'] += len(review['accepted'])
                per_root['ordering_review_rejected'] += len(review['rejected'])
                entry['review_public_only_no_gold_evidence_or_prior_judgment'] = True
            per_root['http_requests'] += 1
            requests.append(entry)
        report = read(run / 'REPORT.json')
        mechanical = read(run / 'MECHANICAL_POOL.json')['questions']
        pool = read(run / 'SOLVER_POOL.json')['questions']
        check(per_root['http_requests'] == report['http_attempts'] and len(mechanical) == report['mechanical_pass'] and len(pool) == report['ready_for_solver'], 'generation count mismatch')
        per_root.update(candidate_slots=report['candidate_slots'], mechanical_pass=len(mechanical), ready_for_solver=len(pool))
        final_rejections = [z for outcome in report['roots'] for z in outcome['mechanical_rejected']]
        per_root['final_mechanical_or_dedup_rejected'] = len(final_rejections)
        per_root['exact_statement_set_duplicates'] = sum(any('duplicate_' in error for error in z['errors']) for z in final_rejections)
        count.update(per_root)
        roots.append({'root_id': root, 'source_turns': len(src['turns']), 'counts': dict(per_root),
                      'final_failures': [{'question_type': x['question_type'],
                                          'mechanical_rejected': [{'candidate_id': z['candidate_id'], 'errors': z['errors']} for z in x['mechanical_rejected']],
                                          'review_rejected': x['review_rejected'], 'abstentions': x['abstentions']} for x in report['roots']],
                      'mechanical_mc_correct_counts': [len(q['gold']) for q in mechanical if q['question_type'] == 'multi_select'],
                      'mechanical_order_stems': [q['question'] for q in mechanical if q['question_type'] == 'ordering']})
    return {'status': 'PASS', 'counts': dict(count), 'roots': roots, 'requests': requests,
            'minimum_completion_to_start_gap_seconds': min(x['completion_to_start_gap_seconds'] for x in requests if x['completion_to_start_gap_seconds'] is not None),
            'last_generation_http_completion_epoch': max(completions),
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
            roots.append({'root_id': row['root_id'], 'full_solver_calls': 0, 'selected': 0,
                          'solver_status': 'EMPTY_POOL_NO_SOLVER', 'all_generation_before_any_solver': 'vacuously_true_no_solver',
                          'selected_display_permutations_and_gold_replayed': 'vacuously_true_empty',
                          'selected_mc_correct_counts': [], 'selected_order_display_gold': [],
                          'selected_all_six': 0, 'selected_order_identity': 0})
            continue
        started = stamp(read(solver / 'STARTED.json')['started_at_utc'])
        check(started > completion_time > generation['last_generation_http_completion_epoch'], 'Solver started before all generation completed')
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
            check(sha(prompt.read_bytes()) == request['prompt_sha256'], 'actual saved Solver prompt changed')
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
        roots.append({'root_id': row['root_id'], 'full_solver_calls': len(records), 'selected': len(canon),
                      'solver_prompt_hashes_verified': len(prompts), 'solver_order_identity_display_calls': identity_orders,
                      'solver_gold_content_and_display_mapping_preserved': True, 'selected_display_permutations_and_gold_replayed': True,
                      'solver_started_after_all_generation': True,
                      'selected_mc_correct_counts': [len(q['gold']) for q in views if q['question_type'] == 'multi_select'],
                      'selected_order_display_gold': [q['gold'] for q in views if q['question_type'] == 'ordering'],
                      'selected_all_six': sum(q['question_type'] == 'multi_select' and len(q['gold']) == 6 for q in views),
                      'selected_order_identity': sum(q['question_type'] == 'ordering' and q['gold'] == list(q['items']) for q in views)})
    return {'status': 'PASS', 'roots': roots, 'all_generation_before_any_solver': True,
            'full_solver_calls': sum(x['full_solver_calls'] for x in roots), 'selected': sum(x['selected'] for x in roots)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=['generation', 'evaluation'], required=True)
    phase = parser.parse_args().phase
    plan = read(HERE / 'EXPERIMENT_PLAN.json')
    check(bind(plan['generation_code']['path']) == plan['generation_code'] and bind(plan['orchestrator']['path']) == plan['orchestrator'], 'implementation changed during live validation')
    result = {'audited_at_utc': datetime.now(timezone.utc).isoformat(), 'audit_script': bind(__file__), 'phase': phase,
              'generation': generation_audit(plan), 'models_called_by_audit': 0, 'old_artifact_mutations': 0}
    if phase == 'evaluation':
        result['evaluation'] = evaluation_audit(plan, result['generation'])
    name = 'PROVENANCE_AUDIT' if phase == 'evaluation' else 'PROVENANCE_GENERATION_AUDIT'
    (HERE / (name + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# v54 live provenance audit', '', 'Status: PASS. Audit makes no model calls and changes no input artifacts.', '',
             '- Every actual HTTP payload contains the complete frozen root source and exact previous-round feedback.',
             '- Public priors preserve all public fact statements and exclude Gold/evidence/Solver output.',
             '- Mechanical repairs may contain the same failed slot’s own original draft Gold; no Solver has run yet.',
             '- Independent ordering reviewers see public questions and full source without writer Gold/evidence.',
             '- All actual HTTP payload bindings and completion-to-next-start intervals replay successfully.', '',
             'Generation counts: ' + json.dumps(result['generation']['counts']),
             'Minimum remote completion-to-start interval: ' + str(result['generation']['minimum_completion_to_start_gap_seconds']) + ' seconds.']
    if 'evaluation' in result:
        lines += ['', 'All generation completed before any Solver started. Every saved Solver prompt hash, Solver content-to-Gold mapping, and selected human display permutation was verified.',
                  'Solver calls: ' + str(result['evaluation']['full_solver_calls']) + '; selected: ' + str(result['evaluation']['selected']) + '.']
    lines += ['', 'These are provenance/display checks, not semantic certification. The source and selected questions remain provisional; no training was started.']
    (HERE / (name + '.md')).write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': 'PASS', 'phase': phase, 'counts': result['generation']['counts'],
                      'evaluation': result.get('evaluation'), 'report': str(HERE / (name + '.json'))}, ensure_ascii=False))


if __name__ == '__main__':
    main()
