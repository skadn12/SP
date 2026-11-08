"""Offline replay/export of the completed, frozen v48 single-current probe.

Run only after evaluate_and_pool.py finalize. No live calls or frozen-file edits.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import json
import re
import shutil
import sys

sys.dont_write_bytecode = True
B = Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_sequential_ab_v48_qa')
G = B / 'runs/root104814_conditions_01_generation'
E = B / 'runs/root104814_conditions_01_evaluation'
O = B / 'results/root104814_conditions_01'
ROOT = 'cc-train-episode-104814'
sys.path.insert(0, str(B))
import evaluate_and_pool as p


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return p.base.digest(Path(path).read_bytes())


def main():
    assert (E / 'training_pool/MANIFEST.json').is_file(), 'finalization_not_complete'
    O.mkdir(parents=True, exist_ok=True)
    old = read(B / 'BASELINE_FILES.json')['files']
    assert len(old) == 2253
    p.base.verify_bindings(old)
    outer = p.verify(E)
    manifest, questions, provenance, inspector = p.verify_generation(G)
    assert manifest['prompt_arm'] == 'current'
    assert manifest['roots'] == [ROOT]
    assert manifest['candidate_slots_per_type_per_root'] == 2
    source_path = Path(manifest['source_rows'][0]['source_path'])
    source = read(source_path)
    upstream_source = read(manifest['candidate_history_path'])
    assert source['turns'] == upstream_source['turns'] and len(source['turns']) == 156
    assert sha(source_path) == 'a09fc90df11a350a4526b385c941baa32e8ada2849c1f671ca44a48e89ffecc8'
    assert sha(manifest['candidate_history_path']) == 'f5139178dae5b23c8c1d31f5be13a82a3d867bf0adfc755fbe09c708c4985928'
    prior = read(G / 'prior' / (ROOT + '.json'))
    frozen_input_check = read(O / 'INDEPENDENT_FROZEN_INPUT_CHECKS.json')
    previous_generation = B.parent / 'h4_sequential_ab_v47_qa/runs/root104814_evidence_first_01_generation'
    previous_prior = read(previous_generation / 'prior' / (ROOT + '.json'))
    previous_selected_pool = (B.parent / 'h4_sequential_ab_v47_qa/runs/'
                              'root104814_evidence_first_01_evaluation/training_pool/TRAINING_POOL.jsonl')
    added_prior = p.generation._prior_pool_rows(previous_selected_pool, ROOT)
    assert len(previous_prior) == 62 and len(added_prior) == 3
    assert prior == previous_prior + added_prior and len(prior) == 65
    assert (G / 'sources' / (ROOT + '.json')).read_bytes() == (previous_generation / 'sources' / (ROOT + '.json')).read_bytes()
    added_prompt_prior = p.generation.prompt_prior(added_prior)
    forbidden_prior_fields = {'gold', 'score', 'scores', 'training_selection', 'selected_by_policy',
                              'full_partial_mean', 'full_exact_rate', 'answers', 'solver_answers',
                              'option_evidence', 'item_evidence', 'relation', 'rationale'}
    def check_public_only(value):
        if isinstance(value, dict):
            assert not (forbidden_prior_fields & set(value)), 'private_prior_field_leaked'
            for child in value.values():
                check_public_only(child)
        elif isinstance(value, list):
            for child in value:
                check_public_only(child)
    check_public_only(added_prior)
    check_public_only(added_prompt_prior)
    assert frozen_input_check['source_byte_identical'] is True
    assert frozen_input_check['source_turn_count'] == 156
    assert frozen_input_check['prior_prefix_equal'] is True
    assert frozen_input_check['prior_old_count'] == 62 and frozen_input_check['prior_new_count'] == 65
    assert frozen_input_check['added_prior_ids'] == [row['candidate_id'] for row in added_prior]
    assert frozen_input_check['added_prior_public_content_only'] is True
    feedback_path = G / 'feedback' / ROOT / 'writer__M_clean_policy_feedback.txt'
    feedback = feedback_path.read_text()
    feedback_hash = sha(feedback_path)
    assert feedback_hash == '883fac0b4daaf6d78768826f06c023330f41edaab528e0a5d027f0e721b7dda8'
    source_lock = read(G / 'feedback/source_lock.json')
    assert source_lock['status'] == 'EXPLICIT_PREVIOUS_TRAINING_ROUND_ONLY'
    assert source_lock['merge_current_generation_feedback'] is False
    test_log = Path('/tmp/v48_qa_all_tests.log').read_text()
    match = re.search(r'Ran (\d+) tests in ', test_log)
    assert match and int(match.group(1)) == 35 and '\nOK\n' in test_log
    test_count = int(match.group(1))

    if questions:
        results = read(E / 'solver/RESULTS.json')
        solver_manifest, records, _ = inspector.solver(E / 'solver')
        selected = p.verify_scores(questions, solver_manifest, records, results)
        assert read(E / 'solver/COMPLETE.json')['actual_call_count'] == len(records)
        assert read(E / 'delivery/SELECTED_QUESTIONS.json')['questions'] == selected
    else:
        results, records, selected = {'questions': [], 'selected_questions': []}, [], []
    generation_report = read(G / 'REPORT.json')
    pool_report = read(E / 'training_pool/REPORT.json')
    mechanical = read(G / 'MECHANICAL_POOL.json')['questions']
    expected_context = {kind: p.generation.initial_context(G, ROOT, kind) for kind in p.generation.TYPES}
    accepted = {kind: [] for kind in p.generation.TYPES}
    last_rejected = {kind: [] for kind in p.generation.TYPES}
    all_abstentions = {kind: [] for kind in p.generation.TYPES}
    review_accepted, review_rejected = [], []
    calls, replays, first, all_drafts, diagnostics = [], [], {}, [], []
    initial_matches = {}
    turn_by_ref = {turn['turn_ref']: turn for turn in source['turns']}

    for directory in sorted((G / 'requests').iterdir()):
        request, proof = read(directory / 'REQUEST.json'), read(directory / 'PROOF.json')
        p.generation.verify_prompt_request(G, request, proof)
        marker = '\nINPUT_JSON\n' if proof['role'] == 'question_generator' else '\nDATA:\n'
        context = p._input_json(request, marker)
        assert context['source_turns'] == source['turns']
        text = request['messages'][0]['content']
        assert text.count(feedback) == 1
        call = {'request': directory.name, 'role': proof['role'], 'source_turns': 156,
                'actual_payload_verified': True, 'feedback_verified': True,
                'request_payload_sha256': p.base.digest(request), 'feedback_sha256': feedback_hash}
        if (directory / 'HTTP_ATTEMPT.json').exists():
            inspector.feedback_request(directory / 'REQUEST.json', G, ROOT)
            throttle, outcome = read(directory / 'THROTTLE.json'), read(directory / 'HTTP_RESULT.json')
            call.update(started=throttle['http_started_at_utc'], completed=outcome['http_completed_at_utc'],
                        http_status=outcome.get('http_status'), outcome=outcome.get('outcome'))
            previous = throttle.get('previous_completion')
            if previous:
                marker_gap = datetime.fromisoformat(call['started']).timestamp() - previous['completed_at_epoch']
                assert marker_gap >= 35
                call['previous_completion_marker_gap_seconds'] = marker_gap
        if (directory / 'ERROR.json').exists():
            call['error'] = read(directory / 'ERROR.json')
        if proof['role'] == 'question_generator':
            kind = proof['question_type']
            assert context == expected_context[kind], 'generation_context_changed:' + directory.name
            rebuilt, rebuilt_receipt = p.generation.make_payload(G, ROOT, kind, context)
            assert request == rebuilt and proof['feedback_receipt'] == rebuilt_receipt
            if directory.name.endswith('generate-1'):
                snapshot = read(O / 'initial' / (kind + '.json'))
                assert request == snapshot['payload'] and proof['feedback_receipt'] == snapshot['receipt']
                assert frozen_input_check[kind]['snapshot_sha256'] == sha(O / 'initial' / (kind + '.json'))
                assert frozen_input_check[kind]['actual_payload_sha256'] == p.base.digest(request)
                assert frozen_input_check[kind]['initial_input_equal_except_candidate_ids_and_prior_inventory'] is True
                assert frozen_input_check[kind]['api_parameters_unchanged'] is True
                assert frozen_input_check[kind]['feedback_identical_and_once'] is True
                old_sequence = ('001-104814-merged-multi-generate-1' if kind == 'multi_select'
                                else '002-104814-merged-order-generate-1')
                previous_request = read(previous_generation / 'requests' / old_sequence / 'REQUEST.json')
                previous_context = p._input_json(previous_request, '\nINPUT_JSON\n')
                shared_context = lambda value: {key: item for key, item in value.items()
                    if key not in {'candidate_ids', 'previous_candidates_dedup_only'}}
                assert shared_context(context) == shared_context(previous_context)
                assert context['previous_candidates_dedup_only'] == p.generation.prompt_prior(prior)
                assert previous_context['previous_candidates_dedup_only'] == p.generation.prompt_prior(previous_prior)
                assert {key: value for key, value in request.items() if key != 'messages'} == {
                    key: value for key, value in previous_request.items() if key != 'messages'}
                initial_matches[kind] = {'actual_equals_snapshot': True,
                    'payload_sha256': p.base.digest(request), 'snapshot_sha256': sha(O / 'initial' / (kind + '.json')),
                    'previous_context_equal_except_ids_and_prior': True,
                    'new_prior_rows_public_text_only': 3}
        else:
            candidates = [q for q in mechanical if q['candidate_id'] == proof['candidate_id']]
            assert len(candidates) == 1
            rebuilt, rebuilt_receipt = p.generation.make_review_payload(G, ROOT, candidates[0], source)
            assert request == rebuilt and proof['feedback_receipt'] == rebuilt_receipt
        if not (directory / 'RESPONSE.json').exists():
            assert (directory / 'ERROR.json').exists()
            if proof['role'] == 'question_generator':
                last_rejected[kind] = [{'candidate_id': qid, 'errors': [call['error']['error_type']]}
                                      for qid in context['candidate_ids']]
            else:
                review_rejected.append({'candidate_id': proof['candidate_id'], 'reasons': ['review_request_failed']})
            calls.append(call)
            continue
        response = read(directory / 'RESPONSE.json')
        metadata = response['metadata']
        call['usage'] = {key: metadata.get(key) for key in ('input_tokens', 'output_tokens', 'total_tokens')}
        call['finish_reason'] = metadata.get('finish_reason')
        assert metadata['requested_model'] == metadata['returned_model'] == p.generation.MODEL
        if proof['role'] == 'question_generator':
            valid, rejected, abstained, error = p.generation.legacy.validate_draft(
                response['text'], context['candidate_ids'], ROOT, kind, source, source_path)
            observed = read(directory / 'VALIDATION.json')
            assert observed['accepted_ids'] == [q['candidate_id'] for q in valid]
            assert observed['rejected'] == rejected and observed['abstentions'] == abstained
            assert observed['envelope_error'] == error
            accepted[kind] += valid
            last_rejected[kind] = rejected
            all_abstentions[kind] += abstained
            replays.append({'request': directory.name, 'raw_validation_replayed': True})
            if directory.name.endswith('generate-1'):
                first[kind] = {'mechanical_valid_before_dedup': len(valid),
                    'rejected': [{'candidate_id': row['candidate_id'], 'errors': row['errors']} for row in rejected],
                    'abstentions': abstained}
            if rejected:
                repair = {'instruction': 'Repair only listed mechanical/source-binding errors; preserve accepted slots.',
                          'failures': p.generation.legacy.repair_failures(rejected)}
                if error:
                    repair['envelope_error'] = error
                expected_context[kind] = {**context, 'candidate_ids': [row['candidate_id'] for row in rejected],
                    'mechanical_repair': repair,
                    'previous_candidates_dedup_only': p.generation.prompt_prior(
                        prior + p.generation.legacy.compact_questions(accepted[kind]))}
            try:
                raw = json.loads(response['text'])
            except ValueError:
                raw = {'unparseable_response': response['text']}
            all_drafts.append({'request': directory.name, 'question_type': kind,
                'response_sha256': sha(directory / 'RESPONSE.json'), 'raw_text': response['text'], 'raw': raw})
            for row in raw.get('questions', []):
                if not isinstance(row, dict):
                    continue
                details = []
                rows = list(zip('ABCDEF', row.get('options', []))) if kind == 'multi_select' else list(row.get('item_evidence', {}).items())
                for label, item in rows:
                    turn = turn_by_ref.get(item.get('turn_ref'), {})
                    quote = item.get('quote', '')
                    keys = list(item)
                    evidence_first = (all(key in keys for key in ('turn_ref','speaker','quote','proposition','relation'))
                                      and max(keys.index(key) for key in ('turn_ref','speaker','quote')) < keys.index('proposition') < keys.index('relation')) if kind == 'multi_select' else None
                    details.append({'label': label, 'keys': keys, 'evidence_before_claim': evidence_first,
                        'quote_exact_contiguous': bool(quote) and quote in turn.get('text', ''),
                        'quote_occurrences': turn.get('text', '').count(quote) if quote else 0,
                        'speaker_matches': item.get('speaker') == turn.get('speaker')})
                keys = list(row)
                order_first = (all(key in keys for key in ('item_evidence','items','gold'))
                               and keys.index('item_evidence') < keys.index('items') < keys.index('gold')) if kind == 'ordering' else None
                diagnostics.append({'request': directory.name, 'candidate_id': row.get('candidate_id'),
                    'question_type': kind, 'question_key_order': keys,
                    'ordering_evidence_before_items_gold': order_first,
                    'exact_quotes': sum(d['quote_exact_contiguous'] for d in details),
                    'total_quotes': len(details), 'evidence_before_claim_count': sum(d['evidence_before_claim'] is True for d in details),
                    'details': details})
        else:
            replay = p.generation.ordering_gate.review_response(response['text'], candidates, source,
                directory / 'RESPONSE.json', directory / 'REQUEST.json', metadata)
            assert replay == read(directory / 'REVIEW.json')
            review_accepted += replay['accepted']
            review_rejected += replay['rejected']
            replays.append({'request': directory.name, 'raw_reviewer_outcome_replayed': True})
        calls.append(call)
    assert set(initial_matches) == set(p.generation.TYPES)
    current, dedup_replays = [], []
    for kind in p.generation.TYPES:
        kept, duplicates = p.v15.dedup.deduplicate(accepted[kind], prior, current)
        actual = [q for q in mechanical if q['question_type'] == kind]
        assert [q['candidate_id'] for q in kept] == [q['candidate_id'] for q in actual]
        for original, observed in zip(kept, actual):
            assert all(observed.get(key) == value for key, value in original.items())
        outcome = next(row for row in generation_report['roots'] if row['question_type'] == kind)
        assert outcome['mechanical_rejected'] == last_rejected[kind] + duplicates
        assert outcome['abstentions'] == all_abstentions[kind]
        assert outcome['mechanical_pass'] == len(kept)
        assert outcome['review_rejected'] == (review_rejected if kind == 'ordering' else [])
        current += kept
        dedup_replays.append({'question_type': kind, 'kept_ids': [q['candidate_id'] for q in kept], 'duplicates': duplicates})
    assert questions == [q for q in mechanical if q['question_type'] == 'multi_select'] + review_accepted

    # Replay final pool rows from independently replayed selections and provenance.
    entries = [{'question': q, 'source': read(q['source_path']), 'is_new': True,
        'provenance': {**provenance[q['candidate_id']], 'input_role': 'new',
            'solver_manifest': inspector.bind(E / 'solver/manifest.json'),
            'solver_results': inspector.bind(E / 'solver/RESULTS.json'),
            'solver_identity': inspector.bind(E / 'solver/solver_identity.json')}} for q in selected]
    expected_rows, expected_duplicates = p.pool_rows(entries, E / 'training_pool')
    pool_rows = [json.loads(line) for line in (E / 'training_pool/TRAINING_POOL.jsonl').read_text().splitlines() if line.strip()]
    assert pool_rows == expected_rows
    assert read(E / 'training_pool/SELECTED_QUESTIONS.json')['questions'] == [row['original_selected_question'] for row in expected_rows]
    assert pool_report['new_unique_added'] == len(expected_rows)
    assert pool_report['selected_before_exact_dedup'] == len(selected)
    assert pool_report['duplicates'] == expected_duplicates
    assert pool_report['full_solver_calls'] == len(records)
    assert len(records) == outer['full_calls_planned']
    assert all(row['full_context']['turns'] == source['turns'] for row in pool_rows)
    assert all(row['training_allowed'] is False and row['training_admitted_questions'] == 0 for row in pool_rows)
    formal_prior = p.generation.DEFAULT_PRIOR
    formal_count = len([line for line in formal_prior.read_text().splitlines() if line.strip()])
    assert formal_count == 30

    actual = sorted([call for call in calls if 'started' in call], key=lambda call: call['started'])
    gaps = [(datetime.fromisoformat(right['started']) - datetime.fromisoformat(left['completed'])).total_seconds()
            for left, right in zip(actual, actual[1:])]
    assert all(gap >= 35 for gap in gaps)
    assert len(actual) == generation_report['http_attempts'] <= manifest['budget']['http_attempts_max']
    assert manifest['automatic_transport_retries'] == 0
    usage = {key: sum((call.get('usage', {}).get(key) or 0) for call in actual)
             for key in ('input_tokens', 'output_tokens', 'total_tokens')}
    unknown_usage = [call['request'] for call in actual if any(call.get('usage', {}).get(key) is None
                     for key in ('input_tokens', 'output_tokens', 'total_tokens'))]
    transport = {'http_calls': len(actual), 'http_statuses': dict(Counter(str(call['http_status']) for call in actual)),
        'usage_known_response_total': usage, 'unknown_usage_requests': unknown_usage,
        'usage_complete': not unknown_usage, 'minimum_completion_gap_seconds': min(gaps) if gaps else None,
        'gaps_seconds': gaps, 'automatic_transport_retries': 0, 'calls': calls}
    unique_ids = {row.get('candidate_id') for draft in all_drafts for row in draft['raw'].get('questions', []) if isinstance(row, dict)}
    summary = {'experiment': 'v48_single_current_template_conditions_relations_probe', 'prompt_arm': 'current',
        'first_pass': first, 'unique_question_ids_generated': len(unique_ids),
        'raw_draft_rows_including_repair': sum(len(row['raw'].get('questions', [])) for row in all_drafts),
        'mechanical_pass_after_repairs_and_dedup': generation_report['mechanical_pass'],
        'ready_for_solver': len(questions), 'solver_full_calls': len(records),
        'scores': results['questions'], 'selected_candidate_ids': results['selected_questions'],
        'qa_selected': pool_report['new_unique_added'], 'qa_selected_by_type': pool_report['by_type'],
        'outcomes': generation_report['roots'],
        'source': {'turns': 156, 'body_tokens': 4566, 'new_dialogue_tokens_this_batch': 0,
            'retained_B_body_tokens': 894, 'body_token_count_basis': 'unchanged_hash_of_previously_audited_full_candidate_source',
            'source_sha256': sha(source_path), 'candidate_history_sha256': sha(manifest['candidate_history_path'])},
        'initial_actual_requests_match_frozen': True, 'initial_inputs_v47_equal_except_ids_and_prior': True,
        'prior_inventory': {'previous_count': len(previous_prior), 'current_count': len(prior),
            'previous_prefix_unchanged': True, 'added_count': len(added_prior),
            'added_ids': [row['candidate_id'] for row in added_prior],
            'added_public_text_only': True, 'source_pool_sha256': sha(previous_selected_pool),
            'current_prior_sha256': sha(G / 'prior' / (ROOT + '.json'))},
        'full_feedback_in_all_requests': True, 'feedback_sha256': feedback_hash,
        'formal_training_pool_questions': formal_count, 'prior_v44_selected_candidates_preserved': 2,
        'prior_v47_selected_candidates_preserved': 3,
        'training_started': False, 'training_admitted_questions': 0, 'upstream_admitted_scenes': 0,
        'offline_tests_passed': test_count, 'historical_files_unchanged': len(old), 'transport': transport,
        'raw_output_diagnostics': diagnostics,
        'independent_diagnostic_references': [
            {'path': str(O / name), 'sha256': sha(O / name), 'diagnostic_only': True}
            for name in ('INDEPENDENT_QUESTION_REVIEW.md', 'INDEPENDENT_RAW_DIAGNOSTICS.json',
                         'INDEPENDENT_REVIEWER_DIAGNOSTICS.json', 'INDEPENDENT_SOLVER_DIAGNOSTICS.json')
            if (O / name).is_file()],
        'independent_diagnostic_note': 'Read independent Solver diagnostics alongside selected scores; these observations do not change raw questions, Gold or selection policy.',
        'limitation': 'Single small probe against a preceding batch, not a simultaneous causal comparison. Output field order does not expose hidden reasoning; selection does not certify Gold.',
        'semantic_gold_certified': False}
    verification = {'full_source_in_all_requests': True, 'raw_output_gold_unchanged': True,
        'all_request_feedback_bindings_verified': True, 'initial_requests': initial_matches,
        'prior_v47_prefix_plus_exact_three_public_selected_rows_verified': True,
        'actual_requests_reconstructed': True, 'request_outcomes_replayed': replays,
        'dedup_replayed': dedup_replays, 'scores_and_final_pool_rows_replayed': True,
        'solver_full_answers_replayed': len(records), 'selected_unique_candidates': len(pool_rows),
        'historical_files_verified': len(old), 'offline_test_log_verified_count': test_count,
        'source_and_feedback_hash_verified': True, 'semantic_certified': False}
    write(O / 'RESULT_SUMMARY.json', summary)
    write(O / 'TRANSPORT_AND_USAGE.json', transport)
    write(O / 'QUESTION_DRAFTS.json', {'model_output_unchanged': True, 'raw_drafts': all_drafts})
    write(O / 'GENERATOR_OUTPUT_DIAGNOSTICS.json', {'diagnostic_only': True, 'questions': diagnostics})
    write(O / 'VERIFICATION.json', verification)
    lines = ['# Raw question drafts', '', 'Model text, evidence, relations and Gold are unchanged. Repaired drafts are shown separately. These are model outputs, not a semantic certification.', '']
    for draft in all_drafts:
        lines += ['## ' + draft['request'], '']
        for row in draft['raw'].get('questions', []):
            lines += ['### ' + row.get('candidate_id', 'Unnamed draft'), '', row.get('question', ''), '']
            if draft['question_type'] == 'multi_select':
                for label, option in zip('ABCDEF', row.get('options', [])):
                    lines += [f"- **{label}.** {option.get('proposition', '')} ({option.get('relation', '')})",
                        f"  - Source: {option.get('turn_ref', '')}; speaker: {option.get('speaker', '')}",
                        '  - Quote: ' + json.dumps(option.get('quote', ''), ensure_ascii=False),
                        '  - Rationale: ' + option.get('rationale', '')]
                lines += ['', 'Original Gold derives from the supports labels; no independent correction is applied.', '']
            else:
                for label, item in row.get('items', {}).items():
                    evidence = row.get('item_evidence', {}).get(label, {})
                    lines += [f'- **{label}.** {item}', f"  - Source: {evidence.get('turn_ref', '')}; speaker: {evidence.get('speaker', '')}",
                        '  - Quote: ' + json.dumps(evidence.get('quote', ''), ensure_ascii=False),
                        '  - Status: ' + evidence.get('status', '')]
                lines += ['', 'Writer Gold: ' + json.dumps(row.get('gold'), ensure_ascii=False), '',
                          'Order reason: ' + row.get('order_reason', ''), '']
        if draft['raw'].get('abstentions'):
            lines += ['Abstentions: ' + json.dumps(draft['raw']['abstentions'], ensure_ascii=False), '']
        if 'unparseable_response' in draft['raw']:
            lines += ['```text', draft['raw']['unparseable_response'], '```', '']
    (O / 'QUESTION_DRAFTS.md').write_text('\n'.join(lines) + '\n')
    for src, name in [('/tmp/v48_qa_all_tests.log', 'OFFLINE_TESTS.log'),
                      ('/tmp/v48_qa_live.log', 'GENERATION_STDOUT.log')]:
        shutil.copyfile(src, O / name)
    p.base.verify_bindings(old)
    p.verify(E)
    print(json.dumps({'ready_for_solver': len(questions), 'solver_full_calls': len(records),
        'qa_selected': pool_report['new_unique_added'], 'selected_by_type': pool_report['by_type'],
        'http_calls': len(actual), 'api_usage': usage, 'minimum_gap': min(gaps) if gaps else None,
        'baseline_files': len(old), 'offline_tests_passed': test_count}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
