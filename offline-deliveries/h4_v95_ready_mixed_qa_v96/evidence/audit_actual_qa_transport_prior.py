"""Read-only replay of actual QA requests, public priors and frozen feedback."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys

BASE = Path(__file__).resolve().parents[1]
SHARD = BASE / 'shards/shard_01'
sys.path.insert(0, str(SHARD))
import common
import prior_rendering
import actor_reference
import ordering_generation

EXPECTED = [('cc-train-episode-25652', '25652', 36, 105),
            ('cc-train-episode-19708', '19708', 35, 104)]
READS = {}


def read(path):
    path = Path(path).resolve()
    READS[path] = common.bind(path)
    return common.read(path)


def text(path):
    path = Path(path).resolve()
    READS[path] = common.bind(path)
    return path.read_text()


def need(condition, reason):
    common.need(condition, reason)


def sha_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def keys(value):
    if isinstance(value, dict):
        return set(value).union(*(keys(child) for child in value.values()))
    if isinstance(value, list):
        return set().union(*(keys(child) for child in value))
    return set()


def raw_json_after(value, marker):
    offset = value.index(marker) + len(marker)
    result, length = json.JSONDecoder().raw_decode(value[offset:])
    return result, offset + length


def feedback_check(run, root, feedback, proof, request_text, context_end):
    frozen = run / 'feedback'
    suffix_path = frozen / root / 'writer__M_clean_policy_feedback.txt'
    suffix = text(suffix_path)
    need(suffix == feedback['writer_suffix'], 'actual_feedback_not_exact_previous_round_recompile')
    need(read(frozen / root / 'candidate_history.json') == feedback['candidate_history'],
         'actual_feedback_history_changed')
    need(read(frozen / root / 'aggregate.json') == feedback['aggregate'], 'actual_aggregate_changed')
    lock = read(frozen / 'source_lock.json')
    need(lock == feedback['source_lock']
         and lock['status'] == 'EXPLICIT_PREVIOUS_TRAINING_ROUND_ONLY'
         and lock['merge_current_generation_feedback'] is False
         and lock['automatic_latest_run_selection'] is False,
         'feedback_scope_not_frozen_previous_round_only')
    need(request_text[context_end:context_end + len(suffix)] == suffix
         and request_text.count(suffix) == 1, 'actual_request_feedback_missing_changed_or_repeated')
    receipt = proof['feedback_receipt']
    need(receipt['feedback_appended'] is True
         and receipt['suffix_sha256'] == sha_text(suffix)
         and receipt['source_lock_sha256'] == common.sha(frozen / 'source_lock.json')
         and receipt['request_user_text_sha256'] == sha_text(request_text[:context_end + len(suffix)])
         and receipt['root_id'] == root and receipt['target_solver_model'] == 'M_clean_policy',
         'actual_feedback_receipt_chain_changed')
    return {'suffix': common.bind(suffix_path), 'source_lock': common.bind(frozen / 'source_lock.json'),
            'candidate_history_count': len(feedback['candidate_history']),
            'exact_historical_recompile_equals_frozen_suffix': True,
            'request_contains_exact_suffix_once_after_full_context': True,
            'intermediate_text_sha256_matches': True,
            'historical_feedback_can_contain_previous_solver_scores_by_design': True,
            'current_generation_solver_feedback_used': False}


def audit():
    assets = common.load_multi('asset_generation')
    config = read(SHARD / 'CONFIG.json')
    _, _, annotate, capacity_function, development = assets.base.transport()
    tokenizer = development.tokenizer_from(str(assets.base.TOKENIZER))
    results = []
    ordering_empty_prior = []
    public_forbidden = {'gold', 'original_gold', 'correct_option', 'evidence', 'evidence_refs',
                        'evidence_groups', 'item_evidence', 'models', 'training_exposure'}
    for root, short, expected_prior_count, expected_turn_count in EXPECTED:
        feedback = assets.source_adapter.feedback_data(root)
        for record in feedback['receipt']['source_files']:
            READS[Path(record['path']).resolve()] = record
        original_pool = read(feedback['receipt']['historical_pool']['path'])
        pool_ids = {row['candidate_id'] for row in original_pool}
        need(len(pool_ids) == expected_prior_count == len(feedback['candidate_history']),
             'original_historical_count_mismatch')
        for lane, run in [
            ('multi_select', SHARD / f'multiselect/assets/root{short}_family'),
            ('ordering', SHARD / f'ordering/root{short}_ordering'),
        ]:
            manifest = read(run / 'MANIFEST.json')
            source = read(run / 'SOURCE.json')
            official_source = BASE / f'sources/root{short}_SOURCE.json'
            need(common.sha(run / 'SOURCE.json') == common.sha(official_source)
                 and len(source['turns']) == expected_turn_count
                 and source['root_id'] == root, 'actual_source_bytes_or_length_changed')
            READS[official_source] = common.bind(official_source)
            prior = read(run / 'PRIOR.json')
            if lane == 'multi_select':
                candidates = sorted(run.glob('rounds/*/requests/*'))
                need(len(candidates) == 1 and candidates[0].parent.parent.name == 'initial',
                     'unexpected_multiselect_rounds_or_http_count')
            else:
                candidates = sorted(run.glob('requests/*'))
                need(len(candidates) == 1, 'unexpected_ordering_http_count')
            request_dir = candidates[0]
            request = read(request_dir / 'REQUEST.json')
            proof = read(request_dir / 'PROOF.json')
            user_text = request['messages'][0]['content']
            context, context_end = raw_json_after(user_text, 'INPUT_JSON\n')
            need(context['source_turns'] == source['turns']
                 and context['source_metadata'] == {key: value for key, value in source.items() if key != 'turns'},
                 'actual_request_source_or_metadata_truncated')
            need(proof['payload_sha256'] == assets.base.digest(request)
                 and proof['source_sha256'] == common.sha(run / 'SOURCE.json'),
                 'request_proof_binding_mismatch')
            private_projection = {}
            if lane == 'multi_select':
                expected_raw, prior_files = assets.source_adapter.prior_candidates(
                    root, feedback, run / 'feedback' / root / 'candidate_history.json', assets.v15)
                for path in prior_files:
                    READS[Path(path).resolve()] = common.bind(path)
                need(prior == expected_raw and len(prior) == expected_prior_count
                     and {row['candidate_id'] for row in prior} == pool_ids,
                     'actual_prepared_prior_missing_historical_candidate')
                projected = assets.legacy.prompt_prior(prior)
                encoded = context['previous_candidates_dedup_only']
                need(prior_rendering.decode_prior(encoded) == projected
                     and len(projected) == expected_prior_count
                     and not (keys(encoded) & public_forbidden),
                     'actual_public_prior_projection_changed_or_private_fields_leaked')
                expected_payload, expected_proof = assets.round_payload(run, 'initial', assets.FACT_IDS)
                need(request == expected_payload
                     and all(proof.get(key) == value for key, value in expected_proof.items())
                     and read(run / 'rounds/initial/INITIAL_REQUEST.json') == request,
                     'actual_multiselect_request_not_frozen_round_replay')
                private_projection = {
                    'prepared_raw_prior_rows': len(prior),
                    'original_pool_ids_exactly_match': True,
                    'all_raw_prior_fields_equal_frozen_history_projection': True,
                    'question_type_counts': dict(Counter(row['question_type'] for row in prior)),
                    'actual_writer_public_statement_sets': len(projected),
                    'actual_writer_groups': len(encoded['groups']),
                    'all_public_rows_and_indices_roundtrip': True,
                    'public_projection_gold_or_evidence_fields': [],
                }
            else:
                slot = next(slot for slot in config['ordering_slots'] if slot['root_id'] == root)
                need(slot['prior_pool_paths'] == [] and prior == [], 'unexpected_actual_ordering_prior')
                question_index = context['previous_public_questions_dedup_only']['question_index']
                prior_text, _ = raw_json_after(user_text,
                    'DISALLOWED PRIOR WHOLE SETS (public text, deduplication only):\n')
                need(question_index == [] and prior_text == {'disallowed_prior_six_item_sets': []},
                     'actual_ordering_empty_prior_not_rendered_exactly')
                policy, _ = raw_json_after(user_text, 'ACTOR_REFERENCE_POLICY:\n')
                need(policy == read(run / 'ACTOR_REFERENCE_POLICY.json')
                     and policy['output_style'] == 'use_source_names_or_explicit_roles_for_people'
                     and policy['forbidden_gendered_pronouns'] == list(actor_reference.TOKENS)
                     and policy['pronoun_status'] == 'unknown_for_all_source_speakers'
                     and policy['automatic_repair'] is False
                     and "The standalone\nwords he, him, his, himself, she, her, hers, herself are forbidden" in user_text,
                     'actual_ordering_names_only_policy_missing')
                writer_plan = read(run / 'WRITER_PLAN.json')
                writer_records, _ = raw_json_after(user_text, 'SOURCE_ID_ANCHOR_RECORDS:\n')
                need(writer_records == writer_plan['questions']
                     and not writer_plan['display_labels_supplied']
                     and not writer_plan['gold_supplied_to_writer'],
                     'actual_ordering_writer_plan_changed')
                need(not (keys(writer_records) & {'gold', 'original_gold', 'labels', 'label', 'items', 'item_evidence'}),
                     'private_labels_or_gold_in_public_ordering_anchors')
                expected_payload, expected_proof = ordering_generation.payload_for(run)
                need(request == expected_payload and read(run / 'REQUEST.json') == request
                     and all(proof.get(key) == value for key, value in expected_proof.items()),
                     'actual_ordering_payload_replay_mismatch')
                ordering_empty_prior.append({'root_id': root, 'actual_prior_rows': 0,
                    'explicit_config_prior_pool_paths': [],
                    'legacy_prior_pool_path_ignored_when_explicit_list_present': slot['prior_pool_path'],
                    'code_reason': "ordering prepare uses slot.get('prior_pool_paths', [slot['prior_pool_path']]); explicit [] yields no prior public rows",
                    'historical_training_feedback_candidates': len(feedback['candidate_history']),
                    'historical_question_texts_not_supplied_by_training_feedback_suffix': True})
                private_projection = {'prepared_raw_prior_rows': 0,
                    'actual_writer_question_index_rows': 0, 'actual_disallowed_prior_six_item_sets': 0,
                    'names_only_policy_explicit_in_request': True,
                    'source_speaker_gender_and_pronouns_not_asserted_known': True,
                    'writer_has_complete_source_anchor_quotes_by_design': True,
                    'private_gold_and_display_labels_absent_from_anchor_records': True}
            feedback_receipt = feedback_check(run, root, feedback, proof, user_text, context_end)
            profile = read(request_dir / 'PROFILE.json')
            cap = annotate(capacity_function(profile, tokenizer, request['messages'], request['max_tokens']))
            need(cap == proof['capacity'] == read(run / 'CAPACITY.json')
                 and cap['status'] == 'PASS' and cap['truncation'] is False
                 and request['max_tokens'] == profile['output_limit'] == 6144,
                 'actual_capacity_replay_or_max_tokens_mismatch')
            expected_by_profile = {'model': profile['model'], 'messages': request['messages'],
                                  **{key: value for key, value in profile['request_parameters'].items() if value is not None}}
            need(request == expected_by_profile, 'actual_payload_profile_parameters_mismatch')
            attempt = read(request_dir / 'HTTP_ATTEMPT.json')
            reservation = read(request_dir / 'RESERVATION.json')
            http_result = read(request_dir / 'HTTP_RESULT.json')
            throttle = read(request_dir / 'THROTTLE.json')
            response = read(request_dir / 'RESPONSE.json')
            metadata = response['metadata']
            raw_response = metadata['raw_response']
            choice = raw_response['choices'][0]
            need(attempt['actual_payload_equals_frozen'] is True
                 and attempt['payload_sha256'] == assets.base.digest(request)
                 and attempt['http_attempt_number'] == 1 and attempt['transport_retries'] == 0
                 and reservation['automatic_retries'] == 0 and reservation['maximum'] == 1
                 and http_result['http_status'] == metadata['http_status'] == 200
                 and http_result['outcome'] == 'response' and not (request_dir / 'ERROR.json').exists()
                 and metadata['finish_reason'] == choice['finish_reason'] == 'stop'
                 and metadata['returned_model'] == metadata['actual_model'] == request['model'] == raw_response['model']
                 and metadata['transport_retry_count'] == metadata['rate_limit_retry_count'] == 0
                 and metadata['http_attempt_count'] == metadata['transport_attempt_count'] == 1
                 and response['text'] == choice['message']['content']
                 and metadata['usage'] == raw_response['usage']
                 and metadata['output_tokens'] < request['max_tokens'],
                 'actual_http_response_or_budget_integrity_failed')
            need(json.loads(metadata['raw_response_text']) == raw_response,
                 'saved_raw_http_body_decodes_differently')
            previous = throttle['previous_completion']
            previous_seconds = (datetime.fromisoformat(throttle['http_started_at_utc']).timestamp()
                                - previous['completed_at_epoch']) if previous else None
            need(throttle['minimum_gap_after_previous_response_seconds'] == 35
                 and (previous_seconds is None or previous_seconds >= 34.99), 'actual_transport_minimum_gap_failed')
            results.append({'lane': lane, 'root_id': root, 'run': str(run),
                'request_directory': str(request_dir), 'request': common.bind(request_dir / 'REQUEST.json'),
                'proof': common.bind(request_dir / 'PROOF.json'),
                'source': common.bind(run / 'SOURCE.json'), 'exact_source_turn_count': expected_turn_count,
                'source_bytes_equal_current_materialized_source': True,
                'actual_request_full_source_turns_and_metadata_exact': True,
                'actual_request_reconstructed_with_frozen_functions': True,
                'prior_and_public_projection': private_projection, 'historical_feedback': feedback_receipt,
                'capacity': cap, 'response': common.bind(request_dir / 'RESPONSE.json'),
                'http_status': metadata['http_status'], 'finish_reason': metadata['finish_reason'],
                'requested_and_returned_model': request['model'], 'output_cap': request['max_tokens'],
                'service_reported_input_tokens': metadata['input_tokens'],
                'service_reported_output_tokens': metadata['output_tokens'],
                'service_reported_total_tokens': metadata['total_tokens'],
                'service_minus_local_proxy_input_tokens': metadata['input_tokens'] - cap['input_tokens'],
                'http_attempts': 1, 'automatic_retries': 0,
                'actual_seconds_after_previous_response': previous_seconds,
                'raw_response_body_and_content_unchanged': True})
    rejection_run = SHARD / 'ordering/root19708_ordering'
    rejection_request = next(rejection_run.glob('requests/*'))
    rejection_raw = read(rejection_request / 'RESPONSE.json')['text']
    draft = assets.strict_json(rejection_raw)
    rejected = read(rejection_run / 'REJECTIONS.json')
    observed = []
    for question in draft['questions']:
        tokens = sorted({match.group().lower() for row in question['records']
                         for match in actor_reference.PATTERN.finditer(row['predicate'])})
        observed.append({'candidate_id': question['candidate_id'], 'forbidden_pronouns_observed': tokens})
    need(len(rejected) == len(observed) == 2 and read(rejection_run / 'PUBLIC_QUESTIONS.json')['questions'] == []
         and all(reason.startswith('ordering_actor_reference_policy_forbidden_pronoun:')
                 for row in rejected for reason in row['reasons']), 'ordering_rejection_record_mismatch')
    public_counts = {}
    for lane in ['multiselect', 'ordering']:
        rows = read(SHARD / lane / 'PUBLIC_QUESTIONS.json')['questions']
        need(not (keys(rows) & public_forbidden), 'gold_or_evidence_in_public_final_questions')
        public_counts[lane] = len(rows)
    batch = read(SHARD / 'BATCH_GENERATION.COMPLETE.json')
    need(batch['http_attempts'] == len(results) == 4 and batch['automatic_retry'] is False
         and batch['errors'] == [], 'batch_http_count_changed')
    production_before = read(BASE / 'evidence/ENGINE_DERIVATION_RECEIPT.json')
    for row in production_before['files']:
        common.verify_bindings([row['current']])
    common.verify_bindings(production_before['production_scripts'])
    preservation = read(BASE / 'PRESERVATION_BEFORE.json')
    common.verify_bindings(preservation['files'])
    result = {'version': 'v96_actual_qa_transport_prior_audit_v1', 'created_at': common.now(),
        'status': 'PASS_WITH_DISCLOSED_EMPTY_ORDERING_TEXT_PRIORS_AND_REJECTED_ORDERING_DRAFTS',
        'transport_integrity_status': 'PASS', 'multiselect_complete_historical_prior_status': 'PASS',
        'all_four_requests_full_source_and_exact_historical_feedback': True,
        'actual_http_attempts': len(results), 'http_200_and_stop_count': len(results),
        'all_request_output_caps': [6144], 'all_capacity_replays': 'PASS',
        'capacity_scope': 'Frozen Qwen3-14B proxy tokenizer with local 40960 context assumption; service capacity unknown, no truncation or fallback',
        'public_projection_checks': 'Public prior projections and final public questions contain no Gold or evidence fields; ordering writer anchors intentionally include complete source quotes',
        'ordering_prior_limitation': ordering_empty_prior,
        'ordering_19708_rejected': {'rejections': common.bind(rejection_run / 'REJECTIONS.json'),
            'candidate_count': 2, 'original_reasons': rejected, 'raw_predicate_pronouns': observed,
            'names_only_policy_was_explicit_in_actual_request': True,
            'violation_is_writer_policy_noncompliance_not_proven_source_gender': True,
            'original_candidates_unchanged_and_not_repaired': True},
        'current_public_generated_counts_before_solver_admission': public_counts,
        'private_read_disclosure': {
            'this_actual_audit_json_decoded': ['raw writer responses for both multi-select families and both ordering runs',
                'full actual frozen historical candidate histories, aggregates, source locks and feedback',
                'public parents through request reconstruction', 'public questions and writer plans',
                'actual requests, proofs, profiles, transport and capacity records'],
            'not_json_decoded_in_this_actual_audit': ['current private multi-select ASSETS/POOL/ALLOCATION/Gold',
                'current private ordering GOLD_PLAN/ANCHOR_PLAN/POOL/Gold', 'any current Solver predictions'],
            'earlier_compatibility_task_did_access': ['public parent/judgment records',
                'temporary private parent projection through builder integration', 'original historical candidate pools with historical answers'],
            'not_a_blind_semantic_source_review': True,
            'manifest_or_pool_private_contents_not_needed_for_this_transport_audit': True},
        'production_code_sha_still_matches_frozen_derivation_receipt': True,
        'sealed_v95_protected_files_unchanged': preservation['protected_count'],
        'model_calls_or_batch_mutations_performed_by_audit': 0,
        'audit_script': common.bind(__file__), 'requests': results,
        'json_or_text_decoded_files_and_bound_historical_dependencies': [READS[path] for path in sorted(READS)]}
    common.write(BASE / 'evidence/ACTUAL_QA_TRANSPORT_PRIOR_AUDIT.json', result)
    print(json.dumps({'status': result['status'], 'http_calls': len(results),
                      'output_tokens': [row['service_reported_output_tokens'] for row in results],
                      'prior_counts': [36, 35], 'ordering_prior_counts': [0, 0],
                      'source_turn_counts': [105, 104]}, ensure_ascii=False))


if __name__ == '__main__':
    audit()
