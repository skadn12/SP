"""Audit real prepared/actual QA inputs without opening current private answers.

This script performs no HTTP, generation, semantic judgment, answer repair or
training. Run only after independent source/public-parent judgments are saved.
A local capacity PASS does not establish model-service capacity or answer truth.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
READS = {}
PUBLIC_FORBIDDEN = {'gold', 'original_gold', 'correct_option', 'evidence', 'evidence_refs',
                    'evidence_groups', 'item_evidence', 'option_evidence', 'models', 'training_exposure'}
EXPECTED = {'cc-train-episode-19053': (80, 36, 9), 'cc-train-episode-9557': (92, 36, 9),
            'cc-train-episode-104814': (94, 32, 10), 'cc-train-episode-194259': (149, 12, 12)}


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    path = Path(path).resolve(); READS[str(path)] = bind(path)
    return json.loads(path.read_text())


def read_text(path):
    path = Path(path).resolve(); READS[str(path)] = bind(path)
    return path.read_text()


def need(condition, reason):
    if not condition: raise ValueError(reason)


def keys(value):
    if isinstance(value, dict): return set(value).union(*(keys(v) for v in value.values()))
    if isinstance(value, list): return set().union(*(keys(v) for v in value))
    return set()


def after(value, marker):
    start = value.index(marker) + len(marker)
    obj, count = json.JSONDecoder().raw_decode(value[start:])
    return obj, start + count


def sha_text(value): return hashlib.sha256(value.encode()).hexdigest()


def check_bindings(rows):
    need(len({r['path'] for r in rows}) == len(rows), 'duplicate_frozen_binding')
    for row in rows: need(bind(row['path']) == row, 'frozen_binding_changed:' + row['path'])


def exact_feedback(run, root, feedback, proof, user, context_end):
    frozen = run / 'feedback'; suffix_path = frozen / root / 'writer__M_clean_policy_feedback.txt'
    suffix = read_text(suffix_path)
    need(suffix == feedback['writer_suffix'], 'previous_round_feedback_suffix_not_exact')
    need(read(frozen / root / 'candidate_history.json') == feedback['candidate_history'], 'historical_feedback_candidate_history_changed')
    need(read(frozen / root / 'aggregate.json') == feedback['aggregate'], 'historical_feedback_aggregate_changed')
    lock = read(frozen / 'source_lock.json')
    need(lock == feedback['source_lock'] and lock['status'] == 'EXPLICIT_PREVIOUS_TRAINING_ROUND_ONLY'
         and lock['merge_current_generation_feedback'] is False
         and lock['automatic_latest_run_selection'] is False, 'feedback_not_frozen_previous_training_round')
    need(user[context_end:context_end + len(suffix)] == suffix and user.count(suffix) == 1,
         'request_full_feedback_missing_changed_or_duplicated')
    receipt = proof['feedback_receipt']
    need(receipt['root_id'] == root and receipt['target_solver_model'] == 'M_clean_policy'
         and receipt['feedback_appended'] is True
         and receipt['suffix_sha256'] == sha_text(suffix)
         and receipt['source_lock_sha256'] == bind(frozen / 'source_lock.json')['sha256']
         and receipt['request_user_text_sha256'] == sha_text(user[:context_end + len(suffix)]),
         'request_feedback_sha_chain_changed')
    return {'suffix': bind(suffix_path), 'candidate_history': bind(frozen / root / 'candidate_history.json'),
            'source_lock': bind(frozen / 'source_lock.json'), 'all_exact_historical_recompile': True,
            'full_suffix_once_after_context': True, 'current_solver_feedback_used': False}


def unchanged_gold_mechanism(shard):
    current = read_text(shard / 'ordering_plan.py')
    old = read_text(BASE.parent / 'h4_v95_ready_mixed_qa_v96/engine/ordering_plan.py')
    def function(code, name):
        return next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == name)
    for name in ('gold_questions', 'writer_plan'):
        need(ast.dump(function(current, name), include_attributes=False) ==
             ast.dump(function(old, name), include_attributes=False), 'program_gold_or_writer_anchor_algorithm_changed:' + name)
    gold_func = function(current, 'gold_questions')
    need('support_quote' not in ast.unparse(gold_func), 'support_quote_controls_program_gold')
    need("'quote'" in ast.unparse(gold_func) and "'turn_index'" in ast.unparse(gold_func),
         'program_gold_source_turn_positions_missing')
    return {'gold_questions_and_writer_plan_ast_equal_sealed_v96': True,
            'program_gold_function_reads_support_quote': False,
            'gold_mechanism': 'Original complete source-turn appearance positions; six distinct complete-turn anchors frozen before writer.',
            'current_gold_plan_json_decoded': False,
            'scope': 'Mechanism and bindings only; independent final review/private crosscheck must certify each actual answer.'}


def actual_http(request_dir, request, proof, digest):
    actual = read(request_dir / 'REQUEST.json'); actual_proof = read(request_dir / 'PROOF.json')
    need(actual == request and all(actual_proof.get(k) == v for k, v in proof.items()), 'actual_request_not_exact_frozen')
    attempt = read(request_dir / 'HTTP_ATTEMPT.json')
    need(attempt['actual_payload_equals_frozen'] is True and attempt['payload_sha256'] == digest(request)
         and attempt['http_attempt_number'] == 1 and attempt['transport_retries'] == 0,
         'actual_http_attempt_or_retry_changed')
    result = read(request_dir / 'HTTP_RESULT.json')
    # Do not JSON-decode RESPONSE.json: it can contain unaudited current drafts.
    response_path = request_dir / 'RESPONSE.json'
    error_path = request_dir / 'ERROR.json'
    return {'directory': str(request_dir), 'request': bind(request_dir / 'REQUEST.json'),
            'actual_equals_frozen': True, 'http_status': result.get('http_status'),
            'http_outcome': result.get('outcome'),
            'response_binding_only': bind(response_path) if response_path.exists() else None,
            'error_binding_only': bind(error_path) if error_path.exists() else None,
            'raw_current_response_json_decoded': False}


def audit(shard, *, require_actual=False):
    shard = Path(shard).resolve()
    if not (shard / 'CONFIG.json').exists():
        skip = BASE / 'evidence/QA_SKIPPED_NO_READY_SOURCE.json'
        need(skip.exists(), 'qa_shard_not_prepared_and_no_explicit_zero_ready_skip')
        record = read(skip)
        need(record.get('status') == 'SKIPPED' and record.get('qa_http_calls') == record.get('solver_calls') == 0,
             'zero_ready_skip_record_invalid')
        return {'version': 'v97_actual_qa_input_audit_v1', 'status': 'SKIPPED_NO_READY_SOURCE',
                'skip_record': bind(skip), 'semantic_certification': False,
                'service_capacity_certification': False, 'audit_http_calls': 0}
    sys.dont_write_bytecode = True; sys.path.insert(0, str(shard))
    import common
    import prior_rendering
    import ordering_prior
    import ordering_generation
    import actor_reference
    assets = common.load_multi('asset_generation')
    config = read(shard / 'CONFIG.json')
    precall = read(BASE / 'PRECALL_PLAN.json'); frozen_roots = precall['candidate_roots']
    roots = [s['root_id'] for s in config['multiselect_config']['slots']]
    need(0 < len(roots) <= 4 and roots == [r for r in frozen_roots if r in roots], 'qa_not_ordered_nonempty_ready_subset')
    need([s['root_id'] for s in config['ordering_slots']] == roots, 'mixed_root_scope_differs')
    admission = read(shard / 'SOURCE_ADMISSION_SUMMARY.json')
    need(admission['ready_roots'] == roots and all(r['decision'] == 'READY' for r in admission['roots']),
         'qa_sources_not_independently_ready')
    scope = read(BASE / 'PILOT_PLAN.json')
    run = Path(scope['upstream_expansion_run']['MANIFEST.json']['path']).parent
    registry = {r['root_id']: r for r in read(BASE / 'expansion_engine/SOURCE_REGISTRY.json')['roots']}
    need(sum(registry[r]['source_turns'] for r in frozen_roots) == 415, 'frozen_four_root_historical_total_not415')
    derivation = read(BASE / 'evidence/ORDERING_MINFIX_RECEIPT.json'); check_bindings(derivation['files'])
    installation = read(shard / 'SHARD_ENGINE_INSTALLATION.json')
    check_bindings(installation['source_files']); check_bindings(installation['files'])
    mechanism = unchanged_gold_mechanism(shard)
    _, _, annotate, capacity, development = assets.base.transport()
    tokenizer = development.tokenizer_from(str(assets.base.TOKENIZER))
    output = []; root_rows = []
    ms_slots = {s['root_id']: s for s in config['multiselect_config']['slots']}
    order_slots = {s['root_id']: s for s in config['ordering_slots']}
    for root in roots:
        short = root.rsplit('-', 1)[-1]
        history_count, feedback_count, expected_ordering_priors = EXPECTED[root]
        baseline_path = run / 'sources' / (root + '.json'); baseline = read(baseline_path)
        need(bind(baseline_path)['sha256'] == registry[root]['source_sha256']
             and len(baseline['turns']) == history_count, 'current_run_historical_source_not_exact_registered_full_source')
        source_path = BASE / f'sources/root{short}_SOURCE.json'; source = read(source_path)
        provenance = read(BASE / f'evidence/expansion_provenance/root{short}_PROVENANCE.json')
        need(provenance['output_source'] == bind(source_path) == provenance['source']
             and source_path.read_bytes() == Path(provenance['accepted_output']['path']).read_bytes()
             and source['turns'][:history_count] == baseline['turns'], 'current_expansion_source_or_prefix_changed')
        suffix = source['turns'][history_count:]
        need(suffix and len(suffix) == provenance['new_turn_count']
             and [t['turn_ref'] for t in suffix] == provenance['new_turn_refs']
             and all(t['turn_ref'].startswith('R02:B:S01:') for t in suffix), 'current_source_new_suffix_missing_or_changed')
        feedback = assets.source_adapter.feedback_data(root)
        need(len(feedback['candidate_history']) == feedback_count, 'frozen_historical_feedback_count_changed')
        root_rows.append({'root_id': root, 'baseline_source': bind(baseline_path),
                          'exact_historical_turns': history_count, 'actual_new_turns': len(suffix),
                          'full_actual_source_turns': len(source['turns']), 'full_prefix_preserved': True,
                          'new_refs': [t['turn_ref'] for t in suffix]})
        for lane, slot, folder in [('multi_select', ms_slots[root], f'multiselect/assets/root{short}_family'),
                                    ('ordering', order_slots[root], f'ordering/root{short}_ordering')]:
            family = shard / folder; manifest = read(family / 'MANIFEST.json')
            need(bind(family / 'SOURCE.json')['sha256'] == bind(source_path)['sha256'], 'qa_family_source_not_exact_full_current_source')
            prior = read(family / 'PRIOR.json')
            compact, prior_files = assets.source_adapter.prior_candidates(root, feedback,
                family / 'feedback' / root / 'candidate_history.json', assets.v15)
            for path in prior_files: READS[str(Path(path).resolve())] = bind(path)
            if lane == 'multi_select':
                expected = list(compact)
                for path in slot.get('prior_pool_paths', []): expected += assets.legacy._prior_pool_rows(Path(path), root)
                need(prior == expected and {h['candidate_id'] for h in feedback['candidate_history']} <= {q['candidate_id'] for q in prior},
                     'multiselect_prior_incomplete_or_changed')
                paths = sorted((family / 'rounds').glob('*/INITIAL_REQUEST.json'))
                # Every actual prepared round uses its own immutable target set.
                need(paths and any(p.parent.name == 'initial' for p in paths), 'initial_multiselect_request_missing')
                candidates = [(p.parent.name, p, p.parent / 'INITIAL_PROOF.json', p.parent) for p in paths]
            else:
                expected, receipt, _ = ordering_prior.collect(root,
                    family / 'feedback' / root / 'candidate_history.json', ordering_prior.configured_paths(slot),
                    compact_rows=compact, compact_files=prior_files)
                need(prior == expected and read(family / 'PRIOR_COVERAGE.json') == receipt,
                     'ordering_public_prior_not_complete_or_changed')
                need(len(prior) == expected_ordering_priors, 'actual_ordering_prior_expected_9_9_10_12_changed')
                need(not (keys(prior) & PUBLIC_FORBIDDEN), 'ordering_public_prior_private_field_leak')
                need(Path(slot['prior_pool_path']).resolve() in {Path(p).resolve() for p in ordering_prior.configured_paths(slot)},
                     'ordering_baseline_prior_suppressed')
                candidates = [('initial', family / 'REQUEST.json', family / 'PROOF.json', family)]
            for round_id, request_path, proof_path, request_home in candidates:
                request = read(request_path); proof = read(proof_path)
                user = request['messages'][0]['content']; context, context_end = after(user, 'INPUT_JSON\n')
                need(context['source_turns'] == source['turns']
                     and context['source_metadata'] == {k:v for k,v in source.items() if k != 'turns'},
                     'actual_qa_source_or_metadata_truncated')
                need(proof['payload_sha256'] == assets.base.digest(request)
                     and proof['source_sha256'] == bind(source_path)['sha256'], 'prepared_request_proof_changed')
                prior_result = {'on_disk_prior_count': len(prior), 'historical_feedback_candidates': feedback_count}
                if lane == 'multi_select':
                    projection = assets.legacy.prompt_prior(prior)
                    encoded = context['previous_candidates_dedup_only']
                    need(prior_rendering.decode_prior(encoded) == projection and not (keys(encoded) & PUBLIC_FORBIDDEN),
                         'multiselect_public_prior_grouping_lost_text_or_private_leak')
                    need(context['used_six_statement_sets'] == assets.legacy.used_statement_sets(projection),
                         'multiselect_dedup_original_indices_changed')
                    prior_result.update(public_rows=len(projection), public_groups=len(encoded['groups']),
                                        type_counts=dict(Counter(q['question_type'] for q in prior)), all_public_indices_lossless=True)
                else:
                    public, _ = after(user, 'PUBLIC HISTORICAL ORDERING QUESTIONS (deduplication only, never evidence):\n')
                    public_expected = [{k:q[k] for k in ('candidate_id','root_id','question_type','question','items')} for q in prior]
                    need(public == {'public_ordering_questions': public_expected}
                         and not (keys(public) & PUBLIC_FORBIDDEN), 'actual_ordering_public_prior_text_missing_or_private_leak')
                    index = context['previous_public_questions_dedup_only']['question_index']
                    need(index == [{k:q[k] for k in ('candidate_id','root_id','question_type')} for q in prior],
                         'ordering_public_prior_identity_index_changed')
                    writer = read(family / 'WRITER_PLAN.json'); records, _ = after(user, 'SOURCE_ID_ANCHOR_RECORDS:\n')
                    need(records == writer['questions'] and writer['display_labels_supplied'] is False
                         and writer['gold_supplied_to_writer'] is False
                         and not (keys(records) & {'gold','original_gold','labels','label','items','item_evidence'}),
                         'ordering_writer_records_changed_or_gold_labels_leaked')
                    lookup = {t['turn_ref']:t for t in source['turns']}; new_refs = set(provenance['new_turn_refs'])
                    for q in records:
                        anchors = q['anchor_records']
                        need(len(anchors) == len({a['anchor_id'] for a in anchors}) == 6, 'ordering_not_six_distinct_turns')
                        for anchor in anchors:
                            turn = lookup[anchor['anchor_id']]
                            need(anchor['anchor_id'] in new_refs and anchor['quote'] == turn['text']
                                 and anchor['speaker'] == turn['speaker']
                                 and anchor['statement_prefix'] == turn['speaker'] + ' ', 'ordering_anchor_not_complete_current_source_turn')
                    policy, _ = after(user, 'ACTOR_REFERENCE_POLICY:\n')
                    need(policy == read(family / 'ACTOR_REFERENCE_POLICY.json')
                         and policy['forbidden_gendered_pronouns'] == list(actor_reference.TOKENS)
                         and policy['automatic_repair'] is False and policy['new_source_facts'] is False,
                         'ordering_names_only_policy_missing')
                    need('"support_quote":"exact contiguous source substring"' in user
                         and 'Select ONE event or speech act inside that complete turn' in user
                         and 'The full original turn remains frozen' in user, 'writer_support_quote_single_event_contract_missing')
                    need(manifest['gold_plan'] == bind(family / 'GOLD_PLAN.json')
                         and manifest['anchor_plan'] == bind(family / 'ANCHOR_PLAN.json')
                         and manifest['writer_plan'] == bind(family / 'WRITER_PLAN.json'), 'ordering_pre_writer_private_artifact_binding_changed')
                    # payload_for uses SOURCE/PRIOR/WRITER_PLAN, never private Gold.
                    replay, replay_proof = ordering_generation.payload_for(family)
                    need(replay == request and replay_proof == proof, 'ordering_public_payload_reconstruction_differs')
                    prior_result.update(public_rows=len(prior), item_counts=[len(q['items']) for q in prior],
                                        complete_original_stems_and_items_retained=True, six_full_turn_anchors_per_slot=True,
                                        support_quote_is_output_contract_only=True, current_private_gold_read=False)
                feedback_result = exact_feedback(family, root, feedback, proof, user, context_end)
                profile = read(family / 'provider_profile.json'); profile['output_limit'] = request['max_tokens']
                profile['request_parameters']['max_tokens'] = request['max_tokens']
                cap = annotate(capacity(profile, tokenizer, request['messages'], request['max_tokens']))
                need(request['max_tokens'] == 6144 and cap['status'] == 'PASS' and cap['truncation'] is False,
                     'prepared_request_not6144_or_local_capacity_blocked')
                expected_payload = {'model':profile['model'],'messages':request['messages'],
                    **{k:v for k,v in profile['request_parameters'].items() if v is not None}}
                need(request == expected_payload, 'request_provider_parameters_changed')
                if lane == 'ordering': need(cap == read(family / 'CAPACITY.json'), 'ordering_frozen_capacity_changed')
                if lane == 'multi_select' and round_id == 'initial':
                    need(cap == read(family / 'CAPACITY.json'), 'multiselect_frozen_initial_capacity_changed')
                actual_dirs = sorted(p.parent for p in (request_home / 'requests').glob('*/REQUEST.json'))
                need(len(actual_dirs) <= 1 and (not require_actual or len(actual_dirs) == 1), 'actual_request_count_missing_or_retried')
                actual = [actual_http(p, request, proof, assets.base.digest) for p in actual_dirs]
                output.append({'root_id':root,'lane':lane,'round':round_id,'request':bind(request_path),
                    'source':bind(family / 'SOURCE.json'),'full_source_turns':len(source['turns']),
                    'complete_historical_prefix_turns':history_count,'actual_new_suffix_turns':len(suffix),
                    'prior':prior_result,'previous_training_feedback':feedback_result,'capacity':cap,
                    'service_capacity_certified':False,'actual_requests':actual})
    return {'version':'v97_actual_qa_input_audit_v1','status':'PASS_STRUCTURAL_INPUTS_ONLY',
            'created_at_utc':datetime.now(timezone.utc).isoformat(),'shard':str(shard),'roots':roots,
            'fixed_preselected_roots':frozen_roots,'fixed_historical_source_turn_total':415,
            'audited_ready_historical_source_turn_total':sum(r['exact_historical_turns'] for r in root_rows),
            'audited_ready_full_actual_source_turn_total':sum(r['full_actual_source_turns'] for r in root_rows),
            'require_actual_http':require_actual,'actual_http_requests_verified':sum(len(r['actual_requests']) for r in output),
            'prepared_request_count':len(output),'source_rows':root_rows,'requests':output,
            'program_gold_mechanism':mechanism,'semantic_certification':False,'answer_reliability_certified':False,
            'service_capacity_certification':False,'local_capacity_limitation':'Frozen Qwen3-14B proxy tokenizer and40960local guard; actual service capacities unknown; no source/feedback/prior truncation.',
            'private_read_disclosure':{'current_private_gold_anchor_pool_assets_or_solver_json_decoded':False,
                'current_gold_and_anchor_plan_only_sha_bound':True,'historical_candidate_history_contains_historical_answers':True,
                'public_current_writer_request_and_complete_full_turn_writer_plan_read':True},
            'audit_http_calls':0,'training_allowed':False,'audit_script':bind(__file__),
            'json_or_text_decoded_files':[READS[p] for p in sorted(READS)]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=Path, default=BASE/'shards/shard_01')
    parser.add_argument('--actual', action='store_true', help='Require exact actual HTTP request records for every prepared round.')
    parser.add_argument('--output', type=Path, default=BASE/'evidence/ACTUAL_QA_INPUT_AUDIT.json')
    args = parser.parse_args()
    result = audit(args.shard, require_actual=args.actual)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps({'status':result['status'],'prepared_requests':result.get('prepared_request_count',0),
                      'actual_http_verified':result.get('actual_http_requests_verified',0)},ensure_ascii=False))
