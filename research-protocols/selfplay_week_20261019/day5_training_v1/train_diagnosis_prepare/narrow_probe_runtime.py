"""CPU preparation and separately released narrow quote/positions probe.

No automatic release, hypothesis promotion or extra request exists here.
"""
from __future__ import annotations
import argparse
import collections
import json
from pathlib import Path
import statistics
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_prepare as b
import run_v2 as r


def prepare():
    r.verify()
    reserve = b.read(HERE / 'NARROW_PROBE_RESERVE_V2.json')
    amendment = b.read(HERE / 'PROBE_RESERVE_ADDENDUM_V2.json')
    r.require(amendment['old_reserve_current_disposition'] == 'RETIRED_DO_NOT_RELEASE_OR_EXECUTE' and amendment['combined_new_probe_cap_day5_day6'] == 24, 'OLD_RESERVE_NOT_RETIRED_OR_CAP_CHANGED')
    r.check([amendment['old_reserve_preserved'], amendment['new_narrow_reserve'], amendment['main_direct_and_brief_evidence_freeze_unchanged']])
    r.check(reserve['public_prompt_bindings'] + reserve['S0_submitted_batch_bindings'] + [reserve[k] for k in ('source_protocol', 'source_manifest', 'source_evidence', 'source_semantic_review', 'source_operational_card_collection', 'preparation_code')])
    endpoints = {role: b.binding(HERE / 'models' / role / 'EXECUTION_MANIFEST.json') for role in ('A1', 'B1')}
    for ep in endpoints.values():
        m = b.read(ep['path'])
        r.check(m['model_bindings'] + m['tokenizer_bindings'] + m['endpoint_evidence_bindings'] + [m['endpoint_manifest'], m['endpoint_proof']])
    r.require(len(reserve['requests_per_model']) == 12 and len(set(x['request_id'] for x in reserve['requests_per_model'])) == 12, 'EXACT12_REQUESTS_REQUIRED')
    for f in reserve['frozen_records']:
        r.check([f['source_binding']])
        source = b.read(f['source_binding']['path'])
        byref = {t['turn_ref']: (i, t) for i, t in enumerate(source['turns'])}
        positions = []
        for item in f['evidence']:
            idx, turn = byref[item['turn_ref']]
            r.require(item['quote'] and turn['text'].count(item['quote']) == 1, 'FROZEN_QUOTE_NOT_EXACT_UNIQUE')
            positions.append({'label': item['label'], 'source_turn_index': idx, 'quote_start_offset': turn['text'].find(item['quote'])})
        r.require(sorted(positions, key=lambda x: x['label']) == sorted(f['positions'], key=lambda x: x['label']), 'FROZEN_POSITIONS_DRIFT')
        target = [x['label'] for x in sorted(positions, key=lambda x: (x['source_turn_index'], x['quote_start_offset']))]
        r.require(target == f['citation_sorted_display'], 'FROZEN_TARGET_DRIFT')
    files = [b.binding(HERE / name) for name in ('NARROW_PROBE_RESERVE_V2.json', 'PROBE_RESERVE_ADDENDUM_V2.json', 'narrow_probe_runtime.py', 'run_v2.py', 'build_prepare.py')]
    b.save(HERE / 'NARROW_PROBE_CPU_FREEZE.json', {'status': 'CPU_PREPARED_NOT_RELEASED_NOT_EXECUTED', 'created_at_utc': r.now(),
        'files': files, 'endpoints': endpoints, 'runtime_versions': r.versions(), 'python_prefix': sys.prefix,
        'requests_per_model': 12, 'maximum_total_new_requests': 24, 'actual_calls': 0,
        'main_diagnosis_not_modified': b.binding(HERE / 'EXECUTION_ADDENDUM_V2.json'),
        'release_required': True, 'probe_specific_independent_audit_required': True})
    print('NARROW_PROBE_CPU_PREPARED_NO_RELEASE_NO_INFERENCE')


def verify():
    r.verify()
    f = b.read(HERE / 'NARROW_PROBE_CPU_FREEZE.json')
    r.check(f['files'] + list(f['endpoints'].values()) + [f['main_diagnosis_not_modified']])
    r.require(f['maximum_total_new_requests'] == 24 and f['requests_per_model'] == 12 and f['runtime_versions'] == r.versions() and f['python_prefix'] == sys.prefix, 'CPU_FREEZE_SCOPE_OR_RUNTIME_CHANGED')
    reserve = b.read(HERE / 'NARROW_PROBE_RESERVE_V2.json')
    r.check(reserve['public_prompt_bindings'] + reserve['S0_submitted_batch_bindings'])
    return f, reserve


def released(role, releasepath, auditpath):
    f, reserve = verify()
    release = b.read(releasepath)
    r.require(release['status'] == 'RELEASED_AFTER_B1_SOURCE_REVIEW' and release['source_review_completed'] is True and release['formal_B1_reproduced'] is False,
        'SOURCE_REVIEW_RELEASE_REQUIRED_NO_PREMATURE_REPRODUCED')
    r.require(release['allowed_request_count_total'] == 24 and release['old_general_reserve_retired'] is True and release['unresolved_hypothesis'], 'EXACT_SHARED_CAP_AND_CONCRETE_HYPOTHESIS_REQUIRED')
    r.require(len(set(release['qualified_distinct_train_roots'])) >= 3 and len(set(release['qualified_distinct_fact_families'])) >= 3, 'SOURCE_REVIEWED_MIN3_ROOTS_FAMILIES_REQUIRED')
    r.check([release['b1_feedback_snapshot'], release['b1_semantic_review'], release['probe_cpu_freeze']])
    r.require(release['probe_cpu_freeze'] == b.binding(HERE / 'NARROW_PROBE_CPU_FREEZE.json'), 'RELEASE_WRONG_CPU_FREEZE')
    # Reviewer must inspect actual supported-case semantics, not merely count fields.
    review = b.read(auditpath)
    r.require(review['status'] == 'PASS' and review['scope'] == 'NARROW_QUOTE_POSITION_PROBE_RELEASE', 'PROBE_SPECIFIC_AUDIT_REQUIRED')
    required = [b.binding(releasepath), b.binding(HERE / 'NARROW_PROBE_CPU_FREEZE.json'), f['endpoints'][role]]
    r.require(all(x in review['audited_files'] for x in required), 'PROBE_REVIEW_NOT_BOUND_TO_RELEASE_FREEZE_ENDPOINT')
    r.check(review['audited_files'])
    m = b.read(f['endpoints'][role]['path'])
    r.check(m['model_bindings'] + m['tokenizer_bindings'] + m['endpoint_evidence_bindings'] + [m['endpoint_proof']])
    return reserve, m


def live(role, releasepath, auditpath):
    reserve, m = released(role, releasepath, auditpath)
    run = HERE / 'models' / role / 'narrow_probe'
    r.require(not run.exists(), 'NO_PROBE_RETRY_OR_OVERWRITE')
    rows = reserve['requests_per_model']
    b.save(run / 'STARTED.json', {'started_at_utc': r.now(), 'model_role': role, 'release': b.binding(releasepath), 'audit': b.binding(auditpath),
        'cpu_freeze': b.binding(HERE / 'NARROW_PROBE_CPU_FREEZE.json'), 'endpoint': b.binding(HERE / 'models' / role / 'EXECUTION_MANIFEST.json'),
        'max_planned_requests': 12, 'global_new_probe_cap': 24, 'Gold_opened': False})
    backend, failed, sent, done = None, None, set(), set()
    start, load_seconds = time.perf_counter(), None
    try:
        a = r.adapter()
        _, solver = a.native()
        solver.ensure_vllm_environment()
        from scripts.current_method.restart_n5_spell.vllm_backends import VLLMSolverBackend
        from vllm import SamplingParams, sampling_params
        backend = VLLMSolverBackend(model_path=Path(m['model_directory']), tokenizer_path=Path(m['tokenizer_directory']), max_model_len=16384,
            temperature=.8, top_p=.9, max_tokens=32, gpu_memory_utilization=.8)
        load_seconds = backend.load_latency_seconds
        r.require(str(backend.model_identifier) == m['model_directory'], 'WRONG_BACKEND_MODEL')
        for offset in range(0, len(rows), 8):
            batch = rows[offset:offset + 8]
            prompts, params = [], []
            for row in batch:
                p = Path(row['prompt_path'])
                r.require(b.sha(p) == row['prompt_sha256'], 'PROBE_PROMPT_CHANGED')
                prompts.append(p.read_text())
                choices = reserve['finite_choices'][reserve['request_choice_keys'][row['request_id']]]
                kw = {'temperature': .8, 'top_p': .9, 'max_tokens': 32, 'seed': row['seed'], 'stop': ['</box>'], 'include_stop_str_in_output': True}
                if m['decoder_contract']['structured_output_api'].startswith('structured_outputs='):
                    kw['structured_outputs'] = sampling_params.StructuredOutputsParams(choice=choices)
                else:
                    kw['guided_decoding'] = sampling_params.GuidedDecodingParams(choice=choices)
                params.append(SamplingParams(**kw))
            bid = 'batch' + str(offset // 8).zfill(4)
            b.save(run / 'batches' / (bid + '_SUBMITTED.json'), {'submitted_at_utc': r.now(), 'request_ids': [x['request_id'] for x in batch],
                'sampling_params': [p.to_dict() if hasattr(p, 'to_dict') else repr(p) for p in params], 'automatic_retry': False})
            sent.update(x['request_id'] for x in batch)
            outputs = backend.engine.generate(prompts, params, use_tqdm=False)
            for i, row in enumerate(batch):
                if i >= len(outputs):
                    continue
                out = outputs[i]
                completions = list(out.outputs)
                first = completions[0] if completions else None
                errors = []
                if out.prompt != prompts[i] or len(out.prompt_token_ids) != row['prompt_tokens']:
                    errors.append('PROMPT_OR_TOKEN_COUNT_DRIFT')
                if len(completions) != 1:
                    errors.append('COMPLETION_COUNT_DRIFT')
                b.save(run / 'responses' / (row['request_id'] + '.json'), {'request_id': row['request_id'], 'status': 'RETURNED',
                    'model_role': role, 'model_directory': m['model_directory'], 'prompt_sha256': row['prompt_sha256'], 'seed': row['seed'],
                    'raw_output': first.text if first else None, 'all_completions': [{'text': x.text, 'token_ids': x.token_ids, 'finish_reason': x.finish_reason} for x in completions],
                    'prompt_token_ids': out.prompt_token_ids, 'output_validation_errors': errors, 'response_committed_at_utc': r.now(),
                    'metadata': {'finish_reason': first.finish_reason if first else None, 'stop_reason': first.stop_reason if first else None,
                        'prompt_token_count': len(out.prompt_token_ids), 'completion_token_count': len(first.token_ids) if first else None}})
                done.add(row['request_id'])
            if len(outputs) > len(batch):
                b.save(run / 'batches' / (bid + '_EXTRA_RETURNS.json'), {'outputs': [{'request_id': x.request_id, 'completions': [{'text': z.text, 'token_ids': z.token_ids} for z in x.outputs]} for x in outputs[len(batch):]]})
            r.require(len(outputs) == len(batch) and all(not b.read(run / 'responses' / (x['request_id'] + '.json'))['output_validation_errors'] for x in batch), 'BACKEND_DRIFT_AFTER_RAW_COMMIT')
            print(json.dumps({'model_role': role, 'probe_completed': len(done), 'planned': 12}), flush=True)
    except BaseException as e:
        failed = {'error': str(e), 'exception_type': type(e).__name__, 'traceback': traceback.format_exc()}
    finally:
        if backend:
            try:
                backend.close()
            except BaseException as e:
                failed = failed or {'error': str(e), 'traceback': traceback.format_exc()}
        for row in rows:
            if row['request_id'] not in done:
                b.save(run / 'responses' / (row['request_id'] + '.json'), {'request_id': row['request_id'], 'raw_output': None,
                    'status': 'BACKEND_ERROR' if row['request_id'] in sent else 'NOT_SUBMITTED', 'response_committed_at_utc': r.now(), 'failure': failed,
                    'usage_status': 'UNKNOWN_NOT_ZERO' if row['request_id'] in sent else 'NO_REQUEST_SENT'})
        b.save(run / ('FAILED.json' if failed else 'COMPLETE.json'), {'status': 'FAILED' if failed else 'COMPLETE', 'completed_at_utc': r.now(),
            'submitted_requests': len(sent), 'returned_requests': len(done), 'wall_seconds': time.perf_counter() - start, 'model_load_wall_seconds': load_seconds, 'failure': failed})
        files = [run / 'responses' / (x['request_id'] + '.json') for x in rows]
        r.require(set(files) == set((run / 'responses').glob('*.json')), 'EXTRA_RAW_FILES')
        b.save(run / 'RAW_RESPONSE_FREEZE.json', {'status': 'COMMITTED', 'created_at_utc': r.now(), 'files': [b.binding(p) for p in files],
            'planned_request_count': 12, 'cpu_freeze': b.binding(HERE / 'NARROW_PROBE_CPU_FREEZE.json'), 'release': b.binding(releasepath), 'Gold_opened': False})
    if failed:
        raise RuntimeError('PROBE_FAILED_NO_AUTOMATIC_RETRY:' + failed['error'])


def score(role):
    _, reserve = verify()
    run = HERE / 'models' / role / 'narrow_probe'
    freeze = b.read(run / 'RAW_RESPONSE_FREEZE.json')
    r.check(freeze['files'] + [freeze['cpu_freeze'], freeze['release']])
    r.require(freeze['planned_request_count'] == 12, 'PROBE_RAW_INCOMPLETE')
    r.check(reserve['S0_response_bindings'])
    frozen = {f['frozen_record_id']: f for f in reserve['frozen_records']}
    results = {}
    for label in ('S0', role):
        records = []
        for row in reserve['requests_per_model']:
            path = b.WEEK / 'day2_frozen_evidence_probe_v2/run/responses' / (row['request_id'] + '.json') if label == 'S0' else run / 'responses' / (row['request_id'] + '.json')
            raw = b.read(path)
            choices = reserve['finite_choices'][reserve['request_choice_keys'][row['request_id']]]
            valid = raw['status'] == 'RETURNED' and not raw.get('output_validation_errors') and raw.get('metadata', {}).get('finish_reason') in ('stop', 'eos') and raw['raw_output'] in choices
            target = frozen[row['frozen_record_id']]['citation_sorted_display']
            predicted = raw['raw_output'][5:-6].split(',') if valid else None
            ranks = {v: i for i, v in enumerate(predicted or [])}
            pairs = [(a, z) for i, a in enumerate(target) for z in target[i + 1:]]
            pair_score = sum(bool(valid and ranks[a] < ranks[z]) for a, z in pairs) / len(pairs)
            records.append({'request_id': row['request_id'], 'root_id': row['root_id'], 'frozen_record_id': row['frozen_record_id'], 'probe_arm': row['probe_arm'],
                'repeat_id': row['repeat_id'], 'valid': bool(valid), 'exact': bool(valid and predicted == target), 'pair_accuracy': pair_score,
                'citation_order_target': target, 'prediction': predicted, 'response_binding': b.binding(path)})
        byroot = {}
        for root in reserve['roots']:
            arms = {}
            for arm in ('freeze_quotes', 'explicit_positions'):
                rs = [x for x in records if x['root_id'] == root and x['probe_arm'] == arm]
                r.require(len(rs) == 2, 'ROOT_ARM_DENOMINATOR_CHANGED')
                arms[arm] = {'requests': 2, 'valid': sum(x['valid'] for x in rs), 'exact': statistics.mean(x['exact'] for x in rs), 'pair_accuracy': statistics.mean(x['pair_accuracy'] for x in rs)}
            byroot[root] = {'arms': arms, 'exact_difference': arms['explicit_positions']['exact'] - arms['freeze_quotes']['exact'],
                'pair_difference': arms['explicit_positions']['pair_accuracy'] - arms['freeze_quotes']['pair_accuracy']}
        pair_delta = statistics.mean(x['pair_difference'] for x in byroot.values())
        exact_delta = statistics.mean(x['exact_difference'] for x in byroot.values())
        positive = sum(x['pair_difference'] > 0 for x in byroot.values())
        passed = all(x['valid'] for x in records) and positive >= 3 and pair_delta > 0 and exact_delta >= 0
        results[label] = {'records': records, 'by_root': byroot, 'valid': sum(x['valid'] for x in records), 'fixed_requests': 12,
            'root_equal_pair_difference': pair_delta, 'root_equal_exact_difference': exact_delta, 'positive_pair_roots': positive,
            'new_prospective_controlled_support_gate_passed': passed, 'formal_reproduced_not_automatically_granted': True}
    b.save(run / 'SCORES.json', {'status': 'COMPLETE' if (run / 'COMPLETE.json').exists() else 'FAILED_FIXED_DENOMINATOR', 'model_role': role,
        'raw_freeze': b.binding(run / 'RAW_RESPONSE_FREEZE.json'), 'reserve': b.binding(HERE / 'NARROW_PROBE_RESERVE_V2.json'),
        'models': results, 'scope': 'Old source-qualified frozen-citation Train subset; no Gold parsed, final-evaluation or internal-mechanism claim.',
        'question_semantic_gold_unchanged': True, 'B1_admission_requires_separate_source_review_and_independent_audit': True})
    print(json.dumps({'role': role, 'controlled_support_gate': results[role]['new_prospective_controlled_support_gate_passed'],
        'pair_difference': results[role]['root_equal_pair_difference'], 'exact_difference': results[role]['root_equal_exact_difference']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    sub.add_parser('verify')
    for cmd in ('live', 'score'):
        q = sub.add_parser(cmd)
        q.add_argument('--role', choices=('A1', 'B1'), required=True)
        if cmd == 'live':
            q.add_argument('--release', type=Path, required=True)
            q.add_argument('--audit', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'verify':
        verify()
        print('NARROW_PROBE_CPU_VERIFY_PASS_NOT_RELEASED')
    elif args.command == 'live':
        live(args.role, args.release, args.audit)
    else:
        score(args.role)
