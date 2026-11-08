"""CPU verify/bind/score and explicitly audited live Train-diagnosis launcher.

Only live instantiates a GPU backend. Probe reserve has no live entry here.
"""
from __future__ import annotations
import argparse
import collections
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WEEK = HERE.parents[1]
sys.path.insert(0, str(HERE))
import build_prepare as b


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def now():
    return b.datetime.now(b.timezone.utc).isoformat()


def check(rows):
    for r in rows:
        p = Path(r['path'])
        require(p.is_file() and p.stat().st_size == r['bytes'] and b.sha(p) == r['sha256'], 'FROZEN_FILE_CHANGED:' + str(p))


def versions():
    return {k: importlib.metadata.version(k) for k in ('torch', 'transformers', 'tokenizers', 'vllm')}


def verify():
    f = b.read(HERE / 'EXECUTION_ADDENDUM_V2.json')
    check(f['files'] + f['upstream_bindings'] + f['upstream_direct_prompt_bindings'])
    require(f['questions'] == 24 and f['roots'] == 6 and f['direct_requests_per_model'] == 360 and f['dev_test_requests'] == 0, 'SCOPE_DRIFT')
    for q in b.read(HERE / 'public/PUBLIC_QUESTIONS.json')['questions']:
        require(b.sha(q['source_path']) == q['source_sha256'], 'SOURCE_CHANGED')
    choice = b.read(HERE / 'public/FINITE_CHOICES.json')
    check(choice['S0_batch_bindings'])
    return f


def adapter():
    return b.import_file('day5_train_native_adapter', WEEK / 'day2/solver/adapter.py')


def bind(role):
    verify()
    expected = WEEK.parents[1] / '.venv-local4b'
    require(Path(sys.prefix).resolve() == expected.resolve(), 'USE_LOCAL4B_EVAL_ENVIRONMENT')
    # Reuse exact final endpoint proof already prepared for common DEV, never DEV responses.
    ep = HERE.parent / 'evaluation/models' / role / 'EXECUTION_MANIFEST.json'
    e = b.read(ep)
    require(e['model_role'] == role, 'ENDPOINT_ROLE_MISMATCH')
    proof = b.read(e['endpoint_proof']['path'])
    require(proof['training_completed'] is True and proof['completed_optimizer_steps'] == proof['fixed_final_optimizer_step'] == 56, 'FINAL_STEP56_REQUIRED')
    require(proof['independent_initialization'] == 'S0' and proof['merge_base'] == '/root/autodl-tmp/Qwen3-4B' and proof['adapter_representation'] == 'FULL_UPDATED_S0_CUSTOM_LORA', 'WRONG_MODEL_LINEAGE')
    check(e['model_bindings'] + e['tokenizer_bindings'] + e['code_bindings'] + e['endpoint_evidence_bindings'] + [e['endpoint_proof']])
    a = adapter()
    _, solver = a.native()
    from scripts.current_method.restart_n5_spell import vllm_backends
    decoder = solver.generation_config('multi_select')
    require(decoder['max_tokens'] == 32 and decoder['temperature'] == .8 and decoder['top_p'] == .9 and decoder['enable_thinking'] is False, 'DECODER_DRIFT')
    frozen_choices = b.read(HERE / 'public/FINITE_CHOICES.json')
    for r in b.read(HERE / 'public/DIRECT_REQUESTS.json')['requests']:
        field = 'options' if r['question_type'] == 'multi_select' else 'items'
        expected_choices = frozen_choices['choices'][frozen_choices['request_keys'][r['request_id']]]
        actual_choices = a.choices(r['question_type'], r['display'][field])
        require(len(actual_choices) == len(expected_choices) and set(actual_choices) == set(expected_choices), 'CHOICE_SPACE_CHANGED')
    code = [b.binding(p) for p in (Path(__file__), HERE / 'build_prepare.py', WEEK / 'day2/solver/adapter.py', Path(vllm_backends.__file__), Path(solver.__file__))]
    b.save(HERE / 'models' / role / 'EXECUTION_MANIFEST.json', {'status': 'BOUND_FINAL_ENDPOINT_REVIEW_REQUIRED',
        'model_role': role, 'model_directory': e['model_directory'], 'tokenizer_directory': e['tokenizer_directory'],
        'endpoint_manifest': b.binding(ep), 'endpoint_proof': e['endpoint_proof'], 'endpoint_evidence_bindings': e['endpoint_evidence_bindings'],
        'model_bindings': e['model_bindings'], 'tokenizer_bindings': e['tokenizer_bindings'], 'code_bindings': code,
        'request_freeze': b.binding(HERE / 'EXECUTION_ADDENDUM_V2.json'), 'python_prefix': sys.prefix, 'runtime_versions': versions(),
        'decoder_contract': decoder, 'created_at_utc': now(), 'training_diagnosis_only': True, 'dev_test_calls': 0,
        'direct_requests': 360, 'brief_evidence_requests': 24, 'probe_requests_released': 0})


def execution_verify(role, audit):
    verify()
    p = HERE / 'models' / role / 'EXECUTION_MANIFEST.json'
    m = b.read(p)
    require(m['model_role'] == role and m['runtime_versions'] == versions() and m['python_prefix'] == sys.prefix, 'MODEL_OR_RUNTIME_CHANGED')
    check(m['model_bindings'] + m['tokenizer_bindings'] + m['code_bindings'] + m['endpoint_evidence_bindings'] + [m['endpoint_manifest'], m['endpoint_proof'], m['request_freeze']])
    a = b.read(audit)
    require(a['status'] == 'PASS' and any(x['path'] == str(p) and x['sha256'] == b.sha(p) for x in a.get('audited_files', [])), 'INDEPENDENT_AUDIT_NOT_BOUND_TO_MANIFEST')
    check(a['audited_files'])
    return m


def live(role, branch, audit):
    m = execution_verify(role, audit)
    run = HERE / 'models' / role / branch
    require(not run.exists(), 'NO_AUTOMATIC_RERUN_OR_OVERWRITE')
    rowfile = HERE / 'public' / ('DIRECT_REQUESTS.json' if branch == 'direct' else 'BRIEF_EVIDENCE_REQUESTS.json')
    rows = b.read(rowfile)['requests']
    require(len(rows) == (360 if branch == 'direct' else 24), 'REQUEST_COUNT_CHANGED')
    b.save(run / 'STARTED.json', {'started_at_utc': now(), 'request_file': b.binding(rowfile), 'execution_manifest': b.binding(HERE / 'models' / role / 'EXECUTION_MANIFEST.json'), 'audit': b.binding(audit), 'Gold_opened': False})
    backend, submitted, done, failed = None, set(), set(), None
    started, load_seconds = time.perf_counter(), None
    try:
        a = adapter()
        _, solver = a.native()
        solver.ensure_vllm_environment()
        from scripts.current_method.restart_n5_spell.vllm_backends import VLLMSolverBackend
        from vllm import SamplingParams, sampling_params
        frozen_choices = b.read(HERE / 'public/FINITE_CHOICES.json')
        maximum = 32 if branch == 'direct' else 2048
        backend = VLLMSolverBackend(model_path=Path(m['model_directory']), tokenizer_path=Path(m['tokenizer_directory']), max_model_len=16384,
            temperature=.8, top_p=.9, max_tokens=maximum, gpu_memory_utilization=.8)
        load_seconds = backend.load_latency_seconds
        require(str(backend.model_identifier) == m['model_directory'], 'BACKEND_IDENTITY_DRIFT')
        for offset in range(0, len(rows), 8):
            batch = rows[offset:offset + 8]
            prompts, params = [], []
            for r in batch:
                p = Path(r['prompt_path'])
                require(b.sha(p) == r['prompt_sha256'], 'PROMPT_DRIFT')
                prompts.append(p.read_text())
                if branch == 'direct':
                    kw = {'temperature': .8, 'top_p': .9, 'max_tokens': 32, 'seed': r['seed'], 'stop': ['</box>'], 'include_stop_str_in_output': True}
                    choice = frozen_choices['choices'][frozen_choices['request_keys'][r['request_id']]]
                    if m['decoder_contract']['structured_output_api'].startswith('structured_outputs='):
                        kw['structured_outputs'] = sampling_params.StructuredOutputsParams(choice=choice)
                    else:
                        kw['guided_decoding'] = sampling_params.GuidedDecodingParams(choice=choice)
                    params.append(SamplingParams(**kw))
                else:
                    params.append(SamplingParams(temperature=.8, top_p=.9, max_tokens=2048, seed=r['seed']))
            bid = 'batch' + str(offset // 8).zfill(4)
            b.save(run / 'batches' / (bid + '_SUBMITTED.json'), {'request_ids': [r['request_id'] for r in batch], 'submitted_at_utc': now(), 'automatic_retry': False,
                'sampling_params': [p.to_dict() if hasattr(p, 'to_dict') else repr(p) for p in params]})
            submitted.update(r['request_id'] for r in batch)
            outputs = backend.engine.generate(prompts, params, use_tqdm=False)
            # Commit every returned payload before deciding whether output shape drift is fatal.
            for i, r in enumerate(batch):
                if i >= len(outputs):
                    continue
                out = outputs[i]
                completions = list(out.outputs)
                first = completions[0] if completions else None
                errors = []
                if out.prompt != prompts[i] or len(out.prompt_token_ids) != r['prompt_tokens']:
                    errors.append('OUTPUT_PROMPT_OR_TOKEN_DRIFT')
                if len(completions) != 1:
                    errors.append('COMPLETION_COUNT_MISMATCH')
                b.save(run / 'responses' / (r['request_id'] + '.json'), {**{k: r[k] for k in ('request_id', 'candidate_id', 'root_id', 'question_type', 'variant_id', 'slot', 'seed', 'prompt_sha256')},
                    'status': 'RETURNED', 'raw_output': first.text if first else None, 'response_committed_at_utc': now(),
                    'model_role': role, 'model_directory': m['model_directory'], 'branch': branch, 'output_validation_errors': errors,
                    'prompt_token_ids': out.prompt_token_ids, 'all_completions': [{'text': c.text, 'token_ids': c.token_ids, 'finish_reason': c.finish_reason, 'stop_reason': c.stop_reason} for c in completions],
                    'metadata': {'finish_reason': first.finish_reason if first else None, 'stop_reason': first.stop_reason if first else None,
                        'prompt_token_count': len(out.prompt_token_ids), 'completion_token_count': len(first.token_ids) if first else None,
                        'temperature': .8, 'top_p': .9, 'max_tokens': maximum, 'enable_thinking': False, 'seed': r['seed'],
                        'structured_output_api': m['decoder_contract']['structured_output_api'] if branch == 'direct' else None}, 'gold_opened_by_solver': False})
                done.add(r['request_id'])
            if len(outputs) > len(batch):
                b.save(run / 'batches' / (bid + '_EXTRA_OUTPUTS.json'), {'outputs': [{'request_id': x.request_id, 'prompt': x.prompt, 'completions': [{'text': z.text, 'token_ids': z.token_ids} for z in x.outputs]} for x in outputs[len(batch):]]})
            require(len(outputs) == len(batch), 'OUTPUT_COUNT_MISMATCH')
            require(all(not b.read(run / 'responses' / (r['request_id'] + '.json'))['output_validation_errors'] for r in batch), 'OUTPUT_VALIDATION_FAILED_AFTER_RAW_COMMIT')
            b.save(run / 'batches' / (bid + '_COMPLETE.json'), {'completed_at_utc': now(), 'request_ids': [r['request_id'] for r in batch]})
            print(json.dumps({'status': 'LIVE', 'role': role, 'branch': branch, 'completed': len(done), 'planned': len(rows)}), flush=True)
    except BaseException as e:
        failed = {'error': str(e), 'exception_type': type(e).__name__, 'traceback': traceback.format_exc()}
    finally:
        if backend:
            try:
                backend.close()
            except BaseException as e:
                failed = failed or {'error': str(e), 'exception_type': type(e).__name__, 'traceback': traceback.format_exc()}
        for r in rows:
            if r['request_id'] not in done:
                b.save(run / 'responses' / (r['request_id'] + '.json'), {**{k: r[k] for k in ('request_id', 'candidate_id', 'root_id', 'question_type', 'variant_id', 'slot', 'seed', 'prompt_sha256')},
                    'status': 'BACKEND_ERROR' if r['request_id'] in submitted else 'NOT_SUBMITTED', 'raw_output': None,
                    'response_committed_at_utc': now(), 'usage': None, 'usage_status': 'UNKNOWN_NOT_ZERO' if r['request_id'] in submitted else 'NO_REQUEST_SENT', 'failure': failed})
        b.save(run / ('FAILED.json' if failed else 'COMPLETE.json'), {'status': 'FAILED' if failed else 'COMPLETE', 'completed_at_utc': now(),
            'actual_submitted_requests': len(submitted), 'returned_requests': len(done), 'model_load_wall_seconds': load_seconds, 'wall_seconds': time.perf_counter() - started, 'failure': failed})
        files = [run / 'responses' / (r['request_id'] + '.json') for r in rows]
        require(set(files) == set((run / 'responses').glob('*.json')), 'RAW_RESPONSE_SET_MISMATCH')
        b.save(run / 'RAW_RESPONSE_FREEZE.json', {'status': 'COMMITTED', 'created_at_utc': now(), 'files': [b.binding(p) for p in files],
            'planned_request_count': len(rows), 'request_freeze': b.binding(HERE / 'EXECUTION_ADDENDUM_V2.json'), 'execution_manifest': b.binding(HERE / 'models' / role / 'EXECUTION_MANIFEST.json'), 'gold_opened_by_live': False})
    if failed:
        raise RuntimeError('LIVE_FAILED_WITH_FIXED_DENOMINATOR:' + failed['error'])


def score(role):
    verify()
    run = HERE / 'models' / role
    freezes = []
    for branch in ('direct', 'brief_evidence'):
        f = b.read(run / branch / 'RAW_RESPONSE_FREEZE.json')
        check(f['files'] + [f['request_freeze'], f['execution_manifest']])
        require(f['status'] == 'COMMITTED' and f['planned_request_count'] == (360 if branch == 'direct' else 24), 'RAW_FREEZE_INCOMPLETE')
        freezes.append(b.binding(run / branch / 'RAW_RESPONSE_FREEZE.json'))
    b.save(run / 'SCORING_STARTED.json', {'started_at_utc': now(), 'raw_freezes': freezes, 'all_raw_committed_before_gold': True})
    base = b.read(HERE / 'private/S0_BASELINE_BINDINGS.json')
    check([base['gold_binding']] + base['selected_raw_responses'])
    gold = {q['candidate_id']: q for q in b.read(base['gold_binding']['path'])['questions']}
    sys.path.insert(0, str(WEEK / 'day3_solver187_v1/runtime'))
    scorer = b.import_file('day5_reused_score187', WEEK / 'day3_solver187_v1/runtime/score.py')
    maps = b.read(HERE / 'public/DIRECT_DISPLAY_MAPS.json')
    rows = b.read(HERE / 'public/DIRECT_REQUESTS.json')['requests']
    pub = b.read(HERE / 'public/PUBLIC_QUESTIONS.json')['questions']
    annotations = {q['candidate_id']: q for q in b.read(HERE / 'SELECTION.json')['question_annotations']}
    out = {}
    for label in ('S0', role):
        records = []
        for row in rows:
            rp = WEEK / 'day3_solver187_v1/runtime/direct_v2/responses' / (row['request_id'] + '.json') if label == 'S0' else run / 'direct/responses' / (row['request_id'] + '.json')
            raw = b.read(rp)
            require(all(raw[k] == row[k] for k in ('request_id', 'candidate_id', 'root_id', 'question_type', 'variant_id', 'slot', 'seed', 'prompt_sha256')), 'RAW_IDENTITY_DRIFT')
            rec = scorer.score_one(row, raw, maps['requests'][row['request_id']], maps['public_to_canonical'][row['candidate_id']], gold[row['candidate_id']]['gold'])
            records.append(rec)
        questions = [scorer.summarize_question(q, [r for r in records if r['candidate_id'] == q['candidate_id']]) | annotations[q['candidate_id']] for q in pub]
        groups = {}
        for typ in ('ALL', 'multi_select', 'ordering'):
            for exposure in ('ALL', 'OWN_ARM_TRAINED_QUESTION', 'OPPOSITE_ARM_QUESTION_SHARED_ROOT_MAY_SHARE_FACTS'):
                qs = [q for q in questions if (typ == 'ALL' or q['question_type'] == typ) and (exposure == 'ALL' or q['model_training_exposure'][role] == exposure)]
                roots = collections.defaultdict(list)
                for q in qs:
                    roots[q['root_id']].append(q['exact_rate'])
                groups[typ + '/' + exposure] = {'questions': len(qs), 'roots': len(roots), 'question_equal_exact': statistics.mean(q['exact_rate'] for q in qs),
                    'root_equal_exact': statistics.mean(statistics.mean(v) for v in roots.values()), 'invalid_calls': sum(q['invalid_calls'] for q in qs)}
        out[label] = {'requests': records, 'questions': questions, 'groups': groups}
    b.save(run / 'private/DIRECT_SCORES.json', {'role': role, 'scope': 'TRAIN_DIAGNOSIS_ONLY', 'models': out, 'dev_test_requests': 0, 'not_for_A_challenger': True})
    b.save(run / 'DIAGNOSIS_REVIEW_REQUIRED.json', {'status': 'DIRECT_SCORED_BRIEF_EVIDENCE_SEMANTIC_REVIEW_PENDING', 'direct_scores': b.binding(run / 'private/DIRECT_SCORES.json'),
        'source_review_required_before_feedback': True, 'B1_reproduced': False, 'controlled_probe_release': False,
        'next': 'Strictly parse evidence without answer rescue, review quote/turn/semantic support against full source, classify source-to-judgment and judgment-to-answer errors, then version B1 feedback and apply original operational gate.'})
    print(json.dumps({'role': role, 'status': 'DIRECT_SCORED_REVIEW_PENDING', 'groups': out[role]['groups']}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('verify')
    for name in ('bind', 'live', 'score'):
        q = sub.add_parser(name)
        q.add_argument('--role', choices=('A1', 'B1'), required=True)
        if name == 'live':
            q.add_argument('--branch', choices=('direct', 'brief_evidence'), required=True)
            q.add_argument('--audit', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'verify':
        verify()
        print('PASS_CPU_PREPARED_NOT_EXECUTED')
    elif a.command == 'bind':
        bind(a.role)
    elif a.command == 'live':
        live(a.role, a.branch, a.audit)
    else:
        score(a.role)


if __name__ == '__main__':
    main()
