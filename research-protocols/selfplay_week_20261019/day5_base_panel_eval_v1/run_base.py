"""Versioned raw-base supplement to the frozen Day5 24-question direct panel."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
UPSTREAM = WEEK / 'day5_training_v1/train_diagnosis_prepare'
sys.path.insert(0, str(UPSTREAM))
import build_prepare as b
import run_v2 as inherited

MODEL = Path('/root/autodl-tmp/Qwen3-4B')
IDENTITY_KEYS = ('request_id', 'candidate_id', 'root_id', 'question_type', 'variant_id', 'slot', 'seed', 'prompt_sha256')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def now():
    return b.datetime.now(b.timezone.utc).isoformat()


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def prepare():
    require(not (HERE / 'PREPARATION_FREEZE.json').exists(), 'ALREADY_PREPARED')
    inherited.verify()
    previous = b.read(UPSTREAM / 'models/A1/EXECUTION_MANIFEST.json')
    require(sys.prefix == previous['python_prefix'], 'USE_INHERITED_EVALUATION_ENVIRONMENT')
    require(inherited.versions() == previous['runtime_versions'], 'RUNTIME_VERSION_DRIFT')
    require(previous['tokenizer_directory'] == str(MODEL), 'INHERITED_TOKENIZER_NOT_ORIGINAL')
    day1 = b.read(WEEK / 'day1/MODEL_BINDINGS.json')
    base = day1['models']['base']
    require(base['directory'] == str(MODEL), 'BASE_IDENTITY_DRIFT')
    inherited.check(base['files'] + previous['tokenizer_bindings'])
    a = inherited.adapter()
    _, solver = a.native()
    decoder = solver.generation_config('multi_select')
    require(decoder == previous['decoder_contract'], 'DECODER_CONTRACT_DRIFT')
    from scripts.current_method.restart_n5_spell import vllm_backends
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    requests = b.read(UPSTREAM / 'public/DIRECT_REQUESTS.json')['requests']
    require(len(requests) == 360 and len({r['candidate_id'] for r in requests}) == 24, 'PANEL_SCOPE_DRIFT')
    require(len({r['root_id'] for r in requests}) == 6, 'ROOT_SCOPE_DRIFT')
    expected_ids = {}
    raw_bindings = b.read(UPSTREAM / 'private/S0_BASELINE_BINDINGS.json')['selected_raw_responses']
    inherited.check(raw_bindings)
    old_paths = {Path(r['path']).stem: r['path'] for r in raw_bindings}
    choices = b.read(UPSTREAM / 'public/FINITE_CHOICES.json')
    inherited.check(choices['S0_batch_bindings'])
    for row in requests:
        path = Path(row['prompt_path'])
        require(b.sha(path) == row['prompt_sha256'], 'PROMPT_HASH_DRIFT')
        ids = tokenizer.encode(path.read_text(), add_special_tokens=False)
        raw = b.read(old_paths[row['request_id']])
        require(all(raw[k] == row[k] for k in IDENTITY_KEYS), 'OLD_REQUEST_IDENTITY_DRIFT')
        require(ids == raw['prompt_token_ids'], 'EXACT_PROMPT_TOKENS_DIFFER_FROM_S0')
        require(len(ids) == row['prompt_tokens'] and len(ids) + 32 <= 16384, 'PROMPT_LENGTH_DRIFT')
        expected_ids[row['request_id']] = token_hash(ids)
        field = 'options' if row['question_type'] == 'multi_select' else 'items'
        frozen = choices['choices'][choices['request_keys'][row['request_id']]]
        require(set(frozen) == set(a.choices(row['question_type'], row['display'][field])), 'ANSWER_SPACE_DRIFT')
    b.save(HERE / 'CPU_VALIDATION.json', {
        'status': 'PASS', 'created_at_utc': now(), 'questions': 24, 'roots': 6,
        'direct_requests': len(requests), 'input_tokens': sum(r['prompt_tokens'] for r in requests),
        'minimum_prompt_tokens': min(r['prompt_tokens'] for r in requests),
        'maximum_prompt_tokens': max(r['prompt_tokens'] for r in requests),
        'all_prompt_token_ids_equal_S0': True, 'prompt_token_ids_sha256': expected_ids,
        'zero_truncation': True, 'same_decoder_contract': True,
        'finite_choices_order': 'Read existing frozen arrays verbatim; never reconstruct or sort.',
        'new_solver_calls': 0, 'new_gold_json_reads': 0,
    })
    code_paths = [Path(__file__), HERE / 'score_and_report.py', UPSTREAM / 'run_v2.py',
                  UPSTREAM / 'build_prepare.py', WEEK / 'day2/solver/adapter.py',
                  WEEK / 'day3_solver187_v1/runtime/score.py',
                  WEEK / 'day3_solver187_v1/runtime/common.py',
                  Path(vllm_backends.__file__), Path(solver.__file__), a.NATIVE / 'scoring.py']
    upstream_paths = [UPSTREAM / 'public' / name for name in
                      ('DIRECT_REQUESTS.json', 'DIRECT_DISPLAY_MAPS.json', 'PUBLIC_QUESTIONS.json', 'FINITE_CHOICES.json')]
    upstream_paths += [UPSTREAM / 'SELECTION.json', UPSTREAM / 'EXECUTION_ADDENDUM_V2.json',
                       UPSTREAM / 'DIAGNOSIS_RECORDS_FREEZE.json', UPSTREAM / 'DIAGNOSIS_SUMMARY.json',
                       UPSTREAM.parent / 'FINAL_MANIFEST.json', WEEK / 'day1/MODEL_BINDINGS.json',
                       UPSTREAM / 'models/A1/EXECUTION_MANIFEST.json', UPSTREAM / 'models/B1/EXECUTION_MANIFEST.json',
                       UPSTREAM / 'models/A1/private/DIRECT_SCORES.json',
                       UPSTREAM / 'models/B1/private/DIRECT_SCORES.json']
    freeze = {
        'schema_version': 'day5_original_base_panel_supplement_v1',
        'created_at_utc': now(), 'status': 'FROZEN_BEFORE_BASE_INFERENCE',
        'authorization': 'User requested filling the missing original-base comparison on this same24-question panel.',
        'model_role': 'BASE', 'model_directory': str(MODEL), 'tokenizer_directory': str(MODEL),
        'model_bindings': base['files'], 'tokenizer_bindings': previous['tokenizer_bindings'],
        'code_bindings': [b.binding(p) for p in code_paths],
        'upstream_bindings': [b.binding(p) for p in upstream_paths],
        'prompt_bindings': [b.binding(Path(r['prompt_path'])) for r in requests],
        'cpu_validation': b.binding(HERE / 'CPU_VALIDATION.json'),
        'python_prefix': sys.prefix, 'runtime_versions': inherited.versions(), 'decoder_contract': decoder,
        'questions': 24, 'train_roots': 6, 'source_arm_questions': {'A': 12, 'B': 12},
        'requests': 360, 'max_model_len': 16384, 'batch_size': 8, 'gpu_memory_utilization': .8,
        'prefix_caching': False, 'automatic_retry': False, 'source_truncation': False,
        'new_DEV_TEST_calls': 0, 'new_brief_evidence_calls': 0, 'new_controlled_probe_calls': 0,
        'selection': 'Reuse exactly the already evaluated frozen24; no new score-based selection.',
        'historical_exposure': 'A1/B1 trained on own-arm12; opposite-arm shares roots/source/facts. Not heldout generalization.',
        'scope': 'New versioned supplementary evaluation; does not amend Day5 frozen records or change feedback eligibility.',
    }
    b.save(HERE / 'PREPARATION_FREEZE.json', freeze)
    b.save(HERE / 'EXECUTION_MANIFEST.json', {
        'status': 'ORIGINAL_BASE_BOUND_REVIEW_REQUIRED', 'created_at_utc': now(),
        'model_role': 'BASE', 'model_directory': str(MODEL), 'tokenizer_directory': str(MODEL),
        'preparation_freeze': b.binding(HERE / 'PREPARATION_FREEZE.json'),
        'model_bindings': base['files'], 'tokenizer_bindings': previous['tokenizer_bindings'],
        'runtime_versions': inherited.versions(), 'python_prefix': sys.prefix,
        'original_base_no_adapter': True, 'requests': 360,
    })
    print(json.dumps({'status': 'PREPARED', 'requests': 360, 'new_solver_calls': 0}), flush=True)


def execution_verify(audit_path):
    inherited.verify()
    manifest = b.read(HERE / 'EXECUTION_MANIFEST.json')
    inherited.check([manifest['preparation_freeze']])
    freeze = b.read(manifest['preparation_freeze']['path'])
    require(freeze['model_directory'] == str(MODEL) == manifest['model_directory'], 'NOT_ORIGINAL_BASE')
    require(manifest['original_base_no_adapter'] is True, 'ADAPTER_NOT_ALLOWED')
    require(freeze['python_prefix'] == sys.prefix and freeze['runtime_versions'] == inherited.versions(), 'RUNTIME_DRIFT')
    inherited.check(freeze['model_bindings'] + freeze['tokenizer_bindings'] + freeze['code_bindings'] +
                    freeze['upstream_bindings'] + freeze['prompt_bindings'] + [freeze['cpu_validation']])
    audit = b.read(audit_path)
    require(audit['status'] == 'PASS', 'INDEPENDENT_PRECHECK_NOT_PASS')
    require(any(r['path'] == str(HERE / 'EXECUTION_MANIFEST.json') and
                r['sha256'] == b.sha(HERE / 'EXECUTION_MANIFEST.json') for r in audit['audited_files']),
            'AUDIT_NOT_BOUND_TO_EXECUTION_MANIFEST')
    inherited.check(audit['audited_files'])
    return freeze


def live(audit_path):
    freeze = execution_verify(audit_path)
    run = HERE / 'direct'
    require(not run.exists(), 'NO_OVERWRITE_OR_AUTOMATIC_RERUN')
    requests = b.read(UPSTREAM / 'public/DIRECT_REQUESTS.json')['requests']
    expected_tokens = b.read(HERE / 'CPU_VALIDATION.json')['prompt_token_ids_sha256']
    choices = b.read(UPSTREAM / 'public/FINITE_CHOICES.json')
    b.save(run / 'STARTED.json', {'started_at_utc': now(), 'model_role': 'BASE',
        'request_file': b.binding(UPSTREAM / 'public/DIRECT_REQUESTS.json'),
        'execution_manifest': b.binding(HERE / 'EXECUTION_MANIFEST.json'),
        'audit': b.binding(audit_path), 'gold_opened': False})
    backend, submitted, done, failed = None, set(), set(), None
    started, load_seconds = time.perf_counter(), None
    try:
        a = inherited.adapter()
        _, solver = a.native()
        solver.ensure_vllm_environment()
        from scripts.current_method.restart_n5_spell.vllm_backends import VLLMSolverBackend
        from vllm import SamplingParams, sampling_params
        backend = VLLMSolverBackend(model_path=MODEL, tokenizer_path=MODEL, max_model_len=16384,
            temperature=.8, top_p=.9, max_tokens=32, gpu_memory_utilization=.8)
        load_seconds = backend.load_latency_seconds
        require(str(backend.model_identifier) == str(MODEL), 'BACKEND_IDENTITY_DRIFT')
        for offset in range(0, len(requests), 8):
            batch = requests[offset:offset + 8]
            prompts, params = [], []
            for row in batch:
                path = Path(row['prompt_path'])
                require(b.sha(path) == row['prompt_sha256'], 'PROMPT_DRIFT')
                prompts.append(path.read_text())
                kw = {'temperature': .8, 'top_p': .9, 'max_tokens': 32, 'seed': row['seed'],
                      'stop': ['</box>'], 'include_stop_str_in_output': True}
                finite = choices['choices'][choices['request_keys'][row['request_id']]]
                if freeze['decoder_contract']['structured_output_api'].startswith('structured_outputs='):
                    kw['structured_outputs'] = sampling_params.StructuredOutputsParams(choice=finite)
                else:
                    kw['guided_decoding'] = sampling_params.GuidedDecodingParams(choice=finite)
                params.append(SamplingParams(**kw))
            bid = f'batch{offset // 8:04d}'
            b.save(run / 'batches' / (bid + '_SUBMITTED.json'), {
                'request_ids': [r['request_id'] for r in batch], 'submitted_at_utc': now(),
                'automatic_retry': False,
                'sampling_params': [p.to_dict() if hasattr(p, 'to_dict') else repr(p) for p in params]})
            submitted.update(r['request_id'] for r in batch)
            outputs = backend.engine.generate(prompts, params, use_tqdm=False)
            for i, row in enumerate(batch):
                if i >= len(outputs):
                    continue
                out = outputs[i]
                completions = list(out.outputs)
                first = completions[0] if completions else None
                errors = []
                if out.prompt != prompts[i] or token_hash(out.prompt_token_ids) != expected_tokens[row['request_id']]:
                    errors.append('OUTPUT_EXACT_PROMPT_TOKEN_DRIFT')
                if len(completions) != 1:
                    errors.append('COMPLETION_COUNT_MISMATCH')
                b.save(run / 'responses' / (row['request_id'] + '.json'), {
                    **{k: row[k] for k in IDENTITY_KEYS}, 'status': 'RETURNED',
                    'raw_output': first.text if first else None, 'response_committed_at_utc': now(),
                    'model_role': 'BASE', 'model_directory': str(MODEL), 'branch': 'direct',
                    'output_validation_errors': errors, 'prompt_token_ids': out.prompt_token_ids,
                    'all_completions': [{'text': c.text, 'token_ids': c.token_ids,
                                         'finish_reason': c.finish_reason, 'stop_reason': c.stop_reason} for c in completions],
                    'metadata': {'finish_reason': first.finish_reason if first else None,
                        'stop_reason': first.stop_reason if first else None,
                        'prompt_token_count': len(out.prompt_token_ids),
                        'completion_token_count': len(first.token_ids) if first else None,
                        'temperature': .8, 'top_p': .9, 'max_tokens': 32, 'enable_thinking': False,
                        'seed': row['seed'], 'structured_output_api': freeze['decoder_contract']['structured_output_api']},
                    'gold_opened_by_solver': False})
                done.add(row['request_id'])
            require(len(outputs) == len(batch), 'OUTPUT_COUNT_MISMATCH')
            require(all(not b.read(run / 'responses' / (r['request_id'] + '.json'))['output_validation_errors']
                        for r in batch), 'RAW_OUTPUT_VALIDATION_FAILED')
            b.save(run / 'batches' / (bid + '_COMPLETE.json'), {
                'completed_at_utc': now(), 'request_ids': [r['request_id'] for r in batch]})
            print(json.dumps({'status': 'LIVE', 'model': 'BASE', 'completed': len(done), 'planned': 360}), flush=True)
    except BaseException as error:
        failed = {'error': str(error), 'exception_type': type(error).__name__, 'traceback': traceback.format_exc()}
    finally:
        if backend:
            try:
                backend.close()
            except BaseException as error:
                failed = failed or {'error': str(error), 'exception_type': type(error).__name__, 'traceback': traceback.format_exc()}
        for row in requests:
            if row['request_id'] not in done:
                b.save(run / 'responses' / (row['request_id'] + '.json'), {
                    **{k: row[k] for k in IDENTITY_KEYS},
                    'status': 'BACKEND_ERROR' if row['request_id'] in submitted else 'NOT_SUBMITTED',
                    'raw_output': None, 'response_committed_at_utc': now(), 'usage': None,
                    'usage_status': 'UNKNOWN_NOT_ZERO' if row['request_id'] in submitted else 'NO_REQUEST_SENT',
                    'failure': failed})
        b.save(run / ('FAILED.json' if failed else 'COMPLETE.json'), {
            'status': 'FAILED' if failed else 'COMPLETE', 'completed_at_utc': now(),
            'actual_submitted_requests': len(submitted), 'returned_requests': len(done),
            'model_load_wall_seconds': load_seconds, 'wall_seconds': time.perf_counter() - started,
            'failure': failed})
        raw_paths = [run / 'responses' / (r['request_id'] + '.json') for r in requests]
        require(set(raw_paths) == set((run / 'responses').glob('*.json')), 'RAW_SET_MISMATCH')
        b.save(run / 'RAW_RESPONSE_FREEZE.json', {'status': 'COMMITTED', 'created_at_utc': now(),
            'planned_request_count': 360, 'files': [b.binding(p) for p in raw_paths],
            'request_freeze': b.binding(HERE / 'PREPARATION_FREEZE.json'),
            'execution_manifest': b.binding(HERE / 'EXECUTION_MANIFEST.json'), 'gold_opened_by_live': False})
    if failed:
        raise RuntimeError('BASE_LIVE_FAILED_WITH_FIXED_DENOMINATOR:' + failed['error'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    live_parser = sub.add_parser('live')
    live_parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    else:
        live(args.audit.resolve())


if __name__ == '__main__':
    main()
