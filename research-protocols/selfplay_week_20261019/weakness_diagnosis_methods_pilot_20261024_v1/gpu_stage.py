#!/usr/bin/env python3
"""Independent B1 runner. CPU preparation never starts a model; no retries.

Jobs are a JSON list, or an object with a jobs/requests list. Relative input
paths resolve against the jobs file. Only declared prompts and schemas are
opened; diagnosis prompts may already contain the coordinator's scoring data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / 'thinking_switch_20261024_structured_v2/run.py'
MODEL_LEN, BATCH = 16384, 8
STRUCTURED_CONFIG = {'reasoning_parser': 'qwen3', 'enable_in_reasoning': False}


def need(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bind(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def save_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        handle.write(value)


def reference_module():
    spec = importlib.util.spec_from_file_location('weakness_pilot_readonly_structured_reference', REFERENCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def jobs_from(path):
    path = Path(path).resolve()
    document = read(path)
    rows = document if isinstance(document, list) else document.get('jobs', document.get('requests'))
    need(isinstance(rows, list) and rows, 'NONEMPTY_JOBS_REQUIRED')
    normalized = []
    for row in rows:
        need(isinstance(row, dict), 'JOB_OBJECT_REQUIRED')
        rid = row.get('request_id')
        need(isinstance(rid, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,199}', rid), 'SAFE_REQUEST_ID_REQUIRED')
        need(all(key in row for key in ('prompt_path', 'schema_path', 'sampling_seed', 'temperature', 'max_tokens', 'top_p')), 'MISSING_JOB_FIELDS:' + rid)
        seed, maximum = row['sampling_seed'], row['max_tokens']
        need(type(seed) is int and 0 <= seed < 2**63, 'BAD_SEED:' + rid)
        need(type(maximum) is int and 0 < maximum < MODEL_LEN, 'BAD_MAX_TOKENS:' + rid)
        for key in ('temperature', 'top_p'):
            need(type(row[key]) in (int, float) and math.isfinite(row[key]), 'BAD_SAMPLING_NUMBER:' + rid)
        need(row['temperature'] >= 0 and 0 < row['top_p'] <= 1, 'BAD_SAMPLING_RANGE:' + rid)
        item = {key: row[key] for key in ('request_id', 'sampling_seed', 'temperature', 'max_tokens', 'top_p')}
        for key in ('prompt_path', 'schema_path'):
            need(isinstance(row[key], str) and row[key], 'BAD_INPUT_PATH:' + rid)
            input_path = Path(row[key])
            item[key] = str((input_path if input_path.is_absolute() else path.parent / input_path).resolve())
        normalized.append(item)
    need(len(normalized) == len({row['request_id'] for row in normalized}), 'DUPLICATE_REQUEST_ID')
    return normalized


def preflight(jobs_path):
    reference = reference_module()
    orig = reference.orig
    endpoint, config = orig.endpoint_receipt()
    from transformers import AutoTokenizer
    import xgrammar as xgr
    from jsonschema import Draft202012Validator
    tokenizer = AutoTokenizer.from_pretrained(str(orig.TOKENIZER), local_files_only=True)
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=tokenizer.vocab_size), max_threads=4)
    rows = jobs_from(jobs_path)
    prepared = []
    for row in rows:
        item = {'job': row, 'error': None}
        try:
            prompt = Path(row['prompt_path']).read_text(encoding='utf-8')
            schema = read(row['schema_path'])
            need(isinstance(schema, dict), 'JSON_SCHEMA_OBJECT_REQUIRED')
            Draft202012Validator.check_schema(schema)
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            item.update(prompt=prompt, schema=schema, prompt_token_ids=ids,
                        source_bindings=[bind(row['prompt_path']), bind(row['schema_path'])])
            need(ids, 'EMPTY_PROMPT')
            need(len(ids) + row['max_tokens'] <= MODEL_LEN, 'CONTEXT_OVERFLOW_NO_TRUNCATION')
            compiler.compile_json_schema(json.dumps(schema), any_whitespace=True)
        except Exception:
            item['error'] = traceback.format_exc()
        prepared.append(item)
    return reference, endpoint, config, tokenizer, prepared


def backend_metrics(metric):
    if metric is None:
        return None
    fields = ('arrival_time', 'first_scheduled_time', 'first_token_time', 'last_token_time', 'finished_time', 'time_in_queue', 'scheduler_time', 'model_forward_time', 'model_execute_time')
    return {key: value for key in fields if isinstance((value := getattr(metric, key, None)), (int, float)) and math.isfinite(value)}


def capture(result):
    return {'backend_request_id': str(result.request_id), 'prompt': result.prompt,
            'prompt_token_ids': list(result.prompt_token_ids or []),
            'finished': getattr(result, 'finished', None),
            'backend_metrics': backend_metrics(getattr(result, 'metrics', None)),
            'outputs': [{'index': completion.index, 'raw_output': completion.text,
                         'output_token_ids': list(completion.token_ids),
                         'finish_reason': completion.finish_reason, 'stop_reason': completion.stop_reason}
                        for completion in result.outputs]}


def execute(args, reference, endpoint, config, tokenizer, prepared):
    orig = reference.orig
    out = Path(args.out).resolve()
    need(not out.exists(), 'FRESH_OUTPUT_DIRECTORY_REQUIRED')
    lock = open('/tmp/gxy-selfplay-gpu.lock', 'a+')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        lock.close()
        raise
    engine, failure, submitted = None, None, []
    started = time.perf_counter()
    rows = [item['job'] for item in prepared]
    ids = [row['request_id'] for row in rows]
    try:
        out.mkdir(parents=True)
        save(out / 'JOBS.json', {'stage': args.stage, 'jobs': rows})
        save(out / 'PROMPT_TOKEN_IDS.json', {item['job']['request_id']: item.get('prompt_token_ids') for item in prepared})
        for item in prepared:
            rid = item['job']['request_id']
            if 'prompt' in item:
                save_text(out / 'prompts' / (rid + '.txt'), item['prompt'])
                save(out / 'schemas' / (rid + '.json'), item['schema'])
            if item['error']:
                save(out / 'raw' / (rid + '.json'), {'stage': args.stage, 'request_id': rid, 'status': 'INPUT_ERROR', 'failure': item['error'], 'raw_output': None})
        prep = {'stage': args.stage, 'requests': len(ids), 'request_ids': ids,
                'input_errors': [item['job']['request_id'] for item in prepared if item['error']],
                'jobs_source_binding': bind(args.jobs), 'runner_binding': bind(__file__),
                'reference_runner_binding': bind(REFERENCE), 'config_receipt': config,
                'runtime_versions': orig.versions(), 'structured_outputs_config': STRUCTURED_CONFIG,
                'max_model_len': MODEL_LEN, 'batch_size': BATCH, 'automatic_retry': False,
                'independent_gold_files_opened': False, 'diagnosis_prompt_may_contain_gold': args.stage == 'diagnosis',
                'inputs': [{'request_id': item['job']['request_id'], 'bindings': item.get('source_bindings', []), 'prompt_tokens': len(item.get('prompt_token_ids', [])), 'max_tokens': item['job']['max_tokens']} for item in prepared]}
        save(out / 'PREPARATION.json', prep)
        save(out / 'STARTED.json', {'stage': args.stage, 'request_ids': ids, 'requests': len(ids), 'time_unix': time.time(), 'preparation_binding': bind(out / 'PREPARATION.json'), 'automatic_retry': False, 'independent_gold_files_opened': False})
        valid = [item for item in prepared if not item['error']]
        for bound in endpoint['model_bindings'] + endpoint['tokenizer_bindings']:
            orig.check(bound)
        if valid:
            from vllm import LLM, SamplingParams, sampling_params
            overrides = {'rope_theta': orig.EXPECTED_THETA, 'rope_parameters': {'rope_type': 'default', 'rope_theta': orig.EXPECTED_THETA}}
            engine = LLM(model=str(orig.MODEL), tokenizer=str(orig.TOKENIZER), tensor_parallel_size=1,
                         max_model_len=MODEL_LEN, gpu_memory_utilization=.8, dtype='bfloat16',
                         trust_remote_code=True, enable_prefix_caching=False, max_num_seqs=BATCH,
                         seed=0, hf_overrides=overrides, structured_outputs_config=STRUCTURED_CONFIG)
            cfg = engine.llm_engine.model_config
            actual = engine.llm_engine.vllm_config.structured_outputs_config
            actual_structured = {'reasoning_parser': actual.reasoning_parser, 'enable_in_reasoning': actual.enable_in_reasoning}
            need(Path(cfg.model).resolve() == orig.MODEL.resolve(), 'WRONG_MODEL')
            need(actual_structured == STRUCTURED_CONFIG, 'STRUCTURED_CONFIG_DRIFT')
            need(str(cfg.dtype) == 'torch.bfloat16' and cfg.max_model_len == MODEL_LEN, 'ACTUAL_DTYPE_OR_CONTEXT_DRIFT')
            receipt = {'stage': args.stage, 'model_role': 'B1', 'model_directory': str(orig.MODEL),
                       'tokenizer_directory': str(orig.TOKENIZER), 'actual_hf_config': orig.checked_runtime_config(cfg.hf_config),
                       'actual_hf_text_config': orig.checked_runtime_config(cfg.hf_text_config),
                       'actual_structured_outputs_config': actual_structured, 'structured_outputs_config': STRUCTURED_CONFIG,
                       'actual_model_config': {'model': str(cfg.model), 'tokenizer': str(cfg.tokenizer), 'dtype': str(cfg.dtype), 'max_model_len': cfg.max_model_len},
                       'engine_parameters': {'dtype': 'bfloat16', 'max_model_len': MODEL_LEN, 'tensor_parallel_size': 1, 'enable_prefix_caching': False, 'max_num_seqs': BATCH, 'seed': 0, 'hf_overrides': overrides},
                       'config_receipt': config, 'runtime_versions': orig.versions(), 'runner_binding': bind(__file__)}
            save(out / 'MODEL_LOAD_RECEIPT.json', receipt)
            print('MODEL_LOADED ' + args.stage, flush=True)
            for start in range(0, len(valid), BATCH):
                batch = valid[start:start + BATCH]
                index, batch_start, batch_wall = start // BATCH, time.time(), time.perf_counter()
                batch_path = out / 'batches' / (str(index).zfill(3) + '.json')
                record = {'batch_index': index, 'request_ids': [item['job']['request_id'] for item in batch], 'started_unix': batch_start, 'returned_results': [], 'failure': None}
                try:
                    params = [SamplingParams(temperature=item['job']['temperature'], top_p=item['job']['top_p'], top_k=-1,
                              repetition_penalty=1.0, max_tokens=item['job']['max_tokens'], seed=item['job']['sampling_seed'],
                              skip_special_tokens=False, structured_outputs=sampling_params.StructuredOutputsParams(json=item['schema'])) for item in batch]
                    for item in batch:
                        for bound in item['source_bindings']:
                            need(bind(bound['path']) == bound, 'INPUT_CHANGED_AFTER_PREFLIGHT')
                        rid = item['job']['request_id']
                        save(out / 'submitted' / (rid + '.json'), {'stage': args.stage, 'request_id': rid, 'sampling': item['job'], 'batch_index': index, 'time_unix': time.time(), 'model_load_receipt_binding': bind(out / 'MODEL_LOAD_RECEIPT.json')})
                        submitted.append(rid)
                    results = engine.generate([item['prompt'] for item in batch], params, use_tqdm=False)
                    record['returned_results'] = [capture(result) for result in results]
                    need(len(results) == len(batch), 'BACKEND_RESULT_COUNT_MISMATCH')
                    backend_ids = [result['backend_request_id'] for result in record['returned_results']]
                    need(len(backend_ids) == len(set(backend_ids)), 'DUPLICATE_BACKEND_REQUEST_ID')
                    capture_failures = []
                    for item, captured in zip(batch, record['returned_results']):
                        rid = item['job']['request_id']
                        errors = []
                        if captured['prompt'] != item['prompt']: errors.append('RETURNED_PROMPT_DRIFT')
                        if captured['prompt_token_ids'] != item['prompt_token_ids']: errors.append('PROMPT_TOKEN_IDS_DRIFT')
                        if len(captured['outputs']) != 1: errors.append('COMPLETION_COUNT_MISMATCH')
                        raw = {'stage': args.stage, 'request_id': rid, 'status': 'RETURNED_INVALID_CAPTURE' if errors else 'RETURNED',
                               'model_role': 'B1', 'model_directory': str(orig.MODEL), 'sampling_seed': item['job']['sampling_seed'],
                               'sampling': {key: item['job'][key] for key in ('sampling_seed', 'temperature', 'max_tokens', 'top_p')},
                               'batch_index': index, 'capture_errors': errors, 'returned_result_count': 1,
                               'returned_results': [captured], 'prompt_token_ids': captured['prompt_token_ids'],
                               'raw_output': None, 'output_token_ids': [], 'finish_reason': None, 'stop_reason': None,
                               'independent_gold_files_opened': False, 'model_load_receipt_binding': bind(out / 'MODEL_LOAD_RECEIPT.json')}
                        if captured['outputs']:
                            raw.update(captured['outputs'][0])
                            raw['decoded_output_skip_special_tokens_false'] = tokenizer.decode(raw['output_token_ids'], skip_special_tokens=False)
                        save(out / 'raw' / (rid + '.json'), raw)
                        capture_failures.extend(errors)
                    need(not capture_failures, 'BACKEND_ASSOCIATION_OR_COMPLETION_DRIFT:' + ','.join(capture_failures))
                except BaseException:
                    record['failure'] = traceback.format_exc()
                    raise
                finally:
                    record.update(finished_unix=time.time(), wall_seconds=time.perf_counter() - batch_wall)
                    save(batch_path, record)
                print(json.dumps({'stage': args.stage, 'completed': start + len(batch), 'total_valid_inputs': len(valid), 'batch_seconds': record['wall_seconds']}), flush=True)
    except BaseException:
        failure = traceback.format_exc()
        print(failure, file=sys.stderr, flush=True)
    finally:
        if engine is not None:
            try:
                core = getattr(engine.llm_engine, 'engine_core', None)
                if callable(getattr(core, 'shutdown', None)):
                    core.shutdown()
            except BaseException:
                failure = (failure or '') + traceback.format_exc()
        if out.exists():
            for row in rows:
                path = out / 'raw' / (row['request_id'] + '.json')
                if not path.exists():
                    save(path, {'stage': args.stage, 'request_id': row['request_id'], 'status': 'BACKEND_ERROR' if row['request_id'] in submitted else 'NOT_SUBMITTED', 'raw_output': None, 'failure': failure})
            paths = [out / 'raw' / (rid + '.json') for rid in ids]
            statuses = Counter(read(path)['status'] for path in paths)
            freeze = {'stage': args.stage, 'requests': len(ids), 'request_ids': ids, 'submitted_ids': submitted,
                      'returned': statuses['RETURNED'], 'status_counts': dict(statuses), 'failure': failure,
                      'wall_seconds': time.perf_counter() - started, 'automatic_retry': False,
                      'bindings': [bind(path) for path in paths],
                      'submission_bindings': [bind(path) for path in sorted((out / 'submitted').glob('*.json'))],
                      'batch_bindings': [bind(path) for path in sorted((out / 'batches').glob('*.json'))],
                      'model_receipt_binding': bind(out / 'MODEL_LOAD_RECEIPT.json') if (out / 'MODEL_LOAD_RECEIPT.json').exists() else None,
                      'preparation_binding': bind(out / 'PREPARATION.json') if (out / 'PREPARATION.json').exists() else None,
                      'started_binding': bind(out / 'STARTED.json') if (out / 'STARTED.json').exists() else None,
                      'prompt_token_ids_binding': bind(out / 'PROMPT_TOKEN_IDS.json') if (out / 'PROMPT_TOKEN_IDS.json').exists() else None,
                      'prompt_bindings': [bind(path) for path in sorted((out / 'prompts').glob('*.txt'))],
                      'schema_bindings': [bind(path) for path in sorted((out / 'schemas').glob('*.json'))]}
            need(len(paths) == len(ids) == sum(statuses.values()), 'RAW_COMPLETENESS_FAILURE')
            save(out / 'RAW_RESPONSE_FREEZE.json', freeze)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
    if failure or any(read(out / 'raw' / (rid + '.json'))['status'] != 'RETURNED' for rid in ids):
        raise RuntimeError('Stage contains frozen failures; no automatic retry')
    return freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('solver', 'diagnosis'), required=True)
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cpu-check', action='store_true', help='Tokenize/compile schemas only; do not create outputs or start GPU.')
    parser.add_argument('--execute-gpu', action='store_true', help='Explicitly allow model loading and generation.')
    args = parser.parse_args()
    need(args.cpu_check != args.execute_gpu, 'CHOOSE_EXACTLY_ONE_OF_CPU_CHECK_OR_EXECUTE_GPU')
    for key, value in {'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'VLLM_USE_FLASHINFER_SAMPLER': '0', 'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false'}.items():
        os.environ[key] = value
    reference, endpoint, config, tokenizer, prepared = preflight(args.jobs)
    if args.cpu_check:
        summary = {'status': 'CPU_CHECK_ONLY_NO_MODEL_STARTED', 'stage': args.stage, 'requests': len(prepared),
                   'jobs': [{'request_id': item['job']['request_id'], 'prompt_tokens': len(item.get('prompt_token_ids', [])), 'max_tokens': item['job']['max_tokens'], 'error': item['error']} for item in prepared], 'gpu_model_started': False}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        need(all(item['error'] is None for item in prepared), 'CPU_INPUT_CHECK_FAILED')
    else:
        result = execute(args, reference, endpoint, config, tokenizer, prepared)
        print(json.dumps({'stage': args.stage, 'requests': result['requests'], 'status_counts': result['status_counts']}))


if __name__ == '__main__':
    main()
