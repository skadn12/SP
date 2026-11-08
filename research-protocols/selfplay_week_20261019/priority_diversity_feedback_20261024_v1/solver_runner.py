#!/usr/bin/env python3
"""Train-only B1 visible-reason evaluation; prepare, generate, then score.

Original experiments are imported read-only. Generation opens only frozen public
requests, their prompts/schemas, and endpoint/config files. Gold is scored only
after the complete raw response set is sealed. No training or automatic retry.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
REPO = HERE.parents[2]
BASE_PATH = WEEK / 'postweek_question_quality_pilot_v1/run.py'
COMPAT_PATH = WEEK / 'overnight_endtoend_20261024_v1/followup_repair_v1/evaluation/model_compat.py'
PRODUCTION_GATE_PATH = REPO / 'scripts/current_method/production_question_gate.py'
NEW = HERE / 'CANDIDATES_PRIVATE.json'
REVIEW = HERE / 'CONTENT_REVIEW_AI.json'
SELECTION = HERE / 'SELECTION_FREEZE.json'
OPERATION_PATH = HERE / 'OPERATION_INSTRUCTIONS.json'
OUT = HERE / 'execution'
MODEL = REPO / 'restart-n5-runs/day5_B1_schedule_v1/merged_model'
TOKENIZER = Path('/root/autodl-tmp/Qwen3-4B')
PYTHON_PREFIX = REPO / '.venv-local4b'
CONDITIONS = ('FULL', 'QUESTION_ONLY')
VARIANTS = ('perm_00', 'perm_01')
DISPLAY_SEED = 2026102411
MAX_TOKENS, MODEL_LEN = 3072, 16384
MAX_QUESTIONS, MAX_REQUESTS = 32, 128
EXPECTED_THETA = 1_000_000


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


base = module('priority_visible_reason_frozen_reuse', BASE_PATH)
compat = module('priority_checked_model_config', COMPAT_PATH)
sys.path.insert(0, str(PRODUCTION_GATE_PATH.parent))
production_gate = module('priority_frozen_production_gate', PRODUCTION_GATE_PATH)
read, save, bind, check = base.read, base.save, base.bind, base.check
fingerprint, gate, diagnostic, capture = base.fingerprint, base.gate, base.diagnostic, base.capture
PUBLIC_KEYS = base.PUBLIC_KEYS | {'variant_id'}


def require(value, code):
    if not value:
        raise ValueError(code)


def versions():
    return {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'tokenizers', 'vllm', 'xgrammar')}


def endpoint_receipt():
    require(Path(sys.prefix).resolve() == PYTHON_PREFIX.resolve(), 'WRONG_PYTHON_ENVIRONMENT')
    endpoint = read(diagnostic.ENDPOINT)
    require(Path(endpoint['model_directory']).resolve() == MODEL.resolve(), 'WRONG_B1_ENDPOINT')
    require(Path(endpoint['tokenizer_directory']).resolve() == TOKENIZER.resolve(), 'WRONG_TOKENIZER_ENDPOINT')
    config, receipt = compat.load_compatible_config(MODEL, TOKENIZER)
    require(config.rope_theta == EXPECTED_THETA, 'CHECKED_THETA_MISMATCH')
    require(config.rope_parameters == {'rope_type': 'default', 'rope_theta': EXPECTED_THETA}, 'CHECKED_NESTED_THETA_MISMATCH')
    return endpoint, receipt


def operation_instructions():
    extra = read(OPERATION_PATH) if OPERATION_PATH.exists() else {}
    require(isinstance(extra, dict) and all(isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in extra.items()), 'INVALID_OPERATION_INSTRUCTIONS')
    diagnostic.OPERATIONS.update(extra)
    return extra


def sampling_seed(candidate_id):
    return int.from_bytes(hashlib.sha256(('priority_diversity_feedback_20261024_v1:' + candidate_id).encode()).digest()[:4], 'big') % 2147483647


def displays(candidate):
    field = 'items' if candidate['question_type'] == 'ordering' else 'options'
    require(4 <= len(candidate[field]) <= 6, 'DISPLAY_ITEM_COUNT_LIMIT')
    seen = set()
    for index, variant in enumerate(VARIANTS):
        seed, skipped = DISPLAY_SEED + index, []
        while True:
            public, private = gate.prepare_display(candidate, seed)
            permutation = tuple(private['display_to_canonical'].values())
            if permutation not in seen:
                break
            skipped.append(seed)
            seed += len(VARIANTS)
        seen.add(permutation)
        private['duplicate_display_seed_skips'] = skipped
        yield variant, public, private


def code_paths():
    paths = [Path(__file__), BASE_PATH, COMPAT_PATH, PRODUCTION_GATE_PATH,
             base.GATE_PATH, base.DIAGNOSTIC_PATH, base.CAPTURE_PATH,
             diagnostic.REPAIR_PATH, diagnostic.ENDPOINT]
    if OPERATION_PATH.exists():
        paths.append(OPERATION_PATH)
    return paths


def prepare():
    require(not OUT.exists(), 'NO_PREPARATION_OVERWRITE')
    endpoint, config_receipt = endpoint_receipt()
    operation_instructions()
    selection = read(SELECTION)
    roots = {row['root_id']: row for row in selection['roots']}
    require(0 < len(roots) == len(selection['roots']) <= 4, 'EXPECTED_UP_TO_FOUR_UNIQUE_TRAIN_ROOTS')
    require(all(row.get('split') == 'train' for row in roots.values()), 'TRAIN_SOURCES_ONLY')
    planned = selection.get('planned_questions', MAX_QUESTIONS)
    require(type(planned) is int and 0 < planned <= MAX_QUESTIONS, 'INVALID_PLANNED_COUNT')
    candidates = read(NEW)
    reviews = read(REVIEW)['reviews']
    require(isinstance(candidates, list) and 0 < len(candidates) <= planned, 'CANDIDATE_LIMIT')
    by_id = {row['candidate_id']: row for row in reviews}
    require(len(by_id) == len(reviews) == len(candidates), 'EXACT_REVIEW_COVERAGE_REQUIRED')
    require(set(by_id) == {c['candidate_id'] for c in candidates}, 'CANDIDATE_REVIEW_SET_DRIFT')
    admitted, ledger = [], []
    for candidate in candidates:
        cid, rid = candidate['candidate_id'], candidate['root_id']
        require(rid in roots, 'UNSELECTED_ROOT:' + cid)
        review = by_id[cid]
        require(review['candidate_sha256'] == fingerprint(candidate), 'CANDIDATE_REVIEW_HASH_DRIFT:' + cid)
        require(candidate.get('author_id') and review.get('author_id') == candidate['author_id'] != review.get('reviewer_id'), 'INDEPENDENT_REVIEW_REQUIRED:' + cid)
        require(review['decision'] in ('PASS', 'HOLD', 'FAIL'), 'UNKNOWN_REVIEW_DECISION:' + cid)
        selected_binding = {k: roots[rid][k] for k in ('path', 'sha256', 'bytes')}
        require(candidate['source_binding'] == selected_binding, 'SOURCE_SELECTION_DRIFT:' + cid)
        check(selected_binding)
        if review['decision'] == 'PASS':
            gate._validate(candidate)
            result = production_gate.validate_review(candidate, review, SELECTION)
            require(result['admitted'], 'CONTENT_REVIEW_GATE_REJECTED:' + cid + ':' + json.dumps(result))
            count = len(candidate['items' if candidate['question_type'] == 'ordering' else 'options'])
            require(4 <= count <= 6 if candidate['question_type'] == 'ordering' else count == 6, 'ITEM_COUNT_LIMIT:' + cid)
            diagnostic.operation_text(candidate)
            admitted.append(candidate)
        ledger.append({'candidate_id': cid, 'root_id': rid, 'decision': review['decision'],
                       'candidate_sha256': fingerprint(candidate), 'author_id': candidate['author_id'],
                       'reviewer_id': review['reviewer_id']})
    require(admitted, 'NO_CONTENT_ADMITTED_QUESTIONS')
    from transformers import AutoTokenizer
    import xgrammar as xgr
    tokenizer = AutoTokenizer.from_pretrained(endpoint['tokenizer_directory'], local_files_only=True)
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=tokenizer.vocab_size), max_threads=4)
    staged, sources = [], {}
    for candidate in admitted:
        source = read(candidate['source_binding']['path'])
        require(source['root_id'] == candidate['root_id'], 'SOURCE_ROOT_DRIFT')
        sources[candidate['source_binding']['path']] = candidate['source_binding']
        refs = [turn['turn_ref'] for turn in source['turns']]
        require(refs and len(refs) == len(set(refs)), 'INVALID_SOURCE_REFS')
        for variant, public, private in displays(candidate):
            field = 'options' if candidate['question_type'] == 'multi_select' else 'items'
            for condition in CONDITIONS:
                row = {k: v for k, v in public.items() if k != field}
                row.update(display=public[field], labels=list(public[field]), variant_id=variant,
                           cohort='new', condition=condition, sampling_seed=sampling_seed(candidate['candidate_id']),
                           request_id=f"{candidate['candidate_id']}__{variant}__{condition}__B1_priority_v1")
                prompt = diagnostic.make_prompt(tokenizer, row, row['display'], source) if condition == 'FULL' else base.question_only_prompt(tokenizer, row)
                tokens = tokenizer.encode(prompt, add_special_tokens=False)
                require(len(tokens) + MAX_TOKENS <= MODEL_LEN, 'CONTEXT_OVERFLOW_NO_TRUNCATION:' + row['request_id'])
                schema = diagnostic.repair.schema(row['labels'], refs, row['question_type']) if condition == 'FULL' else base.question_only_schema(row['labels'], row['question_type'])
                compiler.compile_json_schema(json.dumps(schema), any_whitespace=True)
                scoring = {'request_id': row['request_id'], 'candidate_id': candidate['candidate_id'],
                           'cohort': 'new', 'parent_candidate_id': None, 'strictly_paired': False,
                           'variant_id': variant, 'presentation': private}
                staged.append((row, prompt, tokens, schema, scoring))
    require(len(staged) == len(admitted) * 4 <= MAX_REQUESTS, 'REQUEST_LIMIT_OR_FACTOR_DRIFT')
    OUT.mkdir()
    bindings = [bind(p) for p in code_paths()] + list(sources.values())
    rows, token_ids, private_rows = [], {}, []
    for row, prompt, tokens, schema, scoring in staged:
        rid = row['request_id']
        pp, sp = OUT / 'prompts' / (rid + '.txt'), OUT / 'schemas' / (rid + '.json')
        pp.parent.mkdir(exist_ok=True)
        with pp.open('x', encoding='utf-8') as stream:
            stream.write(prompt); stream.flush(); os.fsync(stream.fileno())
        save(sp, schema)
        row.update(prompt_path=str(pp), schema_path=str(sp), prompt_tokens=len(tokens))
        require(set(row) == PUBLIC_KEYS, 'PUBLIC_FIELD_ALLOWLIST_DRIFT')
        rows.append(row); token_ids[rid] = tokens; private_rows.append(scoring)
        bindings.extend((bind(pp), bind(sp)))
    save(OUT / 'PUBLIC_REQUESTS.json', {'requests': rows})
    save(OUT / 'PROMPT_TOKEN_IDS.json', token_ids)
    bindings.extend(bind(OUT / name) for name in ('PUBLIC_REQUESTS.json', 'PROMPT_TOKEN_IDS.json'))
    private_path = OUT / 'PRIVATE_SCORING.json'
    save(private_path, {'requests': private_rows, 'bindings': [bind(NEW), bind(REVIEW), bind(SELECTION)],
                       'public_requests_binding': bind(OUT / 'PUBLIC_REQUESTS.json'),
                       'prompt_token_ids_binding': bind(OUT / 'PROMPT_TOKEN_IDS.json'),
                       'new_admission': ledger, 'planned_questions': planned,
                       'new_mc_coverage': gate.coverage_report(admitted)})
    private_path.chmod(0o600)
    save(OUT / 'PREPARATION.json', {'requests': len(rows), 'model_role': 'B1', 'bindings': bindings,
        'private_scoring_binding': bind(private_path), 'config_preflight_receipt': config_receipt,
        'runtime_versions': versions(), 'max_output_tokens': MAX_TOKENS, 'max_model_len': MODEL_LEN,
        'temperature': .8, 'top_p': .9, 'display_seed': DISPLAY_SEED, 'variants': list(VARIANTS),
        'conditions': list(CONDITIONS), 'fixed_order': [r['request_id'] for r in rows],
        'automatic_retry': False, 'source_truncation': False, 'native_hidden_thinking': False,
        'independent_requests_no_shared_history_or_memory': True, 'generation_reads_private_scoring': False,
        'effective_rope_theta_required': EXPECTED_THETA, 'historical_scores_reusable': False,
        'namespace': 'train_only_priority_diversity_feedback_diagnostic'})
    print(json.dumps({'prepared_requests': len(rows), 'admitted_questions': len(admitted),
                      'max_prompt_tokens': max(r['prompt_tokens'] for r in rows)}))


def public_preparation():
    prep = read(OUT / 'PREPARATION.json')
    forbidden = {NEW.resolve(), REVIEW.resolve(), SELECTION.resolve(), (OUT / 'PRIVATE_SCORING.json').resolve()}
    for bound in prep['bindings']:
        require(Path(bound['path']).resolve() not in forbidden, 'PRIVATE_INPUT_IN_GENERATION_BINDINGS')
        check(bound)
    rows, tokens = read(OUT / 'PUBLIC_REQUESTS.json')['requests'], read(OUT / 'PROMPT_TOKEN_IDS.json')
    ids = [row['request_id'] for row in rows]
    require(0 < len(ids) == prep['requests'] == len(set(ids)) <= MAX_REQUESTS, 'REQUEST_SET_OR_LIMIT_DRIFT')
    require(ids == prep['fixed_order'] and set(tokens) == set(ids), 'FROZEN_ORDER_OR_TOKEN_SET_DRIFT')
    required = [OUT / 'PUBLIC_REQUESTS.json', OUT / 'PROMPT_TOKEN_IDS.json'] + code_paths()
    for row in rows:
        require(set(row) == PUBLIC_KEYS and row['condition'] in CONDITIONS and row['variant_id'] in VARIANTS, 'NONPUBLIC_REQUEST_FIELDS')
        require(row['cohort'] == 'new' and row['labels'] == list(row['display']), 'PUBLIC_DISPLAY_DRIFT')
        require(len(tokens[row['request_id']]) == row['prompt_tokens'], 'TOKEN_COUNT_DRIFT')
        require(row['prompt_tokens'] + MAX_TOKENS <= MODEL_LEN, 'CONTEXT_OVERFLOW')
        required.extend(Path(row[k]) for k in ('prompt_path', 'schema_path'))
    require(all(bind(p) in prep['bindings'] for p in required), 'UNBOUND_PUBLIC_INPUT')
    operation_instructions()
    return prep, rows, tokens


def checked_runtime_config(hf_config):
    nested = getattr(hf_config, 'rope_parameters', None)
    require(isinstance(nested, dict) and nested.get('rope_type') == 'default' and nested.get('rope_theta') == EXPECTED_THETA, 'LOADED_NESTED_ROPE_THETA_DRIFT')
    top = getattr(hf_config, 'rope_theta', None)
    require(top in (None, EXPECTED_THETA), 'LOADED_TOP_ROPE_THETA_DRIFT')
    return {'rope_parameters': nested, 'rope_theta': top, 'model_type': getattr(hf_config, 'model_type', None),
            'max_position_embeddings': getattr(hf_config, 'max_position_embeddings', None)}


def checked_prompt_tokens(tokenizer, rows, token_ids):
    for row in rows:
        actual = tokenizer.encode(Path(row['prompt_path']).read_text(), add_special_tokens=False)
        require(actual == token_ids[row['request_id']], 'PROMPT_TOKEN_CONTENT_DRIFT:' + row['request_id'])


def checked_raw_manifest(frozen, rows, root):
    ids = [r['request_id'] for r in rows]
    paths = {str((root / 'raw' / (rid + '.json')).resolve()) for rid in ids}
    require(frozen['requests'] == len(ids) == len(frozen['bindings']), 'RAW_FREEZE_COUNT_DRIFT')
    bound_paths = [b['path'] for b in frozen['bindings']]
    require(len(set(bound_paths)) == len(paths) and set(bound_paths) == paths, 'RAW_FREEZE_SET_DRIFT')
    require({str(p.resolve()) for p in (root / 'raw').glob('*.json')} == paths, 'RAW_DIRECTORY_SET_DRIFT')
    require(frozen['request_ids'] == ids, 'RAW_ORDER_DRIFT')
    for b in frozen['bindings'] + frozen['validation_bindings'] + frozen['submission_bindings']:
        check(b)
    require(frozen['returned_requests'] == sum(read(p)['status'] == 'RETURNED' for p in paths), 'RETURNED_COUNT_DRIFT')
    require(set(frozen['submitted_request_ids']) <= set(ids), 'UNKNOWN_SUBMITTED_REQUEST')
    require(frozen['submitted_requests'] == len(frozen['submitted_request_ids']) == len(set(frozen['submitted_request_ids'])), 'SUBMITTED_COUNT_DRIFT')
    submitted = set(frozen['submitted_request_ids'])
    returned = set()
    receipt = frozen['model_load_receipt_binding']
    if receipt is not None:
        check(receipt)
    for path in paths:
        raw = read(path)
        require(raw['request_id'] in ids and path == str((root / 'raw' / (raw['request_id'] + '.json')).resolve()), 'RAW_REQUEST_ID_DRIFT')
        require(raw['status'] in ('RETURNED', 'BACKEND_ERROR', 'NOT_SUBMITTED'), 'UNKNOWN_RAW_TERMINAL_STATUS')
        rid = raw['request_id']
        require((raw['status'] == 'NOT_SUBMITTED') == (rid not in submitted), 'RAW_SUBMISSION_STATUS_DRIFT')
        if raw['status'] == 'RETURNED':
            returned.add(rid)
            require(receipt is not None and raw['model_load_receipt_binding'] == receipt, 'RAW_MODEL_RECEIPT_DRIFT')
    for dirname, names, key in (('submitted', submitted, 'submission_bindings'), ('raw_validation', returned, 'validation_bindings')):
        expected = {str((root / dirname / (rid + '.json')).resolve()) for rid in names}
        actual = {str(p.resolve()) for p in (root / dirname).glob('*.json')}
        bound = [b['path'] for b in frozen[key]]
        require(actual == expected and len(bound) == len(expected) and set(bound) == expected, 'RAW_' + dirname.upper() + '_SET_DRIFT')
        for path in expected:
            record = read(path)
            rid = record['request_id']
            require(path == str((root / dirname / (rid + '.json')).resolve()), 'RAW_' + dirname.upper() + '_ID_DRIFT')
            if dirname == 'raw_validation':
                require(record['raw_binding'] == bind(root / 'raw' / (rid + '.json')), 'RAW_VALIDATION_BINDING_DRIFT')
            else:
                require(receipt is not None and record['model_load_receipt_binding'] == receipt, 'SUBMISSION_MODEL_RECEIPT_DRIFT')
    return True


def run(execute_gpu=False):
    require(execute_gpu, 'EXPLICIT_GPU_FLAG_REQUIRED')
    require(not (OUT / 'STARTED.json').exists() and not (OUT / 'RAW_RESPONSE_FREEZE.json').exists(), 'NO_RERUN_OR_OVERWRITE')
    prep, rows, token_ids = public_preparation()
    endpoint, config_receipt = endpoint_receipt()
    require(config_receipt == prep['config_preflight_receipt'], 'CONFIG_PREFLIGHT_DRIFT')
    require(versions() == prep['runtime_versions'], 'RUNTIME_PACKAGE_DRIFT')
    for bound in endpoint['model_bindings'] + endpoint['tokenizer_bindings']:
        check(bound)
    for key, value in {'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'VLLM_USE_FLASHINFER_SAMPLER': '0',
                       'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false'}.items():
        os.environ[key] = value
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(endpoint['tokenizer_directory'], local_files_only=True)
    checked_prompt_tokens(tokenizer, rows, token_ids)
    from vllm import LLM, SamplingParams, sampling_params
    lock = open('/tmp/gxy-selfplay-gpu.lock', 'a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    ids = [r['request_id'] for r in rows]
    save(OUT / 'STARTED.json', {'time_unix': time.time(), 'preparation': bind(OUT / 'PREPARATION.json'),
        'private_scoring_binding': prep['private_scoring_binding'], 'requests': len(rows), 'request_ids': ids,
        'automatic_retry': False, 'gold_read': False, 'model_directory': str(MODEL)})
    engine, failure, submitted, request_failures = None, None, [], []
    started = time.perf_counter()
    try:
        overrides = {'rope_theta': EXPECTED_THETA, 'rope_parameters': {'rope_type': 'default', 'rope_theta': EXPECTED_THETA}}
        engine = LLM(model=str(MODEL), tokenizer=str(TOKENIZER), tensor_parallel_size=1,
                     max_model_len=MODEL_LEN, gpu_memory_utilization=.8, dtype='bfloat16',
                     trust_remote_code=True, enable_prefix_caching=False, hf_overrides=overrides)
        model_config = engine.llm_engine.model_config
        require(Path(model_config.model).resolve() == MODEL.resolve(), 'ACTUAL_LOADED_MODEL_DRIFT')
        actual = {'hf_config': checked_runtime_config(model_config.hf_config),
                  'hf_text_config': checked_runtime_config(model_config.hf_text_config)}
        receipt = {'status': 'PASS_CONFIG_THETA_GATE', 'model_directory': str(MODEL),
                   'tokenizer_directory': str(TOKENIZER), 'runtime_versions': versions(),
                   'checked_config': config_receipt, 'hf_overrides': overrides, 'actual_engine_config': actual,
                   'effective_rope_theta': EXPECTED_THETA, 'base_model_modified': False,
                   'model_config_binding': bind(MODEL / 'config.json'), 'runner_binding': bind(Path(__file__))}
        save(OUT / 'MODEL_LOAD_RECEIPT.json', receipt)
        for row in rows:
            rid, raw_path = row['request_id'], OUT / 'raw' / (row['request_id'] + '.json')
            try:
                params = SamplingParams(temperature=.8, top_p=.9, max_tokens=MAX_TOKENS, seed=row['sampling_seed'],
                    structured_outputs=sampling_params.StructuredOutputsParams(json=read(row['schema_path'])))
                prompt = Path(row['prompt_path']).read_text()
                save(OUT / 'submitted' / (rid + '.json'), {'request_id': rid, 'seed': row['sampling_seed'], 'time_unix': time.time(),
                     'model_load_receipt_binding': bind(OUT / 'MODEL_LOAD_RECEIPT.json')})
                submitted.append(rid)
                results = engine.generate([prompt], [params], use_tqdm=False)
                raw = base.preserve_response(row, results, endpoint)
                raw['model_load_receipt_binding'] = bind(OUT / 'MODEL_LOAD_RECEIPT.json')
                save(raw_path, raw)
                errors = capture.response_errors(raw, prompt, token_ids[rid])
                save(OUT / 'raw_validation' / (rid + '.json'), {'request_id': rid, 'errors': errors, 'raw_binding': bind(raw_path)})
                if errors:
                    request_failures.append({'request_id': rid, 'errors': errors})
                print('RETURNED ' + rid, flush=True)
            except Exception:
                error = traceback.format_exc()
                request_failures.append({'request_id': rid, 'error': error})
                if not raw_path.exists():
                    save(raw_path, {'request_id': rid, 'status': 'BACKEND_ERROR' if rid in submitted else 'NOT_SUBMITTED', 'raw_output': None, 'failure': error})
    except BaseException:
        failure = traceback.format_exc()
    finally:
        if engine is not None:
            try:
                core = getattr(engine.llm_engine, 'engine_core', None)
                if callable(getattr(core, 'shutdown', None)):
                    core.shutdown()
                del engine
                import gc
                gc.collect()
            except BaseException:
                failure = (failure or '') + traceback.format_exc()
        for row in rows:
            rid = row['request_id']
            path = OUT / 'raw' / (rid + '.json')
            if not path.exists():
                save(path, {'request_id': rid, 'status': 'BACKEND_ERROR' if rid in submitted else 'NOT_SUBMITTED', 'raw_output': None, 'failure': failure})
            validation_path = OUT / 'raw_validation' / (rid + '.json')
            if read(path)['status'] == 'RETURNED' and not validation_path.exists():
                save(validation_path, {'request_id': rid, 'errors': ['CAPTURE_VALIDATION_FAILED_SEE_REQUEST_FAILURES'], 'raw_binding': bind(path)})
        raw_paths = sorted((OUT / 'raw').glob('*.json'))
        frozen = {'requests': len(rows), 'request_ids': ids, 'started_binding': bind(OUT / 'STARTED.json'),
            'wall_seconds': time.perf_counter() - started, 'submitted_requests': len(submitted), 'submitted_request_ids': submitted,
            'returned_requests': sum(read(p)['status'] == 'RETURNED' for p in raw_paths),
            'failure': failure, 'request_failures': request_failures, 'automatic_retry': False,
            'model_load_receipt_binding': bind(OUT / 'MODEL_LOAD_RECEIPT.json') if (OUT / 'MODEL_LOAD_RECEIPT.json').exists() else None,
            'bindings': [bind(p) for p in raw_paths],
            'submission_bindings': [bind(p) for p in sorted((OUT / 'submitted').glob('*.json'))],
            'validation_bindings': [bind(p) for p in sorted((OUT / 'raw_validation').glob('*.json'))]}
        checked_raw_manifest(frozen, rows, OUT)
        save(OUT / 'RAW_RESPONSE_FREEZE.json', frozen)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN); lock.close()
    if failure or request_failures:
        raise RuntimeError('Frozen failures; no automatic retry. Inspect RAW_RESPONSE_FREEZE.json')


original_sealed_inputs = base.sealed_inputs


def sealed_inputs():
    prep, rows, tokens = public_preparation()
    frozen = read(OUT / 'RAW_RESPONSE_FREEZE.json')
    checked_raw_manifest(frozen, rows, OUT)
    require(frozen['model_load_receipt_binding'] is not None, 'NO_CORRECT_MODEL_LOAD_RECEIPT')
    check(frozen['model_load_receipt_binding'])
    receipt = read(frozen['model_load_receipt_binding']['path'])
    require(receipt['status'] == 'PASS_CONFIG_THETA_GATE' and receipt['effective_rope_theta'] == EXPECTED_THETA, 'BAD_MODEL_LOAD_RECEIPT')
    for name in ('hf_config', 'hf_text_config'):
        from types import SimpleNamespace
        checked_runtime_config(SimpleNamespace(**receipt['actual_engine_config'][name]))
    return original_sealed_inputs()


original_score_record = base.score_record


def score_record(row, presentation, answer, valid):
    result = original_score_record(row, presentation, answer, valid)
    result['variant_id'] = row['variant_id']
    return result


def summarize(scores, private):
    cells = [{'condition': c, 'variant_id': v, **base.aggregate([s for s in scores if s['condition'] == c and s['variant_id'] == v])} for c in CONDITIONS for v in VARIANTS]
    conditions = [{'condition': c, **base.aggregate([s for s in scores if s['condition'] == c])} for c in CONDITIONS]
    typed = []
    for c in CONDITIONS:
        for group in ('MC2_5_PRIMARY', 'ORDERING', 'MC6_CONTROL'):
            chosen = [s for s in scores if s['condition'] == c and ((group == 'ORDERING' and s['question_type'] == 'ordering') or (group == 'MC2_5_PRIMARY' and s['gold_count'] in (2, 3, 4, 5)) or (group == 'MC6_CONTROL' and s['gold_count'] == 6))]
            typed.append({'condition': c, 'group': group, **base.aggregate(chosen)})
    pairs = []
    for cid in sorted({s['candidate_id'] for s in scores}):
        for condition in CONDITIONS:
            pair = sorted([s for s in scores if s['candidate_id'] == cid and s['condition'] == condition], key=lambda s: s['variant_id'])
            require(len(pair) == 2, 'MISSING_DISPLAY_PAIR')
            a, b = pair
            same = a['valid'] and b['valid'] and (set(a['prediction']) == set(b['prediction']) if a['question_type'] == 'multi_select' else a['prediction'] == b['prediction'])
            pairs.append({'candidate_id': cid, 'condition': condition, 'question_type': a['question_type'],
                          'both_valid': a['valid'] and b['valid'], 'canonical_prediction_same': same,
                          'both_exact': a['exact'] and b['exact'], 'accuracy_changed': a['exact'] != b['exact']})
    groups = []
    for condition in CONDITIONS:
        for operation in sorted({s['task_operation'] for s in scores}):
            groups.append({'condition': condition, 'task_operation': operation, **base.aggregate([s for s in scores if s['condition'] == condition and s['task_operation'] == operation])})
    root_cells = [{'root_id': rid, 'condition': c, **base.aggregate([s for s in scores if s['root_id'] == rid and s['condition'] == c])} for rid in sorted({s['root_id'] for s in scores}) for c in CONDITIONS]
    return {'requests': len(scores), 'valid': sum(s['valid'] for s in scores), 'correct': sum(s['exact'] for s in scores),
        'planned_questions': private['planned_questions'], 'admitted_questions': len(scores) // 4,
        'cells': cells, 'conditions': conditions, 'primary_control_groups': typed,
        'operation_cells': groups, 'root_cells': root_cells, 'variant_pairs': pairs,
        'new_mc_gold_buckets': [{'condition': c, 'gold_count': n, **base.aggregate([s for s in scores if s['condition'] == c and s['gold_count'] == n])} for c in CONDITIONS for n in (2, 3, 4, 5, 6)],
        'new_admission': private['new_admission'], 'all_frozen_requests_in_exact_denominator': True,
        'invalid_mc_prediction_for_descriptive_fp_fn': 'empty_set', 'semantic_reason_review': 'PENDING_INDEPENDENT_REVIEW',
        'feedback_eligible': False, 'formal_weakness_admission': False, 'training_admission': False,
        'historical_scores_reused': False, 'new_dialogue_turns': 0,
        'inference_limits': 'Train-only four-root maximum, two stochastic displays per question. Related displays are not independent questions; QUESTION_ONLY is shortcut diagnosis. Visible reasons are not faithful internal CoT; no causal feedback claim.'}


base.OUT = OUT
base.public_preparation = public_preparation
base.sealed_inputs = sealed_inputs
base.score_record = score_record
base.summarize = summarize


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'analyze'))
    parser.add_argument('--execute-gpu', action='store_true')
    args = parser.parse_args()
    {'prepare': prepare, 'run': lambda: run(args.execute_gpu), 'analyze': base.analyze}[args.action]()
