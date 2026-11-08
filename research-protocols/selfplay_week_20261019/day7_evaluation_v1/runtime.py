"""Frozen Day7 TEST-only evaluation of S0/A1/B1; A2/B2 were not produced.

verify, bind, and score do not instantiate models. Only live imports a vLLM
engine, and requires a PASS independent review bound to this code, the common
request freeze, and all three execution manifests. Requests are never retried.
Gold values are parsed only by score, after a complete raw-response freeze.
"""
from __future__ import annotations

import argparse
import collections
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
REPO = WEEK.parents[1]
VENV = REPO / '.venv-local4b'
TOKENIZER = Path('/root/autodl-tmp/Qwen3-4B')
ADAPTER = WEEK / 'day2/solver/adapter.py'
DEV = WEEK / 'day5_training_v1/evaluation'
MODEL_ROLES = ('S0', 'A1', 'B1')
QUESTION_COUNT = 89
REQUEST_COUNT = 1446
COHORT_COUNTS = sorted([(75, 1230), (2, 36), (12, 180)])
for key, value in {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                   'VLLM_USE_FLASHINFER_SAMPLER': '0',
                   'TOKENIZERS_PARALLELISM': 'false'}.items():
    os.environ.setdefault(key, value)
sys.dont_write_bytecode = True


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def textsha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def canon(value):
    return textsha(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')))


def bindfile(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def verify_bindings(bindings):
    for row in bindings:
        path = Path(row['path'])
        require(path.is_file() and path.stat().st_size == row['bytes'] and
                sha(path) == row['sha256'], 'BOUND_FILE_CHANGED:' + str(path))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def adapter():
    key = 'day7_preserved_day2_adapter'
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ADAPTER)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def versions():
    return {name: importlib.metadata.version(name)
            for name in ('torch', 'transformers', 'tokenizers', 'vllm')}


def verify():
    """Verify bound bytes and public request structure without parsing Gold."""
    freeze = read(HERE / 'TEST_REQUEST_FREEZE.json')
    require(freeze['question_count'] == QUESTION_COUNT and
            freeze['requests_per_model'] == REQUEST_COUNT and
            tuple(freeze['model_roles']) == MODEL_ROLES, 'TEST_FREEZE_DENOMINATOR_DRIFT')
    bindings = freeze['input_bindings'] + freeze['artifact_bindings']
    code_binding = freeze.get('runtime_binding', freeze.get('wrapper'))
    if code_binding is not None:
        bindings += [code_binding]
    # The common freeze binds the preparation script. Execution manifests bind
    # this runtime separately, avoiding a circular common/manifest dependency.
    verify_bindings(bindings)
    require(freeze['runtime_versions'] == versions(), 'RUNTIME_VERSIONS_CHANGED')
    rows = lines(HERE / 'TEST_REQUESTS.jsonl')
    public = lines(HERE / 'TEST_PUBLIC_QUESTIONS.jsonl')
    require(len(rows) == len({r['request_id'] for r in rows}) == REQUEST_COUNT and
            len(public) == len({q['candidate_id'] for q in public}) == QUESTION_COUNT,
            'TEST_REQUEST_OR_QUESTION_DENOMINATOR_DRIFT')
    questions = {q['candidate_id']: q for q in public}
    choices = read(HERE / 'FINITE_CHOICES.json')
    maps = read(HERE / 'TEST_DISPLAY_MAPS.json')
    pindex = read(HERE / 'PROMPT_INDEX.json')
    if isinstance(pindex, list):
        require(len(pindex) == len({p['prompt_id'] for p in pindex}), 'DUPLICATE_PROMPT_ID')
        pindex = {p['prompt_id']: p for p in pindex}
    require(set(maps) == set(pindex) == {r['prompt_id'] for r in rows}, 'PROMPT_MAPPING_BOUNDARY_DRIFT')
    cohort_questions = collections.Counter(q['cohort_id'] for q in public)
    cohort_requests = collections.Counter(r['cohort_id'] for r in rows)
    require(set(cohort_questions) == set(cohort_requests) and
            sorted((cohort_questions[c], cohort_requests[c]) for c in cohort_questions) == COHORT_COUNTS,
            'COHORT_DENOMINATOR_DRIFT')
    for q in public:
        require(q.get('split') == 'test', 'NON_TEST_PUBLIC_QUESTION')
        adapter().public_question({k: v for k, v in q.items() if k in adapter().PUBLIC_KEYS})
    for row in rows:
        q = questions[row['candidate_id']]
        require(row['split'] == 'test' and row['branch'] == 'direct' and
                row['cohort_id'] == q['cohort_id'] and row['root_id'] == q['root_id'] and
                row['question_type'] == q['question_type'], 'NON_TEST_OR_REQUEST_IDENTITY_DRIFT')
        require(Path(row['request_id']).name == row['request_id'], 'UNSAFE_REQUEST_ID')
        sampling = row['sampling']
        require(set(sampling) == {'temperature', 'top_p', 'max_tokens', 'seed', 'stop',
                                 'include_stop_str_in_output'} and
                sampling['temperature'] == .8 and sampling['top_p'] == .9 and
                sampling['max_tokens'] == 32 and isinstance(sampling['seed'], int) and
                sampling['stop'] == ['</box>'] and sampling['include_stop_str_in_output'] is True,
                'NATIVE_SAMPLING_DRIFT')
        require(row['prompt_tokens'] + 32 <= 16384, 'SOURCE_TRUNCATION_REQUIRED')
        prompt = pindex[row['prompt_id']]
        require(row['prompt_sha256'] == prompt['prompt_sha256'] and
                row['prompt_tokens'] == prompt['prompt_tokens'] and
                Path(row['prompt_file']).resolve() == Path(prompt['prompt_file']).resolve() and
                sha(row['prompt_file']) == row['prompt_sha256'], 'PROMPT_BYTES_OR_LENGTH_DRIFT')
        field = 'options' if row['question_type'] == 'multi_select' else 'items'
        n = len(q[field])
        mapping = maps[row['prompt_id']]
        require(mapping['variant_id'] == row['variant_id'] and
                mapping['candidate_id'] == row['candidate_id'] and len(mapping[field]) == n and
                set(mapping['display_to_content']) == set(mapping[field]) and
                len(set(mapping['display_to_content'].values())) == n, 'DISPLAY_MAPPING_DRIFT')
        key = row['constraint_key']
        require(key == row['question_type'] or key == row['question_type'] + ':' + str(n),
                'CHOICE_KEY_ARITY_DRIFT')
        require(key in choices and len(choices[key]) == len(set(choices[key])) ==
                (2 ** n - 1 if field == 'options' else math.factorial(n)), 'FINITE_CHOICE_ARITY_DRIFT')
    return freeze


def bind_model(role, modeldir, endpoint_proof):
    freeze = verify()
    require(Path(sys.prefix).resolve() == VENV.resolve(), 'USE_PROVEN_LOCAL4B_VENV')
    modeldir, endpoint_proof = modeldir.resolve(), endpoint_proof.resolve()
    devpath = DEV / 'models' / role / 'EXECUTION_MANIFEST.json'
    dev = read(devpath)
    require(dev['model_role'] == role and Path(dev['model_directory']).resolve() == modeldir and
            Path(dev['endpoint_proof']['path']).resolve() == endpoint_proof,
            'FINAL_DEV_ENDPOINT_IDENTITY_CHANGED')
    verify_bindings([dev['endpoint_proof']] + dev['endpoint_evidence_bindings'] +
                    dev['model_bindings'] + dev['tokenizer_bindings'] + dev['code_bindings'])
    require(dev['runtime_versions'] == versions(), 'DEV_TO_TEST_RUNTIME_DRIFT')
    proof = read(endpoint_proof)
    require(proof['model_role'] == role and Path(proof['model_directory']).resolve() == modeldir,
            'ENDPOINT_PROOF_ROLE_OR_PATH_MISMATCH')
    require(proof.get('evidence_bindings'), 'ENDPOINT_EVIDENCE_REQUIRED')
    verify_bindings(proof['evidence_bindings'])
    if role in ('A1', 'B1'):
        require(proof.get('fixed_final_optimizer_step') == proof.get('completed_optimizer_steps') == 56 and
                proof.get('training_completed') is True and proof.get('independent_initialization') == 'S0',
                'FIXED_FINAL_STEP56_PROOF_REQUIRED')
        require(proof.get('merge_base') == str(TOKENIZER) and
                proof.get('adapter_representation') == 'FULL_UPDATED_S0_CUSTOM_LORA',
                'FULL_ADAPTER_MUST_MERGE_INTO_ORIGINAL_BASE')
    else:
        original = read(WEEK / 'day1/MODEL_BINDINGS.json')['models']['S0_M_clean_merged']
        require(modeldir == Path(original['directory']).resolve(), 'S0_PATH_CHANGED')
        verify_bindings(original['files'])
    require((modeldir / 'config.json').is_file() and list(modeldir.glob('*.safetensors')),
            'STANDARD_MERGED_HF_MODEL_REQUIRED')
    fresh_files = [bindfile(p) for p in sorted(modeldir.iterdir()) if p.is_file()]
    require({r['path']: r for r in fresh_files} == {r['path']: r for r in dev['model_bindings']},
            'FRESH_MODEL_HASHES_DIFFER_FROM_DEV_ENDPOINT')
    tokenizers = [bindfile(TOKENIZER / name) for name in
                  ('tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json')]
    require(tokenizers == dev['tokenizer_bindings'], 'FRESH_TOKENIZER_HASHES_DIFFER_FROM_DEV')
    a = adapter()
    _, solver = a.native()
    from scripts.current_method.restart_n5_spell import vllm_backends, structured_qa_box_protocol
    decoder = solver.generation_config('multi_select')
    require(decoder == dev['decoder_contract'], 'NATIVE_DECODER_DRIFT')
    frozen_choices = read(HERE / 'FINITE_CHOICES.json')
    for key, values in frozen_choices.items():
        parts = key.split(':')
        typ, n = parts[0], int(parts[1]) if len(parts) == 2 else 6
        labels = dict.fromkeys(a.labels(typ, n))
        native = a.choices(typ, labels)
        require(len(native) == len(values) and set(native) == set(values), 'NATIVE_CHOICE_SPACE_DRIFT:' + key)
    code = {Path(__file__), ADAPTER, a.NATIVE / 'scoring.py', Path(vllm_backends.__file__),
            Path(structured_qa_box_protocol.__file__), Path(solver.__file__)}
    code.update(Path(m.__file__) for m in list(sys.modules.values())
                if getattr(m, '__file__', None) and Path(m.__file__).suffix == '.py' and
                Path(m.__file__).is_relative_to(a.WORKTREE))
    out = HERE / 'models' / role / 'EXECUTION_MANIFEST.json'
    save(out, {'schema_version': 'day7_TEST_model_execution_manifest_v1',
               'status': 'BOUND_FINAL_ENDPOINT_READY_FOR_REVIEW', 'model_role': role,
               'model_directory': str(modeldir), 'endpoint_proof': bindfile(endpoint_proof),
               'endpoint_evidence_bindings': proof['evidence_bindings'],
               'historical_DEV_identity_manifest': bindfile(devpath),
               'fresh_hashes_match_existing_DEV_model': True,
               'request_freeze': bindfile(HERE / 'TEST_REQUEST_FREEZE.json'),
               'model_bindings': fresh_files, 'tokenizer_bindings': tokenizers,
               'tokenizer_directory': str(TOKENIZER), 'runtime_versions': versions(),
               'python_prefix': sys.prefix, 'decoder_contract': decoder,
               'code_bindings': [bindfile(p) for p in sorted(code)],
               'request_count': freeze['requests_per_model'], 'test_requests': REQUEST_COUNT,
               'batch_size': 8, 'max_model_len': 16384, 'gpu_memory_utilization': .8,
               'prefix_caching': False, 'automatic_retry': False,
               'weights_loaded_during_binding': False, 'A2_B2': 'NOT_PRODUCED',
               'created_at_utc': now()})
    print(json.dumps({'status': 'BOUND_FINAL_ENDPOINT_READY_FOR_REVIEW', 'model_role': role,
                      'requests': REQUEST_COUNT, 'manifest': str(out)}))


def execution_verify(role, audit):
    verify()
    review = read(audit)
    require(review.get('status') == 'PASS', 'EXECUTION_REVIEW_MUST_PASS')
    audited = review.get('audited_files', [])
    require(audited, 'INDEPENDENT_REVIEW_BINDINGS_REQUIRED')
    verify_bindings(audited)
    required = [Path(__file__), HERE / 'TEST_REQUEST_FREEZE.json']
    required += [HERE / 'models' / r / 'EXECUTION_MANIFEST.json' for r in MODEL_ROLES]
    for path in required:
        require(any(Path(x['path']).resolve() == path.resolve() and x['sha256'] == sha(path)
                    for x in audited), 'REVIEW_NOT_BOUND_TO_REQUIRED_FILE:' + str(path))
    manifests = {}
    for current_role in MODEL_ROLES:
        manifest = read(HERE / 'models' / current_role / 'EXECUTION_MANIFEST.json')
        require(manifest['model_role'] == current_role and manifest['request_count'] == REQUEST_COUNT and
                manifest['test_requests'] == REQUEST_COUNT and manifest['automatic_retry'] is False,
                'MODEL_EXECUTION_SCOPE_DRIFT')
        # All endpoints must be fixed before TEST starts. Each selected endpoint's
        # large model bytes are freshly checked again before it is loaded.
        verify_bindings([manifest['request_freeze'], manifest['historical_DEV_identity_manifest'],
                         manifest['endpoint_proof']] + manifest['endpoint_evidence_bindings'] +
                        manifest['tokenizer_bindings'] + manifest['code_bindings'])
        require(Path(sys.prefix).resolve() == VENV.resolve() and
                manifest['runtime_versions'] == versions(), 'RUNTIME_CHANGED')
        manifests[current_role] = manifest
    verify_bindings(manifests[role]['model_bindings'])
    return manifests[role]


def live(role, audit):
    manifest = execution_verify(role, audit)
    run = HERE / 'models' / role
    require(not (run / 'STARTED.json').exists(), 'ALREADY_STARTED_NO_AUTOMATIC_RETRY')
    require(not (run / 'responses').exists(), 'PREEXISTING_RESPONSE_DIRECTORY')
    for other in MODEL_ROLES:
        if other == role:
            continue
        other_run = HERE / 'models' / other
        require(not (other_run / 'STARTED.json').exists() or
                (other_run / 'RAW_RESPONSE_FREEZE.json').is_file(), 'OTHER_ROLE_EXECUTION_NOT_TERMINAL')
    rows = lines(HERE / 'TEST_REQUESTS.jsonl')
    save(run / 'STARTED.json', {'started_at_utc': now(), 'model_role': role,
                               'execution_manifest': bindfile(run / 'EXECUTION_MANIFEST.json'),
                               'review': bindfile(audit), 'private_gold_values_parsed': False})
    backend = None
    submitted, done = set(), set()
    started, load_seconds = time.perf_counter(), None
    close_error = None
    try:
        a = adapter()
        _, solver = a.native()
        solver.ensure_vllm_environment()
        from scripts.current_method.restart_n5_spell.vllm_backends import VLLMSolverBackend
        from vllm import SamplingParams, sampling_params
        backend = VLLMSolverBackend(model_path=Path(manifest['model_directory']), tokenizer_path=TOKENIZER,
                                    max_model_len=16384, temperature=.8, top_p=.9, max_tokens=32,
                                    gpu_memory_utilization=.8)
        load_seconds = backend.load_latency_seconds
        require(backend.model_identifier == manifest['model_directory'], 'MODEL_IDENTITY_DRIFT')
        choices = read(HERE / 'FINITE_CHOICES.json')
        for offset in range(0, len(rows), 8):
            batch = rows[offset:offset + 8]
            prompts, params = [], []
            for row in batch:
                require(sha(row['prompt_file']) == row['prompt_sha256'], 'PROMPT_DRIFT')
                prompts.append(Path(row['prompt_file']).read_text())
                constraint = {'choice': choices[row['constraint_key']]}
                kw = dict(row['sampling'])
                if manifest['decoder_contract']['structured_output_api'].startswith('structured_outputs='):
                    kw['structured_outputs'] = sampling_params.StructuredOutputsParams(**constraint)
                else:
                    kw['guided_decoding'] = sampling_params.GuidedDecodingParams(**constraint)
                params.append(SamplingParams(**kw))
            bid, submitted_at = f'batch{offset // 8:04d}', now()
            save(run / 'batches' / (bid + '_SUBMITTED.json'),
                 {'submitted_at_utc': submitted_at, 'request_ids': [r['request_id'] for r in batch],
                  'automatic_retry': False})
            submitted.update(r['request_id'] for r in batch)
            outputs = backend.engine.generate(prompts, params, use_tqdm=False)
            require(len(outputs) == len(batch), 'OUTPUT_COUNT_MISMATCH')
            for row, prompt, output in zip(batch, prompts, outputs):
                require(output.prompt == prompt and len(output.prompt_token_ids) == row['prompt_tokens'],
                        'OUTPUT_PROMPT_OR_TOKEN_DRIFT')
                require(len(output.outputs) == 1, 'COMPLETION_COUNT_MISMATCH')
                completion = output.outputs[0]
                save(run / 'responses' / (row['request_id'] + '.json'),
                     {'request_id': row['request_id'], 'candidate_id': row['candidate_id'],
                      'cohort_id': row['cohort_id'], 'model_role': role,
                      'model_directory': manifest['model_directory'], 'status': 'RETURNED',
                      'raw_output': completion.text, 'prompt_sha256': row['prompt_sha256'],
                      'seed': row['sampling']['seed'], 'prompt_token_ids': output.prompt_token_ids,
                      'completion_token_ids': completion.token_ids,
                      'request_submitted_at_utc': submitted_at, 'response_committed_at_utc': now(),
                      'metadata': {'finish_reason': completion.finish_reason,
                                   'stop_reason': completion.stop_reason,
                                   'prompt_token_count': len(output.prompt_token_ids),
                                   'completion_token_count': len(completion.token_ids),
                                   'backend_request_id': output.request_id,
                                   'structured_output_api': manifest['decoder_contract']['structured_output_api'],
                                   'sampling': row['sampling']}, 'gold_opened_by_solver': False})
                done.add(row['request_id'])
            save(run / 'batches' / (bid + '_COMPLETE.json'),
                 {'completed_at_utc': now(), 'request_ids': [r['request_id'] for r in batch]})
            print(json.dumps({'status': 'LIVE', 'model_role': role, 'completed': len(done),
                              'planned': len(rows)}), flush=True)
        save(run / 'COMPLETE.json', {'status': 'COMPLETE', 'completed_at_utc': now(),
                                     'actual_submitted_requests': len(submitted), 'returned_requests': len(done),
                                     'model_load_wall_seconds': load_seconds,
                                     'wall_seconds': time.perf_counter() - started})
    except BaseException as error:
        err = {'error': str(error), 'traceback': traceback.format_exc()}
        save(run / 'FAILED.json', {'status': 'FAILED', **err, 'failed_at_utc': now(),
                                   'actual_submitted_requests': len(submitted), 'returned_requests': len(done),
                                   'wall_seconds': time.perf_counter() - started})
        for row in rows:
            if row['request_id'] not in done:
                save(run / 'responses' / (row['request_id'] + '.json'),
                     {'request_id': row['request_id'], 'candidate_id': row['candidate_id'],
                      'cohort_id': row['cohort_id'], 'model_role': role,
                      'status': 'BACKEND_ERROR' if row['request_id'] in submitted else 'NOT_SUBMITTED',
                      'raw_output': None, 'response_committed_at_utc': now(), **err})
        raise
    finally:
        if backend is not None:
            try:
                backend.close()
            except BaseException as error:
                close_error = {'error': str(error), 'traceback': traceback.format_exc()}
                save(run / 'BACKEND_CLOSE_FAILED.json', close_error)
        files = [run / 'responses' / (row['request_id'] + '.json') for row in rows]
        if all(p.is_file() for p in files):
            require({p.stem for p in (run / 'responses').glob('*.json')} ==
                    {row['request_id'] for row in rows}, 'UNEXPECTED_RAW_RESPONSE')
            save(run / 'RAW_RESPONSE_FREEZE.json',
                 {'status': 'COMMITTED', 'created_at_utc': now(),
                  'execution_manifest': bindfile(run / 'EXECUTION_MANIFEST.json'),
                  'request_freeze': bindfile(HERE / 'TEST_REQUEST_FREEZE.json'),
                  'planned_request_count': len(rows), 'files': [bindfile(p) for p in files],
                  'gold_opened_by_live': False, 'backend_close_error': close_error})
        if close_error is not None:
            raise RuntimeError('BACKEND_CLOSE_FAILED_RAW_RECORDS_PRESERVED')


def cluster_ci(root_values):
    values = list(root_values)
    if not values:
        return None
    rng = random.Random(20261021)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(2000))
    return {'method': 'percentile_root_cluster_bootstrap', 'replicates': 2000, 'seed': 20261021,
            'lower_95': means[49], 'upper_95': means[1949], 'root_count': len(values),
            'scope': 'descriptive historical TEST regression only'}


def score(role):
    verify()
    # Keep every endpoint blind until all three matched raw populations are
    # immutable. A failed endpoint still contributes its full fixed denominator.
    for other_role in MODEL_ROLES:
        other_run = HERE / 'models' / other_role
        other_freeze = read(other_run / 'RAW_RESPONSE_FREEZE.json')
        require(other_freeze['status'] == 'COMMITTED' and
                other_freeze['planned_request_count'] == REQUEST_COUNT and
                len(other_freeze['files']) == REQUEST_COUNT, 'ALL_ROLES_RAW_FREEZE_REQUIRED_BEFORE_GOLD')
        verify_bindings(other_freeze['files'] + [other_freeze['execution_manifest'],
                                                other_freeze['request_freeze']])
    run = HERE / 'models' / role
    freeze = read(run / 'RAW_RESPONSE_FREEZE.json')
    verify_bindings(freeze['files'] + [freeze['execution_manifest'], freeze['request_freeze']])
    require(freeze['status'] == 'COMMITTED' and freeze['planned_request_count'] == REQUEST_COUNT and
            len(freeze['files']) == REQUEST_COUNT, 'RAW_DENOMINATOR_DRIFT')
    rows = lines(HERE / 'TEST_REQUESTS.jsonl')
    require({Path(x['path']).name for x in freeze['files']} ==
            {r['request_id'] + '.json' for r in rows}, 'RAW_REQUEST_BOUNDARY_DRIFT')
    save(run / 'SCORING_STARTED.json',
         {'started_at_utc': now(), 'raw_response_freeze': bindfile(run / 'RAW_RESPONSE_FREEZE.json'),
          'all_raw_records_committed_before_this_scorer_read_gold': True})
    gold_rows = lines(HERE / 'private/TEST_GOLD.jsonl')
    gold = {x['candidate_id']: x for x in gold_rows}
    public = {x['candidate_id']: x for x in lines(HERE / 'TEST_PUBLIC_QUESTIONS.jsonl')}
    require(len(gold) == len(gold_rows) == QUESTION_COUNT and set(gold) == set(public),
            'PRIVATE_TEST_QUESTION_BOUNDARY_DRIFT')
    maps, choices = read(HERE / 'TEST_DISPLAY_MAPS.json'), read(HERE / 'FINITE_CHOICES.json')
    native_scorer = adapter().native_scoring()
    scored, raw_records = [], []
    for row in rows:
        path = run / 'responses' / (row['request_id'] + '.json')
        raw = read(path)
        raw_records.append(raw)
        require(raw['request_id'] == row['request_id'] and
                raw['status'] in ('RETURNED', 'BACKEND_ERROR', 'NOT_SUBMITTED'), 'RESPONSE_ID_OR_STATUS_MISMATCH')
        if raw['status'] == 'RETURNED':
            require(raw['seed'] == row['sampling']['seed'] and
                    raw['metadata']['sampling'] == row['sampling'] and
                    raw['prompt_sha256'] == row['prompt_sha256'] and raw['model_role'] == role and
                    raw['metadata']['prompt_token_count'] == row['prompt_tokens'],
                    'RESPONSE_REQUEST_CONTRACT_DRIFT')
        result = native_scorer.score_response(gold[row['candidate_id']],
                                              {'variant': maps[row['prompt_id']]}, raw['raw_output'])
        valid = (raw['status'] == 'RETURNED' and raw['metadata']['finish_reason'] in ('stop', 'eos') and
                 raw['raw_output'] in choices[row['constraint_key']] and result.get('scored') is True)
        metrics = {}
        if valid:
            pred, g = result['content_prediction'], gold[row['candidate_id']]['gold_content_ids']
            if row['question_type'] == 'multi_select':
                pred, g = set(pred), set(g)
                metrics = {'fp': len(pred - g), 'fn': len(g - pred), 'jaccard': len(pred & g) / len(pred | g),
                           'overselected': len(pred) > len(g), 'selected_count': len(pred), 'gold_count': len(g)}
            else:
                rank = {v: i for i, v in enumerate(pred)}
                pairs = [(a, b) for i, a in enumerate(g) for b in g[i + 1:]]
                metrics = {'pair_accuracy': sum(rank[a] < rank[b] for a, b in pairs) / len(pairs),
                           'within_across_scene': None,
                           'within_across_scene_reason': 'Minimal frozen Gold packet lacks per-item scene anchors; not reconstructed from outcomes'}
        scored.append({'request_id': row['request_id'], 'candidate_id': row['candidate_id'],
                       'cohort_id': row['cohort_id'], 'root_id': row['root_id'],
                       'question_type': row['question_type'], 'reporting_role': row['reporting_role'],
                       'variant_id': row['variant_id'], 'sample_index': row['sample_index'],
                       'valid': valid, 'exact': bool(valid and result.get('exact_match')),
                       'partial_score': float(Fraction(result['partial_score_fraction'])) if valid else 0.0,
                       'prediction_content_ids': result.get('content_prediction') if valid else None,
                       'metrics': metrics, 'score': result, 'response_binding': bindfile(path)})
    questions = []
    for cid, q in public.items():
        rs = [r for r in scored if r['candidate_id'] == cid]
        require(rs and all(r['reporting_role'] == gold[cid]['reporting_role'] for r in rs),
                'QUESTION_REPORTING_ROLE_DRIFT')
        predictions = [json.dumps(r['prediction_content_ids']) for r in rs if r['valid']]
        questions.append({'candidate_id': cid, 'cohort_id': q['cohort_id'], 'root_id': q['root_id'],
                          'question_type': q['question_type'], 'reporting_role': gold[cid]['reporting_role'],
                          'planned_requests': len(rs), 'valid_requests': sum(r['valid'] for r in rs),
                          'exact_rate': statistics.mean(r['exact'] for r in rs),
                          'partial_score_mean': statistics.mean(r['partial_score'] for r in rs),
                          'display_permutation_consistency': max(collections.Counter(predictions).values()) / len(rs) if predictions else 0,
                          'any_correct': any(r['exact'] for r in rs), 'all_correct': all(r['exact'] for r in rs)})
    groups = {}
    for cohort, reporting in sorted({(q['cohort_id'], q['reporting_role']) for q in questions}):
        for typ in ('ALL', 'multi_select', 'ordering'):
            qs = [q for q in questions if q['cohort_id'] == cohort and q['reporting_role'] == reporting and
                  (typ == 'ALL' or q['question_type'] == typ)]
            if not qs:
                continue
            ids = {q['candidate_id'] for q in qs}
            rs = [r for r in scored if r['candidate_id'] in ids]
            roots = collections.defaultdict(list)
            for q in qs:
                roots[q['root_id']].append(q['exact_rate'])
            rootmeans = [statistics.mean(v) for v in roots.values()]
            metric = {'questions': len(qs), 'roots': len(roots), 'planned_requests': len(rs),
                      'valid_requests': sum(r['valid'] for r in rs), 'invalid_requests': sum(not r['valid'] for r in rs),
                      'exact_requests': sum(r['exact'] for r in rs),
                      'question_equal_exact': statistics.mean(q['exact_rate'] for q in qs),
                      'root_equal_exact': statistics.mean(rootmeans), 'root_cluster_uncertainty': cluster_ci(rootmeans),
                      'question_equal_partial': statistics.mean(q['partial_score_mean'] for q in qs),
                      'any_correct_questions': sum(q['any_correct'] for q in qs),
                      'all_correct_questions': sum(q['all_correct'] for q in qs),
                      'display_permutation_consistency': statistics.mean(q['display_permutation_consistency'] for q in qs)}
            for name in ('fp', 'fn', 'jaccard', 'overselected', 'selected_count', 'pair_accuracy'):
                vals = [r['metrics'][name] for r in rs if name in r['metrics']]
                metric[name + '_valid_call_mean'] = statistics.mean(vals) if vals else None
            ordering = [r for r in rs if r['question_type'] == 'ordering']
            metric['ordering_pair_accuracy_fixed_denominator'] = statistics.mean(r['metrics'].get('pair_accuracy', 0) for r in ordering) if ordering else None
            metric['invalid_ordering_requests'] = sum(not r['valid'] for r in ordering)
            metric['fixed_denominator_invalid_policy'] = 'Invalid responses score zero; valid-only metrics are reported separately'
            groups[cohort + '/' + reporting + '/' + typ] = metric
    save(run / 'private/SCORED_REQUESTS.json', {'requests': scored, 'questions': questions, 'not_for_challenger': True})
    complete = (run / 'COMPLETE.json').exists() and not (run / 'FAILED.json').exists()
    aggregate = {'schema_version': 'day7_TEST_aggregate_v1',
                 'status': 'COMPLETE' if complete else 'FAILED_FIXED_DENOMINATOR', 'model_role': role,
                 'execution_manifest': bindfile(run / 'EXECUTION_MANIFEST.json'),
                 'raw_response_freeze': bindfile(run / 'RAW_RESPONSE_FREEZE.json'),
                 'private_scores': bindfile(run / 'private/SCORED_REQUESTS.json'),
                 'denominator_questions': len(questions), 'denominator_requests': len(rows), 'groups': groups,
                 'known_prompt_tokens': sum(x.get('metadata', {}).get('prompt_token_count', 0) for x in raw_records),
                 'known_completion_tokens': sum(x.get('metadata', {}).get('completion_token_count', 0) for x in raw_records),
                 'unknown_usage_requests': sum(x['status'] == 'BACKEND_ERROR' for x in raw_records),
                 'not_submitted_requests': sum(x['status'] == 'NOT_SUBMITTED' for x in raw_records),
                 'runtime': read(run / ('COMPLETE.json' if complete else 'FAILED.json')),
                 'test_requests': REQUEST_COUNT, 'A2_B2': 'NOT_PRODUCED',
                 'checkpoint_selected_by_test': False, 'challenger_release': False,
                 'cohorts_have_separate_denominators': True, 'historical_root_exposure': True,
                 'historical_regression_only': True, 'no_unseen_root_claim': True,
                 'scored_at_utc': now()}
    save(run / 'AGGREGATE.json', aggregate)
    print(json.dumps({k: aggregate[k] for k in ('status', 'model_role', 'denominator_questions',
                                               'denominator_requests', 'groups')}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('verify')
    bind = sub.add_parser('bind')
    bind.add_argument('--role', choices=MODEL_ROLES, required=True)
    bind.add_argument('--model-dir', type=Path, required=True)
    bind.add_argument('--endpoint-proof', type=Path, required=True)
    for name in ('live', 'score', 'preflight'):
        command = sub.add_parser(name)
        command.add_argument('--role', choices=MODEL_ROLES, required=True)
        if name != 'score':
            command.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'verify':
        verify()
        print('PASS: frozen TEST requests, source bytes, prompts, maps, Gold packet hashes, runtime')
    elif args.command == 'bind':
        bind_model(args.role, args.model_dir, args.endpoint_proof)
    elif args.command == 'live':
        live(args.role, args.audit)
    elif args.command == 'preflight':
        execution_verify(args.role, args.audit)
        print('PASS: independent review, all fixed endpoints, selected model fresh hashes; no model loaded')
    else:
        score(args.role)


if __name__ == '__main__':
    main()
