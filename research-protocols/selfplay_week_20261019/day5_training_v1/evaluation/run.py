"""Day 5 DEV-only wrapper around the preserved native backend and v15 scorer.

prepare/verify/score are CPU operations. bind imports native runtime metadata but
does not load model weights. live is the only command that may instantiate vLLM.
No TEST content is exported; no evaluation output is released to Challenger.
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
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
WEEK = HERE.parents[1]
REPO = WEEK.parents[1]
VENV = REPO / '.venv-local4b'
TOKENIZER = Path('/root/autodl-tmp/Qwen3-4B')
ADAPTER = WEEK / 'day2/solver/adapter.py'
MODEL_ROLES = ('S0', 'A1', 'B1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
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
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def textsha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def canon(value):
    return textsha(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')))


def bindfile(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def verify_bindings(rows):
    for row in rows:
        p = Path(row['path'])
        require(p.is_file() and p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], 'BOUND_FILE_CHANGED:' + str(p))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def save_lines(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')


def adapter():
    key = 'day5_preserved_day2_adapter'
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ADAPTER)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def versions():
    return {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'tokenizers', 'vllm')}


def prepare():
    require(not (HERE / 'DEV_REQUEST_FREEZE.json').exists(), 'DEV_ALREADY_FROZEN')
    lockpath = WEEK / 'day4_schedule_v1/EVALUATION_LOCK.json'
    lock = read(lockpath)
    historical = next(x for x in lock['cohorts'] if x['cohort_id'] == 'historical115')
    previous = next(x for x in lock['cohorts'] if x['cohort_id'] == 'previous4MC')
    inputs = [bindfile(lockpath), historical['request_freeze'], previous['request_freeze'], lock['unchanged_sampling_policy'], bindfile(ADAPTER)]
    for cohort in (historical, previous):
        inputs += cohort['request_evidence_bindings']
    verify_bindings(inputs)
    histroot = Path(historical['request_freeze']['path']).parent / 'evidence'
    hp = {x['candidate_id']: x for x in lines(histroot / 'ADMITTED_PRIVATE.jsonl') if x['split'] == 'dev'}
    hq = {x['candidate_id']: x for x in lines(histroot / 'ADMITTED_PUBLIC.jsonl') if x['candidate_id'] in hp}
    hr = [x for x in lines(histroot / 'ADMITTED_REQUESTS.jsonl') if x['split'] == 'dev']
    hm = {x['prompt_id']: x for x in read(histroot / 'ADMITTED_DISPLAY_MAPS.json')['variants'] if x['split'] == 'dev'}
    prompts = {x['prompt_id']: x for x in lines(histroot / 'ADMITTED_PROMPTS.jsonl') if x['candidate_id'] in hp}
    choices = read(histroot / 'FINITE_ANSWER_CHOICES.json')['choices']
    require(len(hp) == 40 and len(hr) == 624 and len(hm) == 336, 'HISTORICAL_DEV_DENOMINATOR_CHANGED')
    rows = []
    questions = []
    private = []
    maps = {}
    for cid, q in hq.items():
        adapter().public_question(q)
        questions.append({'cohort_id': 'historical115_DEV', 'split': 'dev', **q})
        private.append({'cohort_id': 'historical115_DEV', 'root_id': q['root_id'], 'question_type': q['question_type'], **hp[cid]})
    for r in hr:
        pid = r['prompt_id']
        m = hm[pid]
        native_variant = {key: value for key, value in m.items() if key not in ('prompt_id', 'candidate_id', 'split', 'family_id', 'canonical_label_to_content', 'all_answer_space_roundtrips', 'unique_variant')}
        require(canon(native_variant) == r['display_mapping_sha256'], 'UPSTREAM_MAPPING_HASH_MISMATCH')
        maps[pid] = m
        rows.append({'cohort_id': 'historical115_DEV', 'root_id': hq[r['candidate_id']]['root_id'], **r, 'prompt_tokens': prompts[pid]['prompt_tokens']})
    prevroot = Path(previous['request_freeze']['path']).parent
    pr = [x for x in lines(prevroot / 'REQUESTS.jsonl') if x['original_split'] == 'dev']
    pm = {(x['candidate_id'], x['variant_id']): x for x in read(prevroot / 'DISPLAY_MAPS.json')}
    require(len(pr) == 36 and len({x['candidate_id'] for x in pr}) == 2, 'PREVIOUS_DEV_DENOMINATOR_CHANGED')
    for cid in sorted({x['candidate_id'] for x in pr}):
        pubpath = prevroot.parent / 'content/public' / (cid + '.json')
        privpath = prevroot.parent / 'content/private' / (cid + '.json')
        inputs.extend([bindfile(pubpath), bindfile(privpath)])
        q = read(pubpath)
        q = {key: value for key, value in q.items() if key in adapter().PUBLIC_KEYS}
        adapter().public_question(q)
        p = read(privpath)
        require(p['original_split'] == 'dev', 'NON_DEV_PRIVATE_DATA')
        ids = adapter().content_ids(q)
        questions.append({'cohort_id': 'previous2MC_DEV', 'split': 'dev', **q})
        private.append({'cohort_id': 'previous2MC_DEV', 'candidate_id': cid, 'root_id': q['root_id'], 'question_type': q['question_type'], 'split': 'dev', 'reporting_role': 'SEPARATE_PREVIOUS_MC_SUPPLEMENT', 'gold': p['gold'], 'gold_content_ids': [ids[x] for x in p['gold']]})
    for r in pr:
        pid = r['candidate_id'] + '::' + r['variant_id']
        m = pm[r['candidate_id'], r['variant_id']]
        require(m == {'candidate_id': r['candidate_id'], **r['display']}, 'PREVIOUS_MAP_CHANGED')
        ph = textsha(r['prompt'])
        pp = {'prompt_id': pid, 'candidate_id': r['candidate_id'], 'variant_id': r['variant_id'], 'prompt': r['prompt'], 'prompt_sha256': ph, 'prompt_tokens': r['prompt_tokens']}
        require(pid not in prompts or prompts[pid] == pp, 'CONFLICTING_PROMPT_BYTES')
        prompts[pid] = pp
        maps[pid] = m
        rows.append({'cohort_id': 'previous2MC_DEV', 'request_id': r['request_id'], 'candidate_id': r['candidate_id'], 'root_id': r['root_id'], 'split': 'dev', 'reporting_role': 'SEPARATE_PREVIOUS_MC_SUPPLEMENT', 'question_type': r['question_type'], 'branch': 'direct', 'variant_id': r['variant_id'], 'sample_index': r['sample_index'], 'prompt_id': pid, 'prompt_sha256': ph, 'prompt_tokens': r['prompt_tokens'], 'sampling': {'temperature': r['temperature'], 'top_p': r['top_p'], 'max_tokens': r['max_output_tokens'], 'seed': r['seed'], 'stop': r['stop'], 'include_stop_str_in_output': r['include_stop_str_in_output']}, 'constraint_key': r['question_type'], 'context_mode': 'full_context'})
    require(len(rows) == len({x['request_id'] for x in rows}) == 660 and len(questions) == 42, 'COMBINED_DENOMINATOR_CHANGED')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    for pid, p in prompts.items():
        require(textsha(p['prompt']) == p['prompt_sha256'], 'PROMPT_BYTES_CHANGED')
        ids = tok.encode(p['prompt'], add_special_tokens=False)
        require(len(ids) == p['prompt_tokens'] and len(ids) + 32 <= 16384, 'TOKEN_COUNT_OR_LENGTH_DRIFT')
        if 'token_ids_sha256' in p:
            require(canon(ids) == p['token_ids_sha256'], 'TOKEN_IDS_DRIFT')
        target = HERE / 'prompts' / (textsha(pid) + '.txt')
        target.parent.mkdir(exist_ok=True)
        with target.open('xb') as f:
            f.write(p['prompt'].encode())
        p['prompt_file'] = str(target)
    for q in questions:
        require(sha(q['source_path']) == q['source_sha256'], 'SOURCE_CHANGED')
        inputs.append(bindfile(q['source_path']))
    for r in rows:
        require(r['split'] == 'dev' and r['branch'] == 'direct', 'NON_DEV_OR_NON_DIRECT_REQUEST')
        p = prompts[r['prompt_id']]
        require(r['prompt_sha256'] == p['prompt_sha256'], 'REQUEST_PROMPT_DRIFT')
        r['prompt_file'] = p['prompt_file']
    save_lines(HERE / 'DEV_REQUESTS.jsonl', rows)
    save_lines(HERE / 'DEV_PUBLIC_QUESTIONS.jsonl', questions)
    save_lines(HERE / 'private/DEV_GOLD.jsonl', private)
    save(HERE / 'DEV_DISPLAY_MAPS.json', maps)
    save(HERE / 'FINITE_CHOICES.json', choices)
    save(HERE / 'PROMPT_INDEX.json', [{key: value for key, value in p.items() if key != 'prompt'} for p in prompts.values()])
    artifacts = [p for p in HERE.rglob('*') if p.is_file() and p.name != 'run.py']
    frozen = {'schema_version': 'day5_same_request_DEV_only_v1', 'status': 'CPU_PASS_AWAITING_FINAL_MODEL_RUNTIME_BINDINGS', 'created_at_utc': now(), 'model_roles': list(MODEL_ROLES), 'question_count': 42, 'requests_per_model': 660, 'total_planned_solver_requests': 1980, 'cohorts': [{'cohort_id': 'historical115_DEV', 'questions': 40, 'requests_per_model': 624, 'primary_questions': 35, 'gold6_control_questions': 5}, {'cohort_id': 'previous2MC_DEV', 'questions': 2, 'requests_per_model': 36, 'separate_denominator': True}], 'prompt_count': len(prompts), 'planned_prompt_tokens_per_model': sum(r['prompt_tokens'] for r in rows), 'maximum_output_tokens_per_model': len(rows) * 32, 'maximum_prompt_tokens': max(r['prompt_tokens'] for r in rows), 'batch_size': 8, 'max_model_len': 16384, 'gpu_memory_utilization': .8, 'prefix_caching': False, 'source_truncation': False, 'automatic_retry': False, 'sampling_inherited_byte_equivalent': True, 'all_prompt_and_token_bytes_verified': True, 'test_requests': 0, 'supplement12_requests': 0, 'checkpoint_selection': 'FIXED_FINAL_OPTIMIZER_STEP_56_NO_DEV_SELECTION', 'challenger_release': 'No DEV facts, examples, Gold, item outcomes or diagnosis; aggregate reporting only', 'historically_exposed': True, 'no_unseen_root_claim': True, 'raw_generation_gold_access': False, 'preparation_private_access': 'DEV split metadata/Gold read only to make isolated private scoring packet; no model called', 'execution_permitted': False, 'new_solver_calls': 0, 'new_actual_answer_scorer_runs': 0, 'model_weights_loaded': False, 'model_identity': {role: None for role in MODEL_ROLES}, 'input_bindings': list({x['path']: x for x in inputs}.values()), 'artifact_bindings': [bindfile(p) for p in sorted(artifacts)], 'wrapper': bindfile(__file__), 'runtime_versions': versions()}
    save(HERE / 'DEV_REQUEST_FREEZE.json', frozen)
    print(json.dumps({key: frozen[key] for key in ('status', 'question_count', 'requests_per_model', 'total_planned_solver_requests', 'planned_prompt_tokens_per_model', 'maximum_prompt_tokens', 'test_requests', 'new_solver_calls')}))


def verify():
    f = read(HERE / 'DEV_REQUEST_FREEZE.json')
    verify_bindings(f['input_bindings'] + f['artifact_bindings'] + [f['wrapper']])
    require(f['runtime_versions'] == versions(), 'RUNTIME_VERSIONS_CHANGED')
    rows = lines(HERE / 'DEV_REQUESTS.jsonl')
    require(len(rows) == 660 and all(x['split'] == 'dev' for x in rows), 'DEV_REQUEST_BOUNDARY_CHANGED')
    return f


def bind_model(role, modeldir, endpoint_proof):
    f = verify()
    require(Path(sys.prefix) == VENV, 'USE_PROVEN_LOCAL4B_VENV')
    proof = read(endpoint_proof)
    require(proof['model_role'] == role and Path(proof['model_directory']).resolve() == modeldir.resolve(), 'ENDPOINT_PROOF_ROLE_OR_PATH_MISMATCH')
    require(proof.get('evidence_bindings'), 'ENDPOINT_EVIDENCE_REQUIRED')
    verify_bindings(proof['evidence_bindings'])
    if role in ('A1', 'B1'):
        require(proof.get('fixed_final_optimizer_step') == proof.get('completed_optimizer_steps') == 56 and proof.get('training_completed') is True and proof.get('independent_initialization') == 'S0', 'FIXED_FINAL_STEP56_PROOF_REQUIRED')
        require(proof.get('merge_base') == '/root/autodl-tmp/Qwen3-4B' and proof.get('adapter_representation') == 'FULL_UPDATED_S0_CUSTOM_LORA', 'FULL_ADAPTER_MUST_MERGE_INTO_ORIGINAL_BASE')
    else:
        original = read(WEEK / 'day1/MODEL_BINDINGS.json')['models']['S0_M_clean_merged']
        require(modeldir.resolve() == Path(original['directory']).resolve(), 'S0_PATH_CHANGED')
        verify_bindings(original['files'])
    require((modeldir / 'config.json').is_file() and list(modeldir.glob('*.safetensors')), 'STANDARD_MERGED_HF_MODEL_REQUIRED')
    a = adapter()
    _, solver = a.native()
    from scripts.current_method.restart_n5_spell import vllm_backends, structured_qa_box_protocol
    decoder = solver.generation_config('multi_select')
    require(decoder['temperature'] == .8 and decoder['top_p'] == .9 and decoder['max_tokens'] == 32 and decoder['enable_thinking'] is False, 'NATIVE_DECODER_DRIFT')
    frozen_choices = read(HERE / 'FINITE_CHOICES.json')
    for typ in ('multi_select', 'ordering'):
        labels = dict.fromkeys('ABCDEF' if typ == 'multi_select' else ['I1', 'I2', 'I3', 'I4', 'I5', 'I6'])
        require(a.choices(typ, labels) == frozen_choices[typ], 'NATIVE_CHOICE_SPACE_DRIFT')
    code = {Path(__file__), ADAPTER, a.NATIVE / 'scoring.py', Path(vllm_backends.__file__), Path(structured_qa_box_protocol.__file__), Path(solver.__file__)}
    code.update(Path(m.__file__) for m in list(sys.modules.values()) if getattr(m, '__file__', None) and Path(m.__file__).suffix == '.py' and Path(m.__file__).is_relative_to(a.WORKTREE))
    files = [bindfile(p) for p in sorted(modeldir.iterdir()) if p.is_file()]
    out = HERE / 'models' / role / 'EXECUTION_MANIFEST.json'
    save(out, {'schema_version': 'day5_DEV_model_execution_manifest_v1', 'status': 'BOUND_FINAL_ENDPOINT_READY_FOR_REVIEW', 'model_role': role, 'model_directory': str(modeldir.resolve()), 'endpoint_proof': bindfile(endpoint_proof), 'endpoint_evidence_bindings': proof['evidence_bindings'], 'request_freeze': bindfile(HERE / 'DEV_REQUEST_FREEZE.json'), 'model_bindings': files, 'tokenizer_bindings': [bindfile(TOKENIZER / name) for name in ('tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json')], 'tokenizer_directory': str(TOKENIZER), 'runtime_versions': versions(), 'python_prefix': sys.prefix, 'decoder_contract': decoder, 'code_bindings': [bindfile(p) for p in sorted(code)], 'request_count': f['requests_per_model'], 'test_requests': 0, 'batch_size': 8, 'max_model_len': 16384, 'automatic_retry': False, 'weights_loaded_during_binding': False, 'created_at_utc': now()})
    print(json.dumps({'status': 'BOUND_FINAL_ENDPOINT_READY_FOR_REVIEW', 'model_role': role, 'requests': 660, 'manifest': str(out)}))


def execution_verify(role, audit):
    verify()
    mpath = HERE / 'models' / role / 'EXECUTION_MANIFEST.json'
    m = read(mpath)
    verify_bindings([m['request_freeze'], m['endpoint_proof']] + m['endpoint_evidence_bindings'] + m['model_bindings'] + m['tokenizer_bindings'] + m['code_bindings'])
    require(Path(sys.prefix) == VENV and m['runtime_versions'] == versions(), 'RUNTIME_CHANGED')
    review = read(audit)
    require(review.get('status') == 'PASS', 'EXECUTION_REVIEW_MUST_PASS')
    require(any(x['path'] == str(mpath) and x['sha256'] == sha(mpath) for x in review.get('audited_files', [])), 'REVIEW_NOT_BOUND_TO_EXECUTION_MANIFEST')
    verify_bindings(review['audited_files'])
    return m


def live(role, audit):
    m = execution_verify(role, audit)
    run = HERE / 'models' / role
    require(not (run / 'STARTED.json').exists(), 'ALREADY_STARTED_NO_AUTOMATIC_RETRY')
    rows = lines(HERE / 'DEV_REQUESTS.jsonl')
    save(run / 'STARTED.json', {'started_at_utc': now(), 'model_role': role, 'execution_manifest': bindfile(run / 'EXECUTION_MANIFEST.json'), 'review': bindfile(audit), 'private_gold_opened': False})
    backend = None
    submitted = set()
    done = set()
    started = time.perf_counter()
    load_seconds = None
    try:
        a = adapter()
        _, solver = a.native()
        solver.ensure_vllm_environment()
        from scripts.current_method.restart_n5_spell.vllm_backends import VLLMSolverBackend
        from vllm import SamplingParams, sampling_params
        backend = VLLMSolverBackend(model_path=Path(m['model_directory']), tokenizer_path=TOKENIZER, max_model_len=16384, temperature=.8, top_p=.9, max_tokens=32, gpu_memory_utilization=.8)
        load_seconds = backend.load_latency_seconds
        require(backend.model_identifier == m['model_directory'], 'MODEL_IDENTITY_DRIFT')
        choices = read(HERE / 'FINITE_CHOICES.json')
        for offset in range(0, len(rows), 8):
            batch = rows[offset:offset + 8]
            prompts = []
            params = []
            for r in batch:
                p = Path(r['prompt_file'])
                require(sha(p) == r['prompt_sha256'], 'PROMPT_DRIFT')
                prompts.append(p.read_text())
                constraint = {'choice': choices[r['constraint_key']]}
                kw = dict(r['sampling'])
                api = m['decoder_contract']['structured_output_api']
                if api.startswith('structured_outputs='):
                    kw['structured_outputs'] = sampling_params.StructuredOutputsParams(**constraint)
                else:
                    kw['guided_decoding'] = sampling_params.GuidedDecodingParams(**constraint)
                params.append(SamplingParams(**kw))
            bid = f'batch{offset // 8:04d}'
            submitted_at = now()
            save(run / 'batches' / (bid + '_SUBMITTED.json'), {'submitted_at_utc': submitted_at, 'request_ids': [r['request_id'] for r in batch], 'automatic_retry': False})
            submitted.update(r['request_id'] for r in batch)
            outputs = backend.engine.generate(prompts, params, use_tqdm=False)
            require(len(outputs) == len(batch), 'OUTPUT_COUNT_MISMATCH')
            for r, prompt, out in zip(batch, prompts, outputs):
                require(out.prompt == prompt and len(out.prompt_token_ids) == r['prompt_tokens'], 'OUTPUT_PROMPT_OR_TOKEN_DRIFT')
                require(len(out.outputs) == 1, 'COMPLETION_COUNT_MISMATCH')
                c = out.outputs[0]
                save(run / 'responses' / (r['request_id'] + '.json'), {'request_id': r['request_id'], 'candidate_id': r['candidate_id'], 'cohort_id': r['cohort_id'], 'model_role': role, 'model_directory': m['model_directory'], 'status': 'RETURNED', 'raw_output': c.text, 'prompt_sha256': r['prompt_sha256'], 'seed': r['sampling']['seed'], 'prompt_token_ids': out.prompt_token_ids, 'completion_token_ids': c.token_ids, 'request_submitted_at_utc': submitted_at, 'response_committed_at_utc': now(), 'metadata': {'finish_reason': c.finish_reason, 'stop_reason': c.stop_reason, 'prompt_token_count': len(out.prompt_token_ids), 'completion_token_count': len(c.token_ids), 'backend_request_id': out.request_id, 'structured_output_api': m['decoder_contract']['structured_output_api'], 'sampling': r['sampling']}, 'gold_opened_by_solver': False})
                done.add(r['request_id'])
            save(run / 'batches' / (bid + '_COMPLETE.json'), {'completed_at_utc': now(), 'request_ids': [r['request_id'] for r in batch]})
            print(json.dumps({'status': 'LIVE', 'model_role': role, 'completed': len(done), 'planned': len(rows)}), flush=True)
        save(run / 'COMPLETE.json', {'status': 'COMPLETE', 'completed_at_utc': now(), 'actual_submitted_requests': len(submitted), 'returned_requests': len(done), 'model_load_wall_seconds': load_seconds, 'wall_seconds': time.perf_counter() - started})
    except Exception as error:
        err = {'error': str(error), 'traceback': traceback.format_exc()}
        save(run / 'FAILED.json', {'status': 'FAILED', **err, 'actual_submitted_requests': len(submitted), 'returned_requests': len(done), 'wall_seconds': time.perf_counter() - started})
        for r in rows:
            if r['request_id'] not in done:
                save(run / 'responses' / (r['request_id'] + '.json'), {'request_id': r['request_id'], 'status': 'BACKEND_ERROR' if r['request_id'] in submitted else 'NOT_SUBMITTED', 'raw_output': None, 'response_committed_at_utc': now(), **err})
        raise
    finally:
        if backend:
            backend.close()
        files = [run / 'responses' / (r['request_id'] + '.json') for r in rows]
        if all(p.is_file() for p in files):
            require({p.stem for p in (run / 'responses').glob('*.json')} == {r['request_id'] for r in rows}, 'UNEXPECTED_RAW_RESPONSE')
            save(run / 'RAW_RESPONSE_FREEZE.json', {'status': 'COMMITTED', 'created_at_utc': now(), 'execution_manifest': bindfile(run / 'EXECUTION_MANIFEST.json'), 'request_freeze': bindfile(HERE / 'DEV_REQUEST_FREEZE.json'), 'planned_request_count': len(rows), 'files': [bindfile(p) for p in files], 'gold_opened_by_live': False})


def cluster_ci(root_values):
    values = list(root_values)
    if not values:
        return None
    rng = random.Random(20261021)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(2000))
    return {'method': 'percentile_root_cluster_bootstrap', 'replicates': 2000, 'seed': 20261021, 'lower_95': means[49], 'upper_95': means[1949], 'root_count': len(values), 'scope': 'descriptive historical regression only'}


def score(role):
    verify()
    run = HERE / 'models' / role
    freeze = read(run / 'RAW_RESPONSE_FREEZE.json')
    verify_bindings(freeze['files'] + [freeze['execution_manifest'], freeze['request_freeze']])
    require(freeze['planned_request_count'] == 660, 'RAW_DENOMINATOR_DRIFT')
    rows = lines(HERE / 'DEV_REQUESTS.jsonl')
    save(run / 'SCORING_STARTED.json', {'started_at_utc': now(), 'raw_response_freeze': bindfile(run / 'RAW_RESPONSE_FREEZE.json'), 'all_raw_records_committed_before_this_scorer_read_gold': True})
    gold = {x['candidate_id']: x for x in lines(HERE / 'private/DEV_GOLD.jsonl')}
    public = {x['candidate_id']: x for x in lines(HERE / 'DEV_PUBLIC_QUESTIONS.jsonl')}
    maps = read(HERE / 'DEV_DISPLAY_MAPS.json')
    choices = read(HERE / 'FINITE_CHOICES.json')
    native_scorer = adapter().native_scoring()
    scored = []
    raw_records = []
    for r in rows:
        p = run / 'responses' / (r['request_id'] + '.json')
        raw = read(p)
        raw_records.append(raw)
        require(raw['request_id'] == r['request_id'], 'RESPONSE_ID_MISMATCH')
        s = native_scorer.score_response(gold[r['candidate_id']], {'variant': maps[r['prompt_id']]}, raw['raw_output'])
        valid = raw['status'] == 'RETURNED' and raw['metadata']['finish_reason'] in ('stop', 'eos') and raw['raw_output'] in choices[r['constraint_key']] and s.get('scored') is True
        if raw['status'] == 'RETURNED':
            require(raw['seed'] == r['sampling']['seed'] and raw['metadata']['sampling'] == r['sampling'] and raw['prompt_sha256'] == r['prompt_sha256'] and raw['model_role'] == role, 'RESPONSE_REQUEST_CONTRACT_DRIFT')
        metrics = {}
        if valid:
            pred = s['content_prediction']
            g = gold[r['candidate_id']]['gold_content_ids']
            if r['question_type'] == 'multi_select':
                pred, g = set(pred), set(g)
                metrics = {'fp': len(pred - g), 'fn': len(g - pred), 'jaccard': len(pred & g) / len(pred | g), 'overselected': len(pred) > len(g), 'selected_count': len(pred), 'gold_count': len(g)}
            else:
                rank = {v: i for i, v in enumerate(pred)}
                pairs = [(a, b) for i, a in enumerate(g) for b in g[i + 1:]]
                metrics = {'pair_accuracy': sum(rank[a] < rank[b] for a, b in pairs) / len(pairs), 'within_across_scene': None, 'within_across_scene_reason': 'Minimal frozen historical Gold packet does not contain per-item scene anchors; not reconstructed from model outcomes'}
        scored.append({'request_id': r['request_id'], 'candidate_id': r['candidate_id'], 'cohort_id': r['cohort_id'], 'root_id': r['root_id'], 'question_type': r['question_type'], 'reporting_role': r['reporting_role'], 'variant_id': r['variant_id'], 'valid': valid, 'exact': bool(valid and s.get('exact_match')), 'partial_score': float(Fraction(s['partial_score_fraction'])) if valid else 0.0, 'metrics': metrics, 'score': s, 'response_binding': bindfile(p)})
    questions = []
    for cid, q in public.items():
        rs = [x for x in scored if x['candidate_id'] == cid]
        predictions = [json.dumps(x['score']['content_prediction']) for x in rs if x['valid']]
        questions.append({'candidate_id': cid, 'cohort_id': q['cohort_id'], 'root_id': q['root_id'], 'question_type': q['question_type'], 'reporting_role': gold[cid]['reporting_role'], 'planned_requests': len(rs), 'valid_requests': sum(x['valid'] for x in rs), 'exact_rate': statistics.mean(x['exact'] for x in rs), 'partial_score_mean': statistics.mean(x['partial_score'] for x in rs), 'display_permutation_consistency': max(collections.Counter(predictions).values()) / len(rs) if predictions else 0, 'any_correct': any(x['exact'] for x in rs), 'all_correct': all(x['exact'] for x in rs)})
    groups = {}
    for cohort, reporting in [('historical115_DEV', 'HISTORICAL_REGRESSION_PRIMARY'), ('historical115_DEV', 'GOLD6_CONTROL_ONLY'), ('previous2MC_DEV', 'SEPARATE_PREVIOUS_MC_SUPPLEMENT')]:
        for typ in ('ALL', 'multi_select', 'ordering'):
            qs = [q for q in questions if q['cohort_id'] == cohort and q['reporting_role'] == reporting and (typ == 'ALL' or q['question_type'] == typ)]
            if not qs:
                continue
            rs = [r for r in scored if r['candidate_id'] in {q['candidate_id'] for q in qs}]
            roots = collections.defaultdict(list)
            for q in qs:
                roots[q['root_id']].append(q['exact_rate'])
            rootmeans = [statistics.mean(v) for v in roots.values()]
            metric = {'questions': len(qs), 'roots': len(roots), 'planned_requests': len(rs), 'valid_requests': sum(r['valid'] for r in rs), 'exact_requests': sum(r['exact'] for r in rs), 'question_equal_exact': statistics.mean(q['exact_rate'] for q in qs), 'root_equal_exact': statistics.mean(rootmeans), 'root_cluster_uncertainty': cluster_ci(rootmeans), 'question_equal_partial': statistics.mean(q['partial_score_mean'] for q in qs), 'any_correct_questions': sum(q['any_correct'] for q in qs), 'all_correct_questions': sum(q['all_correct'] for q in qs), 'display_permutation_consistency': statistics.mean(q['display_permutation_consistency'] for q in qs)}
            for name in ('fp', 'fn', 'jaccard', 'overselected', 'selected_count', 'pair_accuracy'):
                vals = [r['metrics'][name] for r in rs if name in r['metrics']]
                metric[name + '_valid_call_mean'] = statistics.mean(vals) if vals else None
            ordering = [r for r in rs if r['question_type'] == 'ordering']
            metric['ordering_pair_accuracy_fixed_denominator'] = statistics.mean(r['metrics'].get('pair_accuracy', 0) for r in ordering) if ordering else None
            groups[cohort + '/' + reporting + '/' + typ] = metric
    save(run / 'private/SCORED_REQUESTS.json', {'requests': scored, 'questions': questions, 'not_for_challenger': True})
    aggregate = {'schema_version': 'day5_DEV_aggregate_v1', 'status': 'COMPLETE' if (run / 'COMPLETE.json').exists() else 'FAILED_FIXED_DENOMINATOR', 'model_role': role, 'execution_manifest': bindfile(run / 'EXECUTION_MANIFEST.json'), 'raw_response_freeze': bindfile(run / 'RAW_RESPONSE_FREEZE.json'), 'private_scores': bindfile(run / 'private/SCORED_REQUESTS.json'), 'denominator_questions': len(questions), 'denominator_requests': len(rows), 'groups': groups, 'known_prompt_tokens': sum(x.get('metadata', {}).get('prompt_token_count', 0) for x in raw_records), 'known_completion_tokens': sum(x.get('metadata', {}).get('completion_token_count', 0) for x in raw_records), 'unknown_usage_requests': sum(x['status'] == 'BACKEND_ERROR' for x in raw_records), 'not_submitted_requests': sum(x['status'] == 'NOT_SUBMITTED' for x in raw_records), 'runtime': read(run / ('COMPLETE.json' if (run / 'COMPLETE.json').exists() else 'FAILED.json')), 'test_requests': 0, 'checkpoint_selected_by_dev': False, 'challenger_release': False, 'historical_regression_only': True, 'no_unseen_root_claim': True, 'scored_at_utc': now()}
    save(run / 'AGGREGATE.json', aggregate)
    print(json.dumps({key: aggregate[key] for key in ('status', 'model_role', 'denominator_questions', 'denominator_requests', 'groups')}, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    sub.add_parser('verify')
    q = sub.add_parser('bind')
    q.add_argument('--role', choices=MODEL_ROLES, required=True)
    q.add_argument('--model-dir', type=Path, required=True)
    q.add_argument('--endpoint-proof', type=Path, required=True)
    for name in ('live', 'score'):
        q = sub.add_parser(name)
        q.add_argument('--role', choices=MODEL_ROLES, required=True)
        if name == 'live':
            q.add_argument('--audit', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'verify':
        verify()
        print('PASS: frozen DEV requests, source bytes, prompt bytes, maps, Gold packet, wrapper and runtime versions')
    elif args.command == 'bind':
        bind_model(args.role, args.model_dir, args.endpoint_proof)
    elif args.command == 'live':
        live(args.role, args.audit)
    else:
        score(args.role)


if __name__ == '__main__':
    main()
