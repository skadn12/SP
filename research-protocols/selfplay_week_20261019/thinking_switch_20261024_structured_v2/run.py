"""Independent structured thinking-switch engineering replication, v2."""
from __future__ import annotations
import argparse, collections, difflib, fcntl, hashlib, importlib.util, json, os, re, statistics, sys, time, traceback
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'priority_diversity_feedback_20261024_v1'
V1 = HERE.parent / 'thinking_switch_20261024_v1'
OUT = HERE / 'execution'
MODES = ('THINK_OFF', 'THINK_ON')
MAX_TOKENS, MODEL_LEN, BATCH = 6144, 16384, 8
STRUCTURED_CONFIG = {'reasoning_parser': 'qwen3', 'enable_in_reasoning': False}

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec); sys.modules[name] = obj; spec.loader.exec_module(obj)
    return obj

orig = load('thinking_switch_structured_v2_readonly_original', OLD / 'solver_runner.py')
read, save, bind, check = orig.read, orig.save, orig.bind, orig.check

def need(value, message):
    if not value: raise ValueError(message)

def structured_api_paths():
    package = Path(sys.prefix) / 'lib' / ('python' + str(sys.version_info.major) + '.' + str(sys.version_info.minor)) / 'site-packages' / 'vllm'
    return [package / relative for relative in ('entrypoints/llm.py', 'sampling_params.py', 'config/structured_outputs.py', 'reasoning/qwen3_reasoning_parser.py', 'reasoning/basic_parsers.py', 'v1/structured_output/__init__.py')]

def cpu_api_check():
    import inspect
    from vllm import LLM, SamplingParams, sampling_params
    from vllm.config import StructuredOutputsConfig
    need('structured_outputs_config' in inspect.signature(LLM.__init__).parameters, 'NO_STRUCTURED_ENGINE_API')
    config = StructuredOutputsConfig(**STRUCTURED_CONFIG)
    schema = {'type': 'object', 'properties': {'final_answer': {'type': 'string'}}, 'required': ['final_answer'], 'additionalProperties': False}
    params = SamplingParams(max_tokens=MAX_TOKENS, structured_outputs=sampling_params.StructuredOutputsParams(json=schema))
    need(config.reasoning_parser == 'qwen3' and config.enable_in_reasoning is False and params.structured_outputs.json == schema, 'STRUCTURED_API_OBJECT_DRIFT')
    return {'status': 'PASS_CPU_API_OBJECT_CONSTRUCTION_NO_MODEL', 'structured_outputs_config': STRUCTURED_CONFIG, 'runtime_versions': orig.versions(), 'implementation_bindings': [bind(p) for p in structured_api_paths()], 'gpu_model_started': False}

def checked_model_receipt(receipt):
    need(receipt['experiment_id'] == HERE.name, 'MODEL_RECEIPT_EXPERIMENT_DRIFT')
    need(Path(receipt['model_directory']).resolve() == orig.MODEL.resolve() and Path(receipt['tokenizer_directory']).resolve() == orig.TOKENIZER.resolve(), 'MODEL_RECEIPT_PATH_DRIFT')
    need(receipt['structured_outputs'] is True and receipt['structured_outputs_config'] == STRUCTURED_CONFIG and receipt['actual_structured_outputs_config'] == STRUCTURED_CONFIG, 'MODEL_STRUCTURED_CONFIG_DRIFT')
    need(receipt['runtime_versions'] == orig.versions() and receipt['runner_binding'] == bind(Path(__file__)), 'MODEL_RECEIPT_RUNTIME_OR_CODE_DRIFT')
    from types import SimpleNamespace
    for name in ('actual_hf_config', 'actual_hf_text_config'):
        orig.checked_runtime_config(SimpleNamespace(**receipt[name]))

def checked_raw_manifest(frozen, rows, token_ids):
    ids = [r['request_id'] for r in rows]
    need(len(ids) == len(set(ids)) == 120 and frozen['request_ids'] == ids and frozen['requests'] == len(ids), 'RAW_REQUEST_SET_DRIFT')
    check(frozen['started_binding'])
    need(frozen['started_binding'] == bind(OUT/'STARTED.json'), 'STARTED_PATH_DRIFT')
    started = read(OUT/'STARTED.json')
    need(started['experiment_id'] == HERE.name and started['request_ids'] == ids and started['requests'] == len(ids), 'STARTED_REQUEST_SET_DRIFT')
    need(started['preparation_binding'] == bind(OUT/'PREPARATION.json') and started['structured_outputs_config'] == STRUCTURED_CONFIG and started['gold_read'] is False, 'STARTED_CONFIG_OR_PREPARATION_DRIFT')
    submissions = frozen['submitted_ids']
    need(len(submissions) == len(set(submissions)) and submissions == [rid for rid in ids if rid in set(submissions)], 'SUBMISSION_ID_DRIFT')
    expected_raw = {str((OUT/'raw'/(rid+'.json')).resolve()) for rid in ids}
    expected_submitted = {str((OUT/'submitted'/(rid+'.json')).resolve()) for rid in submissions}
    for directory, bindings, expected in (('raw', frozen['bindings'], expected_raw), ('submitted', frozen['submission_bindings'], expected_submitted)):
        paths = [str(Path(b['path']).resolve()) for b in bindings]
        need(len(paths) == len(set(paths)) and set(paths) == expected, directory.upper()+'_BINDING_SET_DRIFT')
        need({str(p.resolve()) for p in (OUT/directory).glob('*.json')} == expected, directory.upper()+'_DIRECTORY_SET_DRIFT')
        for bound in bindings: check(bound)
    receipt_binding = frozen['model_receipt_binding']
    if receipt_binding is not None:
        check(receipt_binding); need(receipt_binding == bind(OUT/'MODEL_LOAD_RECEIPT.json'), 'MODEL_RECEIPT_BINDING_PATH_DRIFT')
        receipt = read(receipt_binding['path']); checked_model_receipt(receipt)
    else:
        need(not submissions and frozen['returned'] == 0, 'MISSING_MODEL_RECEIPT_AFTER_SUBMISSION')
    batches = frozen['batch_bindings']; batch_paths = [str(Path(b['path']).resolve()) for b in batches]
    need(len(batch_paths) == len(set(batch_paths)) and set(batch_paths) == {str(p.resolve()) for p in (OUT/'batches').glob('*.json')}, 'BATCH_DIRECTORY_SET_DRIFT')
    finished_ids, finished_batches = [], []
    for bound in batches:
        check(bound); batch = read(bound['path']); index = batch['batch_index']
        need(isinstance(index, int) and 0 <= index < (len(ids)+BATCH-1)//BATCH, 'BATCH_INDEX_DRIFT')
        need(str(Path(bound['path']).resolve()) == str((OUT/'batches'/(str(index).zfill(3)+'.json')).resolve()), 'BATCH_PATH_DRIFT')
        need(batch['request_ids'] == ids[index*BATCH:(index+1)*BATCH], 'BATCH_REQUEST_SET_DRIFT')
        finished_batches.append(index); finished_ids.extend(batch['request_ids'])
    need(finished_batches == list(range(len(finished_batches))) and len(finished_ids) == len(set(finished_ids)) and set(finished_ids) <= set(submissions), 'BATCH_ORDER_OR_SUBMISSION_DRIFT')
    returned = 0
    for row in rows:
        rid = row['request_id']; raw = read(OUT/'raw'/(rid+'.json'))
        need(raw['request_id'] == rid and raw['mode'] == row['mode'] and raw['base_request_id'] == row['base_request_id'], 'RAW_ID_OR_ARM_DRIFT')
        need(raw['status'] in ('RETURNED', 'BACKEND_ERROR', 'NOT_SUBMITTED'), 'UNKNOWN_RAW_STATUS')
        need((raw['status'] != 'NOT_SUBMITTED') == (rid in set(submissions)), 'RAW_SUBMISSION_STATUS_DRIFT')
        if rid in set(submissions):
            submission = read(OUT/'submitted'/(rid+'.json'))
            need(submission['request_id'] == rid and submission['experiment_id'] == HERE.name and submission['model_load_receipt_binding'] == receipt_binding, 'SUBMISSION_RECEIPT_DRIFT')
        if raw['status'] == 'RETURNED':
            returned += 1
            need(raw['model_load_receipt_binding'] == receipt_binding, 'RAW_MODEL_RECEIPT_DRIFT')
            errors = orig.capture.response_errors(raw, Path(row['prompt_path']).read_text(), token_ids[rid])
            need(raw['capture_errors'] == errors, 'CAPTURE_VALIDATION_DRIFT')
    need(returned == frozen['returned'], 'RETURNED_COUNT_DRIFT')
    if frozen['failure'] is None:
        need(submissions == ids and returned == len(ids) and finished_batches == list(range((len(ids)+BATCH-1)//BATCH)), 'INCOMPLETE_SUCCESS_FREEZE')

def public():
    prep = read(OUT / 'PREPARATION.json')
    for b in prep['bindings']: check(b)
    rows = read(OUT / 'PUBLIC_REQUESTS.json')['requests']; tokens = read(OUT / 'PROMPT_TOKEN_IDS.json')
    need([r['request_id'] for r in rows] == prep['fixed_order'], 'REQUEST_ORDER_DRIFT')
    need(len(rows) == len({r['request_id'] for r in rows}) == 120 and set(tokens) == {r['request_id'] for r in rows}, 'REQUEST_SET_DRIFT')
    need(all(r['condition'] == 'FULL' and r['mode'] in MODES for r in rows), 'PUBLIC_ARM_DRIFT')
    need(all(r['enable_thinking'] == (r['mode']=='THINK_ON') and r['prompt_tokens'] == len(tokens[r['request_id']]) and r['prompt_tokens']+MAX_TOKENS<=MODEL_LEN for r in rows), 'PUBLIC_THINK_OR_BUDGET_DRIFT')
    need(prep['structured_outputs_config'] == STRUCTURED_CONFIG, 'PREP_STRUCTURED_CONFIG_DRIFT')
    check(prep['cpu_api_check_binding'])
    return prep, rows, tokens

def prepare():
    need(not OUT.exists(), 'NO_PREPARE_OVERWRITE')
    api_receipt = cpu_api_check()
    oldprep, oldrows, oldtokens = orig.public_preparation()
    endpoint, config = orig.endpoint_receipt()
    full = [r for r in oldrows if r['condition'] == 'FULL']
    need(len(full) == 60 and len({r['candidate_id'] for r in full}) == 30, 'FROZEN_30_REQUIRED')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(orig.TOKENIZER), local_files_only=True)
    from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
    reasoner = Qwen3ReasoningParser(tokenizer=tok)
    v1_rows = read(V1/'execution/PUBLIC_REQUESTS.json')['requests']
    v1_by_key = {(r['base_request_id'],r['mode']):r for r in v1_rows}
    v1_tokens = read(V1/'execution/PROMPT_TOKEN_IDS.json')
    need(len(v1_rows) == len(v1_by_key) == 120, 'V1_PUBLIC_REQUEST_SET_DRIFT')
    oldpattern = r'<\|im_start\|>system\n(.*?)<\|im_end\|>\n<\|im_start\|>user\n(.*?)<\|im_end\|>\n<\|im_start\|>assistant\n<think>\n\n</think>\n\n'
    oldtail = 'Return only JSON, with\nitem_evidence first, answer_reason second and final_answer last. Provide brief\nevidence-based explanations, not extended private deliberation.'
    newtail = 'Your final response must be JSON, with item_evidence first, answer_reason second and final_answer last. Keep the explanations in the final JSON brief and evidence-based.'
    rows, tokens, bindings, diffs, messages_hash = [], {}, list(oldprep['bindings']), [], {}
    for index, old in enumerate(full):
        text = Path(old['prompt_path']).read_text(); need(tok.encode(text, add_special_tokens=False) == oldtokens[old['request_id']], 'OLD_TOKEN_DRIFT')
        match = re.fullmatch(oldpattern, text, re.S); need(match, 'UNEXPECTED_OLD_CHAT_TEMPLATE')
        system, user = match.groups(); need(user.count(oldtail) == 1, 'UNEXPECTED_OLD_INSTRUCTIONS')
        newuser = user.replace(oldtail, newtail)
        newsystem = system.replace('Give concise, verifiable explanations', 'In the final response, give concise, verifiable explanations')
        messages = [{'role': 'system', 'content': newsystem}, {'role': 'user', 'content': newuser}]
        mh = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        messages_hash[old['request_id']] = mh
        diffs.append({'base_request_id': old['request_id'], 'system_diff': list(difflib.unified_diff(system.splitlines(), newsystem.splitlines())), 'user_diff': list(difflib.unified_diff(user.splitlines(), newuser.splitlines()))})
        modes = MODES if index % 2 == 0 else tuple(reversed(MODES))
        for mode in modes:
            on = mode == 'THINK_ON'; prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=on)
            ids = tok.encode(prompt, add_special_tokens=False); need(len(ids) + MAX_TOKENS <= MODEL_LEN, 'NO_INPUT_TRUNCATION')
            previous = v1_by_key[(old['request_id'],mode)]
            need(all(previous[k] == old[k] for k in ('candidate_id','root_id','question_type','task_operation','question','source_binding','display','labels','variant_id','sampling_seed')), 'V1_QUESTION_OR_SEED_DRIFT')
            need(previous['messages_sha256'] == mh and Path(previous['prompt_path']).read_text() == prompt and v1_tokens[previous['request_id']] == ids, 'V1_MESSAGES_OR_TOKENS_DRIFT')
            need(reasoner.is_reasoning_end(ids) is (not on), 'QWEN3_REASONING_GATE_CPU_DRIFT')
            row = dict(old); rid = old['request_id'].replace('__B1_priority_v1', '__' + mode + '__structured_v2')
            pp = OUT / 'prompts' / (rid + '.txt'); pp.parent.mkdir(parents=True, exist_ok=True)
            with pp.open('x') as f: f.write(prompt)
            row.update(request_id=rid, base_request_id=old['request_id'], mode=mode, enable_thinking=on, prompt_path=str(pp), prompt_tokens=len(ids), messages_sha256=mh)
            rows.append(row); tokens[rid] = ids; bindings.append(bind(pp))
    need(len(rows) == 120 and len({r['request_id'] for r in rows}) == 120, 'REQUEST_COUNT')
    need(all(len({r['messages_sha256'] for r in rows if r['base_request_id'] == old['request_id']}) == 1 for old in full), 'MODE_MESSAGE_DIFFERENCE')
    audit = []
    for root in sorted({r['root_id'] for r in full}):
        for operation in ('MC_QUANTITATIVE_RELATION', 'ORDER_EXPLICIT_TIME'):
            choices = sorted({r['candidate_id'] for r in full if r['root_id'] == root and r['task_operation'] == operation})
            audit.append({'root_id': root, 'operation': operation, 'candidate_id': choices[0] if choices else None, 'variant_id': 'perm_00', 'modes': list(MODES)})
    save(OUT / 'PUBLIC_REQUESTS.json', {'requests': rows})
    save(OUT / 'PROMPT_TOKEN_IDS.json', tokens)
    save(OUT / 'COMMON_MESSAGE_DIFFS.json', {'records': diffs})
    need(audit == read(V1/'execution/SEMANTIC_AUDIT_SELECTION.json')['records'], 'V1_AUDIT_SELECTION_DRIFT')
    save(OUT / 'SEMANTIC_AUDIT_SELECTION.json', {'selection_before_generation': True, 'records': audit})
    save(OUT / 'CPU_API_CHECK.json', {**api_receipt, 'reasoning_gate_prompts_checked':len(rows), 'v1_public_prompts_and_tokens_exactly_matched':True, 'v1_audit_selection_exactly_matched':True})
    protocol = {'experiment_id': HERE.name, 'design': 'paired_same_messages_same_budget_thinking_switch_structured_v2', 'engineering_followup_reason': 'observed_v1_output_contract_failures_not_accuracy_selection', 'questions': 30, 'roots': sorted({r['root_id'] for r in full}), 'requests_per_mode': 60, 'modes': list(MODES), 'max_tokens': MAX_TOKENS, 'max_model_len': MODEL_LEN, 'batch_size': BATCH, 'temperature': .8, 'top_p': .9, 'top_k': -1, 'repetition_penalty': 1.0, 'engine_seed': 0, 'structured_outputs': True, 'structured_outputs_config': STRUCTURED_CONFIG, 'schema': 'same_frozen_old_P_schema_final_only', 'prefix_caching': False, 'automatic_retries': 0, 'historical_scores_used_as_control': False, 'generation_reads_gold': False, 'no_training': True, 'no_dev_test': True, 'native_thinking_observed_separately': True, 'old_run_max_tokens_for_disclosure': 3072}
    save(OUT / 'PROTOCOL_FREEZE.json', protocol)
    for p in (Path(__file__), HERE / 'PROTOCOL_ZH.md', OUT / 'PROTOCOL_FREEZE.json', OUT / 'PUBLIC_REQUESTS.json', OUT / 'PROMPT_TOKEN_IDS.json', OUT / 'COMMON_MESSAGE_DIFFS.json', OUT / 'SEMANTIC_AUDIT_SELECTION.json', OUT / 'CPU_API_CHECK.json', OLD / 'FINAL_MANIFEST.json', V1/'run.py', V1/'PROTOCOL_ZH.md', V1/'execution/PREPARATION.json', V1/'execution/PROTOCOL_FREEZE.json', V1/'execution/PUBLIC_REQUESTS.json', V1/'execution/PROMPT_TOKEN_IDS.json', V1/'execution/SEMANTIC_AUDIT_SELECTION.json', *structured_api_paths()):
        bindings.append(bind(p))
    save(OUT / 'PREPARATION.json', {'experiment_id': HERE.name, 'requests': len(rows), 'fixed_order': [r['request_id'] for r in rows], 'bindings': bindings, 'private_scoring_binding': bind(OLD / 'execution/PRIVATE_SCORING.json'), 'old_content_bindings': [bind(OLD / n) for n in ('SELECTION_FREEZE.json', 'CANDIDATES_PRIVATE.json', 'CONTENT_REVIEW_AI.json', 'CONTENT_FREEZE.json')], 'endpoint_path': str(orig.diagnostic.ENDPOINT), 'config_receipt': config, 'runtime_versions': orig.versions(), 'max_prompt_tokens': max(r['prompt_tokens'] for r in rows), 'protocol_binding': bind(OUT / 'PROTOCOL_FREEZE.json'), 'structured_outputs_config': STRUCTURED_CONFIG, 'cpu_api_check_binding': bind(OUT/'CPU_API_CHECK.json')})
    print(json.dumps({'prepared': len(rows), 'unique_questions': 30, 'max_prompt_tokens': max(r['prompt_tokens'] for r in rows), 'max_output_tokens': MAX_TOKENS, 'audit_questions': sum(a['candidate_id'] is not None for a in audit)}), flush=True)

def metric_dict(metric):
    if metric is None: return {}
    names = ('arrival_time', 'first_scheduled_time', 'first_token_time', 'last_token_time', 'finished_time', 'time_in_queue', 'scheduler_time', 'model_forward_time', 'model_execute_time')
    return {name: getattr(metric, name) for name in names if isinstance(getattr(metric, name, None), (str, int, float, bool))}

def run():
    need(not (OUT / 'STARTED.json').exists(), 'NO_RERUN_OR_OVERWRITE')
    prep, rows, token_ids = public(); endpoint, config = orig.endpoint_receipt()
    need(config == prep['config_receipt'] and orig.versions() == prep['runtime_versions'], 'RUNTIME_DRIFT')
    for b in endpoint['model_bindings'] + endpoint['tokenizer_bindings']: check(b)
    for k, v in {'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'VLLM_USE_FLASHINFER_SAMPLER': '0', 'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false'}.items(): os.environ[k] = v
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(orig.TOKENIZER), local_files_only=True)
    orig.checked_prompt_tokens(tok, rows, token_ids)
    from vllm import LLM, SamplingParams, sampling_params
    lock = open('/tmp/gxy-selfplay-gpu.lock', 'a+'); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    save(OUT / 'STARTED.json', {'experiment_id': HERE.name, 'time_unix': time.time(), 'preparation_binding': bind(OUT / 'PREPARATION.json'), 'requests': len(rows), 'request_ids':[r['request_id'] for r in rows], 'gold_read': False, 'structured_outputs_config': STRUCTURED_CONFIG})
    engine, failure, submitted = None, None, []
    started = time.perf_counter()
    try:
        overrides = {'rope_theta': orig.EXPECTED_THETA, 'rope_parameters': {'rope_type': 'default', 'rope_theta': orig.EXPECTED_THETA}}
        engine = LLM(model=str(orig.MODEL), tokenizer=str(orig.TOKENIZER), tensor_parallel_size=1, max_model_len=MODEL_LEN, gpu_memory_utilization=.8, dtype='bfloat16', trust_remote_code=True, enable_prefix_caching=False, max_num_seqs=BATCH, seed=0, hf_overrides=overrides, structured_outputs_config=STRUCTURED_CONFIG)
        cfg = engine.llm_engine.model_config
        actual_structured = engine.llm_engine.vllm_config.structured_outputs_config
        receipt = {'experiment_id':HERE.name, 'model_directory': str(orig.MODEL), 'tokenizer_directory': str(orig.TOKENIZER), 'config': config, 'actual_hf_config': orig.checked_runtime_config(cfg.hf_config), 'actual_hf_text_config': orig.checked_runtime_config(cfg.hf_text_config), 'runtime_versions': orig.versions(), 'runner_binding': bind(Path(__file__)), 'structured_outputs': True, 'structured_outputs_config': STRUCTURED_CONFIG, 'actual_structured_outputs_config':{'reasoning_parser':actual_structured.reasoning_parser,'enable_in_reasoning':actual_structured.enable_in_reasoning}}
        checked_model_receipt(receipt)
        need(Path(cfg.model).resolve() == orig.MODEL.resolve(), 'WRONG_MODEL')
        save(OUT / 'MODEL_LOAD_RECEIPT.json', receipt)
        print('MODEL_LOADED', flush=True)
        for i in range(0, len(rows), BATCH):
            batch = rows[i:i+BATCH]; prompts = [Path(r['prompt_path']).read_text() for r in batch]
            params = [SamplingParams(temperature=.8, top_p=.9, top_k=-1, repetition_penalty=1.0, max_tokens=MAX_TOKENS, seed=r['sampling_seed'], skip_special_tokens=False, structured_outputs=sampling_params.StructuredOutputsParams(json=read(r['schema_path']))) for r in batch]
            t0 = time.perf_counter()
            for r in batch:
                save(OUT / 'submitted' / (r['request_id'] + '.json'), {'experiment_id':HERE.name, 'request_id': r['request_id'], 'time_unix': time.time(), 'model_load_receipt_binding': bind(OUT / 'MODEL_LOAD_RECEIPT.json')}); submitted.append(r['request_id'])
            results = engine.generate(prompts, params, use_tqdm=False)
            need(len(results) == len(batch), 'BACKEND_RESULT_COUNT')
            for r, result in zip(batch, results):
                raw = orig.base.preserve_response(r, [result], endpoint)
                raw.update(mode=r['mode'], base_request_id=r['base_request_id'], model_load_receipt_binding=bind(OUT / 'MODEL_LOAD_RECEIPT.json'), backend_metrics=metric_dict(getattr(result, 'metrics', None)), batch_index=i//BATCH)
                raw['decoded_output_skip_special_tokens_false'] = tok.decode(raw.get('output_token_ids', []), skip_special_tokens=False)
                raw['capture_errors'] = orig.capture.response_errors(raw, Path(r['prompt_path']).read_text(), token_ids[r['request_id']])
                save(OUT / 'raw' / (r['request_id'] + '.json'), raw)
            save(OUT / 'batches' / (str(i//BATCH).zfill(3) + '.json'), {'batch_index': i//BATCH, 'request_ids': [r['request_id'] for r in batch], 'wall_seconds': time.perf_counter() - t0})
            print(json.dumps({'completed': i+len(batch), 'total': len(rows), 'batch_seconds': round(time.perf_counter()-t0, 3), 'finishes': [x.outputs[0].finish_reason for x in results], 'output_tokens': [len(x.outputs[0].token_ids) for x in results]}), flush=True)
    except BaseException:
        failure = traceback.format_exc(); print(failure, flush=True)
    finally:
        if engine is not None:
            try:
                core = getattr(engine.llm_engine, 'engine_core', None)
                if callable(getattr(core, 'shutdown', None)): core.shutdown()
            except BaseException:
                failure = (failure or '') + traceback.format_exc()
        for r in rows:
            p = OUT / 'raw' / (r['request_id'] + '.json')
            if not p.exists(): save(p, {'request_id': r['request_id'], 'status': 'BACKEND_ERROR' if r['request_id'] in submitted else 'NOT_SUBMITTED', 'raw_output': None, 'failure': failure, 'mode': r['mode'], 'base_request_id':r['base_request_id']})
        paths = [OUT / 'raw' / (r['request_id'] + '.json') for r in rows]
        frozen = {'requests': len(rows), 'request_ids': [r['request_id'] for r in rows], 'submitted_ids': submitted, 'returned': sum(read(p)['status'] == 'RETURNED' for p in paths), 'failure': failure, 'wall_seconds': time.perf_counter()-started, 'bindings': [bind(p) for p in paths], 'submission_bindings': [bind(p) for p in sorted((OUT/'submitted').glob('*.json'))], 'batch_bindings': [bind(p) for p in sorted((OUT/'batches').glob('*.json'))], 'model_receipt_binding': bind(OUT / 'MODEL_LOAD_RECEIPT.json') if (OUT/'MODEL_LOAD_RECEIPT.json').exists() else None, 'started_binding': bind(OUT/'STARTED.json')}
        checked_raw_manifest(frozen, rows, token_ids)
        save(OUT / 'RAW_RESPONSE_FREEZE.json', frozen)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN); lock.close()
    if failure: raise RuntimeError('Frozen backend failure; no automatic retry')

def split_output(raw):
    need(isinstance(raw, str), 'NO_OUTPUT_TEXT')
    text = raw.strip()
    if text.endswith('<|im_end|>'): text = text[:-len('<|im_end|>')].rstrip()
    need('<tool_call>' not in text and '<|im_start|>' not in text, 'UNEXPECTED_TOOL_OR_CHAT_WRAPPER')
    if '<think>' not in text and '</think>' not in text: return None, text, 'NO_THINK_BLOCK'
    need(text.startswith('<think>') and text.count('<think>') == text.count('</think>') == 1, 'UNBALANCED_OR_MULTIPLE_THINK')
    think, final = text[len('<think>'):].split('</think>', 1)
    need(final.strip(), 'MISSING_FINAL_AFTER_THINK')
    return think.strip(), final.strip(), 'CLOSED_NONEMPTY' if think.strip() else 'CLOSED_EMPTY'

def legal_answer(final, row):
    obj = json.loads(final); need(isinstance(obj, dict) and isinstance(obj.get('final_answer'), str), 'NO_FINAL_ANSWER_FIELD')
    match = re.fullmatch(r'<box>([A-F](?: [A-F]){0,5})</box>', obj['final_answer']); need(match, 'INVALID_ANSWER_BOX')
    ans = match.group(1).split(); need(len(ans) == len(set(ans)) and set(ans) <= set(row['labels']), 'INVALID_ANSWER_LABELS')
    if row['question_type'] == 'ordering': need(set(ans) == set(row['labels']), 'INCOMPLETE_ORDERING')
    return obj, ans

def average(values): return statistics.mean(values) if values else None

def analyze():
    prep, rows, token_ids = public(); frozen = read(OUT/'RAW_RESPONSE_FREEZE.json')
    checked_raw_manifest(frozen, rows, token_ids)
    need(frozen['request_ids'] == [r['request_id'] for r in rows] and len(frozen['bindings']) == len(rows), 'RAW_SET_DRIFT')
    for b in frozen['bindings'] + frozen['submission_bindings'] + frozen['batch_bindings']: check(b)
    check(prep['private_scoring_binding'])
    for b in prep['old_content_bindings']: check(b)
    private = {r['request_id']: r for r in read(prep['private_scoring_binding']['path'])['requests']}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(orig.TOKENIZER), local_files_only=True)
    records = []
    for row in rows:
        path = OUT/'raw'/(row['request_id']+'.json'); raw = read(path)
        think, final, boundary, parsed, answer = None, None, 'UNPARSED', None, []
        errors, valid, answer_readable = [], False, False
        try:
            need(raw['status'] == 'RETURNED' and raw.get('finish_reason') == 'stop', 'BACKEND_OR_LENGTH_FAILURE')
            need(not raw.get('capture_errors'), 'CAPTURE_ERRORS')
            think, final, boundary = split_output(raw['raw_output'])
            parsed, answer = legal_answer(final, row); answer_readable = True
            orig.base.validate_output(final, row, read(row['schema_path'])); valid = True
        except Exception as e: errors.append(str(e))
        if final is None and isinstance(raw.get('raw_output'), str):
            try: think, final, boundary = split_output(raw['raw_output'])
            except Exception as e: errors.append(str(e)); boundary = str(e)
        presentation = private[row['base_request_id']]['presentation']
        strict = orig.score_record(row, presentation, answer, valid)
        readable = orig.score_record(row, presentation, answer, answer_readable)
        counts = {'true_positive': len(set(readable['prediction']) & set(readable['gold'])) if row['question_type']=='multi_select' else None}
        if row['question_type']=='multi_select':
            tp = counts['true_positive']; fp = readable['false_positive']; fn = readable['false_negative']
            counts.update(precision=tp/(tp+fp) if tp+fp else 0.0, recall=tp/(tp+fn), f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0)
        mechanical = {}
        if isinstance(parsed, dict):
            ev = parsed.get('item_evidence', {})
            if isinstance(ev, dict):
                supported = {k for k,v in ev.items() if isinstance(v,dict) and v.get('judgment')=='supported'}
                mechanical['supported_labels'] = sorted(supported)
                mechanical['supported_set_equals_final'] = supported == set(answer) if answer_readable and row['question_type']=='multi_select' else None
        metrics = raw.get('backend_metrics', {})
        latency = metrics.get('finished_time', 0)-metrics.get('arrival_time', 0) if metrics.get('finished_time') is not None and metrics.get('arrival_time') is not None else None
        records.append({'request_id':row['request_id'], 'base_request_id':row['base_request_id'], 'mode':row['mode'], 'strict':strict, 'answer_readable':answer_readable, 'answer_readable_score':readable, 'mc_partial':counts, 'errors':errors, 'finish_reason':raw.get('finish_reason'), 'thinking_boundary':boundary, 'thinking_text':think, 'final_text':final, 'parsed':parsed, 'output_tokens':len(raw.get('output_token_ids',[])), 'thinking_text_tokens':len(tok.encode(think, add_special_tokens=False)) if think else 0, 'final_text_tokens':len(tok.encode(final,add_special_tokens=False)) if final else 0, 'backend_latency_seconds':latency, 'mechanical':mechanical, 'raw_binding':bind(path)})
    save(OUT/'SCORES.json', {'records':records, 'raw_freeze_binding':bind(OUT/'RAW_RESPONSE_FREEZE.json'), 'private_scoring_binding':prep['private_scoring_binding']})
    summaries=[]
    for mode in MODES:
        for group in ('ALL','MC2_5_PRIMARY','MC6_CONTROL','ORDERING'):
            chosen=[r for r in records if r['mode']==mode and (group=='ALL' or group=='ORDERING' and r['strict']['question_type']=='ordering' or group=='MC2_5_PRIMARY' and r['strict']['gold_count'] in (2,3,4,5) or group=='MC6_CONTROL' and r['strict']['gold_count']==6)]
            summaries.append({'mode':mode,'group':group,**orig.base.aggregate([r['strict'] for r in chosen]),'answer_readable_correct':sum(r['answer_readable_score']['exact'] for r in chosen),'closed_nonempty_thinking':sum(r['thinking_boundary']=='CLOSED_NONEMPTY' for r in chosen),'length_truncated':sum(r['finish_reason']=='length' for r in chosen),'mean_output_tokens':average([r['output_tokens'] for r in chosen]),'mean_thinking_text_tokens':average([r['thinking_text_tokens'] for r in chosen]),'mean_backend_latency_seconds':average([r['backend_latency_seconds'] for r in chosen if r['backend_latency_seconds'] is not None]),'mc_macro_f1':average([r['mc_partial']['f1'] for r in chosen if 'f1' in r['mc_partial']]),'item_final_mismatch':sum(r['mechanical'].get('supported_set_equals_final') is False for r in chosen)})
    bybase=collections.defaultdict(dict)
    for r in records: bybase[r['base_request_id']][r['mode']]=r
    paired=[]
    for key,m in bybase.items():
        a,b=m['THINK_OFF'],m['THINK_ON']; paired.append({'base_request_id':key,'candidate_id':a['strict']['candidate_id'],'variant_id':a['strict']['variant_id'],'off_exact':a['strict']['exact'],'on_exact':b['strict']['exact'],'transition':('BOTH_CORRECT' if a['strict']['exact'] else 'ON_GAIN') if b['strict']['exact'] else ('ON_LOSS' if a['strict']['exact'] else 'BOTH_WRONG')})
    save(OUT/'ANALYSIS_SUMMARY.json',{'summaries':summaries,'paired_transitions':dict(collections.Counter(p['transition'] for p in paired)),'pairs':paired,'unique_questions':30,'roots':4,'raw_run_wall_seconds':frozen['wall_seconds'],'structured_outputs':True,'structured_outputs_config':STRUCTURED_CONFIG,'inference_limits':'Exploratory Train-only structured engineering followup, two dependent displays/question; neutral shared prompt and reasoning-aware final-JSON constraint; separate from v1; no training or causal feedback claims; semantic audit covers fixed subset only.'})
    print(json.dumps({'summaries':summaries,'paired':dict(collections.Counter(p['transition'] for p in paired))},ensure_ascii=False),flush=True)

def cpu_check():
    cpu_api_check()
    need(split_output('{"a":1}') == (None,'{"a":1}','NO_THINK_BLOCK'),'PLAIN_PARSE')
    need(split_output('<think>x</think>{"a":1}')[0]=='x','THINK_PARSE')
    for s in ('<think>x','x</think>{}', '<think>x</think>', '<think>x</think><think>y</think>{}', '<tool_call>x'):
        try: split_output(s)
        except ValueError: continue
        raise ValueError('INVALID_BOUNDARY_ACCEPTED:'+s)
    print('CPU boundary checks passed',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run','analyze','cpu-check'));a=p.parse_args()
    {'prepare':prepare,'run':run,'analyze':analyze,'cpu-check':cpu_check}[a.action]()
