"""Original-base replay of the two complete, already trained Day4 pools.

Preparation may inspect training answers to bind the exact existing tasks.
The separate live process opens only public requests, hashes and old base raws.
"""
from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
REPO = WEEK.parents[1]
DAY3 = WEEK / 'day3_solver187_v1/runtime/direct_v2'
OLD = REPO / 'offline-deliveries/h4_codex_train_reeval_v137/evaluation'
PANEL = WEEK / 'day5_base_panel_eval_v1'
SCHEDULE = WEEK / 'day4_schedule_v1'
MODEL = Path('/root/autodl-tmp/Qwen3-4B')
sys.path.insert(0, str(WEEK / 'day5_training_v1/train_diagnosis_prepare'))
import build_prepare as b

IDENTITY = ('request_id', 'candidate_id', 'root_id', 'question_type', 'variant_id', 'slot', 'seed', 'prompt_sha256')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def now():
    return b.datetime.now(b.timezone.utc).isoformat()


def check(bindings):
    for entry in bindings:
        path = Path(entry['path'])
        require(path.is_file() and path.stat().st_size == entry['bytes'] and b.sha(path) == entry['sha256'],
                'FROZEN_FILE_CHANGED:' + str(path))


def versions():
    return {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'tokenizers', 'vllm')}


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def adapter():
    return b.import_file('full_train_eval_adapter', WEEK / 'day2/solver/adapter.py')


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def choices_from_repr(value):
    tree = ast.parse(value, mode='eval').body
    for keyword in tree.keywords:
        if keyword.arg in ('structured_outputs', 'guided_decoding'):
            return ast.literal_eval(next(k.value for k in keyword.value.keywords if k.arg == 'choice'))
    raise RuntimeError('NO_FINITE_CHOICE_ARRAY_IN_OLD_SUBMISSION')


def prepare():
    require(not (HERE / 'PREPARATION_FREEZE.json').exists(), 'NO_OVERWRITE_PREPARATION')
    previous = b.read(PANEL / 'EXECUTION_MANIFEST.json')
    require(previous['python_prefix'] == sys.prefix and previous['runtime_versions'] == versions(), 'INHERIT_EVAL_ENVIRONMENT')
    check(previous['model_bindings'] + previous['tokenizer_bindings'])
    schedule_summary = b.read(SCHEDULE / 'SCHEDULE_SUMMARY.json')
    check(list(schedule_summary['train_pools'].values()))
    pools = {arm: lines(SCHEDULE / ('TRAIN_POOL_' + arm + '.jsonl')) for arm in ('A', 'B')}
    arm_ids = {arm: sorted({r['candidate_id'] for r in rows}) for arm, rows in pools.items()}
    selected = b.read(SCHEDULE / 'SELECTION.json')
    common = set(selected['selected_foundation'])
    new_ids = set(selected['selected_new_A'] + selected['selected_new_B'])
    all_ids = set(arm_ids['A']) | set(arm_ids['B'])
    require(len(common) == 24 and len(new_ids) == 176 and len(all_ids) == 200, 'POOL_SIZE_DRIFT')
    require(set(arm_ids['A']) & set(arm_ids['B']) == common, 'SHARED_POOL_DRIFT')
    for arm in ('A', 'B'):
        require(len(pools[arm]) == 224 and len(arm_ids[arm]) == 112, 'TWO_EPOCHS_ARE_NOT_NEW_QUESTIONS')
        require(set(arm_ids[arm]) == set(selected['selected_new_' + arm]) | common, 'ACTUAL_TRAIN_IDS_DRIFT')
        require(all(sorted(r['epoch'] for r in pools[arm] if r['candidate_id'] == cid) == [1, 2]
                    for cid in arm_ids[arm]), 'TRAIN_EPOCH_MEMBERSHIP_DRIFT')
    a = adapter()
    _, solver = a.native()
    from scripts.current_method.restart_n5_spell import vllm_backends, structured_qa_box_protocol
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    decoder = solver.generation_config('multi_select')
    require(decoder == b.read(DAY3 / 'manifest.json')['decoder_contract'] ==
            b.read(OLD / 'SHARED_REQUESTS.json')['decoder_contract'], 'COHORT_DECODER_DRIFT')
    old_tools = b.read(OLD / 'TOOLS_FREEZE.json')
    old_tool_bindings = [x for x in old_tools['files'] if x['path'] in
                         (str(OLD / 'batched_live.py'), str(Path(solver.__file__)),
                          str(Path(vllm_backends.__file__)), str(Path(structured_qa_box_protocol.__file__)))]
    require(len(old_tool_bindings) == 4, 'V137_CHOICE_CODE_BINDING_MISSING')
    check(old_tool_bindings)
    qnew = {q['candidate_id']: q for q in b.read(DAY3 / 'PUBLIC_QUESTIONS.json')['questions']}
    goldnew = {q['candidate_id']: q for q in b.read(DAY3.parent / 'direct_v2_private/PRIVATE_QUESTIONS.json')['questions']}
    qold_private = {q['candidate_id']: q for q in b.read(OLD / 'EVALUATION_POOL.json')['questions']}
    allowed = a.PUBLIC_KEYS
    questions = [qnew[cid] for cid in sorted(new_ids)] + [
        {k: v for k, v in qold_private[cid].items() if k in allowed} for cid in sorted(common)]
    by_id = {q['candidate_id']: q for q in questions}
    gold = [{**q, 'gold': (goldnew if q['candidate_id'] in new_ids else qold_private)[q['candidate_id']]['gold']}
            for q in questions]
    gold_by_id = {q['candidate_id']: q['gold'] for q in gold}
    for q in questions:
        a.public_question(q)
        require(b.sha(q['source_path']) == q['source_sha256'], 'SOURCE_HASH_DRIFT')
    oldbase = b.read(WEEK / 'day3_mainline_readiness_v1/training/COMMON_OLD_BASE_SET.json')
    check([oldbase['foundation']['candidate_data_binding'], oldbase['foundation']['lineage_binding']])
    lineage = {r['candidate_id']: r for r in lines(oldbase['foundation']['lineage_binding']['path']) if r['candidate_id'] in common}
    production = {}
    production_bindings = []
    for directory in ('day3_production_v1', 'day3_production_batch2_v1'):
        path = WEEK / directory / 'exports/content_accepted.private.jsonl'
        production_bindings.append(b.binding(path))
        for row in lines(path):
            if row['candidate_id'] in new_ids:
                ref = b.binding(Path(row['public']['path']))
                require(ref['sha256'] == row['public']['sha256'], 'PRODUCTION_PUBLIC_CHANGED')
                production_bindings.append(ref)
                production[row['candidate_id']] = b.read(ref['path'])['options_or_items']
    require(set(production) == new_ids, 'PRODUCTION_INPUTS_MISSING')
    displays = b.read(SCHEDULE / 'DISPLAY_MAP_CHECKS.json')['records']
    dmap = {(r['candidate_id'], r['epoch'], r['arm']): r for r in displays}
    for arm, pool in pools.items():
        for r in pool:
            cid = r['candidate_id']
            q = by_id[cid]
            field = 'options' if q['question_type'] == 'multi_select' else 'items'
            original = lineage[cid]['display_variant'][field] if cid in common else production[cid]
            mapping = dmap[(cid, r['epoch'], 'COMMON' if cid in common else arm)]
            require(set(original.values()) == set(q[field].values()), 'TRAIN_OPTION_SET_DRIFT:' + cid)
            # Production and evaluation canonical labels can differ. Compare
            # answer content after undoing the epoch's display permutation.
            actual_gold = [original[mapping['display_to_original'][k]] for k in mapping['mapped_gold']]
            canonical_gold = [q[field][k] for k in gold_by_id[cid]]
            require((actual_gold == canonical_gold if field == 'items' else set(actual_gold) == set(canonical_gold)), 'TRAIN_GOLD_CONTENT_DRIFT:' + cid)
            require(r['messages'][-1]['content'] == '<box>' + ','.join(mapping['mapped_gold']) + '</box>', 'TRAIN_TARGET_DRIFT')
            block = '\n'.join(f'{k}. {original[v]}' for k, v in mapping['display_to_original'].items())
            require(block in r['messages'][1]['content'], 'TRAIN_OPTION_CONTENT_DRIFT')
    maps3 = b.read(DAY3 / 'DISPLAY_MAPPINGS.json')
    manifest3 = b.read(DAY3 / 'manifest.json')
    rows = [{**r, 'prompt_path': str(DAY3 / 'prompts' / (r['request_id'] + '.txt')), 'cohort': 'day3'}
            for r in manifest3['requests'] if r['candidate_id'] in new_ids]
    maps = {'requests': {r['request_id']: maps3['requests'][r['request_id']] for r in rows},
            'public_to_canonical': {cid: maps3['public_to_canonical'][cid] for cid in sorted(new_ids)}}
    old_run = OLD / 'run/clean'
    legacy = b.read(old_run / 'manifest.json')
    require(legacy['model'] == manifest3['model'], 'S0_COHORT_MODEL_DRIFT')
    for old in legacy['requests']:
        if old['candidate_id'] not in common:
            continue
        field = 'options' if old['question_type'] == 'multi_select' else 'items'
        row = {**{k: v for k, v in old.items() if k not in ('variant', 'mode')},
               'variant_id': old['variant']['variant_id'], 'display': {field: old['variant'][field]},
               'branch': 'direct', 'prompt_path': str(old_run / 'prompts' / (old['request_id'] + '.txt')),
               'cohort': 'foundation_v137', 'max_output_tokens': 32}
        rows.append(row)
        maps['requests'][row['request_id']] = old['variant']
    maps['public_to_canonical'].update({cid: a.content_ids(by_id[cid]) for cid in sorted(common)})
    require(len(rows) == len({r['request_id'] for r in rows}) == 2910, 'FIXED_REQUEST_COUNT_DRIFT')
    for q in questions:
        require(sum(r['candidate_id'] == q['candidate_id'] for r in rows) ==
                (18 if q['question_type'] == 'multi_select' else 12), 'QUESTION_BUDGET_DRIFT')
    finite = {'choices': {}, 'request_keys': {}, 'provenance': {
        'day3': 'EXTRACTED_ACTUAL_SUBMITTED_ARRAY',
        'foundation_v137': 'CODE_RECONSTRUCTED_FROM_HASH_MATCHED_V137_EXECUTION_CODE'}}
    wanted = {r['request_id'] for r in rows if r['cohort'] == 'day3'}
    choice_source_bindings = []
    for path in sorted((DAY3 / 'batches').glob('*_SUBMITTED.json')):
        batch = b.read(path)
        if not wanted.intersection(batch['request_ids']):
            continue
        choice_source_bindings.append(b.binding(path))
        for rid, value in zip(batch['request_ids'], batch['sampling_params']):
            if rid not in wanted:
                continue
            choices = choices_from_repr(value)
            key = hashlib.sha256(json.dumps(choices).encode()).hexdigest()
            finite['choices'][key] = choices
            finite['request_keys'][rid] = key
    for r in rows:
        field = 'options' if r['question_type'] == 'multi_select' else 'items'
        native_choices = a.choices(r['question_type'], r['display'][field])
        if r['cohort'] == 'foundation_v137':
            key = hashlib.sha256(json.dumps(native_choices).encode()).hexdigest()
            finite['choices'][key] = native_choices
            finite['request_keys'][r['request_id']] = key
        require(set(finite['choices'][finite['request_keys'][r['request_id']]]) == set(native_choices), 'ANSWER_SPACE_DRIFT')
    s0_sources, token_hashes, prompt_bindings, token_cache = {}, {}, {}, {}
    legacy_complete = b.read(old_run / 'COMPLETE.json')
    for index, r in enumerate(rows):
        rid, path = r['request_id'], Path(r['prompt_path'])
        require(b.sha(path) == r['prompt_sha256'], 'PROMPT_HASH_DRIFT')
        if str(path) not in token_cache:
            token_cache[str(path)] = tokenizer.encode(path.read_text(), add_special_tokens=False)
        ids = token_cache[str(path)]
        require(len(ids) == r['prompt_tokens'] and len(ids) + 32 <= 16384, 'ZERO_TRUNCATION_FAILED')
        reference = (DAY3 if r['cohort'] == 'day3' else old_run) / 'responses' / (rid + '.json')
        raw = b.read(reference)
        keys = IDENTITY if r['cohort'] == 'day3' else IDENTITY[:-1]
        require(all(raw[k] == r[k] for k in keys), 'S0_IDENTITY_DRIFT')
        if r['cohort'] == 'day3':
            require(raw['prompt_token_ids'] == ids, 'EXACT_S0_TOKEN_ID_DRIFT')
        else:
            require(raw['metadata']['prompt_token_count'] == len(ids), 'LEGACY_PROMPT_TOKEN_COUNT_DRIFT')
        s0_sources[rid] = {**b.binding(reference), 'cohort': r['cohort']}
        token_hashes[rid] = token_hash(ids)
        prompt_bindings[str(path)] = b.binding(path)
        if (index + 1) % 500 == 0:
            print(json.dumps({'status': 'CPU_PREPARING', 'checked_requests': index + 1}), flush=True)
    reused = {}
    panel_freeze = b.read(PANEL / 'direct/RAW_RESPONSE_FREEZE.json')
    check(panel_freeze['files'] + [panel_freeze['execution_manifest'], panel_freeze['request_freeze']])
    row_lookup = {r['request_id']: r for r in rows}
    for entry in panel_freeze['files']:
        raw = b.read(entry['path'])
        require(raw['request_id'] in row_lookup, 'PREVIOUS_BASE_PANEL_NOT_IN_ACTUAL_TRAIN_POOLS')
        r = row_lookup[raw['request_id']]
        require(all(raw[k] == r[k] for k in IDENTITY) and raw['model_role'] == 'BASE' and
                raw['model_directory'] == str(MODEL), 'REUSE_BASE_IDENTITY_DRIFT')
        require(token_hash(raw['prompt_token_ids']) == token_hashes[r['request_id']], 'REUSE_TOKEN_DRIFT')
        require(raw['status'] == 'RETURNED' and not raw['output_validation_errors'], 'INVALID_BASE_REUSE')
        reused[r['request_id']] = entry
    require(len(reused) == 360, 'REUSED_PANEL_SIZE_DRIFT')
    members = [{'candidate_id': q['candidate_id'], 'root_id': q['root_id'], 'question_type': q['question_type'],
                'category': 'foundation' if q['candidate_id'] in common else 'new_structure',
                'arms': [arm for arm in ('A', 'B') if q['candidate_id'] in arm_ids[arm]]} for q in questions]
    b.save(HERE / 'public/PUBLIC_QUESTIONS.json', {'questions': questions})
    b.save(HERE / 'public/DIRECT_REQUESTS.json', {'requests': rows})
    b.save(HERE / 'public/DISPLAY_MAPPINGS.json', maps)
    b.save(HERE / 'public/FINITE_CHOICES.json', finite)
    b.save(HERE / 'private/GOLD.json', {'questions': gold})
    b.save(HERE / 'POOL_MEMBERSHIP.json', {'questions': members, 'arms': arm_ids,
           'unique_union_questions': 200, 'shared_foundation_questions': 24, 'epochs_per_question': 2,
           'independent_question_count_uses_candidate_id': True})
    b.save(HERE / 'S0_REFERENCE_BINDINGS.json', {'requests': s0_sources,
           'legacy_terminal_at_utc': legacy_complete['completed_at_utc'],
           'legacy_token_ids_unavailable': True, 'legacy_schema_conversion_only_after_base_raw_freeze': True})
    b.save(HERE / 'REUSED_BASE_BINDINGS.json', {'requests': reused, 'raw_freeze': b.binding(PANEL / 'direct/RAW_RESPONSE_FREEZE.json')})
    b.save(HERE / 'CPU_VALIDATION.json', {'status': 'PASS', 'created_at_utc': now(), 'questions': 200,
           'requests': 2910, 'new_requests': 2550, 'reused_requests': 360,
           'input_tokens_all': sum(r['prompt_tokens'] for r in rows),
           'input_tokens_new': sum(r['prompt_tokens'] for r in rows if r['request_id'] not in reused),
           'maximum_prompt_tokens': max(r['prompt_tokens'] for r in rows),
           'prompt_token_ids_sha256': token_hashes, 'zero_truncation': True,
           'day3_full_prompt_token_ids_equal_S0': True,
           'foundation_old_full_token_ids_available': False,
           'foundation_prompt_bytes_and_old_token_counts_match': True,
           'preparation_authoring_reads_gold': True, 'new_solver_calls': 0})
    code_paths = [Path(__file__), HERE / 'score_and_report.py',
                  WEEK / 'day2/solver/adapter.py', WEEK / 'day3_solver187_v1/runtime/score.py',
                  WEEK / 'day3_solver187_v1/runtime/common.py', Path(b.__file__),
                  Path(solver.__file__), Path(vllm_backends.__file__), Path(structured_qa_box_protocol.__file__),
                  a.NATIVE / 'scoring.py', OLD / 'batched_live.py']
    input_paths = [SCHEDULE / name for name in ('TRAIN_POOL_A.jsonl', 'TRAIN_POOL_B.jsonl', 'SELECTION.json',
                   'SCHEDULE_SUMMARY.json', 'DISPLAY_MAP_CHECKS.json', 'FINAL_MANIFEST.json')]
    input_paths += [DAY3 / name for name in ('manifest.json', 'PUBLIC_QUESTIONS.json', 'DISPLAY_MAPPINGS.json', 'RAW_RESPONSE_FREEZE.json')]
    input_paths += [OLD / 'TOOLS_FREEZE.json', OLD / 'PREFLIGHT_FREEZE.json', OLD / 'SHARED_REQUESTS.json',
                    OLD / 'EVALUATION_POOL.json', old_run / 'manifest.json', old_run / 'COMPLETE.json',
                    old_run / 'solver_identity.json', PANEL / 'FINAL_MANIFEST.json', WEEK / 'day1/MODEL_BINDINGS.json']
    local_paths = sorted(p for p in HERE.rglob('*.json') if p.is_file())
    freeze = {'status': 'FROZEN_BEFORE_NEW_BASE_INFERENCE', 'created_at_utc': now(),
              'model_role': 'BASE', 'model_directory': str(MODEL), 'tokenizer_directory': str(MODEL),
              'model_bindings': previous['model_bindings'], 'tokenizer_bindings': previous['tokenizer_bindings'],
              'code_bindings': [b.binding(p) for p in code_paths], 'input_bindings': [b.binding(p) for p in input_paths + local_paths] + production_bindings,
              'prompt_bindings': list(prompt_bindings.values()), 'choice_source_bindings': choice_source_bindings,
              's0_reference_bindings': list(s0_sources.values()), 'reused_base_bindings': list(reused.values()),
              'source_bindings': list({q['source_path']: b.binding(Path(q['source_path'])) for q in questions}.values()),
              'gold_binding': b.binding(HERE / 'private/GOLD.json'),
              'python_prefix': sys.prefix, 'runtime_versions': versions(), 'decoder_contract': decoder,
              'questions': 200, 'arm_questions': {'A': 112, 'B': 112}, 'requests': 2910,
              'new_requests': 2550, 'reused_requests': 360, 'batch_size': 8, 'max_model_len': 16384,
              'prefix_caching': False, 'automatic_retry': False, 'source_truncation': False,
              'new_DEV_TEST_calls': 0, 'new_R1_R2_calls': 0, 'training_runs': 0,
              'cohorts': {'day3': '176 new questions; original prompts/display/seed and submitted choice arrays.',
                          'foundation_v137': '24 shared foundation; original prompts/display/seed; choices reconstructed from hash-matched frozen code.'},
              'numerical_equivalence_to_old_batch_compositions_not_claimed': True,
              'scope': 'Complete actual training questions, not heldout generalization or proof of feedback causality.',
              'old_frozen_records_are_read_only': True}
    b.save(HERE / 'PREPARATION_FREEZE.json', freeze)
    b.save(HERE / 'EXECUTION_MANIFEST.json', {'status': 'FULL_TRAIN_BASE_BOUND_REVIEW_REQUIRED',
           'created_at_utc': now(), 'model_role': 'BASE', 'model_directory': str(MODEL),
           'tokenizer_directory': str(MODEL), 'original_base_no_adapter': True,
           'preparation_freeze': b.binding(HERE / 'PREPARATION_FREEZE.json'),
           'reused_request_ids': list(reused), 'new_request_ids': [r['request_id'] for r in rows if r['request_id'] not in reused],
           'requests': 2910, 'questions': 200, 'python_prefix': sys.prefix, 'runtime_versions': versions()})
    print(json.dumps({'status': 'PREPARED', 'questions': 200, 'requests': 2910, 'new_requests': 2550, 'reused_requests': 360}), flush=True)


def execution_verify(audit_path):
    manifest = b.read(HERE / 'EXECUTION_MANIFEST.json')
    check([manifest['preparation_freeze']])
    freeze = b.read(manifest['preparation_freeze']['path'])
    require(manifest['model_role'] == 'BASE' and manifest['model_directory'] == str(MODEL) and
            manifest['original_base_no_adapter'] is True, 'NOT_ORIGINAL_BASE')
    require(freeze['python_prefix'] == sys.prefix and freeze['runtime_versions'] == versions(), 'RUNTIME_DRIFT')
    for key in ('model_bindings', 'tokenizer_bindings', 'code_bindings', 'input_bindings', 'prompt_bindings',
                'choice_source_bindings', 's0_reference_bindings', 'reused_base_bindings', 'source_bindings'):
        check(freeze[key])
    audit = b.read(audit_path)
    require(audit['status'] == 'PASS' and any(r['path'] == str(HERE / 'EXECUTION_MANIFEST.json') and
            r['sha256'] == b.sha(HERE / 'EXECUTION_MANIFEST.json') for r in audit['audited_files']), 'INDEPENDENT_AUDIT_NOT_BOUND')
    check(audit['audited_files'])
    return manifest, freeze


def live(audit_path):
    manifest, freeze = execution_verify(audit_path)
    run = HERE / 'direct'
    require(not run.exists(), 'NO_OVERWRITE_OR_AUTOMATIC_RETRY')
    all_rows = b.read(HERE / 'public/DIRECT_REQUESTS.json')['requests']
    reused = b.read(HERE / 'REUSED_BASE_BINDINGS.json')['requests']
    rows = [r for r in all_rows if r['request_id'] not in reused]
    tokens = b.read(HERE / 'CPU_VALIDATION.json')['prompt_token_ids_sha256']
    finite = b.read(HERE / 'public/FINITE_CHOICES.json')
    require(len(rows) == 2550 and len(reused) == 360 and len(all_rows) == 2910, 'REQUEST_SCOPE_DRIFT')
    b.save(run / 'STARTED.json', {'started_at_utc': now(), 'model_role': 'BASE', 'planned_new_requests': 2550,
           'reused_requests': 360, 'request_file': b.binding(HERE / 'public/DIRECT_REQUESTS.json'),
           'execution_manifest': b.binding(HERE / 'EXECUTION_MANIFEST.json'), 'audit': b.binding(audit_path),
           'gold_opened_by_live': False})
    for rid, entry in reused.items():
        target = run / 'responses' / (rid + '.json')
        target.parent.mkdir(parents=True, exist_ok=True)
        require(not target.exists(), 'REUSE_TARGET_EXISTS')
        shutil.copyfile(entry['path'], target)
        require(b.sha(target) == entry['sha256'], 'REUSE_COPY_DRIFT')
    backend, submitted, done, failed = None, set(), set(), None
    started, load_seconds = time.perf_counter(), None
    try:
        a = adapter()
        _, solver = a.native()
        solver.ensure_vllm_environment()
        from scripts.current_method.restart_n5_spell.vllm_backends import VLLMSolverBackend
        from vllm import SamplingParams, sampling_params
        backend = VLLMSolverBackend(model_path=MODEL, tokenizer_path=MODEL, max_model_len=16384,
                    temperature=.8, top_p=.9, max_tokens=32, gpu_memory_utilization=.8)
        load_seconds = backend.load_latency_seconds
        require(str(backend.model_identifier) == str(MODEL), 'BACKEND_IDENTITY_DRIFT')
        for offset in range(0, len(rows), 8):
            batch = rows[offset:offset + 8]
            prompts, params = [], []
            for row in batch:
                path = Path(row['prompt_path'])
                require(b.sha(path) == row['prompt_sha256'], 'PROMPT_DRIFT')
                prompts.append(path.read_text())
                kw = {'temperature': .8, 'top_p': .9, 'max_tokens': 32, 'seed': row['seed'],
                      'stop': ['</box>'], 'include_stop_str_in_output': True}
                choices = finite['choices'][finite['request_keys'][row['request_id']]]
                kw['structured_outputs'] = sampling_params.StructuredOutputsParams(choice=choices)
                params.append(SamplingParams(**kw))
            bid = f'batch{offset // 8:04d}'
            b.save(run / 'batches' / (bid + '_SUBMITTED.json'), {'request_ids': [r['request_id'] for r in batch],
                   'submitted_at_utc': now(), 'automatic_retry': False,
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
                if out.prompt != prompts[i] or token_hash(out.prompt_token_ids) != tokens[row['request_id']]:
                    errors.append('OUTPUT_EXACT_PROMPT_TOKEN_DRIFT')
                if len(completions) != 1:
                    errors.append('COMPLETION_COUNT_MISMATCH')
                b.save(run / 'responses' / (row['request_id'] + '.json'), {
                    **{k: row[k] for k in IDENTITY}, 'status': 'RETURNED', 'raw_output': first.text if first else None,
                    'response_committed_at_utc': now(), 'model_role': 'BASE', 'model_directory': str(MODEL),
                    'branch': 'direct', 'cohort': row['cohort'], 'output_validation_errors': errors,
                    'prompt_token_ids': out.prompt_token_ids,
                    'all_completions': [{'text': c.text, 'token_ids': c.token_ids, 'finish_reason': c.finish_reason,
                                         'stop_reason': c.stop_reason} for c in completions],
                    'metadata': {'finish_reason': first.finish_reason if first else None,
                        'stop_reason': first.stop_reason if first else None, 'prompt_token_count': len(out.prompt_token_ids),
                        'completion_token_count': len(first.token_ids) if first else None,
                        'temperature': .8, 'top_p': .9, 'max_tokens': 32, 'enable_thinking': False, 'seed': row['seed'],
                        'structured_output_api': freeze['decoder_contract']['structured_output_api']},
                    'gold_opened_by_solver': False})
                done.add(row['request_id'])
            if len(outputs) > len(batch):
                b.save(run / 'batches' / (bid + '_EXTRA_OUTPUTS.json'), {'outputs': [
                    {'request_id': o.request_id, 'prompt': o.prompt, 'completions': [{'text': c.text, 'token_ids': c.token_ids}
                     for c in o.outputs]} for o in outputs[len(batch):]]})
            require(len(outputs) == len(batch), 'OUTPUT_COUNT_MISMATCH')
            require(all(not b.read(run / 'responses' / (r['request_id'] + '.json'))['output_validation_errors'] for r in batch), 'RAW_OUTPUT_VALIDATION_FAILED')
            b.save(run / 'batches' / (bid + '_COMPLETE.json'), {'completed_at_utc': now(), 'request_ids': [r['request_id'] for r in batch]})
            print(json.dumps({'status': 'LIVE', 'model': 'BASE', 'completed_new': len(done), 'planned_new': 2550,
                              'reused': 360, 'completed_total': len(done) + 360, 'planned_total': 2910}), flush=True)
    except BaseException as error:
        failed = {'error': str(error), 'exception_type': type(error).__name__, 'traceback': traceback.format_exc()}
    finally:
        if backend:
            try:
                backend.close()
            except BaseException as error:
                failed = failed or {'error': str(error), 'exception_type': type(error).__name__, 'traceback': traceback.format_exc()}
        for row in rows:
            if row['request_id'] not in done:
                b.save(run / 'responses' / (row['request_id'] + '.json'), {
                    **{k: row[k] for k in IDENTITY}, 'status': 'BACKEND_ERROR' if row['request_id'] in submitted else 'NOT_SUBMITTED',
                    'raw_output': None, 'response_committed_at_utc': now(), 'usage': None,
                    'usage_status': 'UNKNOWN_NOT_ZERO' if row['request_id'] in submitted else 'NO_REQUEST_SENT', 'failure': failed})
        b.save(run / ('FAILED.json' if failed else 'COMPLETE.json'), {'status': 'FAILED' if failed else 'COMPLETE',
               'completed_at_utc': now(), 'planned_request_count': 2910, 'actual_submitted_requests': len(submitted),
               'returned_requests': len(done), 'reused_requests': 360, 'total_returned_requests': len(done) + 360,
               'model_load_wall_seconds': load_seconds, 'wall_seconds': time.perf_counter() - started, 'failure': failed})
        paths = [run / 'responses' / (r['request_id'] + '.json') for r in all_rows]
        require(set(paths) == set((run / 'responses').glob('*.json')), 'FIXED_RAW_DENOMINATOR_DRIFT')
        b.save(run / 'RAW_RESPONSE_FREEZE.json', {'status': 'COMMITTED', 'created_at_utc': now(),
               'planned_request_count': 2910, 'files': [b.binding(p) for p in paths],
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
