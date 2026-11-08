#!/usr/bin/env python3
"""Exploratory paired rendering and prototype difficulty pilot; frozen B1.

Preparation and scoring may read private answers. Generation reads only public
requests, prompts, schemas, runtime configuration and public hashes. Old artifacts
remain read-only. No training, truncation, automatic retry or result-based edits.
"""
from __future__ import annotations

import argparse
import collections
import fcntl
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
OLD = WEEK / 'feedback_closed_loop_20261024_v1'
DESIGN = WEEK / 'round2_data_design_20261026_v1'
OUT = HERE / 'execution'
MAX_TOKENS, MODEL_LEN, BATCH = 6144, 16384, 8
ROOTS = ('218519', '222937', '225575', '229686')
FORMATS = ('ORIGINAL', 'SHORT')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


engine_module = load('format_pilot_frozen_engine', WEEK / 'thinking_switch_20261024_structured_v2/run.py')
orig = engine_module.orig


def need(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    p = Path(path).resolve()
    data = p.read_bytes()
    return {'path': str(p), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def check(b):
    need(binding(b['path']) == b, 'BINDING_DRIFT:' + b['path'])


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')


def textfile(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        f.write(text)


def original_messages(prompt):
    pattern = r'<\|im_start\|>system\n(.*?)<\|im_end\|>\n<\|im_start\|>user\n(.*?)<\|im_end\|>\n<\|im_start\|>assistant\n<think>\n\n</think>\n\n'
    match = re.fullmatch(pattern, prompt, re.S)
    need(match, 'UNEXPECTED_FROZEN_TEMPLATE')
    return [{'role': 'system', 'content': match[1]}, {'role': 'user', 'content': match[2]}]


def prototype_messages(candidate, dialogue, display):
    view = candidate['solver_view']
    common = orig.diagnostic.COMMON.replace(
        'Return only JSON, with\nitem_evidence first, answer_reason second and final_answer last. Provide brief\nevidence-based explanations, not extended private deliberation.',
        'Your final response must be JSON, with item_evidence first, answer_reason second and final_answer last. Keep the explanations in the final JSON brief and evidence-based.')
    user = (f"Dialogue:\n{dialogue}\n\nQuestion (multi_select):\n{view['question']}\n\nDisplayed items:\n"
            + '\n'.join(f'{label}. {text}' for label, text in display.items())
            + f"\n\nRequested operation: {candidate['operation']}\n{view['response_instruction']}\n\n{common}")
    return [{'role': 'system', 'content': 'Use only the supplied dialogue. Follow the question\'s requested operation. In the final response, give concise, verifiable explanations before the final boxed answer.'},
            {'role': 'user', 'content': user}]


def prepare():
    need(not OUT.exists(), 'NO_PREPARATION_OVERWRITE')
    converter = load('format_pilot_renderer', HERE / 'render.py')
    endpoint, config = orig.endpoint_receipt()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(orig.TOKENIZER), local_files_only=True)
    from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
    reasoner = Qwen3ReasoningParser(tokenizer=tok)
    api = engine_module.cpu_api_check()
    legacy_public = read(OLD / 'execution/PUBLIC_REQUESTS.json')['requests']
    legacy_private = {r['request_id']: r for r in read(OLD / 'execution/PRIVATE_SCORING.json')['requests']}
    # Selection is a complete predeclared slot/root cross, without reading scores.
    selected = [r for r in legacy_public if r['root_id'].split('-')[-1] in ROOTS
                and r['arm'] == 'F' and r['slot_id'] in ('MC2', 'ORDER1')]
    need(len(selected) == 16 and len({r['candidate_id'] for r in selected}) == 8, 'LEGACY_SELECTION_INCOMPLETE')
    prototypes = read(DESIGN / 'question_design/TASK_ORIENTED_MC_PROTOTYPES.json')
    need([c['candidate_id'] for c in prototypes['candidates']] == [f'MC-PROT-{i:02}' for i in range(1, 5)], 'PROTOTYPE_SET_DRIFT')
    sources, rendered = {}, {}
    for r in selected:
        sources[r['root_id']] = r['source_binding']
    for root, b in sources.items():
        check(b)
        source = read(b['path'])
        need(source['root_id'] == root, 'SOURCE_ROOT_MISMATCH')
        rendered[root] = converter.render_source(source)
    staged = []
    for row in selected:
        root = row['root_id']
        prompt = Path(row['prompt_path']).read_text()
        messages = original_messages(prompt)
        need(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False) == prompt, 'FROZEN_PROMPT_DRIFT')
        need('Dialogue:\n' + rendered[root]['original_text'] + '\n\nQuestion' in messages[1]['content'], 'ORIGINAL_DIALOGUE_DRIFT')
        presentation = legacy_private[row['request_id']]['presentation']
        staged.append({'cohort': 'legacy_format', 'candidate_id': row['candidate_id'], 'root_id': root,
                       'question_type': row['question_type'], 'task_operation': row['task_operation'],
                       'question': row['question'], 'variant_id': row['variant_id'], 'display': row['display'],
                       'sampling_seed': row['sampling_seed'], 'messages': messages,
                       'display_to_canonical': presentation['display_to_canonical'],
                       'canonical_gold': presentation['canonical_gold'], 'display_gold': presentation['display_gold'],
                       'legacy_request_id': row['request_id']})
    prototype_binding = binding(DESIGN / 'question_design/TASK_ORIENTED_MC_PROTOTYPES.json')
    for candidate in prototypes['candidates']:
        root = candidate['source_binding']['root_id']
        need({k: candidate['source_binding'][k] for k in ('path', 'sha256', 'bytes')} == sources[root], 'PROTOTYPE_SOURCE_MISMATCH')
        labels = list(candidate['solver_view']['options'])
        seen = set()
        for perm in range(2):
            # Freeze nonidentity, distinct shuffles before model scoring.
            shuffle_seed = 2026102600 + perm
            while True:
                canonical = labels[:]
                rng = random.Random(candidate['candidate_id'] + ':' + str(shuffle_seed))
                rng.shuffle(canonical)
                if canonical != labels and tuple(canonical) not in seen:
                    break
                shuffle_seed += 2
            seen.add(tuple(canonical))
            dtc = dict(zip(labels, canonical))
            ctd = {v: k for k, v in dtc.items()}
            display = {label: candidate['solver_view']['options'][dtc[label]] for label in labels}
            seed = int.from_bytes(hashlib.sha256((candidate['candidate_id'] + ':solver').encode()).digest()[:4], 'big') % 2147483647
            staged.append({'cohort': 'prototype_difficulty', 'candidate_id': candidate['candidate_id'], 'root_id': root,
                           'question_type': 'multi_select', 'task_operation': candidate['operation'],
                           'question': candidate['solver_view']['question'], 'variant_id': f'perm_{perm:02}',
                           'display': display, 'sampling_seed': seed,
                           'messages': prototype_messages(candidate, rendered[root]['original_text'], display),
                           'display_to_canonical': dtc, 'canonical_gold': candidate['gold'],
                           'display_gold': sorted(ctd[g] for g in candidate['gold']), 'shuffle_seed': shuffle_seed})
    need(len(staged) == 24, 'BASE_REQUEST_COUNT')
    rows, tokens, scoring, public_bindings = [], {}, [], []
    source_stats = []
    for root, conversion in rendered.items():
        textfile(HERE / 'converted' / (root + '__ORIGINAL.txt'), conversion['original_text'])
        textfile(HERE / 'converted' / (root + '__SHORT.txt'), conversion['short_text'])
        save(HERE / 'private_maps' / (root + '.json'), conversion)
        (HERE / 'private_maps' / (root + '.json')).chmod(0o600)
        source_stats.append({'root_id': root, 'turns': len(conversion['refs']), 'scenes': conversion['scenes'],
                             'lossless_validation': conversion['lossless_validation'],
                             'original_dialogue_tokens': len(tok.encode(conversion['original_text'], add_special_tokens=False)),
                             'short_dialogue_tokens': len(tok.encode(conversion['short_text'], add_special_tokens=False)),
                             'source_binding': sources[root]})
    for index, base in enumerate(staged):
        conversion = rendered[base['root_id']]
        # Counterbalance order across pairs; all pairs retain identical seeds.
        format_order = FORMATS if index % 2 == 0 else tuple(reversed(FORMATS))
        for fmt in format_order:
            messages = [dict(m) for m in base['messages']]
            if fmt == 'SHORT':
                old_dialogue = 'Dialogue:\n' + conversion['original_text'] + '\n\nQuestion'
                need(messages[1]['content'].count(old_dialogue) == 1, 'NONUNIQUE_DIALOGUE_BLOCK')
                messages[1]['content'] = messages[1]['content'].replace(old_dialogue, 'Dialogue:\n' + conversion['short_text'] + '\n\nQuestion')
                messages[1]['content'] = messages[1]['content'].replace('turn_refs exactly as printed in the\ndialogue', 'turn_refs using the complete scene-id:Txxx citation format described above')
                refs = conversion['refs']
            else:
                refs = list(conversion['short_to_original'].values())
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok.encode(prompt, add_special_tokens=False)
            need(len(ids) + MAX_TOKENS <= MODEL_LEN, 'CONTEXT_OVERFLOW_NO_TRUNCATION:' + base['candidate_id'])
            need(reasoner.is_reasoning_end(ids), 'THINK_OFF_GATE_FAILURE')
            rid = f"{base['candidate_id']}__{base['variant_id']}__{fmt}"
            pp, sp = OUT / 'prompts' / (rid + '.txt'), OUT / 'schemas' / (rid + '.json')
            schema = orig.diagnostic.repair.schema(list(base['display']), refs, base['question_type'])
            textfile(pp, prompt)
            save(sp, schema)
            row = {k: base[k] for k in ('cohort', 'candidate_id', 'root_id', 'question_type', 'task_operation', 'question', 'variant_id', 'display', 'sampling_seed')}
            row.update(request_id=rid, format=fmt, condition='FULL', mode='THINK_OFF', labels=list(base['display']),
                       source_binding=sources[base['root_id']], prompt_path=str(pp), schema_path=str(sp),
                       prompt_tokens=len(ids), messages_sha256=hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest())
            rows.append(row)
            tokens[rid] = ids
            scoring.append({'request_id': rid, **{k: base[k] for k in ('display_to_canonical', 'canonical_gold', 'display_gold')},
                            'citation_to_original': conversion['short_to_original'] if fmt == 'SHORT' else {r: r for r in refs},
                            'legacy_request_id': base.get('legacy_request_id')})
            public_bindings.extend([binding(pp), binding(sp)])
    need(len(rows) == len({r['request_id'] for r in rows}) == 48, 'REQUEST_COUNT')
    save(OUT / 'PUBLIC_REQUESTS.json', {'requests': rows})
    save(OUT / 'PROMPT_TOKEN_IDS.json', tokens)
    save(OUT / 'PRIVATE_SCORING.json', {'requests': scoring, 'prototype_binding': prototype_binding})
    (OUT / 'PRIVATE_SCORING.json').chmod(0o600)
    protocol = {
        'experiment_id': HERE.name, 'exploratory': True, 'independent_validation': False,
        'selection_rule': 'All four original roots, arm F, slots MC2 and ORDER1; all four prototypes. No Solver scores read for selection.',
        'legacy_semantic_questions': 8, 'prototype_semantic_questions': 4, 'roots': list(ROOTS),
        'display_variants_per_question': 2, 'formats': list(FORMATS), 'requests': 48,
        'calendar_policy': 'No date inference or insertion. Explicit existing metadata only; missing dates are unknown.',
        'paired_controls': ['source text', 'speaker', 'turn and scene order', 'question', 'display permutation', 'gold', 'sampling seed', 'sampling settings', 'output budget'],
        'intervention': 'Display coordinates, scene headers, citation convention and corresponding schema refs. Original arm is the frozen legacy prompt for old questions.',
        'sampling': {'temperature': .8, 'top_p': .9, 'top_k': -1, 'repetition_penalty': 1., 'engine_seed': 0},
        'max_tokens': MAX_TOKENS, 'max_model_len': MODEL_LEN, 'batch_size': BATCH,
        'thinking': False, 'structured_outputs_config': engine_module.STRUCTURED_CONFIG,
        'prefix_caching': False, 'automatic_retries': 0, 'source_truncation': False, 'training': False,
        'primary_mc_partial': 'J = TP/g - FP/(n-g); negatives retained, no clipping. Report J*100 as score, not accuracy.',
        'invalid_policy': 'Strict exact = 0 with all frozen requests as denominator; original partial J is None/unscored for invalid responses. Report valid J denominator.',
        'limits': 'Dependent permutations, four previously exposed roots, one checkpoint; not an equivalence, training-effect or population-difficulty test.'}
    save(HERE / 'PROTOCOL_FREEZE.json', protocol)
    save(HERE / 'TOKEN_CONVERSION_AUDIT.json', {'sources': source_stats, 'source_mutation': False})
    save(OUT / 'CPU_API_CHECK.json', api)
    public_bindings.extend(binding(p) for p in [Path(__file__), HERE / 'render.py', HERE / 'PROTOCOL_FREEZE.json', OUT / 'PUBLIC_REQUESTS.json', OUT / 'PROMPT_TOKEN_IDS.json'])
    public_bindings.extend(sources.values())
    public_bindings.extend(binding(p) for p in orig.code_paths())
    public_bindings.append(binding(WEEK / 'thinking_switch_20261024_structured_v2/run.py'))
    save(OUT / 'PREPARATION.json', {'requests': len(rows), 'fixed_order': [r['request_id'] for r in rows],
                                   'bindings': list({b['path']: b for b in public_bindings}.values()),
                                   'private_scoring_binding': binding(OUT / 'PRIVATE_SCORING.json'),
                                   'runtime_versions': orig.versions(), 'config_receipt': config,
                                   'max_prompt_tokens': max(r['prompt_tokens'] for r in rows)})
    print(json.dumps({'prepared': len(rows), 'max_prompt_tokens': max(r['prompt_tokens'] for r in rows), 'max_output_tokens': MAX_TOKENS}), flush=True)


def public():
    prep = read(OUT / 'PREPARATION.json')
    forbidden = {str((OUT / 'PRIVATE_SCORING.json').resolve())}
    for b in prep['bindings']:
        need(b['path'] not in forbidden and '/private_maps/' not in b['path'], 'PRIVATE_BINDING_IN_GENERATION')
        check(b)
    rows = read(OUT / 'PUBLIC_REQUESTS.json')['requests']
    tokens = read(OUT / 'PROMPT_TOKEN_IDS.json')
    need(prep['fixed_order'] == [r['request_id'] for r in rows] and len(rows) == len(tokens) == 48, 'PUBLIC_SET_DRIFT')
    for row in rows:
        need(not ({'gold', 'canonical_gold', 'display_gold', 'display_to_canonical', 'citation_to_original'} & set(row)), 'PRIVATE_PUBLIC_FIELD')
        need(row['prompt_tokens'] == len(tokens[row['request_id']]) and row['prompt_tokens'] + MAX_TOKENS <= MODEL_LEN, 'TOKEN_BUDGET_DRIFT')
    return prep, rows, tokens


def run():
    need(not (OUT / 'STARTED.json').exists(), 'NO_RERUN_OR_OVERWRITE')
    prep, rows, tokens = public()
    endpoint, config = orig.endpoint_receipt()
    need(config == prep['config_receipt'] and orig.versions() == prep['runtime_versions'], 'RUNTIME_CONFIG_DRIFT')
    for b in endpoint['model_bindings'] + endpoint['tokenizer_bindings']:
        check(b)
    for k, v in {'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'VLLM_USE_FLASHINFER_SAMPLER': '0', 'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false'}.items():
        os.environ[k] = v
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(orig.TOKENIZER), local_files_only=True)
    orig.checked_prompt_tokens(tok, rows, tokens)
    from vllm import LLM, SamplingParams, sampling_params
    lock = open('/tmp/gxy-selfplay-gpu.lock', 'a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    save(OUT / 'STARTED.json', {'experiment_id': HERE.name, 'time_unix': time.time(), 'preparation_binding': binding(OUT / 'PREPARATION.json'), 'requests': len(rows), 'gold_read': False})
    started = time.perf_counter()
    submitted, failure = [], None
    try:
        engine = LLM(model=str(orig.MODEL), tokenizer=str(orig.TOKENIZER), tensor_parallel_size=1,
                     max_model_len=MODEL_LEN, gpu_memory_utilization=.8, dtype='bfloat16', trust_remote_code=True,
                     enable_prefix_caching=False, max_num_seqs=BATCH, seed=0,
                     hf_overrides={'rope_theta': orig.EXPECTED_THETA, 'rope_parameters': {'rope_type': 'default', 'rope_theta': orig.EXPECTED_THETA}},
                     structured_outputs_config=engine_module.STRUCTURED_CONFIG)
        cfg = engine.llm_engine.model_config
        need(Path(cfg.model).resolve() == orig.MODEL.resolve(), 'WRONG_MODEL')
        actual = engine.llm_engine.vllm_config.structured_outputs_config
        need(actual.reasoning_parser == 'qwen3' and not actual.enable_in_reasoning, 'STRUCTURED_CONFIG_DRIFT')
        save(OUT / 'MODEL_LOAD_RECEIPT.json', {'experiment_id': HERE.name, 'model_directory': str(orig.MODEL),
             'tokenizer_directory': str(orig.TOKENIZER), 'config': config, 'runtime_versions': orig.versions(),
             'actual_hf_config': orig.checked_runtime_config(cfg.hf_config),
             'actual_hf_text_config': orig.checked_runtime_config(cfg.hf_text_config),
             'structured_outputs_config': engine_module.STRUCTURED_CONFIG, 'runner_binding': binding(Path(__file__))})
        print('MODEL_LOADED', flush=True)
        for i in range(0, len(rows), BATCH):
            batch = rows[i:i + BATCH]
            params = [SamplingParams(temperature=.8, top_p=.9, top_k=-1, repetition_penalty=1., max_tokens=MAX_TOKENS,
                       seed=r['sampling_seed'], skip_special_tokens=False,
                       structured_outputs=sampling_params.StructuredOutputsParams(json=read(r['schema_path']))) for r in batch]
            for row in batch:
                save(OUT / 'submitted' / (row['request_id'] + '.json'), {'request_id': row['request_id'], 'time_unix': time.time(), 'gold_read': False})
                submitted.append(row['request_id'])
            t0 = time.perf_counter()
            results = engine.generate([Path(r['prompt_path']).read_text() for r in batch], params, use_tqdm=False)
            need(len(results) == len(batch), 'RESULT_COUNT_MISMATCH')
            for row, result in zip(batch, results):
                raw = orig.base.preserve_response(row, [result], endpoint)
                raw['capture_errors'] = orig.capture.response_errors(raw, Path(row['prompt_path']).read_text(), tokens[row['request_id']])
                raw['batch_index'] = i // BATCH
                raw['decoded_output_skip_special_tokens_false'] = tok.decode(raw.get('output_token_ids', []), skip_special_tokens=False)
                save(OUT / 'raw' / (row['request_id'] + '.json'), raw)
            save(OUT / 'batches' / (f'{i // BATCH:03}.json'), {'batch_index': i // BATCH, 'request_ids': [r['request_id'] for r in batch], 'wall_seconds': time.perf_counter() - t0})
            print(json.dumps({'completed': i + len(batch), 'total': len(rows), 'batch_seconds': round(time.perf_counter() - t0, 3),
                              'finishes': [x.outputs[0].finish_reason for x in results], 'output_tokens': [len(x.outputs[0].token_ids) for x in results]}), flush=True)
    except BaseException:
        failure = traceback.format_exc()
        print(failure, flush=True)
    for row in rows:
        p = OUT / 'raw' / (row['request_id'] + '.json')
        if not p.exists():
            save(p, {'request_id': row['request_id'], 'status': 'BACKEND_ERROR' if row['request_id'] in submitted else 'NOT_SUBMITTED', 'raw_output': None, 'failure': failure})
    save(OUT / 'RAW_RESPONSE_FREEZE.json', {'experiment_id': HERE.name, 'request_ids': [r['request_id'] for r in rows],
         'submitted_ids': submitted, 'returned': sum(read(OUT / 'raw' / (r['request_id'] + '.json'))['status'] == 'RETURNED' for r in rows),
         'wall_seconds': time.perf_counter() - started, 'failure': failure,
         'bindings': [binding(OUT / 'raw' / (r['request_id'] + '.json')) for r in rows]})
    if failure:
        raise RuntimeError('Pilot run failed; all attempted requests preserved, no retry. See raw freeze.')


def avg(values):
    return statistics.mean(values) if values else None


def summarize(records):
    return {'requests': len(records), 'valid': sum(r['valid'] for r in records),
            'exact_correct': sum(r['exact'] for r in records), 'exact_accuracy': avg([int(r['exact']) for r in records]),
            'J_score': avg([r['J'] * 100 for r in records if r['J'] is not None]),
            'J_scored_requests': sum(r['J'] is not None for r in records),
            'ordering_pairwise_accuracy': avg([r['pairwise_accuracy'] for r in records if r['question_type'] == 'ordering']),
            'mean_prompt_tokens': avg([r['prompt_tokens'] for r in records]), 'mean_output_tokens': avg([r['output_tokens'] for r in records]),
            'length_finishes': sum(r['finish_reason'] == 'length' for r in records)}


def analyze():
    prep, rows, tokens = public()
    freeze = read(OUT / 'RAW_RESPONSE_FREEZE.json')
    need(freeze['request_ids'] == prep['fixed_order'], 'RAW_FREEZE_SET_DRIFT')
    for b in freeze['bindings']:
        check(b)
    check(prep['private_scoring_binding'])
    private = {r['request_id']: r for r in read(OUT / 'PRIVATE_SCORING.json')['requests']}
    records = []
    for row in rows:
        raw = read(OUT / 'raw' / (row['request_id'] + '.json'))
        scoring = private[row['request_id']]
        valid, parsed, answer, errors = False, None, [], []
        try:
            need(raw['status'] == 'RETURNED' and raw.get('finish_reason') == 'stop', 'BACKEND_OR_LENGTH_FAILURE')
            need(not raw.get('capture_errors'), 'CAPTURE_ERRORS')
            thinking, final, _ = engine_module.split_output(raw['raw_output'])
            need(not thinking, 'UNEXPECTED_THINKING')
            parsed, answer = orig.base.validate_output(final, row, read(row['schema_path']))
            valid = True
        except Exception as e:
            errors.append(str(e))
        # Invalid answers receive no exact credit; original J remains unscored.
        canonical = [scoring['display_to_canonical'][a] for a in answer] if valid else []
        gold = scoring['canonical_gold']
        exact = valid and (set(canonical) == set(gold) if row['question_type'] == 'multi_select' else canonical == gold)
        j, pairwise = None, None
        tp = len(set(canonical) & set(gold))
        fp = len(set(canonical) - set(gold))
        if row['question_type'] == 'multi_select' and valid:
            g, n = len(gold), len(row['labels'])
            need(0 < g < n, 'J_REQUIRES_NONDEGENERATE_GOLD')
            j = tp / g - fp / (n - g)
        elif row['question_type'] == 'ordering' and valid:
            rank = {x: i for i, x in enumerate(canonical)}
            pairs = list(itertools.combinations(gold, 2))
            pairwise = sum(rank[a] < rank[b] for a, b in pairs) / len(pairs)
        elif row['question_type'] == 'ordering':
            pairwise = 0.
        citation_audit = []
        if parsed:
            for label, evidence in parsed['item_evidence'].items():
                citation_audit.append({'label': label, 'original_turn_refs': [scoring['citation_to_original'].get(ref) for ref in evidence['turn_refs']]})
        records.append({k: row[k] for k in ('request_id', 'candidate_id', 'cohort', 'root_id', 'question_type', 'task_operation', 'variant_id', 'format', 'prompt_tokens')} |
                       {'valid': valid, 'exact': exact, 'canonical_prediction': sorted(canonical) if row['question_type'] == 'multi_select' else canonical,
                        'canonical_gold': gold, 'TP': tp, 'FP': fp, 'FN': len(set(gold) - set(canonical)),
                        'J': j, 'pairwise_accuracy': pairwise, 'errors': errors, 'parsed': parsed,
                        'citations': citation_audit, 'finish_reason': raw.get('finish_reason'), 'output_tokens': len(raw.get('output_token_ids', []))})
    save(OUT / 'SCORES.json', {'records': records, 'raw_freeze_binding': binding(OUT / 'RAW_RESPONSE_FREEZE.json')})
    summaries = []
    for cohort in ('legacy_format', 'prototype_difficulty'):
        for fmt in FORMATS:
            summaries.append({'cohort': cohort, 'format': fmt, **summarize([r for r in records if r['cohort'] == cohort and r['format'] == fmt])})
    pair_groups = collections.defaultdict(dict)
    for r in records:
        pair_groups[(r['candidate_id'], r['variant_id'])][r['format']] = r
    paired = []
    for (cid, variant), arms in pair_groups.items():
        a, b = arms['ORIGINAL'], arms['SHORT']
        paired.append({'candidate_id': cid, 'variant_id': variant, 'cohort': a['cohort'], 'root_id': a['root_id'],
                       'original_exact': a['exact'], 'short_exact': b['exact'],
                       'same_canonical_prediction': a['valid'] and b['valid'] and a['canonical_prediction'] == b['canonical_prediction'],
                       'original_J': a['J'], 'short_J': b['J'], 'token_saving': a['prompt_tokens'] - b['prompt_tokens'],
                       'both_valid': a['valid'] and b['valid']})
    questions = []
    for cid in dict.fromkeys(r['candidate_id'] for r in records):
        rs = [r for r in records if r['candidate_id'] == cid]
        questions.append({'candidate_id': cid, 'cohort': rs[0]['cohort'], 'root_id': rs[0]['root_id'],
                          'gold_count': len(rs[0]['canonical_gold']),
                          'by_format': {fmt: summarize([r for r in rs if r['format'] == fmt]) for fmt in FORMATS},
                          'permutation_consistency': {fmt: len({tuple(r['canonical_prediction']) for r in rs if r['format'] == fmt and r['valid']}) == 1
                                                     and all(r['valid'] for r in rs if r['format'] == fmt) for fmt in FORMATS}})
    save(OUT / 'ANALYSIS_SUMMARY.json', {'summaries': summaries, 'paired': paired, 'questions': questions,
         'requests': len(records), 'unique_semantic_questions': len(questions), 'source_roots': len(ROOTS),
         'prototype_roots': 1, 'model_calls': len(freeze['submitted_ids']), 'returned': freeze['returned'],
         'wall_seconds': freeze['wall_seconds'], 'total_prompt_tokens': sum(r['prompt_tokens'] for r in records),
         'total_output_tokens': sum(r['output_tokens'] for r in records), 'failure': freeze['failure'],
         'limits': read(HERE / 'PROTOCOL_FREEZE.json')['limits']})
    print(json.dumps({'summaries': summaries, 'model_calls': len(freeze['submitted_ids']), 'returned': freeze['returned']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run', 'analyze'])
    globals()[parser.parse_args().action]()
