"""Independent closed-loop evaluation, reusing the frozen structured-v2 engine."""
from __future__ import annotations
import argparse, collections, hashlib, importlib.util, json, re, sys
from pathlib import Path

sys.dont_write_bytecode = True
P = Path(__file__).resolve().parent
OUT = P / 'execution'
W = P.parent
MAX_TOKENS, MODEL_LEN = 6144, 16384
ARMS = ('F', 'G')
SLOTS = {
    'MC2': ('multi_select', 'MC_QUANTITATIVE_RELATION', 2),
    'MC3': ('multi_select', 'MC_CONSTRAINT_CONJUNCTION', 3),
    'MC4': ('multi_select', 'MC_EVIDENCE_SELECTION', 4),
    'MC5': ('multi_select', 'MC_CAUSAL_SUPPORT', 5),
    'MC6': ('multi_select', 'MC_CONSTRAINT_CONJUNCTION', 6),
    'ORDER1': ('ordering', 'ORDER_EXPLICIT_TIME', None),
    'ORDER2': ('ordering', 'ORDER_EXPLICIT_PROCEDURE', None),
    'ORDER3': ('ordering', 'ORDER_DISCOURSE_RECONSTRUCTION', None),
}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


v2 = load('closed_loop_frozen_structured_engine', W / 'thinking_switch_20261024_structured_v2/run.py')
orig = v2.orig
read, bind, check, need = v2.read, v2.bind, v2.check, v2.need
PUBLIC_KEYS = orig.PUBLIC_KEYS | {'base_request_id', 'mode', 'enable_thinking', 'arm', 'slot_id', 'messages_sha256'}


def paths_equal(bindings, expected, code):
    paths = [str(Path(b['path']).resolve()) for b in bindings]
    need(len(paths) == len(set(paths)) and set(paths) == {str(Path(p).resolve()) for p in expected}, code)


def display_variants(candidate):
    """Two fixed seed bases; actual F/G permutations need not be identical."""
    seen = set()
    for index in range(2):
        seed, skipped = 2026102427 + index, []
        while True:
            public, presentation = orig.gate.prepare_display(candidate, seed)
            permutation = tuple(presentation['display_to_canonical'].values())
            shown_order = tuple(public['items' if candidate['question_type'] == 'ordering' else 'options'])
            if permutation not in seen and not (candidate['question_type'] == 'ordering' and tuple(presentation['display_gold']) == shown_order):
                break
            skipped.append(seed)
            seed += 2
        seen.add(permutation)
        presentation['display_seed_base'] = 2026102427 + index
        presentation['rejected_display_seeds'] = skipped
        yield index, public, presentation


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def prepare():
    need(not (OUT / 'PREPARATION.json').exists(), 'NO_PREPARATION_OVERWRITE')
    endpoint, config = orig.endpoint_receipt()
    candidates = read(P / 'CANDIDATES_PRIVATE.json')
    review_rows = read(P / 'CONTENT_REVIEW_AI.json')['reviews']
    reviews = {r['candidate_id']: r for r in review_rows}
    need(isinstance(candidates, list) and 0 < len(candidates) <= 64, 'CANDIDATE_COUNT')
    need(len(candidates) == len({c['candidate_id'] for c in candidates}) == len(reviews) == len(review_rows), 'DUPLICATE_OR_MISSING_CANDIDATE_REVIEW')
    need(set(reviews) == {c['candidate_id'] for c in candidates}, 'REVIEW_SET_DRIFT')
    selection = read(P / 'SOURCE_SELECTION_FINAL.json')
    roots = {r['root_id']: r for r in selection['roots']}
    need(len(roots) == len(selection['roots']) == 4 and all(r.get('split', selection.get('split')) == 'train' for r in roots.values()), 'FOUR_TRAIN_ROOTS_REQUIRED')
    config_doc = read(P / 'CONFIG.json')
    need(set(roots) == {'cc-train-episode-' + root for root in config_doc['roots']} and config_doc['arms'] == list(ARMS)
        and config_doc['planned_semantic_questions'] == 64 and config_doc['planned_solver_requests'] == 128 and config_doc['display_variants'] == 2, 'FROZEN_DESIGN_DRIFT')
    content_freeze = read(P / 'CONTENT_FREEZE.json')
    for b in content_freeze['bindings']:
        check(b)
    for path in (P / 'CANDIDATES_PRIVATE.json', P / 'CONTENT_REVIEW_AI.json', P / 'SOURCE_SELECTION_FINAL.json'):
        need(bind(path) in content_freeze['bindings'], 'UNBOUND_CONTENT_INPUT:' + path.name)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(orig.TOKENIZER), local_files_only=True)
    from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
    reasoner = Qwen3ReasoningParser(tokenizer=tok)
    api = v2.cpu_api_check()
    save(OUT / 'CPU_API_CHECK.json', api)
    rows, tokens, scoring, bindings, ledger, quality = [], {}, [], [], [], []
    bindings += [bind(Path(__file__)), bind(W / 'thinking_switch_20261024_structured_v2/run.py')]
    bindings += [bind(P / 'PROTOCOL_ZH.md'), bind(P / 'CONFIG.json'), bind(P / 'question_authoring/COMMON_AUTHOR_CONTRACT_ZH.md'), bind(OUT / 'CPU_API_CHECK.json')]
    bindings += api['implementation_bindings']
    bindings += [bind(x) for x in orig.code_paths()]
    mc_counts, arm_counts, slot_keys = collections.Counter(), collections.Counter(), set()
    for candidate in sorted(candidates, key=lambda c: (c['root_id'], c['slot_id'], c['arm'])):
        cid = candidate['candidate_id']
        rv = reviews[cid]
        arm, slot, root = candidate['arm'], candidate['slot_id'], candidate['root_id']
        need(arm in ARMS and slot in SLOTS and root in roots, 'UNPLANNED_ARM_ROOT_OR_SLOT:' + cid)
        key = (root, slot, arm)
        need(key not in slot_keys, 'DUPLICATE_PLANNED_SLOT:' + cid)
        slot_keys.add(key)
        kind, operation, count = SLOTS[slot]
        need(candidate['question_type'] == kind and candidate['task_operation'] == operation, 'SLOT_OPERATION_DRIFT:' + cid)
        field = 'options' if kind == 'multi_select' else 'items'
        need(set(candidate[field]) == set('ABCDEF') and (count is None or len(candidate['gold']) == count), 'SLOT_ITEM_OR_GOLD_COUNT_DRIFT:' + cid)
        selected_binding = {k: roots[root][k] for k in ('path', 'sha256', 'bytes')}
        need(candidate['source_binding'] == selected_binding, 'SOURCE_SELECTION_DRIFT:' + cid)
        need(rv['decision'] in ('PASS', 'HOLD', 'FAIL'), 'UNKNOWN_REVIEW_DECISION:' + cid)
        need(rv['candidate_sha256'] == orig.fingerprint(candidate), 'REVIEW_HASH_DRIFT:' + cid)
        need(candidate['author_id'] and rv['author_id'] == candidate['author_id'] != rv['reviewer_id'], 'AUTHOR_REVIEW_CONFLICT')
        ledger.append({'candidate_id': cid, 'root_id': root, 'arm': arm, 'slot_id': slot, 'decision': rv['decision']})
        if rv['decision'] != 'PASS':
            continue
        orig.gate._validate(candidate)
        result = orig.production_gate.validate_review(candidate, rv, P / 'SOURCE_SELECTION_FINAL.json')
        need(result['admitted'], 'CONTENT_GATE:' + cid + ':' + json.dumps(result))
        source = read(candidate['source_binding']['path'])
        check(candidate['source_binding'])
        need(source['root_id'] == candidate['root_id'], 'SOURCE_ROOT_DRIFT')
        refs = [t['turn_ref'] for t in source['turns']]
        need(len(refs) == len(set(refs)), 'DUPLICATE_SOURCE_REFS')
        arm_counts[candidate['arm']] += 1
        if candidate['question_type'] == 'multi_select':
            mc_counts[(candidate['arm'], len(candidate['gold']))] += 1
        option_words = {label: len(text.split()) for label, text in candidate[field].items()}
        quality.append({'candidate_id': cid, 'arm': arm, 'root_id': root, 'slot_id': slot,
            'question_words': len(candidate['question'].split()), 'option_words': option_words,
            'option_length_guideline': '10..35 English whitespace-delimited words, descriptive pre-Solver check; independent content review governs admission',
            'option_length_guideline_flags': [label for label, n in option_words.items() if not 10 <= n <= 35]})
        # Both arms use the same seed bases; gate hashing includes candidate_id.
        base_seed = int.from_bytes(hashlib.sha256((candidate['root_id'] + ':' + candidate['slot_id']).encode()).digest()[:4], 'big') % 2147483647
        for index, public, presentation in display_variants(candidate):
            rid = f'{cid}__perm_{index:02}__FULL__THINK_OFF__CL1'
            row = {k: v for k, v in public.items() if k != field}
            row.update(request_id=rid, base_request_id=rid, cohort='new', condition='FULL', mode='THINK_OFF', enable_thinking=False,
                       arm=candidate['arm'], slot_id=candidate['slot_id'], variant_id=f'perm_{index:02}', display=public[field], labels=list(public[field]), sampling_seed=base_seed)
            old = orig.diagnostic.make_prompt(tok, row, row['display'], source)
            pattern = r'<\|im_start\|>system\n(.*?)<\|im_end\|>\n<\|im_start\|>user\n(.*?)<\|im_end\|>\n<\|im_start\|>assistant\n<think>\n\n</think>\n\n'
            match = re.fullmatch(pattern, old, re.S)
            need(match, 'PROMPT_TEMPLATE_DRIFT')
            system, user = match.groups()
            oldtail = 'Return only JSON, with\nitem_evidence first, answer_reason second and final_answer last. Provide brief\nevidence-based explanations, not extended private deliberation.'
            newtail = 'Your final response must be JSON, with item_evidence first, answer_reason second and final_answer last. Keep the explanations in the final JSON brief and evidence-based.'
            need(user.count(oldtail) == 1, 'NEUTRAL_INSTRUCTION_DRIFT')
            messages = [{'role': 'system', 'content': system.replace('Give concise, verifiable explanations', 'In the final response, give concise, verifiable explanations')}, {'role': 'user', 'content': user.replace(oldtail, newtail)}]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok.encode(prompt, add_special_tokens=False)
            need(len(ids) + MAX_TOKENS <= MODEL_LEN, 'CONTEXT_OVERFLOW_NO_TRUNCATION:' + rid + ':' + str(len(ids)))
            need(reasoner.is_reasoning_end(ids) is True, 'THINK_OFF_REASONING_GATE_DRIFT:' + rid)
            schema = orig.diagnostic.repair.schema(row['labels'], refs, row['question_type'])
            pp, sp = OUT / 'prompts' / (rid + '.txt'), OUT / 'schemas' / (rid + '.json')
            pp.parent.mkdir(parents=True, exist_ok=True)
            with pp.open('x') as f:
                f.write(prompt)
            save(sp, schema)
            row.update(prompt_path=str(pp), schema_path=str(sp), prompt_tokens=len(ids), messages_sha256=hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest())
            need(set(row) == PUBLIC_KEYS, 'PUBLIC_FIELD_ALLOWLIST_DRIFT')
            rows.append(row)
            tokens[rid] = ids
            scoring.append({'request_id': rid, 'candidate_id': cid, 'arm': candidate['arm'], 'slot_id': candidate['slot_id'], 'presentation': presentation})
            bindings += [bind(pp), bind(sp), candidate['source_binding']]
    need(rows and len(rows) <= 128 and len({r['request_id'] for r in rows}) == len(rows), 'REQUEST_COUNT')
    save(OUT / 'PUBLIC_REQUESTS.json', {'requests': rows})
    save(OUT / 'PROMPT_TOKEN_IDS.json', tokens)
    save(OUT / 'PRIVATE_SCORING.json', {'requests': scoring, 'candidate_binding': bind(P / 'CANDIDATES_PRIVATE.json'), 'review_binding': bind(P / 'CONTENT_REVIEW_AI.json'),
        'selection_binding': bind(P / 'SOURCE_SELECTION_FINAL.json'), 'admission_ledger': ledger, 'planned_root_ids': sorted(roots)})
    (OUT / 'PRIVATE_SCORING.json').chmod(0o600)
    bindings += [bind(OUT / 'PUBLIC_REQUESTS.json'), bind(OUT / 'PROMPT_TOKEN_IDS.json')]
    save(OUT / 'PRE_SOLVER_QUALITY.json', {'selection_before_solver': True, 'records': quality,
        'display_seed_bases': [2026102427, 2026102428], 'matched_sampling_seed_key': ['root_id', 'slot_id'],
        'actual_permutations_in_private_scoring': True, 'same_cross_arm_permutation_claimed': False})
    bindings.append(bind(OUT / 'PRE_SOLVER_QUALITY.json'))
    save(OUT / 'PREPARATION.json', {'requests': len(rows), 'planned_requests': 128, 'bindings': list({b['path']: b for b in bindings}.values()),
        'private_scoring_binding': bind(OUT / 'PRIVATE_SCORING.json'), 'content_freeze_binding': bind(P / 'CONTENT_FREEZE.json'),
        'config_receipt': config, 'runtime_versions': orig.versions(), 'structured_outputs_config': v2.STRUCTURED_CONFIG,
        'fixed_order': [r['request_id'] for r in rows], 'max_tokens': MAX_TOKENS, 'max_model_len': MODEL_LEN,
        'sampling': {'temperature': .8, 'top_p': .9, 'top_k': -1, 'repetition_penalty': 1, 'engine_seed': 0},
        'batch_size': 8, 'prefix_cache': False, 'automatic_retry': False, 'source_truncation': False,
        'arm_question_counts': dict(arm_counts), 'planned_semantic_questions': 64, 'planned_root_ids': sorted(roots),
        'public_generation_reads_private_scoring': False, 'reasoning_gate_prompts_checked': len(rows),
        'max_prompt_tokens': max(r['prompt_tokens'] for r in rows)})
    print(json.dumps({'prepared': len(rows), 'arms': dict(arm_counts), 'max_prompt_tokens': max(r['prompt_tokens'] for r in rows)}), flush=True)


def public():
    prep = read(OUT / 'PREPARATION.json')
    forbidden = {str((P / name).resolve()) for name in ('CANDIDATES_PRIVATE.json', 'CONTENT_REVIEW_AI.json', 'CONTENT_FREEZE.json', 'SOURCE_SELECTION_FINAL.json')}
    forbidden.add(str((OUT / 'PRIVATE_SCORING.json').resolve()))
    for b in prep['bindings']:
        need(str(Path(b['path']).resolve()) not in forbidden, 'PRIVATE_INPUT_IN_GENERATION_BINDINGS')
        check(b)
    rows = read(OUT / 'PUBLIC_REQUESTS.json')['requests']
    tokens = read(OUT / 'PROMPT_TOKEN_IDS.json')
    need(prep['fixed_order'] == [r['request_id'] for r in rows], 'FROZEN_ORDER_DRIFT')
    need(0 < len(rows) == prep['requests'] == len(set(prep['fixed_order'])) <= 128 and set(tokens) == set(prep['fixed_order']), 'REQUEST_SET_DRIFT')
    need(all(set(r) == PUBLIC_KEYS and r['condition'] == 'FULL' and r['cohort'] == 'new' and r['arm'] in ARMS and r['slot_id'] in SLOTS and r['labels'] == list(r['display']) == list('ABCDEF') for r in rows), 'NONPUBLIC_REQUEST_FIELDS_OR_DISPLAY_DRIFT')
    need(all(r['mode'] == 'THINK_OFF' and r['enable_thinking'] is False and r['prompt_tokens'] == len(tokens[r['request_id']]) and r['prompt_tokens'] + MAX_TOKENS <= MODEL_LEN for r in rows), 'MODE_OR_BUDGET_DRIFT')
    need(prep['structured_outputs_config'] == v2.STRUCTURED_CONFIG and prep['max_tokens'] == MAX_TOKENS and prep['max_model_len'] == MODEL_LEN, 'PREP_ENGINE_CONFIG_DRIFT')
    need(prep['sampling'] == {'temperature': .8, 'top_p': .9, 'top_k': -1, 'repetition_penalty': 1, 'engine_seed': 0} and prep['batch_size'] == v2.BATCH and prep['prefix_cache'] is False and prep['automatic_retry'] is False and prep['source_truncation'] is False, 'PREP_SAMPLING_DRIFT')
    required = [Path(__file__), W / 'thinking_switch_20261024_structured_v2/run.py', P / 'PROTOCOL_ZH.md', P / 'CONFIG.json', P / 'question_authoring/COMMON_AUTHOR_CONTRACT_ZH.md', OUT / 'CPU_API_CHECK.json', OUT / 'PRE_SOLVER_QUALITY.json', OUT / 'PUBLIC_REQUESTS.json', OUT / 'PROMPT_TOKEN_IDS.json', *orig.code_paths()]
    api = read(OUT / 'CPU_API_CHECK.json')
    need(api['gpu_model_started'] is False and api['structured_outputs_config'] == v2.STRUCTURED_CONFIG and api['runtime_versions'] == prep['runtime_versions'], 'CPU_API_RECEIPT_DRIFT')
    required.extend(Path(b['path']) for b in api['implementation_bindings'])
    groups = collections.defaultdict(list)
    for row in rows:
        need(row['root_id'] in prep['planned_root_ids'], 'UNPLANNED_PUBLIC_ROOT')
        need(row['question_type'] == SLOTS[row['slot_id']][0] and row['task_operation'] == SLOTS[row['slot_id']][1], 'PUBLIC_SLOT_OPERATION_DRIFT')
        expected_seed = int.from_bytes(hashlib.sha256((row['root_id'] + ':' + row['slot_id']).encode()).digest()[:4], 'big') % 2147483647
        need(row['sampling_seed'] == expected_seed and row['base_request_id'] == row['request_id'], 'PUBLIC_SAMPLING_SEED_OR_BASE_ID_DRIFT')
        groups[(row['root_id'], row['slot_id'], row['arm'], row['candidate_id'])].append(row)
        required.extend([Path(row['prompt_path']), Path(row['schema_path']), Path(row['source_binding']['path'])])
    for group in groups.values():
        need(len(group) == 2 and {r['variant_id'] for r in group} == {'perm_00', 'perm_01'}, 'MISSING_OR_DUPLICATE_DISPLAY_PAIR')
    need(len({(root, slot, arm) for root, slot, arm, cid in groups}) == len(groups), 'DUPLICATE_PUBLIC_SLOT')
    bound_paths = [str(Path(b['path']).resolve()) for b in prep['bindings']]
    need(len(bound_paths) == len(set(bound_paths)) and all(bind(p) in prep['bindings'] for p in required), 'UNBOUND_PUBLIC_INPUT')
    return prep, rows, tokens


def check_raw(frozen, rows, tokens):
    ids = [r['request_id'] for r in rows]
    need(0 < len(ids) == len(set(ids)) <= 128 and frozen['request_ids'] == ids and frozen['requests'] == len(ids), 'RAW_SET_DRIFT')
    check(frozen['started_binding'])
    need(frozen['started_binding'] == bind(OUT / 'STARTED.json'), 'STARTED_PATH_DRIFT')
    started = read(OUT / 'STARTED.json')
    need(started['experiment_id'] == P.name and started['request_ids'] == ids and started['requests'] == len(ids), 'STARTED_REQUEST_SET_DRIFT')
    need(started['preparation_binding'] == bind(OUT / 'PREPARATION.json') and started['structured_outputs_config'] == v2.STRUCTURED_CONFIG and started['gold_read'] is False, 'STARTED_CONFIG_OR_PREPARATION_DRIFT')
    submitted = frozen['submitted_ids']
    need(len(submitted) == len(set(submitted)) and submitted == [rid for rid in ids if rid in set(submitted)], 'SUBMISSION_ID_DRIFT')
    for dirname, bindings, expected in (
        ('raw', frozen['bindings'], [OUT / 'raw' / (rid + '.json') for rid in ids]),
        ('submitted', frozen['submission_bindings'], [OUT / 'submitted' / (rid + '.json') for rid in submitted])):
        paths_equal(bindings, expected, dirname.upper() + '_BINDING_SET_DRIFT')
        need({str(p.resolve()) for p in (OUT / dirname).glob('*.json')} == {str(p.resolve()) for p in expected}, dirname.upper() + '_DIRECTORY_SET_DRIFT')
        for binding in bindings:
            check(binding)
    receipt_binding = frozen['model_receipt_binding']
    if receipt_binding is not None:
        check(receipt_binding)
        need(receipt_binding == bind(OUT / 'MODEL_LOAD_RECEIPT.json'), 'MODEL_RECEIPT_PATH_DRIFT')
        v2.HERE, v2.OUT = P, OUT
        v2.checked_model_receipt(read(receipt_binding['path']))
    else:
        need(not submitted and frozen['returned'] == 0, 'MISSING_MODEL_RECEIPT_AFTER_SUBMISSION')
    paths_equal(frozen['batch_bindings'], list((OUT / 'batches').glob('*.json')), 'BATCH_DIRECTORY_SET_DRIFT')
    finished_ids, finished_batches = [], []
    for binding in frozen['batch_bindings']:
        check(binding)
        batch = read(binding['path'])
        index = batch['batch_index']
        need(type(index) is int and 0 <= index < (len(ids) + v2.BATCH - 1) // v2.BATCH, 'BATCH_INDEX_DRIFT')
        need(Path(binding['path']).resolve() == (OUT / 'batches' / (str(index).zfill(3) + '.json')).resolve(), 'BATCH_PATH_DRIFT')
        need(batch['request_ids'] == ids[index * v2.BATCH:(index + 1) * v2.BATCH], 'BATCH_REQUEST_SET_DRIFT')
        finished_batches.append(index)
        finished_ids.extend(batch['request_ids'])
    need(finished_batches == list(range(len(finished_batches))) and len(finished_ids) == len(set(finished_ids)) and set(finished_ids) <= set(submitted), 'BATCH_ORDER_OR_SUBMISSION_DRIFT')
    returned = 0
    for r in rows:
        raw = read(OUT / 'raw' / (r['request_id'] + '.json'))
        rid = r['request_id']
        need(raw['request_id'] == rid and raw['mode'] == r['mode'] and raw['base_request_id'] == r['base_request_id'], 'RAW_ID_OR_MODE_DRIFT')
        need(raw['status'] in ('RETURNED', 'BACKEND_ERROR', 'NOT_SUBMITTED'), 'UNKNOWN_RAW_STATUS')
        need((raw['status'] != 'NOT_SUBMITTED') == (rid in set(submitted)), 'RAW_SUBMISSION_STATUS_DRIFT')
        if rid in set(submitted):
            submission = read(OUT / 'submitted' / (rid + '.json'))
            need(submission['request_id'] == rid and submission['experiment_id'] == P.name and submission['model_load_receipt_binding'] == receipt_binding, 'SUBMISSION_RECEIPT_DRIFT')
        if raw['status'] == 'RETURNED':
            returned += 1
            need(raw['model_load_receipt_binding'] == receipt_binding, 'RAW_MODEL_RECEIPT_DRIFT')
            need(raw.get('capture_errors') == orig.capture.response_errors(raw, Path(r['prompt_path']).read_text(), tokens[r['request_id']]), 'CAPTURE_CHECK_DRIFT')
    need(returned == frozen['returned'], 'RETURNED_COUNT_DRIFT')
    if frozen['failure'] is None:
        need(returned == len(rows) and submitted == ids and finished_batches == list(range((len(ids) + v2.BATCH - 1) // v2.BATCH)), 'INCOMPLETE_SUCCESS')


def run():
    # Reuse exactly the validated engine implementation, only replacing batch-local paths/checks.
    v2.HERE, v2.OUT, v2.public, v2.checked_raw_manifest = P, OUT, public, check_raw
    v2.run()


def in_group(record, group):
    strict = record['strict']
    return (group == 'ALL' or group == 'MC2_5_PRIMARY' and strict['gold_count'] in (2, 3, 4, 5)
        or group == 'MC6_CONTROL' and strict['gold_count'] == 6
        or group == 'ORDERING' and strict['question_type'] == 'ordering'
        or group == 'NON_MC6_PRIMARY' and strict['gold_count'] != 6)


def summary_metrics(records):
    strict = [r['strict'] for r in records]
    valid = [r for r in records if r['strict']['valid']]
    result = orig.base.aggregate(strict)
    result['response_valid_conditioned'] = orig.base.aggregate([r['strict'] for r in valid])
    for label, subset in (('all_admitted_requests', records), ('valid_responses_only', valid)):
        mc = [r for r in subset if r['strict']['question_type'] == 'multi_select']
        result[label + '_mc_macro'] = {metric: sum(r['partial'][metric] for r in mc) / len(mc) if mc else None for metric in ('precision', 'recall', 'f1')}
    result['unique_semantic_questions'] = len({r['strict']['candidate_id'] for r in records})
    result['invalid_responses'] = sum(not r['strict']['valid'] for r in records)
    result['valid_but_wrong_responses'] = sum(r['strict']['valid'] and not r['strict']['exact'] for r in records)
    return result


def analyze():
    prep, rows, tokens = public()
    frozen = read(OUT / 'RAW_RESPONSE_FREEZE.json')
    v2.HERE, v2.OUT = P, OUT
    check_raw(frozen, rows, tokens)
    check(prep['private_scoring_binding'])
    need(prep['private_scoring_binding'] == bind(OUT / 'PRIVATE_SCORING.json'), 'PRIVATE_SCORING_PATH_DRIFT')
    check(prep['content_freeze_binding'])
    need(prep['content_freeze_binding'] == bind(P / 'CONTENT_FREEZE.json'), 'CONTENT_FREEZE_PATH_DRIFT')
    for binding in read(P / 'CONTENT_FREEZE.json')['bindings']:
        check(binding)
    private_doc = read(OUT / 'PRIVATE_SCORING.json')
    private = {r['request_id']: r for r in private_doc['requests']}
    need(len(private) == len(private_doc['requests']) == len(rows) and list(private) == prep['fixed_order'], 'PRIVATE_REQUEST_SET_DRIFT')
    for key, name in [('candidate_binding', 'CANDIDATES_PRIVATE.json'), ('review_binding', 'CONTENT_REVIEW_AI.json'), ('selection_binding', 'SOURCE_SELECTION_FINAL.json')]:
        check(private_doc[key])
        need(private_doc[key] == bind(P / name), 'PRIVATE_CONTENT_PATH_DRIFT')
    need(private_doc['planned_root_ids'] == prep['planned_root_ids'], 'PRIVATE_PLANNED_ROOT_DRIFT')
    candidates = {c['candidate_id']: c for c in read(P / 'CANDIDATES_PRIVATE.json')}
    records = []
    for row in rows:
        path = OUT / 'raw' / (row['request_id'] + '.json')
        raw = read(path)
        final, parsed, ans, errors, valid = None, None, [], [], False
        try:
            need(raw['status'] == 'RETURNED' and raw.get('finish_reason') == 'stop', 'BACKEND_OR_LENGTH_FAILURE')
            need(not raw.get('capture_errors'), 'CAPTURE_ERRORS')
            thinking, final, boundary = v2.split_output(raw['raw_output'])
            need(not thinking, 'UNEXPECTED_NONEMPTY_THINKING')
            parsed, ans = v2.legal_answer(final, row)
            orig.base.validate_output(final, row, read(row['schema_path']))
            valid = True
        except Exception as e:
            errors.append(str(e))
        scoring = private[row['request_id']]
        need(all(scoring[k] == row[k] for k in ('candidate_id', 'arm', 'slot_id')), 'PRIVATE_PUBLIC_ID_DRIFT')
        index = int(row['variant_id'][-2:])
        expected_public, expected_presentation = list(display_variants(candidates[row['candidate_id']]))[index][1:]
        need(scoring['presentation'] == expected_presentation and expected_public['options' if row['question_type'] == 'multi_select' else 'items'] == row['display'], 'PRIVATE_PRESENTATION_DRIFT')
        strict = orig.base.score_record(row, scoring['presentation'], ans, valid)
        strict['variant_id'] = row['variant_id']
        if strict['question_type'] == 'multi_select':
            tp = len(set(strict['prediction']) & set(strict['gold']))
            fp, fn = strict['false_positive'], strict['false_negative']
            partial = {'precision': tp / (tp + fp) if tp + fp else 0, 'recall': tp / (tp + fn), 'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0}
        else:
            partial = {'pairwise_accuracy': strict['pairwise_correct'] / strict['pairwise_total']}
        records.append({'request_id': row['request_id'], 'arm': row['arm'], 'slot_id': row['slot_id'], 'strict': strict,
            'partial': partial, 'errors': errors, 'parsed': parsed, 'final_text': final,
            'raw_binding': bind(path), 'prompt_tokens': row['prompt_tokens'], 'output_tokens': len(raw.get('output_token_ids', [])), 'finish_reason': raw.get('finish_reason')})
    save(OUT / 'SCORES.json', {'records': records, 'raw_freeze_binding': bind(OUT / 'RAW_RESPONSE_FREEZE.json'), 'private_scoring_binding': prep['private_scoring_binding']})
    summaries = []
    for arm in ARMS:
        for group in ['ALL', 'MC2_5_PRIMARY', 'MC6_CONTROL', 'ORDERING', 'NON_MC6_PRIMARY']:
            chosen = [r for r in records if r['arm'] == arm and in_group(r, group)]
            summaries.append({'arm': arm, 'group': group, **summary_metrics(chosen)})
    admitted = {(r['strict']['root_id'], r['slot_id'], r['arm']) for r in records}
    matched = {(root, slot) for root in prep['planned_root_ids'] for slot in SLOTS if all((root, slot, arm) in admitted for arm in ARMS)}
    paired_summaries = [{'arm': arm, 'group': group, 'matched_semantic_slot_pairs': len({(r['strict']['root_id'], r['slot_id']) for r in records if (r['strict']['root_id'], r['slot_id']) in matched and in_group(r, group)}),
        **summary_metrics([r for r in records if r['arm'] == arm and (r['strict']['root_id'], r['slot_id']) in matched and in_group(r, group)])}
        for arm in ARMS for group in ('ALL', 'MC2_5_PRIMARY', 'MC6_CONTROL', 'ORDERING', 'NON_MC6_PRIMARY')]
    strata = []
    for arm in ARMS:
        for dimension in ('root_id', 'task_operation', 'gold_count'):
            values = sorted({r['strict'][dimension] for r in records if r['arm'] == arm}, key=str)
            for value in values:
                chosen = [r for r in records if r['arm'] == arm and r['strict'][dimension] == value]
                strata.append({'arm': arm, 'dimension': dimension, 'value': value, **summary_metrics(chosen)})
    slot_completion = [{'root_id': root, 'slot_id': slot, 'arm': arm,
        'content_admitted': (root, slot, arm) in admitted,
        'content_matched_across_arms': (root, slot) in matched,
        'planned_displays': 2, 'frozen_requests': sum(r['arm'] == arm and r['slot_id'] == slot and r['strict']['root_id'] == root for r in records)}
        for root in prep['planned_root_ids'] for slot in SLOTS for arm in ARMS]
    by_question = collections.defaultdict(list)
    for r in records:
        by_question[r['strict']['candidate_id']].append(r)
    consistency = [{'candidate_id': cid, 'arm': rs[0]['arm'], 'root_id': rs[0]['strict']['root_id'], 'slot_id': rs[0]['slot_id'],
        'display_exact': [r['strict']['exact'] for r in rs], 'both_correct': all(r['strict']['exact'] for r in rs),
        'both_valid': all(r['strict']['valid'] for r in rs),
        'same_canonical_prediction': all(r['strict']['valid'] for r in rs) and len({tuple(r['strict']['prediction']) for r in rs}) == 1} for cid, rs in by_question.items()]
    save(OUT / 'ANALYSIS_SUMMARY.json', {'summaries': summaries, 'content_matched_slot_summaries': paired_summaries,
        'stratified_summaries': strata, 'planned_slot_completion': slot_completion,
        'denominators': {'planned_semantic_questions': 64, 'planned_solver_requests': 128,
            'content_admitted_semantic_questions': len(by_question), 'frozen_requests': len(records),
            'content_matched_slot_pairs': len(matched), 'submitted_requests': len(frozen['submitted_ids']), 'returned_requests': frozen['returned'],
            'valid_responses': sum(r['strict']['valid'] for r in records),
            'held_or_unavailable_semantic_slots': 64 - len(by_question),
            'accuracy_denominator_policy': 'Strict accuracy uses all frozen requests for content-admitted questions; held/missing content slots are completion deficits, not Solver failures. Response-valid-conditioned metrics are separate. Two displays/question are dependent measurements.'},
        'question_consistency': consistency, 'raw_wall_seconds': frozen['wall_seconds'],
        'total_prompt_tokens': sum(r['prompt_tokens'] for r in records), 'total_output_tokens': sum(r['output_tokens'] for r in records),
        'inference_limits': 'Same-source F/G question-generation guidance comparison, 4 discovery Train roots, dependent displays; no training, unseen-root or separate expansion-feedback causal claim.'})
    print(json.dumps({'summaries': summaries, 'raw_wall_seconds': frozen['wall_seconds']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run', 'analyze'])
    globals()[parser.parse_args().action]()
