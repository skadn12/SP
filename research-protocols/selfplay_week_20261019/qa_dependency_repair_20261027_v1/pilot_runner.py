"""Reuses the frozen v2 B1 engine; adds dependency diagnostics and equal prompts."""
from __future__ import annotations
import argparse, collections, copy, hashlib, importlib.util, json, re, statistics, sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'natural_question_forms_fg_15root_extension_20261026_v1'
OUT = HERE / 'execution'

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

prior = load('dependency_original_runner', OLD / 'runner.py')
quality = load('dependency_new_quality_gate', HERE / 'quality_gate.py')
engine, orig = prior.engine, prior.orig
read, bind, check, need, save = prior.read, prior.bind, prior.check, prior.need, prior.save
MAX_TOKENS, MODEL_LEN, BATCH = 6144, 16384, 8
SAMPLING = {'temperature': .8, 'top_p': .9, 'top_k': -1, 'repetition_penalty': 1, 'engine_seed': 0}
PUBLIC_KEYS = {'candidate_id', 'root_id', 'question_type', 'question', 'request_id',
               'base_request_id', 'condition', 'cohort', 'mode', 'enable_thinking',
               'variant_id', 'display', 'labels', 'prompt_path', 'schema_path',
               'prompt_tokens', 'sampling_seed'}
BASE_CONDITIONS = ('FULL', 'NO_CONTEXT')
NEW_FIXED_CONDITIONS = ('FULL', 'NO_CONTEXT', 'PARENT_ONLY', 'EXTENSION_ONLY', 'EVIDENCE_ONLY')

def runtime_code_paths():
    return [Path(__file__), HERE / 'quality_gate.py', OLD / 'runner.py', prior.OLD / 'solver_runner.py',
            prior.SCORER_PATH, Path(engine.__file__), quality.LEGACY_PATH,
            *orig.code_paths()]

def paths_equal(bindings, expected, code):
    paths = [str(Path(b['path']).resolve()) for b in bindings]
    need(len(paths) == len(set(paths)) and set(paths) == {str(Path(p).resolve()) for p in expected}, code)

def review_records(value):
    # Frozen review artifacts may use either supported JSON container.
    rows = value.get('reviews') if isinstance(value, dict) else value
    need(isinstance(rows, list) and all(isinstance(row, dict) for row in rows), 'REVIEW_CONTAINER_DRIFT')
    return rows

def sampling_seed(root, index):
    material = f'{root}:{index}:dependency_v1'.encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], 'big') % 2147483647

def output_schema():
    item = {'type': 'object', 'properties': {
        'judgment': {'type': 'string', 'enum': ['supported', 'contradicted', 'not_supported', 'ambiguous']},
        'turn_refs': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 3, 'uniqueItems': True},
        'reason': {'type': 'string', 'minLength': 1, 'maxLength': 320}},
        'required': ['judgment', 'turn_refs', 'reason'], 'additionalProperties': False}
    return {'type': 'object', 'properties': {
        'item_evidence': {'type': 'object', 'properties': {k: copy.deepcopy(item) for k in 'ABCDEF'},
                          'required': list('ABCDEF'), 'additionalProperties': False},
        'answer_reason': {'type': 'string', 'minLength': 1, 'maxLength': 800},
        'final_answer': {'type': 'string', 'pattern': '^<box>(?:[A-F](?: [A-F]){0,5})?</box>$'}},
        'required': ['item_evidence', 'answer_reason', 'final_answer'], 'additionalProperties': False}

def make_prompt(tokenizer, question, display, turns):
    dialogue = '\n'.join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in turns)
    dialogue = dialogue or '[No dialogue supplied]'
    instruction = (
        'Give your best prediction of the option set that answers the question. '
        'Assess all options; the number selected is not specified. For each label, '
        'give judgment supported, contradicted, not_supported or ambiguous, up to '
        'three turn_refs that actually appear in the supplied dialogue, and a brief '
        'reason. Use empty turn_refs when no source evidence is available. '
        'Do not invent missing evidence. Then give answer_reason and final_answer '
        'as <box>A B</box>, using selected labels once each in alphabetical order; '
        'the example shows format only. An empty selection is <box></box>. '
        'Return only JSON in this key order: item_evidence, answer_reason, final_answer. '
        'Keep reasons brief and verifiable. These instructions apply equally to all inputs.')
    messages = [
        {'role': 'system', 'content': 'Answer the question using the supplied information. Give concise evidence-based judgments and the best final option set.'},
        {'role': 'user', 'content': f"Dialogue:\n{dialogue}\n\nQuestion:\n{question}\n\nOptions:\n" +
         '\n'.join(f'{key}. {text}' for key, text in display.items()) + '\n\n' + instruction}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

def conditions(candidate, source, cohort):
    turns = source['turns']
    result = [('FULL', turns), ('NO_CONTEXT', [])]
    if cohort == 'baseline':
        return result
    # Parent is the cumulative pre-R03 source; it is not the raw original dataset.
    result += [('PARENT_ONLY', [t for t in turns if not t['turn_ref'].startswith('R03:')]),
               ('EXTENSION_ONLY', [t for t in turns if t['turn_ref'].startswith('R03:')])]
    cert = candidate['dependency_certificate']
    evidence_refs = {e['turn_ref'] for a in candidate['item_analysis'].values() for e in a['evidence']}
    evidence_refs |= {r for g in cert['premise_groups'] for r in g['turn_refs']}
    result += [('EVIDENCE_ONLY', [t for t in turns if t['turn_ref'] in evidence_refs])]
    for index, group in enumerate(cert['premise_groups']):
        refs = set(group['turn_refs'])
        result += [(f'ABLATE_G{index+1}', [t for t in turns if t['turn_ref'] not in refs]),
                   (f'ONLY_G{index+1}', [t for t in turns if t['turn_ref'] in refs])]
    return result

def prepare():
    need(not (OUT / 'PREPARATION.json').exists(), 'NO_PREPARATION_OVERWRITE')
    _, config = orig.endpoint_receipt()
    plan = read(HERE / 'PILOT_PLAN.json')
    need(len(plan['display_seeds']) >= 3 and len(set(plan['display_seeds'])) == len(plan['display_seeds']), 'AT_LEAST_THREE_FIXED_DISPLAY_SEEDS')
    candidates = read(HERE / 'pilot/CANDIDATES_FINAL.json')
    review_rows = review_records(read(HERE / 'review/REVIEWS.json'))
    need(isinstance(candidates, list) and candidates and len({c['candidate_id'] for c in candidates}) == len(candidates), 'CANDIDATE_SET_DRIFT')
    reviews = {r['candidate_id']: r for r in review_rows}
    need(len(reviews) == len(review_rows) and set(reviews) == {c['candidate_id'] for c in candidates}, 'REVIEW_SET_DRIFT')
    need(all(c['root_id'] in plan['roots'] for c in candidates), 'UNPLANNED_ROOT')
    admissions = [quality.validate_review(c, reviews.get(c['candidate_id'], {})) for c in candidates]
    save(HERE / 'review/ADMISSION_GATE.json', {'admissions': admissions})
    admitted_ids = {r['candidate_id'] for r in admissions if r['admitted']}
    need(admitted_ids, 'NO_ADMITTED_NEW_QUESTIONS')
    new = [c for c in candidates if c['candidate_id'] in admitted_ids]
    old_by_id = {c['candidate_id']: c for c in read(OLD / 'CANDIDATES_PRIVATE.json')}
    baseline = [old_by_id[cid] for cid in plan['baseline_candidates']]
    need(len({c['candidate_id'] for c in baseline}) == len(baseline) and not set(admitted_ids).intersection(plan['baseline_candidates']), 'BASELINE_OR_COHORT_ID_DRIFT')
    from transformers import AutoTokenizer
    from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
    tokenizer = AutoTokenizer.from_pretrained(str(orig.TOKENIZER), local_files_only=True)
    reasoner = Qwen3ReasoningParser(tokenizer=tokenizer)
    api = engine.cpu_api_check()
    save(OUT / 'CPU_API_CHECK.json', api)
    rows, private, token_ids, bindings, condition_details = [], [], {}, [], []
    schema = output_schema()
    for cohort, cs in [('baseline', baseline), ('new', new)]:
        for candidate in cs:
            check(candidate['source_binding'])
            source = read(candidate['source_binding']['path'])
            refs = [t['turn_ref'] for t in source['turns']]
            need(source['root_id'] == candidate['root_id'] and refs and len(refs) == len(set(refs)), 'SOURCE_ROOT_OR_REFS_DRIFT')
            need(candidate['question_type'] == 'multi_select' and len(candidate['options']) == 6 and 1 <= len(candidate['gold']) <= 5, 'PILOT_MULTISELECTION_DOMAIN')
            variants = []
            for index, seed in enumerate(plan['display_seeds']):
                public_view, presentation = quality.prepare_display(candidate, seed)
                mapping = tuple(presentation['display_to_canonical'].values())
                need(mapping not in variants, 'DUPLICATE_DISPLAY_VARIANT')
                variants.append(mapping)
                variant = f'perm_{index:02}'
                for condition, turns in conditions(candidate, source, cohort):
                    rid = f'{candidate["candidate_id"]}__{variant}__{condition}__DP1'
                    prompt = make_prompt(tokenizer, public_view['question'], public_view['options'], turns)
                    ids = tokenizer.encode(prompt, add_special_tokens=False)
                    need(len(ids) + 6144 <= 16384, 'CONTEXT_OVERFLOW_NO_TRUNCATION:'+rid)
                    need(reasoner.is_reasoning_end(ids), 'THINK_OFF_GATE_DRIFT')
                    prompt_path = OUT / 'prompts' / (rid + '.txt')
                    schema_path = OUT / 'schemas' / (rid + '.json')
                    prompt_path.parent.mkdir(parents=True, exist_ok=True)
                    with prompt_path.open('x') as handle:
                        handle.write(prompt)
                    save(schema_path, schema)
                    row = {k: public_view[k] for k in ('candidate_id', 'root_id', 'question_type', 'question')}
                    row.update(request_id=rid, base_request_id=rid, condition=condition, cohort=cohort,
                               mode='THINK_OFF', enable_thinking=False, variant_id=variant,
                               display=public_view['options'], labels=list('ABCDEF'),
                               prompt_path=str(prompt_path), schema_path=str(schema_path),
                               prompt_tokens=len(ids), sampling_seed=sampling_seed(candidate['root_id'], index))
                    rows.append(row)
                    token_ids[rid] = ids
                    private.append({'request_id': rid, 'candidate_id': candidate['candidate_id'], 'presentation': presentation,
                                    'visible_refs': [t['turn_ref'] for t in turns],
                                    'dependency_certificate': candidate.get('dependency_certificate')})
                    bindings.extend([bind(prompt_path), bind(schema_path), candidate['source_binding']])
                    condition_details.append({'request_id': rid, 'condition': condition,
                                              'source_turns': len(source['turns']), 'visible_turns': len(turns)})
    save(OUT / 'PUBLIC_REQUESTS.json', {'requests': rows})
    save(OUT / 'PROMPT_TOKEN_IDS.json', token_ids)
    save(OUT / 'CONDITION_AUDIT.json', {'requests': condition_details,
                                     'parent_definition': 'All pre-R03 turns in the cumulative source; not raw original data.'})
    save(OUT / 'PRIVATE_SCORING.json', {'requests': private})
    (OUT / 'PRIVATE_SCORING.json').chmod(0o600)
    need(0 < len(rows) <= 600 and len({r['request_id'] for r in rows}) == len(rows), 'REQUEST_COUNT_OR_DUPLICATE')
    required = [*runtime_code_paths(), HERE / 'PILOT_PLAN.json',
                HERE / 'AUTHOR_CONTRACT_ZH.md', OUT / 'PUBLIC_REQUESTS.json',
                OUT / 'PROMPT_TOKEN_IDS.json', OUT / 'CPU_API_CHECK.json', OUT / 'CONDITION_AUDIT.json',
                *[Path(b['path']) for b in api['implementation_bindings']]]
    bindings += [bind(p) for p in required]
    private_bindings = [bind(p) for p in [HERE / 'pilot/CANDIDATES_FINAL.json', HERE / 'review/REVIEWS.json',
                                       HERE / 'review/ADMISSION_GATE.json', OUT / 'PRIVATE_SCORING.json',
                                       OLD / 'CANDIDATES_PRIVATE.json']]
    save(OUT / 'PREPARATION.json', {
        'experiment_id': HERE.name, 'requests': len(rows),
        'fixed_order': [r['request_id'] for r in rows],
        'config_receipt': config, 'runtime_versions': orig.versions(),
        'bindings': list({b['path']: b for b in bindings}.values()),
        'private_bindings': private_bindings, 'structured_outputs_config': engine.STRUCTURED_CONFIG,
        'max_tokens': MAX_TOKENS, 'max_model_len': MODEL_LEN, 'batch_size': BATCH,
        'sampling': SAMPLING, 'prefix_cache': False,
        'automatic_retry': False, 'source_truncation': False, 'generation_reads_gold': False,
        'new_questions': len(new), 'baseline_questions': len(baseline),
        'max_prompt_tokens': max(r['prompt_tokens'] for r in rows)})
    print(json.dumps({'requests': len(rows), 'new_questions': len(new), 'baseline_questions': len(baseline),
                      'max_prompt_tokens': max(r['prompt_tokens'] for r in rows)}, ensure_ascii=False))

def public():
    prep = read(OUT / 'PREPARATION.json')
    forbidden = {str(p.resolve()) for p in (HERE / 'pilot/CANDIDATES_FINAL.json',
                 HERE / 'review/REVIEWS.json', HERE / 'review/ADMISSION_GATE.json',
                 OUT / 'PRIVATE_SCORING.json', OLD / 'CANDIDATES_PRIVATE.json')}
    for b in prep['bindings']:
        need(str(Path(b['path']).resolve()) not in forbidden, 'PRIVATE_INPUT_IN_GENERATION_BINDINGS')
        check(b)
    rows = read(OUT / 'PUBLIC_REQUESTS.json')['requests']
    tokens = read(OUT / 'PROMPT_TOKEN_IDS.json')
    ids = [r['request_id'] for r in rows]
    need(prep['experiment_id'] == HERE.name and 0 < len(ids) <= 600
         and len(ids) == len(set(ids)) == prep['requests'] and ids == prep['fixed_order']
         and set(tokens) == set(ids), 'PUBLIC_SET_DRIFT')
    need((prep['max_tokens'], prep['max_model_len'], prep['batch_size']) == (MAX_TOKENS, MODEL_LEN, BATCH)
         and prep['sampling'] == SAMPLING and prep['structured_outputs_config'] == engine.STRUCTURED_CONFIG,
         'PUBLIC_ENGINE_CONFIG_DRIFT')
    need(all(prep[k] is False for k in ('prefix_cache', 'automatic_retry', 'source_truncation', 'generation_reads_gold')),
         'PUBLIC_EXECUTION_POLICY_DRIFT')
    plan = read(HERE / 'PILOT_PLAN.json')
    seeds = plan['display_seeds']
    need(len(seeds) >= 3 and len(seeds) == len(set(seeds)), 'DISPLAY_SEED_PLAN_DRIFT')
    grouped, required = collections.defaultdict(list), []
    for row in rows:
        need(set(row) == PUBLIC_KEYS and row['cohort'] in ('new', 'baseline')
             and row['root_id'] in plan['roots'] and row['question_type'] == 'multi_select', 'PUBLIC_FIELD_OR_DOMAIN_DRIFT')
        need(row['mode'] == 'THINK_OFF' and row['enable_thinking'] is False
             and row['base_request_id'] == row['request_id'], 'PUBLIC_MODE_DRIFT')
        need(row['labels'] == list(row['display']) == list('ABCDEF')
             and len(set(row['display'].values())) == 6, 'PUBLIC_DISPLAY_DRIFT')
        variants = [f'perm_{i:02}' for i in range(len(seeds))]
        need(row['variant_id'] in variants, 'PUBLIC_VARIANT_DRIFT')
        index = variants.index(row['variant_id'])
        need(row['sampling_seed'] == sampling_seed(row['root_id'], index), 'PUBLIC_SEED_DRIFT')
        need(row['request_id'] == f"{row['candidate_id']}__{row['variant_id']}__{row['condition']}__DP1", 'REQUEST_ID_DRIFT')
        ids_for_row = tokens[row['request_id']]
        need(isinstance(ids_for_row, list) and all(type(x) is int and x >= 0 for x in ids_for_row)
             and len(ids_for_row) == row['prompt_tokens'] and 0 < row['prompt_tokens']
             and row['prompt_tokens'] + MAX_TOKENS <= MODEL_LEN, 'TOKEN_BUDGET_DRIFT')
        need(Path(row['prompt_path']).resolve() == (OUT / 'prompts' / (row['request_id'] + '.txt')).resolve()
             and Path(row['schema_path']).resolve() == (OUT / 'schemas' / (row['request_id'] + '.json')).resolve(), 'REQUEST_INPUT_PATH_DRIFT')
        need(read(row['schema_path']) == output_schema(), 'FROZEN_SCHEMA_DRIFT')
        required.extend((Path(row['prompt_path']), Path(row['schema_path'])))
        grouped[(row['cohort'], row['candidate_id'])].append(row)
    for (cohort, cid), group in grouped.items():
        conditions_here = {r['condition'] for r in group}
        if cohort == 'baseline':
            need(conditions_here == set(BASE_CONDITIONS) and cid in plan['baseline_candidates'], 'BASELINE_CONDITION_DRIFT')
        else:
            ablates = {c for c in conditions_here if re.fullmatch(r'ABLATE_G[1-9][0-9]*', c)}
            onlys = {c for c in conditions_here if re.fullmatch(r'ONLY_G[1-9][0-9]*', c)}
            need(len(ablates) >= 2 and ablates == {f'ABLATE_G{i+1}' for i in range(len(ablates))}
                 and onlys == {c.replace('ABLATE_', 'ONLY_') for c in ablates}
                 and conditions_here == set(NEW_FIXED_CONDITIONS) | ablates | onlys, 'DEPENDENCY_CONDITION_DRIFT')
        need(len(group) == len(conditions_here) * len(seeds)
             and {(r['condition'], r['variant_id']) for r in group}
             == {(c, f'perm_{i:02}') for c in conditions_here for i in range(len(seeds))}, 'CONDITION_VARIANT_COVERAGE_DRIFT')
        for index in range(len(seeds)):
            pair = [r for r in group if r['variant_id'] == f'perm_{index:02}']
            need(all(r['display'] == pair[0]['display'] and r['question'] == pair[0]['question']
                     and r['root_id'] == pair[0]['root_id'] and r['sampling_seed'] == pair[0]['sampling_seed'] for r in pair),
                 'CONDITION_PAIR_DRIFT')
        need(len({tuple(next(r for r in group if r['variant_id'] == f'perm_{i:02}')['display'].values())
                  for i in range(len(seeds))}) == len(seeds), 'DUPLICATE_DISPLAY_VARIANT')
    need({cid for cohort, cid in grouped if cohort == 'baseline'} == set(plan['baseline_candidates'])
         and sum(cohort == 'baseline' for cohort, cid in grouped) == prep['baseline_questions']
         and sum(cohort == 'new' for cohort, cid in grouped) == prep['new_questions'], 'COHORT_COUNT_DRIFT')
    api = read(OUT / 'CPU_API_CHECK.json')
    need(api['gpu_model_started'] is False and api['structured_outputs_config'] == engine.STRUCTURED_CONFIG
         and api['runtime_versions'] == prep['runtime_versions'], 'CPU_API_RECEIPT_DRIFT')
    required.extend([*runtime_code_paths(), HERE / 'PILOT_PLAN.json', HERE / 'AUTHOR_CONTRACT_ZH.md',
                     OUT / 'PUBLIC_REQUESTS.json', OUT / 'PROMPT_TOKEN_IDS.json',
                     OUT / 'CPU_API_CHECK.json', OUT / 'CONDITION_AUDIT.json',
                     *[Path(b['path']) for b in api['implementation_bindings']]])
    paths = [str(Path(b['path']).resolve()) for b in prep['bindings']]
    need(len(paths) == len(set(paths)) and all(bind(p) in prep['bindings'] for p in required), 'UNBOUND_PUBLIC_INPUT')
    return prep, rows, tokens

def checked_raw(frozen, rows, tokens):
    # Retain every original strict check: sets, paths, identities, submissions,
    # STARTED/preparation/model receipts, completed batches and capture recomputation.
    prior.HERE, prior.OUT = HERE, OUT
    engine.HERE, engine.OUT = HERE, OUT
    prior.check_raw(frozen, rows, tokens)
    for index, row in enumerate(rows):
        raw = read(OUT / 'raw' / (row['request_id'] + '.json'))
        if raw['status'] == 'RETURNED':
            need(type(raw.get('batch_index')) is int and raw['batch_index'] == index // BATCH, 'RAW_BATCH_INDEX_DRIFT')


def run():
    engine.HERE, engine.OUT = HERE, OUT
    engine.public, engine.checked_raw_manifest = public, checked_raw
    engine.run()

def summarize(xs):
    js = [r['J'] for r in xs if r['J'] is not None]
    return {'requests': len(xs), 'questions': len({r['candidate_id'] for r in xs}),
            'valid': sum(r['valid'] for r in xs), 'invalid': sum(not r['valid'] for r in xs), 'exact_count': sum(r['exact'] for r in xs),
            'exact_accuracy': statistics.mean(r['exact'] for r in xs) if xs else None,
            'J': statistics.mean(js) if js else None, 'J_denominator': len(js)}

def validate_private(rows, private_rows, candidates):
    need(len(private_rows) == len(rows) and len({r['request_id'] for r in private_rows}) == len(rows)
         and {r['request_id'] for r in private_rows} == {r['request_id'] for r in rows}, 'PRIVATE_REQUEST_SET_DRIFT')
    private = {r['request_id']: r for r in private_rows}
    plan = read(HERE / 'PILOT_PLAN.json')
    for row in rows:
        info, candidate = private[row['request_id']], candidates[row['candidate_id']]
        need(info['candidate_id'] == row['candidate_id'], 'PRIVATE_CANDIDATE_ID_DRIFT')
        index = int(row['variant_id'].split('_')[1])
        public_view, expected = quality.prepare_display(candidate, plan['display_seeds'][index])
        need(info['presentation'] == expected and row['display'] == public_view['options']
             and all(row[k] == public_view[k] for k in ('candidate_id', 'root_id', 'question_type', 'question')),
             'PRIVATE_PRESENTATION_OR_GOLD_DRIFT')
        check(candidate['source_binding'])
        source = read(candidate['source_binding']['path'])
        expected_conditions = dict(conditions(candidate, source, row['cohort']))
        need(row['condition'] in expected_conditions and info['visible_refs'] == [t['turn_ref'] for t in expected_conditions[row['condition']]]
             and info['dependency_certificate'] == candidate.get('dependency_certificate'), 'PRIVATE_CONTEXT_OR_CERTIFICATE_DRIFT')
    return private

def strict_json(text):
    def pairs(items):
        need(len(items) == len({key for key, _ in items}), 'DUPLICATE_JSON_KEY')
        return dict(items)
    return json.loads(text, object_pairs_hook=pairs)

def score_one(row, info, raw, schema):
    from jsonschema import validate
    need(schema == output_schema(), 'SCORING_SCHEMA_DRIFT')
    valid, errors, answer, parsed = False, [], [], None
    try:
        need(raw['request_id'] == row['request_id'] and raw['status'] == 'RETURNED'
             and raw.get('finish_reason') == 'stop', 'BACKEND_OR_LENGTH_FAILURE')
        need(not raw.get('capture_errors'), 'CAPTURE_ERRORS')
        think, final, _ = engine.split_output(raw['raw_output'])
        need(not think, 'UNEXPECTED_THINKING')
        parsed = strict_json(final)
        validate(parsed, schema)
        need(list(parsed) == ['item_evidence', 'answer_reason', 'final_answer'], 'KEY_ORDER')
        for item in parsed['item_evidence'].values():
            need(set(item['turn_refs']) <= set(info['visible_refs']), 'REFERENCE_NOT_VISIBLE')
        match = re.fullmatch(r'<box>([A-F](?: [A-F]){0,5})?</box>', parsed['final_answer'])
        need(match, 'ANSWER_FORMAT')
        answer = (match.group(1) or '').split()
        need(len(answer) == len(set(answer)) and answer == sorted(answer), 'ANSWER_ORDER_OR_DUPLICATE')
        valid = True
    except Exception as error:
        errors.append(str(error))
    presentation = info['presentation']
    prediction = sorted(presentation['display_to_canonical'][x] for x in answer) if valid else []
    gold = presentation['canonical_gold']
    exact = valid and set(prediction) == set(gold)
    tp, fp = len(set(prediction) & set(gold)), len(set(prediction) - set(gold))
    question = {'question_type': 'multi_select', 'candidate_id': row['candidate_id'],
                'options': {v: row['display'][k] for k, v in presentation['display_to_canonical'].items()}, 'gold': gold}
    request = {'question_type': 'multi_select', 'candidate_id': row['candidate_id'], 'request_id': row['request_id'],
               'variant': {'display_to_canonical': presentation['display_to_canonical'], 'options': row['display'], 'displayed_gold': presentation['display_gold']}}
    scored = prior.scorer.score_response(question, request, {'request_id': row['request_id'],
                  'raw_output': '<box>'+','.join(answer)+'</box>' if valid else None})
    j = scored['partial_score']
    if valid:
        expected_j = tp/len(gold)-fp/(6-len(gold))
        need(scored['scored'] is True and scored['canonical_prediction'] == prediction
             and scored['exact_match'] == exact and j is not None and abs(j-expected_j) < 1e-12, 'ORIGINAL_J_DRIFT')
    else:
        need(scored['scored'] is False and j is None and scored['exact_match'] is None, 'INVALID_RESPONSE_WAS_SCORED')
    return {k: row[k] for k in ('request_id', 'candidate_id', 'root_id', 'cohort', 'condition', 'variant_id')} | {
        'valid': valid, 'exact': exact, 'J': j, 'J_status': scored['scoring_status'],
        'prediction': prediction, 'gold': gold, 'TP': scored['tp'], 'FP': scored['fp'], 'FN': scored['fn'],
        'errors': errors, 'parsed': parsed, 'finish_reason': raw.get('finish_reason')}

def analyze():
    prep, rows, tokens = public()
    freeze = read(OUT / 'RAW_RESPONSE_FREEZE.json')
    checked_raw(freeze, rows, tokens)
    required_private = [HERE / 'pilot/CANDIDATES_FINAL.json', HERE / 'review/REVIEWS.json',
                        HERE / 'review/ADMISSION_GATE.json', OUT / 'PRIVATE_SCORING.json', OLD / 'CANDIDATES_PRIVATE.json']
    paths_equal(prep['private_bindings'], required_private, 'PRIVATE_BINDING_SET_DRIFT')
    for b in prep['private_bindings']:
        check(b)
    candidates = {c['candidate_id']: c for path in (HERE / 'pilot/CANDIDATES_FINAL.json', OLD / 'CANDIDATES_PRIVATE.json') for c in read(path)}
    private = validate_private(rows, read(OUT / 'PRIVATE_SCORING.json')['requests'], candidates)
    records = [score_one(row, private[row['request_id']], read(OUT / 'raw' / (row['request_id'] + '.json')),
                         read(row['schema_path'])) for row in rows]
    summaries = []
    for cohort in ('baseline', 'new'):
        for cond in sorted({r['condition'] for r in records if r['cohort'] == cohort}):
            xs = [r for r in records if (r['cohort'], r['condition']) == (cohort, cond)]
            summaries.append({'cohort': cohort, 'condition': cond, **summarize(xs)})
    per_item = []
    for cid in sorted({r['candidate_id'] for r in records}):
        for cond in sorted({r['condition'] for r in records if r['candidate_id'] == cid}):
            xs = [r for r in records if (r['candidate_id'], r['condition']) == (cid, cond)]
            per_item.append({'candidate_id': cid, 'condition': cond, **summarize(xs),
                             'permutation_same_prediction': all(r['valid'] for r in xs) and len({tuple(r['prediction']) for r in xs}) == 1})
    save(OUT / 'SCORES.json', {'records': records, 'raw_freeze': bind(OUT / 'RAW_RESPONSE_FREEZE.json')})
    paired = []
    by_key = {(r['candidate_id'], r['variant_id'], r['condition']): r for r in records}
    for row in records:
        if row['condition'] != 'FULL':
            continue
        no = by_key[(row['candidate_id'], row['variant_id'], 'NO_CONTEXT')]
        paired.append({k: row[k] for k in ('candidate_id', 'variant_id', 'cohort')} | {
            'exact_full_minus_no': int(row['exact']) - int(no['exact']),
            'J_full_minus_no': row['J'] - no['J'] if row['J'] is not None and no['J'] is not None else None})
    paired_summary = []
    for cohort in ('baseline', 'new'):
        xs = [r for r in paired if r['cohort'] == cohort]
        js = [r['J_full_minus_no'] for r in xs if r['J_full_minus_no'] is not None]
        paired_summary.append({'cohort': cohort, 'paired_requests': len(xs),
            'exact_full_minus_no': statistics.mean(r['exact_full_minus_no'] for r in xs) if xs else None,
            'J_full_minus_no': statistics.mean(js) if js else None, 'J_paired_valid_denominator': len(js)})
    review_by_id = {r['candidate_id']: r for r in review_records(read(HERE / 'review/REVIEWS.json'))}
    logical_reviews = [{'candidate_id': cid,
                        'target_option': candidates[cid]['dependency_certificate']['target_option'],
                        'groups': review_by_id[cid]['ablation_review']}
                       for cid in sorted({r['candidate_id'] for r in records if r['cohort'] == 'new'})]
    result = {'summaries': summaries, 'per_item': per_item, 'full_minus_no': paired_summary,
              'full_minus_no_pairs': paired, 'logical_ablation_review': logical_reviews,
              'requests': len(records), 'submitted': len(freeze['submitted_ids']), 'returned': freeze['returned'],
              'failure': freeze['failure'], 'wall_seconds': freeze['wall_seconds'],
              'limits': read(HERE / 'PILOT_PLAN.json')['limit'],
              'ablation_policy': 'Scores on altered contexts measure retention of the FULL-source Gold, not correctness relative to a new ablated-source Gold. Logical underdetermination is assessed separately by independent review.'}
    save(OUT / 'ANALYSIS_SUMMARY.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_item'}, ensure_ascii=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'analyze'])
    getattr(sys.modules[__name__], parser.parse_args().action)()
