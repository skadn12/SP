"""Single-choice/ordering diagnostic: CPU prepare, one frozen-engine run, CPU analysis.

Only this branch's candidates/reviews are admitted. Generation reads frozen public
prompts/token IDs/schemas, never private Gold/source/review artifacts. Prior code
is reused for the checkpoint, capture and complete raw receipt validation only.
"""
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

prior = load('single_order_prior_capture_runner', OLD / 'runner.py')
quality = load('single_order_quality_gate', HERE / 'quality_gate.py')
engine, orig = prior.engine, prior.orig
read, bind, check, need, save = prior.read, prior.bind, prior.check, prior.need, prior.save
MAX_TOKENS, MODEL_LEN, BATCH = 6144, 16384, 8
SAMPLING = {'temperature': .8, 'top_p': .9, 'top_k': -1, 'repetition_penalty': 1, 'engine_seed': 0}
PUBLIC_KEYS = {'candidate_id', 'root_id', 'question_type', 'question', 'request_id',
               'base_request_id', 'condition', 'cohort', 'mode', 'enable_thinking',
               'variant_id', 'display', 'labels', 'prompt_path', 'schema_path',
               'prompt_tokens', 'sampling_seed'}
FIXED_CONDITIONS = ('FULL', 'NO_CONTEXT', 'PARENT_ONLY', 'EXTENSION_ONLY', 'EVIDENCE_ONLY')

def runtime_code_paths():
    return [Path(__file__), HERE / 'quality_gate.py', OLD / 'runner.py', prior.OLD / 'solver_runner.py',
            prior.SCORER_PATH, Path(engine.__file__), quality.LEGACY_PATH, *orig.code_paths()]

def protocol_paths():
    return [HERE / name for name in ('PILOT_PLAN.json', 'AUTHOR_CONTRACT_ZH.md', 'ANALYSIS_PLAN.json')]

def paths_equal(bindings, expected, code):
    paths = [str(Path(b['path']).resolve()) for b in bindings]
    need(len(paths) == len(set(paths)) and set(paths) == {str(Path(p).resolve()) for p in expected}, code)

def review_records(value):
    rows = value.get('reviews') if isinstance(value, dict) else value
    need(isinstance(rows, list) and all(isinstance(row, dict) for row in rows), 'REVIEW_CONTAINER_DRIFT')
    return rows

def sampling_seed(candidate_id):
    # One item has one sampling seed across ALL conditions AND display permutations.
    material = f'{candidate_id}:single_order_dependency_v1'.encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], 'big') % 2147483647

def output_schema():
    # No uniqueItems: both installed grammar backends must compile this exact schema.
    # Answer cardinality, ordering completeness and ref compliance are scored separately.
    return {'type': 'object', 'properties': {
        'final_answer': {'type': 'string', 'pattern': '^<box>(?:[A-F](?: [A-F]){0,5})?</box>$'},
        'reason': {'type': 'string', 'minLength': 1, 'maxLength': 800},
        'turn_refs': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 8}},
        'required': ['final_answer', 'reason', 'turn_refs'], 'additionalProperties': False}


def cpu_schema_preflight(tokenizer, schema):
    """Actually validate and compile with both installed vLLM backends on CPU."""
    import torch
    from types import SimpleNamespace
    from vllm import SamplingParams, sampling_params
    from vllm.config import StructuredOutputsConfig
    from vllm.v1.structured_output import backend_guidance, backend_xgrammar
    from vllm.v1.structured_output.backend_types import StructuredOutputOptions
    import xgrammar, llguidance, llguidance.hf
    need(not torch.cuda.is_initialized(), 'CPU_SCHEMA_CHECK_STARTED_WITH_CUDA')
    config = StructuredOutputsConfig(**engine.STRUCTURED_CONFIG)
    hf_config = read(orig.MODEL / 'config.json')
    vocab_size = hf_config.get('text_config', hf_config)['vocab_size']
    backend_config = SimpleNamespace(structured_outputs_config=config, speculative_config=None)
    params = SamplingParams(temperature=.8, top_p=.9, top_k=-1, repetition_penalty=1.,
        max_tokens=MAX_TOKENS, seed=0, skip_special_tokens=False,
        structured_outputs=sampling_params.StructuredOutputsParams(json=copy.deepcopy(schema)))
    backend_xgrammar.validate_xgrammar_grammar(params)
    xbackend = backend_xgrammar.XgrammarBackend(vllm_config=backend_config, tokenizer=tokenizer, vocab_size=vocab_size)
    xcompiled = xbackend.compile_grammar(StructuredOutputOptions.JSON, json.dumps(schema))
    gbackend = backend_guidance.GuidanceBackend(vllm_config=backend_config, tokenizer=tokenizer, vocab_size=vocab_size)
    backend_guidance.validate_guidance_grammar(params, tokenizer=gbackend.ll_tokenizer)
    gcompiled = gbackend.compile_grammar(StructuredOutputOptions.JSON, json.dumps(schema))
    need(not torch.cuda.is_initialized(), 'CPU_SCHEMA_CHECK_INITIALIZED_CUDA')
    modules = (backend_xgrammar, backend_guidance, xgrammar, llguidance, llguidance.hf)
    return {'status': 'PASS_CPU_BOTH_BACKENDS_VALIDATED_AND_COMPILED',
            'schema_sha256': quality.fingerprint(schema), 'model_vocab_size': vocab_size,
            'tokenizer_length': len(tokenizer), 'cuda_initialized': False, 'gpu_model_started': False,
            'backends': {'xgrammar': type(xcompiled).__name__, 'guidance': type(gcompiled).__name__},
            'implementation_bindings': [bind(Path(module.__file__)) for module in modules]}


def make_prompt(tokenizer, question_type, question, display, turns):
    dialogue = '\n'.join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in turns) or '[No dialogue supplied]'
    task = ('Select exactly one option label.' if question_type == 'single_choice' else
            'Return every option label exactly once, in the complete order requested by the question.')
    instruction = (
        task + ' Give your best prediction even when the supplied information is incomplete. '
        'If you cannot make a prediction, use <box></box>; this is an abstention and is not a complete answer. '
        'Put the answer in final_answer as <box>A</box> for one label or <box>B D A C</box> for an order; '
        'examples show format only. Separate labels with one space. Give a concise reason and up to eight '
        'turn_refs that actually appear in the supplied dialogue. Use an empty turn_refs list when no '
        'source evidence is available. Do not invent evidence. Return only JSON with keys '
        'final_answer, reason, turn_refs. These instructions apply equally to all inputs.')
    messages = [
        {'role': 'system', 'content': 'Answer the question using the supplied information and give your best final prediction.'},
        {'role': 'user', 'content': f'Dialogue:\n{dialogue}\n\nQuestion:\n{question}\n\nOptions:\n' +
         '\n'.join(f'{key}. {text}' for key, text in display.items()) + '\n\n' + instruction}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def conditions(candidate, source):
    turns = source['turns']
    result = [('FULL', turns), ('NO_CONTEXT', []),
              ('PARENT_ONLY', [t for t in turns if not t['turn_ref'].startswith('R03:')]),
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


def private_paths(plan):
    return [HERE / 'pilot/CANDIDATES_FINAL.json', HERE / 'review/REVIEWS.json',
            HERE / 'review/ADMISSION_GATE.json', OUT / 'PRIVATE_SCORING.json',
            *[Path(b['path']) for b in plan['source_bindings'] + plan['warning_bindings']]]


def check_plan(plan):
    need(len(plan['display_seeds']) == 3 and len(set(plan['display_seeds'])) == 3, 'THREE_FIXED_DISPLAY_SEEDS')
    need(plan['planned_candidates'] == len(plan['attempts']) == 6 and len(set(plan['roots'])) == 3,
         'PLANNED_ATTEMPTS_DRIFT')
    expected = {(root, kind) for root in plan['roots'] for kind in ('single_choice', 'ordering')}
    need({(a['root_id'], a['question_type']) for a in plan['attempts']} == expected, 'ATTEMPT_DESIGN_DRIFT')
    need(len(plan['source_bindings']) == len(plan['warning_bindings']) == len(plan['roots']), 'SOURCE_WARNING_COUNT_DRIFT')
    return expected


def prepare():
    need(not (OUT / 'PREPARATION.json').exists(), 'NO_PREPARATION_OVERWRITE')
    _, config = orig.endpoint_receipt()
    plan = read(HERE / 'PILOT_PLAN.json')
    expected_attempts = check_plan(plan)
    for b in plan['source_bindings'] + plan['warning_bindings']:
        check(b)
    candidates = read(HERE / 'pilot/CANDIDATES_FINAL.json')
    review_rows = review_records(read(HERE / 'review/REVIEWS.json'))
    need(isinstance(candidates, list) and candidates and all(isinstance(c, dict) for c in candidates)
         and len({c['candidate_id'] for c in candidates}) == len(candidates), 'CANDIDATE_SET_DRIFT')
    need(len(candidates) <= plan['planned_candidates']
         and len({(c['root_id'], c['question_type']) for c in candidates}) == len(candidates)
         and {(c['root_id'], c['question_type']) for c in candidates} <= expected_attempts, 'UNPLANNED_OR_DUPLICATE_ATTEMPT')
    reviews = {r['candidate_id']: r for r in review_rows}
    need(len(reviews) == len(review_rows) and set(reviews) == {c['candidate_id'] for c in candidates}, 'REVIEW_SET_DRIFT')
    for c in candidates:
        need(c['source_binding'] in plan['source_bindings'], 'UNPLANNED_SOURCE_BINDING')
    admissions = [quality.validate_review(c, reviews[c['candidate_id']]) for c in candidates]
    need(read(HERE / 'review/ADMISSION_GATE.json') == {'admissions': admissions}, 'FROZEN_ADMISSION_GATE_DRIFT')
    admitted = [c for c, a in zip(candidates, admissions) if a['admitted']]
    need(admitted, 'NO_ADMITTED_QUESTIONS')
    from transformers import AutoTokenizer
    from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
    tokenizer = AutoTokenizer.from_pretrained(str(orig.TOKENIZER), local_files_only=True)
    reasoner = Qwen3ReasoningParser(tokenizer=tokenizer)
    api = engine.cpu_api_check()
    api['schema_compilation'] = cpu_schema_preflight(tokenizer, output_schema())
    api['implementation_bindings'].extend(api['schema_compilation']['implementation_bindings'])
    save(OUT / 'CPU_API_CHECK.json', api)
    rows, private, token_ids, bindings, details = [], [], {}, [], []
    for candidate in admitted:
        check(candidate['source_binding'])
        source = read(candidate['source_binding']['path'])
        refs = [t['turn_ref'] for t in source['turns']]
        need(source['root_id'] == candidate['root_id'] and refs and len(refs) == len(set(refs)), 'SOURCE_ROOT_OR_REFS_DRIFT')
        for index, (public_view, presentation) in enumerate(quality.prepare_displays(candidate, plan['display_seeds'])):
            variant = f'perm_{index:02}'
            for condition, turns in conditions(candidate, source):
                rid = f'{candidate["candidate_id"]}__{variant}__{condition}__SO1'
                prompt = make_prompt(tokenizer, candidate['question_type'], public_view['question'], public_view['options'], turns)
                ids = tokenizer.encode(prompt, add_special_tokens=False)
                need(len(ids) + MAX_TOKENS <= MODEL_LEN, 'CONTEXT_OVERFLOW_NO_TRUNCATION:'+rid)
                need(reasoner.is_reasoning_end(ids), 'THINK_OFF_GATE_DRIFT')
                prompt_path, schema_path = OUT / 'prompts' / (rid + '.txt'), OUT / 'schemas' / (rid + '.json')
                prompt_path.parent.mkdir(parents=True, exist_ok=True)
                with prompt_path.open('x', encoding='utf-8') as handle:
                    handle.write(prompt)
                save(schema_path, output_schema())
                row = {k: public_view[k] for k in ('candidate_id', 'root_id', 'question_type', 'question')}
                row.update(request_id=rid, base_request_id=rid, condition=condition, cohort='new',
                           mode='THINK_OFF', enable_thinking=False, variant_id=variant,
                           display=public_view['options'], labels=list(public_view['options']),
                           prompt_path=str(prompt_path), schema_path=str(schema_path),
                           prompt_tokens=len(ids), sampling_seed=sampling_seed(candidate['candidate_id']))
                rows.append(row)
                token_ids[rid] = ids
                private.append({'request_id': rid, 'candidate_id': candidate['candidate_id'], 'presentation': presentation,
                                'visible_refs': [t['turn_ref'] for t in turns],
                                'dependency_certificate': candidate['dependency_certificate']})
                bindings.extend([bind(prompt_path), bind(schema_path)])
                details.append({'request_id': rid, 'condition': condition, 'source_turns': len(refs),
                                'visible_turns': len(turns), 'sampling_seed': row['sampling_seed']})
    save(OUT / 'PUBLIC_REQUESTS.json', {'requests': rows})
    save(OUT / 'PROMPT_TOKEN_IDS.json', token_ids)
    save(OUT / 'CONDITION_AUDIT.json', {'requests': details,
         'parent_definition': 'All pre-R03 turns in cumulative source, not raw original dataset.',
         'same_sampling_seed_all_permutations_and_conditions_per_item': True})
    save(OUT / 'PRIVATE_SCORING.json', {'requests': private, 'admissions': admissions,
         'missing_attempts': [a for a in plan['attempts'] if (a['root_id'], a['question_type']) not in
                              {(c['root_id'], c['question_type']) for c in candidates}]})
    (OUT / 'PRIVATE_SCORING.json').chmod(0o600)
    need(0 < len(rows) <= 600 and len({r['request_id'] for r in rows}) == len(rows), 'REQUEST_COUNT_OR_DUPLICATE')
    required = [*runtime_code_paths(), *protocol_paths(), OUT / 'PUBLIC_REQUESTS.json',
                OUT / 'PROMPT_TOKEN_IDS.json', OUT / 'CPU_API_CHECK.json', OUT / 'CONDITION_AUDIT.json',
                *[Path(b['path']) for b in api['implementation_bindings']]]
    bindings.extend(bind(p) for p in required)
    save(OUT / 'PREPARATION.json', {
        'experiment_id': HERE.name, 'requests': len(rows), 'fixed_order': [r['request_id'] for r in rows],
        'config_receipt': config, 'runtime_versions': orig.versions(),
        'bindings': list({b['path']: b for b in bindings}.values()),
        'private_bindings': list({str(p.resolve()): bind(p) for p in private_paths(plan)}.values()),
        'structured_outputs_config': engine.STRUCTURED_CONFIG,
        'max_tokens': MAX_TOKENS, 'max_model_len': MODEL_LEN, 'batch_size': BATCH, 'sampling': SAMPLING,
        'prefix_cache': False, 'automatic_retry': False, 'source_truncation': False, 'generation_reads_gold': False,
        'admitted_questions': len(admitted), 'candidate_questions': len(candidates), 'planned_questions': 6,
        'sampling_seed_policy': 'one candidate_id seed shared across every condition and permutation',
        'max_prompt_tokens': max(r['prompt_tokens'] for r in rows)})
    print(json.dumps({'requests': len(rows), 'admitted_questions': len(admitted),
                      'max_prompt_tokens': max(r['prompt_tokens'] for r in rows)}, ensure_ascii=False))


def public():
    prep, plan = read(OUT / 'PREPARATION.json'), read(HERE / 'PILOT_PLAN.json')
    check_plan(plan)
    forbidden = {str(p.resolve()) for p in private_paths(plan)}
    for b in prep['bindings']:
        need(str(Path(b['path']).resolve()) not in forbidden, 'PRIVATE_INPUT_IN_GENERATION_BINDINGS')
        check(b)
    rows, tokens = read(OUT / 'PUBLIC_REQUESTS.json')['requests'], read(OUT / 'PROMPT_TOKEN_IDS.json')
    ids = [r['request_id'] for r in rows]
    need(prep['experiment_id'] == HERE.name and 0 < len(ids) <= 600
         and len(ids) == len(set(ids)) == prep['requests'] and ids == prep['fixed_order'] and set(tokens) == set(ids), 'PUBLIC_SET_DRIFT')
    need((prep['max_tokens'], prep['max_model_len'], prep['batch_size']) == (MAX_TOKENS, MODEL_LEN, BATCH)
         and prep['sampling'] == SAMPLING and prep['structured_outputs_config'] == engine.STRUCTURED_CONFIG, 'PUBLIC_ENGINE_CONFIG_DRIFT')
    need(all(prep[k] is False for k in ('prefix_cache', 'automatic_retry', 'source_truncation', 'generation_reads_gold')), 'PUBLIC_EXECUTION_POLICY_DRIFT')
    grouped, required = collections.defaultdict(list), []
    for row in rows:
        need(set(row) == PUBLIC_KEYS and row['cohort'] == 'new' and row['root_id'] in plan['roots']
             and row['question_type'] in ('single_choice', 'ordering'), 'PUBLIC_FIELD_OR_DOMAIN_DRIFT')
        n = len(row['labels'])
        need((row['question_type'] == 'single_choice' and n == 4) or
             (row['question_type'] == 'ordering' and 4 <= n <= 6), 'PUBLIC_OPTION_COUNT_DRIFT')
        need(row['mode'] == 'THINK_OFF' and row['enable_thinking'] is False
             and row['base_request_id'] == row['request_id'], 'PUBLIC_MODE_DRIFT')
        need(row['labels'] == list(row['display']) == list('ABCDEF')[:n]
             and len(set(row['display'].values())) == n, 'PUBLIC_DISPLAY_DRIFT')
        need(row['variant_id'] in ('perm_00', 'perm_01', 'perm_02')
             and row['sampling_seed'] == sampling_seed(row['candidate_id']), 'PUBLIC_SEED_OR_VARIANT_DRIFT')
        need(row['request_id'] == f"{row['candidate_id']}__{row['variant_id']}__{row['condition']}__SO1", 'REQUEST_ID_DRIFT')
        ids_for_row = tokens[row['request_id']]
        need(isinstance(ids_for_row, list) and all(type(x) is int and x >= 0 for x in ids_for_row)
             and len(ids_for_row) == row['prompt_tokens'] and 0 < row['prompt_tokens']
             and row['prompt_tokens'] + MAX_TOKENS <= MODEL_LEN, 'TOKEN_BUDGET_DRIFT')
        need(Path(row['prompt_path']).resolve() == (OUT / 'prompts' / (row['request_id'] + '.txt')).resolve()
             and Path(row['schema_path']).resolve() == (OUT / 'schemas' / (row['request_id'] + '.json')).resolve(), 'REQUEST_INPUT_PATH_DRIFT')
        need(read(row['schema_path']) == output_schema(), 'FROZEN_SCHEMA_DRIFT')
        required.extend((Path(row['prompt_path']), Path(row['schema_path'])))
        grouped[row['candidate_id']].append(row)
    for cid, group in grouped.items():
        conds = {r['condition'] for r in group}
        ablates = {c for c in conds if re.fullmatch(r'ABLATE_G[1-9][0-9]*', c)}
        onlys = {c for c in conds if re.fullmatch(r'ONLY_G[1-9][0-9]*', c)}
        need(len(ablates) >= 2 and ablates == {f'ABLATE_G{i+1}' for i in range(len(ablates))}
             and onlys == {c.replace('ABLATE_', 'ONLY_') for c in ablates}
             and conds == set(FIXED_CONDITIONS) | ablates | onlys, 'DEPENDENCY_CONDITION_DRIFT')
        need(len(group) == len(conds) * 3 and {(r['condition'], r['variant_id']) for r in group}
             == {(c, f'perm_{i:02}') for c in conds for i in range(3)}, 'CONDITION_VARIANT_COVERAGE_DRIFT')
        need(len({r['sampling_seed'] for r in group}) == 1 and len({r['question_type'] for r in group}) == 1
             and len({r['question'] for r in group}) == 1 and len({r['root_id'] for r in group}) == 1, 'ITEM_PAIR_DRIFT')
        for i in range(3):
            pair = [r for r in group if r['variant_id'] == f'perm_{i:02}']
            need(all(r['display'] == pair[0]['display'] for r in pair), 'CONDITION_DISPLAY_DRIFT')
        need(len({tuple(next(r for r in group if r['variant_id'] == f'perm_{i:02}')['display'].values()) for i in range(3)}) == 3,
             'DUPLICATE_DISPLAY_VARIANT')
    need(len(grouped) == prep['admitted_questions'] <= prep['candidate_questions'] <= prep['planned_questions'] == 6, 'QUESTION_COUNT_DRIFT')
    need(len({(g[0]['root_id'], g[0]['question_type']) for g in grouped.values()}) == len(grouped), 'DUPLICATE_ATTEMPT')
    api = read(OUT / 'CPU_API_CHECK.json')
    need(api['schema_compilation']['status'] == 'PASS_CPU_BOTH_BACKENDS_VALIDATED_AND_COMPILED'
         and api['schema_compilation']['schema_sha256'] == quality.fingerprint(output_schema())
         and api['schema_compilation']['cuda_initialized'] is False, 'SCHEMA_COMPILATION_RECEIPT_DRIFT')
    need(api['gpu_model_started'] is False and api['structured_outputs_config'] == engine.STRUCTURED_CONFIG
         and api['runtime_versions'] == prep['runtime_versions'], 'CPU_API_RECEIPT_DRIFT')
    required += [*runtime_code_paths(), *protocol_paths(), OUT / 'PUBLIC_REQUESTS.json', OUT / 'PROMPT_TOKEN_IDS.json',
                 OUT / 'CPU_API_CHECK.json', OUT / 'CONDITION_AUDIT.json', *[Path(b['path']) for b in api['implementation_bindings']]]
    paths_equal(prep['bindings'], required, 'PUBLIC_BINDING_SET_DRIFT')
    return prep, rows, tokens


def checked_raw(frozen, rows, tokens):
    # Keep all frozen checks for paths/sets/STARTED/model/submission/batches/capture.
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


def validate_private(rows, private_rows, candidates):
    need(len(private_rows) == len(rows) and len({r['request_id'] for r in private_rows}) == len(rows)
         and {r['request_id'] for r in private_rows} == {r['request_id'] for r in rows}, 'PRIVATE_REQUEST_SET_DRIFT')
    private, plan = {r['request_id']: r for r in private_rows}, read(HERE / 'PILOT_PLAN.json')
    displays = {cid: quality.prepare_displays(c, plan['display_seeds']) for cid, c in candidates.items()}
    for row in rows:
        info, candidate = private[row['request_id']], candidates[row['candidate_id']]
        need(info['candidate_id'] == row['candidate_id'], 'PRIVATE_CANDIDATE_ID_DRIFT')
        index = int(row['variant_id'].split('_')[1])
        view, expected = displays[row['candidate_id']][index]
        need(info['presentation'] == expected and row['display'] == view['options']
             and all(row[k] == view[k] for k in ('candidate_id', 'root_id', 'question_type', 'question')), 'PRIVATE_PRESENTATION_OR_GOLD_DRIFT')
        check(candidate['source_binding'])
        source = read(candidate['source_binding']['path'])
        expected_conditions = dict(conditions(candidate, source))
        need(row['condition'] in expected_conditions and info['visible_refs'] == [t['turn_ref'] for t in expected_conditions[row['condition']]]
             and info['dependency_certificate'] == candidate['dependency_certificate'], 'PRIVATE_CONTEXT_OR_CERTIFICATE_DRIFT')
    return private


def strict_json(text):
    def pairs(items):
        need(len(items) == len({key for key, _ in items}), 'DUPLICATE_JSON_KEY')
        return dict(items)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=lambda x: (_ for _ in ()).throw(ValueError('NONFINITE_JSON_NUMBER')))


def lcs_length(a, b):
    prev = [0] * (len(b)+1)
    for x in a:
        now = [0]
        for i, y in enumerate(b):
            now.append(prev[i]+1 if x == y else max(prev[i+1], now[-1]))
        prev = now
    return prev[-1]


def score_one(row, info, raw, schema):
    from jsonschema import validate
    need(schema == output_schema(), 'SCORING_SCHEMA_DRIFT')
    parsed, answer, prediction = None, [], []
    response_errors, answer_errors, reference_errors, structure_errors = [], [], [], []
    final, response_valid, schema_valid, reference_compliant, answer_valid = None, False, False, False, False
    abstention, duplicates, missing, unknown = False, [], [], []
    try:
        need(raw['request_id'] == row['request_id'] and raw['status'] == 'RETURNED'
             and raw.get('finish_reason') == 'stop', 'BACKEND_OR_LENGTH_FAILURE')
        need(not raw.get('capture_errors'), 'CAPTURE_ERRORS')
        think, final, _ = engine.split_output(raw['raw_output'])
        need(not think, 'UNEXPECTED_THINKING')
        parsed = strict_json(final)
        need(isinstance(parsed, dict), 'JSON_OBJECT_REQUIRED')
        response_valid = True
    except Exception as error:
        response_errors.append(str(error))
    if response_valid:
        try:
            validate(parsed, schema)
            schema_valid = True
        except Exception as error:
            structure_errors.append(str(error))
        # Reference and reason/schema failures cannot erase an otherwise readable answer.
        try:
            refs = parsed.get('turn_refs')
            need(isinstance(refs, list) and all(isinstance(x, str) for x in refs), 'TURN_REFS_NOT_STRING_LIST')
            need(len(refs) <= 8, 'TOO_MANY_TURN_REFERENCES')
            need(len(refs) == len(set(refs)), 'DUPLICATE_TURN_REFERENCE')
            need(set(refs) <= set(info['visible_refs']), 'REFERENCE_NOT_VISIBLE')
            reference_compliant = True
        except Exception as error:
            reference_errors.append(str(error))
        try:
            boxed = parsed.get('final_answer')
            need(isinstance(boxed, str), 'ANSWER_NOT_STRING')
            match = re.fullmatch(r'<box>([A-Z](?: [A-Z])*)?</box>', boxed)
            need(match is not None, 'ANSWER_FORMAT')
            answer = (match.group(1) or '').split()
            abstention = not answer
            duplicates = sorted({x for x in answer if answer.count(x) > 1})
            unknown = sorted(set(answer) - set(row['labels']))
            missing = [x for x in row['labels'] if x not in answer] if row['question_type'] == 'ordering' else []
            # Preserve any decoded in-domain prediction even when incomplete/invalid.
            prediction = [info['presentation']['display_to_canonical'][x] for x in answer if x in row['labels']]
            if abstention: answer_errors.append('ABSTENTION_EMPTY_BOX')
            if duplicates: answer_errors.append('DUPLICATE_ANSWER_LABELS')
            if unknown: answer_errors.append('UNKNOWN_ANSWER_LABELS')
            if row['question_type'] == 'single_choice' and len(answer) != 1: answer_errors.append('SINGLE_CHOICE_REQUIRES_ONE_LABEL')
            if row['question_type'] == 'ordering' and (len(answer) != len(row['labels']) or missing): answer_errors.append('INCOMPLETE_ORDERING')
            answer_valid = not answer_errors
        except Exception as error:
            answer_errors.append(str(error))
    else:
        answer_errors.append('RESPONSE_UNREADABLE_OR_BACKEND_INVALID')
    gold = info['presentation']['canonical_gold']
    exact = answer_valid and prediction == gold
    concordant, total_pairs, pairwise, lcs = None, None, None, None
    if row['question_type'] == 'ordering' and answer_valid:
        n, pred_pos = len(gold), {x: i for i, x in enumerate(prediction)}
        total_pairs = n * (n-1) // 2
        concordant = sum(pred_pos[gold[i]] < pred_pos[gold[j]] for i in range(n) for j in range(i+1, n))
        pairwise, lcs = concordant/total_pairs, lcs_length(prediction, gold)/n
    target = info['dependency_certificate'].get('target_option') if row['question_type'] == 'single_choice' else info['dependency_certificate'].get('target_pair')
    target_retained = None
    if answer_valid:
        target_retained = prediction == [target] if row['question_type'] == 'single_choice' else prediction.index(target[0]) < prediction.index(target[1])
    return {k: row[k] for k in ('request_id', 'candidate_id', 'root_id', 'question_type', 'cohort', 'condition', 'variant_id')} | {
        'answer_valid': answer_valid, 'valid': answer_valid, 'exact': exact, 'prediction': prediction, 'gold': gold,
        'display_prediction': answer, 'abstention': abstention, 'duplicate_labels': duplicates,
        'missing_labels': missing, 'unknown_labels': unknown, 'answer_errors': answer_errors,
        'response_valid': response_valid, 'response_errors': response_errors,
        'schema_valid': schema_valid, 'structure_errors': structure_errors,
        'reference_compliant': reference_compliant, 'reference_errors': reference_errors,
        'concordant_pairs': concordant, 'all_pairs': total_pairs, 'pairwise_accuracy': pairwise,
        'normalized_lcs': lcs, 'partial_status': 'SCORED_COMPLETE_ORDER' if pairwise is not None else
                          ('UNSCORED_INVALID_ORDER' if row['question_type'] == 'ordering' else 'NOT_APPLICABLE_SINGLE_CHOICE'),
        'target': target, 'full_gold_target_retained': target_retained,
        'parsed': parsed, 'final_text': final, 'finish_reason': raw.get('finish_reason'),
        'output_tokens': len(raw.get('output_token_ids', []))}


def mean(xs):
    return statistics.mean(xs) if xs else None


def summarize(xs):
    pairwise = [r['pairwise_accuracy'] for r in xs if r['pairwise_accuracy'] is not None]
    lcs = [r['normalized_lcs'] for r in xs if r['normalized_lcs'] is not None]
    return {'requests': len(xs), 'questions': len({r['candidate_id'] for r in xs}),
            'answer_valid': sum(r['answer_valid'] for r in xs), 'answer_invalid': sum(not r['answer_valid'] for r in xs),
            'abstentions': sum(r['abstention'] for r in xs), 'duplicate_answers': sum(bool(r['duplicate_labels']) for r in xs),
            'incomplete_orders': sum(bool(r['missing_labels']) for r in xs),
            'exact_count': sum(r['exact'] for r in xs), 'exact_accuracy': mean([int(r['exact']) for r in xs]),
            'reference_compliant': sum(r['reference_compliant'] for r in xs), 'schema_valid': sum(r['schema_valid'] for r in xs),
            'answer_valid_reference_noncompliant': sum(r['answer_valid'] and not r['reference_compliant'] for r in xs),
            'pairwise_accuracy': mean(pairwise), 'normalized_lcs': mean(lcs), 'partial_denominator': len(pairwise),
            'full_gold_target_retained': mean([int(r['full_gold_target_retained']) for r in xs if r['full_gold_target_retained'] is not None]),
            'target_valid_denominator': sum(r['full_gold_target_retained'] is not None for r in xs)}


def analyze():
    prep, rows, tokens = public()
    freeze = read(OUT / 'RAW_RESPONSE_FREEZE.json')
    checked_raw(freeze, rows, tokens)
    plan = read(HERE / 'PILOT_PLAN.json')
    paths_equal(prep['private_bindings'], private_paths(plan), 'PRIVATE_BINDING_SET_DRIFT')
    for b in prep['private_bindings']:
        check(b)
    candidates = {c['candidate_id']: c for c in read(HERE / 'pilot/CANDIDATES_FINAL.json')}
    reviews = {r['candidate_id']: r for r in review_records(read(HERE / 'review/REVIEWS.json'))}
    admissions = [quality.validate_review(c, reviews[c['candidate_id']]) for c in candidates.values()]
    need(read(HERE / 'review/ADMISSION_GATE.json') == {'admissions': admissions}, 'ANALYSIS_ADMISSION_DRIFT')
    need({a['candidate_id'] for a in admissions if a['admitted']} == {r['candidate_id'] for r in rows}, 'ANALYSIS_ADMITTED_SET_DRIFT')
    private = validate_private(rows, read(OUT / 'PRIVATE_SCORING.json')['requests'], candidates)
    records = [score_one(row, private[row['request_id']], read(OUT / 'raw' / (row['request_id'] + '.json')), read(row['schema_path'])) for row in rows]
    summaries, per_item = [], []
    for kind in ('single_choice', 'ordering'):
        for cond in sorted({r['condition'] for r in records if r['question_type'] == kind}):
            xs = [r for r in records if (r['question_type'], r['condition']) == (kind, cond)]
            summaries.append({'question_type': kind, 'condition': cond, **summarize(xs)})
    for cid in sorted({r['candidate_id'] for r in records}):
        for cond in sorted({r['condition'] for r in records if r['candidate_id'] == cid}):
            xs = [r for r in records if (r['candidate_id'], r['condition']) == (cid, cond)]
            per_item.append({'candidate_id': cid, 'question_type': xs[0]['question_type'], 'condition': cond, **summarize(xs),
                             'permutation_same_prediction': all(r['answer_valid'] for r in xs) and len({tuple(r['prediction']) for r in xs}) == 1})
    by_key = {(r['candidate_id'], r['variant_id'], r['condition']): r for r in records}
    paired = []
    for full in records:
        if full['condition'] != 'FULL': continue
        for cond in sorted({r['condition'] for r in records if r['candidate_id'] == full['candidate_id']} - {'FULL'}):
            other = by_key[(full['candidate_id'], full['variant_id'], cond)]
            paired.append({k: full[k] for k in ('candidate_id', 'question_type', 'variant_id')} | {
                'comparison': 'FULL_MINUS_'+cond, 'exact_delta': int(full['exact'])-int(other['exact']),
                'pairwise_delta': full['pairwise_accuracy']-other['pairwise_accuracy'] if full['pairwise_accuracy'] is not None and other['pairwise_accuracy'] is not None else None,
                'lcs_delta': full['normalized_lcs']-other['normalized_lcs'] if full['normalized_lcs'] is not None and other['normalized_lcs'] is not None else None,
                'same_prediction': full['answer_valid'] and other['answer_valid'] and full['prediction'] == other['prediction']})
    logical = [{'candidate_id': cid, 'question_type': candidates[cid]['question_type'],
                'target': candidates[cid]['dependency_certificate'].get('target_option', candidates[cid]['dependency_certificate'].get('target_pair')),
                'groups': reviews[cid]['ablation_review']} for cid in sorted({r['candidate_id'] for r in records})]
    result = {'summaries': summaries, 'per_item': per_item, 'full_minus_condition_pairs': paired,
              'logical_ablation_review': logical, 'requests': len(records), 'submitted': len(freeze['submitted_ids']),
              'returned': freeze['returned'], 'failure': freeze['failure'], 'wall_seconds': freeze['wall_seconds'],
              'limits': plan['limits'], 'permutations_are_independent_n': False,
              'ablation_policy': 'All altered-context scores retain the FULL-source Gold only as answer-retention diagnostics. Logical target underdetermination comes from independent source review.',
              'reference_policy': 'Final-answer scores are independent of reference and reason/schema compliance from the first analysis.',
              'ordering_partial_policy': 'Only complete valid permutations receive pairwise concordance/allpairs and LCS/n. Invalid answers count incorrect for exact and are unscored for partial.'}
    save(OUT / 'SCORES.json', {'records': records, 'raw_freeze': bind(OUT / 'RAW_RESPONSE_FREEZE.json')})
    save(OUT / 'ANALYSIS_SUMMARY.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('per_item', 'full_minus_condition_pairs')}, ensure_ascii=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'analyze'])
    getattr(sys.modules[__name__], parser.parse_args().action)()
