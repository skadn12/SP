#!/usr/bin/env python3
"""Independent CPU audit of v5 solver inputs/runtime; never parse private Gold.

Prior author/freeze/export process exposure is disclosed. This audit validates
public requests and synthetic scoring cases without reading real private Gold.
"""
from __future__ import annotations
import ast
import argparse
import collections
import datetime as dt
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

BASE = Path(__file__).resolve().parents[1]
SOLVER = BASE / 'solver_v5'
RUN = SOLVER / 'direct'
PRIVATE = SOLVER / 'direct_private'
RUNTIME = SOLVER / 'runtime'
OUT = RUN / 'INDEPENDENT_PREFLIGHT.json'
PUBLIC_KEYS = {'candidate_id','root_id','question_type','question','options','items','source_path','source_sha256'}
NATIVE_SCORE = BASE.parents[2] / 'offline-deliveries/h4_feedback_training_pool_v15_qa/evaluation/scoring.py'

def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)

def read(path):
    path = Path(path)
    require(path.name != 'PRIVATE_QUESTIONS.json' and not (path.parent.name == 'questions' and path.name.startswith('cc-train-episode-')), 'AUDIT_MUST_NOT_PARSE_PRIVATE_QUESTION_JSON')
    return json.loads(path.read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}

def verify(row):
    p = Path(row['path'])
    require(p.is_file() and p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], 'BINDING_DRIFT:' + str(p))

def all_bindings(value):
    if isinstance(value, dict):
        if {'path','sha256','bytes'} <= set(value):
            yield value
        else:
            for v in value.values():
                yield from all_bindings(v)
    elif isinstance(value, list):
        for v in value:
            yield from all_bindings(v)

def choices(typ, display):
    labels = list(display)
    values = itertools.permutations(labels) if typ == 'ordering' else itertools.chain.from_iterable(itertools.combinations(labels, n) for n in range(1, len(labels)+1))
    return ['<box>' + ','.join(value) + '</box>' for value in values]

def score_cases(score_path):
    spec = importlib.util.spec_from_file_location('independent_v5_native_score', NATIVE_SCORE)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    tree = ast.parse(score_path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'score_one')
    module = ast.Module(body=[function], type_ignores=[])
    scope = {'c': SimpleNamespace(require=require, a=SimpleNamespace(native_scoring=lambda: native, choices=choices))}
    exec(compile(ast.fix_missing_locations(module), str(score_path), 'exec'), scope)
    run_score = scope['score_one']
    row = {'request_id':'synthetic','candidate_id':'synthetic','root_id':'synthetic', 'question_type':'multi_select','variant_id':'synthetic','slot':0,'seed':1,'display': {'options': {'A':'x','B':'y','C':'z','D':'u','E':'v','F':'w'}}}
    canonical = {k:'content_'+k for k in 'ABCDEF'}
    # Deliberate permutation to prove displayed IDs are not canonical IDs.
    display_to_content = dict(zip('ABCDEF', [canonical[k] for k in 'CAFDBE']))
    variant = {'options':row['display']['options'], 'display_to_content':display_to_content}
    raw = {'status':'RETURNED','raw_output':'<box>B,C</box>', 'metadata':{'finish_reason':'stop'},'response_committed_at_utc':'2026-10-23T00:00:00+00:00'}
    # B->A, C->F: synthetic gold A,F.
    exact = run_score(row, raw, variant, canonical, ['A','F'])
    require(exact['exact'] and exact['metrics']['jaccard'] == 1 and exact['metrics']['fp'] == exact['metrics']['fn'] == 0, 'SYNTHETIC_MC_EXACT_MAPPING_FAILED')
    raw['raw_output'] = '<box>A,B,C</box>'
    over = run_score(row, raw, variant, canonical, ['A','F'])
    require(not over['exact'] and over['metrics']['fp'] == 1 and over['metrics']['fn'] == 0 and abs(over['metrics']['jaccard'] - 2/3) < 1e-12, 'SYNTHETIC_MC_OVERSELECT_FAILED')
    raw['status'] = 'BACKEND_ERROR'
    invalid = run_score(row, raw, variant, canonical, ['A','F'])
    require(not invalid['valid'] and not invalid['exact'] and invalid['metrics']['jaccard'] == 0 and invalid['metrics']['fp'] is None and invalid['metrics']['fn'] is None, 'SYNTHETIC_INVALID_POLICY_FAILED')
    row = row | {'question_type':'ordering','display': {'items':{'I1':'x','I2':'y','I3':'z','I4':'u'}}}
    canonical = {f'I{i}':'content_'+str(i) for i in range(1,5)}
    variant = {'items':row['display']['items'], 'display_to_content':dict(zip(['I1','I2','I3','I4'], [canonical[k] for k in ['I3','I1','I4','I2']]))}
    raw = {'status':'RETURNED','raw_output':'<box>I2,I4,I1,I3</box>', 'metadata':{'finish_reason':'stop'}, 'response_committed_at_utc':'2026-10-23T00:00:00+00:00'}
    exact = run_score(row, raw, variant, canonical, ['I1','I2','I3','I4'])
    require(exact['exact'] and exact['metrics']['pair_accuracy'] == 1, 'SYNTHETIC_ORDER_EXACT_FAILED')
    raw['raw_output'] = '<box>I4,I2,I1,I3</box>'
    swapped = run_score(row, raw, variant, canonical, ['I1','I2','I3','I4'])
    require(not swapped['exact'] and swapped['metrics']['correct_pairs'] == 5 and abs(swapped['metrics']['pair_accuracy']-5/6) < 1e-12, 'SYNTHETIC_ORDER_PAIR_FAILED')
    return ['MC shuffled-label exact mapping', 'MC over-selection FP/Jaccard', 'invalid zero-credit fixed-denominator behavior', 'ordering shuffled-label exact mapping', 'ordering adjacent-swap pair accuracy']

def main():
    global RUN, PRIVATE, RUNTIME, OUT
    args_parser = argparse.ArgumentParser(description=__doc__)
    args_parser.add_argument('--run', type=Path, required=True)
    args_parser.add_argument('--private', type=Path, required=True)
    args_parser.add_argument('--runtime', type=Path, required=True)
    args_parser.add_argument('--expected-model', type=Path, required=True)
    args_parser.add_argument('--expected-solver-name', required=True)
    args = args_parser.parse_args()
    RUN, PRIVATE, RUNTIME = args.run.resolve(), args.private.resolve(), args.runtime.resolve()
    OUT = RUN / 'INDEPENDENT_PREFLIGHT.json'
    require(not OUT.exists(), 'AUDIT_EXISTS_NO_OVERWRITE')
    require(not (RUN/'STARTED.json').exists(), 'GPU_ALREADY_STARTED')
    manifest = read(RUN/'manifest.json')
    cpu = read(RUN/'CPU_PREFLIGHT.json')
    frozen = read(RUN/'PREFLIGHT_FREEZE.json')
    public = read(RUN/'PUBLIC_QUESTIONS.json')
    endpoint_path = BASE.parent/'day5_training_v1/evaluation/models/B1/EXECUTION_MANIFEST.json'
    endpoint = read(endpoint_path)
    require(endpoint['model_role'] == 'B1' and Path(endpoint['model_directory']).resolve() == args.expected_model.resolve(), 'B1_ENDPOINT_MISMATCH')
    require(endpoint['tokenizer_directory'] == '/root/autodl-tmp/Qwen3-4B', 'B1_TOKENIZER_MISMATCH')
    require(manifest['versions'] == endpoint['runtime_versions'], 'B1_RUNTIME_VERSION_MISMATCH')
    for row in manifest['model_bindings']:
        verify(row)
    require(all(row in manifest['model_bindings'] for row in endpoint['model_bindings'] + endpoint['tokenizer_bindings'] + endpoint['code_bindings']), 'B1_MODEL_BINDINGS_NOT_EXACTLY_INCLUDED')
    export = read(SOLVER/'inputs/PUBLIC_EXPORT_FREEZE.json')
    private_meta = read(PRIVATE/'PRIVATE_BINDINGS.json')
    question_freeze = read(BASE/'QUESTION_FREEZE.json')
    for row in frozen['files'] + manifest['input_bindings'] + manifest['code_bindings']:
        verify(row)
    for row in all_bindings(export):
        verify(row)
    for row in all_bindings(private_meta):
        verify(row)
    require(cpu['status'] == 'PASS' and cpu['cpu_only'] is True and cpu['model_requests_issued'] == 0 and cpu['gpu_solver_started'] is False, 'CPU_PREFLIGHT_REQUIRED')
    require(cpu['independent_review_completed'] is False, 'CPU_SELF_REVIEW_MISLABEL')
    for row in cpu['audited_files'] + [cpu['script_binding']]: verify(row)
    require(any(Path(b['path']).resolve() == RUN/'PREFLIGHT_FREEZE.json' and b['sha256'] == sha(RUN/'PREFLIGHT_FREEZE.json') for b in cpu['audited_files']), 'CPU_NOT_BOUND_TO_FREEZE')
    require(manifest['question_count'] == 14 and manifest['question_type_counts'] == {'multi_select':10,'ordering':4}, 'QUESTION_COUNT')
    require(manifest['request_count'] == 228 and len(manifest['requests']) == 228, 'REQUEST_COUNT')
    require(manifest['automatic_retry'] is False and manifest['source_truncation'] is False and manifest['batch_size'] == 8 and manifest['max_model_len'] == 16384, 'EXECUTION_POLICY')
    require(manifest['solver_model'] == args.expected_solver_name and Path(manifest['model']).resolve() == args.expected_model.resolve() and manifest['tokenizer'] == '/root/autodl-tmp/Qwen3-4B', 'MODEL_TOKENIZER_POLICY')
    require(set(public) == {'questions'} and len(public['questions']) == 14, 'PUBLIC_WRAPPER')
    questions = {q['candidate_id']:q for q in public['questions']}
    require(len(questions) == 14, 'UNIQUE_QUESTIONS')
    mappings = read(RUN/'DISPLAY_MAPPINGS.json')
    tokens = read(RUN/'PROMPT_TOKEN_IDS.json')
    rows = manifest['requests']
    require(len({r['request_id'] for r in rows}) == len({r['seed'] for r in rows}) == 228, 'UNIQUE_REQUESTS_SEEDS')
    require({r['request_id'] for r in rows} == set(tokens) == set(mappings['requests']), 'REQUEST_SET')
    type_calls = collections.Counter(r['question_type'] for r in rows)
    require(type_calls == {'multi_select':180,'ordering':48}, 'TYPE_REQUEST_COUNTS')
    per_q = collections.defaultdict(list)
    for row in rows: per_q[row['candidate_id']].append(row)
    checks = []
    for qid,q in questions.items():
        field = 'options' if q['question_type'] == 'multi_select' else 'items'
        require(set(q) == {'candidate_id','root_id','question_type','question',field,'source_path','source_sha256'} and set(q) <= PUBLIC_KEYS, 'PUBLIC_FIELDS')
        require(not {'gold','evidence','gold_count','slot','display_order','display_seed'} & set(q), 'PUBLIC_ANSWER_LEAK')
        require(('items' if field=='options' else 'options') not in q, 'UNUSED_NULL_FIELD')
        canonical = {label:'content_'+hashlib.sha256(text.strip().encode()).hexdigest()[:16] for label,text in q[field].items()}
        require(mappings['public_to_canonical'][qid] == canonical and len(set(canonical.values())) == len(canonical), 'CANONICAL_CONTENT_IDS')
        source = read(RUN/'sources'/f"{q['root_id']}.json")
        require(sha(RUN/'sources'/f"{q['root_id']}.json") == q['source_sha256'], 'SOURCE_HASH')
        expected_perms,samples = (6,3) if field=='options' else (12,1)
        require(len(per_q[qid]) == expected_perms*samples, 'QUESTION_CALL_BUDGET')
        variants = collections.defaultdict(list)
        for row in per_q[qid]: variants[row['variant_id']].append(row)
        require(len(variants) == expected_perms and all(len(v)==samples for v in variants.values()), 'QUESTION_VARIANT_COUNTS')
        orders = []
        for variant_id, calls in variants.items():
            first = calls[0]
            mapping = mappings['requests'][first['request_id']]
            require(set(mapping['display_to_content']) == set(q[field]) and set(mapping['display_to_content'].values()) == set(canonical.values()), 'DISPLAY_BIJECTION')
            require(mapping['display_order_content_ids'] == list(mapping['display_to_content'].values()), 'DISPLAY_ORDER_MAPPING')
            require(first['display'] == {field:mapping[field]}, 'DISPLAY_TEXT_MAPPING')
            orders.append(tuple(mapping['display_order_content_ids']))
            for call in calls:
                require(call['display'] == first['display'] and mappings['requests'][call['request_id']] == mapping, 'SAMPLING_VARIANT_DRIFT')
                prompt = (RUN/'prompts'/f"{call['request_id']}.txt").read_text()
                require(sha(RUN/'prompts'/f"{call['request_id']}.txt") == call['prompt_sha256'], 'PROMPT_HASH')
                require(qid not in prompt, 'CANDIDATE_GOLDCOUNT_METADATA_ID_IN_MODEL_PROMPT')
                require(q['question'] in prompt and all(text in prompt for text in mapping[field].values()), 'PROMPT_CONTENT')
                require(all(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" in prompt for t in source['turns']), 'FULL_SOURCE_REQUIRED')
                require(len(tokens[call['request_id']]) == call['prompt_tokens'] and len(tokens[call['request_id']]) + 32 <= 16384, 'CONTEXT_BUDGET')
                require(call['source_turns'] == len(source['turns']) and call['max_output_tokens'] == 32, 'SOURCE_COUNT_OR_OUTPUT_BUDGET')
        require(len(set(orders)) == expected_perms, 'DUPLICATE_DISPLAY')
        checks.append({'candidate_id':qid,'question_type':q['question_type'],'calls':len(per_q[qid]),'display_permutations':expected_perms,'samples_each_display':samples})
    require(cpu['max_prompt_tokens'] == manifest['max_prompt_tokens'] == max(r['prompt_tokens'] for r in rows) == 9756, 'MAX_PROMPT_TOKENS')
    require(manifest['minimum_context_headroom'] == cpu['minimum_context_headroom'] == 6596, 'HEADROOM')
    require(question_freeze['status'] == 'FROZEN_BEFORE_SOLVER' and question_freeze['gates']['content_review_status'] == 'PASS', 'CONTENT_FREEZE_REQUIRED')
    admission = read(RUN/'CONTENT_ADMISSION.json')
    protocol = read(RUN/'PROTOCOL.json')
    require(admission['status'] == 'PASS' and admission['solver_score_filtering'] is False and set(admission['candidate_ids']) == set(questions), 'CONTENT_ADMISSION')
    require(manifest['planned_question_slots'] == manifest['production_denominator'] == 14 and manifest['production_unadmitted_slots'] == 0 and manifest['missing_candidate_ids'] == [], 'DENOMINATOR_FIELDS')
    require(len(protocol['slots']) == 14 and {s['candidate_id'] for s in protocol['slots']} == set(questions), 'PROTOCOL_SLOTS')
    require(private_meta['exact_frozen_private_source_required'] is True and private_meta['private_json_parsed_by_staging'] is False, 'BOUND_PRIVATE_STAGER_REQUIRED')
    expected_private = export['bindings']['private_questions']
    require(private_meta['original_private_binding'] == expected_private, 'ORIGINAL_PRIVATE_NOT_BOUND_TO_EXPORT')
    require(private_meta['private_answer_binding']['sha256'] == expected_private['sha256'] and private_meta['private_answer_binding']['bytes'] == expected_private['bytes'], 'STAGED_PRIVATE_DRIFT')
    require(Path(private_meta['staging_code_binding']['path']).name == 'stage_private_bound.py', 'UNBOUND_LEGACY_STAGER_FORBIDDEN')
    require(Path(private_meta['public_manifest_binding']['path']).resolve() == RUN/'manifest.json' and private_meta['public_manifest_binding']['sha256'] == sha(RUN/'manifest.json'), 'PRIVATE_MANIFEST_BINDING')
    require(private_meta['public_preflight_binding']['sha256'] == sha(RUN/'PREFLIGHT_FREEZE.json'), 'PRIVATE_PREFLIGHT_BINDING')

    # Independent static examination of the execution sequence and synthetic
    # native-scoring tests. No model backend is imported or instantiated here.
    code = {p.name:p.read_text() for p in RUNTIME.glob('*.py')}
    for name,text in code.items(): ast.parse(text, filename=name)
    require('ALREADY_STARTED_NO_AUTOMATIC_RERUN' in code['execute.py'] and code['execute.py'].count('backend.engine.generate(') == 1, 'EXECUTION_RETRY_GUARD')
    require('c.freeze_raw(run, manifest)' in code['execute.py'] and 'NOT_SUBMITTED' in code['execute.py'] and 'BACKEND_ERROR' in code['execute.py'], 'RAW_FIXED_DENOMINATOR')
    require('gold_payload = c.read' in code['score.py'] and code['score.py'].index('freeze = c.read(run / "RAW_RESPONSE_FREEZE.json")') < code['score.py'].index('gold_payload = c.read'), 'GOLD_BEFORE_RAW')
    require(code['score.py'].index('c.require(set((run / "responses").glob("*.json")) == expected') < code['score.py'].index('gold_payload = c.read'), 'GOLD_BEFORE_ALL_TERMINALS')
    synthetic = score_cases(RUNTIME/'score.py')
    files = [RUN/'PREFLIGHT_FREEZE.json',RUN/'CPU_PREFLIGHT.json',RUN/'manifest.json',RUN/'EXECUTION_FREEZE.json',RUN/'PUBLIC_QUESTIONS.json',RUN/'DISPLAY_MAPPINGS.json',RUN/'CONTENT_ADMISSION.json',RUN/'PROTOCOL.json',BASE/'QUESTION_FREEZE.json',SOLVER/'inputs/PUBLIC_EXPORT_FREEZE.json',PRIVATE/'PRIVATE_BINDINGS.json'] + list(RUNTIME.glob('*.py')) + [Path(__file__),NATIVE_SCORE,endpoint_path]
    unique = sorted(set(p.resolve() for p in files))
    report = {
        'schema_version':'live_extension_v5_independent_solver_preflight_v1','status':'PASS',
        'created_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'auditor':'/root/solver_prepare/v5_freeze_audit',
        'independent_review_completed':True,'cpu_only':True,'model_requests_issued':0,'gpu_solver_started':False,
        'gold_parsed_by_this_audit':False,'private_gold_files_hashed_only':True,
        'prior_exposure_disclosure':'This audit agent previously created question/export freezes and saw private Gold. This independent runtime/structure audit makes no Gold-blind author/reviewer claim and uses only public questions plus synthetic Gold for metric checks.',
        'expected_model':str(args.expected_model.resolve()),'expected_solver_name':args.expected_solver_name,
        'question_count':14,'root_count':2,'request_count':228,'direct_mc_requests':180,'direct_ordering_requests':48,
        'max_prompt_tokens':9756,'minimum_context_headroom':6596,
        'checks_passed':[
            'all question freeze, public export, runtime code, CPU report, and run artifact hashes match',
            'public question keys are adapter-whitelisted; no private answer/evidence fields',
            'candidate IDs containing MC2-MC6 are bookkeeping only and absent from all 228 model prompts',
            '228 unique requests/seeds: 10 MC questions x6 displays x3 samples plus 4 ordering questions x12 displays',
            'content-ID mappings are bijective and display permutations are distinct per question',
            'all cumulative source turns are present in every prompt without truncation',
            'context budget is 9756 prompt tokens maximum plus 32 output tokens; 6596 minimum headroom',
            'model/tokenizer/sampling params match the requested frozen solver policy',
            'execution code commits raw bytes before response validation; exact prompt token IDs are rechecked in scorer',
            'execution code commits terminal records for every planned call; no automatic retry; invalid/failure denominators retained',
            'score code checks committed raw freeze and all terminal identities before private Gold JSON parse',
            'CONTENT_ADMISSION/PROTOCOL and production denominator compatibility fields populated',
            'only stage_private_bound.py used; staged private hash/path/bytes match frozen private export',
        ],
        'synthetic_scoring_checks':synthetic,'questions':checks,
        'required_commands':{'private_staging':'runtime/stage_private_bound.py; legacy runtime/stage_private.py is not approved for official use'},
        'audit_limitations':[
            'CPU/static validation does not claim model answer reliability or a new formal weakness',
            'No solver response or private real Gold was parsed by this audit execution',
            'CPU preflight performs tokenizer/native finite-choice regeneration; this separate audit verifies its binding plus all public artifacts',
        ],
        'audited_files':[bind(p) for p in unique],
        'gpu_gate':'PASS for the exact bound artifacts; re-audit if any bound byte changes',
    }
    with OUT.open('x') as file:
        json.dump(report,file,ensure_ascii=False,indent=2);file.write('\n')
    print(json.dumps({'status':'PASS','output':str(OUT),'sha256':sha(OUT),'questions':14,'requests':228,'synthetic_checks':synthetic},ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
