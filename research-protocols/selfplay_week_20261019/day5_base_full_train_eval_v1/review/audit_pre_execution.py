"""Independent CPU-only pre-inference audit for the full Day5 AB train union."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import ast, hashlib, importlib.metadata, json, re, sys
from pathlib import Path

N = Path(__file__).resolve().parent.parent
W, R = N.parent, N.parent.parents[1]
DAY3 = W / 'day3_solver187_v1/runtime/direct_v2'
OLD = R / 'offline-deliveries/h4_codex_train_reeval_v137/evaluation'
PREV = W / 'day5_base_panel_eval_v1'
SCHEDULE = W / 'day4_schedule_v1'
rd = lambda p: json.loads(Path(p).read_text())
checks, hashes = [], {}

def sha(path):
    path = Path(path).resolve()
    if path not in hashes:
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(block)
        hashes[path] = h.hexdigest()
    return hashes[path]

def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}

def check(ok, label, detail=None):
    checks.append({'check': label, 'pass': bool(ok), 'detail': detail})
    if not ok:
        raise RuntimeError(label + ':' + str(detail))

def verify(bindings):
    for ref in bindings:
        p = Path(ref['path'])
        check(p.is_file() and p.stat().st_size == ref['bytes'] and sha(p) == ref['sha256'],
              'BOUND_FILE_UNCHANGED', str(p))

def lines(path):
    return [json.loads(row) for row in Path(path).read_text().splitlines() if row.strip()]

def main():
    audit_path = N / 'review/PRE_EXECUTION_AUDIT.json'
    check(not audit_path.exists() and not (N / 'direct').exists(), 'NO_PRIOR_AUDIT_OR_BASE_RUN')
    prep, manifest, cpu = (rd(N / name) for name in ('PREPARATION_FREEZE.json','EXECUTION_MANIFEST.json','CPU_VALIDATION.json'))
    verify([manifest['preparation_freeze']])
    for key in ('model_bindings','tokenizer_bindings','code_bindings','input_bindings','prompt_bindings',
                'choice_source_bindings','s0_reference_bindings','reused_base_bindings','source_bindings'):
        verify(prep[key])
    check(manifest['model_role'] == 'BASE' and manifest['original_base_no_adapter'] is True and
          manifest['model_directory'] == prep['model_directory'] == prep['tokenizer_directory'] == '/root/autodl-tmp/Qwen3-4B',
          'ORIGINAL_BASE_AND_NO_ADAPTER')
    versions = {x: importlib.metadata.version(x) for x in ('torch','transformers','tokenizers','vllm')}
    check(prep['python_prefix'] == sys.prefix and prep['runtime_versions'] == versions,
          'INHERITED_EVALUATION_RUNTIME')

    rows = rd(N / 'public/DIRECT_REQUESTS.json')['requests']
    public = rd(N / 'public/PUBLIC_QUESTIONS.json')['questions']
    private = rd(N / 'private/GOLD.json')['questions']
    gold = {q['candidate_id']: q for q in private}
    questions = {q['candidate_id']: q for q in public}
    membership = rd(N / 'POOL_MEMBERSHIP.json')
    members = {q['candidate_id']: q for q in membership['questions']}
    selection = rd(SCHEDULE / 'SELECTION.json')
    common = set(selection['selected_foundation'])
    new = set(selection['selected_new_A'] + selection['selected_new_B'])
    check(len(common) == 24 and len(new) == 176 and len(common | new) == 200 and common.isdisjoint(new),
          'UNION_IS_200_UNIQUE_QUESTIONS')
    check(set(questions) == set(gold) == set(members) == common | new, 'PUBLIC_GOLD_MEMBERSHIP_IDENTITY')
    check(len(rows) == len({r['request_id'] for r in rows}) == 2910 and cpu['requests'] == 2910 and
          cpu['new_requests'] == 2550 and cpu['reused_requests'] == 360, 'FIXED2910_2550_360_DENOMINATORS')
    check(manifest['requests'] == 2910 and manifest['questions'] == 200 and
          set(manifest['reused_request_ids']) == set(rd(N / 'REUSED_BASE_BINDINGS.json')['requests']) and
          len(manifest['reused_request_ids']) == 360 and len(manifest['new_request_ids']) == 2550,
          'EXECUTION_MANIFEST_PARTITION')
    check(Counter(q['question_type'] for q in public) == {'multi_select':85,'ordering':115} and
          Counter(q['category'] for q in members.values()) == {'new_structure':176,'foundation':24},
          'FIXED_UNION_TYPE_AND_COHORT_COUNTS')
    check(Counter(r['cohort'] for r in rows) == {'day3':2532,'foundation_v137':378}, 'REQUEST_COHORT_COUNTS')
    check(set(rd(N / 'public/DISPLAY_MAPPINGS.json')['requests']) ==
          {r['request_id'] for r in rows} == set(rd(N / 'public/FINITE_CHOICES.json')['request_keys']) ==
          set(rd(N / 'S0_REFERENCE_BINDINGS.json')['requests']), 'ALL_REQUESTS_HAVE_MAP_CHOICES_S0_BINDING')

    old_day3 = {q['candidate_id']:q for q in rd(DAY3 / 'PUBLIC_QUESTIONS.json')['questions']}
    old_day3_gold = {q['candidate_id']:q for q in rd(DAY3.parent / 'direct_v2_private/PRIVATE_QUESTIONS.json')['questions']}
    old_v137 = {q['candidate_id']:q for q in rd(OLD / 'EVALUATION_POOL.json')['questions']}
    allowed = {'candidate_id','root_id','question_type','question','options','items','source_path','source_sha256'}
    for cid,q in questions.items():
        expected = old_day3[cid] if cid in new else {k:v for k,v in old_v137[cid].items() if k in allowed}
        check(q == expected and set(q) <= allowed and 'gold' not in q, 'CANONICAL_PUBLIC_WHITELIST', cid)
        source_gold = old_day3_gold[cid]['gold'] if cid in new else old_v137[cid]['gold']
        check(gold[cid]['gold'] == source_gold and all(gold[cid][k] == v for k,v in q.items()),
              'PRIVATE_GOLD_AND_PUBLIC_CONTENT_BINDING', cid)
        check(sha(q['source_path']) == q['source_sha256'], 'SOURCE_HASH', cid)

    day3_manifest = rd(DAY3 / 'manifest.json')
    day3_rows = {r['request_id']:r for r in day3_manifest['requests']}
    old_run = OLD / 'run/clean'
    old_manifest = rd(old_run / 'manifest.json')
    old_rows = {r['request_id']:r for r in old_manifest['requests']}
    d3_submissions = {}
    for ref in prep['choice_source_bindings']:
        batch = rd(ref['path'])
        for rid,payload in zip(batch['request_ids'],batch['sampling_params']):
            d3_submissions[rid] = ast.literal_eval(re.search(r'choice=(\[[^\]]*\])',payload).group(1)) if isinstance(payload,str) else payload['structured_outputs']['choice']
    choices = rd(N / 'public/FINITE_CHOICES.json')
    maps = rd(N / 'public/DISPLAY_MAPPINGS.json')['requests']
    s0refs = rd(N / 'S0_REFERENCE_BINDINGS.json')['requests']
    reused = rd(N / 'REUSED_BASE_BINDINGS.json')['requests']
    reused_ids = set(reused)
    sys.path.insert(0, str(W / 'day2/solver'))
    import adapter as native_adapter
    check(reused_ids == set(manifest['reused_request_ids']) and set(manifest['new_request_ids']) ==
          {r['request_id'] for r in rows} - reused_ids, 'REUSE_NEW_REQUESTS_DISJOINT_EXHAUSTIVE')
    token_hashes = cpu['prompt_token_ids_sha256']
    for row in rows:
        rid,cid = row['request_id'],row['candidate_id']
        q = questions[cid]
        field = 'options' if q['question_type'] == 'multi_select' else 'items'
        check(sum(x['candidate_id'] == cid for x in rows) == (18 if field == 'options' else 12),
              'PER_QUESTION_FIXED_REQUEST_COUNT', cid)
        check(sha(row['prompt_path']) == row['prompt_sha256'] and row['prompt_tokens'] + 32 <= 16384,
              'PROMPT_HASH_AND_CONTEXT', rid)
        arr = choices['choices'][choices['request_keys'][rid]]
        if row['cohort'] == 'day3':
            old = day3_rows[rid]
            check(all(row[k] == old[k] for k in old), 'DAY3_REQUEST_EXACT_REUSE', rid)
            check(arr == d3_submissions[rid], 'DAY3_ACTUAL_FINITE_CHOICE_ARRAY_REUSE', rid)
        else:
            old = old_rows[rid]
            check(all(row[k] == old[k] for k in ('request_id','candidate_id','root_id','question_type','slot','seed','prompt_sha256','prompt_tokens')) and
                  row['variant_id'] == old['variant']['variant_id'] and row['display'][field] == old['variant'][field],
                  'V137_REQUEST_DISPLAY_AND_SEED_REUSE', rid)
            reconstructed = native_adapter.choices(row['question_type'], row['display'][field])
            check(arr == reconstructed, 'V137_RECONSTRUCTED_FINITE_ARRAY_FROM_HASHED_NATIVE_CODE', rid)
        ref = s0refs[rid]
        check(ref['cohort'] == row['cohort'], 'S0_COHORT_BINDING', rid)
        if rid in reused_ids:
            refbase = reused[rid]
            check(refbase['sha256'] == sha(refbase['path']) and sha(refbase['path']) == sha(refbase['path']),
                  'REUSED_BASE_BINDING_HASH', rid)
            base_raw = rd(refbase['path'])
            check(base_raw['status'] == 'RETURNED' and base_raw['model_role'] == 'BASE' and
                  base_raw['model_directory'] == '/root/autodl-tmp/Qwen3-4B' and
                  all(base_raw[k] == row[k] for k in ('request_id','candidate_id','root_id','question_type','variant_id','slot','seed','prompt_sha256')),
                  'REUSED_BASE_RESPONSE_IDENTITY', rid)
            check(hashlib.sha256(json.dumps(base_raw['prompt_token_ids'],separators=(',',':')).encode()).hexdigest() == token_hashes[rid],
                  'REUSED_BASE_TOKEN_IDS', rid)
    check(len(reused_ids) == 360 and len(choices['request_keys']) == 2910, 'COMPLETE_REUSE_AND_CHOICES')

    oldbase_pool = rd(W / 'day3_mainline_readiness_v1/training/COMMON_OLD_BASE_SET.json')
    lineages = {x['candidate_id']:x for x in lines(oldbase_pool['foundation']['lineage_binding']['path']) if x['candidate_id'] in common}
    production, production_public = {}, {}
    for directory in ('day3_production_v1','day3_production_batch2_v1'):
        for record in lines(W / directory / 'exports/content_accepted.private.jsonl'):
            if record['candidate_id'] in new:
                production_public[record['candidate_id']] = rd(record['public']['path'])
                production[record['candidate_id']] = production_public[record['candidate_id']]['options_or_items']
    display_map = {(x['candidate_id'],x['epoch'],x['arm']):x for x in rd(SCHEDULE / 'DISPLAY_MAP_CHECKS.json')['records']}
    train_rows = {arm:lines(SCHEDULE / f'TRAIN_POOL_{arm}.jsonl') for arm in ('A','B')}
    label_adaptations = set()
    for arm,training in train_rows.items():
        ids = set(membership['arms'][arm])
        check(len(training) == 224 and len(ids) == 112 and ids == set((rd(SCHEDULE/'SELECTION.json')['selected_new_'+arm]) + list(common)),
              'ACTUAL_TRAINING_IDS_AND_TWO_EPOCHS', arm)
        check(Counter(members[cid]['question_type'] for cid in ids) == {'multi_select':50,'ordering':62},
              'ARM_TASK_COUNTS', arm)
        for t in training:
            cid,epoch = t['candidate_id'],t['epoch']
            q = questions[cid]
            field = 'options' if q['question_type'] == 'multi_select' else 'items'
            original = lineages[cid]['display_variant'][field] if cid in common else production[cid]
            mapping = display_map[(cid,epoch,'COMMON' if cid in common else arm)]
            actual = [original[mapping['display_to_original'][key]] for key in mapping['mapped_gold']]
            expected = [q[field][key] for key in gold[cid]['gold']]
            check(set(original.values()) == set(q[field].values()) and
                  (actual == expected if field == 'items' else set(actual) == set(expected)),
                  'TRAIN_GOLD_AND_OPTION_TEXT_MATCH_EVALUATION', arm+':'+cid+':'+str(epoch))
            source = rd(q['source_path'])
            dialogue = '\n'.join(f"[{z['turn_ref']}] {z['speaker']}: {z['text']}" for z in source['turns'])
            raw_question = production_public[cid]['question'] if cid in new else old_v137[cid]['question']
            displayed = '\n'.join(f"{key}. {original[value]}" for key,value in mapping['display_to_original'].items())
            user = t['messages'][1]['content']
            check(raw_question in user, 'TRAIN_QUESTION_TEXT_PRESENT', arm+':'+cid+':'+str(epoch))
            check(dialogue in user and displayed in user, 'TRAIN_DIALOGUE_AND_PERMUTED_OPTIONS_MATCH_EVALUATION', arm+':'+cid+':'+str(epoch))
            if raw_question != q['question']:
                check(q['question'].replace('six displayed item IDs','six letters') == raw_question,
                      'ONLY_ORDERING_OUTPUT_LABEL_ADAPTATION',cid)
                label_adaptations.add(cid)
            else:
                check(raw_question in user, 'TRAIN_QUESTION_MATCHES_EVALUATION', arm+':'+cid+':'+str(epoch))

    runner = (N / 'run_base.py').read_text()
    live = ast.get_source_segment(runner,next(x for x in ast.parse(runner).body if isinstance(x,ast.FunctionDef) and x.name == 'live'))
    check('shutil.copyfile' in live and "require(b.sha(target) == entry['sha256']" in live,
          'REUSE_BASE_IS_BYTE_IDENTICAL_COPY')
    check('GOLD.json' not in live and 'EVALUATION_POOL' not in live and 'DIRECT_SCORES' not in live,
          'LIVE_DOES_NOT_READ_PRIVATE_LABELS_OR_SCORES')
    check('OUTPUT_COUNT_MISMATCH' in live and '_EXTRA_OUTPUTS.json' in live and "'planned_request_count': 2910" in live,
          'FIXED_DENOMINATOR_AND_EXTRA_OUTPUT_RETENTION')
    check('load_adapter' not in runner and 'PeftModel' not in runner, 'NO_ADAPTER_LOAD_PATH')
    score = (N / 'score_and_report.py').read_text()
    check('Legacy foundation payloads are normalized only after the full base freeze.' in score and
          'legacy_conversion' in score and 'no token IDs fabricated' in score,
          'LEGACY_NORMALIZATION_DISCLOSED_NO_FABRICATED_TOKEN_IDS')
    check('not_independent_heldout_generalization' in score and 'cannot_establish_feedback_causality' in score,
          'SCIENTIFIC_SCOPE_REPORTED')

    refs = [N/'EXECUTION_MANIFEST.json',N/'PREPARATION_FREEZE.json',N/'CPU_VALIDATION.json',N/'POOL_MEMBERSHIP.json',
            N/'REUSED_BASE_BINDINGS.json',N/'S0_REFERENCE_BINDINGS.json',N/'public/PUBLIC_QUESTIONS.json',
            N/'public/DIRECT_REQUESTS.json',N/'public/DISPLAY_MAPPINGS.json',N/'public/FINITE_CHOICES.json',N/'private/GOLD.json']
    refs += [Path(x['path']) for x in prep['code_bindings']]
    report = {'schema_version':'full_AB_base_independent_pre_execution_audit_v1',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'PASS','reviewer':'/root/base_panel_independent_check',
        'checks_count':len(checks),'failed_checks':0,'checks':checks,'audited_files':[binding(x) for x in dict.fromkeys(refs)],
        'scope':{'unique_questions':200,'A_questions':112,'B_questions':112,'shared_foundation':24,
                 'requests':2910,'new_base_requests':2550,'byte_reused_base_requests':360},
        'content_audit':{'training_display_rows':448,'all_training_source_option_gold_text_matches_evaluation':True,
            'output_label_adapted_ordering_questions':len(label_adaptations),'adapted_candidate_ids':sorted(label_adaptations)},
        'limitations':['Foundation v137 finite-choice arrays are reconstructed from code bound by matching hashes, not actual submitted arrays.',
            'Foundation historical raw lacks full prompt token IDs; frozen prompt bytes and recorded token counts are checked without fabricating IDs.',
            'Only the 360 previous base-panel responses are byte-reused. No numerical-equivalence claim across historical/new batch composition.',
            'This evaluates the training union, not independent held-out generalization or feedback causality.'],
        'reviewer_solver_calls':0,'old_files_modified':False}
    out=N/'review/PRE_EXECUTION_AUDIT.json';out.parent.mkdir(exist_ok=True)
    with out.open('x') as stream:json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({'status':'PASS','checks':len(checks),'adapted_ordering_questions':len(label_adaptations),
        'audited_files':len(report['audited_files']),'audit':str(out),'sha256':sha(out)}))

if __name__ == '__main__': main()
