"""CPU-only independent raw, scoring and closeout audit for the full AB pool."""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import ast, csv, hashlib, importlib.util, json, re, sys
from pathlib import Path

N = Path(__file__).resolve().parent.parent
W = N.parent
R = W.parents[1]
DAY3 = W/'day3_solver187_v1/runtime/direct_v2'
OLD = R/'offline-deliveries/h4_codex_train_reeval_v137/evaluation'
PREV = W/'day5_base_panel_eval_v1'
DAY5 = W/'day5_training_v1'
P = DAY5/'train_diagnosis_prepare'
rd = lambda p: json.loads(Path(p).read_text())
HASHES, CHECKS = {}, []

def sha(path):
    path = Path(path).resolve()
    if path not in HASHES:
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''): h.update(block)
        HASHES[path] = h.hexdigest()
    return HASHES[path]

def bind(path):
    path=Path(path).resolve(); return {'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}

def check(ok, name, detail=None):
    CHECKS.append({'check':name,'pass':bool(ok),'detail':detail})
    if not ok: raise RuntimeError(name+':'+str(detail))

def verify(refs):
    for ref in refs:
        path=Path(ref['path'])
        check(path.is_file() and path.stat().st_size==ref['bytes'] and sha(path)==ref['sha256'],
              'BOUND_FILE_UNCHANGED',str(path))

def mean(vals):
    vals=list(vals)
    check(bool(vals),'NONEMPTY_METRIC')
    return sum(vals,Fraction())/len(vals)

def aggregate(questions, records):
    grouped=defaultdict(list)
    for rec in records: grouped[rec['candidate_id']].append(rec)
    roots=defaultdict(list); rates=[]
    for q in questions:
        rs=grouped[q['candidate_id']]
        check(len(rs)==q['denominator_calls'],'QUESTION_SCORE_DENOMINATOR',q['candidate_id'])
        rate=Fraction(sum(x['exact'] for x in rs),len(rs)); rates.append(rate); roots[q['root_id']].append(rate)
    out={'questions':len(questions),'roots':len(roots),'planned_calls':len(records),
         'valid_calls':sum(x['valid'] for x in records),'invalid_calls':sum(not x['valid'] for x in records),
         'exact_calls':sum(x['exact'] for x in records),'question_equal_exact':mean(rates),
         'root_equal_exact':mean(mean(x) for x in roots.values()),
         'call_pooled_exact_secondary':Fraction(sum(x['exact'] for x in records),len(records))}
    types={q['question_type'] for q in questions}
    if types=={'multi_select'}:
        qs=[]
        for q in questions:
            vals=[]
            for rec in grouped[q['candidate_id']]:
                pred,gold=set(rec['prediction_content'] or []),set(rec['gold_content'])
                vals.append(Fraction(len(pred&gold),len(pred|gold)) if rec['valid'] else Fraction())
            qs.append(mean(vals))
        out['question_equal_mc_jaccard_secondary']=mean(qs)
    if types=={'ordering'}:
        out['question_equal_order_pair_accuracy_secondary']=mean(mean(
            Fraction(r['metrics']['correct_pairs'],r['metrics']['total_pairs']) for r in grouped[q['candidate_id']])
            for q in questions)
    return out

def groups(questions, records):
    out={}
    for cohort in ('UNION','A','B'):
      for category in ('ALL','new_structure','foundation'):
       for typ in ('ALL','multi_select','ordering'):
        subset=[q for q in questions if (cohort=='UNION' or cohort in q['arms']) and
                (category=='ALL' or q['category']==category) and (typ=='ALL' or q['question_type']==typ)]
        if not subset: continue
        ids={q['candidate_id'] for q in subset}
        out[f'{cohort}/{category}/{typ}']=aggregate(subset,[r for r in records if r['candidate_id'] in ids])
    return out

def load_scorer():
    d=W/'day3_solver187_v1/runtime';sys.path.insert(0,str(d))
    spec=importlib.util.spec_from_file_location('independent_full_ab_score',d/'score.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    audit_path=N/'review/POST_EXECUTION_AUDIT.json'; manifest_path=N/'FINAL_MANIFEST.json'
    check(not audit_path.exists() and not manifest_path.exists(),'NO_OVERWRITE_POST_AUDIT_OR_FINAL')
    pre=rd(N/'review/PRE_EXECUTION_AUDIT.json');check(pre['status']=='PASS','PRE_AUDIT_PASS');verify(pre['audited_files'])
    prep=rd(N/'PREPARATION_FREEZE.json');em=rd(N/'EXECUTION_MANIFEST.json');cpu=rd(N/'CPU_VALIDATION.json')
    check(em['model_role']=='BASE' and em['original_base_no_adapter'] is True and em['model_directory']=='/root/autodl-tmp/Qwen3-4B','BASE_IDENTITY')
    # Verify all frozen inputs, source/prompt lineage and base reuse bindings again.
    for key in ('model_bindings','tokenizer_bindings','code_bindings','input_bindings','prompt_bindings',
                'choice_source_bindings','s0_reference_bindings','reused_base_bindings','source_bindings'): verify(prep[key])
    rows=rd(N/'public/DIRECT_REQUESTS.json')['requests'];pub=rd(N/'public/PUBLIC_QUESTIONS.json')['questions']
    membership=rd(N/'POOL_MEMBERSHIP.json');members={q['candidate_id']:q for q in membership['questions']}
    public={q['candidate_id']:q for q in pub};qids=set(public);rids={r['request_id'] for r in rows}
    finite=rd(N/'public/FINITE_CHOICES.json');maps=rd(N/'public/DISPLAY_MAPPINGS.json')
    check(len(rows)==len(rids)==2910 and len(pub)==len(qids)==200,'FIXED2910_200_DENOMINATORS')
    check(len(membership['arms']['A'])==len(set(membership['arms']['A']))==112 and
          len(membership['arms']['B'])==len(set(membership['arms']['B']))==112,'A_B_112_QUESTIONS')
    check(len(set(membership['arms']['A']) & set(membership['arms']['B']))==24,'SHARED24_COUNTED_ONCE_IN_UNION')
    newids=set(em['new_request_ids']); reuseids=set(em['reused_request_ids'])
    check(len(newids)==2550 and len(reuseids)==360 and newids.isdisjoint(reuseids) and newids|reuseids==rids,'2550_NEW_360_REUSED_PARTITION')
    rawdir=N/'direct';complete=rd(rawdir/'COMPLETE.json');rawfreeze=rd(rawdir/'RAW_RESPONSE_FREEZE.json')
    started=rd(rawdir/'STARTED.json');scoring=rd(N/'SCORING_STARTED.json')
    check(complete['status']=='COMPLETE' and complete['failure'] is None and complete['actual_submitted_requests']==2550 and
          complete['returned_requests']==2550 and complete['reused_requests']==360 and complete['total_returned_requests']==2910,
          'LIVE_COMPLETE_COUNTS')
    check(rawfreeze['status']=='COMMITTED' and rawfreeze['planned_request_count']==2910 and rawfreeze['gold_opened_by_live'] is False,
          'RAW_TERMINAL_FREEZE')
    verify(rawfreeze['files']+[rawfreeze['request_freeze'],rawfreeze['execution_manifest']])
    check(Path(rawfreeze['request_freeze']['path']).resolve()==N/'PREPARATION_FREEZE.json' and
          Path(rawfreeze['execution_manifest']['path']).resolve()==N/'EXECUTION_MANIFEST.json','FREEZE_BINDINGS_MATCH_RUN')
    check(datetime.fromisoformat(complete['completed_at_utc'])<=datetime.fromisoformat(rawfreeze['created_at_utc'])<
          datetime.fromisoformat(scoring['gold_first_read_at_utc']),'RAW_FREEZE_BEFORE_GOLD')
    verify([scoring['raw_freeze'],scoring['scoring_code']])
    expected_paths={rawdir/'responses'/(r['request_id']+'.json') for r in rows}
    check(len(rawfreeze['files'])==2910 and {Path(x['path']).resolve() for x in rawfreeze['files']}==expected_paths and
          set((rawdir/'responses').glob('*.json'))==expected_paths,'RAW_SET_EXACT2910')
    # Every new request was submitted once in order, no retries, no dropped/extra outputs.
    newrows=[r for r in rows if r['request_id'] in newids]
    submitted=sorted((rawdir/'batches').glob('*_SUBMITTED.json'));completed=sorted((rawdir/'batches').glob('*_COMPLETE.json'))
    check(len(submitted)==len(completed)==319,'319_BATCHES_FOR_2550_NEW_REQUESTS')
    check(len(list((rawdir/'batches').glob('*_EXTRA_OUTPUTS.json')))==0,'NO_EXTRA_OUTPUT_COUNT_EVENTS')
    params_by_id={};submitted_order=[]
    for i,(sp,cp) in enumerate(zip(submitted,completed)):
        sb,cb=rd(sp),rd(cp);expected=[r['request_id'] for r in newrows[i*8:(i+1)*8]]
        check(sb['request_ids']==cb['request_ids']==expected and len(sb['sampling_params'])==len(expected),'BATCH_EXACT_IDS_AND_SIZE',i)
        check(sb['automatic_retry'] is False and datetime.fromisoformat(sb['submitted_at_utc'])<=datetime.fromisoformat(cb['completed_at_utc'])<=datetime.fromisoformat(rawfreeze['created_at_utc']),'BATCH_NO_RETRY_AND_PRE_FREEZE',i)
        for rid,reprp in zip(sb['request_ids'],sb['sampling_params']): params_by_id[rid]=reprp
        submitted_order.extend(sb['request_ids'])
    check(submitted_order==[r['request_id'] for r in newrows] and len(set(submitted_order))==2550,'NO_DUPLICATE_SUBMISSION')
    # Compare complete recorded params including choice ordering with the frozen actual-array provenance.
    for row in newrows:
        rid=row['request_id'];reprp=params_by_id[rid]
        # SamplingParams continues with grammar/json fields after the finite-choice
        # array; anchor the choice capture at that field boundary instead of
        # assuming StructuredOutputsParams closes immediately after ``choice``.
        match=re.search(r"temperature=([^,]+), top_p=([^,]+), top_k=([^,]+).*?seed=([^,]+).*?stop=(\[[^]]*\]).*?include_stop_str_in_output=([^,]+).*?max_tokens=([^,]+).*?structured_outputs=StructuredOutputsParams\(json=None, regex=None, choice=(\[.*?\]), grammar=",reprp)
        check(bool(match),'SAMPLING_PARAMS_PARSE',rid)
        if not match: continue
        vals=match.groups();choice=ast.literal_eval(vals[7]);expected_choice=finite['choices'][finite['request_keys'][rid]]
        check(float(vals[0])==.8 and float(vals[1])==.9 and int(vals[3])==row['seed'] and
              ast.literal_eval(vals[4])==['</box>'] and vals[5]=='True' and int(vals[6])==32,
              'SAMPLING_SEED_AND_DECODER',rid)
        check(choice==expected_choice,'SAMPLING_FINITE_CHOICE_ORDER',rid)
    old_base_dir=PREV/'direct/responses';reuse_binding=rd(N/'REUSED_BASE_BINDINGS.json')['requests']
    raws={};identity=('request_id','candidate_id','root_id','question_type','variant_id','slot','seed','prompt_sha256')
    for row in rows:
        rid=row['request_id'];path=rawdir/'responses'/(rid+'.json');raw=rd(path)
        check(all(raw[k]==row[k] for k in identity),'RAW_REQUEST_IDENTITY',rid)
        check(raw['status']=='RETURNED' and raw['model_role']=='BASE' and raw['model_directory']=='/root/autodl-tmp/Qwen3-4B' and
              raw['branch']=='direct' and raw['gold_opened_by_solver'] is False and raw['output_validation_errors']==[],
              'RAW_BASE_VALID_STATUS',rid)
        check(len(raw['all_completions'])==1 and raw['metadata']['finish_reason'] in ('stop','eos') and
              raw['raw_output'] in finite['choices'][finite['request_keys'][rid]],'RAW_FINITE_VALID_OUTPUT',rid)
        check(datetime.fromisoformat(raw['response_committed_at_utc'])<=datetime.fromisoformat(rawfreeze['created_at_utc']),
              'RAW_BEFORE_FREEZE',rid)
        prompt=Path(row['prompt_path'])
        check(sha(prompt)==row['prompt_sha256'] and len(raw['prompt_token_ids'])==row['prompt_tokens'] and
              hashlib.sha256(json.dumps(raw['prompt_token_ids'],separators=(',',':')).encode()).hexdigest()==cpu['prompt_token_ids_sha256'][rid],
              'PROMPT_TOKEN_ID_HASH_AND_COUNT',rid)
        if rid in reuseids:
            src=Path(reuse_binding[rid]['path']);check(sha(path)==sha(src)==reuse_binding[rid]['sha256'],'REUSED_BASE_BYTE_IDENTICAL',rid)
        else:
            check(raw['metadata']['seed']==row['seed'] and raw['metadata']['temperature']==.8 and raw['metadata']['top_p']==.9 and
                  raw['metadata']['max_tokens']==32 and raw['metadata']['structured_output_api']==prep['decoder_contract']['structured_output_api'],
                  'RAW_DECODER_METADATA',rid)
        raws[rid]=raw
    # Existing frozen Day5 corpus is read-only; verify its complete original file inventory.
    old_day5=rd(DAY5/'FINAL_MANIFEST.json');check(old_day5['status']=='D5_EXECUTED_CLOSED_SECOND_WAVE_B_NO_GO' and
        old_day5['full_day5_GO'] is False and old_day5['second_wave_formal_B_GO'] is False,'OLD_DAY5_NO_GO_UNCHANGED')
    verify(old_day5['files']+old_day5['external_execution_model_artifacts'])
    check(len(old_day5['files'])==4339 and len(old_day5['external_execution_model_artifacts'])==45,'OLD_DAY5_SEALED_INVENTORY')
    prev_final=rd(PREV/'FINAL_MANIFEST.json');verify(prev_final['files']);check(prev_final['status']=='BASE_FIXED24_SUPPLEMENT_EXECUTED_AUDITED_CLOSED','PRIOR_BASE_PANEL_CLOSED_UNCHANGED')
    # Gold is first read here only after all base raws and old hashes have been verified.
    private={q['candidate_id']:q for q in rd(N/'private/GOLD.json')['questions']}
    check(set(private)==qids,'PRIVATE_GOLD_SET')
    for cid,q in public.items(): check(all(private[cid][k]==v for k,v in q.items()),'PRIVATE_PUBLIC_CONTENT',cid)
    scorer=load_scorer();bindings=rd(N/'S0_REFERENCE_BINDINGS.json')['requests']
    check(set(bindings)==rids,'S0_REFERENCE_FIXED2910')
    records={'BASE':[],'S0':[]}
    for row in rows:
        rid=row['request_id'];cid=row['candidate_id'];gold=private[cid]['gold']
        rec=scorer.score_one(row,raws[rid],maps['requests'][rid],maps['public_to_canonical'][cid],gold)
        records['BASE'].append(rec)
        ref=bindings[rid];old=rd(ref['path'])
        if ref['cohort']=='foundation_v137':
            old={**{k:row[k] for k in identity},'status':'RETURNED','raw_output':old['raw_output'],'metadata':old['metadata'],
                 'output_validation_errors':old.get('data_errors') or [],'response_committed_at_utc':rd(N/'S0_REFERENCE_BINDINGS.json')['legacy_terminal_at_utc'],
                 'legacy_conversion':'Legacy terminal fields normalized after freeze; token IDs not fabricated.'}
        else: check(old.get('prompt_token_ids')==raws[rid]['prompt_token_ids'],'S0_DAY3_EXACT_PROMPT_IDS',rid)
        records['S0'].append(scorer.score_one(row,old,maps['requests'][rid],maps['public_to_canonical'][cid],gold))
    scored={'BASE':rd(N/'BASE_SCORES.json'),'S0':rd(N/'S0_SCORES.json')}
    qmetrics={}
    for role in ('BASE','S0'):
        stored=scored[role]['records'];check(records[role]==stored,'ALL2910_REQUEST_SCORES_RECOMPUTED',role)
        questions=[scorer.summarize_question(q,[r for r in records[role] if r['candidate_id']==q['candidate_id']])|members[q['candidate_id']] for q in pub]
        check(questions==scored[role]['questions'],'ALL200_QUESTION_METRICS_RECOMPUTED',role)
        calcgroups=groups(questions,records[role]);storedgroups=rd(N/'COMPARISON.json')['models'][role]['groups']
        check(set(calcgroups)==set(storedgroups),'GROUP_SET_RECOMPUTED',role)
        for key,val in calcgroups.items():
            expected=storedgroups[key]
            check(set(val)==set(expected),'GROUP_METRIC_KEYS',role+':'+key)
            for metric,value in val.items():
                if isinstance(value,Fraction):
                    check(Fraction(expected[metric]['fraction'])==value and abs(expected[metric]['value']-float(value))<1e-12,
                          'GROUP_RATIONAL_RECOMPUTED',role+':'+key+':'+metric)
                else: check(expected[metric]==value,'GROUP_COUNT_RECOMPUTED',role+':'+key+':'+metric)
        qmetrics[role]=calcgroups
    with (N/'per_question.csv').open(newline='') as stream: csvrows=list(csv.DictReader(stream))
    check(len(csvrows)==200 and {r['candidate_id'] for r in csvrows}==qids,'CSV_200_QUESTIONS')
    for csvrow in csvrows:
        cid=csvrow['candidate_id']
        for role in ('BASE','S0'):
            q=next(x for x in scored[role]['questions'] if x['candidate_id']==cid)
            check(int(csvrow[role+'_exact_calls'])==q['exact_calls'] and Fraction(csvrow[role+'_exact_fraction'])==Fraction(q['exact_calls'],q['denominator_calls']),
                  'CSV_EXACT_QUESTION_SCORE',role+':'+cid)
    # Verify reported actual-token cost from raw responses independently.
    newraw=[raws[x] for x in newids];reusedraw=[raws[x] for x in reuseids];usage=rd(N/'COMPARISON.json')['usage']
    for label,subset in (('new_base_only',newraw),('reused_old_base_not_new_cost',reusedraw)):
        expected=usage[label]
        check(expected['planned_requests']==len(subset)==expected['returned_requests'],'USAGE_REQUEST_COUNTS',label)
        check(sum(x['metadata']['prompt_token_count'] for x in subset)==expected['input_tokens'] and
              sum(x['metadata']['completion_token_count'] for x in subset)==expected['output_tokens'],'USAGE_TOKENS_FROM_RAW',label)
    check(usage['new_base_only']['input_tokens']==18987402 and usage['new_base_only']['output_tokens']==31521,
          'EXPECTED_NEW_BASE_TOKEN_USAGE')
    check(rd(N/'COMPARISON.json')['interpretation']['not_independent_heldout_generalization'] is True and
          rd(N/'COMPARISON.json')['interpretation']['cannot_establish_feedback_causality'] is True,
          'TRAIN_POOL_LIMITATIONS_DISCLOSED')
    # Preserve a concise fully recomputed result object in the audit.
    ratio={role:{key:{m:({'fraction':str(v),'value':float(v)} if isinstance(v,Fraction) else v) for m,v in val.items()}
                 for key,val in qmetrics[role].items()} for role in ('BASE','S0')}
    audited=[bind(p) for p in sorted(N.rglob('*')) if p.is_file()]
    report={'schema_version':'full_AB_base_independent_post_execution_audit_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
       'status':'PASS','reviewer':'/root/base_panel_independent_check','checks_count':len(CHECKS),'failed_checks':0,'checks':CHECKS,
       'audited_files':audited,'scope':{'unique_questions':200,'roots':len({q['root_id'] for q in pub}),
           'A_questions':112,'B_questions':112,'shared_foundation_questions':24,'union_requests':2910,
           'new_base_requests':2550,'byte_reused_base_requests':360},
       'execution':{'new_batches':319,'expected_batch_sizes':'318 batches of 8; final batch 6','new_returns':2550,
           'reused_raws':360,'all_raws_valid':2910,'raw_frozen_before_gold':True,'sampling_params_seed_choice_checked':2550,
           'token_ids_checked':2910,'automatic_retries':0,'truncations':0,'unexpected_output_count_events':0,
           'new_input_tokens':18987402,'new_output_tokens':31521,'new_wall_seconds':complete['wall_seconds']},
       'independent_recomputation':{'scorer':'unchanged native day3_solver187_v1/runtime/score.py',
           'all_base_and_S0_records_recomputed':True,'all_group_metrics_recomputed':True,'groups_by_role':ratio},
       'old_frozen_records':{'day5_status':old_day5['status'],'day5_status_unchanged':True,
           'day5_frozen_internal_files_verified':len(old_day5['files']),'day5_external_model_files_verified':len(old_day5['external_execution_model_artifacts']),
           'prior_full_base_panel_closed':True},
       'scientific_boundary':'All200 are actual training-pool questions. This is not independent held-out generalization or causal feedback evidence.'}
    with audit_path.open('x') as stream: json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
    final_files=[bind(p) for p in sorted(N.rglob('*')) if p.is_file() and p.name!='FINAL_MANIFEST.json']
    final={'schema_version':'day5_original_base_full_AB_train_closed_manifest_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
       'status':'FULL_AB_BASE_TRAIN_POOL_EXECUTED_AUDITED_CLOSED','execution_records_closed':True,
       'questions':200,'A_questions':112,'B_questions':112,'shared_foundation_questions':24,'unique_union_questions':200,
       'base_requests':2910,'new_base_requests':2550,'byte_reused_base_requests':360,'all_base_requests_valid':True,
       'full_S0_comparison_available':True,'full_R2_comparison_available':False,
       'old_day5_status':old_day5['status'],'old_day5_GO_changed':False,'heldout_generalization':False,'feedback_causality_established':False,
       'post_execution_audit':bind(audit_path),'files_count':len(final_files),'bytes_total':sum(x['bytes'] for x in final_files),
       'files':final_files,'exclusions':['FINAL_MANIFEST.json itself.'],'base_weights_duplicated':False}
    with manifest_path.open('x') as stream:json.dump(final,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({'status':'PASS','checks':len(CHECKS),'groups':len(ratio['BASE']),'sealed_files':len(final_files),
       'sealed_bytes':final['bytes_total'],'audit_sha256':sha(audit_path),'final_manifest_sha256':sha(manifest_path),
       'base_primary':{k:v['question_equal_exact'] for k,v in ratio['BASE'].items() if k.endswith('/ALL')},
       's0_primary':{k:v['question_equal_exact'] for k,v in ratio['S0'].items() if k.endswith('/ALL')}}))

if __name__=='__main__':main()
