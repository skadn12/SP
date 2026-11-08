import copy, itertools, json, sys, tempfile
from pathlib import Path
sys.path.insert(0, '/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/qa_dependency_repair_20261027_v1')
import pilot_runner as p
q = p.quality
checks = []
def check(name, fn):
    fn()
    checks.append({'name': name, 'status': 'PASS'})
def expect_error(fn, marker=None):
    try:
        fn()
    except Exception as e:
        if marker is not None:
            assert marker in str(e), (marker, str(e))
        return
    raise AssertionError('failure was not detected: ' + str(marker))
def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False))
def assert_true(x):
    assert x
with tempfile.TemporaryDirectory(prefix='qa-dependency-cpu-') as tmp:
    d = Path(tmp)
    source = {'root_id': 'cc-train-episode-109164', 'turns': [
        {'turn_ref': ref, 'speaker': 'speaker', 'text': txt} for ref, txt in [
            ('R01:001','The package requires a valid signature.'),
            ('R01:002','A valid signature remains necessary.'),
            ('R02:001','The route is prepared.'),
            ('R03:001','The signature is now valid.'),
            ('R03:002','The package is blue.'),
            ('R03:003','The route remains prepared.')]]}
    source_path = d/'SOURCE.json'; write(source_path, source)
    c = {'candidate_id': 'QA-CPU-01', 'root_id': source['root_id'], 'author_id':'author',
         'question_type':'multi_select', 'question':'Which conclusions follow for this package?',
         'source_binding':p.bind(source_path), 'options':{f'item_{i}':f'Conclusion {i}' for i in range(6)}, 'gold':['item_0'],
         'item_analysis':{f'item_{i}':{'should_select':i==0,'rationale':'Source-specific rationale.',
              'evidence':[{'turn_ref':'R01:001','speaker':'speaker','quote':'requires a valid signature'}]} for i in range(6)},
         'dependency_certificate':{'target_option':'item_0','premise_groups':[
             {'group_id':'G1','turn_refs':['R01:001','R01:002'],'premise':'Signature required.', 'why_necessary':'Defines prerequisite.', 'removal_effect':'Cannot infer eligibility.'},
             {'group_id':'G2','turn_refs':['R03:001'],'premise':'Signature valid.', 'why_necessary':'Satisfies prerequisite.', 'removal_effect':'Cannot infer eligibility.'}],
             'joint_inference':'Valid signature satisfies the source-defined prerequisite.', 'single_group_insufficient':True,'redundant_turn_refs':['R01:002']}}
    review = {'candidate_id':c['candidate_id'],'candidate_sha256':q.fingerprint(c),'reviewer_id':'reviewer', 'gold_blind':True,
              'decision':'PASS','independent_gold':c['gold'], 'checks':{key:{'decision':'PASS','rationale':'Independently checked.'} for key in q.REVIEW_CHECKS},
              'ablation_review':[{'group_id':g['group_id'],'removed_turn_refs':g['turn_refs'], 'target_underdetermined_after_removal':True,'rationale':'Source-specific logical judgment.'} for g in c['dependency_certificate']['premise_groups']]}
    check('one-Gold source-bound candidate and independent complete-deletion review admitted',lambda:assert_true(q.validate_review(c,review)['admitted']))
    def gate_bad(name, mutate):
        bad=copy.deepcopy(c); mutate(bad)
        check(name,lambda:assert_true(not q.validate_candidate(bad)['mechanical_pass']))
    gate_bad('duplicate refs within a group rejected',lambda x:x['dependency_certificate']['premise_groups'][0]['turn_refs'].append('R01:001'))
    gate_bad('overlapping premise groups rejected',lambda x:x['dependency_certificate']['premise_groups'][1]['turn_refs'].append('R01:001'))
    gate_bad('out-of-source-order premise refs rejected',lambda x:x['dependency_certificate']['premise_groups'][0]['turn_refs'].reverse())
    gate_bad('redundant evidence outside deletion groups rejected',lambda x:x['dependency_certificate']['redundant_turn_refs'].append('R02:001'))
    gate_bad('missing explicit redundancy audit rejected',lambda x:x['dependency_certificate'].pop('redundant_turn_refs'))
    gate_bad('unknown premise reference rejected',lambda x:x['dependency_certificate']['premise_groups'][1]['turn_refs'].append('R99:999'))
    gate_bad('source-root mismatch rejected',lambda x:x.update(root_id='wrong'))
    gate_bad('Gold-all-six control excluded',lambda x:x.update(gold=list(x['options'])))
    gate_bad('nonexact source quote rejected',lambda x:x['item_analysis']['item_0']['evidence'][0].update(quote='invented fact'))
    for label, edit in [('blind-stage not affirmed',lambda x:x.update(gold_blind=False)),('incomplete repeated-premise deletion',lambda x:x['ablation_review'][0].update(removed_turn_refs=['R01:001'])),('duplicate group review',lambda x:x['ablation_review'].append(copy.deepcopy(x['ablation_review'][0]))),('Gold mismatch',lambda x:x.update(independent_gold=['item_1']))]:
        bad=copy.deepcopy(review); edit(bad)
        check('review '+label+' rejected',lambda b=bad:assert_true(not q.validate_review(c,b)['admitted']))
    cond=dict(p.conditions(c,source,'new'))
    check('ablation removes all repeated premise refs, ONLY keeps full group, original order preserved',lambda:assert_true(
        [t['turn_ref'] for t in cond['ABLATE_G1']]==['R02:001','R03:001','R03:002','R03:003'] and
        [t['turn_ref'] for t in cond['ONLY_G1']]==['R01:001','R01:002'] and
        [t['turn_ref'] for t in cond['EVIDENCE_ONLY']]==['R01:001','R01:002','R03:001'] and len(source['turns'])==6))
    seeds=p.read(p.HERE/'PILOT_PLAN.json')['display_seeds']
    permutations=[q.prepare_display(c,s)[1]['display_to_canonical'] for s in seeds]
    check('three predeclared distinct permutations',lambda:assert_true(len({tuple(x.values()) for x in permutations})==3))
    changed=copy.deepcopy(c); changed['gold']=['item_1','item_2','item_3']
    check('display permutation independent of Gold',lambda:assert_true(all(q.prepare_display(c,s)[0]==q.prepare_display(changed,s)[0] for s in seeds)))
    def scoring_data(candidate, seed):
        public,pres=q.prepare_display(candidate,seed)
        row={k:public[k] for k in ('candidate_id','root_id','question_type','question')}
        row.update(request_id='CPU',display=public['options'],cohort='new',condition='FULL',variant_id='perm_00')
        info={'presentation':pres,'visible_refs':['R01:001']}
        return row,info
    def response(labels):
        obj={'item_evidence':{k:{'judgment':'ambiguous','turn_refs':[],'reason':'CPU diagnostic.'} for k in 'ABCDEF'},'answer_reason':'CPU diagnostic.', 'final_answer':'<box>'+' '.join(labels)+'</box>'}
        return {'request_id':'CPU','status':'RETURNED','finish_reason':'stop','capture_errors':[],'raw_output':json.dumps(obj)}
    score_count=0
    for g in range(1,6):
        cc=copy.deepcopy(c); cc['gold']=list(cc['options'])[:g]
        for seed in seeds:
            row,info=scoring_data(cc,seed)
            for n in range(7):
                for ans in itertools.combinations('ABCDEF',n):
                    scored=p.score_one(row,info,response(ans),p.output_schema())
                    pred={info['presentation']['display_to_canonical'][k] for k in ans}
                    expected=len(pred & set(cc['gold']))/g-len(pred-set(cc['gold']))/(6-g)
                    assert scored['valid'] and scored['J']==expected and scored['exact']==(pred==set(cc['gold']))
                    score_count+=1
    checks.append({'name':'original J exhaustive subsets, Gold counts 1..5, three permutations including empty and negative scores','status':'PASS','scored_responses':score_count})
    row,info=scoring_data(c,seeds[0])
    invalids=[None,'not json','{"item_evidence":{}}',response(['B','A'])['raw_output'],response(['A','A'])['raw_output']]
    for index,raw_text in enumerate(invalids):
        raw=response([]);raw['raw_output']=raw_text
        check(f'invalid response {index} unscored J=None and exact=false without assertion crash',lambda raw=raw:assert_true((lambda s:not s['valid'] and s['J'] is None and s['exact'] is False)(p.score_one(row,info,raw,p.output_schema()))))
    raw=response([]); obj=json.loads(raw['raw_output']);obj['item_evidence']['A']['turn_refs']=['HIDDEN:1'];raw['raw_output']=json.dumps(obj)
    check('hidden evidence refs invalidate response',lambda:assert_true(not p.score_one(row,info,raw,p.output_schema())['valid']))
    wrong=copy.deepcopy(info);wrong['presentation']['display_gold']=['A','B']
    check('original scorer rejects inconsistent display Gold metadata',lambda:expect_error(lambda:p.score_one(row,wrong,response([]),p.output_schema()),'displayed_gold_mismatch'))
    # Public manifests and private mapping are exercised without preparation, models or real candidates.
    original_out=p.OUT;p.OUT=d/'execution'
    plan=p.read(p.HERE/'PILOT_PLAN.json')
    cs=[]
    for cid in plan['baseline_candidates']:
        cc=copy.deepcopy(c);cc['candidate_id']=cid;cc['options']={chr(65+i):v for i,v in enumerate(c['options'].values())};cc['gold']=['A'];cs.append(('baseline',cc))
    cs.append(('new',c));rows=[];priv=[];tokens={};bindings=[]
    for cohort,cc in cs:
        for index,seed in enumerate(seeds):
            pub,pres=q.prepare_display(cc,seed)
            for condition,turns in p.conditions(cc,source,cohort):
                rid=f'{cc["candidate_id"]}__perm_{index:02}__{condition}__DP1'
                pp=p.OUT/'prompts'/(rid+'.txt');pp.parent.mkdir(parents=True,exist_ok=True);pp.write_text('CPU PROMPT')
                sp=p.OUT/'schemas'/(rid+'.json');write(sp,p.output_schema())
                rr={k:pub[k] for k in ('candidate_id','root_id','question_type','question')}
                rr.update(request_id=rid,base_request_id=rid,condition=condition,cohort=cohort,mode='THINK_OFF',enable_thinking=False,variant_id=f'perm_{index:02}',display=pub['options'],labels=list('ABCDEF'),prompt_path=str(pp),schema_path=str(sp),prompt_tokens=1,sampling_seed=p.sampling_seed(cc['root_id'],index))
                rows.append(rr);tokens[rid]=[1];bindings.extend([p.bind(pp),p.bind(sp)])
                priv.append({'request_id':rid,'candidate_id':cc['candidate_id'],'presentation':pres,'visible_refs':[t['turn_ref'] for t in turns],'dependency_certificate':cc.get('dependency_certificate')})
    write(p.OUT/'PUBLIC_REQUESTS.json',{'requests':rows});write(p.OUT/'PROMPT_TOKEN_IDS.json',tokens);write(p.OUT/'CONDITION_AUDIT.json',{})
    write(p.OUT/'CPU_API_CHECK.json',{'gpu_model_started':False,'structured_outputs_config':p.engine.STRUCTURED_CONFIG,'runtime_versions':{},'implementation_bindings':[]})
    required=[*p.runtime_code_paths(),p.HERE/'PILOT_PLAN.json',p.HERE/'AUTHOR_CONTRACT_ZH.md',*[p.OUT/n for n in ('PUBLIC_REQUESTS.json','PROMPT_TOKEN_IDS.json','CPU_API_CHECK.json','CONDITION_AUDIT.json')]]
    bindings += [p.bind(path) for path in required]
    prep={'experiment_id':p.HERE.name,'requests':len(rows),'fixed_order':[r['request_id'] for r in rows], 'bindings':list({b['path']:b for b in bindings}.values()), 'max_tokens':p.MAX_TOKENS,'max_model_len':p.MODEL_LEN,'batch_size':p.BATCH,'sampling':p.SAMPLING,'structured_outputs_config':p.engine.STRUCTURED_CONFIG,'prefix_cache':False,'automatic_retry':False,'source_truncation':False,'generation_reads_gold':False,'runtime_versions':{},'baseline_questions':6,'new_questions':1}
    write(p.OUT/'PREPARATION.json',prep)
    check('public generation manifest complete bindings and 63 condition/permutation rows',lambda:assert_true(len(p.public()[1])==63))
    def bad_prep_test(name,mutate,marker):
        bad=copy.deepcopy(prep);mutate(bad);write(p.OUT/'PREPARATION.json',bad)
        check(name,lambda:expect_error(p.public,marker));write(p.OUT/'PREPARATION.json',prep)
    bad_prep_test('missing bound prompt rejected',lambda x:x['bindings'].pop(0),'UNBOUND_PUBLIC_INPUT')
    bad_prep_test('private file in generation bindings rejected',lambda x:x['bindings'].insert(0,{'path':str(p.HERE/'pilot/CANDIDATES_FINAL.json')}),'PRIVATE_INPUT_IN_GENERATION_BINDINGS')
    byid={cc['candidate_id']:cc for _,cc in cs}
    check('private Gold and display verified against candidate and fixed seed',lambda:assert_true(len(p.validate_private(rows,priv,byid))==63))
    wrong=copy.deepcopy(priv);wrong[0]['presentation']['canonical_gold']=['B']
    check('private canonical Gold drift rejected',lambda:expect_error(lambda:p.validate_private(rows,wrong,byid),'PRIVATE_PRESENTATION_OR_GOLD_DRIFT'))
    wrong=copy.deepcopy(priv);wrong[-1]['visible_refs']=[]
    check('private visible-context/deletion drift rejected',lambda:expect_error(lambda:p.validate_private(rows,wrong,byid),'PRIVATE_CONTEXT_OR_CERTIFICATE_DRIFT'))
    # Real strict receipt/capture logic; only actual model-runtime receipt validation is stubbed.
    rawrows=rows[:2]; rawids=[r['request_id'] for r in rawrows]
    rawdir=d/'rawchecks';p.OUT=rawdir;write(p.OUT/'PREPARATION.json',{})
    started={'experiment_id':p.HERE.name,'requests':2,'request_ids':rawids,'preparation_binding':p.bind(p.OUT/'PREPARATION.json'),'structured_outputs_config':p.engine.STRUCTURED_CONFIG,'gold_read':False};write(p.OUT/'STARTED.json',started)
    write(p.OUT/'MODEL_LOAD_RECEIPT.json',{'synthetic_model_receipt':True});receipt=p.bind(p.OUT/'MODEL_LOAD_RECEIPT.json')
    real_receipt_checker=p.engine.checked_model_receipt
    p.engine.checked_model_receipt=lambda x:assert_true(x=={'synthetic_model_receipt':True})
    for rr in rawrows:
        rid=rr['request_id'];prompt=Path(rr['prompt_path']).read_text();comp={'raw_output':'{}','finish_reason':'stop','stop_reason':None,'output_token_ids':[9]}
        raw={'request_id':rid,'status':'RETURNED','mode':rr['mode'],'base_request_id':rid,'model_load_receipt_binding':receipt,'batch_index':0,'capture_errors':[],'returned_result_count':1,'returned_results':[{'prompt':prompt,'prompt_token_ids':[1],'outputs':[comp]}],'prompt_token_ids':[1],**comp}
        write(p.OUT/'raw'/(rid+'.json'),raw);write(p.OUT/'submitted'/(rid+'.json'),{'request_id':rid,'experiment_id':p.HERE.name,'model_load_receipt_binding':receipt})
    write(p.OUT/'batches/000.json',{'batch_index':0,'request_ids':rawids})
    freeze={'requests':2,'request_ids':rawids,'submitted_ids':rawids,'returned':2,'failure':None,'wall_seconds':0,
            'bindings':[p.bind(p.OUT/'raw'/(rid+'.json')) for rid in rawids], 'submission_bindings':[p.bind(p.OUT/'submitted'/(rid+'.json')) for rid in rawids], 'batch_bindings':[p.bind(p.OUT/'batches/000.json')], 'started_binding':p.bind(p.OUT/'STARTED.json'),'model_receipt_binding':receipt}
    check('strict checked_raw accepts bound complete success using real capture validation',lambda:p.checked_raw(freeze,rawrows,tokens))
    bad=copy.deepcopy(freeze);bad['bindings'].pop()
    check('raw missing binding rejected',lambda:expect_error(lambda:p.checked_raw(bad,rawrows,tokens),'RAW_BINDING_SET_DRIFT'))
    write(p.OUT/'raw/extra.json',{})
    check('raw unbound directory addition rejected',lambda:expect_error(lambda:p.checked_raw(freeze,rawrows,tokens),'RAW_DIRECTORY_SET_DRIFT'));(p.OUT/'raw/extra.json').unlink()
    bad=copy.deepcopy(freeze);bad['model_receipt_binding']=None
    check('submitted requests without model receipt rejected',lambda:expect_error(lambda:p.checked_raw(bad,rawrows,tokens),'MISSING_MODEL_RECEIPT_AFTER_SUBMISSION'))
    def mutate_bound(path, field, mutate, marker):
        before=p.read(path);after=copy.deepcopy(before);mutate(after);write(path,after);bad=copy.deepcopy(freeze)
        if isinstance(bad[field],list):
            bad[field]=[p.bind(path) if Path(b['path'])==path else b for b in bad[field]]
        else:bad[field]=p.bind(path)
        expect_error(lambda:p.checked_raw(bad,rawrows,tokens),marker);write(path,before)
    for name,mut,marker in [('identity',lambda x:x.update(request_id='WRONG'),'RAW_ID_OR_MODE_DRIFT'),('status',lambda x:x.update(status='NOT_SUBMITTED'),'RAW_STATUS_DRIFT'),('capture prompt',lambda x:x['returned_results'][0].update(prompt='DRIFT'),'CAPTURE_CHECK_DRIFT'),('batch index',lambda x:x.update(batch_index=1),'RAW_BATCH_INDEX_DRIFT'),('model receipt',lambda x:x.update(model_load_receipt_binding=None),'RAW_MODEL_RECEIPT_DRIFT')]:
        check('raw '+name+' drift rejected even after rebinding',lambda mut=mut,marker=marker:mutate_bound(p.OUT/'raw'/(rawids[0]+'.json'),'bindings',mut,marker))
    check('STARTED preparation drift rejected after rebinding',lambda:mutate_bound(p.OUT/'STARTED.json','started_binding',lambda x:x.update(gold_read=True),'STARTED_CONFIG_DRIFT'))
    check('batch request-set drift rejected after rebinding',lambda:mutate_bound(p.OUT/'batches/000.json','batch_bindings',lambda x:x.update(request_ids=rawids[::-1]),'BATCH_REQUEST_SET_DRIFT'))
    check('submission experiment drift rejected after rebinding',lambda:mutate_bound(p.OUT/'submitted'/(rawids[0]+'.json'),'submission_bindings',lambda x:x.update(experiment_id='wrong'),'SUBMISSION_RECEIPT_DRIFT'))
    p.engine.checked_model_receipt=real_receipt_checker;p.OUT=original_out
result={'status':'PASS','checks':checks,'count':len(checks),'exhaustive_scoring_responses':score_count,'gpu_model_started':False,'prepare_called':False,'real_candidate_content_read':False,'model_receipt_test_limit':'Synthetic receipt used only in isolated strict-raw success tests; frozen runtime model-receipt checker is reused unchanged in actual execution.'}
Path('/tmp/qa_dependency_cpu_check_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False))
