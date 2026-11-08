"""Read-only independent v134 preflight. No live inference, training, or QA mutation."""
import collections,copy,datetime,hashlib,importlib.util,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
B=Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_codex_held_roots_eval_v134')
E=B/'evaluation';REPO=B.parent.parent
CHECKS=[];HASH={}
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 p=Path(p).resolve();st=p.stat();key=(str(p),st.st_size,st.st_mtime_ns)
 if key not in HASH:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
  HASH[key]=h.hexdigest()
 return HASH[key]
def check(x,message):
 if not x:raise AssertionError(message)
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def verify(doc,base):
 rows=read(doc)['files'];check(bool(rows),'empty freeze '+str(doc))
 for r in rows:
  p=Path(r['path']);p=p if p.is_absolute() else base/p
  check(p.is_file() and sha(p)==r['sha256'],'bound file changed '+str(p))
  if 'bytes' in r:check(p.stat().st_size==r['bytes'],'bound byte size changed')
 return len(rows)
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def record(name,detail):CHECKS.append({'check':name,'status':'PASS','detail':detail});print(name,'PASS',flush=True)
freezes={}
for path in [B/'MASTER_FREEZE.json',B/'DEV_QA_CLOSED.json',B/'QA_FREEZE.json',B.parent/'h4_codex_held_roots_retry_v133/COMPLETE_MANIFEST.json',E/'TOOLS_FREEZE.json',E/'PREFLIGHT_FREEZE.json']:
 freezes[str(path)]=verify(path,path.parent)
record('all_transitive_input_and_code_freezes_hashes',freezes)
check(sha(E/'PREFLIGHT_FREEZE.json')=='6b7f5c54e657165c04099514c8c3fe49d6f5e0b31bdb2a723875580ece1b8326','coder preflight unexpectedly changed')
check(not (E/'RUN_STARTED.json').exists(),'inference already started')
for side in ['base','clean','new']:
 check(not (E/'run'/side/'STARTED.json').exists(),'side already started')
 check(not list((E/'run'/side/'responses').glob('*.json')),'raw responses already exist')
record('all_qa_and_test_closed_before_solver',{'qa_closed':read(B/'QA_FREEZE.json')['created_at_utc'],'dev_closed':read(B/'DEV_QA_CLOSED.json')['created_at_utc'],'preflight':read(E/'PREFLIGHT_FREEZE.json')['created_at_utc'],'no_started_or_responses':True})
w=module('v134_independent_qa_bridge',B/'qa_workflow.py')
sys.path.insert(0,str(B/'scripts'))
from evidence_export import export_root_questions
manifest=read(B/'ROOT_MANIFEST.json')['roots'];all_q=[];root_rows=[];review_binds=[]
for split,expected_count in [('dev',35),('test',14)]:
 delivered=read(B/'delivery'/split/'QUESTIONS_WITH_ANSWERS.json')['questions'];check(len(delivered)==expected_count,'split count')
 original_path=B.parent/'h4_codex_dev_test_ab_v124/delivery'/split/'QUESTIONS_WITH_ANSWERS.json';old=read(original_path)['questions'];extended=read(B/'delivery'/split/'EXTENDED_QUESTIONS_WITH_ANSWERS.json')
 check(extended['old_pool_sha256']==sha(original_path),'old pool hash')
 check(extended['questions']==old+delivered,'old/new extended losslessness')
 check(len(old)==(48 if split=='dev' else 87),'old fixed questions')
 check(not {q['candidate_id'] for q in old}&{q['candidate_id'] for q in delivered},'old/new ID intersection')
 for row in [r for r in manifest if r['split']==split]:
  rid=row['root_id'];f=w.folder(rid);u=w.unit(rid);src=w.ready(f);turns={t['turn_ref']:t for t in src['turns']}
  qq=export_root_questions(f)
  expected=[q for q in delivered if q['root_id']==rid]
  for q in qq:q.update(split=split,evaluation_only=True,solver_evaluated=False,baseline_conflicts_resolved=False,question_evidence_scope=list(w.SCENES))
  check(qq==expected,'native lossless export differs from delivery '+rid)
  check(len(qq)==7,'root question count')
  counts=collections.Counter('ordering' if q['question_type']=='ordering' else 'gold'+str(len(q['gold'])) for q in qq)
  check(counts==collections.Counter({'ordering':2,'gold2':1,'gold3':1,'gold4':1,'gold5':1,'gold6':1}),'root quota')
  names=['ORDERING_ANCHOR_PLAN.json','QA_DRAFT.json','PUBLIC_ASSETS.json','ASSET_SOURCE_JUDGMENTS.json','PUBLIC_QUESTIONS.json','PUBLIC_SOURCE_JUDGMENTS.json','PRIVATE_GOLD.json','FINAL_CROSSCHECK.json']
  times=[(f/n).stat().st_mtime_ns for n in names];check(times==sorted(times),'QA saving order '+rid)
  check(read(f/'AUTHOR_QA_RECEIPT.json')['actual_author']==row['author'],'actual author mismatch')
  check(read(f/'AUTHOR_QA_RECEIPT.json')['attempt']==1 and read(f/'AUTHOR_QA_RECEIPT.json')['semantic_repairs']==0,'semantic retry')
  for stage,reviewname,pubname in [('assets','ASSET_SOURCE_JUDGMENTS.json','PUBLIC_ASSETS.json'),('public','PUBLIC_SOURCE_JUDGMENTS.json','PUBLIC_QUESTIONS.json')]:
   rv=read(f/reviewname);ctx=read(u/'reads'/stage/'READ_CONTEXT.json')
   check(rv['actual_reviewer']==row['reviewer'] and rv['actual_reviewer']!=row['author'],'non-author reviewer')
   check(ctx['actual_reader']==row['reviewer'] and ctx['private_answers_exposed'] is False,'read identity or visibility')
   check(rv['turn_refs_read']==ctx['turn_refs_read']==[t['turn_ref'] for t in src['turns']],'full source coverage')
   check(rv['full_source_read'] is True and rv['private_answers_read_before_save'] is False and rv['independent_of_question_author'] is True,'review visibility claims')
   check(rv['quarantine_compliance']['status']=='PASS' and rv['baseline_conflicts_resolved'] is False,'quarantine attestation')
   check(not any(x['severity']=='BLOCKING' for x in rv.get('findings',[])),'blocking final source')
   paths=[Path(x['path']) for x in ctx['files']]
   check(len(paths)==5 and {p.name for p in paths}=={'CUMULATIVE_SOURCE.json','SAFE_PLAN.json','HISTORICAL_CONTINUATION_CONSTRAINTS.json','PROTOCOL_ZH.md',pubname},'review read context leaks private input')
   for rr in ctx['files']:check(sha(rr['path'])==rr['sha256'],'review input changed')
   check((f/pubname).stat().st_mtime_ns<=(u/'reads'/stage/'READ_CONTEXT.json').stat().st_mtime_ns<=(f/reviewname).stat().st_mtime_ns,'read before review order')
   if split=='test':check(ctx['created_at_utc']>read(B/'DEV_QA_CLOSED.json')['created_at_utc'],'TEST opened before DEV closure')
   review_binds.append(f/reviewname)
  cross=read(f/'FINAL_CROSSCHECK.json')['results'];check(len(cross)==7 and all(x['status']=='CONTENT_PASS' and x['original_gold_unchanged'] is True for x in cross),'content results filtering')
  for q in qq:
   ev=q['option_evidence' if q['question_type']=='multi_select' else 'item_evidence'];scenes=[]
   for label,j in ev.items():
    w.citations(j,src);ref=j['turn_ref'];check(turns[ref]['scene_ref'] in w.SCENES,'primary outside new scope');scenes.append(turns[ref]['scene_ref'])
    if q['question_type']=='multi_select':check(j['relation']==('supports' if label in q['gold'] else 'contradicts'),'wrong support or contradiction relation')
    else:check(j['order_anchor']['turn_ref'] in j['support_refs'],'order anchor support mismatch')
   check(set(scenes)==set(w.SCENES) and len(scenes)==6,'one fact/event from each new scene')
  root_rows.append({'root_id':rid,'split':split,'content_pass':len(qq),'counts':dict(counts),'source_turns':len(src['turns']),'source_sha256':sha(f/'CUMULATIVE_SOURCE.json')})
 all_q+=delivered
check(read(B/'EVALUATION_POOL.json')['questions']==all_q==read(E/'EVALUATION_POOL.json')['questions'],'global/new eval pool differs')
record('qa_provenance_source_first_order_and_evidence_scope',{'roots':root_rows,'questions':len(all_q),'main_gold2345':28,'gold6_controls':7,'ordering':14,'no_current_score_selection':True,'same_model_shared_filesystem_limit_retained':True})
record('old_pool_extended_losslessly',{'dev_old':48,'dev_new':35,'dev_extended':83,'test_old':87,'test_new':14,'test_extended':101,'evaluation_new_only':49})

# All display/randomization functions are compared with the untouched native v15.
ev=module('v134_independent_eval',E/'evaluate.py');runner=ev.native();adapter=module('v134_independent_protocol',E/'adapter_protocol.py')
check((E/'adapter_protocol.py').read_text()==ev.protocol_text(),'provenance adapter changes beyond exact allowed substitutions')
shared=read(E/'SHARED_REQUESTS.json');requests=shared['requests'];scope=read(E/'SCOPE.json');pool={q['candidate_id']:q for q in all_q}
check(len(requests)==798 and len({r['request_id'] for r in requests})==798,'request denominator')
check(shared['seed']==2026100800 and shared['policy_version']==ev.POLICY,'policy seed')
check(scope['calls_per_model']==798 and scope['total_planned_calls']==2394,'scope denominator')
check(len(scope['fixed_full_heldout_roots']['dev'])==15 and len(scope['fixed_full_heldout_roots']['test'])==16,'heldout denominator')
train={r['root_id'] for r in read(ev.TRAIN_ROOTS)};held=set(sum(scope['fixed_full_heldout_roots'].values(),[]));check(len(train)==68 and len(held)==31 and train.isdisjoint(held),'train/heldout overlap')
variant_by_candidate={};request_counts=collections.Counter();splitcounts=collections.Counter()
for q in all_q:
 n=6 if q['question_type']=='multi_select' else 12
 vv=adapter.make_variants(q,seed=2026100800,count=n)
 poison=copy.deepcopy(q);poison['gold']=['PRIVATE_SENTINEL'];poison['option_evidence']={'PRIVATE_SENTINEL':'leak'};poison['item_evidence']={'PRIVATE_SENTINEL':'leak'};poison['gold_content_ids']=['PRIVATE_SENTINEL']
 check(vv==adapter.make_variants(poison,seed=2026100800,count=n),'Gold-dependent displays')
 check(len({tuple(v['display_order_content_ids']) for v in vv})==n,'duplicate permutation')
 variant_by_candidate[q['candidate_id']]={v['variant_id']:v for v in vv}
for i,r in enumerate(requests):
 q=pool[r['candidate_id']];request_counts[q['candidate_id']]+=1;splitcounts[q['split']]+=1
 check(r['seed']==2026100800+i and r['mode']=='full_context','seed/full context')
 check(r['variant']==variant_by_candidate[q['candidate_id']][r['variant']['variant_id']],'native variant mismatch')
 check(r['slot'] in (range(3) if q['question_type']=='multi_select' else range(1)),'rollout slot')
for q in all_q:check(request_counts[q['candidate_id']]==(18 if q['question_type']=='multi_select' else 12),'question planned calls')
check(dict(splitcounts)=={'dev':570,'test':228},'split requests')
record('paired_gold_blind_native_random_request_plan',{'calls_per_model':798,'calls_all_models':2394,'dev_calls_per_model':570,'test_calls_per_model':228,'unique_display_prompts':378,'mcq': '6 distinct permutations x 3 rollouts','ordering':'12 distinct permutations x 1 rollout','gold_evidence_perturbation_invariant_all_49':True})

# Independent prompt reconstruction uses only public question fields, full source and display text.
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained('/root/autodl-tmp/Qwen3-4B',local_files_only=True)
probe=runner.load_probe('v134_independent_full_context_probe');sources={r['root_id']:w.source(r['root_id']) for r in manifest};token_cache={};source_counts={}
for side,model in ev.MODELS.items():
 run=E/'run'/side;mf=read(run/'manifest.json');check(mf['requests']==requests and mf['decoder_contract']==shared['decoder_contract'],'models not paired')
 check(mf['solver_model']==model and mf['model']==read(run/'solver_identity.json')['model_path'],'identity manifest')
 for name in ['questions.json','gold.json','canonical_bindings.json']:check(read(run/name)==read(E/'run/base'/name),'paired private/public input mismatch')
 for root,source in sources.items():check((run/'sources'/(root+'.json')).read_bytes()==Path(w.row(root)['source_path']).read_bytes(),'run source truncated')
 gold=read(run/'gold.json')
 for r in requests:
  q=pool[r['candidate_id']];field='options' if q['question_type']=='multi_select' else 'items';promptfile=run/'prompts'/(r['request_id']+'.txt');text=promptfile.read_text();check(sha(promptfile)==r['prompt_sha256'],'prompt sha')
  if r['prompt_sha256'] not in token_cache:
   public={'question_type':q['question_type'],'question':q['question']};field='options' if q['question_type']=='multi_select' else 'items';pv={field:r['variant'][field]};dialogue=probe.contract.dialogue_rows(sources[q['root_id']])
   expected=probe.contract.build_prompt(tokenizer,public,pv,dialogue,context_mode='full_context')
   check(text==expected,'public full-context prompt differs from frozen text')
   tokens=len(tokenizer.encode(text,add_special_tokens=False));check(tokens==r['prompt_tokens'] and tokens+32<=16384,'token count or budget')
   token_cache[r['prompt_sha256']]=tokens
  else:check(token_cache[r['prompt_sha256']]==r['prompt_tokens'],'shared token count')
  ids=['content_'+hashlib.sha256(q[field][l].strip().encode()).hexdigest()[:16] for l in q['gold']]
  check(gold[q['candidate_id']]==ids,'display-independent gold content binding')
check(len(token_cache)==378 and max(token_cache.values())==8239,'prompt summary')
record('all_2394_prompt_files_public_only_and_full_context',{'unique_reconstructed_prompts':378,'all_prompt_files_checked':2394,'minimum_prompt_tokens':min(token_cache.values()),'maximum_prompt_tokens':max(token_cache.values()),'mean_unique_prompt_tokens':sum(token_cache.values())/len(token_cache),'output_reserve':32,'context_limit':16384,'minimum_margin_after_output_reserve':16384-max(token_cache.values())-32})

# Model identities are immutable audited merged checkpoints, not fresh adapters.
identities={};checkpoint_paths=set()
for side,model in ev.MODELS.items():
 identity=read(E/'run'/side/'solver_identity.json');check(identity['model_identifier']==model,'model identity label')
 for key in ['audit_path','tokenizer_audit_path']:
  if identity.get(key):check(sha(identity[key])==identity[key.replace('_path','_sha256')],'model historical audit')
 for r in identity['checkpoint_files']:
  p=Path(r['path']);check(sha(p)==r['sha256'] and p.stat().st_size==r['bytes'],'checkpoint hash mismatch '+str(p));checkpoint_paths.add(str(p))
 if side!='base':check(identity['adapter_enabled'] is True and identity['adapter_already_merged'] is True,'unmerged identity')
 identities[side]={'identifier':model,'model_path':identity['model_path'],'tokenizer_path':identity['tokenizer_path'],'checkpoint_files':len(identity['checkpoint_files']),'audit_sha256':identity['audit_sha256'],'adapter_already_merged':identity.get('adapter_already_merged',False)}
check(len({x['model_path'] for x in identities.values()})==3,'duplicate model paths')
native_text=(ev.NATIVE/'run_solver.py').read_text();check('backend = VLLMSolverBackend(model_path=Path(manifest["model"]), tokenizer_path=TOKENIZER,' in native_text,'merged model loading changed');check('canonical_answer_text=""' in native_text,'Gold passed to generation')
record('all_three_historical_model_identities_and_weight_hashes',{'models':identities,'unique_checkpoint_files_hashed':len(checkpoint_paths),'no_extra_adapter_argument_in_native_live':True})

# Exercise reporting behavior without model inference or writing production outputs.
report=module('v134_independent_report',E/'report.py');helper,scoring=report.libraries();analytic=[]
for q in all_q:
 r=next(r for r in requests if r['candidate_id']==q['candidate_id']);gold=read(E/'run/base/gold.json')[q['candidate_id']];mf=read(E/'run/base/manifest.json')
 for found in [[],[(Path('/tmp/duplicate-a.json'),{}),(Path('/tmp/duplicate-b.json'),{})]]:
  result=report.score_call(q,r,mf,gold,found,scoring);check(result['exact_match']==0 and result['valid'] is False and result['signed_partial'] is None,'missing/duplicate denominator')
 if q['question_type']=='multi_select':
  result=report.all_select_call(q,r,gold,scoring);expected=1 if len(q['gold'])==6 else 0;check(result['exact_match']==expected and float(result['signed_partial'])==expected,'all select analytic');analytic.append({'candidate_id':q['candidate_id'],'gold_count':len(q['gold']),'exact':expected,'signed_partial':expected})
empty=report.group_summary([],[],helper)
for model in ev.MODELS:
 check(empty['models'][model]['root_equal_among_covered']['exact_rate'] is None,'uncovered root should be null')
 check(empty['models'][model]['question_equal']['exact_rate_equal_candidate'] is None,'uncovered question group should be null')
record('report_fixed_denominators_null_roots_and_analytic_baseline',{'missing_and_duplicate_exact_zero_all_49':True,'uncovered_roots':24,'fixed_root_rows':31,'uncovered_metrics_null':True,'all_select_gold2345_exact_and_partial':0,'all_select_gold6_exact_and_partial':1,'ordering_pair_accuracy':'correct_pairs/15','ordering_signed_partial':'(correct_pairs-wrong_pairs)/15','primary_candidates_excluding_gold6':42})

result={'status':'PASS','actual_reviewer':'/root/frontier_quality','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model_solver_training_invoked':False,'production_files_modified':False,'noncoder_of_evaluation_tools':True,'checks':CHECKS,'question_count':49,'calls_per_model':798,'total_planned_calls':2394,'maximum_prompt_tokens':8239,'findings':[],'limitations':['Mechanical/source-binding checks complement actual source-first semantic reviews; no program certifies entailment.','This reader directly performed the source-first semantic reviews for roots147355,229160,151581 and inspected other roots reviewed propositions and cited evidence after their reviews were fixed.','Same inherited model and shared filesystem do not provide strict external blind review.','31 heldout roots are retained; 24 have no new questions and keep null metrics. Seven conflict-isolated roots supply this increment.','Permutations, rollouts, Gold families and same-root questions are correlated; the incremental comparison is descriptive, not causal.'],'audited_files':[bind(p) for p in [E/'PREFLIGHT_FREEZE.json',E/'TOOLS_FREEZE.json',E/'SCOPE.json',E/'SHARED_REQUESTS.json',E/'evaluate.py',E/'report.py',E/'run_all.py',E/'adapter_protocol.py',B/'MASTER_FREEZE.json',B/'DEV_QA_CLOSED.json',B/'QA_FREEZE.json',B/'EVALUATION_POOL.json',B/'evidence/QA_CONTROLLER_PREFLIGHT.json']+review_binds]}
Path('/tmp/v134_independent_evaluation_frontier_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'PASS','checks':len(CHECKS),'report':'/tmp/v134_independent_evaluation_frontier_result.json'}),flush=True)
