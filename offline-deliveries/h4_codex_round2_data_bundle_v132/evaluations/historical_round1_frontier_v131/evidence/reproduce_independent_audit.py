"""Independent mechanical audit of frozen historical frontier, no inference or TEST."""
import argparse,collections,csv,datetime,fractions,hashlib,itertools,json,pathlib,re,sys
sys.dont_write_bytecode=True
R=pathlib.Path('/root/autodl-tmp/gxy-selfplay');P=R/'offline-deliveries/h4_codex_historical_frontier_eval_v131';V127=R/'offline-deliveries/h4_codex_dev_eval_v127'
H=R/'restart-n5-runs/h4_round0_trained_solver_train_reeval_v1_live_1';M0=R/'restart-n5-runs/h4_round0_b0_k3_difficulty_v1_live_1';HIST={'historical_m0':M0,'historical_clean':H/'clean_train'}
E=[];B={};HASHCACHE={};SIDES={'live_clean':('M_clean_policy',V127/'run/baseline/solver_identity.json'),'live_new':('M_v125_newonly',V127/'run/new/solver_identity.json')}
def read(p):return json.loads(pathlib.Path(p).read_text())
def jsonl(p):return [json.loads(l) for l in pathlib.Path(p).read_text().splitlines() if l.strip()]
def ck(ok,msg):
 if not ok:E.append(msg)
def sha(p):
 p=pathlib.Path(p).resolve();st=p.stat();k=(str(p),st.st_size,st.st_mtime_ns)
 if k in HASHCACHE:return HASHCACHE[k]
 hh=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):hh.update(b)
 value=hh.hexdigest();HASHCACHE[k]=value;return value
def bind(p,h=None,size=None):
 p=pathlib.Path(p).resolve();v=sha(p);B[str(p)]={'path':str(p),'sha256':v,'bytes':p.stat().st_size}
 if h is not None:ck(v==h,'SHA:'+str(p))
 if size is not None:ck(p.stat().st_size==size,'SIZE:'+str(p))
 return v
def sidecar(p):
 bind(p);q=pathlib.Path(str(p)+'.sha256');bind(q);ck(sha(p)==q.read_text().strip().split()[0],'SIDECAR:'+str(p))
def freeze(p):
 bind(p)
 for r in read(p)['files']:
  path=pathlib.Path(r['path']);path=path if path.is_absolute() else p.parent/path;bind(path,r['sha256'],r.get('bytes'))
def candidate_file(run,cid):return run/'candidate_results'/(hashlib.sha256(cid.encode()).hexdigest()+'.json')
def isfront(r):
 a=r['models']['M0'];b=r['models']['M_clean_policy'];return a['full_context_correct_count']==0 and 0<b['full_context_correct_count']<b['full_context_total_count']
def independent_score(raw,q,req):
 typ=q['question_type'];field='items' if typ=='ordering' else 'options';mapping=req['display_to_canonical'];allowed=set(mapping);mm=re.fullmatch(r'<box>([^<>]+)</box>',raw.strip());ck(mm is not None,'RAW_NOT_BOX:'+req['request_id'])
 if not mm:return None
 ds=mm[1].split(',');ck(len(ds)==len(set(ds)) and set(ds)<=allowed,'RAW_LABELS:'+req['request_id'])
 if typ=='single_choice':ck(len(ds)==1,'SC_NOT_SINGLE:'+req['request_id'])
 if typ=='ordering':ck(len(ds)==len(mapping),'ORDER_NOT_COMPLETE:'+req['request_id'])
 pred=[mapping[d] for d in ds];gold=list(req['canonical_gold']);out={'display_prediction':ds,'canonical_prediction':sorted(set(pred)) if typ=='multi_select' else pred,'exact':int(set(pred)==set(gold) if typ=='multi_select' else pred==gold)}
 if typ=='multi_select':
  pp,gg=set(pred),set(gold);tp=len(pp&gg);fp=len(pp-gg);fn=len(gg-pp);neg=len(mapping)-len(gg);j=fractions.Fraction(tp,len(gg))-(fractions.Fraction(fp,neg) if neg else 0)
  out.update(tp=tp,fp=fp,fn=fn,predicted_count=len(pp),gold_count=len(gg),signed_partial=str(j),jaccard=len(pp&gg)/len(pp|gg),precision=tp/len(pp),recall=tp/len(gg))
 elif typ=='ordering':
  pos={c:i for i,c in enumerate(pred)};corr=sum(pos[a]<pos[b] for a,b in itertools.combinations(gold,2));den=len(gold)*(len(gold)-1)//2;out.update(correct_pairs=corr,total_pairs=den,pair_accuracy=corr/den,signed_partial=str(fractions.Fraction(2*corr-den,den)))
 return out
def check_native(s,own,q,tag):
 ck(s.get('scored') is True and bool(s.get('correct'))==bool(own['exact']),tag+':EXACT_NATIVE')
 ck(s['display_prediction']==own['display_prediction'] and s['canonical_prediction']==own['canonical_prediction'],tag+':MAP_NATIVE')
 metrics=s.get('metrics',{})
 if q['question_type']=='multi_select':
  ck(metrics['exact_match']==bool(own['exact']),tag+':MS_EXACT')
  for k in ['precision','recall','jaccard']:ck(abs(metrics[k]-own[k])<1e-12,tag+':MS_'+k)
 if q['question_type']=='ordering':ck(metrics['exact_order_match']==bool(own['exact']) and abs(metrics['pairwise_accuracy']-own['pair_accuracy'])<1e-12,tag+':ORDER_SCORE')
def historical():
 for n in ['candidate_transition_matrix.jsonl','m0_full_zero_transition_report.json','preflight_report.json','m0_reuse_audit.json','model_identity_report.json']:sidecar(H/n)
 ck(sha(H/'candidate_transition_matrix.jsonl')=='853118ab1f484b70c440dd486bb39b5be48dabefb7e567a998bc7e49e04aafae','FIXED_HIST_TRANSITION_SHA')
 ts=jsonl(H/'candidate_transition_matrix.jsonl');selected=[r for r in ts if isfront(r)];ck(len(selected)==123,'FRONTIER_NOT123');ck(collections.Counter(r['question_type'] for r in selected)=={'ordering':62,'multi_select':51,'single_choice':10},'FRONTIER_TYPES');ck(all(r['split']=='train' and r['training_exposure']=='UNEXPOSED_TO_BOTH' for r in selected),'FRONTIER_SCOPE_EXPOSURE')
 inv=read(H/'preflight_report.json')['source_root_inventory'];qmap={};hist={};reqs={};paths={};totals=collections.Counter()
 for tr in selected:
  cid=tr['candidate_id'];root=tr['root_id'];origin=inv[root]
  for pathkey,hkey in [('candidate_pool_path','candidate_pool_sha256'),('dialogue_path','dialogue_sha256')]:bind(origin[pathkey],origin[hkey])
  qq=[q for q in read(origin['candidate_pool_path']) if q['candidate_id']==cid];ck(len(qq)==1,'CANONICAL_ID:'+cid);q=qq[0];qmap[cid]=q;hist[cid]={};reqs[cid]={};paths[cid]={}
  for key,run in HIST.items():
   fp=candidate_file(run,cid);sidecar(fp);row=read(fp);paths[cid][key]=str(fp);ck(row['candidate_id']==cid and row['question_type']==q['question_type'] and row['candidate_pool_sha256']==origin['candidate_pool_sha256'] and row['dialogue_sha256']==origin['dialogue_sha256'],'HIST_CANDIDATE_BIND:'+cid+':'+key)
   rr=[x for x in row['requests'] if x['context_mode']=='full_context'];model='M0' if key=='historical_m0' else 'M_clean_policy';state=tr['models'][model];ck(len(rr)==state['full_context_total_count'] and sum(x['correct'] for x in rr)==state['full_context_correct_count'],'HIST_TRANSITION_COUNTS:'+cid+':'+key)
   reqs[cid][key]=rr;hist[cid][key]=[]
   for r in rr:
    rid=r['request_id'];cache=r['request_cache_key'];prompt=run/'raw_prompts'/(r['prompt_sha256']+'.txt');envelope=run/'raw_responses'/(cache+'.json');ledger=run/'request_ledger'/root/(cache+'.json');bind(prompt,r['prompt_sha256']);bind(envelope);bind(ledger);env=read(envelope);led=read(ledger)
    ck(led['status']=='completed' and led['solver_generation_count']==1 and len(led['attempts'])==1,'HIST_RETRY_OR_INCOMPLETE:'+rid+':'+key)
    ck(led['prompt_sha256']==r['prompt_sha256'] and led['response_envelope_sha256']==sha(envelope),'HIST_LEDGER_BIND:'+rid+':'+key)
    ck(env['raw_text']==r['raw_output'] and env['raw_response_sha256']==hashlib.sha256(r['raw_output'].encode()).hexdigest(),'HIST_RAW:'+rid+':'+key)
    rc=led['request_contract']
    for f in ['request_id','candidate_id','root_id','question_type','variant_id','context_mode','slot_id','seed','prompt_sha256','display_to_canonical','canonical_gold']:
     if f in rc:ck(rc[f]==r[f],'HIST_LEDGER_DESCRIPTOR:'+f+':'+rid+':'+key)
    md=env['response_metadata'];ck(md['sampling_seed']==r['seed'] and md['prompt_token_count']==r['prompt_token_count'] and md['finish_reason'] in ['stop','eos'],'HIST_METADATA:'+rid+':'+key)
    gc=rc['generation_config'];structured=q['question_type']!='single_choice';ck(gc['vllm_version']=='0.10.2' and gc['structured_output_version']==('finite_choice_set_v1' if structured else None) and gc['enable_thinking'] is False,'HIST_RUNTIME:'+rid+':'+key)
    for f in ['temperature','top_p','max_tokens','structured_output_api','structured_output_version']:ck(md.get(f)==gc[f],'HIST_DECODER:'+f+':'+rid+':'+key)
    own=independent_score(r['raw_output'],q,r);check_native(r,own,q,'HIST_NATIVE:'+rid+':'+key);hist[cid][key].append({'request_id':rid,'candidate_id':cid,'root_id':root,'question_type':q['question_type'],**own})
  aa={x['request_id']:x for x in reqs[cid]['historical_m0']};bb={x['request_id']:x for x in reqs[cid]['historical_clean']};ck(set(aa)==set(bb),'HIST_PAIRED_KEYS:'+cid)
  for rid,a in aa.items():
   b=bb[rid]
   for f in ['request_id','candidate_id','root_id','question_type','variant_id','shift','context_mode','slot_id','seed','prompt_sha256','prompt_token_count','display_to_canonical','canonical_gold','displayed_gold']:ck(a[f]==b[f],'HIST_PAIR_DESCRIPTOR:'+f+':'+rid)
  totals[q['question_type']]+=len(bb)
 ck(dict(totals)=={'multi_select':765,'ordering':744,'single_choice':120},'HIST_REQUEST_COUNTS')
 return selected,qmap,hist,reqs,paths,inv

def preflight(final=False):
 selected,qs,hist,oldreqs,oldpaths,inv=historical();bind(V127/'COMPLETE_MANIFEST.json','4fb446adcf81d8ce6db72accf78dd3e47b64c2e2548add3191561856e8389d61');v127f={r['path']:r for r in read(V127/'COMPLETE_MANIFEST.json')['files']};bind(P/'manifest.json');manifest=read(P/'manifest.json');qpath=pathlib.Path(manifest['questions_path']);bind(qpath);actualq=read(qpath)['candidates'];ck(actualq==[qs[r['candidate_id']] for r in selected],'CANONICAL_PROJECTION_LOSSLESS')
 mc=manifest['candidates'];ck([r['candidate_id'] for r in mc]==[r['candidate_id'] for r in selected],'CANDIDATE_SELECTION_EXACT_ALL123')
 for rr,tr in zip(mc,selected):
  cid=tr['candidate_id'];ck(rr['transition']==tr and rr['root_id']==tr['root_id'] and rr['question_type']==tr['question_type'] and rr['training_exposure']==tr['training_exposure'],'CANDIDATE_TRUTH_BIND:'+cid)
  ck(rr['historical_m0_path']==oldpaths[cid]['historical_m0'] and rr['historical_clean_path']==oldpaths[cid]['historical_clean'],'HIST_RESULT_PATH:'+cid)
 nr=manifest['requests'];expect=[r for tr in selected for r in oldreqs[tr['candidate_id']]['historical_clean']];ck(len(nr)==len(expect)==1629 and len({r['request_id'] for r in nr})==1629,'NEW_REQUEST_COUNT')
 ck([r['request_id'] for r in nr]==[r['request_id'] for r in expect],'REQUEST_ORDER_EXACT_PROJECTION')
 common=['request_id','candidate_id','root_id','question_type','variant_id','shift','slot_id','seed','context_mode','prompt_sha256','prompt_token_count','display_to_canonical','canonical_gold','displayed_gold']
 for n,o in zip(nr,expect):
  rid=o['request_id'];cid=o['candidate_id'];q=qs[cid]
  for f in common:ck(n[f]==o[f],'NEW_DESCRIPTOR:'+f+':'+rid)
  ck(n['original_response']==o,'ORIGINAL_RESPONSE_LOSSLESS:'+rid)
  pp=pathlib.Path(n['prompt_path']);oldp=H/'clean_train/raw_prompts'/(o['prompt_sha256']+'.txt');ck(pp.read_bytes()==oldp.read_bytes(),'NEW_PROMPT_BYTES:'+rid);bind(pp,o['prompt_sha256'])
  field='items' if q['question_type']=='ordering' else 'options';keys=list(q[field]);shift=o['shift'];v=n['variant'];mapping=o['display_to_canonical'];displayed={k:q[field][mapping[k]] for k in (keys[shift:]+keys[:shift] if q['question_type']=='ordering' else keys)}
  ck(v['variant_id']==o['variant_id'] and v['shift']==shift and v['display_to_canonical']==mapping and v['canonical_gold']==o['canonical_gold'] and v['displayed_gold']==o['displayed_gold'],'VARIANT_BIND:'+rid)
  ck(list(v[field].items())==list(displayed.items()),'VARIANT_DISPLAY_CONTENT_ORDER:'+rid)
  ck([mapping[k] for k in o['displayed_gold']]==o['canonical_gold'],'GOLD_ROUNDTRIP:'+rid)
 for side,(mid,priorpath) in SIDES.items():
  frozen=v127f[str(priorpath.relative_to(V127))];bind(priorpath,frozen['sha256'],frozen['bytes'])
  ident=manifest['identities'][side];prior=read(priorpath);bind(priorpath);ck(ident==prior,'EXACT_V127_MODEL_IDENTITY:'+side);ck(ident['model_identifier']==mid,'MODEL_ID:'+side)
  bind(ident['audit_path'],ident['audit_sha256'])
  for row in ident['checkpoint_files']:bind(row['path'],row['sha256'],row['bytes'])
  ck(ident.get('adapter_enabled') is True and ident.get('adapter_already_merged') is True,'MERGED_MODEL_DECL:'+side)
 ck(manifest['native_import_root']==str(R/'current_method_github'),'NATIVE_IMPORT_ROOT')
 ck(manifest['question_count']==123 and manifest['request_count_per_side']==1629 and manifest['response_filename']=='sha256(request_id UTF-8).json','MANIFEST_COUNTS_NAMING')
 ck(manifest['runtime']['vllm']=='0.21.0','CURRENT_RUNTIME_VLLM021')
 for t,gc in manifest['decoder_contract'].items():
  structured=t!='single_choice';expected={'temperature':.8 if structured else .7,'top_p':.9,'max_tokens':32,'structured_output_api':'structured_outputs=StructuredOutputsParams(choice)' if structured else None,'structured_output_version':'finite_choice_set_v1' if structured else None,'vllm_version':'0.21.0','stop':['</box>'] if structured else [],'include_stop_str_in_output':structured,'enable_thinking':False};ck(gc==expected,'DECODER_CONTRACT:'+t)
 bind(P/'SCOPE.json');sc=read(P/'SCOPE.json')
 for k,v in {'questions':123,'roots':58,'question_types':{'ordering':62,'multi_select':51,'single_choice':10},'split':'train','historical_question_exposure':'UNEXPOSED_TO_BOTH','context_modes':['full_context'],'requests_per_side':1629,'new_inference_calls':3258,'reused_historical_calls':3258,'new_sides':['live_clean','live_new'],'historical_sides':['historical_m0','historical_clean'],'native_generation_algorithm_changed':False,'current_runtime':manifest['runtime'],'prompt_and_gold_replay_checks':3258,'historical_code_sha_equal_to_current_claimed':False,'test_evaluation':False,'new_training':False,'new_question_generation':False,'automatic_retry':False,'score_based_filtering_of_this_selected_set':False,'gold_modification':False}.items():ck(sc[k]==v,'SCOPE:'+k)
 for root in {r['root_id'] for r in selected}:bind(P/'sources'/(root+'.json'),inv[root]['dialogue_sha256'])
 freeze(P/'TOOLS_FREEZE.json');freeze(P/'PREFLIGHT_FREEZE.json');tools={r['path'] for r in read(P/'TOOLS_FREEZE.json')['files']};ck(str(P/'evaluate.py') in tools and str(P/'report_frontier.py') in tools,'WRAPPER_REPORT_TOOL_FREEZE')
 bind(pathlib.Path(__file__))
 if not final:ck(not (P/'RUN_STATUS.json').exists() and not (P/'responses').exists(),'PREAUDIT_AFTER_START')
 checks={'population':{'questions':123,'roots':len({r['root_id'] for r in selected}),'question_types':dict(collections.Counter(r['question_type'] for r in selected)),'full_calls_per_live_model':1629,'single_choice_calls':120,'multi_select_calls':765,'ordering_calls':744,'all_unexposed_to_first_round_training_records':True,'historical_train_scope_only':True},'projection':{'all123_selected_by_original_transition_condition':True,'no_extra_quality_score_filter':True,'historical_prompts_copied_byte_exact':True,'old_cyclic_variant_and_label_semantics_retained':True,'seeds_and_decoding_values_preserved':True,'historical_runtime':'vllm0.10.2','current_bridge_and_new_runtime':'vllm0.21.0','direct_comparison_is_current_live_clean_to_current_live_new':True},'quality_boundary':'Historical Gold retained for frozen diagnostic comparison; progress transition does not certify semantic correctness or held-out generalization.','no_TEST':True}
 return manifest,qs,hist,checks

def summarize(rows):
 def ss(rs):
  out={'calls':len(rs),'exact_correct':sum(r['exact'] for r in rs),'exact_rate':sum(r['exact'] for r in rs)/len(rs) if rs else None}
  ms=[r for r in rs if r['question_type']=='multi_select'];oo=[r for r in rs if r['question_type']=='ordering']
  if ms:
   out['multi_select']={'calls':len(ms),'tp':sum(r['tp'] for r in ms),'fp':sum(r['fp'] for r in ms),'fn':sum(r['fn'] for r in ms),'mean_signed_partial':float(sum((fractions.Fraction(r['signed_partial']) for r in ms),fractions.Fraction())/len(ms))}
  if oo:out['ordering']={'calls':len(oo),'correct_pairs':sum(r['correct_pairs'] for r in oo),'total_pairs':sum(r['total_pairs'] for r in oo),'pair_accuracy':sum(r['correct_pairs'] for r in oo)/sum(r['total_pairs'] for r in oo)}
  return out
 return {'overall':ss(rows),'by_type':{t:ss([r for r in rows if r['question_type']==t]) for t in ['multi_select','ordering','single_choice']},'per_question':{cid:ss([r for r in rows if r['candidate_id']==cid]) for cid in sorted({r['candidate_id'] for r in rows})},'per_call':rows}

def final_scores(manifest,qs,hist):
 out={key:summarize([r for cid,mm in hist.items() for r in mm[key]]) for key in HIST}
 for side,(mid,ip) in SIDES.items():
  response_dir=P/'responses'/side;names={hashlib.sha256(r['request_id'].encode()).hexdigest()+'.json' for r in manifest['requests']};ck({p.name for p in response_dir.glob('*.json')}==names,'LIVE_RESPONSE_FILE_SET:'+side);rows=[]
  for req in manifest['requests']:
   rp=response_dir/(hashlib.sha256(req['request_id'].encode()).hexdigest()+'.json');bind(rp);r=read(rp);rid=req['request_id'];cid=req['candidate_id'];q=qs[cid]
   for f in ['request_id','candidate_id','root_id','question_type','variant_id','shift','slot_id','seed','context_mode','prompt_sha256','prompt_token_count','display_to_canonical','canonical_gold','displayed_gold']:ck(r[f]==req[f],'LIVE_RESPONSE_BIND:'+f+':'+side+':'+rid)
   ck(r['solver_model']==mid and r['model_path']==manifest['identities'][side]['model_path'],'LIVE_MODEL:'+side+':'+rid)
   ck(r['side']==side and r['model_fingerprint_sha256']==manifest['identities'][side]['model_fingerprint_sha256'] and r['runtime']==manifest['runtime'],'LIVE_ID_RUNTIME:'+side+':'+rid)
   ck(r['automatic_retry'] is False,'LIVE_RETRY:'+side+':'+rid)
   md=r['metadata'];expected_temp=.7 if q['question_type']=='single_choice' else .8;ck(md['sampling_seed']==req['seed'] and md['prompt_token_count']==req['prompt_token_count'] and md['finish_reason'] in ['stop','eos'],'LIVE_METADATA:'+side+':'+rid)
   ck(md['temperature']==expected_temp and md['top_p']==.9 and md['max_tokens']==32 and 0<md['completion_token_count']<=32,'LIVE_SAMPLING:'+side+':'+rid)
   if q['question_type']!='single_choice':ck(md['structured_output_version']=='finite_choice_set_v1' and md['structured_output_api']=='structured_outputs=StructuredOutputsParams(choice)','LIVE_STRUCTURED_DECODER:'+side+':'+rid)
   ck(r['evaluation_valid'] is True,'LIVE_EVALUATION_INVALID:'+side+':'+rid);own=independent_score(r['raw_output'],q,req);check_native(r['score'],own,q,'LIVE_NATIVE:'+side+':'+rid)
   rows.append({'request_id':rid,'candidate_id':cid,'root_id':req['root_id'],'question_type':q['question_type'],**own})
  out[side]=summarize(rows)
 return out

def close(a,b,tag):
 ck(a is None and b is None or a is not None and b is not None and abs(float(a)-float(b))<1e-12,tag)
def average(vals):return sum(vals,fractions.Fraction())/len(vals) if vals else None
def report_audit(manifest,qs,results):
 for fn in ['SUMMARY.json','PER_QUESTION.json','PER_QUESTION.csv','PER_CALL.json','PER_ROOT.json','REPORT_ZH.md']:bind(P/fn)
 summary=read(P/'SUMMARY.json');ck(summary['status']=='COMPLETE' and not summary['errors'],'REPORT_NOT_COMPLETE')
 ck(summary['population']=={'questions':123,'train_roots':58,'question_type_counts':{'ordering':62,'multi_select':51,'single_choice':10},'planned_calls_per_model':1629,'new_current_runtime_models':2,'new_inference_calls':3258,'reused_historical_full_calls':3258,'comparison_calls_total':6516,'no_context_calls_added':0,'test_calls_added':0},'REPORT_POPULATION')
 allrows={m:r['per_call'] for m,r in results.items()};lookup={m:{r['request_id']:r for r in rows} for m,rows in allrows.items()};models=list(allrows)
 def agg(rows,report,tag):
  ck(report['planned_calls']==len(rows) and report['valid_calls']==len(rows) and report['invalid_or_missing_calls']==0 and report['exact_correct']==sum(r['exact'] for r in rows),tag+':CALL_COUNTS')
  exact=average([fractions.Fraction(r['exact']) for r in rows]);close(report['exact_rate_fixed'],exact,tag+':EXACT');ck(report['exact_rate_fraction']==(str(exact) if exact is not None else None),tag+':EXACT_FRAC')
  oo=[r for r in rows if r['question_type']=='ordering'];ms=[r for r in rows if r['question_type']=='multi_select']
  for branch,subset,key,own in [('ordering',oo,'pair_accuracy','pair_accuracy'),('multi_select',ms,'jaccard','jaccard')]:
   rr=report[branch];mm=rr[key];n=len(subset);mean=average([fractions.Fraction(r['correct_pairs'],r['total_pairs']) if branch=='ordering' else fractions.Fraction(r['tp'],r['tp']+r['fp']+r['fn']) for r in subset]);ck(rr['planned_calls']==n and mm['planned_denominator']==n and mm['defined_count']==n and mm['undefined_count']==0,tag+':'+branch+':DENOM')
   for field in ['mean','lower','upper']:close(mm[field],mean,tag+':'+branch+':'+field)
   if branch=='ordering':ck(rr['known_correct_pairs']==sum(r['correct_pairs'] for r in subset) and rr['planned_pairs']==sum(r['total_pairs'] for r in subset) and rr['unknown_pairs']==0,tag+':PAIR_COUNTS')
   else:
    for source,target in [('tp','known_tp'),('fp','known_fp'),('fn','known_fn')]:ck(rr[target]==sum(r[source] for r in subset),tag+':'+target)
    close(rr['predicted_count']['mean'],average([fractions.Fraction(r['predicted_count']) for r in subset]),tag+':PREDICTED_COUNT')
 def eq(ids,model,report,tag):
  rows=[r for r in allrows[model] if r['candidate_id'] in ids];per={cid:[r for r in rows if r['candidate_id']==cid] for cid in ids};rate=average([average([fractions.Fraction(r['exact']) for r in rr]) for rr in per.values()]);ck(report['question_count']==len(ids) and report['planned_calls']==len(rows) and report['valid_calls']==len(rows),tag+':EQ_COUNTS');close(report['exact_rate_equal_question'],rate,tag+':EQUAL_QUESTION');ck(report['exact_rate_equal_question_fraction']==(str(rate) if rate is not None else None),tag+':EQ_EXACT_FRAC')
  for typ,branch,key in [('ordering','ordering','pair_accuracy'),('multi_select','multi_select','jaccard')]:
   qids=[cid for cid in ids if qs[cid]['question_type']==typ];vals=[average([fractions.Fraction(r['correct_pairs'],r['total_pairs']) if typ=='ordering' else fractions.Fraction(r['tp'],r['tp']+r['fp']+r['fn']) for r in per[cid]]) for cid in qids];rr=report[branch+'_'+key+'_equal_question'];ck(rr['question_count']==len(qids),tag+':'+typ+':EQ_COUNT')
   for field in ['mean','lower','upper']:close(rr[field],average(vals),tag+':'+typ+':EQ_'+field)
  return rate
 for m,rows in allrows.items():agg(rows,summary['models'][m]['summary'],'REPORT_TOTAL:'+m)
 pc=read(P/'PER_CALL.json');ck(len(pc)==1629 and {r['request_id'] for r in pc}==set(lookup['live_clean']),'REPORT_CALL_SET')
 for row in pc:
  rid=row['request_id']
  for m in models:
   rr=row['models'][m];own=lookup[m][rid];ck(rr['valid'] is True and rr['errors']==[] and rr['exact_match']==own['exact'] and rr['canonical_prediction']==own['canonical_prediction'] and rr['display_prediction']==own['display_prediction'],'REPORT_CALL_SCORE:'+m+':'+rid)
   if own['question_type']=='ordering':ck(rr['correct_pairs']==own['correct_pairs'] and rr['wrong_pairs']==own['total_pairs']-own['correct_pairs'],'REPORT_CALL_PAIRS:'+m+':'+rid);close(fractions.Fraction(rr['pair_accuracy']),own['pair_accuracy'],'REPORT_CALL_PAIR_RATE:'+m+':'+rid)
   if own['question_type']=='multi_select':
    for k in ['tp','fp','fn','predicted_count']:ck(rr[k]==own[k],'REPORT_CALL_MS:'+k+':'+m+':'+rid)
    close(fractions.Fraction(rr['jaccard']),own['jaccard'],'REPORT_CALL_JACCARD:'+m+':'+rid)
 pq=read(P/'PER_QUESTION.json');ck(len(pq)==123 and {r['candidate_id'] for r in pq}==set(qs),'REPORT_QUESTION_SET');byq={r['candidate_id']:r for r in pq}
 pairs=[('historical_m0','historical_clean'),('historical_clean','live_clean'),('live_clean','live_new')]
 for row in pq:
  cid=row['candidate_id'];q=qs[cid];ck(row['root_id']==q['root_id'] and row['question_type']==q['question_type'] and row['split']=='train' and row['training_exposure']=='UNEXPOSED_TO_BOTH','REPORT_QUESTION_BIND:'+cid)
  for m in models:agg([r for r in allrows[m] if r['candidate_id']==cid],row['models'][m],'REPORT_QUESTION:'+cid+':'+m)
  for a,b in pairs:close(row['paired_deltas'][a+'_to_'+b]['exact_delta'],results[b]['per_question'][cid]['exact_rate']-results[a]['per_question'][cid]['exact_rate'],'REPORT_QUESTION_DELTA:'+cid+':'+a+':'+b)
 transitions=[r['transition'] for r in manifest['candidates']];both={r['candidate_id'] for r in transitions if 0<r['models']['M_inclusive_policy']['full_context_correct_count']<r['models']['M_inclusive_policy']['full_context_total_count']};seedpath=R/'restart-n5-runs/h4_round1_feedback_compiler_v1_live_1/feedback_seed_manifest.jsonl';bind(seedpath);seeds={r['seed_candidate_id'] for r in jsonl(seedpath)}
 ck(len(both)==93 and len(set(qs)-both)==30 and len(seeds)==10 and seeds<=both,'REPORT_STRATA_POPULATIONS')
 groups={'all123':set(qs),**{typ:{cid for cid,q in qs.items() if q['question_type']==typ} for typ in ['single_choice','multi_select','ordering']},'historical_both93':both,'historical_clean_only30':set(qs)-both,'formal_feedback_seeds10':seeds};ck(set(summary['strata'])==set(groups),'REPORT_STRATA_SET')
 for name,ids in groups.items():
  rr=summary['strata'][name];ck(rr['question_count']==len(ids),'REPORT_STRATUM_COUNT:'+name);rates={}
  for m in models:rates[m]=eq(ids,m,rr['models'][m]['equal_question'],'REPORT_STRATUM_EQ:'+name+':'+m);agg([r for r in allrows[m] if r['candidate_id'] in ids],rr['models'][m]['pooled_calls'],'REPORT_STRATUM_POOL:'+name+':'+m)
  for a,b in pairs:close(rr['paired_equal_question_exact_delta'][a+'_to_'+b],rates[b]-rates[a],'REPORT_STRATUM_DELTA:'+name+':'+a+':'+b)
 for cid,row in byq.items():ck(row['historical_frontier_group']==('BOTH93' if cid in both else 'CLEAN_ONLY30') and row['formal_feedback_seed']==(cid in seeds),'REPORT_QUESTION_STRATUM_TAG:'+cid)
 roots={q['root_id'] for q in qs.values()};pr=read(P/'PER_ROOT.json');ck(len(pr)==58 and {r['root_id'] for r in pr}==roots,'REPORT_ROOT_SET');rootrates={m:[] for m in models}
 for row in pr:
  root=row['root_id'];ids={cid for cid,q in qs.items() if q['root_id']==root};ck(row['question_count']==len(ids),'REPORT_ROOT_COUNT:'+root)
  for m in models:rootrates[m].append(eq(ids,m,row['models'][m],'REPORT_ROOT_EQ:'+root+':'+m))
 for m in models:ck(summary['root_macro'][m]['covered_train_roots']==58,'REPORT_ROOT_MACRO_COUNT:'+m);close(summary['root_macro'][m]['exact_rate_equal_root'],average(rootrates[m]),'REPORT_ROOT_MACRO_RATE:'+m)
 with (P/'PER_QUESTION.csv').open() as stream:csvrows=list(csv.DictReader(stream))
 ck(len(csvrows)==123 and {r['candidate_id'] for r in csvrows}==set(qs),'REPORT_CSV_QUESTION_SET')
 for row in csvrows:
  cid=row['candidate_id']
  for m in models:
   own=results[m]['per_question'][cid];ck(int(row[m+'_planned_calls'])==own['calls'] and int(row[m+'_valid_calls'])==own['calls'],'REPORT_CSV_CALL_COUNTS:'+m+':'+cid);close(float(row[m+'_exact']),own['exact_rate'],'REPORT_CSV_EXACT:'+m+':'+cid)
 for rr in summary['input_bindings']:bind(rr['path'],rr['sha256'])
 qpath=P/'evidence/HISTORICAL_QUALITY_CROSSCHECK.json'
 if qpath.exists():
  bind(qpath);quality=read(qpath);ck(len(quality['rows'])==123 and {r['candidate_id'] for r in quality['rows']}==set(qs),'QUALITY_APPLICABILITY_SET');ck(summary['historical_gold_review_applicability']==quality['summary'],'QUALITY_SUMMARY_RETAINED')
 return {'report_files':6,'all_four_model_raws_independently_reconstructed':True,'calls_per_model':1629,'per_question_and_call_and_root_verified':True,'question_equal_and_root_equal_and_call_equal_verified':True,'strata':{k:len(v) for k,v in groups.items()},'report_semantic_gold_certification':False}

def final_status_audit(prior):
 audit_time=datetime.datetime.fromisoformat(prior['created_at_utc']);status=read(P/'RUN_STATUS.json');bind(P/'RUN_STATUS.json');ck(status['status']=='REPORTED_AWAITING_INDEPENDENT_AUDIT' and status['actual_new_calls']==3258 and status['planned_new_calls']==3258 and status['pid'] is None and status['automatic_retry'] is False and status['test_evaluation'] is False,'ROOT_STATUS_NOT_STOPPED_COMPLETE')
 ck(datetime.datetime.fromisoformat(status['started_at_utc'])>=audit_time,'INFERENCE_BEFORE_AUDIT')
 for side in SIDES:
  fp=P/(side+'_STATUS.json');bind(fp);ss=read(fp);ck(ss['status']=='COMPLETE' and ss['planned_calls']==1629 and ss['actual_calls']==1629 and ss['pid'] is None,'SIDE_NOT_STOPPED_COMPLETE:'+side);ck(datetime.datetime.fromisoformat(ss['started_at_utc'])>=audit_time,'SIDE_BEFORE_AUDIT:'+side);bind(P/(side+'.log'))
 for log in P.glob('*.log'):bind(log)
 running=[]
 for proc in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
  try:args=proc.read_bytes().split(b'\0')
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  if str(P/'evaluate.py').encode() in args and any(x in args for x in [b'live',b'live_clean',b'live_new']):running.append(str(proc))
 ck(not running,'LIVE_INFERENCE_PROCESS_STILL_RUNNING:'+str(running))
 return {'root_and_side_statuses_stopped_complete':True,'new_calls':3258,'logs_bound':True,'no_live_wrapper_processes':not running}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','final']);a=ap.parse_args();manifest,qs,hist,checks=preflight(a.stage=='final')
 if a.stage=='final':
  pa=P/'evidence/INDEPENDENT_PREFLIGHT_AUDIT.json';prior=read(pa);ck(prior['status']=='PASS' and not prior['errors'],'PREAUDIT_NOT_PASS');bind(pa)
  for row in prior['audited_files']:bind(row['path'],row['sha256'],row.get('bytes'))
  checks['independent_results']=final_scores(manifest,qs,hist)
  checks['report_crosscheck']=report_audit(manifest,qs,checks['independent_results'])
  checks['final_status']=final_status_audit(prior)
 out={'schema_version':'v131_independent_'+a.stage+'_audit_v1','status':'FAIL' if E else 'PASS','errors':E,'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'auditor':'/root/baseline_order_errors','checks':checks,'audited_files':list(B.values()),'new_inference_calls_by_auditor':0,'TEST_read_or_inference':False}
 dest=P/'evidence'/('INDEPENDENT_PREFLIGHT_AUDIT.json' if a.stage=='preflight' else 'INDEPENDENT_FINAL_AUDIT.json');dest.parent.mkdir(exist_ok=True,parents=True)
 with dest.open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({'status':out['status'],'errors':E,'bound_file_count':len(B),'path':str(dest),'results':{k:v['by_type'] for k,v in checks.get('independent_results',{}).items()}},ensure_ascii=False,indent=2));return bool(E)
if __name__=='__main__':sys.exit(main())
