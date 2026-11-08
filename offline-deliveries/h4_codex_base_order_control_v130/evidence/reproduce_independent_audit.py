#!/usr/bin/env python3
"""Independent v130 projection/model/completeness/score audit. No inference imports."""
import argparse,collections,datetime,fractions,hashlib,itertools,json,pathlib,re,sys
ROOT=pathlib.Path('/root/autodl-tmp/gxy-selfplay')
OLD=ROOT/'offline-deliveries/h4_codex_dev_eval_v127'
NEW=ROOT/'offline-deliveries/h4_codex_base_order_control_v130'
M0=pathlib.Path('/root/autodl-tmp/Qwen3-4B')
MODEL_AUDIT=ROOT/'offline-deliveries/h4_multiselect_partial_admission_v1/followup_model_identity.json'
MODEL_AUDIT_SHA='59e23710e2373a00aef3b7d50d117d4089eea47cd17fd16a00eebcc83fba0035'
CHECKED={};ERRORS=[]
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
 return h.hexdigest()
def check(ok,msg):
 if not ok:ERRORS.append(msg)
 return ok
def bind(p,expected=None,size=None):
 p=pathlib.Path(p).resolve();h=sha(p);CHECKED[str(p)]=h
 if expected is not None:check(h==expected,'SHA_MISMATCH:'+str(p))
 if size is not None:check(p.stat().st_size==size,'SIZE_MISMATCH:'+str(p))
 return h
def rows(d):return d['files']
def freeze(p):
 bind(p)
 for row in rows(read(p)):
  rp=pathlib.Path(row['path']);rp=rp if rp.is_absolute() else pathlib.Path(p).parent/rp
  bind(rp,row['sha256'],row.get('bytes'))
def old_frozen(p):
 rel=str(p.relative_to(OLD));r=OLDROWS[rel];bind(p,r['sha256'],r.get('bytes'))
def preflight(run,final=False):
 checks={};base=OLD/'run/baseline';trained=OLD/'run/new';new=NEW/run
 om=read(base/'manifest.json');tm=read(trained/'manifest.json');nm=read(new/'manifest.json')
 oq=read(OLD/'EVALUATION_POOL.json')['questions'];cq=[q for q in oq if q['question_type']=='ordering'];ids={q['candidate_id'] for q in cq};roots={q['root_id'] for q in cq}
 check(len(cq)==14 and len(ids)==14 and len(roots)==7,'REFERENCE_POPULATION_NOT14_7')
 oldreq=[r for r in om['requests'] if r['question_type']=='ordering'];treq=[r for r in tm['requests'] if r['question_type']=='ordering'];nr=nm['requests']
 check(nr==oldreq==treq,'EXACT_FULL_REQUEST_PROJECTION_MISMATCH');check(len(nr)==168 and len({r['request_id'] for r in nr})==168,'REQUEST_COUNT_OR_DUPLICATE')
 check(nm['candidate_count']==14 and nm['request_count']==168,'MANIFEST_COUNTS')
 for key in ['policy_version','execution','tokenizer','random_display','decoder_contract','code_sha256']:
  check(nm[key]==om[key]==tm[key],'MANIFEST_PROTOCOL_MISMATCH:'+key)
 check(set(nm['input_sha256'])==set(om['input_sha256'])==set(tm['input_sha256']),'NATIVE_INPUT_BINDING_KEYS')
 check(nm['pool_path']==str(NEW/'EVALUATION_POOL.json') and nm['pool_sha256']==sha(NEW/'EVALUATION_POOL.json'),'NATIVE_POOL_BINDING')
 for name,h in nm['input_sha256'].items():bind(NEW/'EVALUATION_POOL.json' if name=='pool' else new/name,h)
 for name,h in nm['code_sha256'].items():bind(name,h)
 check({p.name for p in (new/'sources').iterdir()}=={r+'.json' for r in roots},'SOURCE_FILE_SET')
 check({p.name for p in (new/'prompts').iterdir()}=={r['request_id']+'.txt' for r in nr},'PROMPT_FILE_SET')
 check(nm['model']==str(M0) and nm['solver_model']=='M0','M0_MANIFEST_PATH_IDENTITY')
 check(not nm.get('training_started') and not nm.get('training_promotion') and nm.get('new_generation_calls')==0,'TRAINING_OR_GENERATION_FLAG')
 check(all(r['mode']=='full_context' and r['question_type']=='ordering' and r['slot']==0 for r in nr),'REQUEST_SCOPE')
 counts=collections.Counter(r['candidate_id'] for r in nr);check(set(counts)==ids and set(counts.values())=={12},'QUESTION_CALL_COUNTS')
 for name in ['questions.json','canonical_bindings.json']:
  expected=[q for q in read(base/name)['candidates'] if q['candidate_id'] in ids]
  expectednew=[q for q in read(trained/name)['candidates'] if q['candidate_id'] in ids]
  actual=read(new/name)['candidates'];check(actual==expected==expectednew,'PREPARED_CANONICAL_PROJECTION:'+name)
 ng=read(new/'gold.json');bg=read(base/'gold.json');tg=read(trained/'gold.json');check(ng=={i:bg[i] for i in ids}=={i:tg[i] for i in ids},'GOLD_PROJECTION')
 nqp=read(NEW/'EVALUATION_POOL.json')['questions'];check(nqp==cq,'FULL_LOSSLESS_POOL_CANONICAL_PROJECTION')
 for r in nr:
  fn=r['request_id']+'.txt';paths=[d/'prompts'/fn for d in [base,trained,new]];data=[p.read_bytes() for p in paths];check(data[0]==data[1]==data[2],'PROMPT_BYTES:'+fn)
  for p in paths:bind(p,r['prompt_sha256'])
 for rid in roots:
  fn=rid+'.json';paths=[d/'sources'/fn for d in [base,trained,new]];data=[p.read_bytes() for p in paths];check(data[0]==data[1]==data[2],'SOURCE_BYTES:'+fn)
  for p in paths:bind(p)
 for d in [base,trained]:
  for name in ['manifest.json','questions.json','canonical_bindings.json','gold.json','solver_identity.json']:old_frozen(d/name)
 old_frozen(OLD/'EVALUATION_POOL.json')
 ident=read(new/'solver_identity.json');check(ident==nm['solver_identity'],'MANIFEST_IDENTITY_MISMATCH')
 check(ident['model_path']==str(M0) and ident['model_identifier']=='M0' and ident['tokenizer_path']==str(M0),'M0_IDENTITY_PATH')
 check(not ident.get('adapter_enabled') and not ident.get('adapter_already_merged') and not ident.get('adapter_path'),'M0_ADAPTER_ENABLED')
 bind(MODEL_AUDIT,MODEL_AUDIT_SHA);ba=read(MODEL_AUDIT)
 check(ident['audit_path']==str(MODEL_AUDIT) and ident['audit_sha256']==MODEL_AUDIT_SHA,'M0_FIXED_AUDIT')
 check(ident['checkpoint_files']==ba['files'],'M0_CHECKPOINT_LIST_NOT_OFFICIAL')
 for row in ident['checkpoint_files']:bind(row['path'],row['sha256'],row['bytes'])
 cfg=read(M0/'config.json');check(cfg.get('model_type')=='qwen3','BASE_CONFIG_TYPE')
 index=read(M0/'model.safetensors.index.json');shards=set(index['weight_map'].values());recorded={r['name'] for r in ident['checkpoint_files']}
 check(shards<=recorded and len(shards)==3,'ALL_BASE_SHARDS_REGISTERED')
 check(not list(M0.glob('adapter*')),'BASE_HAS_UNEXPECTED_ADAPTER_FILE')
 freeze(NEW/'TOOLS_FREEZE.json');freeze(NEW/'PREFLIGHT_FREEZE.json')
 tools={r['path'] for r in read(NEW/'TOOLS_FREEZE.json')['files']}
 check(str(NEW/'evaluate.py') in tools,'WRAPPER_NOT_FROZEN')
 check(str(NEW/'report_control.py') in tools,'REPORT_NOT_FROZEN')
 check(any(path.endswith('/evaluation/run_solver.py') for path in tools),'NATIVE_RUNNER_NOT_FROZEN')
 check(any(path.endswith('/evaluation/scoring.py') for path in tools),'NATIVE_SCORER_NOT_FROZEN')
 check(any(path.endswith('/evaluation/selection.py') for path in tools),'NATIVE_SELECTION_NOT_FROZEN')
 for filename in ['manifest.json','questions.json','gold.json','canonical_bindings.json','solver_identity.json']:bind(new/filename)
 scope=read(NEW/'SCOPE.json');oldscope=read(OLD/'SCOPE.json')
 check(scope['fixed_dev_root_ids']==oldscope['fixed_dev_root_ids'] and len(scope['fixed_dev_root_ids'])==15,'FIXED_DEV_ROOTS')
 check(scope['covered_dev_root_ids']==oldscope['covered_dev_root_ids'] and set(scope['covered_dev_root_ids'])==roots,'COVERED_DEV_ROOTS')
 check(scope['candidate_ids']==[q['candidate_id'] for q in cq] and scope['question_count']==14 and scope['new_inference_calls']==168,'FROZEN_SCOPE_COUNTS_IDS')
 check(scope['test_evaluation'] is False and scope['automatic_retry'] is False and scope['selection_based_filtering'] is False and scope['native_algorithm_changed'] is False,'FROZEN_SCOPE_EXCLUSIONS')
 bind(NEW/'SCOPE.json');bind(pathlib.Path(__file__))
 if not final:check(not (new/'STARTED.json').exists() and not (new/'responses').exists() and not (NEW/'RUN_STATUS.json').exists(),'INFERENCE_STARTED_BEFORE_INDEPENDENT_AUDIT')
 checks.update(population={'questions':14,'covered_dev_roots':7,'requests':168,'full_context_only':True,'ordering_only':True,'test_question_source_content_read':False},projection={'all_request_dicts_equal_both_v127_models':nr==oldreq==treq,'prompt_bytes_equal_all168':True,'canonical_and_gold_lossless_projection':True,'source_files_identical':7,'sampled_seed_offset_preserved':True,'no_new_generation':True},model={'identifier':'M0','path':str(M0),'adapter_enabled':False,'checkpoint_files_hashed':len(ident['checkpoint_files']),'shards':sorted(shards),'official_historical_audit_sha256':MODEL_AUDIT_SHA,'loaded_by_auditor':False},protocol={'decoder_contract':nm['decoder_contract'],'request_seed_min':min(r['seed'] for r in nr),'request_seed_max':max(r['seed'] for r in nr),'min_prompt_tokens':min(r['prompt_tokens'] for r in nr),'max_prompt_tokens':max(r['prompt_tokens'] for r in nr)})
 return checks,new,nm,cq

def score_side(run,manifest,canonical):
 reqs=[r for r in manifest['requests'] if r['question_type']=='ordering'];gold=read(run/'gold.json');qs={q['candidate_id']:q for q in canonical};out=[]
 expectedfiles={r['request_id']+'.json' for r in reqs};actualfiles={p.name for p in (run/'responses').glob('*ordering*.json')};check(actualfiles==expectedfiles,'RESPONSE_FILE_SET:'+str(run))
 for req in reqs:
  rp=run/'responses'/(req['request_id']+'.json');old=rp.is_relative_to(OLD)
  if old:old_frozen(rp)
  else:bind(rp)
  r=read(rp);s=r['score'];raw=r['raw_output'];m=re.fullmatch(r'<box>(I[1-6](?:,I[1-6]){5})</box>',raw)
  check(m is not None,'RAW_PARSE:'+str(rp))
  labels=m[1].split(',') if m else [];check(len(set(labels))==6,'SIX_UNIQUE:'+str(rp))
  pred=[req['variant']['display_to_content'][l] for l in labels];gg=gold[req['candidate_id']];pos={c:i for i,c in enumerate(pred)};corr=sum(pos[a]<pos[b] for a,b in itertools.combinations(gg,2));exact=pred==gg;partial=fractions.Fraction(2*corr-15,15)
  check(s['display_prediction']==labels and s['content_prediction']==pred and s['content_gold']==gg,'RESPONSE_MAPPING:'+str(rp))
  check(s['tp']==corr and s['fp']==15-corr and s['fn']==0 and s['exact_match']==exact,'RESPONSE_INDEPENDENT_SCORE:'+str(rp))
  check(fractions.Fraction(s['partial_score_fraction'])==partial and fractions.Fraction(s['partial_score'])==partial,'RESPONSE_SIGNED_PARTIAL:'+str(rp))
  check(r['evaluation_valid'] and not r['data_errors'] and s['parse_status']=='success' and s['scored'],'RESPONSE_INVALID:'+str(rp))
  for key in ['request_id','candidate_id','root_id','question_type','mode','slot','seed']:check(r[key]==req[key],'RESPONSE_BIND:'+key+':'+str(rp))
  check(r['variant_id']==req['variant']['variant_id'],'RESPONSE_VARIANT:'+str(rp))
  check(r['solver_model']==manifest['solver_model'] and r['model_path']==manifest['model'],'RESPONSE_MODEL_ID:'+str(rp))
  md=r['metadata'];check(md['sampling_seed']==req['seed'] and md['prompt_token_count']==req['prompt_tokens'] and md['finish_reason'] in ('stop','eos'),'RESPONSE_METADATA:'+str(rp))
  for key in ['temperature','top_p','max_tokens','structured_output_api','structured_output_version']:check(md[key]==manifest['decoder_contract'][key],'RESPONSE_DECODER:'+key+':'+str(rp))
  check(0<md['completion_token_count']<=32,'RESPONSE_COMPLETION_TOKENS:'+str(rp))
  q=qs[req['candidate_id']];text_to_item={t:i for i,t in q['items'].items()};route={c:q['order_anchor_spans'][text_to_item[t]]['turn_ref'].split(':')[1] for c,t in req['variant']['content_to_text'].items()}
  pcs={name:[0,0] for name in ['A_internal','B_internal','cross_route']}
  for aa,bb in itertools.combinations(gg,2):
   category=route[aa]+'_internal' if route[aa]==route[bb] else 'cross_route';pcs[category][0]+=pos[aa]<pos[bb];pcs[category][1]+=1
  block_violation=any(pos[aa]>pos[bb] for aa in gg for bb in gg if route[aa]=='A' and route[bb]=='B')
  out.append({'pair_categories':pcs,'block_violation':int(block_violation),'candidate_id':req['candidate_id'],'root_id':req['root_id'],'request_id':req['request_id'],'exact':int(exact),'correct_pairs':corr,'signed_partial':str(partial)})
 def stats(rs):return {'pair_categories':{c:{'correct':sum(r['pair_categories'][c][0] for r in rs),'total':sum(r['pair_categories'][c][1] for r in rs)} for c in ['A_internal','B_internal','cross_route']},'block_violation_calls':sum(r['block_violation'] for r in rs),'calls':len(rs),'exact_correct':sum(r['exact'] for r in rs),'exact_rate':sum(r['exact'] for r in rs)/len(rs),'correct_pairs':sum(r['correct_pairs'] for r in rs),'total_pairs':15*len(rs),'pair_accuracy':sum(r['correct_pairs'] for r in rs)/(15*len(rs)),'mean_signed_partial':str(sum((fractions.Fraction(r['signed_partial']) for r in rs),fractions.Fraction())/len(rs))}
 return {'overall':stats(out),'per_question':{cid:stats([r for r in out if r['candidate_id']==cid]) for cid in qs},'per_root':{root:stats([r for r in out if r['root_id']==root]) for root in sorted({r['root_id'] for r in out})},'per_call':out}

def main():
 global OLDROWS
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','final']);ap.add_argument('--run',default='run/base');a=ap.parse_args()
 bind(OLD/'COMPLETE_MANIFEST.json');OLDROWS={r['path']:r for r in read(OLD/'COMPLETE_MANIFEST.json')['files']}
 checks,new,nm,cq=preflight(a.run,final=a.stage=='final')
 if a.stage=='final':
  prior=read(NEW/'evidence/INDEPENDENT_PREFLIGHT_AUDIT.json');check(prior['status']=='PASS' and not prior['errors'],'PREFLIGHT_NOT_PASS')
  for row in prior['audited_files']:bind(row['path'],row['sha256'])
  summaries={}
  for name,path,manifest in [('M0',new,nm),('M_clean_policy',OLD/'run/baseline',read(OLD/'run/baseline/manifest.json')),('M_v125_newonly',OLD/'run/new',read(OLD/'run/new/manifest.json'))]:summaries[name]=score_side(path,manifest,cq)
  complete=read(new/'COMPLETE.json');check(complete['status']=='COMPLETE' and complete['actual_call_count']==168 and not complete['training_started'] and not complete['training_promotion'],'M0_NATIVE_COMPLETE')
  started=read(new/'STARTED.json');check(started['manifest_sha256']==sha(new/'manifest.json'),'NATIVE_START_MANIFEST')
  check(not (new/'FAILED.json').exists(),'FAILED_RUN_PRESENT')
  check(prior['created_at_utc']<started['started_at_utc'],'AUDIT_DID_NOT_PRECEDE_INFERENCE')
  state=read(NEW/'RUN_STATUS.json');check(state['status']=='BASE_CONTROL_REPORTED_AWAITING_INDEPENDENT_AUDIT' and state['actual_new_calls']==168 and state['test_evaluation'] is False and state['automatic_retry'] is False,'RUN_STATUS_NOT_COMPLETE')
  bind(NEW/'RUN_STATUS.json')
  results=read(new/'RESULTS.json');check(not results['call_set_errors'] and results['ground_calls']==0 and len(results['questions'])==14,'M0_NATIVE_RESULTS')
  for q in results['questions']:
   ref=summaries['M0']['per_question'][q['candidate_id']];check(fractions.Fraction(q['full_exact_rate'])==fractions.Fraction(ref['exact_correct'],12),'NATIVE_EXACT_SUMMARY:'+q['candidate_id']);check(fractions.Fraction(q['full_partial_mean'])==fractions.Fraction(ref['mean_signed_partial']),'NATIVE_PARTIAL_SUMMARY:'+q['candidate_id'])
  bind(new/'COMPLETE.json');bind(new/'RESULTS.json');bind(new/'STARTED.json')
  checks['independently_recomputed_results']=summaries
  summary=read(NEW/'SUMMARY.json');check(summary['status']=='COMPLETE' and not summary['errors'],'REPORTED_SUMMARY_NOT_COMPLETE');bind(NEW/'SUMMARY.json')
  for key,mid in [('base','M0'),('baseline','M_clean_policy'),('new','M_v125_newonly')]:
   reported=summary['models'][key];rs=reported['summary'];ref=summaries[mid]['overall'];check(reported['solver_model']==mid,'REPORT_MODEL_KEY:'+key)
   check(rs['planned_calls']==168 and rs['valid_calls']==168 and rs['exact_correct']==ref['exact_correct'] and abs(rs['exact_rate_fixed']-ref['exact_rate'])<1e-12,'REPORT_EXACT_COUNTS:'+key)
   check(abs(rs['metrics']['pair_accuracy']['mean']-ref['pair_accuracy'])<1e-12,'REPORT_PAIR_MEAN:'+key)
   check(abs(rs['metrics']['signed_partial']['mean']-float(fractions.Fraction(ref['mean_signed_partial'])))<1e-12,'REPORT_PARTIAL_MEAN:'+key)
   check(abs(rs['metrics']['block_violation']['mean']-ref['block_violation_calls']/168)<1e-12,'REPORT_BLOCK_VIOLATION:'+key)
   for cat in ['A_internal','B_internal','cross_route']:
    rp=rs['pair_categories'][cat];pc=ref['pair_categories'][cat];check(rp['planned_pairs']==pc['total'] and rp['known_correct_pairs']==pc['correct'] and rp['unknown_pairs']==0,'REPORT_PAIR_CATEGORY_COUNTS:'+key+':'+cat)
    metric=cat+'_accuracy' if cat!='cross_route' else 'cross_route_accuracy';check(abs(rs['metrics'][metric]['mean']-pc['correct']/pc['total'])<1e-12,'REPORT_PAIR_CATEGORY_MEAN:'+key+':'+cat)
  checks['threeway_report_crosscheck']='All three model identity/count/exact/pair/signed-partial/block-violation/A-internal/B-internal/cross-route figures independently match raw outputs.'
 result={'schema_version':'v130_independent_'+a.stage+'_audit_v1','status':'FAIL' if ERRORS else 'PASS','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'auditor':'/root/baseline_order_errors','errors':ERRORS,'checks':checks,'audited_files':[{'path':p,'sha256':h} for p,h in sorted(CHECKED.items())],'test_inference_calls':0,'audit_model_inference_calls':0}
 op=NEW/'evidence'/('INDEPENDENT_PREFLIGHT_AUDIT.json' if a.stage=='preflight' else 'INDEPENDENT_FINAL_AUDIT.json')
 if op.exists():raise RuntimeError('REFUSE_OVERWRITE:'+str(op))
 op.parent.mkdir(parents=True,exist_ok=True);op.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'status':result['status'],'errors':ERRORS,'path':str(op),'audited_files':len(CHECKED),'summaries':{k:v['overall'] for k,v in checks.get('independently_recomputed_results',{}).items()}},ensure_ascii=False,indent=2))
 return bool(ERRORS)
if __name__=='__main__':sys.exit(main())
