import json, pathlib, hashlib, re, math, statistics, itertools, csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
P=pathlib.Path('/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/day3_targeted_4root_v1')
R=P/'runtime/direct_v1'
def read(p): return json.loads(pathlib.Path(p).read_text())
def digest(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
checks=Counter();fail=[]
def ck(category,condition,detail=None):
 checks[category]+=1
 if not condition:fail.append({'category':category,'detail':detail})
def eq(category,a,b,detail=None):
 ok=math.isclose(a,b,abs_tol=1e-12,rel_tol=1e-12) if isinstance(a,(int,float)) and isinstance(b,(int,float)) else a==b
 ck(category,ok,{'context':detail,'independent':a,'recorded':b} if not ok else None)
M=read(R/'manifest.json');D=read(R/'DISPLAY_MAPPINGS.json');S=read(R/'SCORES.json');F=read(R/'RAW_RESPONSE_FREEZE.json');PRE=read(R/'PREFLIGHT_FREEZE.json');SS=read(R/'SCORING_STARTED.json');START=read(R/'STARTED.json');COMP=read(R/'COMPLETE.json');SUM=read(P/'SUMMARY.json');GF=read(P/'GENERATION_FREEZE.json');SF=read(P/'SOURCE_FREEZE.json');ADM=read(P/'CONTENT_ADMISSION.json');EX=read(P/'EXECUTION_FREEZE.json');PRO=read(R/'PROTOCOL.json');LED=read(P/'exports/CANDIDATE_LEDGER.json');PUB=read(P/'exports/PUBLIC_QUESTIONS.json');PRIV=read(P/'exports/PRIVATE_QUESTIONS.json')
Q={x['candidate_id']:x for x in PRIV['questions']};U={x['candidate_id']:x for x in PUB['questions']};L={x['candidate_id']:x for x in LED['records']};SQ={x['candidate_id']:x for x in S['questions']};SUQ={x['candidate_id']:x for x in SUM['questions']};SR={x['request_id']:x for x in S['requests']};EXP={x['request_id']:x for x in map(json.loads,(P/'exports/SOLVER_RESPONSES.jsonl').read_text().splitlines())};PER={x['candidate_id']:x for x in read(P/'exports/PER_QUESTION.json')['questions']};CSV={x['candidate_id']:x for x in csv.DictReader((P/'per_question.csv').open())}
A={}
for f in ADM['author_bindings']:
 for line in pathlib.Path(f['path']).read_text().splitlines():
  x=json.loads(line);A[x['candidate_id']]=x
# Hash all explicit path+sha256+bytes bindings in all final/current evidence documents,
# and descendants reached through JSON binding files in this pilot only.
seed=['GENERATION_FREEZE.json','SOURCE_FREEZE.json','CONTENT_ADMISSION.json','EXECUTION_FREEZE.json','SUMMARY.json','exports/CANDIDATE_LEDGER.json','review/REPORT_RECOMPUTATION.json','review/INDEPENDENT_PUBLIC_PREFLIGHT.json','runtime/direct_v1/manifest.json','runtime/direct_v1/PREFLIGHT_FREEZE.json','runtime/direct_v1/RAW_RESPONSE_FREEZE.json','runtime/direct_v1/SCORING_STARTED.json','runtime/direct_v1/SCORES.json']
queue=[P/f for f in seed];visited=set();bindings={};binding_owners=defaultdict(list)
def collect(x,owner):
 if isinstance(x,dict):
  if isinstance(x.get('path'),str) and isinstance(x.get('sha256'),str) and isinstance(x.get('bytes'),int):
   k=(x['path'],x['sha256'],x['bytes']);bindings[k]=x;binding_owners[k].append(owner)
   p=pathlib.Path(x['path'])
   if p.suffix=='.json' and p.is_relative_to(P) and p not in visited and '/responses/' not in str(p) and p.name!='PROMPT_TOKEN_IDS.json':queue.append(p)
  for v in x.values():collect(v,owner)
 elif isinstance(x,list):
  for v in x:collect(v,owner)
while queue:
 p=queue.pop()
 if p in visited:continue
 visited.add(p)
 try:collect(read(p),str(p))
 except Exception as e:ck('binding_document_read',False,{'path':str(p),'error':str(e)})
binding_results=[];hash_cache={}
for (path,sha,n),b in sorted(bindings.items()):
 p=pathlib.Path(path);exists=p.is_file();actual_size=p.stat().st_size if exists else None
 if exists and path not in hash_cache:hash_cache[path]=digest(p)
 actual_sha=hash_cache.get(path);ok=exists and actual_size==n and actual_sha==sha
 ck('all_final_explicit_bindings',ok,{'binding':b,'actual_bytes':actual_size,'actual_sha256':actual_sha})
 binding_results.append({'path':path,'expected_sha256':sha,'bytes':n,'verified':ok})
# Byte equality and immutable Gold-source-author-public link.
eq('public_runtime_copy',(R/'PUBLIC_QUESTIONS.json').read_bytes(),(P/'exports/PUBLIC_QUESTIONS.json').read_bytes())
eq('private_runtime_copy',pathlib.Path(S['private_answer_binding']['path']).read_bytes(),(P/'exports/PRIVATE_QUESTIONS.json').read_bytes())
eq('production_slots',len(GF['slots']),24);eq('production_slots',len(L),24);eq('production_slots',ADM['admitted_count'],23)
eq('sets',set(Q),set(U));eq('sets',set(Q),set(SQ));eq('sets',set(Q),set(SUQ));eq('sets',set(Q),set(PER));eq('sets',set(Q),set(CSV))
for cid,q in Q.items():
 eq('author_gold',q['gold'],A[cid]['gold'],cid)
 eq('public_gold_removed',{k:v for k,v in q.items() if k!='gold'},U[cid],cid)
 eq('source_bound',q['source_sha256'],digest(q['source_path']),cid)
 eq('content_admission',L[cid]['status'],'CONTENT_PASS',cid)
 eq('gold_summary',q['gold'],SUQ[cid]['canonical_gold'],cid)
 ck('train_only_author',A[cid]['split'].lower()=='train',cid)
 for label,text in (q.get('options') or q.get('items')).items():
  eq('canonical_content_hash',D['public_to_canonical'][cid][label],'content_'+hashlib.sha256(text.encode()).hexdigest()[:16],{'cid':cid,'label':label})
missing=set(L)-set(Q)
eq('shortfall_retained',missing,{'T4R-R4-ORD3'})
eq('shortfall_retained',L['T4R-R4-ORD3']['status'],'SOURCE_SHORTFALL')
eq('shortfall_retained',M['missing_candidate_ids'],['T4R-R4-ORD3'])
eq('shortfall_retained',SUM['ordering_unadmitted_slots_without_actual_arity'],['T4R-R4-ORD3'])
ck('shortfall_no_fabricated_arity',L['T4R-R4-ORD3']['item_count'] is None)
eq('production_yield',SUM['production_yield']['admitted_slots'],23);eq('production_yield',SUM['production_yield']['planned_slots'],24)
# Full sources/prefix preservation, routes, question-source provenance.
source_stats={}
for g in GF['roots']:
 root=g['root_id'];fs=SF['sources'][root];prefix=read(g['prefix']['path']);source=read(fs['source']['path'])
 eq('prefix_preserved',source['turns'][:len(prefix['turns'])],prefix['turns'],root)
 eq('source_turns',len(source['turns']),fs['turns'],root)
 eq('source_turns',len(source['turns'])-len(prefix['turns']),fs['added_turns'],root)
 if fs['mode']=='targeted_extension':eq('scene_count',len(source['targeted_extension_scenes']),6,root)
 source_stats[root]={'mode':fs['mode'],'prefix_turns':len(prefix['turns']),'turns':len(source['turns']),'added_turns':len(source['turns'])-len(prefix['turns'])}
eq('total_turns',sum(x['turns'] for x in source_stats.values()),923)
eq('new_turns',sum(x['added_turns'] for x in source_stats.values()),186)
# All 342 responses parsed independently: literal syntax -> display labels -> canonical labels.
raw_paths={pathlib.Path(x['path']).stem:x['path'] for x in F['files']}
reqs={x['request_id']:x for x in M['requests']}
for name,ids in [('freeze',raw_paths),('scores',SR),('export',EXP),('display',D['requests'])]:eq('request_id_sets',set(ids),set(reqs),name)
eq('request_count',len(reqs),342);eq('request_count',len(reqs),M['request_count']);eq('request_count',len(reqs),F['planned_request_count'])
TOKENIDS=read(R/'PROMPT_TOKEN_IDS.json');byq=defaultdict(list);rows=[]
for rid,r in reqs.items():
 cid=r['candidate_id'];q=Q[cid];raw=read(raw_paths[rid]);score=SR[rid];e=EXP[rid];mp=D['requests'][rid];qt=q['question_type'];canonical_items=q.get('options') or q.get('items');n=len(canonical_items)
 eq('response_identity',raw['request_id'],rid);eq('response_identity',raw['candidate_id'],cid);eq('response_identity',raw['root_id'],q['root_id']);eq('response_identity',raw['question_type'],qt)
 eq('prompt_token_ids',raw['prompt_token_ids'],TOKENIDS[rid],rid)
 eq('prompt_token_count',len(raw['prompt_token_ids']),r['prompt_tokens'],rid)
 eq('prompt_token_count',len(raw['prompt_token_ids']),raw['metadata']['prompt_token_count'],rid)
 prompt=(R/'prompts'/f'{rid}.txt').read_bytes();eq('prompt_sha',hashlib.sha256(prompt).hexdigest(),r['prompt_sha256'],rid);eq('prompt_sha',raw['prompt_sha256'],r['prompt_sha256'],rid)
 eq('source_turns',r['source_turns'],source_stats[q['root_id']]['turns'],rid);eq('source_bound',r['source_sha256'],q['source_sha256'],rid)
 eq('seed',raw['seed'],r['seed'],rid)
 for flag in ['gold_read_by_solver','automatic_retry','source_truncation']:eq('runtime_flags',raw[flag],False,{'rid':rid,'flag':flag})
 ctoc={v:k for k,v in D['public_to_canonical'][cid].items()};dtoc={label:ctoc[content] for label,content in mp['display_to_content'].items()}
 eq('mapping_bijection',set(dtoc.values()),set(canonical_items),rid)
 eq('mapping_display_text',r['display'].get('options') or r['display'].get('items'),{k:canonical_items[v] for k,v in dtoc.items()},rid)
 eq('mapping_export',dtoc,e['display_to_canonical'],rid)
 match=re.fullmatch(r'<box>([^<>]*)</box>',raw['raw_output'].strip())
 pred=match.group(1).split(',') if match else []
 valid=bool(match) and len(pred)==len(set(pred)) and all(x in dtoc for x in pred) and (bool(pred) if qt=='multi_select' else set(pred)==set(dtoc)) and raw['status']=='RETURNED'
 eq('validity',valid,score['valid'],rid);eq('validity',valid,e['valid'],rid)
 canonical=[dtoc[x] for x in pred] if valid else None;gold=q['gold'];gset=set(gold)
 exact=bool(valid and (set(canonical)==gset if qt=='multi_select' else canonical==gold))
 eq('exact',exact,score['exact'],rid);eq('exact',exact,e['exact'],rid)
 eq('display_prediction',pred,score['native_score']['display_prediction'],rid);eq('display_prediction',pred,e['display_prediction'],rid)
 # Canonical MC arrays are sets; do not treat display output order as prediction error.
 eq('canonical_prediction',set(canonical) if qt=='multi_select' else canonical,set(score['canonical_prediction']) if qt=='multi_select' else score['canonical_prediction'],rid)
 eq('canonical_prediction_export',set(canonical) if qt=='multi_select' else canonical,set(e['canonical_prediction']) if qt=='multi_select' else e['canonical_prediction'],rid)
 eq('gold_mapping',gold,score['gold_canonical'],rid);eq('gold_mapping',gold,e['canonical_gold'],rid)
 display_gold=[k for g in gold for k,v in dtoc.items() if v==g]
 eq('display_gold_exact_invariance',exact,(set(pred)==set(display_gold) if qt=='multi_select' else pred==display_gold),rid)
 z={'request_id':rid,'candidate_id':cid,'root_id':q['root_id'],'source_mode':source_stats[q['root_id']]['mode'],'question_type':qt,'item_count':n,'template_id':L[cid]['template_id'],'primary_relation':L[cid]['primary_relation'],'valid':valid,'exact':exact,'display_prediction':pred,'canonical_prediction':canonical,'canonical_gold':gold,'display_gold':display_gold,'prompt_tokens':len(raw['prompt_token_ids']),'completion_tokens':len(raw['completion_token_ids'])}
 if qt=='multi_select':
  pset=set(canonical);tp=len(pset&gset);fp=len(pset-gset);fn=len(gset-pset);j=tp/len(pset|gset)
  z.update(tp=tp,fp=fp,fn=fn,jaccard=j,response_error_category='exact' if exact else 'FP_and_FN' if fp and fn else 'FP_only' if fp else 'FN_only',nonexact_with_partial=not exact and tp>0)
  for name,value in [('tp',tp),('fp',fp),('fn',fn),('jaccard',j)]:eq('MC_metrics',value,score['metrics'][name],{'rid':rid,'metric':name})
  for name,value in [('true_positive_options',tp),('false_positive_options',fp),('false_negative_options',fn),('jaccard',j)]:eq('MC_export',value,e[name],{'rid':rid,'metric':name})
  eq('native_MC_normalized_TP_minus_FP',float(Fraction(score['native_score']['partial_score_fraction'])),tp/len(gset)-fp/(n-len(gset)) if len(gset)<n else tp/len(gset),rid)
  eq('MC_item_count',len(score['items']),n,rid)
  for item in score['items']:
   label=item['canonical_label'];outcome=('TP' if label in gset else 'FP') if label in pset else ('FN' if label in gset else 'TN')
   eq('MC_item_outcome',item['outcome'],outcome,{'rid':rid,'label':label})
  eq('MC_FP_labels',set(e['FP_labels']),pset-gset,rid);eq('MC_FN_labels',set(e['FN_labels']),gset-pset,rid)
 else:
  ranks={x:i for i,x in enumerate(canonical)};pairs=list(itertools.combinations(gold,2));pc=sum(ranks[a]<ranks[b] for a,b in pairs);pt=n*(n-1)//2
  z.update(pair_correct=pc,pair_total=pt,pair_accuracy=pc/pt,nonexact_with_partial=not exact and pc>0)
  for name,value in [('correct_pairs',pc),('total_pairs',pt),('pair_accuracy',pc/pt)]:eq('ordering_metrics',value,score['metrics'][name],{'rid':rid,'metric':name})
  eq('ordering_export',pc,e['pair_correct'],rid);eq('ordering_export',pt,e['pair_total'],rid)
  eq('ordering_pairs',len(score['pairs']),pt,rid)
  for pair in score['pairs']:
   a,b=pair['before_canonical_label'],pair['after_canonical_label'];ck('ordering_Gold_pair',(a,b) in pairs,{'rid':rid,'pair':[a,b]});eq('ordering_pair_bit',ranks[a]<ranks[b],pair['correct'],{'rid':rid,'pair':[a,b]})
  eq('native_ordering_Kendall_tau',float(Fraction(score['native_score']['partial_score_fraction'])),(2*pc-pt)/pt,rid)
 eq('nonexact_partial_export',z['nonexact_with_partial'],e['nonexact_with_partial'],rid)
 eq('export_raw_output',raw['raw_output'],e['raw_output'],rid)
 rows.append(z);byq[cid].append(z)
 eq('frozen_before_gold',raw['response_committed_at_utc']<F['created_at_utc']<S['gold_first_read_at_utc'],True,rid)
# Independent per-question aggregate and fixed display/sample design.
qrows={}
for cid,rr in byq.items():
 q=Q[cid];qt=q['question_type'];n=len(q.get('options') or q.get('items'));rate=sum(x['exact'] for x in rr)/len(rr)
 expected=18 if qt=='multi_select' else min(12,math.factorial(n));eq('fixed_calls_per_question',len(rr),expected,cid)
 variants=defaultdict(list)
 for x in rr:variants[reqs[x['request_id']]['variant_id']].append(x)
 eq('fixed_display_count',len(variants),6 if qt=='multi_select' else min(12,math.factorial(n)),cid)
 eq('fixed_samples_per_display',sorted(len(x) for x in variants.values()),[3]*6 if qt=='multi_select' else [1]*len(variants),cid)
 displays=[]
 for vv in variants.values():displays.append(tuple(D['requests'][vv[0]['request_id']]['display_order_content_ids']))
 eq('distinct_displays',len(set(displays)),len(displays),cid)
 z={'candidate_id':cid,'root_id':q['root_id'],'source_mode':source_stats[q['root_id']]['mode'],'question_type':qt,'item_count':n,'template_id':L[cid]['template_id'],'primary_relation':L[cid]['primary_relation'],'calls':len(rr),'valid_calls':sum(x['valid'] for x in rr),'exact_calls':sum(x['exact'] for x in rr),'exact_rate':rate,'partial_metric':statistics.mean(x['jaccard'] if qt=='multi_select' else x['pair_accuracy'] for x in rr),'nonexact_with_partial_calls':sum(x['nonexact_with_partial'] for x in rr)}
 if qt=='multi_select':z.update(fp_option_events=sum(x['fp'] for x in rr),fn_option_events=sum(x['fn'] for x in rr),gold_count=len(q['gold']))
 else:z.update(pair_correct=sum(x['pair_correct'] for x in rr),pair_total=sum(x['pair_total'] for x in rr))
 qrows[cid]=z
 for name,val in [('denominator_calls',z['calls']),('valid_calls',z['valid_calls']),('exact_calls',z['exact_calls']),('exact_rate',rate)]:
  eq('question_scores',val,SQ[cid][name],{'cid':cid,'metric':name});eq('question_summary',val,SUQ[cid][name],{'cid':cid,'metric':name});eq('question_export',val,PER[cid][name],{'cid':cid,'metric':name});eq('question_csv',float(val),float(CSV[cid][name]),{'cid':cid,'metric':name})
 eq('question_partial',z['partial_metric'],SUQ[cid]['partial_metric'],cid)
 if qt=='multi_select':
  eq('question_FP_events',z['fp_option_events'],SQ[cid]['fp_calls'],cid);eq('question_FN_events',z['fn_option_events'],SQ[cid]['fn_calls'],cid)
# All reporting strata are recomputed using questions as primary independent units.
def agg(rr):
 ids={x['candidate_id'] for x in rr};qs=[qrows[c] for c in ids];roots={x['root_id'] for x in rr}
 out={'questions':len(ids),'calls':len(rr),'exact_calls':sum(x['exact'] for x in rr),'question_equal_exact':statistics.mean(q['exact_rate'] for q in qs),'call_equal_exact':sum(x['exact'] for x in rr)/len(rr),'admitted_roots':len(roots),'root_equal_exact':statistics.mean(statistics.mean(q['exact_rate'] for q in qs if q['root_id']==root) for root in roots),'nonexact_with_partial_calls':sum(x['nonexact_with_partial'] for x in rr),'nonexact_with_partial_rate':sum(x['nonexact_with_partial'] for x in rr)/len(rr)}
 types={x['question_type'] for x in rr}
 if len(types)==1:
  out['question_equal_partial_metric']=statistics.mean(q['partial_metric'] for q in qs)
  if 'multi_select' in types:out.update(question_equal_jaccard=out['question_equal_partial_metric'],FP_option_events=sum(x['fp'] for x in rr),FN_option_events=sum(x['fn'] for x in rr))
  else:out.update(correct_pairs=sum(x['pair_correct'] for x in rr),total_pairs=sum(x['pair_total'] for x in rr),question_equal_pair_accuracy=out['question_equal_partial_metric'],pair_weighted_accuracy=sum(x['pair_correct'] for x in rr)/sum(x['pair_total'] for x in rr))
 return out
def compareagg(ind,reported,context):
 for key,val in ind.items():eq('aggregate_'+context,val,reported.get(key),key)
result={'overall':agg(rows),'by_type':{},'by_root':{},'by_source_mode_descriptive_only':{},'by_template':{},'ordering_by_relation':{},'ordering_by_arity':{}}
compareagg(result['overall'],SUM['overall'],'overall')
for target,key,values in [('by_type','question_type',sorted({x['question_type'] for x in rows})),('by_root','root_id',sorted(source_stats)),('by_source_mode_descriptive_only','source_mode',sorted({x['source_mode'] for x in rows})),('by_template','template_id',sorted({x['template_id'] for x in rows})),('ordering_by_relation','primary_relation',sorted({x['primary_relation'] for x in rows if x['question_type']=='ordering'})),('ordering_by_arity','item_count',sorted({x['item_count'] for x in rows if x['question_type']=='ordering'}))]:
 for val in values:
  rr=[x for x in rows if x[key]==val and (x['question_type']=='ordering' if target.startswith('ordering_') else True)];a=agg(rr);result[target][str(val)]=a
  reported=SUM['roots'][val]['scores'] if target=='by_root' else SUM[target][str(val)]
  compareagg(a,reported,target)
# Scalar score-file summaries (SCORES by-type uses its own field names).
for metric,val in [('denominator_calls',342),('denominator_questions',23),('denominator_roots',4),('valid_calls',sum(x['valid'] for x in rows)),('exact_calls',sum(x['exact'] for x in rows)),('question_equal_exact',result['overall']['question_equal_exact']),('root_equal_exact',result['overall']['root_equal_exact']),('production_denominator',24),('production_admitted_numerator',23),('production_unadmitted_slots',1)]:eq('score_global',val,S[metric],metric)
mc=[x for x in rows if x['question_type']=='multi_select'];mcerrs={'denominator_calls':len(mc),'valid_calls':sum(x['valid'] for x in mc),'unknown_invalid_calls':sum(not x['valid'] for x in mc),'FP_option_events':sum(x['fp'] for x in mc),'FN_option_events':sum(x['fn'] for x in mc),'responses_with_FP':sum(x['fp']>0 for x in mc),'responses_with_FN':sum(x['fn']>0 for x in mc)}
eq('MC_observable_errors',mcerrs,SUM['MC_observable_errors'])
mc_categories=dict(Counter(x['response_error_category'] for x in mc))
# Option-level observable summary.
for item in SUM['MC_option_observations']:
 rr=byq[item['candidate_id']];label=item['canonical_label'];gold=label in Q[item['candidate_id']]['gold'];selected=sum(label in x['canonical_prediction'] for x in rr)
 eq('option_observations',selected,item['selected_calls'],item['candidate_id']+':'+label)
 eq('option_observations',0 if gold else selected,item['FP_events'],item['candidate_id']+':'+label)
 eq('option_observations',len(rr)-selected if gold else 0,item['FN_events'],item['candidate_id']+':'+label)
# Token costs and execution chronology.
prompt_tokens=sum(x['prompt_tokens'] for x in rows);completion_tokens=sum(x['completion_tokens'] for x in rows)
for key,val in [('known_prompt_tokens',prompt_tokens),('known_completion_tokens',completion_tokens),('known_total_tokens',prompt_tokens+completion_tokens),('planned_calls',342),('returned_calls',342),('submitted_attempts',342),('not_submitted_calls',0),('requests_with_unknown_usage',0)]:eq('cost',val,S['cost'][key],key)
ck('chronology',GF['created_at_utc']<SF['created_at_utc']<ADM['created_at_utc']<=EX['created_at_utc']<PRE['created_at_utc']<F['created_at_utc']<S['gold_first_read_at_utc']<S['scored_at_utc']<SUM['created_at_utc'])
eq('scoring_started',SS['gold_first_read_at_utc'],S['gold_first_read_at_utc'])
# Prior pilot / old187 integrity: hash only; do not consume their question or raw content.
prior=[];prior_before=read(P/'review/SEALED_PRIOR_INTEGRITY_BEFORE_SOLVER.json')
for old in prior_before['checks']:
 d=P.parent/old['directory'];mp=d/'FINAL_MANIFEST.json';om=read(mp);errors=[]
 eq('prior_manifest_unchanged',digest(mp),old['manifest_sha256'],old['directory'])
 for b in om['files']:
  path=pathlib.Path(b['path']);ok=path.is_file() and path.stat().st_size==b['bytes'] and digest(path)==b['sha256']
  ck('prior_files_unchanged',ok,b['path'])
  if not ok:errors.append(b['path'])
 prior.append({'directory':old['directory'],'manifest_sha256':digest(mp),'files_verified':len(om['files']),'errors':errors,'method':'Hash-only integrity check; question/response contents not parsed.'})
# Scope checks and narrative interpretation; no training or heldout inference in current artifacts.
for key,val in [('training_started',False),('new_training_admitted',0),('feedback_causal_effect_established',False),('controlled_difficulty_increase_established',False),('training_benefit_established',False),('dev_test_calls',0),('source_masked_control_performed',False)]:eq('scope',SUM[key],val,key)
eq('scope',M['training_started'],False);eq('scope',M['use_as_final_eval'],False);eq('scope',LED['solver_scores_used_for_admission'],False);eq('scope',ADM['solver_scores_available'],False);eq('scope',ADM['training_admission'],False);eq('scope',ADM['dev_test_used'],False)
for key in ['A_B_effect_estimate','use_as_final_eval','training_started','automatic_retry','source_truncation']:eq('scope',S[key],False,key)
report=(P/'REPORT_ZH.md').read_text()
for phrase in ['不是随机对照实验','不能用于证明扩充反馈有效','没有反馈因果 A/B','不同项数','未按分数重新准入']:ck('narrative_limits',phrase in report,phrase)
eq('narrative_exact',SUM['by_type']['multi_select']['question_equal_exact'],105/216)
eq('narrative_exact',SUM['by_type']['ordering']['question_equal_exact'],sum(qrows[c]['exact_rate'] for c in qrows if Q[c]['question_type']=='ordering')/11)
# Explicit MC Gold quotas preserved; procedure shortfall omitted from arity, not hidden.
golds=dict(Counter(str(len(q['gold'])) for q in Q.values() if q['question_type']=='multi_select'))
eq('MC_Gold_distribution',golds,SUM['gold_count_distribution_MC']);eq('MC_Gold_distribution',golds,SUM['planned_MC_gold_count_distribution'])
eq('arity_distribution',dict(Counter(str(q['item_count']) for q in qrows.values() if q['question_type']=='ordering')),{'3':1,'4':5,'5':1,'6':4})
# Additional source-citation export verification, without changing source/question content.
COV=read(P/'exports/SOURCE_COVERAGE.json');ind_cov={};total_new=0;total_cited=0
def refs(x):
 found=set()
 if isinstance(x,dict):
  for k,v in x.items():
   if k=='turn_ref' and isinstance(v,str):found.add(v)
   else:found.update(refs(v))
 elif isinstance(x,list):
  for v in x:found.update(refs(v))
 return found
for root,st in source_stats.items():
 cids=[c for c in Q if Q[c]['root_id']==root];turnrefs=set();scenes=defaultdict(list)
 for cid in cids:
  rr=refs(A[cid]);turnrefs.update(rr)
  for sc in sorted({':'.join(r.split(':')[:3]) for r in rr if r.startswith('T4R:')}):scenes[sc].append(cid)
 source=read(P/'sources'/f'{root}.json');allsc={x['scene_ref'] for x in source.get('targeted_extension_scenes',[])}
 expected=COV['roots'][root];eq('source_coverage',len(turnrefs),expected['evidence_unique_turns'],root);eq('source_coverage',dict(scenes),expected['new_evidence_scenes'],root);eq('source_coverage',sorted(allsc-set(scenes)),expected['new_scenes_without_question_evidence'],root)
 ind_cov[root]={'new_scenes':len(allsc),'cited_new_scenes':len(scenes),'unreferenced_new_scenes':sorted(allsc-set(scenes))};total_new+=len(allsc);total_cited+=len(scenes)
eq('source_coverage',total_new,COV['new_scenes_total']);eq('source_coverage',total_cited,COV['new_scenes_cited']);eq('source_coverage',(total_cited,total_new),(9,12))
correction={'status':'AUDITOR_DEFINITION_CORRECTED_AND_RECOMPUTED','initial_failures':96,'initial_failure_category':'native_MC_jaccard','cause':'The independent auditor initially assumed legacy native_score.partial_score_fraction was Jaccard. Frozen scoring.py actually defines normalized TP-minus-FP. The report Jaccard and all other 24035 initial checks were already correct.','correct_native_MC_definition':'TP/g − FP/(n−g) when g<n; TP/g when g=n','report_MC_definition':'TP/|prediction union Gold| (Jaccard)','data_or_score_or_report_changed':False,'initial_fail_artifact':{'path':str(P/'review/history/INDEPENDENT_POST_SCORE_AUDIT.initial_metric_definition_FAIL.json'),'sha256':digest(P/'review/history/INDEPENDENT_POST_SCORE_AUDIT.initial_metric_definition_FAIL.json')},'initial_tool_artifact':{'path':str(P/'review/history/AUDIT_POST_SCORE_TOOL.initial_metric_definition.py'),'sha256':digest(P/'review/history/AUDIT_POST_SCORE_TOOL.initial_metric_definition.py')}}

# Recheck immutable files after computations; final audit plus explicitly requested initial history are the only pilot outputs.
output=P/'review/INDEPENDENT_POST_SCORE_AUDIT.json'
input_files=[P/'REPORT_ZH.md',P/'SUMMARY.json',P/'per_question.csv',P/'exports/SOLVER_RESPONSES.jsonl',P/'exports/PER_QUESTION.json',P/'exports/PRIVATE_QUESTIONS.json']
input_binding=[{'path':str(p),'sha256':digest(p),'bytes':p.stat().st_size} for p in input_files]
notes=[
 '342 repeated requests and 23 admitted questions are not 342 independent questions; four historically exposed Train roots remain the root-level sample.',
 'The two targeted and two reused roots were selected purposively. The independent recomputation supports operational descriptive reporting only and no feedback causal effect.',
 'Arity3/4/5/6 is stratified. Arity, source, question lineage and relation type remain confounded; exact accuracy cannot establish a controlled difficulty increase.',
 'Native MC partial_score_fraction is TP/Gold_count minus FP/nonGold_count; it is not Jaccard. Native ordering partial_score_fraction is Kendall tau; the report deliberately uses correct Gold pairs / all Gold pairs. They are different metrics, and both were independently verified.',
 'fp_calls/fn_calls in SCORES per-question MC records count option events. Response-level FP-only/FN-only/both categories are separately recomputed here.',
 'Legal-answer constrained decoding explains 100% validity; it is not evidence of unconstrained format compliance.',
 'Semantic answer reliability remains conditional on frozen source and completed pre-score independent content audits. This postscore audit does not turn low Solver score into Gold validation.',
 'No model calls or training were performed by this audit. Current runtime manifests, raw flags, unchanged model bindings and prior sealed manifests are consistent with no training, heldout calls, or old187 modification; an artifact audit cannot prove the absence of unlogged filesystem activity.',
 'All admitted questions were evaluated, with no postscore selection. R4-ORD3 remains a source shortfall in the fixed24-slot production denominator and has no fabricated Solver response or actual arity.',
 'High intrinsic procedural cues and several self-contained quantitative/constraint stems limit inference about memory reliance; no source-masked or memory ablation was run.'
]
out={'schema_version':'targeted4root_independent_postscore_audit_v1','status':'PASS' if not fail else 'FAIL','created_at_utc':datetime.now(timezone.utc).isoformat(),'auditor':'different-author independent audit agent mc_dataset_forms_v2','audit_scope':'Postscore only after parent completion notification; independently parsed every raw output and recomputed display/canonical/Gold exact, MC FP/FN/Jaccard, all ordering pairs, strata, bindings and immutable prior integrity. No model calls or scorer/report imports.','inference_calls':0,'training_calls':0,'private_gold_opened':True,'written_pilot_files':[str(output),correction['initial_fail_artifact']['path'],correction['initial_tool_artifact']['path']],'auditor_definition_correction':correction,'source_citation_coverage_independent':ind_cov,'check_count':sum(checks.values()),'check_counts_by_category':dict(checks),'failure_count':len(fail),'failures':fail,'denominators':{'production_slots':24,'admitted_questions':23,'unadmitted_shortfall_slots':1,'planned_roots':4,'returned_valid_calls':342,'MC_questions':12,'MC_calls':216,'ordering_questions':11,'ordering_calls':126},'independent_results':result,'MC_observable_errors':mcerrs,'MC_response_error_categories':mc_categories,'MC_gold_count_distribution':golds,'source_statistics':source_stats,'prompt_tokens':{'min':min(x['prompt_tokens'] for x in rows),'median':statistics.median(x['prompt_tokens'] for x in rows),'max':max(x['prompt_tokens'] for x in rows),'sum':prompt_tokens},'completion_tokens':completion_tokens,'all_final_explicit_binding_records':len(bindings),'all_final_explicit_bound_unique_files':len(hash_cache),'model_bytes_hashed':sum(b['bytes'] for b in M['model_bindings']),'bindings':binding_results,'prior_integrity':prior,'final_report_export_bindings':input_binding,'per_question_independent':list(qrows.values()),'per_request_independent':rows,'interpretation_and_limits':notes}
output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':out['status'],'checks':sum(checks.values()),'failure_count':len(fail),'failures':fail[:25],'audit_path':str(output),'sha256':digest(output),'bytes':output.stat().st_size,'overall':result['overall'],'mc_categories':mc_categories,'bindings':len(bindings),'prior_integrity':prior},ensure_ascii=False,indent=2))
