"""Independent stdlib-only arithmetic audit of completed v127 raw responses."""
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

BASE=Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_codex_dev_eval_v127')
OUT=BASE/'evidence/INDEPENDENT_METRIC_AUDIT.json'
errors=[];tracked={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 p=Path(p).resolve();tracked.setdefault(str(p),sha(p));return json.loads(p.read_text())
def check(ok,code,**kw):
 if not ok:errors.append({'code':code,**kw})
def avg(xs):return sum(xs,Fraction())/len(xs) if xs else None
def rec(value):return {'fraction':str(value),'value':float(value)} if value is not None else None
def close(a,b):return a is not None and b is not None and math.isclose(float(a),float(b),abs_tol=1e-12,rel_tol=1e-12)
def metric(rows,key):return avg([Fraction(r[key]) for r in rows]) if rows and all(r.get(key) is not None for r in rows) else None
def groups(q,g):
 if q['question_type']=='ordering':return ['all48','main41_excluding_gold6','ordering14']
 return ['all48','multi_select_all34','gold'+str(len(g))]+(['multi_select_gold6_control7'] if len(g)==6 else ['main41_excluding_gold6','multi_select_gold2345_27'])

if OUT.exists():raise RuntimeError('NO_OVERWRITE_EXISTING_AUDIT')
if not (BASE/'PAIRED_SUMMARY.json').is_file():raise RuntimeError('PAIRED_SUMMARY_NOT_READY')
scope=read(BASE/'SCOPE.json');freeze=read(BASE/'PREFLIGHT_FREEZE.json');tools=read(BASE/'TOOLS_FREEZE.json')
for frozen in [freeze,tools]:
 for row in frozen['files']:
  p=Path(row['path']);check(p.is_file() and sha(p)==row['sha256'],'FROZEN_INPUT_CHANGED',path=str(p))
  if p.is_file():tracked.setdefault(str(p),sha(p))
summary=read(BASE/'PAIRED_SUMMARY.json');questions_report=read(BASE/'PER_QUESTION.json');root_report=read(BASE/'PER_ROOT.json')
pool=read(BASE/'EVALUATION_POOL.json');canonical={q['candidate_id']:q for q in pool['questions']}
source_anchor_checks=[]
for q in canonical.values():
 if q['question_type']!='ordering':continue
 source=read(q['source_path']);check(sha(q['source_path'])==q['source_sha256'],'ORDER_SOURCE_SHA_CHANGED',candidate_id=q['candidate_id'])
 positions={t['turn_ref']:i for i,t in enumerate(source['turns'])};turns={t['turn_ref']:t for t in source['turns']};keys={}
 for label,anchor in q['order_anchor_spans'].items():
  t=turns[anchor['turn_ref']];offset=t['text'].find(anchor['quote'])
  check(offset>=0 and t['speaker']==anchor['speaker'],'ORDER_ANCHOR_SOURCE_MISMATCH',candidate_id=q['candidate_id'],label=label)
  check(hashlib.sha256(t['text'].encode()).hexdigest()==q['item_evidence'][label]['frozen_full_turn_sha256'],'ORDER_ANCHOR_FULLTURN_SHA_MISMATCH',candidate_id=q['candidate_id'],label=label)
  keys[label]=(positions[anchor['turn_ref']],offset)
 derived=sorted(q['items'],key=keys.get)
 check(derived==q['gold'],'ORDER_SOURCE_CHRONOLOGY_NOT_GOLD',candidate_id=q['candidate_id'])
 source_anchor_checks.append({'candidate_id':q['candidate_id'],'source_derived_gold':derived,'matches_canonical_gold':derived==q['gold'],'ordered_anchor_positions':[list(keys[label]) for label in derived]})
check(summary['status']=='COMPLETE','REPORT_NOT_COMPLETE')
check(scope['question_count']==48 and scope['calls_per_model']==780 and scope['test_evaluation'] is False and scope['score_based_filtering'] is False,'SCOPE_CHANGED')
allcalls={};questions={};golds={};manifests={};perq={};allmetrics={};rootmetrics={}
counts=Counter()
for model in ['baseline','new']:
 run=BASE/'run'/model
 manifest=manifests[model]=read(run/'manifest.json')
 qs=read(run/'questions.json')['candidates'];qmap=questions[model]={q['candidate_id']:q for q in qs}
 gold=golds[model]=read(run/'gold.json')
 check(len(qs)==len(qmap)==len(gold)==48 and set(qmap)==set(gold),'QUESTION_SET_OR_GOLD_CARDINALITY',model=model)
 for cid,q in qmap.items():
  orig=canonical[cid];field='options' if q['question_type']=='multi_select' else 'items'
  for k in ['candidate_id','root_id','question_type','question',field,'source_path','source_sha256']:
   check(q.get(k)==orig.get(k),'QUESTION_CANONICAL_FIELD_MISMATCH',model=model,candidate_id=cid,field=k)
  expected_gold=['content_'+hashlib.sha256(orig[field][label].strip().encode()).hexdigest()[:16] for label in orig['gold']]
  check(gold[cid]==expected_gold,'GOLD_CONTENT_ID_CANONICAL_MISMATCH',model=model,candidate_id=cid)
 requests=manifest['requests'];planned={r['request_id']:r for r in requests}
 check(len(planned)==len(requests)==780,'PLANNED_CALL_COUNT',model=model)
 check(manifest['solver_model']==scope['models'][model],'MODEL_IDENTITY',model=model)
 complete=read(run/'COMPLETE.json')
 check(complete['status']=='COMPLETE' and complete['actual_call_count']==780,'RUN_INCOMPLETE',model=model)
 observed=defaultdict(list)
 for p in sorted((run/'responses').glob('*.json')):
  response=read(p);observed[response.get('request_id')].append((p,response))
 check(set(observed)==set(planned),'RESPONSE_SET_MISMATCH',model=model)
 rows=[]
 for request in requests:
  rid=request['request_id'];cid=request['candidate_id'];q=qmap[cid];g=gold[cid];variant=request['variant'];mapping=variant['display_to_content'];found=observed.get(rid,[])
  field='options' if q['question_type']=='multi_select' else 'items'
  for display,text in variant[field].items():
   check(mapping[display]=='content_'+hashlib.sha256(text.strip().encode()).hexdigest()[:16],'DISPLAY_TEXT_CONTENT_ID_MISMATCH',model=model,request_id=rid,label=display)
  check(set(variant[field].values())==set(q[field].values()),'DISPLAY_CONTENT_NOT_CANONICAL_SIX',model=model,request_id=rid)
  check(len(found)==1,'RESPONSE_NOT_UNIQUE',model=model,request_id=rid,count=len(found))
  if len(found)!=1:continue
  path,response=found[0]
  expected={'request_id':rid,'candidate_id':cid,'root_id':q['root_id'],'question_type':q['question_type'],
            'variant_id':variant['variant_id'],'mode':'full_context','slot':request['slot'],'seed':request['seed'],
            'solver_model':manifest['solver_model'],'model_path':manifest['model']}
  for k,v in expected.items():check(response.get(k)==v,'RESPONSE_BINDING_MISMATCH',model=model,request_id=rid,field=k)
  check(response.get('evaluation_valid') is True and not response.get('data_errors'),'RESPONSE_NATIVE_INVALID',model=model,request_id=rid)
  md=response.get('metadata',{})
  check(md.get('finish_reason') in ['stop','eos'],'BAD_FINISH_REASON',model=model,request_id=rid)
  for k in ['temperature','top_p','max_tokens','structured_output_api','structured_output_version']:
   check(md.get(k)==manifest['decoder_contract'].get(k),'DECODER_BINDING_MISMATCH',model=model,request_id=rid,field=k)
  raw=response.get('raw_output');m=re.fullmatch(r'\s*<box>\s*(.*?)\s*</box>\s*',raw,re.S) if isinstance(raw,str) else None
  check(m is not None,'RAW_OUTPUT_BOX_INVALID',model=model,request_id=rid)
  if m is None:continue
  labels=[x.strip().upper() for x in m.group(1).strip().split(',')] if m.group(1).strip() else []
  check(len(labels)==len(set(labels)) and all(re.fullmatch('[A-Z][A-Z0-9_]*',x) and x in mapping for x in labels),'RAW_OUTPUT_LABELS_INVALID',model=model,request_id=rid)
  if len(labels)!=len(set(labels)) or any(x not in mapping for x in labels):continue
  pred=[mapping[x] for x in labels]
  check(len(set(mapping.values()))==len(mapping)==6 and len(set(pred))==len(pred),'MAPPING_NOT_BIJECTIVE',model=model,request_id=rid)
  row={'request_id':rid,'candidate_id':cid,'root_id':q['root_id'],'question_type':q['question_type'],
       'gold_count':len(g),'strata':groups(q,g),'variant_id':variant['variant_id'],'slot':request['slot'],'display_prediction':labels,'content_prediction':pred}
  if q['question_type']=='multi_select':
   tp=len(set(pred)&set(g));fp=len(set(pred)-set(g));fn=len(set(g)-set(pred));count=len(pred)
   partial=Fraction(tp,len(g))-(Fraction(fp,6-len(g)) if len(g)<6 else Fraction())
   exact=set(pred)==set(g)
   row.update(exact=int(exact),partial=partial,tp=tp,fp=fp,fn=fn,predicted_count=count,
              predicted_minus_gold=count-len(g),all6=int(count==6),overselection=int(count>len(g)),underselection=int(count<len(g)))
  else:
   check(len(pred)==6 and set(pred)==set(g),'ORDERING_NOT_COMPLETE',model=model,request_id=rid)
   if len(pred)!=6 or set(pred)!=set(g):continue
   sequence=[g.index(x) for x in pred]
   wrong=sum(sequence[i]>sequence[j] for i in range(6) for j in range(i+1,6));correct=15-wrong
   partial=Fraction(correct-wrong,15);exact=pred==g;tp,fp,fn=correct,wrong,0
   row.update(exact=int(exact),partial=partial,correct_pairs=correct,wrong_pairs=wrong,pair_accuracy=Fraction(correct,15))
  saved=response['score']
  check(saved['scored'] is True and saved['exact_match']==exact and Fraction(saved['partial_score_fraction'])==partial
        and saved['tp']==tp and saved['fp']==fp and saved['fn']==fn,'SAVED_SCORE_NOT_INDEPENDENT_RECOMPUTATION',model=model,request_id=rid)
  rows.append(row)
 check(len(rows)==780,'RECOMPUTED_CALL_COUNT',model=model)
 allcalls[model]=rows
 candidates={}
 for cid,q in qmap.items():
  subset=[r for r in rows if r['candidate_id']==cid];n=18 if q['question_type']=='multi_select' else 12
  check(len(subset)==n,'CANDIDATE_RECOMPUTED_CALL_COUNT',model=model,candidate_id=cid)
  keys=['exact','partial','tp','fp','fn','predicted_count','predicted_minus_gold','all6','overselection','underselection','pair_accuracy','correct_pairs','wrong_pairs']
  candidates[cid]={'candidate_id':cid,'root_id':q['root_id'],'strata':groups(q,gold[cid]),'planned_calls':n,
                   'metrics':{k:metric(subset,k) for k in keys}}
 perq[model]=candidates
 stratum_metrics={};root_values={}
 for name in ['all48','main41_excluding_gold6','multi_select_all34','multi_select_gold2345_27','multi_select_gold6_control7','gold2','gold3','gold4','gold5','gold6','ordering14']:
  chosen=[v for v in candidates.values() if name in v['strata']];calls=[r for r in rows if name in r['strata']]
  keys=[k for k in chosen[0]['metrics'] if all(v['metrics'][k] is not None for v in chosen)] if chosen else []
  eq={k:avg([r['metrics'][k] for r in chosen]) for k in keys}
  pooled={k:metric(calls,k) for k in keys}
  stratum_metrics[name]={'candidate_count':len(chosen),'planned_calls':sum(r['planned_calls'] for r in chosen),'equal_candidate':{k:rec(v) for k,v in eq.items()},'pooled_calls':{k:rec(v) for k,v in pooled.items()}}
  reported=summary['strata'][name]['models'][model]
  check(reported['equal_candidate']['candidate_count']==len(chosen) and reported['pooled_calls']['planned_calls']==len(calls),'REPORTED_STRATUM_DENOMINATOR_MISMATCH',model=model,stratum=name)
  check(close(eq.get('exact'),reported['equal_candidate']['exact_rate_equal_candidate']) and close(eq.get('partial'),reported['equal_candidate']['signed_partial_equal_candidate']),'REPORTED_EQUAL_CANDIDATE_SCORE_MISMATCH',model=model,stratum=name)
  check(close(pooled.get('exact'),reported['pooled_calls']['exact_rate_fixed']) and close(pooled.get('partial'),reported['pooled_calls']['signed_partial']['mean']),'REPORTED_POOLED_SCORE_MISMATCH',model=model,stratum=name)
  if name.startswith('multi_select') or name.startswith('gold'):
   keys_map={'tp':'tp','fp':'fp','fn':'fn','predicted_count':'predicted_count','predicted_minus_gold':'predicted_minus_gold_count','all6':'all6_rate','overselection':'overselection_rate','underselection':'underselection_rate'}
   for a,b in keys_map.items():check(close(pooled.get(a),reported['pooled_calls']['multi_select'][b]['mean']),'REPORTED_MULTISELECT_BEHAVIOR_MISMATCH',model=model,stratum=name,metric=a)
  if name=='ordering14':check(close(pooled.get('pair_accuracy'),reported['pooled_calls']['ordering']['pair_accuracy']['mean']),'REPORTED_ORDERING_PAIR_ACCURACY_MISMATCH',model=model)
  covered=[]
  for root in scope['fixed_dev_root_ids']:
   family=[r for r in chosen if r['root_id']==root]
   if family:covered.append({'root_id':root,'candidate_count':len(family),**{k:avg([r['metrics'][k] for r in family]) for k in keys}})
  root_values[name]={'covered_root_denominator':len(covered),'fixed_root_population':15,
                     'equal_root':{k:rec(avg([v[k] for v in covered])) for k in keys}}
  rr=summary['root_macro'][name]
  check(rr['covered_root_denominator']==len(covered) and rr['fixed_dev_root_population']==15,'REPORTED_ROOT_DENOMINATOR_MISMATCH',model=model,stratum=name)
  check(close(avg([v['exact'] for v in covered]),rr['models'][model]['exact_rate_equal_root']) and close(avg([v['partial'] for v in covered]),rr['models'][model]['signed_partial_equal_root']),'REPORTED_ROOT_MACRO_MISMATCH',model=model,stratum=name)
 allmetrics[model]=stratum_metrics;rootmetrics[model]=root_values
 for reported in questions_report:
  cid=reported['candidate_id'];value=candidates[cid]
  check(close(value['metrics']['exact'],reported[model]['exact_rate_fixed']) and close(value['metrics']['partial'],reported[model]['signed_partial']['mean']),'PER_QUESTION_REPORTED_SCORE_MISMATCH',model=model,candidate_id=cid)
check(manifests['baseline']['requests']==manifests['new']['requests'] and golds['baseline']==golds['new'] and questions['baseline']==questions['new'],'PAIR_INPUTS_NOT_IDENTICAL')
ordering_display_checks=[]
for cid,q in questions['new'].items():
 if q['question_type']!='ordering':continue
 calls=[r for r in allcalls['new'] if r['candidate_id']==cid]
 raw_sequences={tuple(r['display_prediction']) for r in calls};content_sequences={tuple(r['content_prediction']) for r in calls}
 check(len(calls)==12 and len(content_sequences)==1 and next(iter(content_sequences))==tuple(golds['new'][cid]),'NEW_ORDERING_RESTORED_CONTENT_NOT_GOLD',candidate_id=cid)
 check(len(raw_sequences)>1,'NEW_ORDERING_OUTPUT_CONSTANT_DESPITE_DISPLAY',candidate_id=cid)
 ordering_display_checks.append({'candidate_id':cid,'planned_permutations':12,'distinct_raw_label_sequences':len(raw_sequences),'distinct_restored_content_sequences':len(content_sequences),'all_twelve_restore_to_gold':all(r['content_prediction']==golds['new'][cid] for r in calls)})
check(len(questions_report)==48 and len(root_report)==15,'DETAIL_REPORT_DENOMINATORS')
check({r['root_id'] for r in root_report}==set(scope['fixed_dev_root_ids']),'DETAIL_ROOT_POPULATION')
deltas={}
for name in allmetrics['baseline']:
 ds={}
 for key in ['exact','partial']:
  a=Fraction(allmetrics['baseline'][name]['equal_candidate'][key]['fraction']);b=Fraction(allmetrics['new'][name]['equal_candidate'][key]['fraction'])
  ds[key]=rec(b-a)
  k='exact_rate' if key=='exact' else 'signed_partial'
  check(close(b-a,summary['strata'][name]['paired_equal_candidate_delta'][k]),'REPORTED_PAIRED_DELTA_MISMATCH',stratum=name,metric=key)
 deltas[name]=ds
changed=[p for p,h in tracked.items() if sha(p)!=h]
check(not changed,'AUDITED_INPUT_CHANGED',paths=changed)
report={'schema_version':'v127_independent_raw_metric_audit_v1','status':'PASS' if not errors else 'FAIL','errors':errors,
        'method':'Independent standard-library parser, content-ID set arithmetic and pairwise inversion counting. No report/native scoring import, inference or model use.',
        'fixed_population':{'questions':48,'multi_select_gold2345':27,'multi_select_gold6':7,'ordering':14,'planned_calls_per_model':780,'dev_roots':15,'covered_roots':7},
        'metrics':allmetrics,'root_equal_weight_metrics':rootmetrics,'paired_equal_candidate_deltas':deltas,'inputs_changed_during_audit':changed,
        'ordering_source_anchor_checks':source_anchor_checks,'ordering_new_model_display_restoration_checks':ordering_display_checks,
        'limitations':['Descriptive paired dev comparison only; permutations/rollouts/root families correlated. No significance or unseen-test claim.','No test question/source file read; no score-based admission, response filtering, retry or inference performed.'],
        'audit_script_binding':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},
        'audited_files':[{'path':p,'sha256':h} for p,h in sorted(tracked.items())]}
OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':report['status'],'errors':errors,'report':str(OUT),'sha256':sha(OUT),
                  'main_metrics':{m:{k:allmetrics[m][k] for k in ['multi_select_gold2345_27','multi_select_gold6_control7','ordering14']} for m in allmetrics}},indent=2))
if errors:raise SystemExit(1)
