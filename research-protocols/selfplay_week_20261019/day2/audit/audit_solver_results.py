#!/usr/bin/env python3
"""CPU-only independent reconstruction of completed Solver traces and scores.
Does not import the scoring/runtime adapter or load model weights.
"""
import argparse, collections, hashlib, itertools, json, math, re, statistics
from fractions import Fraction
from pathlib import Path
from datetime import datetime, timezone
HERE=Path(__file__).resolve().parent
DAY2=HERE.parent
BOX=re.compile(r'\s*<box>\s*(.*?)\s*</box>\s*\Z',re.S)
CHECKS=[]

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):p=Path(p);return {'path':str(p.resolve()),'sha256':sha(p),'bytes':p.stat().st_size}
def check(name,c):
 CHECKS.append({'check':name,'status':'PASS' if c else 'FAIL'})
 if not c:raise AssertionError(name)
def verify_ref(r):
 p=Path(r['path']);check('sha:'+str(p),p.exists() and sha(p)==r['sha256'])
 if 'bytes'in r:check('bytes:'+str(p),p.stat().st_size==r['bytes'])
def equal(name,a,b):
 check(name,math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12) if isinstance(a,(int,float)) and isinstance(b,(int,float)) else a==b)
def event(t):return datetime.fromisoformat(t.replace('Z','+00:00'))
def cids(q):
 f='options'if q['question_type']=='multi_select'else'items'
 return {k:'content_'+hashlib.sha256(v.strip().encode()).hexdigest()[:16] for k,v in q[f].items()}
def pred_from_text(raw,row):
 if row['branch']=='evidence':
  try:
   import jsonschema
   d=json.loads(raw);jsonschema.validate(d,row['json_schema'])
   if len({e['label']for e in d['evidence']})!=len(row['display'].get('options',row['display'].get('items'))):return None
   labels=d['answer'];raw='<box>'+','.join(sorted(labels)if row['question_type']=='multi_select'else labels)+'</box>'
  except Exception:return None
 field='options'if row['question_type']=='multi_select'else'items';keys=list(row['display'][field]);m=BOX.fullmatch(raw or'')
 if not m:return None
 labels=m.group(1).split(',')
 if any(x not in keys for x in labels)or len(labels)!=len(set(labels)):return None
 if field=='options':
  if not labels or labels!=sorted(labels):return None
 else:
  if set(labels)!=set(keys)or len(labels)!=len(keys):return None
 if raw!='<box>'+','.join(labels)+'</box>':return None
 return labels

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DAY2/'solver/direct');p.add_argument('--output',type=Path,default=HERE/'RESULTS_AUDIT_DIRECT.json');a=p.parse_args();run=a.run
 need=['manifest.json','PREFLIGHT_FREEZE.json','RAW_RESPONSE_FREEZE.json','SCORING_STARTED.json','SCORES.json','STARTED.json']
 missing=[n for n in need if not(run/n).exists()]
 if missing:print(json.dumps({'status':'PENDING','missing':missing}));return
 started=datetime.now(timezone.utc).isoformat();m=read(run/'manifest.json');pre=read(run/'PREFLIGHT_FREEZE.json');freeze=read(run/'RAW_RESPONSE_FREEZE.json');gold_event=read(run/'SCORING_STARTED.json');scores=read(run/'SCORES.json');start=read(run/'STARTED.json')
 for r in pre['files']:verify_ref(r)
 for r in m['code_bindings']:verify_ref(r)
 for r in m['input_bindings']:verify_ref(r)
 for r in [m['private_answer_binding'],m['content_admission_binding'],start['manifest_binding'],start['independent_audit_binding'],freeze['manifest_binding'],gold_event['raw_response_freeze_binding'],gold_event['private_answer_binding'],scores['raw_response_freeze_binding']]:
  if r:verify_ref(r)
 audit=read(start['independent_audit_binding']['path']);check('preflight_independent_PASS',audit['status']=='PASS')
 check('independent_audit_binds_exact_preflight',any(Path(r['path']).resolve()==(run/'PREFLIGHT_FREEZE.json').resolve()and r['sha256']==sha(run/'PREFLIGHT_FREEZE.json')for r in audit['audited_files']))
 check('no_automatic_retry_and_no_truncation',m['automatic_retry']is False and m['source_truncation']is False)
 check('noGold_prepare_live',m['gold_opened_by_prepare']is False and start['gold_opened']is False and freeze['gold_opened_by_live']is False)
 check('namespace_train_only',m['namespace']=='train.feedback'and scores['use_as_final_eval']is False and scores['training_started']is False)
 check('raw_frozen_before_gold',event(freeze['created_at_utc'])<=event(gold_event['gold_first_read_at_utc'])<=event(scores['gold_read_completed_at_utc'])<=event(scores['scored_at_utc']))
 check('gold_event_matches_score',gold_event['gold_first_read_at_utc']==scores['gold_first_read_at_utc'])
 check('event_order_prepare_preflight_start',event(m['prepared_at_utc'])<=event(pre['created_at_utc'])<=event(start['started_at_utc']))
 check('complete_before_freeze',not(run/'COMPLETE.json').exists()or event(read(run/'COMPLETE.json')['completed_at_utc'])<=event(freeze['created_at_utc']))
 # Private Gold JSON opened only here, after complete raw freeze and its timestamp checks.
 private={q['candidate_id']:q for q in read(m['private_answer_binding']['path'])['questions']}
 public=read(run/'PUBLIC_QUESTIONS.json')['questions'];pub={q['candidate_id']:q for q in public};maps=read(run/'private/DISPLAY_MAPPINGS.json')
 rows=m['requests'];ids=[r['request_id']for r in rows]
 check('request_IDs_unique',len(ids)==len(set(ids)))
 check('no_missing_extra_response_files',set(ids)=={p.stem for p in(run/'responses').glob('*.json')})
 check('rawfreeze_matches_every_request',len(freeze['files'])==len(ids)and {Path(r['path']).stem for r in freeze['files']}==set(ids))
 check('score_rows_match_requests',{r['request_id']for r in scores['requests']}==set(ids)and len(scores['requests'])==len(ids))
 check('score_questions_match_public',{q['candidate_id']for q in scores['questions']}==set(pub)and len(scores['questions'])==len(pub))
 is_hint=m.get('purpose')=='controlled_evidence_hint_probe';is_evidence=m['branch']=='evidence'
 if not is_hint:
  check('39_questions20roots',len(pub)==39 and len({q['root_id']for q in public})==20)
  check('19MC20ordering',collections.Counter(q['question_type']for q in public)=={'multi_select':19,'ordering':20})
  check('fixed_582_or39',len(rows)==(39 if is_evidence else 582))
  allowed=set(read(DAY2.parent/'day1/evaluation/ROOT_SELECTION.json')['wave1']['root_ids'])
  check('exact_frozen_wave1_roots',{q['root_id']for q in public}==allowed)
 else:check('hint_max6question90calls',len(pub)<=6 and len(rows)<=90 and len({q['root_id']for q in public})==len(pub))
 # Frozen tokenization is decoded independently from model outputs.
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(m['tokenizer'],local_files_only=True)
 frozen_ids=read(run/'PROMPT_TOKEN_IDS.json');scoreby={r['request_id']:r for r in scores['requests']};rawby={}
 records=[];totals=collections.Counter();decode_modes=collections.Counter();source_cache={}
 for q in public:
  cid=q['candidate_id'];field='options'if q['question_type']=='multi_select'else'items';qq=private[cid]
  check('private_public:'+cid,{k:qq[k]for k in q}==q)
  check('publiccanonical:'+cid,maps['public_to_canonical'][cid]==cids(q))
  source=Path(q['source_path']);check('source_hash:'+cid,sha(source)==q['source_sha256'])
  copy=run/'sources'/f'{q["root_id"]}.json';check('copied_source_exact:'+cid,source.read_bytes()==copy.read_bytes())
  source_cache[q['root_id']]=read(copy)
 for f in freeze['files']:verify_ref(f)
 for row in rows:
  rid=row['request_id'];cid=row['candidate_id'];q=pub[cid];qq=private[cid];sc=scoreby[rid];raw=read(run/'responses'/f'{rid}.json');rawby[rid]=raw
  field='options'if q['question_type']=='multi_select'else'items';canonical=cids(q);texttoc={q[field][k]:v for k,v in canonical.items()};displaymap={k:texttoc[v]for k,v in row['display'][field].items()}
  variant=maps['requests'][rid]
  check('map_bijection:'+rid,len(set(displaymap.values()))==len(q[field])and variant['display_to_content']==displaymap and variant[field]==row['display'][field])
  check('request_identity:'+rid,all(raw[k]==row[k]for k in ['request_id','candidate_id','root_id','question_type','branch']))
  check('score_identity:'+rid,all(sc[k]==row[k]for k in ['request_id','candidate_id','root_id','question_type','variant_id']))
  verify_ref(sc['response_binding']);check('score_rawbinding_path:'+rid,Path(sc['response_binding']['path']).resolve()==(run/'responses'/f'{rid}.json').resolve())
  check('raw_committed_before_freeze_gold:'+rid,event(raw['response_committed_at_utc'])<=event(freeze['created_at_utc'])<=event(sc['gold_first_read_at_utc']))
  check('score_commit_stamp:'+rid,raw['response_committed_at_utc']==sc['response_committed_at_utc'])
  prompt=(run/'prompts'/f'{rid}.txt').read_text();check('request_prompt_hash:'+rid,hashlib.sha256(prompt.encode()).hexdigest()==row['prompt_sha256'])
  check('all_public_turns_present:'+rid,all(f'[{t["turn_ref"]}] {t["speaker"]}: {t["text"]}'in prompt for t in source_cache[q['root_id']]['turns']))
  check('frozen_prompt_tokenization:'+rid,tok.encode(prompt,add_special_tokens=False)==frozen_ids[rid]and len(frozen_ids[rid])==row['prompt_tokens'])
  check('full_context_budget:'+rid,row['prompt_tokens']+row['max_output_tokens']<=m['max_model_len'])
  if raw['status']=='RETURNED':
   check('output_prompt_exact_token_ids:'+rid,raw['prompt_token_ids']==frozen_ids[rid])
   check('output_prompt_ID_decode:'+rid,tok.decode(raw['prompt_token_ids'],skip_special_tokens=False,clean_up_tokenization_spaces=False)==prompt)
   decoded=tok.decode(raw['completion_token_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)
   check('completion_token_ID_decode:'+rid,decoded==raw['raw_output']);decode_modes['skip_special_tokens=True']+=1
   md=raw['metadata'];check('raw_decoder_contract:'+rid,md['temperature']==.8 and md['top_p']==.9 and md['max_tokens']==row['max_output_tokens']and md['enable_thinking']is False)
   check('raw_model_and_noGold:'+rid,raw['model_path']==m['model']and raw['gold_read_by_solver']is False)
   check('raw_seed_variant_slot:'+rid,all(raw[k]==row[k]for k in ['seed','variant_id','slot','prompt_sha256']))
   check('usage_counts_fromIDs:'+rid,md['prompt_token_count']==len(raw['prompt_token_ids'])and md['completion_token_count']==len(raw['completion_token_ids'])and md['completion_token_count']<=row['max_output_tokens'])
   totals['prompt_tokens']+=md['prompt_token_count'];totals['completion_tokens']+=md['completion_token_count']
  labels=pred_from_text(raw.get('raw_output'),row)if raw['status']=='RETURNED'else None
  valid=raw['status']=='RETURNED'and raw['metadata']['finish_reason']in ('stop','eos')and labels is not None
  prediction=[displaymap[x]for x in labels]if labels is not None else None;gold=[canonical[x]for x in qq['gold']]
  exact=bool(valid and(set(prediction)==set(gold)if field=='options'else prediction==gold))
  equal('valid:'+rid,valid,sc['valid']);equal('exact:'+rid,exact,sc['exact'])
  metrics={};partial=None
  if valid:
   check('native_display_prediction:'+rid,sc['score']['display_prediction']==labels)
   check('native_content_prediction:'+rid,sc['score']['content_prediction']==(sorted(prediction)if field=='options'else prediction))
   check('native_gold_mapping:'+rid,sc['score']['content_gold']==gold and sc['score']['exact_match']==exact)
   if field=='options':
    ps,gs=set(prediction),set(gold);tp=len(ps&gs);fp=len(ps-gs);fn=len(gs-ps);partial=Fraction(tp,len(gs))-Fraction(fp,len(q[field])-len(gs))if len(gs)<len(q[field])else Fraction(tp,len(gs))
    metrics={'fp':fp,'fn':fn,'tp':tp,'jaccard':len(ps&gs)/len(ps|gs),'overselected':len(ps)>len(gs),'selected_count':len(ps),'gold_count':len(gs)}
    totals.update({'MC_TP':tp,'MC_FP':fp,'MC_FN':fn,'MC_overselected_calls':metrics['overselected']})
   else:
    rank={c:i for i,c in enumerate(prediction)};pairs=[(l,r)for i,l in enumerate(gold)for r in gold[i+1:]];right=sum(rank[l]<rank[r]for l,r in pairs);wrong=len(pairs)-right;tp,fp,fn=right,wrong,0;partial=Fraction(right-wrong,len(pairs));metrics['pair_accuracy']=right/len(pairs)
    totals.update({'ordering_correct_pairs':right,'ordering_total_pairs':len(pairs)})
    anchors=qq.get('order_anchor_spans')or qq.get('item_evidence')or{};scene={}
    for label,e in anchors.items():
     tr=e.get('turn_ref')or(e.get('order_anchor')or{}).get('turn_ref')or(e.get('primary_citation')or{}).get('turn_ref')
     if tr:scene[canonical[label]]=tr.rsplit(':T',1)[0]
    for kind,same in [('within_scene',True),('across_scene',False)]:
     selected=[(l,r)for l,r in pairs if l in scene and r in scene and(scene[l]==scene[r])==same];correct=sum(rank[l]<rank[r]for l,r in selected)
     metrics[kind]={'correct_pairs':correct,'total_pairs':len(selected),'accuracy':correct/len(selected)if selected else None};totals.update({kind+'_correct_pairs':correct,kind+'_total_pairs':len(selected)})
   for key,expected in [('tp',tp),('fp',fp),('fn',fn)]:equal('native_'+key+':'+rid,expected,sc['score'][key])
   equal('native_partial:'+rid,str(partial),sc['score']['partial_score'])
  equal('all_secondary_metrics:'+rid,metrics,sc['metrics'])
  records.append({'request_id':rid,'candidate_id':cid,'root_id':q['root_id'],'question_type':q['question_type'],'valid':valid,'exact':exact,'canonical_prediction':prediction,'canonical_gold':gold,'metrics':metrics})
 qs=[]
 for q in public:
  rs=[r for r in records if r['candidate_id']==q['candidate_id']];expected=1 if is_evidence else (18 if q['question_type']=='multi_select' else 12)
  check('all_scheduled_samples:'+q['candidate_id'],len(rs)==expected)
  rate=sum(r['exact']for r in rs)/len(rs);stable='incorrect'if rate==0else'stably_correct'if rate==1else'occasionally_correct'
  predictions=[json.dumps(sorted(r['canonical_prediction'])if q['question_type']=='multi_select'else r['canonical_prediction'])for r in rs if r['valid']]
  consistency=max(collections.Counter(predictions).values())/len(rs)if predictions else 0
  expectedq={'denominator_calls':len(rs),'valid_calls':sum(r['valid']for r in rs),'exact_calls':sum(r['exact']for r in rs),'exact_rate':rate,'stratum':stable,'display_permutation_consistency':consistency,'source_coverage':1.0}
  scoredq=next(z for z in scores['questions']if z['candidate_id']==q['candidate_id'])
  for k,v in expectedq.items():equal('question_'+k+':'+q['candidate_id'],v,scoredq[k])
  qs.append({'candidate_id':q['candidate_id'],'root_id':q['root_id'],'question_type':q['question_type'],**expectedq})
 rr=collections.defaultdict(list)
 for q in qs:rr[q['root_id']].append(q['exact_rate'])
 aggregate={'denominator_calls':len(rows),'denominator_questions':len(qs),'denominator_roots':len(rr),'valid_calls':sum(r['valid']for r in records),'exact_calls':sum(r['exact']for r in records),'question_equal_exact':statistics.mean(q['exact_rate']for q in qs),'root_equal_exact':statistics.mean(statistics.mean(v)for v in rr.values())}
 for k,v in aggregate.items():equal('aggregate_'+k,v,scores[k])
 bytype={}
 for typ in ['multi_select','ordering']:
  sub=[q for q in qs if q['question_type']==typ];rt=collections.defaultdict(list)
  if not sub:continue
  for q in sub:rt[q['root_id']].append(q['exact_rate'])
  bytype[typ]={'questions':len(sub),'calls':sum(q['denominator_calls']for q in sub),'question_equal_exact':statistics.mean(q['exact_rate']for q in sub),'root_equal_exact':statistics.mean(statistics.mean(v)for v in rt.values())}
  equal('aggregate_bytype:'+typ,bytype[typ],scores['by_type'][typ])
 equal('known_prompt_tokens',totals['prompt_tokens'],scores['cost']['known_prompt_tokens']);equal('known_completion_tokens',totals['completion_tokens'],scores['cost']['known_completion_tokens'])
 rawstatuses=collections.Counter(r['status']for r in rawby.values());equal('unknown_usage_calls',rawstatuses['BACKEND_ERROR'],scores['cost']['requests_with_unknown_usage']);equal('not_submitted_calls',rawstatuses['NOT_SUBMITTED'],scores['cost']['not_submitted_requests'])
 report={'status':'PASS','auditor':'/root/day2_fact_memory_prep','created_at_utc':datetime.now(timezone.utc).isoformat(),'audit_started_at_utc':started,
  'scope':'Independent CPU recomputation; no Solver/model-weight inference; code from frozen runtime read but scoring functions not imported.',
  'audited_files':[bind(run/n)for n in need]+[bind(HERE/'audit_solver_results.py')],
  'check_count':len(CHECKS),'checks':CHECKS,'aggregate':aggregate,'by_type':bytype,'secondary_totals':dict(totals),'questions':qs,
  'raw_status_counts':dict(rawstatuses),'completion_decode_modes':dict(decode_modes),
  'event_chain':{'last_response_committed_at_utc':max(r['response_committed_at_utc']for r in rawby.values()),'raw_freeze_committed_at_utc':freeze['created_at_utc'],'scoring_gold_first_read_at_utc':gold_event['gold_first_read_at_utc']},
  'limitations':['Current-process Gold chronology supported by frozen code/events and hashes; no OS-enforced proof that other historical/shared tasks never saw Gold.',
   'Model-weight file identities are bound by preflight and runtime verification; this CPU result audit does not rehash model weight binaries.',
   'Content admission binds historical accepted exact QA/source versions; this audit does not perform new semantic adjudication.',
   'Train feedback outcomes, not final generalization benchmark; no new learning or memory benefit shown by scores alone.']}
 with a.output.open('x')as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in report.items()if k in ['status','check_count','aggregate','by_type','secondary_totals','event_chain']}))
if __name__=='__main__':main()
