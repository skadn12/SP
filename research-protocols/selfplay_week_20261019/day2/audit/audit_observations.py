#!/usr/bin/env python3
"""Independently check observation summaries and every model citation row."""
import argparse,collections,hashlib,json,re,statistics
from pathlib import Path
from datetime import datetime,timezone
B=Path(__file__).resolve().parent.parent

def read(p):return json.loads(Path(p).read_text())
def bind(p):p=Path(p);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
def scene(ref):return ref.rsplit(':T',1)[0]if isinstance(ref,str)and':T'in ref else None
def author_citations(v):
 rows=[]
 def walk(x):
  if isinstance(x,dict):
   if isinstance(x.get('turn_ref'),str):rows.append({k:x.get(k)for k in ['turn_ref','speaker','quote']})
   for y in x.values():
    if isinstance(y,(dict,list)):walk(y)
  elif isinstance(x,list):
   for y in x:walk(y)
 walk(v);out=[];seen=set()
 for r in rows:
  key=tuple(r[k]for k in ['turn_ref','speaker','quote'])
  if key not in seen:out.append(r);seen.add(key)
 return out

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--observations',type=Path,default=B/'analysis/results_01/OBSERVATIONS.json');parser.add_argument('--output',type=Path,default=B/'audit/RESULTS_AUDIT_OBSERVATIONS_01.json');args=parser.parse_args();o=read(args.observations);checks=[];issues=[]
 def ck(name,c):
  checks.append({'check':name,'status':'PASS'if c else'FAIL'})
  if not c:issues.append(name)
 contexts={'direct':B/'solver/direct','evidence':B/'solver/evidence_v2'};summary_out={};invalid=[]
 for name,obs in o['runs'].items():
  run=contexts[name];s=read(run/'SCORES.json');m=read(run/'manifest.json');rows={r['request_id']:r for r in m['requests']};pub={q['candidate_id']:q for q in read(run/'PUBLIC_QUESTIONS.json')['questions']};priv={q['candidate_id']:q for q in read(m['private_answer_binding']['path'])['questions']};maps=read(run/'private/DISPLAY_MAPPINGS.json');source={q['root_id']:read(run/'sources'/f'{q["root_id"]}.json')for q in pub.values()};raw={rid:read(run/'responses'/f'{rid}.json')for rid in rows};scored={r['request_id']:r for r in s['requests']}
  for binding in obs['bindings']:
   ck('binding:'+binding['path'],bind(binding['path'])==binding)
  for k,sk in [('question_count','denominator_questions'),('root_count','denominator_roots'),('planned_calls','denominator_calls'),('valid_calls','valid_calls'),('exact_calls','exact_calls'),('question_equal_exact','question_equal_exact'),('root_equal_exact','root_equal_exact')]:ck(name+':summary:'+k,obs['summary'][k]==s[sk])
  ck(name+':request_set',set(r['request_id']for r in obs['requests'])==set(rows))
  for r in obs['requests']:
   rid=r['request_id'];ck(name+':raw_binding:'+rid,r['raw_binding']==bind(run/'responses'/f'{rid}.json'))
   ck(name+':valid_exact:'+rid,r['valid']==scored[rid]['valid']and r['exact']==scored[rid]['exact'])
   ck(name+':coverage:'+rid,r['source_coverage']==1.0)
  actualrows=[]
  if name=='evidence':
   for rid,row in rows.items():
    try:entries=json.loads(raw[rid].get('raw_output')or'{}').get('evidence',[])
    except Exception:entries=[]
    if not isinstance(entries,list):entries=[]
    entries=[e for e in entries if isinstance(e,dict)];field='options'if row['question_type']=='multi_select'else'items';present={e.get('label')for e in entries};entries+= [{'label':label,'status':None,'turn_ref':'','quote':'','_missing_model_entry':True}for label in row['display'][field]if label not in present]
    ck('evidence_entry_and_placeholder_count:'+rid,len([r for r in obs['citations']if r['request_id']==rid])==len(entries))
    if not scored[rid]['valid']:
     invalid.append({'request_id':rid,'finish_reason':raw[rid]['metadata']['finish_reason'],'completion_tokens':len(raw[rid]['completion_token_ids']),'max_output_tokens':row['max_output_tokens'],'parse_error':scored[rid]['parse_error'],'classification':'LENGTH_AND_FORMAT_FAILURE_NOT_MECHANISM_DIAGNOSIS'})
    q=priv[row['candidate_id']];src=source[q['root_id']];turns={t['turn_ref']:t for t in src['turns']};pos={t['turn_ref']:i for i,t in enumerate(src['turns'])};inverse={v:k for k,v in maps['public_to_canonical'][q['candidate_id']].items()};dm=maps['requests'][rid]['display_to_content']
    for ordinal,e in enumerate(entries):
     record=next(r for r in obs['citations']if r['request_id']==rid and r['ordinal']==ordinal);label=e.get('label');canlabel=inverse.get(dm.get(label));tr=e.get('turn_ref');quote=e.get('quote');t=turns.get(tr);empty=tr in('',None)and quote in('',None);exact=bool(t and isinstance(quote,str)and quote and quote in t['text']);allrefs=[t['turn_ref']for t in src['turns']if isinstance(quote,str)and quote and quote in t['text']];offsets=[m.start()for m in re.finditer(re.escape(quote),t['text'])]if exact else[]
     expected={'record_kind':'MISSING_MODEL_ENTRY'if e.get('_missing_model_entry')else'MODEL_ENTRY','display_label':label,'canonical_label':canlabel,'content_id':dm.get(label),'model_status':e.get('status'),'turn_ref':tr,'quote':quote,'turn_exists':t is not None,'quote_exact_in_cited_turn':exact,'quote_offsets':offsets,'quote_occurrence_turn_refs_in_complete_source':allrefs,'quote_unique_across_source':len(allrefs)==1,'source_speaker_at_citation':t['speaker']if t else None,'source_scene_at_citation':scene(tr)if t else None,'source_turn_index':pos.get(tr),'empty_citation':empty,'empty_citation_status_compatible':empty and e.get('status')=='not_supported','citation_mechanically_valid':exact or(empty and e.get('status')=='not_supported')}
     for k,v in expected.items():ck(f'citation:{rid}:{ordinal}:{k}',record[k]==v)
     ev=q.get('option_evidence'if q['question_type']=='multi_select'else'item_evidence',{}).get(canlabel,{});anchor=(q.get('order_anchor_spans')or{}).get(canlabel)or ev.get('order_anchor')or ev.get('primary_citation')or ev;auth=author_citations(ev)+[x for x in author_citations(anchor)if x not in author_citations(ev)];authrefs={x['turn_ref']for x in auth if x.get('turn_ref')};authspeakers={x['speaker']for x in auth if x.get('speaker')};authscenes={scene(x)for x in authrefs}
     loc='NO_CITATION'if empty else'TURN_REF_NOT_IN_SOURCE'if t is None else'SAME_TURN_AS_AUTHOR'if tr in authrefs else'DIFFERENT_TURN_SAME_SCENE_AND_SPEAKER'if scene(tr)in authscenes and t['speaker']in authspeakers else'DIFFERENT_TURN_SAME_SCENE'if scene(tr)in authscenes else'DIFFERENT_SCENE_SAME_SPEAKER'if t['speaker']in authspeakers else'DIFFERENT_SCENE_AND_SPEAKER'
     ck(f'citation:{rid}:{ordinal}:location_relation',record['location_relation_to_author']==loc)
     status={'supports':'supported','contradicts':'contradicted','not_supported':'not_supported','unsupported':'not_supported','ambiguous':'ambiguous'}.get(ev.get('relation'))
     if q['question_type']=='ordering'and status is None and ev.get('supported')is True:status='supported'
     ck(f'citation:{rid}:{ordinal}:author_status',record['author_status']==status)
     agree=e.get('status')==status if status and e.get('status')in {'supported','contradicted','not_supported','ambiguous'}else None
     ck(f'citation:{rid}:{ordinal}:status_comparison',record['model_status_agrees_with_author']==agree)
     ck(f'citation:{rid}:{ordinal}:semantic_not_claimed',record['semantic_judgment']=='NOT_DETERMINED_BY_LOCATION_OR_STRING_CHECK')
     actualrows.append({**expected,'request_id':rid,'location':loc,'agree':agree})
   groups=collections.defaultdict(list)
   for r in actualrows:groups[(r['request_id'],r['display_label'])].append(r)
   summary={'entries':sum(r['record_kind']=='MODEL_ENTRY'for r in actualrows),'expected_labels':len(groups),'labels_missing_entry':sum(all(r['record_kind']=='MISSING_MODEL_ENTRY'for r in g)for g in groups.values()),'labels_with_duplicate_entries':sum(sum(r['record_kind']=='MODEL_ENTRY'for r in g)>1for g in groups.values()),'labels_with_any_valid_mechanical_citation':sum(any(r['citation_mechanically_valid']for r in g)for g in groups.values()),'rows_including_missing_placeholders':len(actualrows),'turn_exists':sum(r['turn_exists']for r in actualrows),'exact_nonempty_quote':sum(r['quote_exact_in_cited_turn']for r in actualrows),'mechanically_valid_including_allowed_empty':sum(r['citation_mechanically_valid']for r in actualrows),'empty_citation':sum(r['empty_citation']for r in actualrows),'model_status_agrees_with_author':sum(r['agree']is True for r in actualrows),'model_status_comparable':sum(r['agree']is not None for r in actualrows),'author_location_relations':dict(collections.Counter(r['location']for r in actualrows))}
   for k,v in summary.items():ck('citation_summary:'+k,obs['citation_summary'][k]==v)
   summary_out[name]=summary
 out={'status':'PASS'if not issues else'ISSUES_FOUND','created_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'Independent request binding, aggregate summary and exact citation provenance checks; no inference/semantic mechanism adjudication.', 'audited_files':[bind(args.observations),bind(Path(__file__))],'check_count':len(checks),'failed_check_count':len(issues),'issues':issues,'checks':checks,'citation_summary_recomputed':summary_out,'invalid_structured_responses':invalid,'new_solver_calls':0,
 'interpretation':'Exact-string/location checks verify provenance only. Same/different author turn or status agreement does not independently establish semantic correctness or causal failure mechanism. Invalid truncated JSON evidence has no salvage/parsing repair and remains in fixed denominator.'}
 with args.output.open('x')as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in out.items()if k in ['status','check_count','failed_check_count','citation_summary_recomputed','invalid_structured_responses']}))
if __name__=='__main__':main()
