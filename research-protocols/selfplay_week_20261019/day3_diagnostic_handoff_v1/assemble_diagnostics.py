"""Adapt only model-emitted fields, after immutable raw freeze; score separately."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import json,hashlib
H=Path(__file__).resolve().parent
P=H.parent/'day3_targeted_4root_v1'
R=H/'runtime/evidence_v1'
def read(p):return json.loads(p.read_text())
def bind(p):
 b=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def save(p,x):
 assert not p.exists(),str(p)+' exists; no overwrite'
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def check(x):
 b=Path(x['path']).read_bytes();assert hashlib.sha256(b).hexdigest()==x['sha256'];assert len(b)==x['bytes']
f=read(R/'RAW_RESPONSE_FREEZE.json')
assert f['status']=='COMMITTED'
for x in f['files']+f['execution_ledger_bindings']+[f['manifest_binding'],f['preflight_binding']]:check(x)
parsed=read(R/'PARSED_RESPONSES.json');manifest=read(R/'manifest.json');maps=read(R/'DISPLAY_MAPPINGS.json')
by={x['request_id']:x for x in parsed['questions']}
authors={x['candidate_id']:x for x in map(json.loads,(P/'exports/ALL_AUTHOR_CANDIDATES.jsonl').read_text().splitlines())}
gold={x['candidate_id']:x['gold'] for x in read(P/'exports/PRIVATE_QUESTIONS.json')['questions']}
direct={x['request_id']:x for x in map(json.loads,(P/'exports/SOLVER_RESPONSES.jsonl').read_text().splitlines())}
coverage=[];responses=[];scores=[]
for r in manifest['requests']:
 x=by[r['request_id']];rawpath=R/'responses'/(r['request_id']+'.json');raw=read(rawpath);content_mapping=maps['requests'][r['request_id']]['display_to_content'];inverse={v:k for k,v in maps['public_to_canonical'][r['candidate_id']].items()};mapping={k:inverse[v] for k,v in content_mapping.items()};q=authors[r['candidate_id']]
 cp={'candidate_id':r['candidate_id'],'root_id':r['root_id'],'fact_family_id':q['fact_family_id'],'question_type':r['question_type'],'request_id':r['request_id'],'direct_request_id':r['direct_request_id'],'display_to_canonical':mapping,'source_path':str(R/'sources'/(r['root_id']+'.json')),'source_sha256':r['source_sha256']}
 coverage.append(cp)
 normalized=None
 if x['format_valid']:
  z=x['parsed_response']
  normalized={'item_judgments':[{'label':v['label'],'judgment':v['judgment'],'evidence':[{'turn':e['turn_ref'],'quote':e['quote']} for e in v['citations']],'rationale':v['reason'],**({'evidence_status':'no_support_found'} if not v['citations'] else {})} for v in z['item_evidence']],'final_answer':z['final_answer']}
 responses.append({'request_id':r['request_id'],'candidate_id':r['candidate_id'],'root_id':r['root_id'],'question_type':r['question_type'],'status':raw['status'],'raw_output':raw.get('raw_output'),'raw_response_binding':bind(rawpath),'diagnostic_valid':x['format_valid'],'parsed':normalized,'invalid_reason':'; '.join(x['invalid_reasons']) if x['invalid_reasons'] else None,'invalid_reasons':x['invalid_reasons'],'truncated':x['truncated'],'model':'S0_M_clean','source_sha256':r['source_sha256'],'training_use':False})
 pred=[inverse[v] for v in x['canonical_prediction']] if x['canonical_prediction'] is not None else None;g=gold[r['candidate_id']];is_mc=r['question_type']=='multi_select'
 exact=bool(x['format_valid'] and (set(pred)==set(g) if is_mc else pred==g))
 d=direct[r['direct_request_id']]
 scores.append({'candidate_id':r['candidate_id'],'request_id':r['request_id'],'question_type':r['question_type'],'root_id':r['root_id'],'template_id':q['template_id'],'format_valid':x['format_valid'],'display_prediction':x['display_prediction'],'canonical_prediction':pred,'canonical_gold':g,'diagnostic_exact':exact,'paired_direct_request_id':r['direct_request_id'],'paired_direct_exact':d['exact'],'same_answer':pred==d['canonical_prediction'],'final_answer_extracted_from_invalid_output':False,'citation_count':x['citation_count'],'exact_quote_count':x['exact_quote_count']})
meta={'schema_version':'evidence_handoff_coverage_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),'max_solver_calls':23,'selection_rule':'All23 accepted questions; original direct perm00/s00; no correctness filtering','items':coverage}
assert read(H/'COVERAGE_PLAN.json')['items']==coverage, 'Frozen coverage drift'
resp=H/'DIAGNOSTIC_RESPONSES.jsonl';assert not resp.exists();resp.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in responses))
report={'schema_version':'evidence_diagnostic_score_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),'planned_calls':23,'format_valid':sum(x['format_valid'] for x in scores),'invalid':sum(not x['format_valid'] for x in scores),'by_type':{},'questions':scores,'raw_freeze':bind(R/'RAW_RESPONSE_FREEZE.json'),'parsed_response_binding':bind(R/'PARSED_RESPONSES.json'),'not_comparable_to_342_direct_as_equal_budget':True,'limitations':['One fixed mapping/sample per question; descriptive only, no CoT benefit estimate.','2048 output budget and free JSON vs direct32 constrained labels: protocol and output budget differ.','Observable rationale is not faithful access to hidden reasoning.','Invalid outputs remain in denominator; no repair or retry.']}
for kind in ['multi_select','ordering']:
 a=[x for x in scores if x['question_type']==kind]
 report['by_type'][kind]={'questions':len(a),'diagnostic_exact':sum(x['diagnostic_exact'] for x in a),'paired_direct_exact':sum(x['paired_direct_exact'] for x in a),'direct_wrong_diagnostic_correct':sum(not x['paired_direct_exact'] and x['diagnostic_exact'] for x in a),'direct_correct_diagnostic_wrong':sum(x['paired_direct_exact'] and not x['diagnostic_exact'] for x in a)}
save(H/'DIAGNOSTIC_SCORES.json',report)
print(json.dumps({k:v for k,v in report.items() if k not in ['questions','limitations']},ensure_ascii=False))
