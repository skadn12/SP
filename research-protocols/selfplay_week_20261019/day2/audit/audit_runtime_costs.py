#!/usr/bin/env python3
"""Aggregate actual returned IDs and failure accounting without invoking models."""
from pathlib import Path
import json,hashlib,collections
from datetime import datetime,timezone
B=Path(__file__).resolve().parent.parent
RUNS=['solver/direct','solver/evidence','probes/evidence_hint','solver/evidence_v2']
def read(p):return json.loads(p.read_text())
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
def main():
 missing=[rel for rel in RUNS if not(B/rel/'SCORES.json').exists()]
 if missing:print(json.dumps({'status':'PENDING','missing_scored_runs':missing}));return
 rows=[];bindings=[];totals=collections.Counter();starts=[];ends=[]
 for rel in RUNS:
  run=B/rel;m=read(run/'manifest.json');s=read(run/'SCORES.json');freeze=read(run/'RAW_RESPONSE_FREEZE.json');raw=[read(Path(r['path']))for r in freeze['files']];term=read(run/('COMPLETE.json'if(run/'COMPLETE.json').exists()else'FAILED.json'));status=collections.Counter(r['status']for r in raw)
  assert len(raw)==len(m['requests'])==s['denominator_calls']
  actualsubmit=term.get('actual_submitted_requests',term.get('submitted_request_count'));returned=term.get('returned_requests',term.get('returned_request_count'))
  assert actualsubmit==len(raw)-status['NOT_SUBMITTED']and returned==status['RETURNED']
  pt=sum(len(r['prompt_token_ids'])for r in raw if r['status']=='RETURNED');ct=sum(len(r['completion_token_ids'])for r in raw if r['status']=='RETURNED')
  assert pt==s['cost']['known_prompt_tokens']and ct==s['cost']['known_completion_tokens']
  assert status['BACKEND_ERROR']==s['cost']['requests_with_unknown_usage']
  record={'run':rel,'planned_record_denominator':len(raw),'submitted_request_attempts':actualsubmit,'returned_outputs':returned,
   'not_submitted_requests':status['NOT_SUBMITTED'],'backend_error_attempts_unknown_usage':status['BACKEND_ERROR'],
   'known_prompt_tokens_from_returned_IDs':pt,'known_completion_tokens_from_returned_IDs':ct,
   'reported_model_load_wall_seconds':term['model_load_wall_seconds'],'reported_live_wall_seconds':term['wall_seconds'],
   'status':term['status'],'source_feedback_only':True}
  rows.append(record);bindings.extend(bind(run/n)for n in ['manifest.json','RAW_RESPONSE_FREEZE.json','SCORES.json','STARTED.json']);bindings.append(bind(run/('COMPLETE.json'if(run/'COMPLETE.json').exists()else'FAILED.json')))
  for k in ['planned_record_denominator','submitted_request_attempts','returned_outputs','not_submitted_requests','backend_error_attempts_unknown_usage','known_prompt_tokens_from_returned_IDs','known_completion_tokens_from_returned_IDs','reported_model_load_wall_seconds','reported_live_wall_seconds']:totals[k]+=record[k]
  starts.append(datetime.fromisoformat(read(run/'STARTED.json')['started_at_utc']));ends.append(datetime.fromisoformat(freeze['created_at_utc']))
 totals['known_total_tokens_from_returned_IDs']=totals['known_prompt_tokens_from_returned_IDs']+totals['known_completion_tokens_from_returned_IDs']
 out={'status':'PASS_ACCOUNTING_WITH_EXPLICIT_UNKNOWN_USAGE','created_at_utc':datetime.now(timezone.utc).isoformat(),'bindings':bindings+[bind(Path(__file__))],
  'runs':rows,'totals':dict(totals),'observed_solver_start_to_last_raw_freeze_elapsed_seconds':(max(ends)-min(starts)).total_seconds(),
  'new_solver_calls_by_auditor':0,'currency_cost':None,
  'limits':['8attempted evidence_v1 requests were backend grammar rejected and returned0outputs; usage is unknown, not silently zero.',
   '31evidence_v1 planned records were NOT_SUBMITTED and remain in failure denominator; they are not actual sent requests.',
   'Reported live wall includes each model load but excludes some cleanup, raw-freeze/score/preflight CPU time and all interactive Codex preparation/review usage. It is not complete monetary or end-to-end research cost.',
   'Start-to-last-freeze elapsed includes audit/scheduling gaps and is separate from sum of reported live wall; do not add them.',
   'Token costs are known returned outputs only; unknown backend attempts must remain separately reported. Memory54scene pilot did not run.']}
 with(B/'audit/RESULTS_AUDIT_RUNTIME_COSTS.json').open('x')as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in out.items()if k in ['status','totals','observed_solver_start_to_last_raw_freeze_elapsed_seconds']}))
if __name__=='__main__':main()
