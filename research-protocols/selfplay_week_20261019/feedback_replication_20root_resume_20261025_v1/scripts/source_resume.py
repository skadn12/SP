#!/usr/bin/env python3
"""Recover exactly the sixteen immutable initial source drafts, review once, revise at most once."""
from __future__ import annotations
import argparse, concurrent.futures, fcntl, json, pathlib, statistics, subprocess, traceback
from codex_client import BASE,atomic,bind,call,load,now,status,strict_response
from event_policy import classify_events
PRE=pathlib.Path(load(BASE/'RESUME_MANIFEST.json')['immutable_predecessor'])
OUT=BASE/'sources'

def compact_turns(source):return [{k:t.get(k) for k in ('turn_ref','speaker','text','origin')} for t in source['turns']]
def mechanical(candidate,inp,parent):
 errors=[]
 if not isinstance(candidate,dict):return ['candidate_not_object']
 if candidate.get('root_id')!=inp['root_id']:errors.append('root_id_mismatch')
 if candidate.get('source_status')!=inp['source_status']:errors.append('source_status_mismatch')
 scenes=candidate.get('scenes',[])
 if [s.get('scene_ref') for s in scenes]!=inp['scene_sequence']:errors.append('scene_sequence_mismatch')
 speakers={t['speaker'] for t in parent['turns']};texts=[];refs=[]
 for s in scenes:
  turns=s.get('turns',[])
  if len(turns)!=8:errors.append('scene_turn_count:'+str(s.get('scene_ref')))
  for i,t in enumerate(turns,1):
   if t.get('turn_ref')!=f"{s.get('scene_ref')}:T{i:03d}":errors.append('turn_ref_mismatch')
   if t.get('speaker') not in speakers:errors.append('new_speaker')
   if t.get('origin')!='codex_synthetic_extension':errors.append('origin_mismatch')
   if not isinstance(t.get('text'),str) or not t['text'].strip():errors.append('empty_text')
   else:texts.append(t['text'])
   refs.append(t.get('turn_ref'))
 if len(texts)!=48:errors.append('not_48_new_turns')
 if len(refs)!=len(set(refs)):errors.append('duplicate_ref')
 if texts and statistics.mean(len(t.split()) for t in texts)>18:errors.append('mean_words_above_18')
 return sorted(set(errors))

def review_ok(receipt,candidate,parent):
 errors=[];r=receipt.get('response')
 if not receipt['accepted']:return receipt.get('failure_reasons',['call_rejected'])
 if not isinstance(r,dict):return ['review_not_object']
 if r.get('root_id')!=candidate['root_id']:errors.append('review_root_mismatch')
 if r.get('decision') not in ['PASS','REVISE','FAIL']:errors.append('invalid_review_decision')
 if r.get('decision')!='PASS':errors.append('semantic_'+str(r.get('decision')))
 turns={t['turn_ref']:t for t in parent['turns']}
 turns.update({t['turn_ref']:t for s in candidate['scenes'] for t in s['turns']})
 quotes=r.get('source_quotes',[])
 if not isinstance(quotes,list) or not quotes:errors.append('missing_source_quotes')
 else:
  for q in quotes:
   t=turns.get(q.get('turn_ref'))
   if not t or not q.get('quote') or q['quote'] not in t['text']:errors.append('invalid_review_quote')
 if not isinstance(r.get('failure_reasons'),list):errors.append('missing_failure_reasons')
 if r.get('decision')=='PASS' and r.get('failure_reasons'):errors.append('pass_has_failures')
 return sorted(set(errors))

def review_prompt(inp,parent,native,candidate,errors):
 return '''You are an independent source reviewer in a fresh session. Use ONLY the supplied data. Do not use any tools, commands, file access, agents, or external facts. Return one compact JSON object, no markdown.
Review this ONE root's fixed R03 source candidate for: continuity with the FULL immutable parent; 6 scenes A1,A2,A3,B1,B2,B3 each 8 turns; same two established speakers; natural sequential conversations; no retrospective same-week assumption; no invented identity; native B scenes faithfully report the EXACT reserved event's utterances, including booking/proposal/refusal states, role/direction, clock and train bindings. Native frames are not spoken evidence. Preserve ambiguity and warn on native inconsistencies, do not resolve them by invention. New A planning is synthetic and permitted. Check all neutral material objectives and verify necessary explicit quantities, times, causal motives, unique discourse contributions and a source-specific ordered procedure exist for downstream tasks; do not demand a particular question or gold. Average English words <=18 per new turn. An old-parent inconsistency alone is a warning, not a new-draft failure unless the extension silently resolves or amplifies it. Mechanical errors below are binding.
Decision PASS only when sound; otherwise REVISE or FAIL with precise targeted reasons. Quote 3-6 exact source substrings with turn_ref covering A and B. Keep total response under 1200 words.
Output: {"root_id":...,"decision":"PASS|REVISE|FAIL","source_quotes":[{"turn_ref":...,"quote":...}],"failure_reasons":[...],"warnings":[...]}. No rewritten source in review.
DATA\n'''+json.dumps({'root_id':inp['root_id'],'constraints':inp['continuity_constraints'],'material_tasks':inp['material_tasks'],'full_parent_turns':compact_turns(parent),'reserved_native_utterances':native,'candidate':candidate,'mechanical_errors':errors},ensure_ascii=False)

def process(path):
 inp=load(path);root=inp['root_id'];d=OUT/root;d.mkdir(parents=True,exist_ok=True)
 if (d/'OUTCOME.json').exists():return load(d/'OUTCOME.json')
 parent=load(inp['parent_source_binding']['path']);native=[]
 for n in inp['native_events']:
  actual=bind(n['binding']['path'])
  if actual['sha256']!=n['binding']['sha256']:raise ValueError('native binding mismatch')
  x=load(n['binding']['path']);native.append({'source_id':n['source_id'],'scene_ref':n['scene_ref'],'turns':[{k:t.get(k) for k in ['turn_id','speaker','utterance']} for t in x['turns']]})
 if bind(inp['parent_source_binding']['path'])['sha256']!=inp['parent_source_binding']['sha256']:raise ValueError('parent binding mismatch')
 old=PRE/'cli_calls'/(root.replace('-','_')+'_source_draft')/'attempt_01'
 terminal=load(old/'TERMINAL.json');raw=(old/'LAST_MESSAGE.txt').read_text()
 cls=classify_events((old/'EVENTS.jsonl').read_text(),timed_out=terminal.get('timed_out',False),backend_error=terminal.get('returncode')!=0)
 candidate,reasons=strict_response(raw,cls,terminal.get('returncode',1),terminal.get('timed_out',False))
 atomic(d/'INITIAL_EVENT_CLASSIFICATION.json',cls)
 atomic(d/'INITIAL_EVIDENCE.json',{'root_id':root,'fixed_draft':bind(old/'LAST_MESSAGE.txt'),'events':bind(old/'EVENTS.jsonl'),'request':bind(old/'REQUEST.txt'),'terminal':bind(old/'TERMINAL.json'),'backend':terminal.get('backend'),'native_bindings':[n['binding'] for n in inp['native_events']],'parent_binding':inp['parent_source_binding'],'transport_errors':reasons})
 outcome={'root_id':root,'status':'FAIL','fixed_initial_draft':bind(old/'LAST_MESSAGE.txt'),'source_ids':[n['source_id'] for n in inp['native_events']],'attempts':[],'failure_reasons':reasons}
 if reasons:
  atomic(d/'OUTCOME.json',outcome);return outcome
 for attempt in [0,1]:
  mech=mechanical(candidate,inp,parent);atomic(d/f'CANDIDATE_{attempt}.json',candidate)
  review=call(root+f'_source_review_{attempt}',review_prompt(inp,parent,native,candidate,mech))
  errors=mech+review_ok(review,candidate,parent)
  outcome['attempts'].append({'attempt':attempt,'candidate':bind(d/f'CANDIDATE_{attempt}.json'),'mechanical_errors':mech,'review_receipt':str(pathlib.Path(review['directory'])/'RECEIPT.json'),'review_errors':errors})
  atomic(d/f'REVIEW_{attempt}.json',review)
  if not errors:
   full=dict(parent);full['turns']=parent['turns']+[t for s in candidate['scenes'] for t in s['turns']]
   full.update(source_status=inp['source_status'],warnings=candidate.get('warnings',[])+review['response'].get('warnings',[]),resume_provenance={'parent':inp['parent_source_binding'],'candidate':bind(d/f'CANDIDATE_{attempt}.json'),'native_source_ids':outcome['source_ids']})
   assert full['turns'][:len(parent['turns'])]==parent['turns']
   atomic(d/'FULL_SOURCE.json',full)
   outcome.update(status='PASS',source=bind(d/'FULL_SOURCE.json'),failure_reasons=[],revisions_used=attempt);break
  outcome['failure_reasons']=errors
  if attempt==0:
   prompt='''You are the targeted source revision author for ONE root. No tools, commands, agents, file access or external facts. This is the only allowed revision. Keep all immutable parent turns and reserved native events unchanged; return ONLY a complete revised candidate JSON with the same root_id, source_status, six scenes in exact order and eight turns per scene, turn_ref/speaker/text/origin fields, native_evidence, warnings. Repair the specific audit failures; preserve sound content. Never invent native facts or silently resolve contradictions. Synthetic A planning is permitted. Every new turn origin=codex_synthetic_extension, mean words <=18. Source data and full original candidate follow.\n'''+json.dumps({'root_id':root,'input_constraints':inp['continuity_constraints'],'scene_sequence':inp['scene_sequence'],'full_parent_turns':compact_turns(parent),'reserved_native_utterances':native,'material_tasks':inp['material_tasks'],'candidate':candidate,'audit':review.get('response'),'mechanical_errors':mech},ensure_ascii=False)
   revision=call(root+'_source_revision_1',prompt);atomic(d/'REVISION_1.json',revision)
   if not revision['accepted'] or not isinstance(revision.get('response'),dict):outcome['failure_reasons']+=['revision_call_failed'];break
   candidate=revision['response']
 atomic(d/'OUTCOME.json',outcome);return outcome

def run():
 lock=(BASE/'.source_resume.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (BASE/'SOURCE_FREEZE.json').exists():return load(BASE/'SOURCE_FREEZE.json')
 check=subprocess.run(['python3',str(BASE/'scripts/self_check.py')],cwd=BASE,capture_output=True,text=True)
 atomic(BASE/'PREFLIGHT.json',{'at':now(),'self_check_exit':check.returncode,'stdout':check.stdout,'stderr':check.stderr,'manifest':bind(BASE/'RESUME_MANIFEST.json'),'plan':bind(BASE/'RESUME_PLAN.json')})
 if check.returncode:raise RuntimeError('preflight failed')
 paths=sorted((PRE/'inputs').glob('cc-train-episode-*_SOURCE_AUTHOR_INPUT.json'))
 assert len(paths)==16
 status('preflight',preflight_passed=True,planned_source_roots=16,source_roots_finished=0,source_roots_passed=0,source_roots_failed=0)
 outcomes=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  for future in concurrent.futures.as_completed({pool.submit(process,p):p for p in paths}):
   try:r=future.result()
   except Exception as e:
    p=next(p for f,p in [] ) if False else None
    raise
   outcomes.append(r)
   atomic(BASE/'SOURCE_PROGRESS.json',{'outcomes':outcomes,'at':now()})
   status('source',source_roots_finished=len(outcomes),source_roots_passed=sum(x['status']=='PASS' for x in outcomes),source_roots_failed=sum(x['status']!='PASS' for x in outcomes))
 freeze={'schema_version':'fgr20_resume_source_freeze_v1','at':now(),'all_initial_drafts_reused':True,'new_initial_drafts_generated':0,'root_count':16,'outcomes':sorted(outcomes,key=lambda x:x['root_id']),'pass_count':sum(x['status']=='PASS' for x in outcomes)}
 atomic(BASE/'SOURCE_FREEZE.json',freeze);status('source_freeze',source_freeze=bind(BASE/'SOURCE_FREEZE.json'))
 return freeze
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--run',action='store_true');args=a.parse_args()
 if args.run:run()
