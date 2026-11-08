#!/usr/bin/env python3
"""Render scene1 only. Pure JSON: no network, model, or branch generation."""
import argparse, hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ARMS=['recent2','recent2_plus_prefix_memory','full_history']
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical_sha(obj):return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def verify_quotes(memory,source,source_hash):
 assert memory['source_sha256']==source_hash and memory['prefix_last_turn']==len(source['turns'])-1
 for f in memory['facts']:
  for e in f['evidence_refs']:
   assert e['source_sha256']==source_hash and e['source_id']==source['source_id']
   assert e['turn_start']==e['turn_end'] and 0<=e['turn_start']<len(source['turns'])
   t=source['turns'][e['turn_start']]
   assert t['speaker']==e['speaker'] and e['quote'] in t['text']

def render(root,arm,preview=False,memory_path=None,memory_review=None):
 index=load(HERE/'SOURCE_BINDINGS.json');r=next(r for r in index['roots'] if r['root_id']==root)
 assert arm in ARMS
 assert sha(r['source']['path'])==r['source']['sha256']
 source=load(r['source']['path']);rows=source['turns']
 memory=None; memory_status='NOT_USED';binding=None
 if arm=='recent2_plus_prefix_memory':
  memory_path=Path(memory_path) if memory_path else Path(r['memory_candidate']['path'])
  m=load(memory_path);verify_quotes(m,source,r['source']['sha256'])
  if preview:
   memory_status='DRAFT_QUOTE_INDEX_PREVIEW_ONLY'
  else:
   if not memory_review:raise SystemExit('Memory arm blocked: exact-hash semantic PASS required. Use --preview only for input inspection.')
   review=load(memory_review)
   assert review['root_id']==root and review['decision']=='PASS' and review['snapshot_sha256']==sha(memory_path)
   assert review['full_source_sha256']==r['source']['sha256'] and review['all_facts_reviewed'] is True
   assert set(review['accepted_fact_ids'])=={f['fact_id'] for f in m['facts']}
   assert review['coverage_status'] in ['BOUNDED','REVIEWED_COMPLETE_WITHIN_DECLARED_SCOPE']
   memory_status='SEMANTIC_PASS_AT_EXACT_HASH'
  memory=[{'fact_id':f['fact_id'],'claim':f['claim'],'validity':f['validity'],
    'evidence_turns':[[e['turn_start'],e['turn_end']] for e in f['evidence_refs']]} for f in m['facts']]
  binding={'snapshot_sha256':sha(memory_path),'review_sha256':sha(memory_review) if memory_review else None}
 if arm!='full_history':rows=[t for t in rows if t['session_ref'] in r['recent2_session_ids']]
 ctx=[{k:t[k] for k in ['turn_ref','session_ref','speaker','text']} for t in rows]
 objective=load(HERE/'PILOT_PROTOCOL.json')['context']['shared_objectives'][0]
 payload={'schema_version':'day2_fact_memory_initial_prompt_v1','root_id':root,'scene_index':1,
  'high_level_objective':objective,
  'instructions':['Write one coherent 12-20 turn follow-up conversation in English.',
   'Only use speakers evidenced in the visible context. Preserve attributed reports, plans, requests and confirmed outcomes.',
   'Do not invent prior events or assume unseen completion. Add at least one explicit small action or decision.',
   'Return a concise plan first; the realizer will receive this same visible context plus that plan.'],
  'context':ctx,'prefix_fact_memory':memory}
 return {'status':'PREVIEW_NOT_EXECUTABLE' if preview else 'PROMPT_RENDER_ONLY_BACKEND_PREFLIGHT_REQUIRED',
  'arm_private':arm,'memory_status':memory_status,'memory_binding_private':binding,'prompt_payload':payload,
  'prompt_sha256':canonical_sha(payload),'visible_turn_count':len(rows),
  'prompt_utf8_bytes':len(json.dumps(payload,ensure_ascii=False).encode()),'actual_input_tokens':None,
  'source_binding_private':r['source'],
  'privacy_note':'Only prompt_payload is visible to planner/realizer. File isolation is not cryptographically enforced in shared internal tasks.'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--arm',required=True,choices=ARMS)
 p.add_argument('--preview',action='store_true');p.add_argument('--memory');p.add_argument('--memory-review');p.add_argument('--output')
 a=p.parse_args();out=render(a.root,a.arm,a.preview,a.memory,a.memory_review)
 if a.output:
  with Path(a.output).open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
 else:print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
