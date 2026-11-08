#!/usr/bin/env python3
"""Validate preparation identities, immutable evidence and fair input structure."""
import argparse, ast, hashlib, json
from pathlib import Path
from collections import Counter
import jsonschema
from render_initial import render,verify_quotes,HERE,canonical_sha
checks=[]
def check(name,condition):
 checks.append({'check':name,'status':'PASS' if condition else 'FAIL'})
 if not condition:raise AssertionError(name)
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='VALIDATION.json');a=p.parse_args()
 protocol=load(HERE/'PILOT_PROTOCOL.json');index=load(HERE/'SOURCE_BINDINGS.json');plan=load(HERE/'PUBLIC_INPUT_PLAN.json')
 roots=load(HERE.parent.parent/'day1/evaluation/ROOT_SELECTION.json')['memory_pilot']['root_ids']
 check('fixed_six_roots',roots==[r['root_id'] for r in index['roots']])
 check('54_reserved_scenes',len(plan['slots'])==54 and plan['actual_scenes']==0)
 check('each_branch_three',set(Counter((s['root_id'],s['arm']) for s in plan['slots']).values())=={3})
 check('scene_review_one_revision_max_one',all(s['review_calls']==1 and s['max_revision_calls']==1 for s in plan['slots']))
 check('generic_shared_goals_equal',all(len({s['high_level_objective'] for s in plan['slots'] if s['scene_index']==i})==1 for i in [1,2,3]))
 check('no_model_execution',protocol['generation']['actual_generation_calls']==0)
 check('initial_memory_review_budget',protocol['generation']['initial_memory_review_calls_planned']==6 and protocol['generation']['initial_memory_review_max_output_tokens']==3072)
 check('twelve_updates_twelve_review_budget',protocol['generation']['maximum_memory_update_calls']==12 and protocol['generation']['maximum_memory_update_review_calls']==12 and protocol['generation']['maintenance_review_max_output_tokens']==3072)
 check('maintenance_reviews_separate_from_scene',protocol['generation']['maximum_memory_maintenance_review_calls_including_initial']==18 and protocol['generation']['review_calls_per_scene']==1)
 check('day1_unchanged_operational_clarification',all(c['day1_modified'] is False and c['frozen_before_any_pilot_generation'] is True for c in protocol['operational_clarifications']))
 schema=load(protocol['fact_memory_schema']['path']);totalfacts=0;totalturns=0
 for r in index['roots']:
  root=r['root_id'];src=load(r['source']['path']);mem=load(r['memory_candidate']['path']);gate=load(r['prefix_content_gate']['path'])
  for key in ['source','historical_source','prefix_content_gate','historical_review','memory_candidate']:
   check(f'{root}:{key}:hash',sha(r[key]['path'])==r[key]['sha256'])
  check(f'{root}:byte_identical_source',Path(r['source']['path']).read_bytes()==Path(r['historical_source']['path']).read_bytes())
  check(f'{root}:historical_content_gate',gate['decision']=='READY' and gate['source_sha256']==r['source']['sha256'] and gate['full_source_read'] is True and gate['baseline_constraint_review']['status']=='PASS')
  check(f'{root}:historical_all_turns_read',set(t['turn_ref'] for t in src['turns'])<=set(gate['turn_refs_read']))
  jsonschema.Draft202012Validator(schema,format_checker=jsonschema.FormatChecker()).validate(mem)
  check(f'{root}:fact_memory_schema',True)
  verify_quotes(mem,src,r['source']['sha256']);check(f'{root}:all_quotes_source_bound',True)
  for f in mem['facts']:
   e=f['evidence_refs'][0];t=src['turns'][e['turn_start']]
   check(f'{f["fact_id"]}:exact_whole_utterance',e['quote']==t['text'])
   check(f'{f["fact_id"]}:older_history_only',t['session_ref'] not in r['recent2_session_ids'])
   check(f'{f["fact_id"]}:uncertain_not_inferred_current_state',f['validity']=='uncertain' and f['supersedes_fact_ids']==[])
  recent=render(root,'recent2',True);memory=render(root,'recent2_plus_prefix_memory',True);full=render(root,'full_history',True)
  rp=recent['prompt_payload'];mp=memory['prompt_payload'];fp=full['prompt_payload']
  check(f'{root}:recent_exactly_two_complete_sessions',list(dict.fromkeys(t['session_ref'] for t in rp['context']))==r['recent2_session_ids'] and len(rp['context'])==r['recent2_turn_count'])
  check(f'{root}:recent_and_memory_same_recent_context',rp['context']==mp['context'])
  check(f'{root}:full_complete',len(fp['context'])==len(src['turns']))
  check(f'{root}:shared_objective_no_fact_injection',rp['high_level_objective']==mp['high_level_objective']==fp['high_level_objective'])
  check(f'{root}:no_memory_for_other_arms',rp['prefix_fact_memory'] is None and fp['prefix_fact_memory'] is None)
  check(f'{root}:memory_candidate_not_admitted',memory['memory_status']=='DRAFT_QUOTE_INDEX_PREVIEW_ONLY' and r['memory_semantic_status']=='PENDING')
  for arm,result in [('recent2',recent),('recent2_plus_prefix_memory',memory),('full_history',full)]:
   allowed={'schema_version','root_id','scene_index','high_level_objective','instructions','context','prefix_fact_memory'}
   check(f'{root}:{arm}:prompt_whitelist',set(result['prompt_payload'])==allowed)
   check(f'{root}:{arm}:context_whitelist',all(set(t)=={'turn_ref','session_ref','speaker','text'} for t in result['prompt_payload']['context']))
   check(f'{root}:{arm}:prompt_hash',canonical_sha(result['prompt_payload'])==result['prompt_sha256'])
   preview_path=HERE/'initial_previews'/f'{root}__{arm}.json'
   check(f'{root}:{arm}:stored_preview_matches_renderer',load(preview_path)==result)
  try: render(root,'recent2_plus_prefix_memory',False)
  except SystemExit: check(f'{root}:memory_execution_blocked_without_review',True)
  else:check(f'{root}:memory_execution_blocked_without_review',False)
  totalfacts+=len(mem['facts']);totalturns+=len(src['turns'])
 for script in HERE.glob('*.py'):
  ast.parse(script.read_text());check('syntax:'+script.name,True)
 out={'status':'PASS','scope':'Mechanical preparation only; not semantic memory admission or expansion outcomes',
  'checks':checks,'check_count':len(checks),'source_turns':totalturns,'candidate_facts':totalfacts,'memory_semantic_pass':0,
  'actual_generation_scenes':0,'actual_model_calls':0}
 target=Path(a.output);target=target if target.is_absolute() else HERE/target
 with target.open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in out.items() if k!='checks'}))
if __name__=='__main__':main()
