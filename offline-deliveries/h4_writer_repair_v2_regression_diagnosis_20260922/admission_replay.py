"""Offline row-isolated admission replay. No model, no production artifact writes."""
from pathlib import Path
import collections, copy, hashlib, json, socket, sys
OUT=Path(__file__).resolve().parent
REPO=Path('/root/autodl-tmp/gxy-selfplay/current_method_github')
RUN=Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_planner_writer_repair_v2_writer_prepare_1')
EX=RUN/'executions/writer_repair_comparison'
sys.path.insert(0,str(REPO))
network_attempts=[]
def forbidden(*a,**k):
    network_attempts.append(True); raise RuntimeError('OFFLINE_NETWORK_FORBIDDEN')
socket.create_connection=forbidden
socket.socket.connect=forbidden
from scripts.current_method.restart_n5_spell.h4_writer_partial_memory_v1.admission import normalize_stage
from scripts.current_method.restart_n5_spell.h4_writer_partial_memory_v1.sources import evidence_catalog
roots=['cc-train-episode-198480','cc-train-episode-122711']
arms=['TEMPORAL_V1','GROUNDED_V2']
def read(p):return json.loads(p.read_text())
def jsonl(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
watched=list((RUN/'inputs').rglob('*.json'))+list(EX.rglob('candidate_generation.json'))+list(EX.rglob('stage_calls/*/*.json'))+list(EX.rglob('raw/prompts/*.txt'))+[REPO/'scripts/current_method/restart_n5_spell/h4_writer_partial_memory_v1/admission.py',REPO/'scripts/current_method/restart_n5_spell/h4_qa_mining_hybrid_generation_v1/candidate_contract.py']
before={str(p):digest(p) for p in watched}
inputs={r:read(RUN/'inputs/TRANSCRIPT_ONLY'/r/'expanded.json') for r in roots}
contracts={r:{x['intent_id']:x for x in read(RUN/'inputs/TRANSCRIPT_ONLY'/r/'contracts.json')} for r in roots}
inv=jsonl(EX/'reports/audited_candidate_inventory.jsonl')
replay=[]; refs=[]; stages=[]; orderings=[]
for arm in arms:
 for root in roots:
  base=EX/'arms'/arm/'roots'/root
  gen=read(base/'candidate_generation.json')
  cats=evidence_catalog(inputs[root]); refset=set(cats)
  for g in gen:
   stage='repair' if 'repair_of' in g else 'primary'; slot=g['intent_id']
   s=read(base/'stage_calls'/slot/(stage+'.json')); raw=s['raw'];v=g['validation']; rows=(raw or {}).get('candidates') or []
   raw_counts=dict(collections.Counter(x.get('question_type','INVALID') for x in rows if isinstance(x,dict)))
   normalized={x['candidate_id']:x for x in v['candidates']}
   checks=[]
   for c in v['row_checks']:
    n=normalized.get(c.get('candidate_id'),{})
    checks.append({**c,'question_type':n.get('question_type')})
   stage_inv=[x for x in inv if x['arm_id']==arm and x['root_id']==root and x['slot_id']==slot and x['stage']==stage]
   stages.append({'arm':arm,'root':root,'slot':slot,'stage':stage,'strict_raw_type_counts':raw_counts,'raw_parse_error':s['parse_error'],'raw_abstentions':(raw or {}).get('abstentions',[]),'response_failure':v['response_failure'],'response_errors':v['errors'],'row_checks':checks,'row_selections':v['minimum_acceptance']['row_selections'],'acceptance':{k:x for k,x in v['minimum_acceptance'].items() if k!='row_selections'},'semantic_inventory':stage_inv})
   orderings.append({'arm':arm,'root':root,'slot':slot,'stage':stage,'ordering_in_strict_raw':raw_counts.get('ordering',0),'ordering_in_normalized':sum(x.get('question_type')=='ordering' for x in v['candidates']),'explicit_ordering_abstention':any(x.get('question_type')=='ordering' for x in (raw or {}).get('abstentions',[]))})
   if raw is None:continue
   for idx,row in enumerate(rows,1):
    if not isinstance(row,dict) or row.get('question_type') not in ('single_choice','multi_select'):continue
    key={'arm':arm,'root':root,'slot':slot,'stage':stage,'raw_row_index':idx,'question_type':row['question_type']}
    outputs={}
    for policy,version in [('partial_v1','ordering_temporal_contract_v1'),('partial_grounded_v2','ordering_grounded_options_v2')]:
     d=OUT/'replay'/arm/root/slot/stage/policy
     d.mkdir(parents=True,exist_ok=True)
     n=normalize_stage({'candidates':[row]},intent_id=slot,source='contract_guided',contract=contracts[root][slot],expanded=inputs[root],root_id=root,stage=stage,request_id=s['request_id'],ordering_policy={'version':version,'node_count':4},ordering_context={'root_dir':d,'run_id':'offline-admission-replay','arm_id':policy,'root_id':root},admission_policy=policy)
     check=n['row_checks'][0];outputs[policy]={'decision':check['decision'],'errors':check['errors']}
    replay.append({**key,**outputs,'same_row_result':outputs['partial_v1']==outputs['partial_grounded_v2']})
    unknown=[r for r in row.get('evidence_refs',[]) if r not in refset]
    if unknown:
     mismatches=[]
     for r in unknown:
      stripped=r.removeprefix(root+':')
      match=next((x for x in inputs[root]['turns'] if x['turn_ref']==stripped),None)
      mismatches.append({'submitted_ref':r,'prefix_stripped_ref':stripped if stripped in refset else None,'matching_source_turn':match})
     refs.append({**key,'unknown_flat_refs':mismatches,'unknown_group_refs':[r for rr in row.get('evidence_groups',{}).values() for r in rr if r not in refset]})
# Compare exact primary and repair raw candidate objects, regardless of top-level abstentions.
repair_pairs=[]
for arm in arms:
 for root in roots:
  for slot in ['intent_1','intent_2','intent_3']:
   base=EX/'arms'/arm/'roots'/root/'stage_calls'/slot
   if not (base/'repair.json').exists():continue
   p=read(base/'primary.json');r=read(base/'repair.json')
   repair_pairs.append({'arm':arm,'root':root,'slot':slot,'strict_primary_parseable':p['raw'] is not None,'candidate_array_identical':p['raw'] is not None and p['raw']['candidates']==r['raw']['candidates'],'raw_object_identical':p['raw']==r['raw']})
# Replaying a deterministic namespace-only correction is diagnostic, not producing accepted candidates.
namespace_counterfactual=[]
for arm in arms:
 for root in roots:
  refset=set(evidence_catalog(inputs[root]))
  for slot in ['intent_1','intent_2','intent_3']:
   for stage in ['primary','repair']:
    p=EX/'arms'/arm/'roots'/root/'stage_calls'/slot/(stage+'.json')
    if not p.exists():continue
    s=read(p)
    for idx,row in enumerate((s['raw'] or {}).get('candidates',[]),1):
     if row.get('question_type') not in ('single_choice','multi_select'):continue
     changed=[];copyrow=copy.deepcopy(row)
     for i,r in enumerate(copyrow.get('evidence_refs',[])):
      stripped=r.removeprefix(root+':')
      if r not in refset and stripped in refset:
       copyrow['evidence_refs'][i]=stripped;changed.append([r,stripped])
     if not changed:continue
     policy='partial_v1' if arm=='TEMPORAL_V1' else 'partial_grounded_v2'; version='ordering_temporal_contract_v1' if arm=='TEMPORAL_V1' else 'ordering_grounded_options_v2'
     n=normalize_stage({'candidates':[copyrow]},intent_id=slot,source='contract_guided',contract=contracts[root][slot],expanded=inputs[root],root_id=root,stage=stage,request_id=s['request_id'],ordering_policy={'version':version,'node_count':4},ordering_context={'root_dir':OUT/'replay/namespace_counterfactual','run_id':'namespace-counterfactual','arm_id':policy,'root_id':root},admission_policy=policy)
     namespace_counterfactual.append({'arm':arm,'root':root,'slot':slot,'stage':stage,'raw_row_index':idx,'changed_flat_refs_only':changed,'row_check_after_namespace_only_change':n['row_checks'][0],'semantic_support_status':'NOT_ASSESSED'})
metrics=[]
for arm in arms:
 for root in roots:
  allrows=[x for x in inv if x['arm_id']==arm and x['root_id']==root and x['question_type'] in ('single_choice','multi_select')]
  final=[x for x in allrows if x['final_retained']]
  passes=[x for x in final if x['content_judgment']=='PASS']
  metrics.append({'arm':arm,'root':root,'raw_scms_occurrences_including_forensic_recovery':len(allrows),'strict_parse_scms_occurrences':sum(sum(x['strict_raw_type_counts'].get(q,0) for q in ('single_choice','multi_select')) for x in stages if x['arm']==arm and x['root']==root),'final_retained_scms':len(final),'final_content':dict(collections.Counter(x['content_judgment'] for x in final)),'final_source_content':dict(collections.Counter(x['source_content_judgment'] for x in final)),'final_citation_support':dict(collections.Counter(x['citation_support_judgment'] for x in final)),'valid_dedup_count':len(set(x['semantic_cluster_key'] for x in passes)),'valid_duplicate_excess':len(passes)-len(set(x['semantic_cluster_key'] for x in passes)),'program_selection_counts_all_scms':dict(collections.Counter(x['program_selection'] for x in allrows))})
findings={'scope':'offline existing-output diagnostic; no API and no production edits','strict_scms_row_replays':len(replay),'old_new_row_results_all_identical':all(x['same_row_result'] for x in replay),'row_result_counts':dict(collections.Counter(x['partial_v1']['decision'] for x in replay)),'replay_omits_top_level_ordering_abstentions_and_other_candidate_types':'This intentionally isolates unchanged SC/MS row semantics; it is not an end-to-end counterfactual generation run. The unparseable original V1 primary is excluded, not silently recovered.','metrics':metrics,'ordering_presence':orderings,'stage_details':stages,'reference_mismatches':refs,'primary_repair_identity':repair_pairs,'namespace_only_counterfactual':namespace_counterfactual,'network_attempts':len(network_attempts),'closed_and_input_selected_files_unchanged':before=={str(p):digest(p) for p in watched},'watched_file_count':len(watched)}
(OUT/'admission_findings.json').write_text(json.dumps(findings,ensure_ascii=False,indent=2)+'\n')
(OUT/'admission_row_replay.json').write_text(json.dumps(replay,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in findings.items() if k in ['strict_scms_row_replays','old_new_row_results_all_identical','row_result_counts','metrics','primary_repair_identity','network_attempts','closed_and_input_selected_files_unchanged','watched_file_count']},ensure_ascii=False,indent=2))
