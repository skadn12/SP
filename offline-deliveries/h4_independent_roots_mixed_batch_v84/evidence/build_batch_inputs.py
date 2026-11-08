"""Construct audited parents and exact new-root configuration before preparation."""
from pathlib import Path
import sys,copy,json
HOME=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HOME))
from common import read,write,bind,now,need,sha
import source_admission
OLD=HOME.parent/'h4_name_policy_mixed_batch_v83'
summary=source_admission.verify();roots=source_admission.ROOTS
inventory={r['root_id']:r for r in read(HOME/'evidence/SOURCE_INVENTORY.json')['root_inventory']}
historical_paths={}
for root in roots:
 inputs=[r['path'] for r in inventory[root]['historical_generation']['historical_public_priors']]
 if root.endswith('-135466'):
  inputs.append(str(HOME.parents[1]/'restart-n5-runs/h4_unexpanded122_qa_generation_only_v1/shards/shard-07/roots/cc-train-episode-135466/candidate_pool.json'))
 rows=[];seen=set()
 for path in dict.fromkeys(inputs):
  obj=read(path);raw=obj if isinstance(obj,list) else obj.get('questions',obj.get('candidates',[]))
  for original in raw:
   q=original.get('original_selected_question',original)
   if q.get('root_id')!=root or q.get('question_type') not in ('multi_select','ordering'):continue
   field='options' if q['question_type']=='multi_select' else 'items'
   values=q.get(field) or q.get('options') or q.get('items')
   if isinstance(values,list):
    values={('I'+str(i+1) if field=='items' else chr(65+i)):(v if isinstance(v,str) else v.get('proposition',v.get('text'))) for i,v in enumerate(values)}
   need(isinstance(values,dict) and all(isinstance(t,str) for t in values.values()),'historical_public_statements_missing:'+str(path)+':'+q['candidate_id'])
   pub={k:copy.deepcopy(q[k]) for k in ('candidate_id','root_id','question_type','question')};pub[field]=copy.deepcopy(values)
   if field=='items':
    pub.update(chronology_kind=q.get('chronology_kind','unspecified_historical_public_inventory'),
      source_path=q.get('source_path') or str(path),source_sha256=q.get('source_sha256') or sha(path))
   key=json.dumps(pub,sort_keys=True)
   if key not in seen:rows.append(pub);seen.add(key)
 p=HOME/'evidence'/f'HISTORICAL_PUBLIC_PRIORS_root{root.split("-")[-1]}.json'
 projection={'questions':rows,'inputs':[bind(p) for p in dict.fromkeys(inputs)],'projection':'verbatim_public_text_only_no_gold_or_evidence','historical_private_fields_encountered':True,'current_private_artifacts_read':False,
  'missing_ordering_metadata_policy':'Unspecified chronology is explicit; missing source metadata binds the historical inventory document for lineage only, not to certify its source or answer.'}
 if p.exists():need(read(p)==projection,'historical_projection_changed')
 else:write(p,projection)
 historical_paths[root]=str(p)
old_pool=read(HOME.parent/'h4_parent_family_flat_writer_v73/MERGED_USABLE_CANDIDATES.json')
parents=[];receipts=[]
for root in roots:
 r=root.split('-')[-1];d=HOME/'parent_review';public_path=d/f'root{r}_PUBLIC.json';jpath=d/f'root{r}_SOURCE_JUDGMENTS.json'
 public=read(public_path);judgment=read(jpath);prep=read(d/f'root{r}_PREPARATION.json')
 need(judgment['public_question']==public and judgment['public_question_sha256']==sha(public_path),'parent_public_binding')
 need(judgment['full_source_read'] is True and judgment['full_source_turn_count']==len(read(public['source_path'])['turns']),'parent_complete_source')
 need(judgment['current_parent_private_gold_read_before_judgments_saved'] is False
      and judgment['current_parent_private_evidence_read_before_judgments_saved'] is False
      and judgment['known_old_parent_gold'] is True,'parent_disclosures')
 js={j['option']:j for j in judgment['judgments']};need(set(js)==set('ABCDEF') and all(js[k]['proposition']==public['options'][k] and js[k]['source_judgment']=='supported' and js[k]['entailed'] is True for k in js),'parent_not_all_supported')
 private_started=now();old=next(q for q in old_pool['questions'] if q['candidate_id']==prep['old_candidate_id'])
 parent=copy.deepcopy(old);parent.update(copy.deepcopy(public));parent['gold']=list('ABCDEF');parent['option_evidence']={}
 turns={t['turn_ref']:t for t in read(public['source_path'])['turns']}
 for k in 'ABCDEF':
  need(bool(js[k]['required_evidence']),'parent_missing_evidence')
  for e in js[k]['required_evidence']:
   need(e['speaker']==turns[e['turn_ref']]['speaker'] and e['quote'] in turns[e['turn_ref']]['text'],'parent_inexact_evidence')
  e=copy.deepcopy(js[k]['required_evidence'][0]);e.update(relation='supports',rationale=js[k]['reason'],reason=js[k]['reason']);parent['option_evidence'][k]=e
 parent.update(parent_origin='v84_known_historical_six_supported_parent_current_source_review_minimal_repair',
    historical_parent_gold_known=True,current_parent_review=bind(jpath),training_allowed=False,
    formal_training_admission=False,baseline_question_recertified=False)
 write(d/f'root{r}_PARENT.json',parent);parents.append(parent)
 provenance={'version':'v84_parent_review_provenance_v1','created_at_utc':now(),
  'parent_candidate_id':parent['candidate_id'],'previous_parent_candidate_id':old['candidate_id'],
  'root_id':root,'source':bind(public['source_path']),'parent_review_path':str(jpath),'parent_review_sha256':sha(jpath),
  'public_question':bind(public_path),'parent':bind(d/f'root{r}_PARENT.json'),'preparation':bind(d/f'root{r}_PREPARATION.json'),
  'source_judgments_saved_at_utc':judgment['created_at_utc'],'current_private_parent_construction_at_utc':private_started,
  'historical_parent_gold_known':True,'fresh_blind_parent_review':False,'full_source_reread_this_batch':True,
  'source_bytes_unchanged':True,'current_parent_changes':prep['changes'],'baseline_question_recertified':False,
  'current_solver_scores_read':False,'training_allowed':False,
  'limitations':'Historical all-six Gold was known; new current public parent was read against the complete source before current private parent construction. This is not blind parent review.'}
 write(d/f'root{r}_PROVENANCE.json',provenance);receipts.append(provenance)
write(HOME/'PARENT_POOL.json',{'questions':parents,'training_allowed':False,'formal_training_admission':False})
value=copy.deepcopy(read(OLD/'CONFIG.json'));value['version']='v84_independent_roots_mixed_batch_v1';value['batch_id']='v84_independent_roots_mixed_01'
m=value['multiselect_config'];template=copy.deepcopy(m['slots'][0]);m['slots']=[];value['ordering_slots']=[]
mpaths=[str(HOME.parent/name/'multiselect/PUBLIC_QUESTIONS.json') for name in ['h4_ready_roots_mixed_batch_v78','h4_output_contract_mixed_batch_v79','h4_scoped_ordering_mixed_batch_v80','h4_bound_scene_ordering_mixed_batch_v81','h4_name_policy_mixed_batch_v83']]
opaths=[str(HOME.parent/name/'ordering/PUBLIC_QUESTIONS.json') for name in ['h4_output_contract_mixed_batch_v79','h4_bound_scene_ordering_mixed_batch_v81','h4_actor_reference_minfix_ordering_v82','h4_name_policy_mixed_batch_v83']]
for parent in parents:
 root=parent['root_id'];r=root.split('-')[-1];s={k:copy.deepcopy(template[k]) for k in ['target_counts','control_count','split','scope_prefix']}
 s.update(slot_id=f'root{r}_family',family_id=f'root{r}_family',root_id=root,parent_candidate_id=parent['candidate_id'],
  parent_pool_path=str(HOME/'PARENT_POOL.json'),source_path=parent['source_path'],source_sha256=parent['source_sha256'],
  history_path=parent['source_path'],history_sha256=parent['source_sha256'],prior_pool_paths=[m['baseline_pool']]+mpaths+[historical_paths[root]],
  parent_review_path=str(HOME/'parent_review'/f'root{r}_SOURCE_JUDGMENTS.json'),parent_review_sha256=sha(HOME/'parent_review'/f'root{r}_SOURCE_JUDGMENTS.json'),
  parent_provenance_path=str(HOME/'parent_review'/f'root{r}_PROVENANCE.json'),parent_provenance_sha256=sha(HOME/'parent_review'/f'root{r}_PROVENANCE.json'),
  parent_origin=parent['parent_origin'],parent_review_mode='saved_pre_generation_full_source_parent_review_with_known_historical_gold',
  parent_gold_known_before_current_review=True,parent_gold_known_before_review=True,fresh_blind_parent_review=False,
  previous_parent_candidate_id=next(v['previous_parent_candidate_id'] for v in receipts if v['root_id']==root),
  current_parent_source_reread_claimed=True)
 m['slots'].append(s)
 o={k:s[k] for k in ['root_id','source_path','source_sha256','history_path','history_sha256']}
 o.update(slot_id=f'root{r}_ordering',candidate_count=2,prior_pool_path=m['baseline_pool'],prior_pool_paths=[m['baseline_pool']]+opaths+[historical_paths[root]]);value['ordering_slots'].append(o)
n=len(parents);m.update(family_count=n,max_http=2*n,max_questions=5*n,max_calls_per_model=90*n,
 target_distribution={**{str(g):n for g in range(2,6)},'6':0},new_child_target_distribution={str(g):n for g in range(2,6)},control_target_distribution={'6':n},final_review_groups=['a','b','c','d'])
value.update(max_generation_http=3*n,max_candidates=7*n,max_solver_calls_per_model=114*n)
for k,name in [('ordering_anchor_plan','SCENE_PLANS.json'),('actor_reference_policy','ACTOR_REFERENCE_POLICY.json'),('character_registry','CHARACTER_REGISTRY.json')]:value[k]=bind(HOME/name)
value['source_consistency']={'upstream_complete':bind(HOME/'SOURCE_ADMISSION.COMPLETE.json'),'selected_roots':list(roots),
 'excluded_held_roots':summary['historical_held_roots_unchanged']+summary['held_roots'],'policy':'explicit_ready_subset_full_sources_unchanged','source_review_is_question_answer_certification':False}
write(HOME/'CONFIG.json',value);write(HOME/'CONFIG_TEMPLATE.json',value)
pins={str(p):sha(p) for p in [HOME/'CONFIG_TEMPLATE.json',HOME/'SOURCE_ADMISSION.COMPLETE.json',HOME/'source_admission.py',HOME/'SCENE_PLANS.json',HOME/'ACTOR_REFERENCE_POLICY.json',HOME/'CHARACTER_REGISTRY.json',HOME/'PARENT_POOL.json',source_admission.GATE]}
source='''"""Validate an explicit immutable new-root plan with unchanged full-source gate."""
from common import HERE, read, need, sha
import source_admission
ROOTS = REPLACE_ROOTS
PINS = REPLACE_PINS

def verified_summary():
    for path, expected in PINS.items():
        need(sha(path)==expected,'upstream_binding_changed:'+path)
    summary=source_admission.verify()
    by_root={r['root_id']:r for r in summary['roots']}
    need(all(r in by_root and by_root[r]['decision']=='READY' for r in ROOTS),'selected_source_not_ready')
    return summary

def validate(value):
    summary=verified_summary();expected=read(HERE/'CONFIG_TEMPLATE.json')
    need(value==expected,'ready_subset_config_changed')
    need(tuple(s['root_id'] for s in value['multiselect_config']['slots'])==ROOTS,'root_selection_changed')
    by_root={r['root_id']:r for r in summary['roots']}
    for s in value['multiselect_config']['slots']:
        need(by_root[s['root_id']]['source']=={'path':s['source_path'],'sha256':s['source_sha256']},'selected_source_binding_changed')
    return value
'''.replace('REPLACE_ROOTS',repr(tuple(roots))).replace('REPLACE_PINS',repr(pins))
with (HOME/'source_subset.py').open('x') as f:f.write(source)
write(HOME/'evidence/PARENT_PREPARATION_COMPLETE.json',{'created_at_utc':now(),'roots':list(roots),'source_before_current_private_gold':True,'historical_parent_gold_known':True,'all_supported':True,'parent_pool':bind(HOME/'PARENT_POOL.json'),'training_allowed':False})
print({'roots':len(roots),'targets':4*n+2*n,'max_generation_http':3*n,'max_solver_calls_per_model':114*n})
