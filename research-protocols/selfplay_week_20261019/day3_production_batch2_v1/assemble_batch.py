"""Export signed, normalized decisions and all inherited slots; never run Solver."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,json
D=Path(__file__).resolve().parent
P=D.parent/'day3_production_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def binding(p):return {'path':str(p),'sha256':sha(p)}
def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def jsonl(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:
  for x in rows:f.write(json.dumps(x,ensure_ascii=False)+'\n')
freeze=read(D/'EXECUTION_FREEZE.json')
slots={x['candidate_slot_id']:x for x in freeze['slots']}
source_reviews=list((D/'source_reviews').glob('cc-*.json'))
assert len(source_reviews)==18
public_paths=sorted((D/'candidates').glob('*/*/public/*.json'))
records={p.stem:read(p) for p in (D/'records').glob('*.json')}
assert len(records)==175 and set(records)=={read(p)['candidate_id'] for p in public_paths}
assert len(records)==len(public_paths)
accepted=[];held=[];allrows=[];bycid={};authors=Counter()
for pubpath in public_paths:
 pub=read(pubpath);cid=pub['candidate_id'];r=records[cid]
 assert r['qa_review']['status'] not in ['NOT_STARTED','CLARIFY_ONCE']
 assert r['fact_family']['duplicate_review'] in ['PASS','FAIL']
 dest=D/'final_records'/(cid+'.json');dump(dest,r)
 entry={'candidate_id':cid,'slot_id':r['inherited_slot_id'],'arm':r['arm'],'root_id':r['root_id'],
   'question_type':r['question_type'],'state':r['disposition']['state'],
   'public':binding(pubpath),'record':binding(dest),
   'source':{'path':r['source_binding']['source_path'],'sha256':r['source_binding']['source_sha256']},
   'fact_family_id':r['fact_family']['family_id'],'target_gold_count':slots[cid]['target_gold_count'],
   'use':'CONTENT_AUDITED_CANDIDATE_NOT_FROZEN_TRAINING_OR_HELD_OUT_EVALUATION_POOL'}
 gold=(r.get('multi_select') or r.get('ordering'))
 entry['gold']=gold.get('gold_option_ids',gold.get('gold_item_ids'))
 bycid[cid]=entry;authors[r['arm']]+=1;allrows.append(entry)
 if r['disposition']['state'].startswith('ACCEPT'):accepted.append(entry)
 else:held.append(entry)
missing=[]
for cid,s in slots.items():
 if cid in bycid:continue
 name=f"{s['question_type']}-{s['local_slot']:02}.json"
 sf=D/'candidates'/s['arm']/s['root_id']/'shortfall'/name
 hold=D/'candidates'/s['arm']/s['root_id']/('GENERATION_HOLD_'+name)
 if sf.exists():state='SHORTFALL';artifact=sf
 elif hold.exists():state='GENERATION_HOLD';artifact=hold
 else:raise RuntimeError('Unresolved inherited slot '+cid)
 entry={'candidate_id':cid,'slot_id':s['slot_id'],'arm':s['arm'],'root_id':s['root_id'],
   'question_type':s['question_type'],'target_gold_count':s['target_gold_count'],
   'state':state,'reason_artifact':binding(artifact),'question_authored':False}
 missing.append(entry);allrows.append(entry)
assert len(allrows)==216
jsonl(D/'exports/content_accepted.public.jsonl',[read(x['public']['path']) for x in accepted])
jsonl(D/'exports/content_accepted.private.jsonl',accepted)
jsonl(D/'exports/held_candidates.jsonl',held)
dump(D/'exports/SLOT_LEDGER.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),'rows':allrows,
    'planned_slots':216,'original_inherited_quotas_changed':False})
pairs=[]
for slot in sorted(set(x['slot_id'] for x in accepted)):
 arms={x['arm']:x['candidate_id'] for x in accepted if x['slot_id']==slot}
 if set(arms)=={'A','B'}:pairs.append({'slot_id':slot,**arms})
dump(D/'MATCHED_SLOT_PAIRS.json',{'pairs':pairs,'matched_slots':len(pairs),'candidates':2*len(pairs),
 'not_training_pool_selection':True,'selection_uses_solver_scores':False})
arms={}
for arm in ['A','B']:
 ar=[x for x in allrows if x['arm']==arm]
 acc=[x for x in accepted if x['arm']==arm]
 arms[arm]={'planned_slots':108,'actual_initial_questions':authors[arm],'accepted':len(acc),
   'content_hold':sum(x['arm']==arm and records[x['candidate_id']]['qa_review']['status']!='PASS' for x in held),
   'dedup_hold':sum(x['arm']==arm and records[x['candidate_id']]['qa_review']['status']=='PASS' and records[x['candidate_id']]['fact_family']['duplicate_review']!='PASS' for x in held),
   'source_shortfalls':sum(x['state']=='SHORTFALL' for x in ar),
   'generation_holds':sum(x['state']=='GENERATION_HOLD' for x in ar),
   'accepted_by_type':dict(Counter(x['question_type'] for x in acc)),
   'accepted_MC_gold_counts':dict(Counter(str(x['target_gold_count']) for x in acc if x['question_type']=='multi_select')),
   'accepted_relations':dict(Counter(records[x['candidate_id']]['target_realization']['primary_relation_id'] for x in acc if x['question_type']=='multi_select')),
   'accepted_ordering_profiles':dict(Counter(records[x['candidate_id']]['target_realization']['ordering_profile_id'] for x in acc if x['question_type']=='ordering'))}
summary={'schema_version':'DAY3_BATCH2_SUMMARY_V1','created_at_utc':datetime.now(timezone.utc).isoformat(),
 'status':'REMAINING_TRAIN_ROOT_PRODUCTION_AND_CONTENT_REVIEW_COMPLETE_COMMON_EVALUATION_PENDING',
 'roots':18,'full_history_semantically_admitted_roots':18,'source_turns_semantically_reviewed':sum(read(p)['actual_read_turn_count'] for p in source_reviews),
 'planned_slots':216,'resolved_slots':len(allrows),'actual_initial_questions':len(records),
 'accepted_questions':len(accepted),'held_questions':len(held),
 'shortfall_slots':sum(x['state']=='SHORTFALL' for x in missing),
 'generation_holds':sum(x['state']=='GENERATION_HOLD' for x in missing),
 'arms':arms,'matched_slot_pairs':len(pairs),'solver_calls':0,'training_runs':0,'source_extensions_created':0,
 'full_day3_complete':False,'training_pool_frozen':False,
 'common_evaluation_actual_questions_frozen':0,
 'original_gold_quota_preserved':True,'accepted_gold_counts_may_differ_due_to_shortfalls':True,
 'presentation_transform_registered':binding(D/'PRESENTATION_TRANSFORM_FREEZE.json'),
 'generation_compute':{'requested_output_tokens_per_item':3072,'actual_provider_cap':None,'actual_tokens':None,
     'scope':'Interactive Codex authored artifacts; not equal realized compute or logged API request evidence.'},
 'boundary':'A input workflow isolation on shared filesystem; not technically blind. B author has narrow historical ordering feedback and cross-arm history disclosed.',
 'fact_independence':'Source roots historically exposed Train. Same-root shared facts grouped conservatively; neither paraphrases nor permutations count as independent knowledge.',
 'causal_claim':'Content/structure process only; no new Solver difficulty, feedback usefulness, or training benefit measured.',
 'next_action':'Finish actual common evaluation material/content/Gold/request freeze, then Day4 symmetric training-pool selection/tokenization/resource freeze.',
 'files':{'accepted_public':binding(D/'exports/content_accepted.public.jsonl'),
          'accepted_private':binding(D/'exports/content_accepted.private.jsonl'),'all_slots':binding(D/'exports/SLOT_LEDGER.json')}}
dump(D/'SUMMARY.json',summary)
print(json.dumps({k:summary[k] for k in ['roots','planned_slots','actual_initial_questions','accepted_questions','held_questions','shortfall_slots','generation_holds','matched_slot_pairs']}))
