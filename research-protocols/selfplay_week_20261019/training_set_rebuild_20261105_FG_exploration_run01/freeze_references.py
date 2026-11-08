"""Freeze a private, score-free reference corpus; never supplied to authors."""
import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;W=R.parent
S=W/'training_set_rebuild_20261105_F_supplement_run01';P=W/'training_set_rebuild_20261104_run01'
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
refs={};bindings=[]
keys=['candidate_id','root_id','arm','question_type','question','options','items','event_identity','cognitive_task','natural_information_need','relation_plan','query_scope','query_pairs','answer_semantics','evidence','option_evidence','item_evidence','evidence_refs','macrostructure']
def add(items,p,origin):
 bindings.append({'path':str(p),'sha256':h(p),'origin':origin})
 for x in items:
  if not isinstance(x,dict) or not x.get('candidate_id') or not x.get('root_id'):continue
  key=x['candidate_id'];y={k:x[k] for k in keys if k in x};y['record_origin']=origin
  if key in refs and refs[key].get('question')!=y.get('question'):
   key=key+'__'+hashlib.sha256(json.dumps(y,sort_keys=True).encode()).hexdigest()[:12]
  if key not in refs:refs[key]=y
for p in sorted((S/'semantic_review').glob('REFERENCE_cc*.json')):add(json.loads(p.read_text()),p,'prior_FG_and_v125_legacy')
for p in sorted(P.glob('batch_*/cc*/[FG]/author/CANDIDATES_FIRST_DRAFT.json')):add(json.loads(p.read_text()).get('candidates',[]),p,'all_prior_FG_first_drafts')
for p in sorted(S.glob('authoring/cc*/CANDIDATES_FIRST_DRAFT.json')):add(json.loads(p.read_text()).get('candidates',[]),p,'F_supplement_all_first_drafts')
for name in ['training_set_rebuild_20261105_three_root_retry_v1','training_set_rebuild_20261105_three_root_G_retry_v1']:
 for p in sorted((W/name).glob('authors/cc*/CANDIDATES_FIRST_DRAFT.json')):
  d=json.loads(p.read_text());add(d if isinstance(d,list) else d.get('candidates',[]),p,'three_root_retry_first_drafts')
for num,shard in [('194259','19'),('87273','54')]:
 p=Path('/root/autodl-tmp/gxy-selfplay/SP-820-publish/first_training_pipeline/data/qa_generation_only/shards')/('shard-'+shard)/'roots'/('cc-train-episode-'+num)/'candidate_pool.json'
 add(json.loads(p.read_text()),p,'round0_extra_root_all_candidates')
# Day1 historical source-first inventory includes older 87273 MC/order tasks.
p=W/'day1/reuse/INVENTORY.json';d=json.loads(p.read_text())
if isinstance(d,list):add(d,p,'day1_legacy_inventory')
else:
 for key in ['questions','items','records','candidates']:
  if isinstance(d.get(key),list):add(d[key],p,'day1_legacy_inventory');break
for p in [P/'training_export_v1/TRAIN.jsonl',S/'repaired_training_export_v1/TRAIN.jsonl',S/'repaired_training_export_v1/QUESTION_LINEAGE.jsonl']:
 bindings.append({'path':str(p),'sha256':h(p),'rows':len(p.read_text().splitlines()),'origin':'sealed_training_and_lineage'})
# The complete 135 training question projection is already score-free and independently bound.
p=S/'semantic_review/SEALED_TRAIN_QUESTION_PROJECTION.json'
if p.exists():
 d=json.loads(p.read_text())
 if isinstance(d,list):add(d,p,'sealed_training_question_projection')
 else:
  for k in ['questions','items','records']:
   if isinstance(d.get(k),list):add(d[k],p,'sealed_training_question_projection');break
for root in json.loads((W/'day1/TRAIN_ROOTS.json').read_text())['root_ids']:
 write(R/'private_semantic_reference'/f'REFERENCE_{root}.json',[v for v in refs.values() if v['root_id']==root])
write(R/'private_semantic_reference/REFERENCE_FREEZE.json',{'created_before_semantic_review':True,'private_not_author_input':True,'record_count':len(refs),'root_count':len({x['root_id'] for x in refs.values()}),'input_bindings':bindings,'projection_excludes_gold_solver_scores_review_decisions':True,'historical_scope_limit':'Extra-root round0 pools and Day1 inventory are included; additional old offline-delivery parent-family candidates require retrieval at semantic review. No novelty determination is made by this corpus.'})
print(json.dumps({'records':len(refs),'roots':len({x['root_id'] for x in refs.values()}),'bindings':len(bindings)}))
