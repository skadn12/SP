"""Generate Gold-blind independent review packets after immutable draft audit."""
import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent

def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
# The audit must be finalized first; no author output is edited.
a=json.loads((R/'ARTIFACT_AUDIT.json').read_text());assert a['frozen_opportunities']==140 and a['pending_opportunities']==0
keys=('candidate_id','root_id','question_type','question','options','query_scope','query_pairs','answer_semantics')
manifest=[];private=[]
for ri,row in enumerate(a['rows']):
 ad=R/'authoring'/row['arm']/row['root_id'];doc=json.loads((ad/'CANDIDATES_FIRST_DRAFT.json').read_text());ai=json.loads((ad/'AUTHOR_INPUT.json').read_text())
 for c in doc['candidates']:
  pid='BR_'+hashlib.sha256(c['candidate_id'].encode()).hexdigest()[:16]
  public={k:c[k] for k in keys if k in c};public.pop('candidate_id',None);public['review_packet_id']=pid
  packet={'review_packet_id':pid,'source_path':ai['source_path'],'source_sha256':ai['source_sha256'],'source_turn_count':ai['source_turn_count'],'neutral_source_info_path':ai['neutral_source_info_path'],'candidate':public,'mechanical_status':row['mechanical_status'],'instruction':'Independently read the complete source and reconstruct answer before any Gold comparison; do not open author directories, private bindings, references, scores or other review outputs. Cite exact turn/speaker/quote. Report each option relation, ambiguity and alternative event matches; distinguish absence from contradiction; no fixes to frozen questions.'}
  op=R/'blind_review_packets'/f'{pid}.json';write(op,packet)
  manifest.append({'review_packet_id':pid,'path':str(op.relative_to(R)),'sha256':h(op)})
  private.append({'review_packet_id':pid,'candidate_id':c['candidate_id'],'root_id':row['root_id'],'arm':row['arm'],'author_draft_path':str(ad/'CANDIDATES_FIRST_DRAFT.json'),'author_draft_sha256':h(ad/'CANDIDATES_FIRST_DRAFT.json'),'mechanical_status':row['mechanical_status']})
write(R/'blind_review_packets/PACKET_MANIFEST.json',{'packets':manifest,'quality_review_started':False,'gold_excluded':True,'solver_scores_absent':True})
write(R/'private_review_bindings.json',private)
write(R/'NEXT_STAGE.json',{'phase':'independent_quality_source_review_then_semantic_dedup_then_new_Mclean_evaluation','review_packet_count':len(manifest),'quality_review_complete':False,'semantic_dedup_complete':False,'evaluation_complete':False,'do_not_train_or_export_unreviewed_drafts':True,'reference_freeze':'private_semantic_reference/REFERENCE_FREEZE.json','sealed135_projection':'private_semantic_reference/SEALED135_QUESTION_PROJECTION_ADDENDUM.json','mechanical_audit':'ARTIFACT_AUDIT.json','evaluation_environment_preflight_required_before_no_retry_attempt':True,'new_evaluation_call_ceiling':5040,'no_legacy_budget_carryover':True})
print(json.dumps({'review_packets':len(manifest),'gold_blind':True,'no_review_or_evaluation_claimed':True}))
