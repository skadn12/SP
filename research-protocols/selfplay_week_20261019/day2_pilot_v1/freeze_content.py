"""Admit exactly source-reviewed slots before any pilot Solver request."""
import json,hashlib,os
from datetime import datetime,timezone
from pathlib import Path
P=Path(__file__).resolve().parent
def read(p): return json.loads(p.read_text())
def bind(p):
    p=p.resolve(); b=p.read_bytes()
    return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f: json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
def verify(row):
    assert bind(Path(row['path']))['sha256']==row['sha256'],row['path']
assert not (P/'CONTENT_FREEZE.json').exists()
assert not (P/'runtime/direct/STARTED.json').exists(),'NO_POST_SCORE_ADMISSION'
base=read(P/'EXECUTION_FREEZE.json'); amendment=read(P/'EXECUTION_AMENDMENT_01.json')
slots={q['candidate_id']:q for q in base['slots']};assert len(slots)==24
reviews={};review_bindings=[]
for arm in ['A','B']:
    path=P/f'reviews/{arm}/INDEPENDENT_REVIEW.json'; review=read(path)
    assert review['status']=='PASS',review['status']
    review_bindings.append(bind(path))
    assert len(review['questions'])==12
    for q in review['questions']:
        assert q['candidate_id'] in slots and q['decision'] in ['PASS','REJECT']
        verify(q['selected_artifact']);reviews[q['candidate_id']]=q
assert set(reviews)==set(slots)
all_attempts=[bind(path) for path in sorted((P/'drafts').rglob('*'))
             if path.is_file() and path.suffix in ('.json','.jsonl','.py')]
attempts_path=P/'AUTHORING_ATTEMPTS_FREEZE.json'
save(attempts_path,{'status':'FROZEN_BEFORE_SOLVER','created_at_utc':datetime.now(timezone.utc).isoformat(),
 'planned_slots':24,'initial_draft_roots_per_arm':6,'files':all_attempts,
 'all_originals_clarifications_read_logs_and_plans_retained':True,'new_solver_calls':0})
public=[];private=[];admitted=[];shortfalls=[];gold_counts={'A':[],'B':[]}
originals=[]
for cid,slot in slots.items():
    review=reviews[cid]
    if review['decision']=='REJECT':
        shortfalls.append({'candidate_id':cid,'slot_id':slot['slot_id'],'reason':review.get('reason',review.get('semantic_review_zh','CONTENT_REJECT'))});continue
    artifact=Path(review['selected_artifact']['path'])
    q=next(q for q in read(artifact)['questions'] if q['candidate_id']==cid)
    assert q['slot_id']==slot['slot_id'] and q['arm']==slot['arm']
    assert q['root_id']==slot['root_id'] and q['question_type']==slot['question_type']
    source=P/'common/sources'/f"{q['root_id']}.json"
    assert bind(source)['sha256']==slot['source']['sha256']
    field='options' if q['question_type']=='multi_select' else 'items'
    assert len(q[field])==6 and len(q['gold'])==slot['gold_count']
    assert len(set(q['gold']))==len(q['gold']) and set(q['gold'])<=set(q[field])
    turns={t['turn_ref']:t for t in read(source)['turns']}
    for label,ev in q['evidence'].items():
        for c in ev['citations']:assert c['turn_ref'].startswith('R02:') and c['quote'] in turns[c['turn_ref']]['text']
        if field=='options':assert (ev['judgment']=='supported')==(label in q['gold'])
    pub={k:q[k] for k in ['candidate_id','root_id','question_type','question',field]}
    pub.update(source_path=str(source.resolve()),source_sha256=bind(source)['sha256'])
    priv=dict(q);priv.update(pub)
    public.append(pub);private.append(priv)
    admitted.append({'candidate_id':cid,'slot_id':slot['slot_id'],'root_id':slot['root_id'],
       'question_type':slot['question_type'],'arm':slot['arm'],
       'content_decision':'PASS','lineage_valid':review.get('lineage_valid',None),
       'target_realized':review.get('target_realized',None),'artifact':bind(artifact)})
    originals.append(bind(artifact))
    if field=='options':gold_counts[slot['arm']].append(len(q['gold']))
public_path=P/'public/PUBLIC_QUESTIONS.json';private_path=P/'private/PRIVATE_QUESTIONS.json'
save(public_path,{'questions':public});save(private_path,{'questions':private})
private_path.chmod(0o600)
lineage_roots=sorted({q['root_id'] for q in admitted if q['arm']=='B' and q['question_type']=='ordering' and q['lineage_valid'] and q['target_realized']})
payload={'schema_version':'day2_pilot_content_freeze_v1','status':'PASS','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
 'namespace':'train.feedback/diagnostic_only','planned_slots':24,'admitted_count':len(public),'questions':admitted,'shortfalls':shortfalls,
 'public_questions':bind(public_path),'private_questions':bind(private_path),
 'base_freeze':bind(P/'EXECUTION_FREEZE.json'),'prospective_amendment':bind(P/'EXECUTION_AMENDMENT_01.json'),
 'independent_source_reviews':review_bindings,'admitted_artifact_bindings':originals,
 'all_authoring_attempts_freeze':bind(attempts_path),
 'MC_gold_counts_in_frozen_slot_order':gold_counts,'formal_narrow_ordering_lineage_target_roots':lineage_roots,
 'formal_narrow_ordering_family_count':len(lineage_roots),'formal_MC_targeted_training_admitted':False,
 'solver_inference_started':False,'training_started':False,'new_source_expansion':False,
 'limitations':['Exploratory different-question A/B; no feedback causal effect from score difference.',
 'Historically exposed Train roots; no pristine evaluation claim.',
 'Requested author token caps unenforced or unknown; no strict equal compute claim.',
 'Ordering remains textual first appearance; verb variation is not structural genre diversity.',
 'Only lineage-valid B ordering targets count toward formal feedback-process readiness.']}
save(P/'CONTENT_FREEZE.json',payload)
print(json.dumps({'status':'PASS','admitted_count':len(public),'planned_slots':24,'lineage_target_roots':lineage_roots,'content_freeze':bind(P/'CONTENT_FREEZE.json')}))
