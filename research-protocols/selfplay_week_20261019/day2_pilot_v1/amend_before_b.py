import pathlib,json,hashlib,datetime,shutil
P=pathlib.Path(__file__).resolve().parent
def bind(q):
    z=q.read_bytes();return dict(path=str(q),sha256=hashlib.sha256(z).hexdigest(),bytes=len(z))
def save(q,d):q.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
assert not (P/'EXECUTION_AMENDMENT_01.json').exists()
assert not list((P/'drafts/B').glob('*.json'))
old=P/'audit/pre_amendment_packets';old.mkdir(exist_ok=True)
for q in (P/'packets/B').glob('*.json'):shutil.copy2(q,old/q.name)
cards=json.loads((P.parent/'day2_feedback_readiness_v1/OPERATIONAL_CARDS.json').read_text())['cards']
cm={x['card_id']:x for x in cards};overrides=[]
for q in (P/'packets/B').glob('*.json'):
    if q.name.endswith('_feedback.json'):continue
    d=json.loads(q.read_text())
    for s in d['slots']:
        if s['question_type']=='multi_select':
            s['expected_B_card_ids']=['D2-MC-ATTRIBUTE']
            overrides.append(dict(slot_id=s['slot_id'],expected_B_card_ids=['D2-MC-ATTRIBUTE']))
    f=P/'packets/B'/(d['root_id']+'_feedback.json')
    save(f,dict(cards=[cm['D2-MC-ATTRIBUTE'],cm['D2-ORDER-FIXED-CITATIONS']],original_collection=bind(P.parent/'day2_feedback_readiness_v1/OPERATIONAL_CARDS.json')))
    d['private_feedback']=bind(f);d['effective_amendment']=str(P/'EXECUTION_AMENDMENT_01.json');save(q,d)
a=dict(schema_version='day2_pilot_prospective_amendment_01',created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='FROZEN_BEFORE_B_AUTHORSHIP_OR_ANY_NEW_SOLVER_INFERENCE',base_freeze=bind(P/'EXECUTION_FREEZE.json'),reason='Enforce PLAN_V2 max two directions: MC attribute-binding hypothesis plus narrow operational cross-scene ordering. Other MC cards unused. Record requested author cap; enforcement unknown and no equal compute claim.',B_author_started=False,B_drafts_existing=0,A_author_started=True,A_drafts_existing_at_amendment=len(list((P/'drafts/A').glob('*.DRAFT.json'))),A_targets_sources_slots_unchanged=True,hypotheses_or_directions=['D2-MC-ATTRIBUTE','D2-ORDER-FIXED-CITATIONS'],slot_overrides=overrides,
author_budget=dict(requested_output_cap_tokens_per_question=3072,applies_to=['A','B'],backend_enforced=None,actual_usage_tokens=None,unknown_not_zero=True,one_initial_draft_one_max_content_clarification=True,agent_context_reads_and_planning_cost_included_when_available=True,no_equal_compute_or_billed_cost_claim=True),packets=[bind(q) for q in sorted((P/'packets').glob('*/*.json'))],source_eligibility_semantic_review_before_score=True)
save(P/'EXECUTION_AMENDMENT_01.json',a)
print(json.dumps(dict(amendment=bind(P/'EXECUTION_AMENDMENT_01.json'),A_drafts_existing=a['A_drafts_existing_at_amendment'],B_directions=a['hypotheses_or_directions']),ensure_ascii=False))
