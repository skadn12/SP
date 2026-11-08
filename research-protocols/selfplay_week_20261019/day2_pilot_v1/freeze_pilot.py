import datetime, hashlib, json, pathlib, shutil
P=pathlib.Path(__file__).resolve().parent
W=P.parent
def bind(p):
    z=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(z).hexdigest(),bytes=len(z))
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');return bind(p)
assert not (P/'EXECUTION_FREEZE.json').exists(), 'Do not overwrite an execution freeze'
for d in ['common/sources','packets/A','packets/B','drafts/A','drafts/B','reviews','private','public','audit','status_before','runtime']:(P/d).mkdir(parents=True,exist_ok=True)
for n in ['CURRENT_STATUS.json','DAY2_LATEST.json']:shutil.copy2(W/n,P/'status_before'/n)
boundary=json.loads((W/'day1/evaluation/ROOT_BOUNDARY.json').read_text())
excluded={x['root_id'] for x in json.loads((W/'day1/diagnosis_inputs/PUBLIC_QUESTIONS.json').read_text())['questions']}
salt='20261019|day2-pilot-v1|'
ordered=sorted([x for x in boundary['roots'] if x['original_split']=='train' and x['root_id'] not in excluded],key=lambda x:(hashlib.sha256((salt+x['root_id']).encode()).hexdigest(),x['root_id']))
eligible=[];accounting=[]
scope=[f'R02:{arm}:S0{i}' for arm in ['A','B'] for i in range(1,4)]
for x in ordered:
    bindings=x['source_lineage_bindings'];ok=False
    if bindings:
        src=pathlib.Path(bindings[0]['path'])
        if src.exists():
            source=json.loads(src.read_text());refs=[t['turn_ref'] for t in source['turns']]
            scenes={r.rsplit(':',1)[0] for r in refs}
            ok=bind(src)['sha256']==bindings[0]['sha256'] and len(refs)==len(set(refs)) and set(scope)<=scenes
    accounting.append(dict(root_id=x['root_id'],hash_rank=hashlib.sha256((salt+x['root_id']).encode()).hexdigest(),source_structurally_eligible=ok,not_semantic_admission=True))
    if ok:eligible.append(x)
roots=eligible[:6];assert len(roots)==6
formats=[('claim_audit','reported_actions'),('final_state_audit','service_final_confirmations'),('joint_condition_audit','actions_and_service_results'),('correction_audit','recorded_decisions'),('person_action_audit','reported_completions'),('status_audit','service_and_person_reports')]
counts=[2,3,4,5,2,3]
mc_cards=['D2-MC-ATTRIBUTE','D2-MC-REVISION','D2-MC-PLAN-COMPLETION','D2-MC-CONJUNCTION','D2-MC-ATTRIBUTE','D2-MC-REVISION']
slots=[];rootrows=[]
for i,x in enumerate(roots):
    rid=x['root_id'];src=pathlib.Path(x['source_lineage_bindings'][0]['path']);dst=P/'common/sources'/f'{rid}.json';shutil.copy2(src,dst)
    sb=bind(dst);rootrows.append(dict(root_id=rid,rank=i,hash_rank=hashlib.sha256((salt+rid).encode()).hexdigest(),source=sb,original_source=bind(src),history=x,scenes=scope))
    for arm in ['A','B']:
        for typ in ['multi_select','ordering']:
            suffix='MC' if typ=='multi_select' else 'ORDER'
            cardids=[] if arm=='A' else [mc_cards[i]] if typ=='multi_select' else ['D2-ORDER-FIXED-CITATIONS']
            slots.append(dict(slot_id=f'D2P1-{i+1:02d}-{arm}-{suffix}',candidate_id=f'd2p1-{rid}-{arm.lower()}-{suffix.lower()}',root_id=rid,root_index=i,arm=arm,question_type=typ,item_count=6,gold_count=counts[i] if typ=='multi_select' else 6,coarse_format=formats[i][0 if typ=='multi_select' else 1],source=sb,family_id=rid+'|'+typ,expected_B_card_ids=cardids,target_status='DIVERSITY_ONLY' if arm=='A' else 'NARROW_PRIOR_OPERATIONAL_REPRODUCED_NEW_CONTENT_UNTESTED' if typ=='ordering' else 'EXPLORATORY_HYPOTHESIS_ONLY'))
common=dict(schema_version='pilot_common_authoring_v1',context_policy='Full history; question truth restricted to six public R02 scenes; no truncation',source_scope=scope,item_count=6,formal_70_percent_targeted_training=False,
format_library={'claim_audit':'Select all claims a careful dialogue summary can include.','final_state_audit':'Select supported claims about final recorded states; each option explicitly states temporal scope.','joint_condition_audit':'Select claims satisfying all stated constraints; every clause must be verified.','correction_audit':'Select all accurate audit notes, including clearly stated corrections.','person_action_audit':'Select all correctly attributed person/action/object reports.','status_audit':'Select supported records distinguishing requested, confirmed and completed statuses.','ordering':'Six reported events in textual first-appearance order within allowed scope, not inferred real-world time. Vary predicates and question focus across roots; cross-scene order for this pilot.'},
content_rules=['Use source facts only; reported service responses do not prove real-world truth.','Use names, no unsupported gender pronouns.','MC six unique independent propositions; follow private author Gold quota, never expose quota to Solver. False options need explicit source contradiction or full-scope unsupported verdict.','Every label needs exact turn_ref and verbatim quote plus concise reason.','Ordering requires six supported events, unambiguous first target occurrence; inspect repeats and plans/completions. Record primary_turn_ref per item.','Do not use scores for authorship or revision. One initial draft per slot, max one content clarification; retain all versions.','Do not extend or alter source in this pilot.','Avoid duplicate/subsuming options and scene ordinals or obvious sequence words as answer shortcuts.'],
output_schema={'questions':[{'slot_id':'exact frozen slot','candidate_id':'exact ID','root_id':'exact root','arm':'A or B','question_type':'multi_select or ordering','question':'English question defining six-scene scope','options_or_items':'options A..F or items I1..I6 object','gold':'list of true labels or ordered labels','evidence':'label -> {judgment:supported|contradicted|not_supported,citations:[{turn_ref,quote}],reason_zh,primary_turn_ref for ordering}','design':{'coarse_format':'frozen slot format','source_material':'actual source facts','target_relation':'specific construction','retrieved_card_ids':'[] for A, actual assigned reads for B','target_labels':'MC target labels; ordering all six','falsification':'specific controls','shortcuts_checked':'brief list'}}]})
commonbinding=save(P/'common/AUTHORING_PROTOCOL.json',common)
cards=json.loads((W/'day2_feedback_readiness_v1/OPERATIONAL_CARDS.json').read_text())['cards'];cardmap={c['card_id']:c for c in cards}
for arm in ['A','B']:
    for row in rootrows:
        selected=[s for s in slots if s['root_id']==row['root_id'] and s['arm']==arm]
        packet=dict(schema_version='day2_challenger_packet_v1',arm=arm,root_id=row['root_id'],source_binding=row['source'],common_protocol=commonbinding,slots=selected,input_boundary='A reads source/common/slots only. B additionally reads exactly assigned feedback. Fresh-context A agents; parent already feedback-exposed. Shared filesystem; not externally blind.',shared_filesystem_disclosed=True)
        if arm=='B':
            ids=sorted({c for s in selected for c in s['expected_B_card_ids']})
            packet['private_feedback']=save(P/'packets/B'/f'{row["root_id"]}_feedback.json',dict(cards=[cardmap[c] for c in ids],original_collection=bind(W/'day2_feedback_readiness_v1/OPERATIONAL_CARDS.json')))
        save(P/f'packets/{arm}/{row["root_id"]}.json',packet)
freeze=dict(schema_version='day2_new_question_pilot_freeze_v1',frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='FROZEN_BEFORE_AUTHORSHIP',namespace='train.feedback/diagnostic_only',roots=rootrows,eligibility=accounting,
selection_rule=dict(population='70 authoritative Train minus original20 diagnostic roots; source structural eligibility required',hash_salt=salt,order='SHA256(salt+full_root_id), full_root_id ascending; first6 eligible; no scores',excluded_diagnostic_roots=sorted(excluded),train_candidates=50,structurally_eligible=len(eligible),not_pristine_or_unseen=True),slots=slots,common_protocol=commonbinding,B_feedback_collection=bind(W/'day2_feedback_readiness_v1/OPERATIONAL_CARDS.json'),question_drafts_per_slot=1,max_content_clarifications_per_slot=1,
solver=dict(name='S0_M_clean',model='/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_common_dev_eval_v1_live_3/clean_merged_model',tokenizer='/root/autodl-tmp/Qwen3-4B',same_model_contract=bind(W/'day2/solver/evidence_v2/manifest.json'),MC_mappings=6,MC_samples_each_mapping=3,ordering_mappings=12,ordering_samples_each_mapping=1,temperature=0.8,top_p=0.9,max_tokens=32,enable_thinking=False,max_model_len=16384,source_truncation=False,automatic_retry=False,seed_base=2026101917),
budget=dict(initial_question_slots=24,max_direct_requests=360,additional_controlled_probe_calls_reserved=24,max_diagnostic_new_question_evidence_calls=0,currency_cost=None,unknown_not_zero=True),
predeclared_predictions=dict(B_ordering='At least3 source-qualified new-root realizations of cross-scene occurrence comparison can be authored using feedback; report target/non-target errors without posthoc error threshold.',B_multi='Realize assigned hypothesis-specific source structure when present; measure targeted item FP/FN; no automatic reproduction upgrade.',A_B_difference='Descriptive only: distinct questions cannot identify feedback causality.'),
GO_rule=dict(existing_D2_01_to_03_verified=True,actual_logged_B_reads_and_plan_to_candidate_lineage=True,content_admitted_narrow_reproduced_target_realizations_min_roots=3,min_data_families=3,family_definition='root|type not structural genres',new_question_error_replication='report, do not add positive error threshold',other_type_gaps_recorded=True,formal_MC_targeted_quota_not_admitted_from_hypotheses=True,no_training_utility_claim=True),failure_policy='No score-selected replacement root/draft; rejected slots remain missing. NO_GO if gate fails.',input_bindings=[bind(W/'CURRENT_STATUS.json'),bind(W/'FEEDBACK_CONTRACT_V2.json'),bind(W/'PLAN_V2_ZH.md'),bind(W/'day1/evaluation/ROOT_BOUNDARY.json')])
save(P/'EXECUTION_FREEZE.json',freeze)
save(P/'EXPOSURE_UPDATE.json',dict(status='NEW_TRAIN_FEEDBACK_DESIGN_EXPOSURE_FROM_FREEZE',roots=[dict(root_id=x['root_id'],previous_week_wave=x['history']['week_wave'],historical_exposure=x['history']['historical_exposure_records'],new_role='pilot_feedback_design_and_S0_diagnostic_not_final_eval') for x in rootrows]))
print(json.dumps(dict(roots=[x['root_id'] for x in rootrows],slots=len(slots),freeze=bind(P/'EXECUTION_FREEZE.json')),ensure_ascii=False))
