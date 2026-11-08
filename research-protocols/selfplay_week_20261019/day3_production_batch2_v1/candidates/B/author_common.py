"""B-author artifact writer. No Solver and no independent QA claim.
Definitions are authored once after full source read and independent source PASS.
Existing PLAN/public/private files are never overwritten.
"""
import hashlib, itertools, json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
D = OUT.parents[1]
V1 = D.parent / 'day3_production_v1'

def now(): return datetime.now(timezone.utc).isoformat()
def bind(p):
    p = Path(p); b = p.read_bytes()
    return {'path':str(p), 'sha256':hashlib.sha256(b).hexdigest(), 'bytes':len(b)}
def load(p): return json.loads(Path(p).read_text())
def save(p, obj):
    p = Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    assert not p.exists(), f'Immutable artifact already exists: {p}'
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def ref(x): return x if x.startswith('R0') or ':P0' in x else 'R02:'+x

def author(root, mc1, mc2, orders, shortfall3, read_at, extra_note='', shortfall2='', recover_existing_manifest=False, generation_holds=None):
    """MC option=(text,truth,refs,reason); ORDER item=(text,selected,mentions).
    Mention=(ref,included,reason). Entire source is read before this function.
    """
    rid='cc-train-episode-'+root
    target=OUT/rid; packet=load(D/'packets/B'/f'{rid}.json')
    sourcep=Path(packet['source_binding']['path']); source=load(sourcep)
    reviewp=D/'source_reviews'/f'{rid}.json'; review=load(reviewp)
    status=review.get('status',review.get('source_review',{}).get('status'))
    assert status in ('PASS','PASS_FOR_SCOPED_AUTHORING'), (rid,status)
    assert bind(sourcep)['sha256']==packet['source_binding']['sha256']
    turnmap={t['turn_ref']:(i,t) for i,t in enumerate(source['turns'])}
    def ev(r):
        r=ref(r); i,t=turnmap[r]
        return {'turn_ref':r,'global_turn_index_zero_based':i,'scene_ref':t.get('scene_ref'),'speaker':t['speaker'],'quote':t['text'],'span_start':0,'span_end':len(t['text'])}
    receipt=OUT/'FEEDBACK_READ_RECEIPT.json'
    manifest={'schema_version':'day3_b2_author_read_v1','root_id':rid,'arm':'B','author':'/root/day3_public_design','full_source_read_completed_at_utc':read_at,
      'source_binding':bind(sourcep),'all_turn_count':len(source['turns']),'read_ranges_zero_based_half_open':[[0,len(source['turns'])]],'full_parent_R01_R02_semantic_read':True,
      'source_review_binding':bind(reviewp),'source_review_status':status,'source_review_read_completed_at_utc':now(),
      'packet_binding':bind(D/'packets/B'/f'{rid}.json'),'public_library_binding':bind(V1/'design/PUBLIC_DESIGN_LIBRARY.json'),
      'common_scope_amendment_binding':bind(V1/'DAY3_SCOPE_AMENDMENT.json'),'common_addendum_binding':bind(V1/'COMMON_INPUT_ADDENDUM_V2.json'),
      'feedback_read_receipt_binding':bind(receipt),'feedback_card_binding':bind(V1/'packets/B/ORDERING_FEEDBACK_BOUNDED.json'),
      'feedback_scope':'Cross-scene ordering only. Within-scene profiles common design. MC common-only despite author arm exposure.',
      'prior_exposure_disclosure':'Author wrote first-batch A103866/public library and reviewed A104814 plus all20 final first-batch publics in dedup. Current work B only; no A batch2 candidates read.',
      'other_scores_or_responses_read':False,'solver_calls':0,'source_read_note':extra_note}
    generation_holds=generation_holds or {}
    if recover_existing_manifest:
        assert generation_holds and (target/'READ_MANIFEST.json').exists() and not (target/'PLAN.json').exists()
        assert load(target/'READ_MANIFEST.json')['source_binding']==bind(sourcep)
    else:
        save(target/'READ_MANIFEST.json',manifest)
    slots=packet['slots']; defs=[mc1,mc2,None]+orders
    shortfall_reasons={2:shortfall2,3:shortfall3}
    plans=[]
    for s,definition in zip(slots,defs):
        if s['slot_id'] in generation_holds:
            plans.append({'slot_id':s['slot_id'],'candidate_id':s['candidate_slot_id'],'design_id':s['design_id'],'target_gold_count':s['target_gold_count'],'decision':'GENERATION_HOLD','reason':generation_holds[s['slot_id']],
                'failed_original_option_propositions':{l:o[0] for l,o in zip('ABCDEF',definition)},'failed_original_proposed_gold':[l for l,o in zip('ABCDEF',definition) if o[1]],'source_of_original_definition':'author_'+root+'.py','no_rewrite':True})
        elif definition is None:
            reason=shortfall_reasons[s['local_slot']]
            assert reason
            plans.append({'slot_id':s['slot_id'],'candidate_id':s['candidate_slot_id'],'design_id':s['design_id'],'target_gold_count':s['target_gold_count'],'decision':'SHORTFALL','reason':reason})
        elif s['question_type']=='multi_select':
            g=[l for l,o in zip('ABCDEF',definition) if o[1]]
            assert len(g)==s['target_gold_count'],(root,s['design_id'],g,s['target_gold_count'])
            plans.append({'slot_id':s['slot_id'],'candidate_id':s['candidate_slot_id'],'design_id':s['design_id'],'target_gold_count':s['target_gold_count'],'gold':g,
                'frozen_option_propositions':{l:o[0] for l,o in zip('ABCDEF',definition)},'evidence_refs_by_label':{l:[ref(r) for r in o[2].split()] for l,o in zip('ABCDEF',definition)}})
        else:
            assert len(definition)==6
            refs=[ref(o[1]) for o in definition]; assert len(set(refs))==6
            gold=sorted('ABCDEF',key=lambda l:turnmap[refs[ord(l)-65]][0])
            scenes=Counter(turnmap[r][1]['scene_ref'] for r in refs)
            expected={'ORDER_3_2_1':[3,2,1],'ORDER_2_2_2':[2,2,2],'ORDER_4_2':[4,2]}[s['design_id']]
            assert sorted(scenes.values(),reverse=True)==expected,(root,s['design_id'],scenes)
            plans.append({'slot_id':s['slot_id'],'candidate_id':s['candidate_slot_id'],'design_id':s['design_id'],'gold':gold,
                'frozen_event_propositions':{l:o[0] for l,o in zip('ABCDEF',definition)},'selected_turn_refs':dict(zip('ABCDEF',refs)),
                'actual_scene_event_counts':dict(scenes),'profile':expected})
    plan={'schema_version':'day3_b2_B_frozen_plan_v1','root_id':rid,'arm':'B','author':'/root/day3_public_design','frozen_at_utc':now(),
          'read_manifest_binding':bind(target/'READ_MANIFEST.json'),'source_binding':bind(sourcep),'source_review_binding':bind(reviewp),
          'slots':plans,'ordering_scope':'First semantically equivalent report in complete source; actor, object and action modality define event. No real-world date ordering.',
          'feedback_to_plan':{'card_id':'D2-ORDER-FIXED-CITATIONS','card_read_before_plan':True,'card_binding':bind(V1/'packets/B/ORDERING_FEEDBACK_BOUNDED.json'),
          'targeted_slots':[p['candidate_id'] for p in plans if p['design_id'].startswith('ORDER')],
          'mapping':'Unambiguous first reported events across scenes, nonmonotonic displayed labels, exact verified positions private, all plausible prior/later mentions retained. Within-scene profile is common design, not a feedback diagnosis.',
          'MC_slots_targeted':False,'diagnostic_internal_cause_claim':False},
          'budget':{'initial_allowed_per_slot':1,'clarification_allowed_per_slot':1,'requested_tokens_per_question':3072,'actual_cap':None,'actual_usage':None,'bulk_artifact_writing_is_not_claimed_API_calls':True},
          'no_rewrite_or_score_selection':True,'qa_status':'NOT_STARTED_INDEPENDENT_REVIEW_REQUIRED'}
    save(target/'PLAN.json',plan)
    for s,definition,p in zip(slots,defs,plans):
        filename=f"{s['question_type']}-{s['local_slot']:02}.json"
        if p.get('decision')=='GENERATION_HOLD':
            save(target/('GENERATION_HOLD_'+filename),{**p,'root_id':rid,'arm':'B','author':'/root/day3_public_design','initial_used':1,'clarification_used':0,'no_public_question_generated':True,'original_attempt_script_binding':bind(OUT/('author_'+root+'.py')),'plan_binding':bind(target/'PLAN.json'),'note':'Unaffected ordering definitions recovered exactly from original author arguments, without reauthoring failed MC propositions.'})
            continue
        common={'candidate_id':s['candidate_slot_id'],'inherited_slot_id':s['slot_id'],'root_id':rid,'arm':'B','question_type':s['question_type'],'design_id':s['design_id'],
          'source_binding':bind(sourcep),'source_review_binding':bind(reviewp),'read_manifest_binding':bind(target/'READ_MANIFEST.json'),'plan_binding':bind(target/'PLAN.json'),
          'created_at_utc':now(),'author':'/root/day3_public_design','initial_used':1 if definition is not None else 0,'clarification_used':0,
          'requested_output_tokens':3072,'actual_cap':None,'actual_usage':None,'solver_calls':0,'disposition':'QA_PENDING' if definition is not None else 'SHORTFALL',
          'feedback_mapping':{'card_read_receipt_binding':bind(receipt),'card_binding':bind(V1/'packets/B/ORDERING_FEEDBACK_BOUNDED.json'),
            'targeted':s['question_type']=='ordering','scope':'cross-scene order only' if s['question_type']=='ordering' else 'common public MC relation; no diagnosed MC target',
            'plan_candidate_id':s['candidate_slot_id'],'within_scene_profile_is_common_design':True},
          'fact_family':{'family_id':rid+'|'+s['design_id'],'duplicate_review':'UNKNOWN_PENDING_INDEPENDENT_BATCH_DEDUP','historical_semantic_independence':'UNKNOWN','new_independent_knowledge_claim':False}}
        if definition is None:
            criteria={'J1':'NOT_REALIZED','J2':'NOT_REALIZED','J3':'NOT_REALIZED'} if s['local_slot']==3 else {'S1':'NOT_REALIZED','S2':'NOT_REALIZED','S3':'NOT_REALIZED'}
            save(target/'shortfall'/filename,{**common,'target_gold_count':s['target_gold_count'],'reason_code':'SHORTFALL_SOURCE_RELATION','reason':shortfall_reasons[s['local_slot']],'no_public_question_generated':True,'designcriteria':criteria})
            continue
        question=("Check only which person actually reports the listed content in the complete dialogue; do not infer what another person might also know. Select all accurate speaker-and-content pairings." if s['local_slot']==1 else
          "By the end of the complete dialogue, which statements preserve the explicitly revised choices and their effective reported status? Select all supported statements, keeping plans, requests and completed actions distinct.") if s['question_type']=='multi_select' else "Order these six reports by their first semantically equivalent appearance in the complete supplied dialogue. Keep intentions, requests, actions in progress and completed actions distinct; use dialogue appearance, not real-world dates or causal expectations. Output the six letters."
        public={'candidate_id':s['candidate_slot_id'],'root_id':rid,'question_type':s['question_type'],'question':question,'options_or_items':{l:o[0] for l,o in zip('ABCDEF',definition)}}
        private={**common,'gold':p['gold'],'evidence_by_label':{},'lexical_shortcuts':{'coordinate_leakage':False,'proper_names_are_locators':True,'numeric_strings_may_localize':True,'scene_only_sufficient':False if s['question_type']=='ordering' else None,'option_lengths':[len(o[0]) for o in definition]},'author_final_QA_claim':False}
        if s['question_type']=='multi_select':
            for l,o in zip('ABCDEF',definition): private['evidence_by_label'][l]={'source_verdict':'SUPPORTED' if o[1] else 'REFUTED','evidence_role':'SUPPORT' if o[1] else 'REFUTE','evidence':[ev(r) for r in o[2].split()],'reason':o[3]}
            private['target_gold_count']=s['target_gold_count']; private['designcriteria']={
              'criteria':['B1 matched lexical content with changed speaker binding','B2 direct source report versus swapped actual speaker','B3 speaker must be checked'] if s['local_slot']==1 else ['S1 explicit same-matter correction/replacement','S2 old/effective value or revision scope contrast','S3 full cutoff/modal check'],
              'author_proposal':'Target realization proposed; independent reconstruction/QA required.',
              'cross_turn_necessary_claim':False,'ordinary_negations_are_not_conditional_joint':True}
        else:
            private['all_mentions']={};private['selected_turn_ref']={};private['event_semantic_key']={}
            for l,o in zip('ABCDEF',definition):
                selected=ref(o[1]); mentions=[(selected,True,'First explicit complete report matching actor/object/modality.')]+list(o[2])
                seen=set(); normalized=[]
                for mr,inc,reason in mentions:
                    mr=ref(mr)
                    if mr in seen:continue
                    seen.add(mr);normalized.append({**ev(mr),'included_for_event_key':inc,'inclusion_or_exclusion_reason':reason})
                normalized.sort(key=lambda x:x['global_turn_index_zero_based'])
                included=[m for m in normalized if m['included_for_event_key']]
                assert included[0]['turn_ref']==selected,(root,l,selected,included)
                private['all_mentions'][l]=normalized;private['selected_turn_ref'][l]=selected
                private['event_semantic_key'][l]=rid+'|'+o[0]
                private['evidence_by_label'][l]={'source_verdict':'SUPPORTED','evidence_role':'OCCURRENCE','evidence':[ev(selected)],'reason':'First full-history report; related mentions and modality exclusions are itemized.'}
            private['ordering_profile']=p['profile'];private['actual_scene_event_counts']=p['actual_scene_event_counts']
            private['all15pairs']=[]
            for a,b in itertools.combinations('ABCDEF',2):
                ia=turnmap[private['selected_turn_ref'][a]][0];ib=turnmap[private['selected_turn_ref'][b]][0]
                private['all15pairs'].append({'left':a,'right':b,'earlier':a if ia<ib else b,'later':b if ia<ib else a,'left_index':ia,'right_index':ib,'same_scene':ev(private['selected_turn_ref'][a])['scene_ref']==ev(private['selected_turn_ref'][b])['scene_ref'],'unique':ia!=ib})
            private['designcriteria']={'exactly_six_atomic_report_events':True,'full_history_semantic_search_completed_by_author':True,'all_reasonable_mentions_included_or_excluded_with_reason':True,'profile_from_actual_selected_scenes':True,'15_pairs_explicit':True,'source_positions_private':True,'target_claim':'Cross-scene appearance ordering; no claim within-scene weakness is diagnosed.','independent_QA':'NOT_STARTED'}
        save(target/'public'/filename,public)
        private['public_binding']=bind(target/'public'/filename)
        save(target/'private'/filename,private)
    initial=sum(d is not None and s['slot_id'] not in generation_holds for s,d in zip(slots,defs));shortfalls=sum(d is None for d in defs)
    by_type=dict(Counter(s['question_type'] for s,d in zip(slots,defs) if d is not None and s['slot_id'] not in generation_holds))
    save(target/'AUTHOR_COMPLETION.json',{'root_id':rid,'arm':'B','completed_at_utc':now(),'planned_slots':6,'initial_candidates':initial,'shortfalls':shortfalls,'generation_holds':len(generation_holds),'by_type':by_type,'independent_qa_status':'NOT_STARTED','artifact_bindings':[bind(p) for p in sorted(target.rglob('*.json'))]})
    print(rid,f'{initial} INITIAL / {shortfalls} SHORTFALL / {len(generation_holds)} GENERATION HOLD',flush=True)
