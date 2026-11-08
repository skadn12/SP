"""Normalize immutable drafts plus signed QA into the original strict schema.

No semantic review is performed here. Missing reviews remain QA_PENDING;
acceptance also requires a separate deduplication decision. Exclusive writes
preserve the first normalized version.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib,json,sys
import jsonschema

D=Path(__file__).resolve().parent
P=D.parent/'day3_production_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def binding(p):return {'path':str(p),'sha256':sha(p)}
def dump(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
freeze=read(D/'EXECUTION_FREEZE.json')
slots={s['candidate_slot_id']:s for s in freeze['slots']}
library=P/'design/PUBLIC_DESIGN_LIBRARY.json'
validator=jsonschema.Draft202012Validator(read(library)['schemas']['candidate_record'],
                                         format_checker=jsonschema.FormatChecker())

effective_path=D/'qa/EFFECTIVE_REVIEW_INDEX.json'
effective=read(effective_path)
reviews={cid:(row,{'reviewer':row['reviewer'],'created_at_utc':row['reviewed_at']},
               Path(row['comparison_binding']['path'])) for cid,row in effective['rows'].items()}
dedup_path=D/'dedup/DECISIONS.json'
dedup={r['candidate_id']:r for r in read(dedup_path)['rows']} if dedup_path.exists() else {}
overrides_path=effective_path
overrides=effective['occurrence_overrides']
written=0
for public_path in sorted((D/'candidates').glob('*/*/public/*.json')):
    pub=read(public_path);cid=pub['candidate_id'];slot=slots[cid]
    dest=D/'records'/(cid+'.json')
    if dest.exists() and '--dry-run' not in sys.argv:
        raise RuntimeError('Normalized record already exists; use a new version after QA/dedup changes: '+str(dest))
    private_path=public_path.parent.parent/'private'/public_path.name
    a=read(private_path)
    source_path=Path(slot['source_binding']['path'])
    src=read(source_path);turns={t['turn_ref']:(i,t) for i,t in enumerate(src['turns'])}
    assert sha(source_path)==slot['source_binding']['sha256']
    root=slot['root_id'];arm=slot['arm'];typ=slot['question_type']
    author_dir=public_path.parent.parent
    manifest_path=author_dir/'READ_MANIFEST.json'
    plan_path=author_dir/'PLAN.json'
    manifest=read(manifest_path);plan=read(plan_path)
    sr_path=D/'source_reviews'/f'{root}.json';sr=read(sr_path)
    assert sr['status'] in ['PASS','PASS_FOR_SCOPED_AUTHORING']
    review_entry=reviews.get(cid)
    qa,c,cp=review_entry if review_entry else ({}, {}, None)
    reconstructed=None;rr=None
    if qa:
        rp=Path(qa['independent_reconstruction']['path'])
        assert sha(rp)==qa['independent_reconstruction']['sha256']
        reconstructed=read(rp)
        rr=next(x for x in reconstructed['rows'] if x['candidate_id']==cid)
    refs=[];refmap={}
    def ref(turn,role):
        key=(turn,role)
        if key not in refmap:
            i,t=turns[turn];ref_id=f'E{len(refs)+1:03}'
            refs.append({'ref_id':ref_id,'scene_id':t.get('scene_ref') or t['session_ref'],
                         'turn_id':turn,'source_position':{'global_turn_index':i,'span_start':0,'span_end':len(t['text'])},
                         'quote':t['text'],'semantic_role':role})
            refmap[key]=ref_id
        return refmap[key]
    options=[]
    ev=a['evidence_by_label']
    for label,text in pub['options_or_items'].items():
        e=ev[label]
        verdict=e.get('verdict',e.get('source_verdict'))
        q=rr.get('evidence_by_label',{}).get(label) if rr else None
        # Reviewer reconstruction is authoritative, but cannot change draft gold.
        if q:verdict=q.get('verdict',q.get('status'))
        assert verdict in ['SUPPORTED','REFUTED','UNSUPPORTED','AMBIGUOUS'],(cid,label,verdict)
        support=[];refute=[]
        role='OCCURRENCE' if typ=='ordering' else ('SUPPORT' if verdict=='SUPPORTED' else 'REFUTE')
        for evidence in e['evidence']:
            r=ref(evidence['turn_ref'],role)
            if verdict=='SUPPORTED':support.append(r)
            else:refute.append(r)
        if q:
            for turn in q['turn_refs']:
                r=ref(turn,role)
                if verdict=='SUPPORTED' and r not in support:support.append(r)
                if verdict!='SUPPORTED' and r not in refute:refute.append(r)
        options.append({'option_id':label,'relation_key':f'{root}|{text}',
                        'source_verdict':verdict,'support_ref_ids':support,'refute_ref_ids':refute,
                        'distractor_mechanism':None if verdict=='SUPPORTED' else 'EXPLICIT_SOURCE_CONTRAST',
                        'truth_rationale':(q or {}).get('rationale',e.get('rationale',e.get('reason','')))})
    revision=a.get('design_realization',{}).get('revision_edge',[])
    if typ=='multi_select' and slot['design_id']=='MC_FINAL_STATE_REVISION' and not revision and rr:
        revision=rr.get('revision_chain_turn_refs',[])
    rev_refs=[ref(t,'REVISION') for t in revision]
    target_pass=qa.get('target_status')=='PASS'
    content_pass=qa.get('content_status')=='PASS'
    rationale=qa.get('independent_semantic_rationale','Pending independent public+full-source review.')
    shortcut=qa.get('shortcut_review','Pending independent review.')
    if not isinstance(shortcut,str):shortcut=json.dumps(shortcut,ensure_ascii=False,sort_keys=True)
    target={'primary_relation_id':slot['design_id'] if typ=='multi_select' else None,
            'ordering_profile_id':slot['design_id'] if typ=='ordering' else None,
            'evidence_structure_tags':['EXPLICIT_REVISION_CHAIN'] if revision else [],
            'criteria':[{'criterion_id':'SIGNED_INDEPENDENT_TARGET_REALIZATION',
                         'result':'PASS' if target_pass else ('FAIL' if qa else 'UNKNOWN'),
                         'evidence_ref_ids':[],'rationale':rationale}],
            'focal_option_ids':[],'premise_groups':[],'minimal_sufficient_ref_sets':[],
            'single_turn_sufficiency_sweep':{'full_scope_searched':bool(qa),
                'status':'NOT_APPLICABLE','sufficient_single_turn_ids':[],
                'rationale':'No cross-turn-necessity label is asserted solely from multiple evidence references.'},
            'premise_ablation':[],'target_implemented':'YES' if target_pass else ('NO' if qa else 'UNKNOWN'),
            'target_implementation_rationale':rationale}
    rels=[x['relation_id'] for x in sr['available_relations'] if x['material_status'].startswith('AVAILABLE')]
    btime=a.get('created_at_utc')
    r={'candidate_id':cid,'inherited_slot_id':slot['slot_id'],'arm':arm,'stage':1,'root_id':root,
       'question_type':typ,'design_assignment_frozen_at':freeze['created_at_utc'],'public_library_sha256':sha(library),
       'source_binding':{'source_path':str(source_path),'source_sha256':sha(source_path),'source_version':'FROZEN_FULL_PARENT_R01_R02',
          'scope_description':'Entire supplied parent+R01+R02 source; first semantic report ordering.',
          'source_scope_sha256':sha(source_path),'train_admission_record':str(sr_path),'suffix_policy_id':None,'full_source_visible_to_qa':True},
       'source_review':{'review_id':str(sr_path),'reviewed_at':sr['reviewed_at_utc'],'reviewer':sr['reviewer'],
          'status':'PASS','source_integrity':'PASS','entities_and_state_consistency':'PASS',
          'available_relation_material':rels,'available_ordering_material':['ORDER_3_2_1','ORDER_2_2_2','ORDER_4_2'],
          'shortfall_codes':['NO_VERIFIED_JOINT_RELATION'] if 'MC_NEGATION_CONDITIONAL_JOINT' not in rels else [],
          'rationale':'Scoped source admission; source restrictions remain binding. See immutable full-history review.'},
       'budget':{'initial_allowed':1,'content_clarification_allowed':1,'initial_used':1,'content_clarification_used':0,
          'requested_output_token_cap':3072,'actual_cap_support':'UNKNOWN','actual_cap_evidence':None,'solver_calls':0},
       'generation_history':[{'attempt_id':cid+'-INITIAL','kind':'INITIAL','request_id':None,'started_at':btime,
          'source_review_id':str(sr_path),'requested_output_token_cap':3072,'actual_cap_support':'UNKNOWN',
          'actual_output_tokens':None,'request_input_sha256':sha(plan_path),'raw_response_path':str(public_path),
          'candidate_content_sha256':sha(public_path)}],
       'fact_family':{'family_id':root+'|PENDING_SEMANTIC_DEDUP','atomic_relation_keys':[root+'|'+slot['design_id']],
          'related_candidate_ids':[],'reuse_class':'OVERLAPPING_FACTS_DIFFERENT_RELATION','duplicate_review':'UNKNOWN'},
       'lexical_shortcuts':{'direct_quote':False,'proper_name_or_id_locator':True,'numeric_or_date_order_cue':False,
          'option_lengths':[len(x) for x in pub['options_or_items'].values()],
          'correctness_correlated_style':'UNKNOWN','scene_only_sufficient':False if typ=='ordering' else None,
          'checks':[{'criterion_id':'SIGNED_SHORTCUT_REVIEW','result':'PASS' if qa else 'UNKNOWN','evidence_ref_ids':[],
                     'rationale':shortcut}],
          'rationale':shortcut},
       'target_realization':target,
       'qa_review':{'review_id':str(cp) if cp else 'PENDING','reviewer':c.get('reviewer','PENDING'),
          'reviewed_at':c.get('created_at_utc',datetime.now(timezone.utc).isoformat()),'phase':'INITIAL' if qa else 'NOT_STARTED',
          'source_review_id':str(sr_path),'source_hash_matches':'PASS',
          'all_option_or_pair_truth_checked':'PASS' if content_pass else ('FAIL' if qa else 'UNKNOWN'),
          'relation_target_checked':'PASS' if target_pass else ('FAIL' if qa else 'UNKNOWN'),
          'shortcut_checked':'PASS' if qa else 'UNKNOWN',
          'gold_independently_reconstructed':'PASS' if qa.get('gold_agreement') else ('FAIL' if qa else 'UNKNOWN'),
          'status':'PASS' if content_pass else ('HOLD' if qa else 'NOT_STARTED'),
          'criterion_results':[{'criterion_id':'SIGNED_INDEPENDENT_CONTENT_QA','result':'PASS' if content_pass else ('FAIL' if qa else 'UNKNOWN'),
                              'evidence_ref_ids':[],'rationale':rationale}],
          'rationale':rationale},
       'disposition':{'state':'QA_PENDING' if not qa else 'HOLD','reason_codes':['PENDING_DUPLICATE_REVIEW'] if content_pass else ['PENDING_OR_FAILED_INDEPENDENT_CONTENT_QA'],
          'content_clarification_request':None,'accepted_content_sha256':None,
          'feedback_binding_location':str(author_dir/'READ_MANIFEST.json') if arm=='B' else None}}
    if typ=='multi_select':
        r['multi_select']={'target_gold_count_inherited':slot['target_gold_count'],'options':options,'gold_option_ids':a['gold'],
            'question_cutoff':'End of full supplied source unless the public stem specifies an earlier same-matter cutoff.',
            'selection_semantics':'SELECT_SOURCE_SUPPORTED_STATEMENTS_WITH_EXPLICITLY_REFUTED_DISTRACTORS',
                'gold_count_verified':'PASS' if len([x for x in options if x['source_verdict']=='SUPPORTED'])==slot['target_gold_count'] and qa.get('gold_agreement') else 'FAIL'}
    else:
        items=[]
        author_items=a.get('items',{})
        for label in 'ABCDEF':
            ai=author_items.get(label,{}) if isinstance(author_items,dict) else next(x for x in author_items if x['item_id']==label)
            selected=ai.get('selected_turn_ref',a.get('selected_turn_ref',{}).get(label))
            occurrences=ai.get('all_mentions',a.get('all_mentions',{}).get(label,[]))
            override=overrides.get(cid,{}).get('occurrences_by_label',{}).get(label)
            if override is not None:occurrences=override
            result=[]
            for o in occurrences:
                inc=o['included_for_event_key'];oref=ref(o['turn_ref'],'OCCURRENCE' if inc else 'EXCLUDED_OCCURRENCE')
                result.append({'ref_id':oref,'included_for_event_key':inc,
                    'exclusion_reason':None if inc else o.get('rationale',o.get('inclusion_or_exclusion_reason',o.get('reason','Related but different event/modal/report.'))),
                    'reviewer_verified':bool(qa)})
            selref=ref(selected,'OCCURRENCE')
            key=effective['canonical_event_keys'].get(cid,{}).get(label,ai.get('event_semantic_key',a.get('event_semantic_key',{}).get(label,root+'|'+pub['options_or_items'][label])))
            items.append({'item_id':label,'event_semantic_key':key,'atomicity':'PASS' if qa else 'UNKNOWN',
                'mention_policy':'FIRST_IN_DECLARED_SCOPE','requested_ordinal':None,'all_occurrences':result,'selected_ref_id':selref,
                'repeat_status':'REPEATED_SAME_FACT' if sum(x['included_for_event_key'] for x in result)>1 else 'SINGLE',
                'selection_rationale':('Earliest semantic report in full source; independently reconstructed.' if content_pass else 'Original author anchor retained in held draft; see effective independent QA and its source-first/addendum evidence.')})
        profile={'ORDER_3_2_1':[3,2,1],'ORDER_2_2_2':[2,2,2],'ORDER_4_2':[4,2]}[slot['design_id']]
        r['ordering']={'gold_definition':'SOURCE_APPEARANCE_ORDER','scope_predeclared':'Full parent+R01+R02 source',
            'scene_event_counts':profile,'within_scene_pair_count':sum(n*(n-1)//2 for n in profile),'items':items,
            'gold_item_ids':a['gold'],'unique_order_verified':'PASS' if content_pass else ('FAIL' if qa else 'UNKNOWN'),
            'all_reasonable_mentions_searched':'PASS' if qa else 'UNKNOWN'}
    sceneids=list(dict.fromkeys(e['scene_id'] for e in refs))
    r['evidence_span']={'turn_ids':list(dict.fromkeys(e['turn_id'] for e in refs)),'scene_ids':sceneids,
        'single_or_multi_turn':'MULTI_TURN' if len(set(e['turn_id'] for e in refs))>1 else 'SINGLE_TURN',
        'same_or_cross_scene':'CROSS_SCENE' if len(sceneids)>1 else 'SAME_SCENE',
        'revision_chain_ref_ids':rev_refs,'evidence_refs':refs}
    if cid in dedup:
        d=dedup[cid]
        r['fact_family']={k:d[k] for k in ['family_id','atomic_relation_keys','related_candidate_ids','reuse_class','duplicate_review']}
        if content_pass and target_pass and d['duplicate_review']=='PASS' and not d.get('within_arm_duplicate',False):
            r['disposition'].update(state='ACCEPT_DIRECT',accepted_content_sha256=sha(public_path),
                                    reason_codes=['SIGNED_SOURCE_CONTENT_TARGET_PASS','SIGNED_DUPLICATE_REVIEW_PASS'])
        elif d['duplicate_review']!='PASS' or d.get('within_arm_duplicate',False):
            r['disposition']['reason_codes']=['FAILED_OR_UNRESOLVED_DUPLICATE_REVIEW']
    if qa and not content_pass:
        reasons=['INDEPENDENT_CONTENT_HOLD']
        if not qa.get('gold_agreement'):reasons.append('INDEPENDENT_GOLD_DISAGREEMENT')
        if qa.get('anchor_agreement') is False:reasons.append('FIRST_OCCURRENCE_DISAGREEMENT')
        r['disposition']['reason_codes']=reasons
    errs=list(validator.iter_errors(r))
    if errs:raise RuntimeError(f'{cid}: '+ '; '.join(f'{list(e.path)} {e.message}' for e in errs))
    provenance={'public':binding(public_path),'private':binding(private_path),
          'read_manifest':binding(manifest_path),'plan':binding(plan_path),'source_review':binding(sr_path),
          'independent_comparison':binding(cp) if cp else None,'dedup':binding(dedup_path) if cid in dedup else None,
          'effective_review_index':binding(effective_path),
          'signed_review_and_annotation_sources':effective['provenance'][cid],
          'annotation_overrides':binding(overrides_path) if cid in overrides else None,
          'source_content_gold_and_anchors_modified':False}
    if '--dry-run' not in sys.argv:
        dump(dest,r)
        dump(D/'record_provenance'/(cid+'.json'),provenance)
    written+=1
print(json.dumps({'dry_run':'--dry-run' in sys.argv,'validated':written,'total_written':len(list((D/'records').glob('*.json')))}))
