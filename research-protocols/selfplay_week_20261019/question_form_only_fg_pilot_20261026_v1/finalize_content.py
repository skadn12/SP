#!/usr/bin/env python3
"""Preserve the single revisions and build Gold/arm-blind re-review packets."""
from pathlib import Path
import argparse
import collections
import copy
from datetime import datetime, timezone
import json
import sys
sys.dont_write_bytecode = True
from assemble_review import HERE, read, save, binding, fingerprint, gate

REQUIRED = ('source_support', 'gold_truth', 'option_distinctness', 'distractor_relation',
            'natural_concrete_task', 'source_dependence', 'r03_necessity', 'operation_alignment')

def verify_freezes():
    for name in ('AUTHORING_INPUT_FREEZE.json','LEGACY_INPUT_FREEZE.json','DRAFT_FREEZE.json'):
        for b in read(HERE/name)['bindings']:
            assert binding(b['path']) == b, b['path']
    for group in (1,2):
        folder=HERE/f'review/group{group}'
        assert read(folder/'INDEPENDENT_KEY_FREEZE.json')['binding'] == binding(folder/'INDEPENDENT_KEY_INITIAL.json')

def pack():
    verify_freezes()
    initial={c['candidate_id']:c for c in read(HERE/'CANDIDATES_INITIAL_PRIVATE.json')}
    candidates=[]; changed=[]; inputs=[]; metadata=[]
    immutable=('candidate_id','arm','slot_id','root_id','author_id','question_type','task_operation',
               'gold','source_binding','source_status','warnings','preserved_plan_candidate_id',
               'preserved_source_relations','feedback_target')
    for arm in ('F','G'):
        folder=HERE/f'question_authoring/{arm}'
        p=folder/'CANDIDATES_REVISION1.json'
        rows=read(p); assert isinstance(rows,list) and len(rows)==20
        targets={r['candidate_id'] for r in read(folder/'REVISION1_INPUT.json')['issues']}
        expected={cid for cid,c in initial.items() if c['arm']==arm}
        assert {c['candidate_id'] for c in rows}==expected
        actual={c['candidate_id'] for c in rows if c != initial[c['candidate_id']]}
        assert actual==targets,(arm,actual,targets)
        for c in rows:
            old=initial[c['candidate_id']]
            for key in immutable:
                assert c.get(key)==old.get(key),(c['candidate_id'],key)
            for label in 'ABCDEF':
                assert c['item_analysis'][label]['should_select']==old['item_analysis'][label]['should_select']
            candidates.append(c)
        changed.extend(sorted(actual))
        inputs.extend(binding(folder/name) for name in ('CANDIDATES_REVISION1.json','REVISION1_CHANGELOG.json','REVISION1_INPUT.json'))
        log=read(folder/'REVISION1_CHANGELOG.json')
        for key in ('initial_draft_binding','initial_file_binding','revision_binding','revised_file_binding'):
            if key in log and log[key] != binding(log[key]['path']):
                metadata.append({'arm':arm,'field':key,'recorded':log[key],'actual':binding(log[key]['path']),
                                 'disposition':'Preserved author metadata; actual byte binding used.'})
    assert len(changed)==12 and len(candidates)==40
    candidates.sort(key=lambda c:(c['root_id'],c['slot_id'],c['arm']))
    module=gate();audits=[module.audit_candidate(c,HERE/'SOURCE_SELECTION_FINAL.json') for c in candidates]
    assert all(r['mechanical_pass'] for r in audits),audits
    save(HERE/'CANDIDATES_PRIVATE.json',candidates)
    save(HERE/'MECHANICAL_FINAL.json',{'records':audits,'passed':40,'total':40,'semantic_proof':False})
    mapping=copy.deepcopy(read(HERE/'review/BLIND_MAP_PRIVATE.json'))
    by_id={c['candidate_id']:c for c in candidates}
    for row in mapping.values():
        row['candidate_sha256']=fingerprint(by_id[row['candidate_id']])
    save(HERE/'review/BLIND_MAP_REVISION1_PRIVATE.json',mapping)
    for group in (1,2):
        folder=HERE/f'review/group{group}'
        old=read(folder/'BLIND_QUESTIONS.json');questions=[]
        for row in old['questions']:
            cid=mapping[row['review_id']]['candidate_id']
            if cid not in changed: continue
            c=by_id[cid]
            questions.append({'review_id':row['review_id'], **{k:c[k] for k in
                ('root_id','question_type','task_operation','question','options','source_binding')}})
        save(folder/'BLIND_QUESTIONS_REVISION1.json',{'questions':questions,
             'source_warnings':old['source_warnings'],
             'review_policy':'Gold/arm-blind single revision re-review. Independently reconstruct all six decisions again. No further author revision is available; retain unresolved HOLD/FAIL.'})
    save(HERE/'REVISION1_FREEZE.json',{'bindings':inputs+[binding(HERE/'CANDIDATES_PRIVATE.json'),
        binding(HERE/'MECHANICAL_FINAL.json'),binding(HERE/'review/BLIND_MAP_REVISION1_PRIVATE.json')]+
        [binding(HERE/f'review/group{g}/BLIND_QUESTIONS_REVISION1.json') for g in (1,2)],
        'changed_candidate_ids':changed,'unchanged_count':28,'metadata_discrepancies':metadata,
        'before_solver':True,'further_author_revisions_allowed':0})
    print(json.dumps({'candidates':40,'changed':12,'unchanged':28,'mechanical_pass':40,
         'metadata_discrepancies':len(metadata)},ensure_ascii=False))

def freeze():
    verify_freezes()
    revision_freeze=read(HERE/'REVISION1_FREEZE.json')
    for b in revision_freeze['bindings']:
        assert binding(b['path'])==b,b['path']
    candidates=read(HERE/'CANDIDATES_PRIVATE.json')
    by_id={c['candidate_id']:c for c in candidates}
    changed=set(revision_freeze['changed_candidate_ids'])
    mapping=read(HERE/'review/BLIND_MAP_REVISION1_PRIVATE.json')
    reviews={r['candidate_id']:copy.deepcopy(r) for r in read(HERE/'CONTENT_REVIEW_INITIAL_AI.json')['reviews']}
    module=gate(); inputs=[]; seen=set()
    for group in (1,2):
        folder=HERE/f'review/group{group}';key_path=folder/'INDEPENDENT_KEY_REVISION1.json'
        kf=read(folder/'INDEPENDENT_KEY_REVISION1_FREEZE.json')
        assert kf['binding']==binding(key_path) and kf['frozen_before_author_gold'] is True
        doc=read(key_path)
        assert doc['gold_and_arm_blind'] is True and doc['reviewer_is_human'] is False
        expected={r['review_id'] for r in read(folder/'BLIND_QUESTIONS_REVISION1.json')['questions']}
        assert len(doc['reviews'])==len(expected)==6
        assert {r['review_id'] for r in doc['reviews']}==expected
        for original in doc['reviews']:
            row=copy.deepcopy(original);bid=row['review_id'];mapped=mapping[bid]
            cid=mapped['candidate_id']; c=by_id[cid]
            assert cid in changed and cid not in seen;seen.add(cid)
            assert fingerprint(c)==mapped['candidate_sha256']
            assert row['decision'] in ('PASS','HOLD','FAIL')
            assert len(row['independent_gold'])==len(set(row['independent_gold']))
            assert set(row['independent_gold'])<=set('ABCDEF') and set(row['options'])==set('ABCDEF')
            assert all(k in row['checks'] and row['checks'][k]['decision'] in ('PASS','HOLD','FAIL')
                       and row['checks'][k]['rationale'].strip() for k in REQUIRED)
            row.update(candidate_id=cid,candidate_sha256=fingerprint(c),source_binding=c['source_binding'],
                author_id=c['author_id'],reviewer_id=doc['reviewer_id'],reviewer_is_human=False,
                gold_blind=True,independent_key_binding=binding(key_path),
                original_independent_decision=row['decision'],review_phase='REVISION1',
                prior_review_binding=binding(HERE/'CONTENT_REVIEW_INITIAL_AI.json'))
            if row['decision']=='PASS' and any(row['checks'][k]['decision']!='PASS' for k in REQUIRED):
                row['decision']='HOLD'
                row['review_decision_consistency_issue']='Overall PASS contained an unresolved review dimension.'
            row['author_gold_agrees']=set(row['independent_gold'])==set(c['gold'])
            if not row['author_gold_agrees']:
                row['decision']='HOLD';row['checks']['gold_truth']={'decision':'HOLD',
                    'rationale':'Frozen independent revision key differs from author Gold; no further revision available.'}
            turns={t['turn_ref']:t for t in read(c['source_binding']['path'])['turns']}
            errors=[]
            for label,option in row['options'].items():
                if not option.get('evidence'): errors.append({'label':label,'error':'NO_INDEPENDENT_EVIDENCE'})
                for ev in option.get('evidence',[]):
                    turn=turns.get(ev.get('turn_ref'))
                    if not turn or ev.get('speaker')!=turn['speaker'] or not ev.get('quote') or ev['quote'] not in turn['text']:
                        errors.append({'label':label,'error':'INDEPENDENT_LITERAL_EVIDENCE_MISMATCH','evidence':ev})
            row['independent_literal_audit']={'pass':not errors,'errors':errors}
            if errors:
                row['decision']='HOLD';row['checks']['source_support']={'decision':'HOLD',
                    'rationale':'Independent revision evidence has a recording mismatch; no post-review repair is used.'}
            if len(c['gold'])==6:
                row['checks']['six_gold_control']=copy.deepcopy(row['checks']['distractor_relation'])
                row['checks']['six_gold_control']['rationale']+=' All-six control; J is not applicable.'
            reviews[cid]=row
        inputs += [binding(key_path),binding(folder/'INDEPENDENT_KEY_REVISION1_FREEZE.json'),
                   binding(folder/'BLIND_QUESTIONS_REVISION1.json')]
    assert seen==changed
    old_plan={r['candidate_id']:r for r in read(HERE/'PLAN_PRESERVATION_AUDIT.json')['records']}
    new_doc=read(HERE/'REVISION1_PLAN_PRESERVATION_AUDIT.json')
    new_plan={r['candidate_id']:r for r in new_doc['records']}
    assert set(new_plan)==changed
    for cid,c in by_id.items():
        row=reviews[cid];assert row['candidate_sha256']==fingerprint(c)
        plan=(new_plan if cid in changed else old_plan)[cid]
        assert plan.get('candidate_canonical_sha256',plan.get('candidate_sha256'))==fingerprint(c)
        assert plan['decision'] in ('PASS','HOLD','FAIL')
        row['plan_preservation']={'decision':plan['decision'],'audit_binding':binding(HERE/(
             'REVISION1_PLAN_PRESERVATION_AUDIT.json' if cid in changed else 'PLAN_PRESERVATION_AUDIT.json'))}
        if plan['decision']!='PASS' and row['decision']=='PASS':
            row['decision']='HOLD'
            row['plan_preservation_hold']=plan.get('rationale','Unresolved original plan/evidence-unit preservation.')
        mechanical=module.audit_candidate(c,HERE/'SOURCE_SELECTION_FINAL.json')
        if not mechanical['mechanical_pass']:
            row['decision']='HOLD';row['mechanical_hold']=mechanical['mechanical_errors']
    final=sorted(reviews.values(),key=lambda r:r['candidate_id'])
    validations=[module.validate_review(by_id[r['candidate_id']],r,HERE/'SOURCE_SELECTION_FINAL.json') for r in final]
    assert not any(r['review_errors'] for r in validations),[r for r in validations if r['review_errors']]
    save(HERE/'CONTENT_REVIEW_AI.json',{'reviews':final,'revision_review_input_bindings':inputs,
        'basis':'AI independent keys frozen before author Gold comparison; at most one content revision; no Solver scores used.'})
    counts=dict(collections.Counter(r['decision'] for r in final))
    matched=[(root,slot) for root in sorted({c['root_id'] for c in candidates}) for slot in ('MC2','MC3','MC4','MC5','MC6')
         if all(reviews[f'CL1-{arm}-{root.split("-")[-1]}-{slot}']['decision']=='PASS' for arm in ('F','G'))]
    save(HERE/'CONTENT_REVIEW_FINAL_AUDIT.json',{'records':validations,'decisions':counts,
        'individual_pass':sum(r['admitted'] for r in validations),'matched_fg_slots':len(matched),
        'execution_questions':2*len(matched),'execution_requests':8*len(matched),
        'unresolved_gold_mismatches':[r['candidate_id'] for r in final if not r['author_gold_agrees']],
        'no_more_author_revisions':True,'new_solver_calls_so_far':0})
    names=['CANDIDATES_PRIVATE.json','CONTENT_REVIEW_AI.json','SOURCE_SELECTION_FINAL.json','PROTOCOL_FREEZE.json',
           'CONTENT_REVIEW_FINAL_AUDIT.json','CONTENT_REVIEW_INITIAL_AI.json','CONTENT_REVIEW_INITIAL_AUDIT.json',
           'CANDIDATES_INITIAL_PRIVATE.json','AUTHORING_INPUT_FREEZE.json','LEGACY_INPUT_FREEZE.json',
           'DRAFT_FREEZE.json','REVISION1_FREEZE.json','MECHANICAL_INITIAL.json','MECHANICAL_FINAL.json',
           'PLAN_PRESERVATION_AUDIT.json','MC4_CONTEXT_DEPENDENCE_AUDIT.json','REVISION1_PLAN_PRESERVATION_AUDIT.json',
           'ERROR_MATCHING_RULES_ZH.md','BLIND_REVIEW_INSTRUCTIONS_ZH.md','RUNNER_CPU_VALIDATION.json','runner.py',
           'assemble_review.py','merge_initial_reviews.py','finalize_content.py',
           'question_authoring/COMMON_AUTHOR_CONTRACT_ZH.md','question_authoring/FORM_ONLY_DELTA.diff']
    bindings=[binding(HERE/n) for n in names]+inputs
    for name in ('AUTHORING_INPUT_FREEZE.json','LEGACY_INPUT_FREEZE.json','DRAFT_FREEZE.json','REVISION1_FREEZE.json'):
        bindings+=read(HERE/name)['bindings']
    for group in (1,2):
        bindings.extend(binding(HERE/f'review/group{group}/{n}') for n in
             ('BLIND_QUESTIONS.json','SOURCE_INPUT.json','INDEPENDENT_KEY_INITIAL.json','INDEPENDENT_KEY_FREEZE.json'))
    bindings += [{k:r[k] for k in ('path','sha256','bytes')} for r in read(HERE/'SOURCE_SELECTION_FINAL.json')['roots']]
    bindings=list({b['path']:b for b in bindings}.values())
    for b in bindings: assert binding(b['path'])==b,b['path']
    save(HERE/'CONTENT_FREEZE.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),
         'bindings':bindings,'frozen_before_solver':True,'decisions':counts,
         'planned_questions':40,'matched_fg_slots':len(matched),'admitted_questions':2*len(matched),
         'no_replacement_sampling':True,'remaining_content_revision_budget':0})
    print(json.dumps({'frozen':True,'decisions':counts,'matched_fg_slots':len(matched),
         'execution_questions':2*len(matched),'solver_requests':8*len(matched)},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['pack','freeze']);a=p.parse_args()
    globals()[a.action]()
