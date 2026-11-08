"""Export only independently delta-reviewed, old-operation-approved repairs."""
import importlib.util
from pathlib import Path
from repair import BASE, PARENT, SLOTS, read, save, bind, now

GATE=BASE.parents[2]/'scripts/current_method/question_presentation_gate_v2.py'
spec=importlib.util.spec_from_file_location('qpr1_shared_public_gate',GATE)
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)

def main():
    review_path=BASE/'PRESENTATION_DELTA_REVIEW.json'; review=read(review_path)
    assert review['status']=='PASS_DELTA_REVIEW_ONLY'
    assert review['independent_of_original_question_authors'] and review['independent_of_presentation_editor']
    rows=read(BASE/'TRANSFORM_RECEIPT.json')['records']
    by={r['candidate_id']:r for r in review['questions']}
    assert len(rows)==len(by)==56 and set(by)=={r['candidate_id'] for r in rows}
    public=[]; private=[]; decisions=[]; seen=set()
    for row in rows:
        q=read(row['repaired']['path']); assert bind(row['repaired']['path'])==row['repaired']
        rr=by[q['candidate_id']]
        assert rr['question_binding']==row['repaired'] and rr['parent_question_binding']==row['original']
        assert rr['reviewer_id'] not in [rr['original_author_id'],rr['presentation_editor_id']]
        assert rr['gold_evidence_source_display_unchanged'] is True
        status=rr['decision']
        assert status in ['ADMIT','FAIL']
        if row['carried_operation_status']!='PASS': assert status=='FAIL'
        if status=='ADMIT':
            assert rr['carried_actual_operation']=='PASS'
            labels=list('ABCDEF' if q['question_type']=='multi_select' else 'ABCD')
            canonical=q['options'] if q['question_type']=='multi_select' else q['items']
            mp=dict(zip(labels,q['display_order'])); reverse={v:k for k,v in mp.items()}
            operation=gate.public_operation(q['task_operation'])
            gate.require_public_selection(q['question'],operation,q['question_type'])
            display={k:canonical[v] for k,v in mp.items()}
            identity=(q['source_binding']['sha256'],q['question'],tuple(display.items()))
            if identity in seen: status='EXACT_WHOLE_DUPLICATE'
            else: seen.add(identity)
        decisions.append({k:row[k] for k in ['root_id','condition','slot_id','candidate_id','parent_candidate_id']}|{'status':status,'question_binding':row['repaired'],'review_binding':bind(review_path)})
        if status!='ADMIT': continue
        public.append({k:q[k] for k in ['candidate_id','root_id','condition','question_type','question','source_binding']}|{'task_operation':operation,'display':display})
        private.append({'candidate_id':q['candidate_id'],'parent_candidate_id':row['parent_candidate_id'],'slot_id':q['slot_id'],
            'private_task_operation':q['task_operation'],'question_type':q['question_type'],'canonical_gold':q['gold'],
            'display_gold':[reverse[k] for k in q['gold']],'display_to_canonical':mp,'question_binding':row['repaired']})
    save(BASE/'QUESTION_FREEZE.json',{'status':'FROZEN_BEFORE_SOLVER','created_at_utc':now(),'planned_slots':56,
        'admitted_questions':len(public),'decisions':decisions,'semantic_review_scope':'Existing full-source/key/operation reviews carried; new independent presentation delta review only',
        'solver_scores_read_for_admission':False,'new_authoring_stability_claim':False})
    save(BASE/'inputs/PUBLIC_QUESTIONS.json',{'questions':public})
    save(BASE/'inputs/PRIVATE_SCORING.json',{'questions':private})
    save(BASE/'inputs/EXPORT_FREEZE.json',{'status':'FROZEN_BEFORE_SOLVER','created_at_utc':now(),
        'public_binding':bind(BASE/'inputs/PUBLIC_QUESTIONS.json'),'private_binding':bind(BASE/'inputs/PRIVATE_SCORING.json'),
        'question_freeze_binding':bind(BASE/'QUESTION_FREEZE.json'),'planned_root_ids':read(BASE/'REPAIR_PROTOCOL.json')['roots'],
        'planned_slots':56,'content_review_status':'PASS','content_review_scope':'carried original semantic gate plus independent presentation delta gate',
        'solver_score_filtering':False,'full_source_no_truncation':True,
        'bindings':[bind(BASE/'REPAIR_PROTOCOL.json'),bind(BASE/'inputs/PUBLIC_SAMPLING_SEEDS.json'),bind(GATE),bind(__file__)]})
    print({'admitted':len(public),'failed':56-len(public),'requests':2*len(public)})

if __name__=='__main__':main()
