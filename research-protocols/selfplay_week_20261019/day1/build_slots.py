"""Freeze score-independent A/B question slots; this does not generate questions."""
from pathlib import Path
from collections import Counter
import hashlib
import json

BASE=Path(__file__).resolve().parent


def main():
    schedule=json.loads((BASE/'evaluation/ROOT_SELECTION.json').read_text())
    protocol=json.loads((BASE/'PROTOCOL_V1.json').read_text())
    slots=[]
    multi_index=0
    for stage, key in [(1,'wave1'),(2,'wave2')]:
        stage_slots=[]
        for root in schedule[key]['root_ids']:
            for typ in ['multi_select','ordering']:
                for local in range(1,4):
                    slot_id=f'week1-stage{stage}-{root}-{typ}-{local:02d}'
                    target_gold=[2,3,4,5][multi_index%4] if typ=='multi_select' else None
                    if typ=='multi_select':multi_index+=1
                    stage_slots.append({'slot_id':slot_id,'stage':stage,'root_id':root,'question_type':typ,
                                        'local_slot':local,'target_gold_count':target_gold,
                                        'options_or_items':6})
        ranked=sorted(stage_slots,key=lambda x:hashlib.sha256(f"{protocol['seeds']['selection']}|{stage}|{x['slot_id']}".encode()).hexdigest())
        targeted={x['slot_id'] for x in ranked[:len(ranked)*7//10]}
        for arm in ['A','B']:
            for slot in stage_slots:
                tar=arm=='B' and slot['slot_id'] in targeted
                slots.append({**slot,'arm':arm,'candidate_slot_id':arm+'-'+slot['slot_id'],
                              'generation_mode':'diagnosis_targeted' if tar else 'free_diverse',
                              'target_cause_id':None,'cause_selection_state':'WAITING_FOR_REPRODUCED_TRAIN_CAUSE' if tar else 'NOT_APPLICABLE',
                              'source_binding':None,'source_admission_state':'PENDING_EXACT_PREFIX_ADMISSION',
                              'question_generated':False})
    assert len(slots)==360
    assert len({x['candidate_slot_id'] for x in slots})==360
    summary={}
    for arm in ['A','B']:
        arr=[s for s in slots if s['arm']==arm]
        summary[arm]={'slots':len(arr),'by_stage':dict(Counter(s['stage'] for s in arr)),
                      'by_type':dict(Counter(s['question_type'] for s in arr)),
                      'gold_target_counts':dict(Counter(s['target_gold_count'] for s in arr if s['question_type']=='multi_select')),
                      'generation_modes':dict(Counter(s['generation_mode'] for s in arr))}
    for stage, expected in [(1,84),(2,42)]:
        assert sum(s['arm']=='B' and s['stage']==stage and s['generation_mode']=='diagnosis_targeted' for s in slots)==expected
    out={'status':'SLOTS_ONLY_NOT_GENERATED_DATA','selection_uses_solver_scores':False,
         'protocol_sha256':hashlib.sha256((BASE/'PROTOCOL_V1.json').read_bytes()).hexdigest(),
         'root_selection_sha256':hashlib.sha256((BASE/'evaluation/ROOT_SELECTION.json').read_bytes()).hexdigest(),
         'summary':summary,'shortfall_policy':'Unadmitted sources or missing reproduced causes remain explicit shortfalls; no score-based replacement or invented targets.',
         'slots':slots}
    with (BASE/'SLOT_PLAN.json').open('x',encoding='utf-8') as f:
        json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
