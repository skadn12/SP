"""Freeze source anchors before writing; never repair generated evidence or Gold."""
from copy import deepcopy
import hashlib
from pathlib import Path
from common import bind, need, read, sha

LABELS=tuple(f'I{i}' for i in range(1,7))
STEM='Order the statements, using left-to-right clause order within a turn, as they APPEAR IN THE SUPPLIED DIALOGUE within the {scene_id} scene.'

def build(config,slot,source,ids):
    path=Path(config['ordering_anchor_plan']['path'])
    need(bind(path)==config['ordering_anchor_plan'],'ordering_plan_binding_changed')
    plans=read(path)['plans']
    selected=[p for p in plans if p['root_id']==slot['root_id']]
    need(len(selected)==len(ids)==2 and sorted(p['slot_number'] for p in selected)==[1,2], 'ordering_plan_slot_coverage')
    turns={t['turn_ref']:t for t in source['turns']}
    rows=[]
    for qid,p in zip(ids,sorted(selected,key=lambda p:p['slot_number'])):
        anchors=p['anchors'];scene=p['scene_id']
        need(isinstance(scene,str) and len(anchors)==6 and len({a['turn_ref'] for a in anchors})==6,'ordering_plan_six_distinct_turns')
        for a in anchors:
            turn=turns.get(a['turn_ref'])
            need(turn is not None and a['turn_ref'].rsplit(':',1)[0]==scene
                 and a['speaker']==turn['speaker'] and isinstance(a['quote'],str)
                 and a['quote'].strip() and turn['text'].count(a['quote'])==1,
                 'ordering_plan_anchor_not_exact_single_scene')
        # Stable label assignment independent of writer Gold and Solver scores.
        perm=sorted(anchors,key=lambda a:hashlib.sha256((qid+'\0'+a['turn_ref']+'\0'+a['quote']).encode()).hexdigest())
        rows.append({'candidate_id':qid,'root_id':slot['root_id'],'scene_id':scene,
                     'question':STEM.format(scene_id=scene),
                     'anchors':{label:{k:a[k] for k in ('turn_ref','speaker','quote')} for label,a in zip(LABELS,perm)}})
    return {'version':'v81_fixed_single_scene_anchors_v1','plan_source':bind(path),
            'root_id':slot['root_id'],'source_sha256':slot['source_sha256'],'questions':rows,
            'semantic_certification':False,'gold_supplied_to_writer':False}

def project(raw,plan):
    """Construct the row from fixed anchors and verbatim writer items and Gold."""
    fields={'candidate_id','root_id','items','gold','order_reason'}
    need(isinstance(raw,dict) and set(raw)==fields,'ordering_bound_writer_schema')
    need(raw['candidate_id']==plan['candidate_id'] and raw['root_id']==plan['root_id'], 'ordering_bound_writer_identity')
    need(isinstance(raw['items'],dict) and set(raw['items'])==set(LABELS)
         and all(isinstance(t,str) and t.strip() for t in raw['items'].values()), 'ordering_bound_writer_items')
    need(isinstance(raw['gold'],list) and len(raw['gold'])==6
         and all(isinstance(x,str) for x in raw['gold']) and set(raw['gold'])==set(LABELS), 'ordering_bound_writer_gold')
    need(isinstance(raw['order_reason'],str) and raw['order_reason'].strip(),'ordering_bound_writer_reason')
    return {'candidate_id':raw['candidate_id'],'root_id':raw['root_id'],
            'question_type':'ordering','chronology_kind':'conversation_order',
            'question':plan['question'],'items':deepcopy(raw['items']),'gold':deepcopy(raw['gold']),
            'item_evidence':{label:{**deepcopy(plan['anchors'][label]),'status':'statement',
                'reason':'Program-bound source excerpt chosen before writer invocation; semantic support requires separate full-source review.'} for label in LABELS},
            'order_reason':raw['order_reason']}
