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

GOLD_ORIGIN = 'source_appearance_order_frozen_before_writer'

def gold_questions(source, anchor_plan):
    """Compute only source positions; semantic support still needs review."""
    turns = {t['turn_ref']: (i, t) for i, t in enumerate(source['turns'])}
    need(len(turns) == len(source['turns']), 'program_gold_duplicate_turns')
    result = []
    for plan in anchor_plan['questions']:
        positions = {}
        need(set(plan['anchors']) == set(LABELS), 'program_gold_anchor_labels')
        for label in LABELS:
            a = plan['anchors'][label]; index, turn = turns[a['turn_ref']]
            need(a['speaker'] == turn['speaker'] and a['quote'] and turn['text'].count(a['quote']) == 1,
                 'program_gold_anchor_not_exact')
            start = turn['text'].index(a['quote'])
            positions[label] = {'turn_ref': a['turn_ref'], 'turn_index': index,
                                'quote_start': start, 'quote_end': start + len(a['quote'])}
        gold = sorted(LABELS, key=lambda k: (positions[k]['turn_index'], positions[k]['quote_start']))
        for left, right in zip(gold, gold[1:]):
            a, b = positions[left], positions[right]
            need(a['turn_index'] < b['turn_index'] or
                 (a['turn_index'] == b['turn_index'] and a['quote_end'] <= b['quote_start']),
                 'program_gold_tied_or_overlapping_anchors')
        result.append({'candidate_id': plan['candidate_id'], 'root_id': plan['root_id'],
                       'gold': gold, 'anchor_positions': positions})
    return result

def project(raw, plan, gold_row):
    """Combine verbatim writer items with the answer fixed before the request."""
    fields = {'candidate_id', 'root_id', 'items'}
    need(isinstance(raw, dict) and set(raw) == fields, 'ordering_bound_writer_schema')
    need(raw['candidate_id'] == plan['candidate_id'] == gold_row['candidate_id']
         and raw['root_id'] == plan['root_id'] == gold_row['root_id'], 'ordering_bound_writer_identity')
    need(isinstance(raw['items'], dict) and set(raw['items']) == set(LABELS)
         and all(isinstance(t, str) and t.strip() for t in raw['items'].values()), 'ordering_bound_writer_items')
    return {'candidate_id': raw['candidate_id'], 'root_id': raw['root_id'],
            'question_type': 'ordering', 'chronology_kind': 'conversation_order',
            'question': plan['question'], 'items': deepcopy(raw['items']), 'gold': deepcopy(gold_row['gold']),
            'gold_origin': GOLD_ORIGIN,
            'item_evidence': {label: {**deepcopy(plan['anchors'][label]), 'status': 'statement',
                'reason': 'Program-bound source excerpt chosen before writer invocation; semantic support requires separate full-source review.'} for label in LABELS},
            'order_reason': 'Program-frozen source appearance order: ' + ', '.join(gold_row['gold'])}
