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
                 and a['quote'].strip() and a['quote']==turn['text'],
                 'ordering_plan_requires_complete_source_turn')
        # Stable label assignment independent of writer Gold and Solver scores.
        perm=sorted(anchors,key=lambda a:hashlib.sha256((qid+'\0'+a['turn_ref']+'\0'+a['quote']).encode()).hexdigest())
        rows.append({'candidate_id':qid,'root_id':slot['root_id'],'scene_id':scene,
                     'question':STEM.format(scene_id=scene),
                     'anchors':{label:{k:a[k] for k in ('turn_ref','speaker','quote')} for label,a in zip(LABELS,perm)}})
    return {'version':'v90_private_full_turn_anchors_v1','plan_source':bind(path),
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
            need(a['speaker'] == turn['speaker'] and a['quote'] and a['quote'] == turn['text'],
                 'program_gold_requires_complete_source_turn')
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

def writer_plan(source, anchor_plan):
    """Source-position records expose no display-label assignment or answer."""
    positions = {t['turn_ref']: i for i, t in enumerate(source['turns'])}
    questions = []
    for plan in anchor_plan['questions']:
        records = []
        for anchor in sorted(plan['anchors'].values(), key=lambda a: positions[a['turn_ref']]):
            records.append({'anchor_id': anchor['turn_ref'], 'speaker': anchor['speaker'],
                            'quote': anchor['quote'], 'statement_prefix': anchor['speaker'] + ' '})
        need(len(records) == len({r['anchor_id'] for r in records}) == 6, 'writer_plan_six_unique_ids')
        questions.append({k: deepcopy(plan[k]) for k in ('candidate_id', 'root_id', 'scene_id', 'question')})
        questions[-1]['anchor_records'] = records
    return {'version': 'v90_full_turn_source_id_predicate_writer_plan_v1', 'root_id': anchor_plan['root_id'],
            'source_sha256': anchor_plan['source_sha256'], 'questions': questions,
            'display_labels_supplied': False, 'gold_supplied_to_writer': False,
            'semantic_certification': False}


def project(raw, plan, gold_row):
    """Bind verbatim predicates by source ID to preassigned private labels."""
    need(isinstance(raw, dict) and set(raw) == {'candidate_id', 'root_id', 'records'},
         'ordering_bound_writer_schema')
    need(raw['candidate_id'] == plan['candidate_id'] == gold_row['candidate_id']
         and raw['root_id'] == plan['root_id'] == gold_row['root_id'], 'ordering_bound_writer_identity')
    records = raw['records']
    need(isinstance(records, list) and len(records) == 6
         and all(isinstance(r, dict) and set(r) == {'anchor_id', 'predicate'} for r in records),
         'ordering_writer_records_schema')
    need(all(isinstance(r['anchor_id'], str) and isinstance(r['predicate'], str)
             and r['predicate'] and r['predicate'].strip() == r['predicate'] for r in records),
         'ordering_writer_record_text')
    indexed = {r['anchor_id']: r['predicate'] for r in records}
    expected = {a['turn_ref'] for a in plan['anchors'].values()}
    need(len(indexed) == 6 and set(indexed) == expected, 'ordering_writer_anchor_id_coverage')
    items = {label: plan['anchors'][label]['speaker'] + ' ' + indexed[plan['anchors'][label]['turn_ref']]
             for label in LABELS}
    return {'candidate_id': raw['candidate_id'], 'root_id': raw['root_id'],
            'question_type': 'ordering', 'chronology_kind': 'conversation_order',
            'question': plan['question'], 'items': items, 'gold': deepcopy(gold_row['gold']),
            'gold_origin': GOLD_ORIGIN,
            'item_evidence': {label: {**deepcopy(plan['anchors'][label]), 'status': 'statement',
                'reason': 'Source-ID-bound predicate within a complete source turn frozen before generation. Independent full-source review must certify meaning, actor, state, modality and unique order.'} for label in LABELS},
            'order_reason': 'Program-frozen source appearance order: ' + ', '.join(gold_row['gold'])}
