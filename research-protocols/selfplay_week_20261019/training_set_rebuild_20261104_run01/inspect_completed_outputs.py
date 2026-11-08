#!/usr/bin/env python3
"""Print bounded private output evidence for a mainline completed-batch review."""
import argparse
import json
from pathlib import Path

R = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('--index', type=int, required=True)
a = p.parse_args()
read = lambda path: json.loads(path.read_text())
hold_file = R / f'source_preparation/BATCH{a.index:02d}_SOURCE_HOLD_APPLICATION.json'
holds = set(read(hold_file)['source_HOLD_candidate_ids']) if hold_file.exists() else set()
for b in sorted((R / 'new_evaluation').glob(f'batch_{a.index:02d}*')):
    s = b / 'solver'
    if not (s / 'COMPLETED.json').exists():
        continue
    scores = read(s / 'SCORES.json')
    raw = {x['request_id']: x for x in read(s / 'RAW_ANSWERS.json')}
    items = {x['candidate_id']: x for x in read(b / 'ITEMS_PRIVATE.json')}
    for qid, modes in scores['items'].items():
        full = modes['full_context']
        fields = ('planned_calls', 'valid_calls', 'exact_correct_calls', 'mean_J', 'mean_directed_pair_f1')
        print(json.dumps({'candidate_id': qid, 'type': items[qid]['question_type'],
            'source_hold_excluded_from_feedback': qid in holds,
            'statistics': {mode: {k: v for k, v in x.items() if k in fields} for mode, x in modes.items()}}))
        if qid in holds:
            continue
        error = next((x for x in scores['requests'] if x['candidate_id'] == qid
            and x['context_mode'] == 'full_context' and x['valid'] and not x['exact_correct']), None)
        if error is None:
            continue
        obj = json.loads(raw[error['request_id']]['raw_prediction'])
        print(json.dumps({'request_id': error['request_id'], 'Gold_stable_ids': items[qid]['gold_option_ids'],
            'predicted_stable_ids': error.get('predicted_option_ids'),
            'answer_reason': obj.get('answer_reason'),
            'judgments': {k: v.get('judgment') for k, v in obj.get('item_evidence', {}).items()},
            'final_answer': obj.get('final_answer')}, ensure_ascii=False))
