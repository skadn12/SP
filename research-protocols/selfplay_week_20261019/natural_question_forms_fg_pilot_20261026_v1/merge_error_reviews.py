#!/usr/bin/env python3
"""Merge complete independent audits of frozen F FULL final responses."""
import collections
import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLASSES = {'OBSERVED_EXPLANATION_MATCH', 'OBSERVED_SELECTION_MATCH',
           'ERROR_OTHER_RELATION', 'NO_TARGET_ERROR_OBSERVED',
           'INDETERMINATE', 'INVALID_RESPONSE'}

def read(path):
    return json.loads(path.read_text())

def bind(path):
    data = path.read_bytes()
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}

def main():
    scores = {s['request_id']: s for s in read(HERE/'execution/SCORES.json')['records']
              if s['arm'] == 'F' and s['condition'] == 'FULL'}
    candidates = {c['candidate_id']: c for c in read(HERE/'CANDIDATES_PRIVATE.json')}
    records, bindings = [], []
    for group in (1, 2):
        packet_path = HERE/f'error_review/group{group}/PACKET.json'
        review_path = HERE/f'error_review/group{group}/REVIEW.json'
        packet, review = read(packet_path), read(review_path)
        expected = {r['score_and_public_final_json']['request_id'] for r in packet['records']}
        rows = review['records']
        assert review['reviewer_id'] and review['reviewer_is_human'] is False
        assert len(rows) == len(expected) and {r['request_id'] for r in rows} == expected
        for original in rows:
            r = dict(original)
            s = scores[r['request_id']]
            c = candidates[s['candidate_id']]
            for k in ('candidate_id', 'root_id', 'slot_id', 'exact', 'J', 'valid', 'canonical_prediction'):
                assert r[k] == s[k], (r['request_id'], k)
            assert r['match_class'] in CLASSES
            assert r['predicted_observable_error'] == c['feedback_target']['predicted_observable_error']
            assert r['rationale'] and r['original_card_scope_assessment']
            assert 'public_response_evidence' in r
            r['reviewer_id'] = review['reviewer_id']
            r['independent_review_binding'] = bind(review_path)
            records.append(r)
        bindings += [bind(packet_path), bind(review_path)]
    assert len(records) == len(scores) and {r['request_id'] for r in records} == set(scores)
    pairs = []
    for cid in sorted({r['candidate_id'] for r in records}):
        subset = [r for r in records if r['candidate_id'] == cid]
        assert len(subset) == 2
        classes = [r['match_class'] for r in subset]
        pairs.append({'candidate_id': cid, 'root_id': subset[0]['root_id'],
                      'slot_id': subset[0]['slot_id'],
                      'request_ids': [r['request_id'] for r in subset],
                      'match_classes': classes,
                      'both_explanation_match': all(k == 'OBSERVED_EXPLANATION_MATCH' for k in classes),
                      'both_observable_target_match': all(k in {'OBSERVED_EXPLANATION_MATCH','OBSERVED_SELECTION_MATCH'} for k in classes),
                      'any_explanation_match': 'OBSERVED_EXPLANATION_MATCH' in classes,
                      'any_observable_target_match': any(k in {'OBSERVED_EXPLANATION_MATCH','OBSERVED_SELECTION_MATCH'} for k in classes)})
    aggregates = []
    for name, slots in [('MC2_5_PRIMARY', {'MC2','MC3','MC4','MC5'}), ('MC6_CONTROL', {'MC6'})]:
        subset = [r for r in records if r['slot_id'] in slots]
        for dimension in ('ALL', 'root_id', 'slot_id'):
            values = ['ALL'] if dimension == 'ALL' else sorted({r[dimension] for r in subset})
            for value in values:
                chosen = subset if dimension == 'ALL' else [r for r in subset if r[dimension] == value]
                ids = {r['candidate_id'] for r in chosen}
                pp = [r for r in pairs if r['candidate_id'] in ids]
                aggregates.append({'group': name, 'dimension': dimension, 'value': value,
                                   'responses': len(chosen), 'semantic_questions': len(ids),
                                   'class_counts': dict(collections.Counter(r['match_class'] for r in chosen)),
                                   'paired_question_counts': {k: sum(p[k] for p in pp) for k in
                                      ('both_explanation_match','both_observable_target_match',
                                       'any_explanation_match','any_observable_target_match')}})
    result = {'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'records': sorted(records, key=lambda r:r['request_id']),
              'aggregates': aggregates, 'paired_permutations': pairs,
              'bindings': bindings,
              'limits': 'All F FULL final public responses audited. Selection-only matches do not establish a reasoning mechanism. Two permutations are dependent measurements of the same question. Four exposed roots and no training; no unseen-root or training-improvement claim.'}
    output = HERE/'FEEDBACK_ERROR_MATCHING_AI.json'
    with output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'responses': len(records), 'aggregates': [a for a in aggregates if a['dimension'] == 'ALL']}))

if __name__ == '__main__':
    main()
