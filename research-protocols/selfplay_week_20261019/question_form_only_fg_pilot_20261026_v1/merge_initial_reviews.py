#!/usr/bin/env python3
"""Map independently frozen blind reviews back to immutable initial drafts."""
from pathlib import Path
import copy
import json
import sys
sys.dont_write_bytecode = True
from assemble_review import HERE, read, save, binding, fingerprint, gate


def main():
    candidates = read(HERE / 'CANDIDATES_INITIAL_PRIVATE.json')
    mapping = read(HERE / 'review/BLIND_MAP_PRIVATE.json')
    by_id = {c['candidate_id']: c for c in candidates}
    assert len(by_id) == len(candidates) == 40
    for b in read(HERE / 'DRAFT_FREEZE.json')['bindings']:
        assert binding(b['path']) == b
    module = gate()
    reviews, inputs, independent_ids = [], [], set()
    for group in (1, 2):
        folder = HERE / f'review/group{group}'
        key_path = folder / 'INDEPENDENT_KEY_INITIAL.json'
        freeze = read(folder / 'INDEPENDENT_KEY_FREEZE.json')
        assert freeze['binding'] == binding(key_path)
        doc = read(key_path)
        assert doc['reviewer_is_human'] is False
        assert doc['gold_and_arm_blind'] is True
        expected = {r['review_id'] for r in read(folder / 'BLIND_QUESTIONS.json')['questions']}
        assert {r['review_id'] for r in doc['reviews']} == expected
        assert len(doc['reviews']) == len(expected) == 20
        for original in doc['reviews']:
            row = copy.deepcopy(original)
            bid = row['review_id']
            assert bid not in independent_ids
            independent_ids.add(bid)
            mapped = mapping[bid]
            candidate = by_id[mapped['candidate_id']]
            assert fingerprint(candidate) == mapped['candidate_sha256']
            assert row['decision'] in ('PASS', 'HOLD', 'FAIL')
            assert len(row['independent_gold']) == len(set(row['independent_gold']))
            assert set(row['independent_gold']) <= set('ABCDEF')
            assert set(row['options']) == set('ABCDEF')
            row.update(candidate_id=candidate['candidate_id'],
                       candidate_sha256=fingerprint(candidate),
                       source_binding=candidate['source_binding'],
                       author_id=candidate['author_id'], reviewer_id=doc['reviewer_id'],
                       reviewer_is_human=False, gold_blind=True,
                       independent_key_binding=binding(key_path),
                       original_independent_decision=row['decision'])
            required = ('source_support','gold_truth','option_distinctness','distractor_relation',
                        'natural_concrete_task','source_dependence','r03_necessity','operation_alignment')
            assert all(k in row['checks'] for k in required), bid
            assert all(row['checks'][k]['decision'] in ('PASS','HOLD','FAIL') and row['checks'][k]['rationale'].strip() for k in required), bid
            if row['decision'] == 'PASS' and any(row['checks'][k]['decision'] != 'PASS' for k in required):
                row['decision'] = 'HOLD'
                row['review_decision_consistency_issue'] = 'Overall PASS contained an unresolved independent-review dimension.'
            row['author_gold_agrees'] = set(row['independent_gold']) == set(candidate['gold'])
            source = read(candidate['source_binding']['path'])
            turns = {t['turn_ref']:t for t in source['turns']}
            independent_quote_errors = []
            for label, option_review in row['options'].items():
                evidence = option_review.get('evidence', [])
                if not evidence:
                    independent_quote_errors.append({'label':label, 'error':'NO_INDEPENDENT_EVIDENCE'})
                for ev in evidence:
                    turn = turns.get(ev.get('turn_ref'))
                    if not turn or ev.get('speaker') != turn['speaker'] or not ev.get('quote') or ev['quote'] not in turn['text']:
                        independent_quote_errors.append({'label':label, 'error':'INDEPENDENT_LITERAL_EVIDENCE_MISMATCH', 'evidence':ev})
            row['independent_literal_audit'] = {'pass':not independent_quote_errors, 'errors':independent_quote_errors}
            if independent_quote_errors:
                row['decision'] = 'HOLD'
                row['checks']['source_support'] = {'decision':'HOLD', 'rationale':
                    'Frozen independent review has literal-evidence recording errors; requires a preserved review supplement before admission.'}
            if len(candidate['gold']) == 6:
                row['checks']['six_gold_control'] = copy.deepcopy(row['checks']['distractor_relation'])
                row['checks']['six_gold_control']['rationale'] += ' Author quota is all-six control; J is not applicable.'
            if not row['author_gold_agrees']:
                row['decision'] = 'HOLD'
                row['checks']['gold_truth'] = {'decision': 'HOLD', 'rationale':
                    'Frozen independent key differs from author Gold; preserve both and resolve before admission.'}
            audit = module.audit_candidate(candidate, HERE / 'SOURCE_SELECTION_FINAL.json')
            if not audit['mechanical_pass']:
                row['decision'] = 'HOLD'
                row['mechanical_hold'] = audit['mechanical_errors']
            reviews.append(row)
        inputs += [binding(key_path), binding(folder / 'INDEPENDENT_KEY_FREEZE.json'),
                   binding(folder / 'BLIND_QUESTIONS.json')]
    assert len(reviews) == 40
    reviews.sort(key=lambda r:r['candidate_id'])
    validations = [module.validate_review(by_id[r['candidate_id']], r, HERE/'SOURCE_SELECTION_FINAL.json') for r in reviews]
    save(HERE/'CONTENT_REVIEW_INITIAL_AI.json', {'reviews':reviews, 'blind_review_input_bindings':inputs,
         'basis':'Independent AI keys frozen before comparison to author Gold; not human adjudication.'})
    save(HERE/'CONTENT_REVIEW_INITIAL_AUDIT.json', {'records':validations,
         'decisions':{d:sum(r['decision']==d for r in reviews) for d in ('PASS','HOLD','FAIL')},
         'author_gold_mismatches':[r['candidate_id'] for r in reviews if not r['author_gold_agrees']],
         'revision_budget':'At most one content-review revision; no Solver results seen.'})
    print(json.dumps({'reviews':40, 'decisions':{d:sum(r['decision']==d for r in reviews) for d in ('PASS','HOLD','FAIL')},
                      'admitted':sum(r['admitted'] for r in validations),
                      'review_record_errors':[r for r in validations if r['review_errors']]},ensure_ascii=False))


if __name__ == '__main__':
    main()
