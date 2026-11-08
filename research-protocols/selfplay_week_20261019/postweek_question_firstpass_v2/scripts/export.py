"""Freeze first-attempt admissions; never repair or select using solver scores."""
import hashlib
import json
from common import BASE, PARENT, SLOTS, bind, gate, mechanical, now, read, save


def identity(question):
    return json.dumps({
        'source': question['source_binding']['sha256'],
        'question': question['question'],
        'choices': question.get('options') or question['items'],
    }, ensure_ascii=False, sort_keys=True)


def main():
    protocol = read(BASE / 'PROTOCOL_FREEZE.json')
    for source_binding in protocol['sources']:
        assert bind(source_binding['path']) == source_binding, 'FROZEN_SOURCE_CHANGED'
    public, private, decisions, bindings, seen = [], [], [], [], set()
    historical = set()
    for path in (PARENT / 'questions').glob('*/*/*.json'):
        q = read(path)
        if 'question' in q and 'source_binding' in q:
            historical.add(identity(q))
    for root in protocol['roots']:
        for condition in protocol['conditions']:
            review_path = BASE / 'question_reviews' / root / condition / 'REVIEW.json'
            receipt_path = BASE / 'questions' / root / condition / 'AUTHOR_RECEIPT.json'
            assert receipt_path.exists(), 'FIRST_ATTEMPT_AUTHOR_RECEIPT_REQUIRED'
            receipt = read(receipt_path)
            review = read(review_path)
            assert review['independent_of_question_author'] and review['independent_of_source_author']
            assert review['full_source_read'] and review['solver_scores_read'] is False
            assert review['root_id'] == root and review['condition'] == condition
            source_path = PARENT / 'sources' / root / condition / 'FINAL_SOURCE.json'
            assert review['source_binding'] == bind(source_path)
            source_refs = [t['turn_ref'] for t in read(source_path)['turns']]
            assert review['turn_refs_read'] == source_refs
            assert receipt['root_id'] == root and receipt['condition'] == condition
            assert receipt['source_binding'] == bind(source_path)
            scope = receipt['source_read_scope']
            assert scope['all_turns_read'] and scope['turn_count'] == len(source_refs)
            author_refs = receipt.get('full_turn_refs_read', scope.get('turn_refs', scope.get('all_turn_refs_read')))
            assert author_refs == source_refs, 'AUTHOR_FULL_SOURCE_RECEIPT_INCOMPLETE'
            receipt_candidates = {row['path']: {k: row[k] for k in ('path', 'sha256', 'bytes')}
                                  for row in receipt['candidate_bindings']}
            assert len(receipt_candidates) == 7
            for row in receipt.get('input_bindings', []):
                assert bind(row['path']) == {k: row[k] for k in ('path', 'sha256', 'bytes')}
            reviewed = {row['slot_id']: row for row in review['questions']}
            assert len(review['questions']) == 7 and set(reviewed) == set(SLOTS)
            bindings.append(bind(review_path))
            bindings.append(bind(receipt_path))
            for slot in SLOTS:
                rr = reviewed[slot]
                path = BASE / 'questions' / root / condition / (slot + '.json')
                row = {'root_id': root, 'condition': condition, 'slot_id': slot,
                       'review_binding': bind(review_path), 'status': rr['decision']}
                assert row['status'] in ('ADMIT', 'FAIL', 'SHORTFALL')
                if not path.exists():
                    short = path.with_name(slot + '_SHORTFALL.json')
                    assert short.exists() and row['status'] in ('SHORTFALL', 'FAIL')
                    row['shortfall_binding'] = bind(short)
                    decisions.append(row)
                    continue
                q = read(path)
                assert rr['question_binding'] == bind(path)
                assert receipt_candidates.get(str(path.resolve())) == bind(path), 'AUTHOR_CANDIDATE_RECEIPT_DRIFT'
                assert q['author_id'] == receipt['author_id']
                assert rr['reviewer_id'] != q['author_id']
                assert q['source_binding'] == bind(source_path)
                assert q['slot_id'] == slot and q['root_id'] == root and q['condition'] == condition
                row.update(draft_candidate_id=q['candidate_id'], question_binding=bind(path))
                try:
                    mechanical(path)
                    row['mechanical_status'] = 'PASS'
                except Exception as e:
                    row['mechanical_status'] = 'FAIL'
                    row['mechanical_error'] = str(e)
                    row['status'] = 'FAIL'
                if row['status'] == 'ADMIT':
                    assert rr['submitted_gold_matches'] and rr['operation_alignment'] == 'PASS'
                    for dimension in ('grounding', 'unique_key', 'actual_operation', 'presentation'):
                        assert rr['checks'][dimension] == 'PASS', (q['candidate_id'], dimension)
                    assert rr['checks']['foils'] != 'FAIL'
                    exact = identity(q)
                    if exact in seen or exact in historical:
                        row['status'] = 'EXACT_WHOLE_DUPLICATE'
                    else:
                        seen.add(exact)
                if row['status'] == 'ADMIT':
                    # The Gold-count slot stays private, including in artifact IDs.
                    opaque = 'QFP2_' + hashlib.sha256(q['candidate_id'].encode()).hexdigest()[:16]
                    labels = list('ABCDEF' if slot.startswith('MC') else 'ABCD')
                    choices = q['options'] if slot.startswith('MC') else q['items']
                    mapping = dict(zip(labels, q['display_order']))
                    inverse = {v: k for k, v in mapping.items()}
                    operation = gate.public_operation(q['task_operation'])
                    gate.require_public_selection(q['question'], operation, q['question_type'])
                    public.append({k: q[k] for k in ('root_id', 'condition', 'question_type', 'question', 'source_binding')} |
                                  {'candidate_id': opaque, 'task_operation': operation,
                                   'display': {k: choices[v] for k, v in mapping.items()}})
                    private.append({'candidate_id': opaque, 'draft_candidate_id': q['candidate_id'],
                                    'slot_id': slot, 'question_type': q['question_type'],
                                    'canonical_gold': q['gold'], 'display_gold': [inverse[k] for k in q['gold']],
                                    'display_to_canonical': mapping, 'question_binding': bind(path)})
                    row['candidate_id'] = opaque
                    bindings.append(bind(path))
                decisions.append(row)
    assert len(decisions) == 56
    save(BASE / 'QUESTION_FREEZE.json', {
        'status': 'FROZEN_BEFORE_SOLVER', 'created_at_utc': now(), 'planned_slots': 56,
        'admitted_questions': len(public), 'decisions': decisions,
        'first_attempts_only': True, 'semantic_revisions': 0, 'solver_scores_read': False})
    save(BASE / 'inputs/PUBLIC_QUESTIONS.json', {'questions': public})
    save(BASE / 'inputs/PRIVATE_SCORING.json', {'questions': private})
    save(BASE / 'inputs/EXPORT_FREEZE.json', {
        'status': 'FROZEN_BEFORE_SOLVER', 'created_at_utc': now(), 'planned_slots': 56,
        'planned_root_ids': protocol['roots'], 'content_review_status': 'PASS',
        'solver_score_filtering': False, 'full_source_no_truncation': True,
        'public_binding': bind(BASE / 'inputs/PUBLIC_QUESTIONS.json'),
        'private_binding': bind(BASE / 'inputs/PRIVATE_SCORING.json'),
        'question_freeze_binding': bind(BASE / 'QUESTION_FREEZE.json'),
        'bindings': [bind(BASE / 'PROTOCOL_FREEZE.json'), bind(BASE / 'AUTHORING_RULE_BINDINGS.json'),
                     bind(__file__), bind(BASE / 'scripts/common.py'), bind(gate.__file__)] + bindings})
    sample = []
    for unit in protocol['reason_semantic_sample']:
        for slot in unit['slots']:
            row = next(d for d in decisions if all(d[k] == unit[k] for k in ('root_id', 'condition')) and d['slot_id'] == slot)
            sample.append({**{k: row[k] for k in ('root_id', 'condition', 'slot_id', 'status')},
                           'candidate_id': row.get('candidate_id'),
                           'request_id': row['candidate_id'] + '__B1_FULL_reason' if row['status'] == 'ADMIT' else None})
    save(BASE / 'inputs/REASON_SAMPLE.json', {'selection': 'preregistered before authoring; failed slots stay missing, no replacement',
                                           'protocol_binding': bind(BASE / 'PROTOCOL_FREEZE.json'), 'records': sample})
    print({'planned': 56, 'admitted': len(public), 'failed_or_shortfall': 56 - len(public), 'requests': 2 * len(public)})


if __name__ == '__main__':
    main()
