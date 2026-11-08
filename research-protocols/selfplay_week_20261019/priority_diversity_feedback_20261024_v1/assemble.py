"""Validate content reviews and feedback realization before any Solver call."""
from pathlib import Path
from collections import Counter
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
GATE_PATH = HERE.parents[2] / 'scripts/current_method/production_question_gate.py'
sys.path.insert(0, str(GATE_PATH.parent))
spec = importlib.util.spec_from_file_location('priority_production_gate', GATE_PATH)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def read(p):
    return json.loads(Path(p).read_text())


def save(name, value):
    with (HERE / name).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def rows(value, field):
    return value if isinstance(value, list) else value[field]


def main():
    selection_path = HERE / 'SELECTION_FREEZE.json'
    selection = read(selection_path)
    feedback_freeze = read(HERE / 'FEEDBACK_INPUT_FREEZE.json')
    input_card_ids = {c['card_id'] for c in feedback_freeze['cards']}
    expected = {f"PDF1-{root['root_id'].split('-')[-1]}-{s['slot']}": (root, s)
                for root in selection['roots'] for s in root['slot_plan']}
    all_candidates, all_reviews, ledger, audits, consumption = [], [], [], [], []
    for group in ['a', 'b']:
        draft_path = HERE / f'author_{group}/CANDIDATES_PRIVATE_DRAFT.json'
        final_path = HERE / f'author_{group}/CANDIDATES_PRIVATE.json'
        review_path = HERE / f'review_{group}/CONTENT_REVIEW_AI.json'
        draft = {c['candidate_id']: c for c in rows(read(draft_path), 'candidates')}
        final = rows(read(final_path), 'candidates')
        reviews = rows(read(review_path), 'reviews')
        by_review = {r['candidate_id']: r for r in reviews}
        initial_reviews = rows(read(HERE / f'review_{group}/INITIAL_CONTENT_REVIEW_AI.json'), 'reviews')
        initial_by_id = {r['candidate_id']: r for r in initial_reviews}
        assert len(by_review) == len(reviews) == len(final)
        assert set(draft) == set(by_review) == {c['candidate_id'] for c in final}
        assert set(initial_by_id) == set(draft) and len(initial_reviews) == len(draft)
        bound, _ = gate.bind_candidates(final, selection_path)
        assert bound == final, 'source binding must be complete before final review'
        submitted = set()
        for c in final:
            cid = c['candidate_id']
            root, slot = expected[cid]
            assert root['author_group'] == group and c['root_id'] == root['root_id']
            assert c['question_type'] == slot['question_type']
            assert c['task_operation'] == slot['task_operation']
            assert c.get('formal_weakness_admission') is False
            if c['question_type'] == 'multi_select':
                assert len(c['gold']) == slot['gold_count']
            else:
                assert 4 <= len(c['items']) <= 6
            feedback = c['feedback_target']
            assert feedback['card_id'] == slot['feedback_card_id']
            assert feedback['dimension'] == slot['feedback_dimension']
            for field in ['predicted_error',
                          'source_specific_realization', 'positive_counterexample']:
                assert isinstance(feedback.get(field), str) and feedback[field].strip(), (cid, field)
            alternatives = feedback.get('competing_explanation')
            assert ((isinstance(alternatives, str) and alternatives.strip()) or
                    (isinstance(alternatives, list) and alternatives and all(
                        isinstance(v, str) and v.strip() for v in alternatives))), cid
            review = by_review[cid]
            initial_result = gate.validate_review(draft[cid], initial_by_id[cid], selection_path)
            assert initial_result['review_record_complete'], (cid, initial_result['review_errors'])
            result = gate.validate_review(c, review, selection_path)
            assert result['review_record_complete'], (cid, result['review_errors'])
            realization = review.get('feedback_realization', {})
            assert realization.get('card_id') == feedback['card_id']
            assert realization.get('decision') in ['PASS', 'HOLD', 'FAIL']
            assert isinstance(realization.get('rationale'), str) and realization['rationale'].strip()
            if review['decision'] == 'PASS':
                assert result['admitted'], (cid, result)
                assert realization['decision'] == 'PASS', cid
            submitted.add(cid)
            all_candidates.append(c)
            all_reviews.append(review)
            audits.append(result)
            ledger.append({'candidate_id': cid, 'root_id': root['root_id'],
                           'slot': slot['slot'], 'task_operation': slot['task_operation'],
                           'status': review['decision'], 'feedback_card_id': feedback['card_id'],
                           'initial_status': initial_by_id[cid]['decision'],
                           'candidate_sha256': gate.fingerprint(c),
                           'initial_candidate_sha256': gate.fingerprint(draft[cid]),
                           'revised_before_solver': c != draft[cid]})
        author_ledger = rows(read(HERE / f'author_{group}/SLOT_LEDGER_DRAFT.json'), 'slots')
        by_slot = {r.get('candidate_id', r.get('slot_id')): r for r in author_ledger}
        group_expected = {cid for cid, (r, _) in expected.items() if r['author_group'] == group}
        assert set(by_slot) == group_expected
        for cid in sorted(group_expected - submitted):
            row = by_slot[cid]
            assert row['status'] == 'MATERIAL_SHORTFALL', (cid, row)
            root, slot = expected[cid]
            ledger.append({**row, 'candidate_id': cid, 'root_id': root['root_id'],
                           'slot': slot['slot'], 'task_operation': slot['task_operation'],
                           'feedback_card_id': slot['feedback_card_id']})
        receipt = read(HERE / f'author_{group}/AUTHOR_READ_RECEIPT.json')
        assert receipt.get('solver_outputs_read', receipt.get('solver_results_read')) is False
        assert set(receipt.get('card_ids_read', receipt.get('feedback_card_ids_read', []))) == input_card_ids
        assert receipt.get('source_card_binding', receipt.get('feedback_source_binding')) == feedback_freeze['source_cards_binding']
        protocol_bindings = receipt.get('protocols_read', [])
        for name, field in [('SELECTION_FREEZE.json', 'selection_binding'),
                            ('PROTOCOL_FREEZE.json', 'protocol_binding'),
                            ('FEEDBACK_INPUT_FREEZE.json', 'feedback_binding')]:
            expected_binding = gate.file_binding(HERE/name)
            if field in receipt:
                assert receipt[field] == expected_binding
            else:
                assert any(x.get('path') == expected_binding['path'] and
                           x.get('sha256') == expected_binding['sha256'] for x in protocol_bindings)
        expected_sources = {r['path']: r for r in selection['roots'] if r['author_group'] == group}
        source_receipts = receipt['sources']
        assert len(source_receipts) == len(expected_sources)
        assert {s.get('binding', s.get('source_binding'))['path'] for s in source_receipts} == set(expected_sources)
        for source_receipt in source_receipts:
            source_binding = source_receipt.get('binding', source_receipt.get('source_binding'))
            assert gate.file_binding(source_binding['path']) == source_binding
            source = read(source_binding['path'])
            assert source_receipt['turn_refs_read'] == [t['turn_ref'] for t in source['turns']]
        plans_document = read(HERE / f'author_{group}/TASK_PLANS.json')
        plans = rows(plans_document, 'plans')
        assert len(plans) == len(group_expected) and {q['candidate_id'] for q in plans} == group_expected
        plans_by_id = {q['candidate_id']: q for q in plans}
        for c in final:
            plan = plans_by_id[c['candidate_id']]
            assert plan['card_id'] == c['feedback_target']['card_id']
            assert plan['task_operation'] == c['task_operation'] and plan['root_id'] == c['root_id']
        consumption.append({'author_group':group,'author_id':receipt['author_id'],
            'questions_submitted':len(final),'cards_read':sorted(input_card_ids),
            'source_turn_refs_verified':True,'plans_cover_all_slots':True,
            'receipt_binding':gate.file_binding(HERE/f'author_{group}/AUTHOR_READ_RECEIPT.json'),
            'plans_binding':gate.file_binding(HERE/f'author_{group}/TASK_PLANS.json'),
            'initial_review_binding':gate.file_binding(HERE/f'review_{group}/INITIAL_CONTENT_REVIEW_AI.json'),
            'final_review_binding':gate.file_binding(review_path)})
    assert len(ledger) == len(expected) == 32
    signatures = {}
    old_questions = []
    for dirname in ['postweek_production_batch_v1', 'postweek_production_batch_v2']:
        f = HERE.parent / dirname / 'CANDIDATES_PRIVATE.json'
        if f.exists():
            old_questions.extend(read(f))
    def identity(c):
        choices = c.get('options', c.get('items'))
        return gate.fingerprint({'question_type': c['question_type'],
                                 'question': c['question'], 'choices': sorted(choices.values())})
    old_signatures = {identity(c) for c in old_questions}
    for c in all_candidates:
        key = identity(c)
        assert key not in signatures and key not in old_signatures, 'WHOLE_QUESTION_EXACT_DUPLICATE'
        signatures[key] = c['candidate_id']
    all_candidates.sort(key=lambda c: c['candidate_id'])
    all_reviews.sort(key=lambda r: r['candidate_id'])
    ledger.sort(key=lambda r: r['candidate_id'])
    save('CANDIDATES_PRIVATE.json', all_candidates)
    save('CONTENT_REVIEW_AI.json', {'reviews': all_reviews, 'reviewer_is_human': False,
         'gold_blind': False, 'solver_results_read': False, 'author_reviewer_separation': True})
    save('SLOT_LEDGER.json', {'planned_slots': 32, 'slots': ledger,
         'counts': dict(Counter(r['status'] for r in ledger)), 'solver_score_filtering': False})
    save('CONTENT_FREEZE.json', {'status': 'CONTENT_AND_FEEDBACK_REALIZATION_REVIEW_COMPLETE',
         'planned_slots': 32, 'submitted': len(all_candidates),
         'admitted': sum(a['admitted'] for a in audits), 'mechanical_audits': audits,
         'initial_review_counts':dict(Counter(r.get('initial_status','MATERIAL_SHORTFALL') for r in ledger)),
         'author_consumption':consumption,
         'card_consumption_is_formal_weakness_proof': False,
         'duplicate_boundary': 'Exact normalized representation only; semantic equivalence independently checked in review.',
         'bindings': [gate.file_binding(HERE / name) for name in
                      ['SELECTION_FREEZE.json', 'PROTOCOL_FREEZE.json', 'FEEDBACK_INPUT_FREEZE.json',
                       'CANDIDATES_PRIVATE.json', 'CONTENT_REVIEW_AI.json', 'SLOT_LEDGER.json']]})
    print(json.dumps({'planned': 32, 'submitted': len(all_candidates),
                      'admitted': sum(a['admitted'] for a in audits),
                      'slot_status': dict(Counter(r['status'] for r in ledger))}))


if __name__ == '__main__':
    main()
