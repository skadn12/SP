"""Read-only reconciliation of final exports, ledgers, and frozen population."""
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
B = HERE.parent
sys.path.insert(0, str(HERE))
from production_inventory_v1 import inventory
from status import population


def read(path):
    return json.loads(Path(path).read_text())


def verify(binding):
    assert hashlib.sha256(Path(binding['path']).read_bytes()).hexdigest() == binding['sha256'], binding['path']


def checks():
    output = B / 'delivery/all'
    final = read(output / 'FINAL_STATUS.json')
    state = population()
    current_inventory = inventory()
    attempts = read(output / 'ALL_ATTEMPTS_INDEX.json')
    assert state['population_denominator'] == state['terminal_roots'] == 68
    assert final['population_denominator'] == final['terminal_roots'] == 68
    assert final['status'] == 'FULL_FIXED_POPULATION_PROCESSED'
    assert final['statuses'] == state['statuses'] == attempts['statuses']
    assert len(attempts['roots']) == 68
    assert attempts['new_materialized_scenes'] == current_inventory['new_materialized_scenes']
    assert attempts['source_only_complete_qa_hold_roots'] == current_inventory['source_only_complete_qa_hold_roots']
    for key in ['training_started', 'training_pool_merged', 'external_semantic_certification', 'unseen_holdout']:
        assert final[key] is False
    for item in read(B / 'MASTER_FREEZE.json')['files']:
        verify(item)
    for item in read(B / 'evidence/HISTORICAL_PROTECTION.json')['files']:
        verify(item)
    audit = read(B / 'evidence/ALL_CONTENT_AUDIT.json')
    assert audit['status'] == 'PASS' and not audit['errors'] and audit['root_denominator'] == 53
    for item in audit['input_bindings']:
        verify(item)
    content = read(output / 'CONTENT_QUESTIONS.json')['questions']
    selected = read(output / 'SELECTED_CANDIDATES.json')['questions']
    by_id = {q['candidate_id']: q for q in content}
    selected_by_id = {q['candidate_id']: q for q in selected}
    assert len(by_id) == len(content) == final['content_questions']
    assert len(selected_by_id) == len(selected) == final['selected_candidates']
    assert set(selected_by_id) <= set(by_id)
    assert all(q == by_id[cid] and q.get('role') != 'control6' for cid, q in selected_by_id.items())
    for questions, prefix in [(content, 'content'), (selected, 'selected')]:
        assert dict(Counter(q['question_type'] for q in questions)) == final[prefix + '_type_counts']
        assert dict(Counter(str(len(q['gold'])) for q in questions if q['question_type'] == 'multi_select')) == final[prefix + '_gold_counts']
    export_roots = {r['root_id']: r for r in final['roots']}
    assert set(export_roots) == {r['root_id'] for r in state['roots']}
    from deliver_v2 import engine
    engine()['verify_candidates'](content)
    counts = Counter(q['root_id'] for q in content)
    chosen_counts = Counter(q['root_id'] for q in selected)
    for rid, row in export_roots.items():
        assert row['content_questions'] == counts[rid]
        assert row['selected_candidates'] == chosen_counts[rid]
        if row['complete_dialogue']:
            verify(row['source_binding'])
            source_path = Path(row['source_binding']['path'])
            source = read(source_path)
            lines = ['# ' + rid, '', 'Source: ' + str(source_path),
                     'SHA256: ' + row['source_binding']['sha256'], '']
            lines.extend('[' + t['turn_ref'] + '] ' + t['speaker'] + ': ' + t['text'] for t in source['turns'])
            assert Path(row['readable_dialogue_path']).read_text() == '\n\n'.join(lines) + '\n'
            assert Path(row['readable_questions_path']).is_file()
        if row['action'] in ['SCREEN', 'INHERIT_SCREEN'] and row['status'] != 'NEW_CONTENT_COMPLETED':
            assert counts[rid] == chosen_counts[rid] == 0
    assert final['complete_dialogues'] == sum(r['complete_dialogue'] for r in final['roots'])
    assert final['new_complete_dialogues'] == sum(r['complete_dialogue'] for r in final['roots'] if r['action'] in ['SCREEN', 'INHERIT_SCREEN'])
    for row in attempts['roots']:
        for item in row['stage_artifacts']:
            verify(item)
        if row['source_only_complete_qa_hold']:
            verify(row['source_only_readable_dialogue_binding'])
    calls = {}
    for wave in ['validation', 'production']:
        scope = B / 'evaluations' / wave
        paired = read(scope / 'ADAPTER_BINDINGS.json')
        for item in paired['files']:
            verify(item)
        raw_audit = read(scope / 'INDEPENDENT_SOLVER_AUDIT.json')
        assert raw_audit['status'] == 'PASS' and not raw_audit['errors']
        for item in raw_audit['input_bindings']:
            verify(item)
        for side in ['base', 'trained']:
            assert read(scope / (side.upper() + '_COMPLETE.json'))['actual_calls'] == paired['calls_per_model']
        pool = read(scope / 'EVALUATION_POOL.json')['questions']
        results = {q['candidate_id']: q for q in read(scope / 'run/trained/RESULTS.json')['questions']}
        assert set(results) == {q['candidate_id'] for q in pool}
        for q in pool:
            cid = q['candidate_id']
            assert q == by_id[cid]
            result = results[cid]
            expected = result['selected_by_policy'] and q.get('role') != 'control6'
            assert (cid in selected_by_id) == expected
            if cid in selected_by_id:
                assert 0 < Fraction(result['full_partial_mean']) < 1
                assert not result['data_errors']
        calls[wave] = paired['calls_per_model']
    ledger = read(output / 'QUESTION_STAGE_LEDGER.json')
    assert ledger['population_roots'] == len(ledger['roots']) == 68
    assert len({r['root_id'] for r in ledger['roots']}) == 68
    assert {r['root_id'] for r in ledger['roots']} == set(export_roots)
    unfilled = [s for r in ledger['roots'] for s in r['unfilled_composition_slots']]
    assert ledger['unfilled_composition_slots'] == len(unfilled)
    assert ledger['unfilled_slot_types'] == dict(Counter(s['question_type'] for s in unfilled))
    assert not set(by_id).intersection(s['candidate_id'] for s in unfilled if 'candidate_id' in s)
    for row in ledger['roots']:
        unit = B / 'units' / ('root' + row['root_id'].rsplit('-', 1)[1])
        composition = unit / 'roots' / unit.name / 'COMPOSITION_RECEIPT.json'
        crosscheck = unit / 'roots' / unit.name / 'FINAL_CROSSCHECK.json'
        assert row['population_status'] == export_roots[row['root_id']]['status']
        assert row['unfilled_composition_slots'] == (read(composition)['unfilled_slots'] if composition.exists() else [])
        assert row['final_crosscheck_results'] == (read(crosscheck)['results'] if crosscheck.exists() else [])
        for item in row['bindings']:
            verify(item)
    notes = read(output / 'PROCEDURE_DEVIATIONS.json')
    for item in notes['input_bindings']:
        verify(item)
    return {'status': 'PASS', 'population': 68, 'content_questions': len(content),
            'selected_candidates': len(selected), 'complete_dialogues': final['complete_dialogues'],
            'source_only_qa_hold_dialogues': len(attempts['source_only_complete_qa_hold_roots']),
            'unfilled_composition_slots': len(unfilled), 'paired_calls_per_side': calls,
            'source_question_gold_or_selection_changed': False,
            'scope': 'Mechanical reconciliation only; final reviewer must independently assess this tool, records, and report limitations.'}


if __name__ == '__main__':
    print(json.dumps(checks(), ensure_ascii=False, indent=2))
