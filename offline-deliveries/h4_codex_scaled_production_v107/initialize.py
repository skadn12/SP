"""Validate source assignments, permanently reserve them, and freeze execution inputs."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path.insert(0, str(PROJECT / 'scripts'))
sys.dont_write_bytecode = True
from codex_controlled_batch import initialize, read, save, digest
from codex_batch_delivery import snapshot

def main():
    plan = read(HERE / 'PRECALL_PLAN.json')
    pool = {r['row']['original_index']: r for r in read(plan['source_pool_path'])['rows']}
    allocations = []
    for group in plan['groups']:
        doc = read(group['selection_output'])
        assert doc['group_id'] == group['group_id']
        assert [r['root_id'] for r in doc['allocations']] == [r['root_id'] for r in group['roots']]
        permitted = {r['row']['original_index'] for r in read(group['event_pool_path'])['rows']}
        for row, binding in zip(doc['allocations'], group['roots']):
            source = read(row['baseline_path'])
            assert row['baseline_sha256'] == digest(row['baseline_path']) == binding['baseline_sha256']
            assert row['source_original_index'] in permitted
            assert row['source_id'] == f"soda:train:{row['source_original_index']}"
            assert row['source_id'] not in plan['consumed_source_ids']
            assert row['full_baseline_read'] is True and row['full_event_read'] is True
            assert row['baseline_turn_refs_read'] == [t['turn_ref'] for t in source['turns']]
            speakers = {t['speaker'] for t in source['turns']}
            assert row['role_mapping'] and set(row['role_mapping'].values()) <= speakers
            turns = {t['turn_ref']: t for t in source['turns']}
            for c in row['continuity_constraints']:
                assert c['quote'] and c['quote'] in turns[c['turn_ref']]['text']
            event = pool[row['source_original_index']]
            assert not event.get('truncated_cells')
            allocations.append(row)
    assert len(allocations) == len({r['root_id'] for r in allocations}) == len({r['source_id'] for r in allocations}) == 12
    selection = HERE / 'evidence/SOURCE_SELECTION.json'
    save(selection, {'allocations': allocations, 'source_pool_path': plan['source_pool_path'],
        'source_pool_sha256': plan['source_pool_sha256'],
        'consumed_source_ids': plan['consumed_source_ids'],
        'consumption_scan': plan['consumption_scan'],
        'selection_method': plan['root_selection'] + ' ' + plan['event_selection'],
        'precall_plan_sha256': digest(HERE / 'PRECALL_PLAN.json'),
        'selection_group_bindings': [{'path': g['selection_output'], 'sha256': digest(g['selection_output'])} for g in plan['groups']],
        'solver_scores_used_for_selection': False})
    initialize(HERE, selection)
    feedback = HERE.parent / 'h4_codex_controlled_batch_v105/inputs/HISTORICAL_FEEDBACK.json'
    target = HERE / 'inputs/HISTORICAL_FEEDBACK.json'
    with target.open('xb') as f:
        f.write(feedback.read_bytes())
    save(HERE / 'evidence/FEEDBACK_FREEZE.json', {'path': str(target), 'sha256': digest(target),
        'upstream_path': str(feedback), 'upstream_sha256': digest(feedback),
        'usage': 'Read matching training-root record if available; otherwise explicitly no matching feedback. No validation/test scores used.'})
    # Freeze code before authors begin, in addition to the final delivery copy.
    for filename in ('codex_controlled_batch.py', 'codex_batch_questions.py', 'codex_batch_delivery.py'):
        source = PROJECT / 'scripts' / filename
        target = HERE / 'evidence/precall_code' / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as f:
            f.write(source.read_bytes())
    snapshot(HERE, plan['protected_batches'])
    save(HERE / 'EXECUTION_INPUTS_FROZEN.json', {
        'status': 'READY_FOR_ONE_DRAFT_PER_ROOT',
        'precall_plan': {'path': str(HERE / 'PRECALL_PLAN.json'), 'sha256': digest(HERE / 'PRECALL_PLAN.json')},
        'run_config': {'path': str(HERE / 'RUN_CONFIG.json'), 'sha256': digest(HERE / 'RUN_CONFIG.json')},
        'content_contract': {'path': str(HERE / 'evidence/CONTENT_CONTRACT.md'), 'sha256': digest(HERE / 'evidence/CONTENT_CONTRACT.md')},
        'source_reservations': {'path': str(HERE / 'SOURCE_RESERVATIONS.json'), 'sha256': digest(HERE / 'SOURCE_RESERVATIONS.json')},
        'training_started': False})

if __name__ == '__main__':
    main()
