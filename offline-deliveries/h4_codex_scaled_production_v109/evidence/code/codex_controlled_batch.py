"""Offline controller for explicit Codex-session artifacts; never invokes a model.

Every run is configured by RUN_CONFIG.json. Semantic decisions must be supplied
by a declared reader; structural checks never claim to replace source review.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

VERSION = 'codex_controlled_batch_v1'

def now():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':')).encode('utf-8')).hexdigest()

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def copy_once(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        stream.write(Path(source).read_bytes())

def root_folder(batch, root_id):
    assert re.fullmatch(r'cc-train-episode-\d+', root_id), root_id
    return Path(batch) / 'roots' / ('root' + root_id.rsplit('-', 1)[-1])

def config(batch):
    value = read(Path(batch) / 'RUN_CONFIG.json')
    assert value['controller_version'] == VERSION
    return value

def initialize(batch, selection_path):
    batch = Path(batch).resolve()
    selection = read(selection_path)
    allocations = selection['allocations']
    assert allocations and len({r['root_id'] for r in allocations}) == len(allocations)
    assert len({r['source_id'] for r in allocations}) == len(allocations)
    consumed = set(selection['consumed_source_ids'])
    pool_path = Path(selection['source_pool_path'])
    pool = {int(x['row']['original_index']): x for x in read(pool_path)['rows']}
    assert not (batch / 'RUN_CONFIG.json').exists()
    # Validate all allocations before reserving any of them.
    for row in allocations:
        root_folder(batch, row['root_id'])
        assert digest(row['baseline_path']) == row['baseline_sha256']
        assert row['source_id'] not in consumed
        assert row['source_id'] == f"soda:train:{row['source_original_index']}"
        assert row['source_original_index'] in pool
        assert read(row['baseline_path'])['root_id'] == row['root_id']
    started = now()
    frozen = []
    for row in allocations:
        short = row['root_id'].rsplit('-', 1)[-1]
        baseline = batch / 'inputs' / f'root{short}_BASELINE.json'
        event = batch / 'inputs' / f'root{short}_EVENT.json'
        copy_once(row['baseline_path'], baseline)
        save(event, pool[row['source_original_index']])
        frozen.append({**row, 'baseline_path': str(baseline),
            'baseline_sha256': digest(baseline), 'baseline_hash_mode': 'file_bytes',
            'source_item_path': str(event), 'source_item_sha256': digest(event),
            'source_item_hash_mode': 'file_bytes',
            'source_item_canonical_sha256': canonical_digest(read(event)),
            'status': 'PERMANENTLY_RESERVED_BEFORE_AUTHORING'})
    save(batch / 'SOURCE_RESERVATIONS.json', {'created_at_utc': started,
        'batch_id': batch.name, 'allocation': frozen,
        'selection_sha256': digest(selection_path),
        'consumption_scan': selection['consumption_scan']})
    save(batch / 'RUN_CONFIG.json', {'controller_version': VERSION,
        'batch_id': batch.name, 'created_at_utc': started,
        'roots': [r['root_id'] for r in frozen], 'scene_ref': 'R02:C:S01',
        'new_turn_count': {'min': 12, 'max': 16},
        'author_attempts_per_root': 1, 'qa_author_attempts_per_root': 1,
        'multiselect_target_cardinalities': [2, 3, 4, 5],
        'gold6_control_per_ready_root': 1, 'ordering_slots_per_ready_root': 2,
        'seed_namespace': batch.name, 'backend': 'CURRENT_CODEX_SESSION',
        'exact_model_id': None, 'exact_model_id_limitation':
            'Inherited session model identifier not exposed; no model identity guessed.',
        'qwen_transport_used': False, 'codex_cli_used': False,
        'solver_run': False, 'training_run': False,
        'asset_review_required_before_composition': True,
        'public_review_required_before_private_gold': True,
        'semantic_review_is_programmatic': False,
        'immutable_after_save': True, 'source_substitution_allowed': False,
        'sampling': selection['selection_method']})
    print(json.dumps({'initialized': str(batch), 'roots': [r['root_id'] for r in frozen],
        'reserved_sources': [r['source_id'] for r in frozen]}, ensure_ascii=False))

def materialize(batch):
    batch = Path(batch); cfg = config(batch)
    reservations = read(batch / 'SOURCE_RESERVATIONS.json')['allocation']
    for row in reservations:
        folder = root_folder(batch, row['root_id'])
        draft_path = folder / 'EXPANSION_DRAFT.json'
        if not draft_path.exists() or (folder / 'MATERIALIZATION_RECEIPT.json').exists():
            continue
        if (folder / 'STRUCTURAL_FAILURE.json').exists():
            continue
        issues = []
        try:
            baseline = read(row['baseline_path']); draft = read(draft_path)
            assert digest(row['baseline_path']) == row['baseline_sha256'], 'baseline_hash'
            assert digest(row['source_item_path']) == row['source_item_sha256'], 'event_hash'
            turns = draft['turns']; scene = cfg['scene_ref']
            assert draft['root_id'] == row['root_id'], 'root_id'
            assert draft['source_id'] == row['source_id'], 'source_id'
            assert cfg['new_turn_count']['min'] <= len(turns) <= cfg['new_turn_count']['max'], 'turn_count'
            speakers = {t['speaker'] for t in baseline['turns']}
            for i, turn in enumerate(turns, 1):
                assert set(turn) == {'turn_ref', 'session_ref', 'scene_ref', 'speaker', 'text', 'origin'}, 'turn_fields'
                assert turn['turn_ref'] == f'{scene}:T{i:03d}', 'turn_ref'
                assert turn['session_ref'] == turn['scene_ref'] == scene, 'scene_ref'
                assert turn['speaker'] in speakers, 'speaker_not_established'
                assert isinstance(turn['text'], str) and turn['text'].strip(), 'text'
                assert turn['origin'] == 'codex_session_generated', 'origin'
            cumulative = baseline['turns'] + turns
            assert len({t['turn_ref'] for t in cumulative}) == len(cumulative), 'duplicate_turn_refs'
            output = folder / 'CUMULATIVE_SOURCE.json'
            save(output, {'schema_version': VERSION + '_source', 'root_id': row['root_id'],
                'baseline_sha256': digest(row['baseline_path']),
                'draft_sha256': digest(draft_path), 'baseline_turn_count': len(baseline['turns']),
                'new_turn_count': len(turns), 'turns': cumulative})
            save(folder / 'MATERIALIZATION_RECEIPT.json', {'created_at_utc': now(),
                'status': 'STRUCTURAL_PASS_ONLY', 'prefix_exact': cumulative[:len(baseline['turns'])] == baseline['turns'],
                'source_sha256': digest(output), 'draft_sha256': digest(draft_path),
                'source_semantic_ready': False})
            print(row['root_id'], 'STRUCTURAL_PASS', len(turns))
        except (AssertionError, ValueError, KeyError, TypeError) as exc:
            save(folder / 'STRUCTURAL_FAILURE.json', {'created_at_utc': now(),
                'status': 'FAIL', 'error': str(exc), 'draft_sha256': digest(draft_path)})
            print(row['root_id'], 'STRUCTURAL_FAIL', str(exc))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['init', 'materialize', 'publish-assets', 'compose', 'finalize'])
    parser.add_argument('--batch-dir', type=Path, required=True)
    parser.add_argument('--selection', type=Path)
    args = parser.parse_args()
    if args.command == 'init':
        assert args.selection, '--selection required'
        initialize(args.batch_dir, args.selection)
    elif args.command == 'materialize':
        materialize(args.batch_dir)
    else:
        from codex_batch_questions import run_stage
        run_stage(args.batch_dir, args.command)

if __name__ == '__main__':
    main()
