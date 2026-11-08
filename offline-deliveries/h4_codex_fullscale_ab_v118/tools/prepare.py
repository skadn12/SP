"""Source matching and unit freezing; supplied human/model decisions stay explicit."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys

B = Path(__file__).resolve().parents[1]
P = B.parent.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(P / 'scripts'))
from codex_sequential_ab import SCENES, load_config, load_baseline, reservation_for


def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p): return {'path': str(Path(p).resolve()), 'sha256': sha(p)}
def save(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f: json.dump(d, f, ensure_ascii=False, indent=2); f.write('\n')
def copy(p, q):
    q.parent.mkdir(parents=True, exist_ok=True)
    with q.open('xb') as f: f.write(Path(p).read_bytes())
def verify(row): assert sha(row['path']) == row['sha256'], row['path']


def master():
    freeze = read(B / 'MASTER_FREEZE.json')
    assert freeze['status'] == 'FROZEN_BEFORE_NEW_BASELINE_FULL_TEXT_READS'
    for row in freeze['files']: verify(row)
    return read(B / 'MANIFEST.json')


def info(root, executor):
    manifest = master()
    row = next(r for r in manifest['roots'] if r['root_id'] == root)
    role = next(r for r in read(B / 'ROLE_ASSIGNMENTS.json')['roots'] if r['root_id'] == root)
    assert role['author'] == executor and executor != role['reviewer'], 'ACTUAL_ASSIGNED_AUTHOR_REQUIRED'
    unit = Path(row['unit_path'])
    constraints_path = unit / 'inputs/CONTINUATION_CONSTRAINTS.json'
    constraints = read(constraints_path)
    assert constraints['decision'] in ['PASS', 'WARNING'], 'BASELINE_HOLD_NOT_AUTHORABLE'
    assert constraints['root_id'] == root
    verify(constraints['baseline_binding'])
    for item in constraints['review_bindings']: verify(item)
    verify(row['baseline_binding'])
    return row, role, unit, constraints_path, constraints


def match_read(root, executor):
    row, role, unit, cp, constraints = info(root, executor)
    folder = unit / 'evidence/MATCHING'
    assert not (unit / 'selection.json').exists()
    started = now()
    poolp = unit / 'inputs/EVENT_POOL.json'
    source = read(row['baseline_path'])
    chunks = ['BASELINE_FULL', *[t['turn_ref'] + ' | ' + t['speaker'] + ': ' + t['text'] for t in source['turns']],
              'ALL_CONTINUATION_CONSTRAINTS', json.dumps(constraints, ensure_ascii=False),
              'COMPLETE_DISJOINT_NATIVE_EVENT_POOL', json.dumps(read(poolp), ensure_ascii=False),
              'CONTENT_CONTRACT', (B / 'evidence/CONTENT_CONTRACT.md').read_text()]
    folder.mkdir(parents=True, exist_ok=True)
    render = folder / 'FULL_INPUT.txt'
    with render.open('x') as f: f.write('\n'.join(chunks) + '\n')
    save(folder / 'READ_CONTEXT.json', {'started_at_utc': started, 'executor_id': executor, 'root_id': root,
         'baseline_binding': row['baseline_binding'], 'constraints_binding': bind(cp),
         'role_manifest_binding': bind(B / 'ROLE_ASSIGNMENTS.json'), 'event_pool_binding': bind(poolp),
         'input_render_binding': bind(render), 'tool_binding': bind(__file__),
         'full_source_read_is_an_executor_attestation': True})
    print(render.read_text())


def match_save(root, executor, payload):
    row, role, unit, cp, constraints = info(root, executor)
    folder = unit / 'evidence/MATCHING'
    context = read(folder / 'READ_CONTEXT.json')
    assert context['executor_id'] == executor
    for k in ['baseline_binding', 'constraints_binding', 'role_manifest_binding', 'event_pool_binding', 'input_render_binding', 'tool_binding']: verify(context[k])
    raw = read(payload)
    att = raw['read_attestation']
    assert att['full_source_read'] is True and att['all_continuation_constraints_read'] is True and att['all_candidate_events_fully_read'] is True
    source = read(row['baseline_path'])
    turns = {t['turn_ref']: t for t in source['turns']}
    plans = raw['a_continuations']
    assert len(plans) == 3 and [p['scene_ref'] for p in plans] == list(SCENES[:3])
    for n, plan in enumerate(plans, 1):
        assert plan['thread'].strip() and plan['evidence']
        if row['cohort'] == 'standard':
            assert plan['baseline_scene_ref'] == f'R01:S{n:02d}'
        for quote in plan['evidence']:
            t = turns[quote['turn_ref']]
            assert quote['quote'] and quote['quote'] in t['text'] and quote['speaker'] == t['speaker']
    pool = {e['dialogue_id']: e for e in read(unit / 'inputs/EVENT_POOL.json')['events']}
    allocations = raw['allocations']
    assert len(allocations) == 3 and [a['scene_ref'] for a in allocations] == list(SCENES[3:])
    ids = [a['source_id'] for a in allocations]
    assert len(set(ids)) == 3
    consumed = set(read(B / 'evidence/SOURCE_POOL_FREEZE.json')['consumed_source_ids'])
    for a in allocations:
        assert a['source_id'].startswith('multiwoz22:train:') and a['source_id'] not in consumed
        assert a['source_id'].removeprefix('multiwoz22:train:') in pool
        assert a['compatibility_reason'].strip() and a['main_purpose'].strip()
    copy(payload, folder / 'MATCHING_PAYLOAD.json')
    selected = {'root_id': root, 'selector': executor, 'started_at_utc': context['started_at_utc'], 'finished_at_utc': now(),
        'full_baseline_read': True, 'full_source_read': True, 'all_continuation_constraints_read': True,
        'baseline_sha256': row['baseline_sha256'], 'baseline_turn_refs_read': list(turns),
        'constraints_binding': bind(cp), 'continuation_constraints': constraints['continuation_constraints'],
        'all_candidate_native_events_fully_read': True, 'event_pool_binding': context['event_pool_binding'],
        'input_context_binding': bind(folder / 'READ_CONTEXT.json'), 'raw_payload_binding': bind(folder / 'MATCHING_PAYLOAD.json'),
        'a_continuations': plans, 'allocations': [], 'semantic_review_is_programmatic': False}
    for a in allocations:
        event = pool[a['source_id'].removeprefix('multiwoz22:train:')]
        selected['allocations'].append({**a, 'root_id': root, 'native_turn_ids_read': [t['turn_id'] for t in event['turns']], 'full_event_utterances_read': True})
    save(unit / 'selection.json', selected)
    reservations = []
    for n, a in enumerate(selected['allocations'], 1):
        event = pool[a['source_id'].removeprefix('multiwoz22:train:')]
        target = unit / 'inputs' / f'B_S{n:02d}_EVENT.json'
        save(target, event)
        reservations.append({**a, 'source_item_path': str(target), 'source_item_sha256': sha(target),
            'selection_path': str(unit / 'selection.json'), 'selection_sha256': sha(unit / 'selection.json'),
            'status': 'PERMANENTLY_RESERVED_BEFORE_AUTHORING'})
    save(unit / 'SOURCE_RESERVATIONS.json', {'batch_id': B.name, 'root_id': root, 'created_at_utc': now(), 'allocation': reservations, 'permanent_including_failed_and_unexecuted_slots': True})
    cfg = {'controller_version': 'codex_sequential_ab_v1', 'batch_id': B.name, 'created_at_utc': now(),
        'roots': [{'root_id': root, 'baseline_path': row['baseline_path'], 'baseline_sha256': row['baseline_sha256']}],
        'scene_sequence': list(SCENES), 'scope_scene_refs': list(SCENES), 'new_turn_count': {'min': 12, 'max': 16},
        'required_rules': ['time', 'location', 'actor_identity', 'plan_vs_completion', 'cross_scene_continuity', 'semantic_repetition', 'naturalness', 'event_fidelity'],
        'seed_namespace': B.name, 'multiselect_target_cardinalities': [2, 3, 4, 5], 'gold6_control_per_ready_root': 1,
        'ordering_slots_per_ready_root': 2, 'author_attempts_per_scene': 1, 'qa_author_attempts_per_complete_root': 1,
        'source_replacements': 0, 'semantic_repairs': 0, 'backend': 'CURRENT_CODEX_SESSION', 'exact_model_id': None,
        'training_run': False, 'solver_run': True, 'solver_protocol': 'v15_full_only_random_mcq6x3_order12x1_v1',
        'selection': 'Complete-valid 0 < equal-display partial mean < 1; Gold6 excluded',
        'failure_policy': 'Stop later dependent scenes; retain raw first draft, HOLD, errors and denominator.',
        'cohort': row['cohort'], 'baseline_gate_binding': bind(cp), 'master_manifest_binding': bind(B / 'MANIFEST.json')}
    save(unit / 'RUN_CONFIG.json', cfg)
    checked = load_config(unit)
    load_baseline(unit, checked, root)
    for scene in SCENES[3:]: reservation_for(unit, checked, root, scene)
    code = [P / 'scripts' / n for n in ['codex_sequential_ab.py', 'codex_scene_payload.py', 'codex_sequential_questions.py', 'codex_controlled_batch.py']]
    inputs = [unit / 'RUN_CONFIG.json', unit / 'SOURCE_RESERVATIONS.json', unit / 'selection.json', cp, Path(row['baseline_path']), *(unit / 'inputs').glob('*'), *code]
    inputs = list(dict.fromkeys(inputs))
    save(unit / 'EXECUTION_INPUTS_FROZEN.json', {'created_at_utc': now(), 'root_id': root, 'status': 'READY_FOR_A3_THEN_B3',
        'master_freeze_binding': bind(B / 'MASTER_FREEZE.json'), 'files': [bind(p) for p in inputs],
        'author': executor, 'reviewer': role['reviewer'], 'generation_started': False})
    print(json.dumps({'root_id': root, 'status': 'INPUTS_FROZEN', 'events_reserved': ids, 'generation_started': False}))


def freeze():
    assert not (B / 'MASTER_FREEZE.json').exists()
    required = [B / n for n in ['MANIFEST.json', 'ROLE_ASSIGNMENTS.json', 'evidence/BASELINE_POLICY.md', 'evidence/CONTENT_CONTRACT.md',
        'evidence/SOURCE_POOL_FREEZE.json', 'evidence/HISTORICAL_PROTECTION.json', 'bootstrap.py', 'tools/prepare.py', 'tools/baseline.py', 'tools/author.py', 'tools/review.py']]
    required += [P / 'scripts' / n for n in ['codex_sequential_ab.py', 'codex_scene_payload.py', 'codex_sequential_questions.py', 'codex_controlled_batch.py', 'codex_evidence_export.py']]
    required += [Path(r['baseline_path']) for r in read(B / 'MANIFEST.json')['roots']]
    required += sorted(B.glob('units/*/inputs/EVENT_POOL.json'))
    assert all(p.is_file() for p in required)
    save(B / 'MASTER_FREEZE.json', {'status': 'FROZEN_BEFORE_NEW_BASELINE_FULL_TEXT_READS', 'created_at_utc': now(),
        'files': [bind(p) for p in required], 'baseline_full_text_semantically_read': False,
        'new_generation_started': False, 'validation_gate': 'Finish the fixed first wave, audit actual completed units, then release production wave; no replacement of held roots.',
        'execution_identity_note': 'Declared actual executor checked against fixed assignments; shared filesystem is not cryptographic caller authentication.'})
    print(json.dumps({'status': 'MASTER_FROZEN', 'files': len(required)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'match-read', 'match-save'])
    parser.add_argument('--root'); parser.add_argument('--executor'); parser.add_argument('--payload', type=Path)
    args = parser.parse_args()
    if args.action == 'freeze': freeze()
    else:
        assert args.root and args.executor
        if args.action == 'match-read': match_read(args.root, args.executor)
        else:
            assert args.payload
            match_save(args.root, args.executor, args.payload)
