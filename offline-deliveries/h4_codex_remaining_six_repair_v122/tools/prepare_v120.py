"""Frozen retry matching: inherited A/prefix stay unchanged; only new B consumes events.

No semantic decision is made here. The assigned author reads all frozen inputs,
selects compatible native events, and supplies explicit compatibility reasons.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys

sys.dont_write_bytecode = True
B = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(B / 'scripts'))
from codex_sequential_ab import SCENES, load_config, load_baseline, reservation_for


def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p): return {'path': str(Path(p).resolve()), 'sha256': sha(p)}
def need(ok, message):
    if not ok: raise ValueError(message)
def save(p, d):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=2); f.write('\n')
def copy(p, q):
    q = Path(q); q.parent.mkdir(parents=True, exist_ok=True)
    with q.open('xb') as f: f.write(Path(p).read_bytes())
def verify(row):
    need(sha(row['path']) == row['sha256'], 'FROZEN_INPUT_CHANGED:' + row['path'])
    return Path(row['path'])


def master():
    freeze = read(B / 'MASTER_FREEZE.json')
    need(freeze['status'] == 'FROZEN_BEFORE_V122_FORMAL_AUTHOR_MATCHING_AND_GENERATION', 'MASTER_NOT_READY')
    for row in freeze['files']: verify(row)
    return read(B / 'MANIFEST.json')


def info(root, executor):
    row = next(r for r in master()['roots'] if r['root_id'] == root)
    need(row['prefix_reuse_approved'] is True, 'PREFIX_HOLD_NO_RETRY')
    role = next(r for r in read(B / 'ROLE_ASSIGNMENTS.json')['roots'] if r['root_id'] == root)
    need(role['author'] == executor and executor != role['reviewer'], 'ACTUAL_ASSIGNED_AUTHOR_REQUIRED')
    unit = Path(row['unit_path'])
    cp = unit / 'inputs/CONTINUATION_CONSTRAINTS.json'
    constraints = read(cp)
    need(constraints['root_id'] == root and constraints['decision'] in ['PASS', 'WARNING'], 'BASELINE_INELIGIBLE')
    for item in [row['baseline_binding'], row['prefix_import_binding'], row['candidate_prefix_binding'], constraints['baseline_binding'], *constraints['review_bindings']]: verify(item)
    historical = read(unit / 'A_PLAN_POLICY.json')
    if row.get('a_plan_mode', 'INHERIT_A') == 'INHERIT_A':
        need([p['scene_ref'] for p in historical['a_continuations']] == list(SCENES[:3]), 'HISTORICAL_A_PLAN_SCOPE')
    else:
        need(row['a_plan_mode'] == historical.get('plan_mode') == 'NEW_SELECTION'
             and historical['a_continuations'] == [], 'NEW_SELECTION_POLICY_REQUIRED')
    return row, role, unit, cp, constraints, historical


def historical_b(row):
    """Return only accepted retained B slots, including untouched old reservation."""
    if not any(':B:' in scene for scene in row['retained_scene_refs']):
        return []
    oldunit = Path(read(Path(row['unit_path']) / 'CONSTRAINTS_PATH_ADAPTATION.json')['original_binding']['path']).parents[1]
    reservations = read(oldunit / 'SOURCE_RESERVATIONS.json')
    found = []
    for scene in row['retained_scene_refs']:
        if ':B:' not in scene: continue
        rows = [a for a in reservations['allocation'] if a['scene_ref'] == scene and a['root_id'] == row['root_id']]
        need(len(rows) == 1, 'RETAINED_B_HISTORICAL_RESERVATION_MISSING')
        a = rows[0]
        verify({'path': a['source_item_path'], 'sha256': a['source_item_sha256']})
        found.append({'scene_ref': scene, 'allocation': a,
                      'historical_reservations_binding': bind(oldunit / 'SOURCE_RESERVATIONS.json')})
    return found


def matching_inputs(row, unit, cp, constraints, historical):
    sourcep = Path(row['candidate_prefix_binding']['path'])
    source = read(sourcep)
    poolp = unit / 'inputs/EVENT_POOL.json'
    retained = historical_b(row)
    paths = [sourcep, unit / 'inputs/BASELINE.json', cp, unit / 'A_PLAN_POLICY.json',
             unit / 'PREFIX_IMPORT.json', poolp, B / 'ROLE_ASSIGNMENTS.json',
             B / 'evidence/CONTENT_CONTRACT.md', B / 'evidence/RETRY_EXECUTION_POLICY.json',
             B / 'evidence/SOURCE_POOL_FREEZE.json', Path(__file__)]
    for old in retained:
        paths += [Path(old['allocation']['source_item_path']), Path(old['historical_reservations_binding']['path'])]
    paths = list(dict.fromkeys(paths))
    chunks = ['FULL_APPROVED_CUMULATIVE_PREFIX', *[t['turn_ref'] + ' | ' + t['speaker'] + ': ' + t['text'] for t in source['turns']],
              'ALL_CONTINUATION_CONSTRAINTS', json.dumps(constraints, ensure_ascii=False),
              'A_PLAN_POLICY_AND_MODE', json.dumps(historical, ensure_ascii=False),
              'RETAINED_B_ALLOCATION_AND_COMPLETE_NATIVE_EVENTS', json.dumps([
                  {**r, 'native_event': read(r['allocation']['source_item_path'])} for r in retained], ensure_ascii=False),
              'COMPLETE_DISJOINT_NEW_NATIVE_CANDIDATES', json.dumps(read(poolp), ensure_ascii=False),
              'RETRY_EXECUTION_POLICY', (B / 'evidence/RETRY_EXECUTION_POLICY.json').read_text(),
              'CONTENT_CONTRACT_INHERITED', (B / 'evidence/CONTENT_CONTRACT.md').read_text()]
    return paths, '\n'.join(chunks) + '\n', source


def match_read(root, executor):
    row, role, unit, cp, constraints, historical = info(root, executor)
    need(not (unit / 'selection.json').exists(), 'MATCHING_ALREADY_FINISHED')
    folder = unit / 'evidence/MATCHING'
    paths, rendered, source = matching_inputs(row, unit, cp, constraints, historical)
    render = folder / 'FULL_INPUT.txt'
    render.parent.mkdir(parents=True, exist_ok=True)
    with render.open('x', encoding='utf-8') as f: f.write(rendered)
    save(folder / 'READ_CONTEXT.json', {'started_at_utc': now(), 'executor_id': executor, 'root_id': root,
         'input_bindings': [bind(p) for p in paths], 'input_render_binding': bind(render),
         'master_freeze_binding': bind(B / 'MASTER_FREEZE.json'),
         'full_source_turn_refs': [t['turn_ref'] for t in source['turns']],
         'required_new_b_scene_refs': row['new_b_scene_refs'],
         'full_source_read_is_an_executor_attestation': True})
    print(render.read_text())


def validate_new_plans(plans, source):
    need(source is not None, 'NEW_SELECTION_FULL_SOURCE_REQUIRED')
    need(isinstance(plans, list) and [p.get('scene_ref') for p in plans] == list(SCENES[:3]), 'NEW_A_PLAN_SCENE_SCOPE')
    turns = {t['turn_ref']: t for t in source['turns']}
    r01 = {t.get('scene_ref') or t.get('session_ref') for t in source['turns']}
    three = all('R01:S%02d' % i in r01 for i in range(1, 4))
    for i, plan in enumerate(plans, 1):
        need(set(plan) == {'scene_ref', 'baseline_scene_ref', 'thread', 'evidence'}, 'NEW_A_PLAN_FIELDS')
        need(isinstance(plan['thread'], str) and plan['thread'].strip(), 'NEW_A_THREAD_REQUIRED')
        need(plan['baseline_scene_ref'] in r01, 'NEW_A_BASELINE_SCENE_UNKNOWN')
        need(isinstance(plan['evidence'], list) and plan['evidence'], 'NEW_A_EVIDENCE_REQUIRED')
        for cite in plan['evidence']:
            need(set(cite) == {'turn_ref', 'speaker', 'quote'} and cite['turn_ref'] in turns, 'NEW_A_CITATION_FIELDS')
            turn = turns[cite['turn_ref']]
            need(cite['speaker'] == turn['speaker'] and isinstance(cite['quote'], str)
                 and cite['quote'].strip() and cite['quote'] in turn['text'], 'NEW_A_CITATION_MISMATCH')
        if three:
            expected = 'R01:S%02d' % i
            need(plan['baseline_scene_ref'] == expected, 'NEW_A_FIXED_R01_ALIGNMENT')
            need(any((turns[c['turn_ref']].get('scene_ref') or turns[c['turn_ref']].get('session_ref')) == expected
                     for c in plan['evidence']), 'NEW_A_FIXED_R01_EVIDENCE')


def validate_payload(raw, row, historical, pool, consumed, source=None):
    need(set(raw) == {'read_attestation', 'a_continuations', 'allocations'}, 'MATCHING_PAYLOAD_FIELDS')
    att = raw['read_attestation']
    for k in ['full_source_read', 'all_continuation_constraints_read', 'all_candidate_events_fully_read', 'retained_native_events_fully_read']:
        need(att.get(k) is True, 'MATCHING_READ_ATTESTATION:' + k)
    if row.get('a_plan_mode', 'INHERIT_A') == 'NEW_SELECTION':
        validate_new_plans(raw['a_continuations'], source)
    else:
        need(raw['a_continuations'] == historical['a_continuations'], 'INHERITED_A_PLAN_CHANGED')
    allocations = raw['allocations']
    need([a['scene_ref'] for a in allocations] == row['new_b_scene_refs'], 'ONLY_EXACT_NEW_B_SLOTS_REQUIRED')
    ids = [a['source_id'] for a in allocations]
    need(len(set(ids)) == len(ids), 'DUPLICATE_NEW_SOURCE')
    for a in allocations:
        need(set(a) == {'scene_ref', 'source_id', 'compatibility_reason', 'main_purpose'}, 'NEW_ALLOCATION_FIELDS')
        need(a['source_id'] in pool and a['source_id'] not in consumed, 'SOURCE_UNAVAILABLE_OR_CONSUMED')
        need(all(isinstance(a[k], str) and a[k].strip() for k in ['compatibility_reason', 'main_purpose']), 'COMPATIBILITY_REASON_REQUIRED')
    return allocations


def match_save(root, executor, payload):
    row, role, unit, cp, constraints, historical = info(root, executor)
    folder = unit / 'evidence/MATCHING'
    context = read(folder / 'READ_CONTEXT.json')
    need(context['executor_id'] == executor and context['root_id'] == root, 'WRONG_MATCHING_READER')
    paths, rendered, source = matching_inputs(row, unit, cp, constraints, historical)
    need(context['input_bindings'] == [bind(p) for p in paths], 'MATCHING_EXACT_INPUT_SET_REQUIRED')
    need((folder / 'FULL_INPUT.txt').read_text(encoding='utf-8') == rendered, 'MATCHING_RENDER_CHANGED_OR_INCOMPLETE')
    need(context['input_render_binding'] == bind(folder / 'FULL_INPUT.txt'), 'MATCHING_RENDER_BINDING_REQUIRED')
    need(context['master_freeze_binding'] == bind(B / 'MASTER_FREEZE.json'), 'MATCHING_MASTER_BINDING_REQUIRED')
    for item in [*context['input_bindings'], context['input_render_binding'], context['master_freeze_binding']]: verify(item)
    need(context['required_new_b_scene_refs'] == row['new_b_scene_refs'], 'NEW_B_SCOPE_CHANGED')
    source = read(row['candidate_prefix_binding']['path'])
    need(context['full_source_turn_refs'] == [t['turn_ref'] for t in source['turns']], 'MATCHING_READ_COVERAGE')
    pool = {'multiwoz22:train:' + e['dialogue_id']: e for e in read(unit / 'inputs/EVENT_POOL.json')['events']}
    consumed = set(read(B / 'evidence/SOURCE_POOL_FREEZE.json')['consumed_source_ids'])
    for p in B.glob('units/*/SOURCE_RESERVATIONS.json'):
        consumed.update(a['source_id'] for a in read(p)['allocation'])
    raw = read(payload)
    allocations = validate_payload(raw, row, historical, pool, consumed, read(unit / 'inputs/BASELINE.json'))
    retained = historical_b(row)
    ids = [a['source_id'] for a in allocations]
    need(not set(ids).intersection(a['allocation']['source_id'] for a in retained), 'RETAINED_SOURCE_REALLOCATED')
    need(not any((unit / name).exists() for name in ['selection.json', 'SOURCE_RESERVATIONS.json', 'RUN_CONFIG.json', 'EXECUTION_INPUTS_FROZEN.json']), 'MATCHING_OUTPUT_ALREADY_EXISTS')
    copy(payload, folder / 'MATCHING_PAYLOAD.json')
    selected = {'root_id': root, 'selector': executor, 'started_at_utc': context['started_at_utc'], 'finished_at_utc': now(),
        'full_baseline_read': True, 'full_source_read': True, 'all_continuation_constraints_read': True,
        'baseline_sha256': sha(unit / 'inputs/BASELINE.json'), 'source_binding': row['candidate_prefix_binding'],
        'turn_refs_read': context['full_source_turn_refs'], 'constraints_binding': bind(cp),
        'continuation_constraints': constraints['continuation_constraints'],
        'all_candidate_native_events_fully_read': True, 'event_pool_binding': bind(unit / 'inputs/EVENT_POOL.json'),
        'input_context_binding': bind(folder / 'READ_CONTEXT.json'), 'raw_payload_binding': bind(folder / 'MATCHING_PAYLOAD.json'),
        'inherited_selection_binding': bind(unit / 'A_PLAN_POLICY.json'),
        'a_continuations': raw['a_continuations'], 'a_plan_mode': row.get('a_plan_mode', 'INHERIT_A'),
        'allocations': [], 'semantic_review_is_programmatic': False}
    oldmap = {r['scene_ref']: r for r in retained}
    newmap = {a['scene_ref']: a for a in allocations}
    for scene in SCENES[3:]:
        if scene in oldmap:
            old = oldmap[scene]
            selected['allocations'].append({**old['allocation'], 'allocation_kind': 'HISTORICAL_PREFIX_REUSE_NO_NEW_CONSUMPTION',
                 'historical_reservations_binding': old['historical_reservations_binding']})
        else:
            a = newmap[scene]; event = pool[a['source_id']]
            selected['allocations'].append({**a, 'root_id': root, 'native_turn_ids_read': [t['turn_id'] for t in event['turns']],
                'full_event_utterances_read': True, 'allocation_kind': 'NEW_PERMANENT_RESERVATION'})
    save(unit / 'selection.json', selected)
    reservations = []
    for a in selected['allocations']:
        scene = a['scene_ref']; target = unit / 'inputs' / (scene.removeprefix('R02:').replace(':', '_') + '_EVENT.json')
        if scene in oldmap:
            copy(oldmap[scene]['allocation']['source_item_path'], target)
        else:
            save(target, pool[a['source_id']])
        reservations.append({**a, 'source_item_path': str(target), 'source_item_sha256': sha(target),
            'selection_path': str(unit / 'selection.json'), 'selection_sha256': sha(unit / 'selection.json'),
            'status': 'HISTORICAL_PREFIX_REUSE_NO_NEW_CONSUMPTION' if scene in oldmap else 'PERMANENTLY_RESERVED_BEFORE_AUTHORING'})
    save(unit / 'SOURCE_RESERVATIONS.json', {'batch_id': B.name, 'root_id': root, 'created_at_utc': now(), 'allocation': reservations,
        'new_consumption_count': len(allocations), 'permanent_including_failed_and_unexecuted_slots': True})
    cfg = {'controller_version': 'codex_sequential_ab_v1', 'batch_id': row.get('candidate_namespace', B.name), 'created_at_utc': now(),
        'roots': [{'root_id': root, 'baseline_path': str(unit / 'inputs/BASELINE.json'), 'baseline_sha256': sha(unit / 'inputs/BASELINE.json')}],
        'scene_sequence': list(SCENES), 'scope_scene_refs': list(SCENES), 'new_turn_count': {'min': 12, 'max': 16},
        'required_rules': ['time', 'location', 'actor_identity', 'plan_vs_completion', 'cross_scene_continuity', 'semantic_repetition', 'naturalness', 'event_fidelity'],
        'seed_namespace': B.name, 'multiselect_target_cardinalities': [2, 3, 4, 5], 'gold6_control_per_ready_root': 1,
        'ordering_slots_per_ready_root': 2, 'author_attempts_per_scene': 1, 'qa_author_attempts_per_complete_root': 1,
        'source_replacements': 0, 'semantic_repairs': 0, 'backend': 'CURRENT_CODEX_SESSION', 'exact_model_id': None,
        'training_run': False, 'solver_run': True, 'solver_protocol': 'v15_full_only_random_mcq6x3_order12x1_v1',
        'selection': 'Complete-valid 0 < equal-display partial mean < 1; Gold6 excluded',
        'failure_policy': 'Stop later dependent scenes; retain raw first draft, HOLD, errors and denominator.',
        'retry_start': row['retry_start'], 'historical_prefix_import_binding': bind(unit / 'PREFIX_IMPORT.json'),
        'baseline_gate_binding': bind(cp), 'master_manifest_binding': bind(B / 'MANIFEST.json')}
    save(unit / 'RUN_CONFIG.json', cfg)
    checked = load_config(unit); load_baseline(unit, checked, root)
    for scene in SCENES[3:]: reservation_for(unit, checked, root, scene)
    inputs = [unit / n for n in ['RUN_CONFIG.json', 'SOURCE_RESERVATIONS.json', 'selection.json', 'PREFIX_IMPORT.json', 'A_PLAN_POLICY.json']]
    inputs += sorted((unit / 'inputs').glob('*')) + [folder / n for n in ['FULL_INPUT.txt', 'READ_CONTEXT.json', 'MATCHING_PAYLOAD.json']]
    save(unit / 'EXECUTION_INPUTS_FROZEN.json', {'created_at_utc': now(), 'root_id': root, 'status': 'READY_FOR_FIXED_REMAINING_SCENES',
        'master_freeze_binding': bind(B / 'MASTER_FREEZE.json'), 'files': [bind(p) for p in inputs],
        'author': executor, 'reviewer': role['reviewer'], 'generation_started': False})
    save(folder / 'SAVE_RECEIPT.json', {'root_id': root, 'actual_executor': executor, 'saved_at_utc': now(),
        'raw_payload_binding': bind(folder / 'MATCHING_PAYLOAD.json'), 'selection_binding': bind(unit / 'selection.json'),
        'reservations_binding': bind(unit / 'SOURCE_RESERVATIONS.json'), 'execution_freeze_binding': bind(unit / 'EXECUTION_INPUTS_FROZEN.json'),
        'new_source_ids_consumed': ids, 'retained_source_ids_reused': [a['allocation']['source_id'] for a in retained]})
    print(json.dumps({'root_id': root, 'status': 'INPUTS_FROZEN', 'new_events_reserved': ids,
        'retained_b_count': len(retained), 'generation_started': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['match-read', 'match-save'])
    parser.add_argument('--root', required=True); parser.add_argument('--executor', required=True)
    parser.add_argument('--payload', type=Path)
    args = parser.parse_args()
    if args.action == 'match-read': match_read(args.root, args.executor)
    else:
        need(args.payload, 'PAYLOAD_REQUIRED'); match_save(args.root, args.executor, args.payload)
