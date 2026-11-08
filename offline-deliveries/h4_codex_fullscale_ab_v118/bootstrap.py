"""Freeze the complete training-root population; no semantic reads or generation."""
from pathlib import Path
from datetime import datetime, timezone
import collections
import hashlib
import json

B = Path(__file__).resolve().parent
P = B.parent.parent
D = P / 'SP-820-publish/first_training_pipeline/data'
WORKERS = ['/root/v113_setup', '/root/v113_question_recheck', '/root/v113_14304_recheck']
REUSED = {
    'h4_codex_sequential_ab_pilot_v112': [222937],
    'h4_codex_serialized_ab_validation_v113': [761, 9557, 14304, 20432],
    'h4_codex_newroots_ab_validation_v115': [29753, 31457, 45849, 55788],
    'h4_codex_screened_ab_validation_v117': [57679, 68549, 75223, 77882],
}
VALIDATION = [19053, 19708, 24719, 25652, 78203, 87964, 94289, 225575]


def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p): return {'path': str(Path(p).resolve()), 'sha256': sha(p)}
def write(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f: json.dump(d, f, ensure_ascii=False, indent=2); f.write('\n')
def copy(p, q):
    q.parent.mkdir(parents=True, exist_ok=True)
    with q.open('xb') as f: f.write(Path(p).read_bytes())
def lines(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def strings(d):
    if isinstance(d, str): yield d
    elif isinstance(d, list):
        for v in d: yield from strings(v)
    elif isinstance(d, dict):
        for v in d.values(): yield from strings(v)


def main():
    assert not (B / 'MANIFEST.json').exists()
    train = D / 'matched_training/m_clean_policy_training_manifest.jsonl'
    dev = D / 'matched_training/dev_full_eval_manifest.jsonl'
    test = D / 'matched_training/test_full_eval_manifest.jsonl'
    split = {k: {r['root_id'] for r in lines(p)} for k, p in [('train', train), ('dev', dev), ('test', test)]}
    assert len(split['train']) == 68
    inventory_path = D / 'qa_generation_only/manifest.json'
    inventory = {r['root_id']: r for r in read(inventory_path)['roots']}
    originals_path = D / 'selection_manifest_h4_unexpanded122.json'
    originals = {r['root_id']: r for r in read(originals_path)['records']}
    metadata = {}
    original_bindings = []
    for rid in set.union(*split.values()):
        row = originals[rid]
        assert sha(row['input_path']) == row['input_sha256']
        source = read(row['input_path'])['source']
        metadata[rid] = {'source_id': source['source_id'], 'source_record_sha256': source['source_record_sha256']}
        original_bindings.append(bind(row['input_path']))
    isolation = []
    for left, right in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
        overlaps = {'root_id': sorted(split[left] & split[right])}
        for key in ['source_id', 'source_record_sha256']:
            overlaps[key] = sorted({metadata[r][key] for r in split[left]} & {metadata[r][key] for r in split[right]})
        assert not any(overlaps.values()), overlaps
        isolation.append({'left': left, 'right': right, 'overlaps': overlaps})
    assert len({metadata[r]['source_id'] for r in split['train']}) == 68
    assert len({metadata[r]['source_record_sha256'] for r in split['train']}) == 68
    screen_path = B.parent / 'h4_codex_baseline_screen_v116/SCREEN_RESULTS.json'
    screened = {r['root_id']: r for r in read(screen_path)['roots']}
    reuse_by_id = {f'cc-train-episode-{n}': name for name, nums in REUSED.items() for n in nums}
    root_rows, assignments = [], []
    pending_index = 0
    for rid in sorted(split['train'], key=lambda r: int(r.rsplit('-', 1)[1])):
        num = rid.rsplit('-', 1)[1]
        source_path = Path(inventory[rid]['source_path'])
        assert sha(source_path) == inventory[rid]['source_sha256']
        # Parse only structure here; text does not participate in selection.
        source = read(source_path)
        turns = source['turns']
        scenes = sorted({t['scene_ref'] for t in turns if t.get('scene_ref')})
        speakers = sorted({t['speaker'] for t in turns})
        assert len(speakers) >= 2 and len({t['turn_ref'] for t in turns}) == len(turns)
        unit = B / 'units' / ('root' + num)
        target = unit / 'inputs/BASELINE.json'
        copy(source_path, target)
        row = {'root_id': rid, 'unit_path': str(unit), 'baseline_path': str(target),
               'baseline_sha256': sha(target), 'baseline_binding': bind(target),
               'upstream_binding': bind(source_path), 'baseline_turn_count': len(turns),
               'baseline_speakers': speakers, 'baseline_scene_refs': scenes,
               'original_source': metadata[rid], 'cohort': 'standard' if len(speakers) == 2 and scenes == ['R01:S01', 'R01:S02', 'R01:S03'] else 'existing_speakers_or_scene_count_adaptation',
               'wave': 'validation' if int(num) in VALIDATION else 'production'}
        if rid in reuse_by_id:
            old = B.parent / reuse_by_id[rid]
            folder = old / 'roots' / ('root' + num)
            cross = read(folder / 'FINAL_CROSSCHECK.json')
            assert len(cross['results']) == 7 and all(r['status'] == 'CONTENT_PASS' for r in cross['results'])
            assert read(folder / 'CUMULATIVE_SOURCE.json')['turns'][:len(turns)] == turns
            row.update(action='REUSE', reuse_batch=str(old), reuse_bindings=[bind(old / 'FINAL_STATUS.json'), bind(folder / 'CUMULATIVE_SOURCE.json'), bind(folder / 'FINAL_CROSSCHECK.json')])
            if num == '20432':
                row['mandatory_candidate_override'] = bind(B.parent / 'h4_codex_wording_minfix_v114/RECOMMENDED_CANDIDATES.json')
                row['historical_content_pass_not_current_candidate_approval'] = True
        elif rid in screened and screened[rid]['decision'] == 'HOLD':
            row.update(action='HISTORICAL_HOLD', inherited_screen_binding=bind(screen_path), inherited_reviews=screened[rid]['reviews'], historical_decision='HOLD')
        else:
            author = WORKERS[pending_index % 3]
            reviewer = WORKERS[(pending_index + 1) % 3]
            baseline_reviewers = [reviewer, WORKERS[(pending_index + 2) % 3]]
            pending_index += 1
            row.update(action='INHERIT_SCREEN' if rid in screened and screened[rid]['eligible'] else 'SCREEN',
                       author=author, reviewer=reviewer, baseline_reviewers=baseline_reviewers)
            if row['action'] == 'INHERIT_SCREEN':
                assert screened[rid]['baseline_sha256'] == row['baseline_sha256']
                row.update(inherited_screen_binding=bind(screen_path), inherited_reviews=screened[rid]['reviews'])
                # These are historical actual readers, not the new scene roles.
                row['baseline_reviewers'] = [read(r['path'])['reviewer'] for r in row['inherited_reviews']]
            assignments.append({k: row[k] for k in ['root_id', 'author', 'reviewer', 'baseline_reviewers']})
        root_rows.append(row)
    assert collections.Counter(r['action'] for r in root_rows) == {'REUSE': 13, 'HISTORICAL_HOLD': 2, 'INHERIT_SCREEN': 2, 'SCREEN': 51}
    assert {int(r['root_id'].rsplit('-', 1)[1]) for r in root_rows if r['wave'] == 'validation'} == set(VALIDATION)
    # Protect the full preceding delivery and its inherited immutable history.
    previous = B.parent / 'h4_codex_screened_ab_validation_v117'
    protection = {r['path']: r['sha256'] for r in read(previous / 'evidence/HISTORICAL_PROTECTION.json')['files']}
    for name in [*REUSED, 'h4_codex_baseline_screen_v116', 'h4_codex_wording_minfix_v114']:
        for path in (B.parent / name).rglob('*'):
            if path.is_file(): protection[str(path)] = sha(path)
    for path, expected in protection.items(): assert sha(path) == expected, path
    write(B / 'evidence/HISTORICAL_PROTECTION.json', {'created_at_utc': now(), 'files': [{'path': p, 'sha256': s} for p, s in sorted(protection.items())]})
    # Every reservation, including failed/unexecuted slots, remains consumed.
    consumed, consumption = set(), []
    for p in sorted(P.rglob('SOURCE_RESERVATIONS.json')):
        if B in p.parents: continue
        ids = sorted({s for s in strings(read(p)) if s.startswith('multiwoz22:train:')})
        consumed.update(ids)
        consumption.append({**bind(p), 'source_ids': ids})
    legacy = B.parent / 'h4_codex_controlled_batch_v105/evidence/SOURCE_SELECTION.json'
    consumed.update(read(legacy)['consumed_source_ids'])
    poolp = B.parent / 'h4_external_event_source_refresh_v1/multiwoz_train_sample.json'
    pool = read(poolp)
    available = [e for e in pool if 'multiwoz22:train:' + e['dialogue_id'] not in consumed]
    assert len(available) == 456
    # Stratify by service metadata, distribute disjoint candidates without text matching.
    groups = collections.defaultdict(list)
    for e in available: groups[tuple(e['services'])].append(e)
    partitions = [[] for _ in assignments]
    cursor = 0
    for service in sorted(groups):
        for event in sorted(groups[service], key=lambda e: e['dialogue_id']):
            partitions[cursor % len(partitions)].append(event); cursor += 1
    partition_rows = []
    for assigned, events in zip(assignments, partitions):
        assert len(events) >= 3
        rid = assigned['root_id']; unit = B / 'units' / ('root' + rid.rsplit('-', 1)[1])
        dest = unit / 'inputs/EVENT_POOL.json'
        write(dest, {'root_id': rid, 'events': events, 'source_pool_binding': bind(poolp), 'content_semantically_read': False, 'events_reserved': False})
        partition_rows.append({'root_id': rid, 'event_pool_binding': bind(dest), 'event_count': len(events)})
    write(B / 'evidence/SOURCE_POOL_FREEZE.json', {'created_at_utc': now(), 'source_pool_binding': bind(poolp), 'available_count': len(available), 'consumed_source_ids': sorted(consumed), 'consumption_files': consumption, 'legacy_binding': bind(legacy), 'partitions': partition_rows, 'disjoint_and_complete': True, 'source_matching_started': False})
    write(B / 'ROLE_ASSIGNMENTS.json', {'schema_version': 'actual_session_agent_assignments_v1', 'roots': assignments, 'executor_identity_is_declared_not_cryptographically_authenticated': True, 'same_inherited_model': True})
    write(B / 'MANIFEST.json', {'schema_version': 'codex_fullscale_ab_population_v1', 'batch_id': B.name, 'created_at_utc': now(), 'population': 'All 68 distinct clean-policy training roots; never dev/test', 'training_manifest': bind(train), 'dev_manifest': bind(dev), 'test_manifest': bind(test), 'upstream_inventory': bind(inventory_path), 'originals_manifest': bind(originals_path), 'original_bindings': original_bindings, 'split_isolation': isolation, 'population_denominator': 68, 'reused_complete_roots': 13, 'historical_hold_roots': 2, 'new_fixed_attempt_roots': 53, 'new_baseline_screen_roots': 51, 'inherited_eligible_screen_roots': 2, 'validation_root_ids': [f'cc-train-episode-{n}' for n in VALIDATION], 'roots': root_rows, 'new_training_authorized_or_started': False, 'training_pool_merge_authorized_or_started': False, 'full_completion_definition': 'All fixed population entries have explicit provenance and terminal disposition; held roots are not expanded or counted as successful.', 'generation_started': False})
    print(json.dumps({'status': 'POPULATION_PREPARED_NOT_YET_MASTER_FROZEN', 'roots': len(root_rows), 'actions': dict(collections.Counter(r['action'] for r in root_rows)), 'partitions': len(partitions), 'history_files': len(protection)}, ensure_ascii=False))


if __name__ == '__main__': main()
