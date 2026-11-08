"""Read-only evidence audit by the actual reviewer; does not certify semantics."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json

B = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('v133_frontier_audit_workflow', B / 'retry_workflow.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
WHO = '/root/frontier_quality'
ROOTS = (147355, 229160)


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def binding(row, base=None):
    p = Path(row['path'])
    if not p.is_absolute():
        assert base is not None
        p = base / p
    assert sha(p) == row['sha256'], ('BINDING_CHANGED', str(p))
    if 'bytes' in row:
        assert p.stat().st_size == row['bytes']
    return str(p)


def citations(obj, source):
    ts = {t['turn_ref']: t for t in source['turns']}
    count = 0

    def visit(x):
        nonlocal count
        if isinstance(x, dict):
            if {'turn_ref', 'speaker', 'quote'} <= x.keys():
                t = ts[x['turn_ref']]
                assert x['speaker'] == t['speaker']
                assert x['quote'] and x['quote'] in t['text']
                count += 1
            for value in x.values():
                visit(value)
        elif isinstance(x, list):
            for value in x:
                visit(value)

    visit(obj)
    return count


def context_check(p, person, purpose, scene, source_path):
    record = read(p)
    source = read(source_path)
    assert record['actual_reader'] == person and record['purpose'] == purpose
    assert record['scene_ref'] == scene
    assert Path(record['source_path']) == source_path
    assert record['source_sha256'] == sha(source_path)
    assert record['turn_refs'] == [t['turn_ref'] for t in source['turns']]
    text = '\n'.join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in source['turns'])
    for item in record['inputs']:
        q = Path(binding(item))
        if q == source_path:
            continue
        if q.name.endswith('_EVENT.json'):
            e = read(q)
            text += '\n\nNATIVE EVENT ' + e['dialogue_id'] + '\n' + '\n'.join(
                f"[{t['turn_id']}] {t['speaker']}: {t['utterance']}" for t in e['turns'])
        else:
            text += '\n\n' + q.name + '\n' + q.read_text()
    assert p.with_name('FULL_INPUT.txt').read_text() == text + '\n'
    return record


def audit_root(rid):
    u = w.unit(rid)
    config = w.guard(rid, WHO, 'reviewer')
    assert config['split'] == 'dev' and config['author'] != WHO
    assert all(config[k] is False for k in ['solver_run', 'training_run', 'question_generation', 'score_based_selection', 'baseline_rewrite'])
    w.check_execution(rid)
    execution = read(u / 'EXECUTION_INPUTS_FROZEN.json')
    assert all(binding(item) for item in execution['files'])
    baseline_path = u / 'inputs/BASELINE.json'
    base = read(baseline_path)
    constraints = read(u / 'inputs/HISTORICAL_CONTINUATION_CONSTRAINTS.json')
    assert constraints['decision'] == 'HOLD'
    old_base = Path(binding(constraints['baseline_binding']))
    assert old_base.read_bytes() == baseline_path.read_bytes()
    for item in constraints['review_bindings'] + constraints['input_bindings']:
        binding(item)
    old_pool = old_base.with_name('EVENT_POOL.json')
    assert old_pool.read_bytes() == (u / 'inputs/EVENT_POOL.json').read_bytes()
    plan = read(u / 'SAFE_PLAN.json')
    plan_review = read(u / 'SAFE_PLAN_REVIEW.json')
    assert plan['author'] == config['author'] and plan['root_id'] == w.ridfull(rid)
    assert plan_review['reviewer'] == WHO and plan_review['decision'] == 'READY'
    assert plan_review['safe_plan_sha256'] == sha(u / 'SAFE_PLAN.json')
    assert plan['baseline_sha256'] == plan_review['baseline_sha256'] == sha(baseline_path)
    assert plan['baseline_conflicts_resolved'] is False
    assert plan_review['baseline_conflicts_resolved'] is False
    assert plan['quarantined_conflicts'] and plan['forbidden_claims']
    citations(plan, base)
    citations(plan_review, base)
    for role in ('author', 'reviewer'):
        person = config[role]
        context_check(u / 'reads' / ('baseline_plan_' + role) / 'READ_CONTEXT.json', person, 'plan_' + role, None, baseline_path)
    reservations = read(u / 'SOURCE_RESERVATIONS.json')
    rows = reservations['allocation']
    assert reservations['permanent_including_failed_and_unexecuted_slots'] is True
    assert [r['scene_ref'] for r in rows] == list(w.SCENES[3:])
    assert len({r['source_id'] for r in rows}) == 3
    pool = {'multiwoz22:train:' + e['dialogue_id']: e for e in read(u / 'inputs/EVENT_POOL.json')['events']}
    assert len(pool) == 8
    consumed = set(read(B / 'evidence/SOURCE_CONSUMPTION_FREEZE.json')['consumed_source_ids'])
    for row in rows:
        assert row['source_id'] not in consumed
        event_path = Path(row['source_item_path'])
        assert sha(event_path) == row['source_item_sha256']
        assert read(event_path) == pool[row['source_id']]
        assert row['native_turn_ids_read'] == [t['turn_id'] for t in read(event_path)['turns']]
        assert row['status'] == 'PERMANENTLY_RESERVED_BEFORE_AUTHORING'
    states = w.status(rid)
    assert states['full_expansion_complete'] and states['ready_scene_count'] == 6
    assert not states['root_stopped']
    previous_path = baseline_path
    previous = base['turns']
    previous_ready = stamp(plan_review['created_at_utc'])
    scenes = []
    for scene, state in zip(w.SCENES, states['scenes']):
        folder = w.native.scene_folder(u, w.ridfull(rid), scene)
        draft_path = folder / 'EXPANSION_DRAFT.json'
        source_path = folder / 'CUMULATIVE_SOURCE.json'
        review_path = folder / 'SOURCE_REVIEW.json'
        d, source, review = map(read, (draft_path, source_path, review_path))
        assert state['status'] == review['decision'] == 'READY'
        assert source['turns'] == previous + d['turns']
        assert source['input_source_sha256'] == d['input_source_sha256'] == sha(previous_path)
        assert source['draft_sha256'] == sha(draft_path)
        assert source['baseline_sha256'] == sha(baseline_path)
        assert len({t['turn_ref'] for t in source['turns']}) == len(source['turns'])
        assert 12 <= len(d['turns']) <= 16
        for n, t in enumerate(d['turns'], 1):
            assert t['turn_ref'] == f'{scene}:T{n:03d}'
            assert t['scene_ref'] == t['session_ref'] == scene
            assert t['speaker'] in {s['speaker'] for s in base['turns']}
            assert t['text'].strip() and not any('\u3400' <= c <= '\u9fff' for c in t['text'])
        payload = read(folder / 'AUTHOR_PAYLOAD.json')
        assert payload['messages'] == [{'speaker': t['speaker'], 'text': t['text']} for t in d['turns']]
        author_receipt = read(folder / 'AUTHOR_RECEIPT.json')
        assert author_receipt['actual_author'] == d['author'] == config['author']
        assert author_receipt['payload_sha256'] == sha(folder / 'AUTHOR_PAYLOAD.json')
        assert review['reviewer'] == WHO and review['source_sha256'] == sha(source_path)
        assert review['full_source_read'] is True
        assert review['full_event_read'] == (':B:' in scene)
        assert review['baseline_conflicts_resolved'] is False
        assert review['turn_refs_read'] == [t['turn_ref'] for t in source['turns']]
        assert w.CHECKS <= review['checks'].keys()
        assert all(isinstance(f, dict) and f.get('severity') != 'BLOCKING' for f in review['findings'])
        for name in w.CHECKS:
            check = review['checks'][name]
            assert check['status'] in {'PASS', 'WARNING', 'PASS_WITH_WARNING'}
            assert check['reason'] and check['citations']
        count = citations(review, source)
        recs = [read(p) for p in sorted((folder / 'receipts').glob('*.json'))]
        assert len(recs) == 2 and {r['status'] for r in recs} == {'STRUCTURAL_PASS', 'REVIEW_READY'}
        structural = next(r for r in recs if r['status'] == 'STRUCTURAL_PASS')
        accepted = next(r for r in recs if r['status'] == 'REVIEW_READY')
        assert structural['source_sha256'] == accepted['source_sha256'] == sha(source_path)
        assert structural['draft_sha256'] == sha(draft_path)
        assert structural['input_source_sha256'] == sha(previous_path)
        assert structural['run_config_sha256'] == sha(u / 'RUN_CONFIG.json')
        assert structural['source_reservations_sha256'] == sha(u / 'SOURCE_RESERVATIONS.json')
        assert accepted['review_sha256'] == sha(review_path) and accepted['reviewer'] == WHO
        author_read = context_check(u / 'reads' / (scene.replace(':', '_') + '_author') / 'READ_CONTEXT.json', config['author'], 'author', scene, previous_path)
        reviewer_read = context_check(u / 'reads' / (scene.replace(':', '_') + '_reviewer') / 'READ_CONTEXT.json', WHO, 'reviewer', scene, source_path)
        assert previous_ready <= stamp(author_read['started_at_utc']) <= stamp(structural['created_at_utc'])
        assert stamp(structural['created_at_utc']) <= stamp(reviewer_read['started_at_utc']) <= stamp(review['created_at_utc']) <= stamp(accepted['created_at_utc'])
        assert stamp(reservations['created_at_utc']) <= stamp(author_read['started_at_utc'])
        scenes.append({'scene_ref': scene, 'status': 'PASS', 'new_turn_count': len(d['turns']), 'cumulative_turn_count': len(source['turns']), 'source_sha256': sha(source_path), 'review_sha256': sha(review_path), 'exact_citations_checked': count, 'predecessor_ready_before_author_read': True, 'warnings': {k: v['reason'] for k, v in review['checks'].items() if v['status'] != 'PASS'}})
        previous_path, previous, previous_ready = source_path, source['turns'], stamp(accepted['created_at_utc'])
    final = w.native.root_folder(u, w.ridfull(rid)) / 'CUMULATIVE_SOURCE.json'
    assert final.read_bytes() == previous_path.read_bytes()
    return {'root_id': w.ridfull(rid), 'status': 'PASS', 'baseline_turn_count': len(base['turns']), 'new_turn_count': len(previous) - len(base['turns']), 'final_turn_count': len(previous), 'final_source_sha256': sha(final), 'old_baseline_and_reviews_unchanged': True, 'original_eight_event_pool_identical': True, 'native_event_ids': [r['source_id'] for r in rows], 'scenes': scenes}


def schema_recovery():
    f = w.native.scene_folder(w.unit(229160), w.ridfull(229160), 'R02:B:S02')
    old, new = read(f / 'SOURCE_REVIEW.INVALID_SCHEMA_01.json'), read(f / 'SOURCE_REVIEW.json')
    correction, acceptance = read(f / 'SCHEMA_CORRECTION_01.json'), read(f / 'SCHEMA_CORRECTION_01_ACCEPTED.json')
    old_sha, new_sha = sha(f / 'SOURCE_REVIEW.INVALID_SCHEMA_01.json'), sha(f / 'SOURCE_REVIEW.json')
    assert old_sha == correction['original_sha256'] == acceptance['original_archive_sha256'] == '94be7f84714a08540caa5d41d9146c56725c9110b6a887c5c114695d952e9aa2'
    assert new_sha == acceptance['accepted_review_sha256'] == '58b07fe4b17b419dd387aaf0d9c539e27e5fa49691ee74e9c7dca9a72cd0e3cb'
    assert old.keys() == new.keys()
    changed = [k for k in old if old[k] != new[k]]
    assert set(changed) == {'findings', 'created_at_utc'}
    assert old['findings'] == [x['description'] for x in new['findings']]
    assert all(x['severity'] == 'NONBLOCKING' for x in new['findings'])
    assert new['checks']['naturalness']['status'] == 'PASS_WITH_WARNING'
    recs = w.native.receipts(f)
    ready = [r for r in recs if r['status'] == 'REVIEW_READY']
    assert len(ready) == 1 and ready[0] == acceptance['native_receipt']
    assert len(recs) == 2 and not any(r['status'] in w.native.TERMINAL_RECEIPTS for r in recs)
    return {'status': 'PASS', 'original_archive_sha256': old_sha, 'accepted_review_sha256': new_sha, 'changed_fields_only': changed, 'finding_text_preserved': True, 'decisions_reasons_citations_source_and_coverage_identical': True, 'native_ready_receipt_count': 1, 'naturalness_warning_retained': True, 'frozen_tools_unchanged': True}


def main():
    freeze = read(B / 'MASTER_FREEZE.json')
    for item in freeze['files']:
        binding(item)
    roots = [audit_root(rid) for rid in ROOTS]
    global_reservations = []
    for p in (B / 'units').glob('root*/SOURCE_RESERVATIONS.json'):
        global_reservations.extend(r['source_id'] for r in read(p)['allocation'])
    assert len(global_reservations) == len(set(global_reservations))
    sealed = []
    for item in read(B / 'evidence/SEALED_DELIVERIES_BEFORE.json')['sealed_manifests']:
        manifest = Path(binding(item))
        data = read(manifest)
        for row in data['files']:
            binding(row, manifest.parent)
        sealed.append({'manifest_path': str(manifest), 'manifest_sha256': sha(manifest), 'all_manifest_payload_hashes_unchanged': True, 'files_checked': len(data['files']), 'method': 'Hash-only read of old sealed payloads; TEST dialogue contents not parsed or semantically processed.'})
    result = {'schema_version': 'v133_independent_dev_review_audit_v1', 'actual_auditor': WHO, 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS', 'scope_roots': list(ROOTS), 'master_freeze_sha256': sha(B / 'MASTER_FREEZE.json'), 'master_frozen_bindings_checked': len(freeze['files']), 'test_phase_opened_by_this_audit': False, 'v133_test_source_contents_read': False, 'auditor_is_independent_of_authors': True, 'auditor_is_original_scene_reviewer': True, 'external_or_strict_blind_review': False, 'semantic_correctness_certified_by_program': False, 'no_solver_training_qa_or_pool_merge': True, 'limitations': ['Mechanical evidence and integrity audit supplements actual semantic scene reviews; it does not prove semantic correctness.', 'Author and reviewer share the same model family and filesystem.', 'Baseline HOLD and historical contradictions remain preserved; success is only under the new conflict-quarantine retry protocol.'], 'roots': roots, 'schema_recovery_229160_B2': schema_recovery(), 'sealed_deliveries': sealed, 'global_reserved_native_source_count': len(global_reservations), 'global_reservations_unique': True, 'audit_script_sha256': sha(__file__)}
    output = B / 'evidence/INDEPENDENT_DEV_147355_229160_AUDIT.json'
    with output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'status': result['status'], 'path': str(output), 'sha256': sha(output), 'roots': [{k: r[k] for k in ['root_id', 'new_turn_count', 'final_turn_count']} for r in roots], 'sealed_files_checked': sum(x['files_checked'] for x in sealed)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
