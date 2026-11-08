"""Independent read-only integrity check; writes only a new audit report on request."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import argparse

B = Path(__file__).resolve().parent.parent
U = B / 'units/root151581'
SCENES = ['R02:A:S01', 'R02:A:S02', 'R02:A:S03', 'R02:B:S01', 'R02:B:S02', 'R02:B:S03']
NINE = {'time', 'location', 'actor_identity', 'plan_vs_completion',
        'cross_scene_continuity', 'semantic_repetition', 'naturalness',
        'event_fidelity', 'quarantine_compliance'}


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stamp(s):
    return datetime.fromisoformat(s)


def audit():
    checks = []
    def ck(name, ok):
        checks.append({'name': name, 'pass': bool(ok)})
    def bindings(name, rows):
        for i, row in enumerate(rows):
            ck(f'{name}/{i}', sha(row['path']) == row['sha256'])
    def citations(name, obj, source):
        lookup = {t['turn_ref']: t for t in source['turns']}
        count = 0
        def walk(x):
            nonlocal count
            if isinstance(x, dict):
                if {'turn_ref', 'speaker', 'quote'} <= x.keys():
                    count += 1
                    t = lookup.get(x['turn_ref'])
                    ck(f'{name}/citation{count}', t is not None and
                       x['speaker'] == t['speaker'] and isinstance(x['quote'], str)
                       and bool(x['quote']) and x['quote'] in t['text'])
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
        walk(obj)
        return count
    closure = read(B / 'DEV_PHASE_CLOSED.json')
    closed = stamp(closure['closed_at_utc'])
    bindings('master_freeze', read(B / 'MASTER_FREEZE.json')['files'])
    bindings('closed_dev_files', [dict(x, path=str(B / x['path']))
                                 for x in closure['frozen_dev_files']])
    bindings('closed_dev_audits', [dict(x, path=str(B / x['path']))
                                  for x in closure['audits']])
    cfg = read(U / 'RUN_CONFIG.json')
    ck('checker_not_author_or_reviewer', '/root/base_control_report' not in
       [cfg['author'], cfg['reviewer']])
    ck('test_configuration', cfg['split'] == 'test' and
       all(cfg[k] is False for k in ['solver_run', 'training_run', 'question_generation']))
    manifest = next(r for r in read(B / 'ROOT_MANIFEST.json')['roots']
                    if r['root_id'] == 'cc-train-episode-151581')
    for i, x in enumerate(manifest['input_bindings']):
        ck(f'legacy_original_copy_hash/{i}',
           sha(x['original_path']) == sha(x['copied_path']) == x['sha256'])
    ck('legacy_hold_retained', read(U / 'inputs/HISTORICAL_CONTINUATION_CONSTRAINTS.json')['decision'] == 'HOLD')
    bindings('execution_freeze', read(U / 'EXECUTION_INPUTS_FROZEN.json')['files'])
    basepath = U / 'inputs/BASELINE.json'
    base = read(basepath)
    plan = read(U / 'SAFE_PLAN.json')
    pr = read(U / 'SAFE_PLAN_REVIEW.json')
    citations('safe_plan', plan, base)
    citations('plan_review', pr, base)
    ck('plan_ready_hash_identity', pr['decision'] == 'READY' and
       pr['reviewer'] == cfg['reviewer'] and pr['safe_plan_sha256'] == sha(U / 'SAFE_PLAN.json') and
       pr['baseline_sha256'] == sha(basepath) and plan['author'] == cfg['author'])
    ck('quarantine_not_resolution', plan['baseline_conflicts_resolved'] is False and
       pr['baseline_conflicts_resolved'] is False)
    recovery_path = U / 'author/SAFE_PLAN_SAVE_CONTEXT_RECEIPT.json'
    recovery = read(recovery_path)
    ck('plan_context_presave_recovery_preserved', recovery['initial_failure'] == 'AUTHOR_CONTEXT_REQUIRED' and
       recovery['semantic_payload_changed'] is False and
       sha(recovery['initial_full_context_path']) == sha(recovery['accepted_full_context_path']) and
       recovery['candidate_payload_sha256'] == sha(U / 'author/SAFE_PLAN_CANDIDATE.json'))
    allocations = read(U / 'SOURCE_RESERVATIONS.json')
    selection = read(U / 'SELECTION.json')
    pool = {'multiwoz22:train:' + e['dialogue_id']: e
            for e in read(U / 'inputs/EVENT_POOL.json')['events']}
    old_consumed = set(read(B / 'evidence/SOURCE_CONSUMPTION_FREEZE.json')['consumed_source_ids'])
    global_ids = []
    for path in (B / 'units').glob('root*/SOURCE_RESERVATIONS.json'):
        global_ids.extend(r['source_id'] for r in read(path)['allocation'])
    ck('global_new_sources_unique', len(global_ids) == len(set(global_ids)))
    ck('three_fixed_allocations', [x['scene_ref'] for x in allocations['allocation']] == SCENES[3:])
    ck('source_reservations_permanent', allocations['permanent_including_failed_and_unexecuted_slots'] is True)
    for i, a in enumerate(allocations['allocation']):
        event = read(a['source_item_path'])
        ck(f'event{i}/original_pool_exact', a['source_id'] in pool and event == pool[a['source_id']])
        ck(f'event{i}/unused_native_train_hash', a['source_id'].startswith('multiwoz22:train:') and
           a['source_id'] not in old_consumed and sha(a['source_item_path']) == a['source_item_sha256'])
        ck(f'event{i}/full_turn_ids', a['native_turn_ids_read'] == [t['turn_id'] for t in event['turns']])
        ck(f'event{i}/selection_match', all(a[k] == selection['selections'][i][k]
                                          for k in selection['selections'][i]))
    for path in sorted((U / 'reads').glob('*/READ_CONTEXT.json')):
        c = read(path)
        ck('test_context_after_dev/' + path.parent.name, stamp(c['started_at_utc']) >= closed)
        ck('context_source_hash/' + path.parent.name, sha(c['source_path']) == c['source_sha256'])
        bindings('context_inputs/' + path.parent.name, c['inputs'])
    previous = base['turns']
    prevpath = basepath
    prev_review_time = stamp(pr['created_at_utc'])
    rows = []
    total_new = 0
    allowed = {t['speaker'] for t in base['turns']}
    for scene in SCENES:
        folder = U / 'roots/root151581/scenes' / scene.removeprefix('R02:').replace(':', '_')
        if not (folder / 'SOURCE_REVIEW.json').exists():
            return {'status': 'PENDING', 'pending_scene': scene, 'checks': checks}
        draft = read(folder / 'EXPANSION_DRAFT.json')
        source = read(folder / 'CUMULATIVE_SOURCE.json')
        review = read(folder / 'SOURCE_REVIEW.json')
        receipt = read(folder / 'AUTHOR_RECEIPT.json')
        payload = read(folder / 'AUTHOR_PAYLOAD.json')
        count = citations(scene, review, source)
        ck(scene + '/prefix_exact', source['turns'] == previous + draft['turns'])
        ck(scene + '/baseline_exact', source['turns'][:len(base['turns'])] == base['turns'])
        ck(scene + '/input_hash', draft['input_source_sha256'] == sha(prevpath))
        ck(scene + '/new_turns', 12 <= len(draft['turns']) <= 16 and
           all(t['speaker'] in allowed and t['text'].strip() for t in draft['turns']))
        ck(scene + '/unique_expected_refs', [t['turn_ref'] for t in draft['turns']] ==
           [f'{scene}:T{i:03d}' for i in range(1, len(draft['turns']) + 1)])
        ck(scene + '/payload_preserved', payload['messages'] ==
           [{'speaker': t['speaker'], 'text': t['text']} for t in draft['turns']] and
           receipt['payload_sha256'] == sha(folder / 'AUTHOR_PAYLOAD.json'))
        ck(scene + '/identity_ready', draft['author'] == cfg['author'] == receipt['actual_author'] and
           review['reviewer'] == cfg['reviewer'] != cfg['author'] and review['decision'] == 'READY')
        ck(scene + '/nine_checks', NINE <= set(review['checks']) and all(
           c['status'] in ['PASS', 'WARNING', 'PASS_WITH_WARNING'] and c['reason'] and c['citations']
           for c in review['checks'].values()))
        ck(scene + '/full_source_attestation', review['source_sha256'] == sha(folder / 'CUMULATIVE_SOURCE.json') and
           review['turn_refs_read'] == [t['turn_ref'] for t in source['turns']] and
           review['full_source_read'] is True and receipt['full_cumulative_source_read'] is True and
           review['baseline_conflicts_resolved'] is False)
        for purpose, who, input_source in [('author', cfg['author'], previous), ('reviewer', cfg['reviewer'], source['turns'])]:
            ctxpath = U / 'reads' / (scene.replace(':', '_') + '_' + purpose) / 'READ_CONTEXT.json'
            context = read(ctxpath)
            full = (ctxpath.parent / 'FULL_INPUT.txt').read_text()
            ck(scene + '/' + purpose + '_coverage', context['actual_reader'] == who and
               context['turn_refs'] == [t['turn_ref'] for t in input_source] and
               all(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" in full for t in input_source))
            if purpose == 'author':
                ck(scene + '/prior_ready_before_author', stamp(context['started_at_utc']) >= prev_review_time)
                ck(scene + '/freeze_before_author', stamp(context['started_at_utc']) >=
                   stamp(read(U / 'EXECUTION_INPUTS_FROZEN.json')['created_at_utc']))
                ck(scene + '/author_before_draft', stamp(context['started_at_utc']) <= stamp(receipt['created_at_utc']))
            else:
                ck(scene + '/review_read_after_draft_before_ready', stamp(receipt['created_at_utc']) <=
                   stamp(context['started_at_utc']) <= stamp(review['created_at_utc']))
            if ':B:' in scene:
                allocation = next(a for a in allocations['allocation'] if a['scene_ref'] == scene)
                event = read(allocation['source_item_path'])
                ck(scene + '/' + purpose + '_native_coverage', all(
                   f"[{t['turn_id']}] {t['speaker']}: {t['utterance']}" in full for t in event['turns']))
        is_b = ':B:' in scene
        ck(scene + '/native_read_attestation', receipt['full_selected_native_event_read'] == is_b and
           review['full_event_read'] == is_b)
        ck(scene + '/source_id', draft['source_id'] ==
           (next(a['source_id'] for a in allocations['allocation'] if a['scene_ref'] == scene) if is_b else None))
        ready_receipts = [read(p) for p in (folder / 'receipts').glob('*.json') if read(p)['status'] == 'REVIEW_READY']
        ck(scene + '/one_bound_ready_receipt', len(ready_receipts) == 1 and
           ready_receipts[0]['review_sha256'] == sha(folder / 'SOURCE_REVIEW.json'))
        previous = source['turns']
        prevpath = folder / 'CUMULATIVE_SOURCE.json'
        prev_review_time = stamp(review['created_at_utc'])
        total_new += len(draft['turns'])
        rows.append({'scene_ref': scene, 'status': review['decision'], 'new_turns': len(draft['turns']),
                     'source_sha256': sha(prevpath), 'review_sha256': sha(folder / 'SOURCE_REVIEW.json'),
                     'exact_citations_checked': count, 'warnings': [k for k, v in review['checks'].items() if v['status'] != 'PASS']})
    ck('root_final_source_exact', read(U / 'roots/root151581/CUMULATIVE_SOURCE.json')['turns'] == previous)
    ck('all_refs_unique', len({t['turn_ref'] for t in previous}) == len(previous))
    failed = [c['name'] for c in checks if not c['pass']]
    return {'schema_version': 'v133_independent_test_mechanical_check_v1',
            'actual_checker': '/root/base_control_report', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'TEST151581 independent mechanical verification only',
            'semantic_review_performed_by_this_check': False, 'external_or_strict_blind_review': False,
            'limitations': ['Hashes and exact citation checks verify integrity and declared coverage, not semantic truth.',
                           'Saved full-read attestations do not independently prove the reader mental process.',
                           'Authors, reviewers and checker share the same model family and filesystem.'],
            'status': 'PASS' if not failed else 'FAIL', 'check_count': len(checks), 'failed_checks': failed,
            'baseline_turns': len(base['turns']), 'new_turns': total_new, 'cumulative_turns': len(previous),
            'ready_scenes': len(rows), 'scenes': rows, 'checks': checks,
            'dev_closure_sha256': sha(B / 'DEV_PHASE_CLOSED.json'), 'audit_script_sha256': sha(__file__),
            'plan_presave_context_failure_receipt_sha256': sha(recovery_path),
            'plan_presave_failure_not_claimed_as_semantic_retry': True,
            'question_generation': False, 'solver_run': False, 'training_run': False, 'evaluation_pool_modified': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    args = parser.parse_args()
    result = audit()
    if args.output:
        assert result['status'] != 'PENDING', 'WAIT_FOR_ALL_SIX_FORMAL_REVIEWS'
        with Path(args.output).open('x') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}, ensure_ascii=False, indent=2))
