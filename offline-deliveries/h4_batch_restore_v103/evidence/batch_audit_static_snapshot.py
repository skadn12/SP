"""Independent stdlib artifact and budget audit for the bounded v103 batch.

No engine, provider, tokenizer or semantic-grading imports. Audit success binds
frozen inputs, actual calls and reported outcomes; model acceptance and source
content admission remain separate.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOTS = ['cc-train-episode-14304', 'cc-train-episode-20432', 'cc-train-episode-24719', 'cc-train-episode-38601']
SOURCES = ['soda:train:6028165', 'soda:train:6027926', 'soda:train:6028392', 'soda:train:6028117']
SOURCE_HASHES = ['c8f13b844c5912f4fe085a9db22949fb1579362d26ca13adf4332a4001caf9d4',
 '4ac3a213d263b27a339ba2f5c4888e9bb792ebcd36327688175634e52f6efb89',
 'b39532573fcea154a4c6ea8bc7d74f5f1c540d5ef2ea0725ab8a5b64ab7c2e45',
 '8b5bf286dfab9d00ab2950f7c4a95b7f688c40641c07804870057b905326d2ba']


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value, *, ascii_only=False):
    return hashlib.sha256(json.dumps(value, ensure_ascii=ascii_only, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def key(row): return row['dataset'], row['split'], str(row['item_id'])
def check(ok, code, failures, **details):
    if not ok: failures.append({'code': code, **details})
def context(payload, marker):
    return json.JSONDecoder().raw_decode(payload['messages'][0]['content'].split(marker, 1)[1])[0]


def frozen_audit(run, failures, *, live):
    manifest = read(run / 'MANIFEST.json'); selected = read(BASE / 'expansion_engine/SOURCE_SELECTION.json')
    allocation = read(run / 'UNIQUE_EVENT_ALLOCATION.json'); prior = read(run / 'INVENTORY_BEFORE.json')
    registry = {row['root_id']: row for row in read(BASE / 'expansion_engine/SOURCE_REGISTRY.json')['roots']}
    check(manifest.get('roots') == selected.get('roots') == allocation.get('roots') == ROOTS,
          'frozen_root_order_changed', failures)
    check(selected.get('source_ids') == SOURCES, 'preselected_unused_source_ids_changed', failures)
    check(manifest.get('selected_routes') == ['dataset_new_event'] and manifest.get('scenes_per_route_target') == 1,
          'route_or_scene_target_changed', failures)
    check(manifest.get('http_max') == 128 and manifest.get('max_planner_attempts') == 2
          and manifest.get('max_realizer_attempts') == 2 and manifest.get('max_state_extraction_attempts') == 2
          and manifest.get('max_event_state_attempts_per_plan') == 2
          and manifest.get('automatic_transport_retries') == 1 and manifest.get('review_schema_retries') == 1
          and manifest.get('card_extraction_attempts_per_source') == 2,
          'expansion_stage_or_global_budget_changed', failures)
    files = manifest.get('files', []); paths = {row['path']: row['sha256'] for row in files}
    check(len(paths) == len(files), 'duplicate_frozen_file', failures)
    changed = [path for path, value in paths.items() if not Path(path).is_file() or sha(path) != value]
    check(not changed, 'frozen_input_changed', failures, paths=changed)
    preservation = read(BASE / 'PRESERVATION_BEFORE.json')
    changed = [row['path'] for row in preservation['files'] if not Path(row['path']).is_file() or sha(row['path']) != row['sha256']]
    check(not changed, 'historical_protected_file_changed', failures, paths=changed)
    snapshots = read(run / 'EXECUTION_CODE_SNAPSHOT.json')['files']
    for row in snapshots:
        check(sha(row['original_path']) == sha(row['snapshot_path']) == row['sha256'],
              'execution_code_snapshot_changed', failures, path=row['original_path'])
    previous = {key(row) for row in prior['accepted_uses']}
    rows = []
    for index, root in enumerate(ROOTS):
        spec = registry[root]; original = read(spec['source_path']); frozen = read(run / 'sources' / f'{root}.json')
        check(sha(spec['source_path']) == spec['source_sha256'] and frozen == original
              and len(frozen['turns']) == spec['source_turns'] == [106, 99, 102, 116][index],
              'complete_historical_input_changed', failures, root=root)
        assigned = allocation['assignments'][root]
        check(len(assigned) == 1 and assigned[0].get('source_id') == SOURCES[index]
              and assigned[0].get('source_item_sha256') == SOURCE_HASHES[index]
              and key(assigned[0]) not in previous, 'source_assignment_reused_or_changed', failures, root=root)
        rows.append({'root_id': root, 'historical_turns': len(frozen['turns']), 'source_id': SOURCES[index],
                     'root_exposure': 'previously attempted root', 'event_source_exposure': 'unused before this batch'})
    source_manifest = read(run / 'dataset_sources.json')
    check([row['source_id'] for row in source_manifest['items']] == SOURCES,
          'source_raw_manifest_order_changed', failures)
    for row in source_manifest['files']:
        check(sha(row['local_path']) == row['local_sha256']
              and sha(row['source_receipt_path']) == row['source_receipt_sha256'],
              'raw_external_source_file_changed', failures, path=row['local_path'])
    for row in prior['prior_evidence']:
        check(Path(row['path']).is_file() and sha(row['path']) == row['sha256'],
              'historical_inventory_evidence_changed', failures, path=row['path'])
    reservation = read(run / 'SOURCE_RESERVATIONS.json')
    check([x['root_id'] for x in reservation['reservations']] == ROOTS
          and [x['source_id'] for x in reservation['reservations']] == SOURCES
          and reservation['consumption_policy'] == 'reservation_before_model_request_never_release',
          'permanent_source_reservations_changed', failures)
    for root in ROOTS:
        a = allocation['assignments'][root][0]
        reserved = [x for x in reservation['reservations'] if x['root_id'] == root]
        check(len(reserved) == 1 and all(reserved[0][k] == a[k] for k in
              ['dataset', 'split', 'item_id', 'source_id', 'source_item_sha256'])
              and reserved[0]['scene_id'] == 'S01', 'reservation_assignment_changed', failures, root=root)
    return manifest, {'manifest_files': len(files), 'historical_protected_files': len(preservation['files']),
                      'execution_snapshots': len(snapshots), 'sources': rows,
                      'consumed_source_identities_before': len(previous),
                      'reserved_source_identities': len(reservation['reservations']),
                      'live_already_started_at_static_audit': (run / 'LIVE_STARTED.json').exists()}


def inventory_audit(run, failures):
    before = read(run / 'INVENTORY_BEFORE.json')
    def signature(row, default_run=None):
        return (*key(row), str(Path(row.get('run', default_run)).resolve()),
                row['root_id'], row['scene_id'], row['source_id'], row['source_item_sha256'])
    paths = set(BASE.parent.rglob('SOURCE_RESERVATIONS.json'))
    paths.update(Path(row['path']) for row in before['prior_evidence'])
    actual = set(); legacy_count = 0
    for path in sorted(paths):
        data = read(path)
        rows = data.get('reservations', []) if path.name == 'SOURCE_RESERVATIONS.json' else data.get('accepted_uses', [])
        if path.name == 'SOURCE_RESERVATIONS.json':
            check(Path(data['run']).resolve() == path.parent.resolve(), 'inventory_reservation_run_changed', failures, path=str(path))
        actual.update(signature(row, path.parent) for row in rows)
        if path.name == 'RESULT.json' and path.parent.name == 'B':
            old_run = path.parents[3]
            if (old_run / 'SOURCE_RESERVATIONS.json').exists():
                continue
            cards = {x['card_id']: x for x in read(old_run / 'dataset_cards_result.json')['accepted_cards']}
            sources = {x['source_id']: x for x in read(old_run / 'dataset_sources.json')['items']}
            for scene in data.get('scene_metadata', []):
                source = sources[cards[scene['dataset_card_id']]['source_id']]
                actual.add(signature({**source, 'root_id': data['root_id'], 'scene_id': scene['scene_id'], 'run': str(old_run)}))
                legacy_count += 1
    expected_old = {signature(owner) for row in before['accepted_uses'] for owner in row.get('historical_owners', [row])}
    expected_new = {signature(row, run) for row in read(run / 'SOURCE_RESERVATIONS.json')['reservations']}
    check(actual == expected_old | expected_new, 'raw_inventory_not_frozen_before_plus_current', failures,
          missing=[list(x) for x in sorted((expected_old | expected_new) - actual)],
          unexpected=[list(x) for x in sorted(actual - (expected_old | expected_new))])
    owners = defaultdict(set)
    for row in actual: owners[row[:3]].add(row[3:])
    conflicts = {k for k, v in owners.items() if len(v) > 1}
    check(conflicts == {key(x) for x in before['historical_conflicts']}, 'historical_conflicts_changed', failures)
    return {'consumed_identities_before': len({key(x) for x in before['accepted_uses']}),
            'consumed_identities_after': len(owners), 'current_reservations': len(expected_new),
            'historical_conflicts': len(conflicts), 'raw_inventory_artifacts': len(paths),
            'legacy_scene_acceptances': legacy_count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=BASE / 'expansion_engine/runs/four_roots_B_scene01')
    parser.add_argument('--phase', choices=['static', 'live'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): raise SystemExit('refuse to overwrite existing evidence: ' + str(args.out))
    failures = []
    manifest, frozen = frozen_audit(args.run, failures, live=args.phase == 'live')
    inventory = inventory_audit(args.run, failures)
    result = {'version': 'v103_independent_batch_audit_v1', 'phase': args.phase,
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'run': str(args.run.resolve()), 'status': 'FAIL' if failures else 'PASS',
              'failures': failures, 'frozen_inputs': frozen, 'inventory': inventory,
              'manifest_sha256': sha(args.run / 'MANIFEST.json'), 'script_sha256': sha(__file__),
              'model_calls': 0, 'http_calls': 0, 'semantic_certified': False,
              'limitation': 'Static artifact integrity audit does not certify semantic quality or predict uncompleted calls.'}
    if args.phase == 'live':
        result['actual'] = live_audit(args.run, manifest, failures)
        result['status'] = 'FAIL' if failures else 'PASS'
    with args.out.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps({'status': result['status'], 'out': str(args.out.resolve()),
                      'sha256': sha(args.out), 'failures': failures,
                      'inventory': inventory}, ensure_ascii=False, indent=2))
    raise SystemExit(bool(failures))


if __name__ == '__main__': main()
