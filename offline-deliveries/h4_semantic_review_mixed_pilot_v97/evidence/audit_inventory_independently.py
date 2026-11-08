"""Independent real-inventory replay and isolated behavior checks for v97.

No production files are changed; temporary reservations are removed with their
temporary directory. Historical collision findings remain failures of novelty,
separate from this implementation audit passing its stated contract.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import tempfile

PROJECT = Path('/root/autodl-tmp/gxy-selfplay')
DELIVERIES = PROJECT / 'offline-deliveries'
V97 = DELIVERIES / 'h4_semantic_review_mixed_pilot_v97'
ENGINE = V97 / 'expansion_engine'
HERE = Path(__file__).resolve().parent

def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bind(path): return {'path': str(Path(path).resolve()), 'sha256': sha(path), 'bytes': Path(path).stat().st_size}
def canon(value): return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':')).encode()
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
def key(row): return (row['dataset'], row['split'], str(row['item_id']))
def owner(row): return (str(Path(row['run']).resolve()), row['root_id'], row['scene_id'])
def identity(row): return (owner(row), row['source_item_sha256'], row['source_id'])

inventory = load('v97_independent_inventory_subject', ENGINE / 'source_inventory.py')
pool = load('v97_independent_pool_subject', ENGINE / 'source_pool.py')
bound_before = [bind(ENGINE / filename) for filename in ['source_inventory.py', 'source_pool.py', 'SOURCE_SELECTION.json']]
selection = read(ENGINE / 'SOURCE_SELECTION.json')
expected = {}
reservation_files = []
for path in sorted(DELIVERIES.rglob('SOURCE_RESERVATIONS.json')):
    payload = read(path)
    assert payload['schema_version'] == 'h4.r02.source_reservations.v4'
    assert Path(payload['run']).resolve() == path.parent.resolve()
    reservation_files.append(bind(path))
    slots = set()
    keys = set()
    for row in payload['reservations']:
        assert Path(row['run']).resolve() == path.parent.resolve()
        assert key(row) not in keys and owner(row) not in slots
        assert len(row['source_item_sha256']) == 64
        assert all(c in '0123456789abcdef' for c in row['source_item_sha256'])
        expected.setdefault(key(row), set()).add(identity(row))
        slots.add(owner(row)); keys.add(key(row))

# The official legacy collector alone knows pre-reservation v1/v2 accepted
# evidence. It is shared only as an evidence reader, not for merge semantics.
legacy_prior = inventory.legacy.collect_prior_usage(inventory.legacy.prior_runs(inventory.V1_RUNS, inventory.V2_RUNS))
for row in legacy_prior['accepted_uses']:
    expected.setdefault(key(row), set()).add(identity(row))
actual = inventory.collect_prior()
actual_keys = {key(row) for row in actual['accepted_uses']}
assert actual_keys == set(expected)
owner_coverage = []
for row in actual['accepted_uses']:
    identities = {identity(o) for o in row['historical_owners']}
    assert expected[key(row)] <= identities
    owner_coverage.append({'event_key': list(key(row)), 'expected_reservation_or_legacy_owner_bindings': len(expected[key(row)]), 'actual_owner_bindings': len(identities), 'all_recorded_owners_preserved': True})
evidence_checks = []
for row in actual['prior_evidence']:
    path = Path(row['path'])
    assert sha(path) == row['sha256']
    evidence_checks.append(bind(path))
actual_evidence_paths = {row['path'] for row in actual['prior_evidence']}
assert all(row['path'] in actual_evidence_paths for row in reservation_files)

source_dir = Path(selection['source_dir'])
source_files = [source_dir / name for name in ['soda_train_sample.json', 'soda_train_sample_source.json', 'multiwoz_train_sample.json', 'multiwoz_train_source.json']]
soda = read(source_files[0]); soda_receipt = read(source_files[1])
assert sha(source_files[0]) == soda_receipt['sha256'] == soda_receipt['download_sha256']
assert source_files[0].stat().st_size == soda_receipt['bytes'] == soda_receipt['download_bytes']
assert soda_receipt['local_is_exact_download_bytes'] is True
available = [str(x['row']['original_index']) for x in soda['rows'] if ('allenai/soda', 'train', str(x['row']['original_index'])) not in actual_keys]
frozen = pool.freeze_unused_source_subset(actual_keys, 4, source_dir, selected_source_ids=selection['source_ids'])
assert [x['source_id'] for x in frozen['items']] == selection['source_ids']
frozen_rows = {x['source_id']: x for x in frozen['items']}
allocations = []
for root, source_id, allocation in zip(selection['roots'], selection['source_ids'], selection['allocation']):
    native = next(x for x in soda['rows'] if str(x['row']['original_index']) == source_id.split(':')[-1])
    raw = native['row']; chosen = frozen_rows[source_id]
    assert not native['truncated_cells'] and len(raw['speakers']) == len(raw['dialogue'])
    assert hashlib.sha256(canon(raw)).hexdigest() == chosen['source_item_sha256'] == allocation['source_item_sha256']
    units = [{'unit_id': 'narrative', 'kind': 'narrative', 'speaker': None, 'text': raw['narrative']}]
    units += [{'unit_id': f'dialogue:{i}', 'kind': 'dialogue', 'speaker': s, 'text': t} for i, (s, t) in enumerate(zip(raw['speakers'], raw['dialogue']))]
    assert units == chosen['units'] and hashlib.sha256(canon(units)).hexdigest() == chosen['excerpt_sha256']
    assert key(chosen) not in actual_keys
    allocations.append({'root_id': root, 'source_id': source_id, 'source_item_sha256': chosen['source_item_sha256'], 'excerpt_sha256': chosen['excerpt_sha256'], 'full_units_count': len(units), 'available': True, 'all_full_units_and_hashes_match_native_source': True})
assignments = {root: [frozen_rows[sid]] for root, sid in zip(selection['roots'], selection['source_ids'])}
inventory.validate_available(assignments, actual)

fixtures = []
def checked(name, operation, error=None):
    try:
        result = operation()
    except ValueError as exc:
        if error is None or error not in str(exc):
            raise
        fixtures.append({'case': name, 'status': 'PASS', 'expected_rejection': str(exc)})
        return None
    if error is not None:
        raise AssertionError('did not reject: ' + name)
    fixtures.append({'case': name, 'status': 'PASS'})
    return result

with tempfile.TemporaryDirectory(prefix='v97-independent-inventory-') as temporary:
    tmp = Path(temporary)
    empty = tmp / 'empty'; empty.mkdir()
    isolated = {name: empty for name in ['v1_runs', 'v2_runs', 'v4_runs', 'v5_runs', 'v6_runs', 'v7_runs', 'v8_runs', 'v9_runs', 'v11_runs', 'v12_runs', 'v15_runs', 'v90_runs', 'v91_runs', 'v92_runs', 'v93_runs', 'v94_runs', 'v95_runs']}
    archive = tmp / 'archive'; archive.mkdir()
    def collect(): return inventory.collect_prior(empty, historical_archive_runs=archive, **isolated)
    def item(iid='seed', digest='a' * 64, sid='soda:train:seed'):
        return {'dataset': 'allenai/soda', 'split': 'train', 'item_id': iid, 'source_id': sid, 'source_item_sha256': digest}
    def payload(path, row):
        return {'schema_version': 'h4.r02.source_reservations.v4', 'run': str(path.resolve()), 'consumption_policy': 'reservation_before_model_request_never_release', 'reservations': [row]}
    def add(path, root, digest='a' * 64, sid='soda:train:seed'):
        path.mkdir(parents=True)
        row = {**item(digest=digest, sid=sid), 'run': str(path.resolve()), 'root_id': root, 'scene_id': 'S01'}
        (path / 'SOURCE_RESERVATIONS.json').write_text(json.dumps(payload(path, row)))
        return row
    one = add(archive / 'future_unenumerated_generation' / 'first', 'root-A')
    prior = checked('recursive_unknown_generation_consumes_event', collect)
    assert len(prior['accepted_uses']) == 1
    checked('consumed_event_new_owner_rejected', lambda: inventory.validate_available({'root-new': [item()]}, prior), 'source_event_already_consumed')
    two = add(archive / 'v-older' / 'second', 'root-B')
    prior = checked('historical_duplicate_slots_remain_occupied', collect)
    assert len(prior['accepted_uses']) == 1 and len(prior['historical_conflicts']) == 1
    assert {owner(r) for r in prior['historical_conflicts'][0]['owners']} == {owner(one), owner(two)}
    assert prior['historical_conflicts'][0]['reasons'] == ['multiple_slots']
    checked('all_conflicting_historical_owners_preserved', lambda: None)
    checked('existing_conflicted_slot_retry_rejected', lambda: inventory.reserve(Path(one['run']), {'root-A': [item()]}, prior), 'existing_reservation_conflicts_with_prior')
    checked('conflict_does_not_allow_new_owner', lambda: inventory.reserve(tmp / 'new-conflict', {'root-C': [item()]}, prior), 'source_event_already_consumed')
    third = add(archive / 'hash-mismatch' / 'third', 'root-D', 'b' * 64, 'soda:train:changed-id')
    prior = checked('historical_hash_and_source_id_conflicts_recorded', collect)
    conflict = prior['historical_conflicts'][0]
    assert set(conflict['reasons']) == {'multiple_slots', 'multiple_source_hashes', 'multiple_source_ids'}
    assert len(conflict['owners']) == 3
    checked('hash_variant_cannot_reclaim_occupied_event', lambda: inventory.validate_available({'new': [item(digest='b' * 64)]}, prior), 'source_event_hash_changed')
    free = item('never-used', 'c' * 64, 'soda:train:never-used')
    new_run = tmp / 'new-free'
    saved = checked('never_used_source_reserves_despite_old_conflicts', lambda: inventory.reserve(new_run, {'root-new': [free]}, prior))
    assert len(saved['reservations']) == 1
    # Existing identical reservation outside archive appears in caller's prior
    # during a genuine retry; use an independent value, not a re-merge helper.
    retry_prior = {**prior, 'accepted_uses': prior['accepted_uses'] + [{**saved['reservations'][0], 'historical_owners': [saved['reservations'][0]]}]}
    checked('unchanged_same_slot_reservation_retry_allowed', lambda: inventory.reserve(new_run, {'root-new': [free]}, retry_prior))
    checked('same_slot_payload_cannot_be_replaced', lambda: inventory.reserve(new_run, {'root-new': [item('other', 'd' * 64, 'soda:train:other')]}, retry_prior), 'existing_reservations_changed')
    checked('duplicate_source_assignment_rejected', lambda: inventory.validate_available({'root-one': [free], 'root-two': [free]}, prior), 'source_event_assigned_twice')
    evidence_path = Path(prior['prior_evidence'][0]['path'])
    old = evidence_path.read_bytes(); evidence_path.write_bytes(old + b'\n')
    checked('changed_historical_evidence_blocks_new_reservation', lambda: inventory.reserve(tmp / 'hash-changed', {'root-new': [item('another-free', 'e' * 64, 'soda:train:another-free')]}, prior), 'historical_usage_evidence_changed')
    evidence_path.write_bytes(old)
    bad = archive / 'bad-duplicate' / 'run'; bad.mkdir(parents=True)
    row = {**item('bad', sid='soda:train:bad'), 'run': str(bad.resolve()), 'root_id': 'root-bad', 'scene_id': 'S01'}
    data = payload(bad, row); data['reservations'].append(row)
    (bad / 'SOURCE_RESERVATIONS.json').write_text(json.dumps(data))
    checked('duplicate_within_one_reservation_file_fails_closed', collect, 'duplicate_event_or_slot_in_reservations')
    (bad / 'SOURCE_RESERVATIONS.json').unlink(); bad.rmdir()
    report = archive / 'no-reservation' / 'run'; report.mkdir(parents=True)
    (report / 'SOURCE_RESERVATIONS.json').write_text(json.dumps({'schema_version': inventory.SCHEMA_VERSION, 'run': str(report.resolve()), 'reservations': []}))
    accepted = {**item('orphan', sid='soda:train:orphan'), 'run': str(report.resolve()), 'root_id': 'root-orphan', 'scene_id': 'S01'}
    (report / 'REPORT.json').write_text(json.dumps({'accepted_uses': [accepted]}))
    checked('accepted_report_without_matching_reservation_rejected', collect, 'accepted_event_missing_reservation')

assert [bind(ENGINE / filename) for filename in ['source_inventory.py', 'source_pool.py', 'SOURCE_SELECTION.json']] == bound_before
for old in reservation_files:
    assert sha(old['path']) == old['sha256']
out = {'schema_version': 'h4.v97.independent_inventory_audit.v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS_IMPLEMENTATION_CONTRACT_ONLY', 'historical_global_uniqueness_status': 'FAIL_HISTORICAL_REUSE_CONFIRMED_NOT_WAIVED', 'audit_script': bind(__file__), 'audited_engine_files': bound_before, 'network_requests': 0, 'production_reservations_created': 0, 'temporary_fixture_cases': fixtures, 'actual_occupied_identity_count': len(actual_keys), 'independent_expected_identity_count': len(expected), 'all_recorded_event_identities_and_owners_preserved': True, 'historical_conflicting_identity_count': len(actual['historical_conflicts']), 'historical_conflicts': actual['historical_conflicts'], 'owner_coverage': owner_coverage, 'all_history_reservation_file_bindings': reservation_files, 'all_collector_evidence_hash_checks': evidence_checks, 'source_file_bindings': [bind(p) for p in source_files], 'refresh_available_before_new_reservation': len(available), 'actual_selected': allocations, 'selected_assignment_validate_available': 'PASS', 'source_full_units_and_native_hashes_verified': True, 'historical_files_unchanged_during_audit': True, 'limits': ['Read-only pre-reservation snapshot; real prepare/live must recollect and reserve under shared lock.', 'Historical collisions remain recorded failures of global novelty. This passing code audit does not recertify v95 or v96 novelty, semantic quality, question Gold, or production readiness.', 'The collector discovers all local recursive reservation files and legacy accepted reports; unrecorded external calls cannot be audited.']}
target = HERE / 'INVENTORY_INDEPENDENT_AUDIT.json'
if target.exists():
    raise FileExistsError(target)
target.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status': out['status'], 'occupied': len(actual_keys), 'historical_conflicts': len(actual['historical_conflicts']), 'selected_unused': len(allocations), 'temporary_fixture_cases_passed': len(fixtures), 'audit': str(target)}, ensure_ascii=False))
