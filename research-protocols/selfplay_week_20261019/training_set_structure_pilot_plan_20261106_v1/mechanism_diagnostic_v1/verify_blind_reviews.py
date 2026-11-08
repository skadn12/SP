"""Check reviewer coverage and cited quotations before selection is unmasked."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


packets = {}
for path in (HERE / 'blind_slot_review').glob('*.packet.json'):
    data = json.loads(path.read_text())
    packets[data['root_id']] = (path, data)
inventories = {}
for path in (HERE / 'source_inventory').glob('source_inventory_*.json'):
    for root in json.loads(path.read_text())['roots']:
        inventories[root['root_id']] = (path, root)
failures, results, roots_seen = [], [], set()
for path in sorted((HERE / 'blind_slot_review').glob('review_*.json')):
    payload = json.loads(path.read_text())
    for root in payload['roots']:
        root_id = root['root_id']
        if root_id in roots_seen or root_id not in packets:
            failures.append([root_id, 'unexpected or repeated root'])
            continue
        roots_seen.add(root_id)
        packet_path, packet = packets[root_id]
        inventory_path, inventory = inventories[root_id]
        expected_slots = {s['blind_slot_id'] for s in packet['slots']}
        expected_inventory = {x['inventory_id'] for x in inventory['opportunities']}
        got_slots = [x['blind_slot_id'] for x in root['slots']]
        got_inventory = [x['inventory_id'] for x in root['inventory_mapping']]
        if set(got_slots) != expected_slots or len(got_slots) != len(expected_slots):
            failures.append([root_id, 'slot coverage mismatch'])
        if set(got_inventory) != expected_inventory or len(got_inventory) != len(expected_inventory):
            failures.append([root_id, 'inventory coverage mismatch'])
        source_path = Path(packet['source']['path'])
        history_path = Path(packet['fixed_historical_reference']['path'])
        source = json.loads(source_path.read_text())
        refs = {t['turn_ref']: t['text'] for t in source['turns']}
        if sha(source_path) != packet['source']['sha256'] or sha(history_path) != packet['fixed_historical_reference']['sha256']:
            failures.append([root_id, 'input hash mismatch'])
        for item in root['slots']:
            if item.get('positive_non_d_supported') is True and (item.get('d_codes') or not item.get('positive_non_d_evidence')):
                failures.append([root_id, item['blind_slot_id'], 'inconsistent positive non-D'])
            for e in item.get('source_evidence', []):
                if isinstance(e, dict) and 'quote' in e and (e.get('turn_ref') not in refs or e['quote'] not in refs[e['turn_ref']]):
                    failures.append([root_id, item['blind_slot_id'], 'quote mismatch', e])
        for item in root['inventory_mapping']:
            if not set(item.get('related_blind_slot_ids', [])) <= expected_slots:
                failures.append([root_id, item['inventory_id'], 'unknown related slot'])
            if item.get('positive_non_d_supported') is True and item.get('d_codes'):
                failures.append([root_id, item['inventory_id'], 'inconsistent positive non-D'])
        results.append({'root_id': root_id, 'review_path': str(path), 'review_sha256': sha(path),
                        'packet_sha256': sha(packet_path), 'inventory_sha256': sha(inventory_path),
                        'slot_count': len(got_slots), 'inventory_count': len(got_inventory)})
if roots_seen != set(packets):
    failures.append(['root coverage mismatch', sorted(set(packets) - roots_seen)])
result = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS' if not failures else 'FAIL',
          'roots': results, 'failures': failures,
          'limitations': ['Mechanical verification does not establish semantic correctness, completeness of history comparison, or actual reviewer isolation.']}
print(json.dumps(result, ensure_ascii=False, indent=2))
if failures:
    raise SystemExit(1)
with (HERE / 'BLIND_REVIEWS_FREEZE.json').open('x') as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write('\n')
