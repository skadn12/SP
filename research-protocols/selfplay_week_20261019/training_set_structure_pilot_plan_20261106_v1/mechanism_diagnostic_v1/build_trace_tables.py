"""Join frozen retrospective reviews for traceability, without changing verdicts."""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(value):
    if isinstance(value, (str, int, float)) or value is None:
        return value
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def write_csv_new(path, rows):
    keys = list(rows[0])
    with path.open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows({k: compact(r.get(k)) for k in keys} for r in rows)


def main():
    freeze = json.loads((HERE / 'BLIND_REVIEWS_FREEZE.json').read_text())
    assert freeze['status'] == 'PASS'
    join = {x['blind_slot_id']: x for x in json.loads((HERE / 'PRIVATE_JOIN_MAP.json').read_text())['rows']}
    old_slots = list(csv.DictReader((HERE.parent / 'execution_20261106_v1/results/SLOTS.csv').open()))
    old = {(x['opportunity_id'], x['slot_id']): x for x in old_slots}
    slots, mappings = [], []
    for p in sorted((HERE / 'blind_slot_review').glob('review_*.json')):
        data = json.loads(p.read_text())
        for root in data['roots']:
            for item in root['slots']:
                bound = join[item['blind_slot_id']]
                prior = old[(bound['opportunity_id'], bound['slot_id'])]
                slots.append({
                    'root_id': root['root_id'], 'review_file': p.name,
                    'review_limit': 'READ_AND_WRITE_PROTOCOL_DEVIATION' if p.name == 'review_a.json' else 'SEE_REVIEW_DISCLOSURE',
                    'blind_slot_id': item['blind_slot_id'], 'opportunity_id': bound['opportunity_id'],
                    'slot_id': bound['slot_id'], 'group': bound['group'], 'selected': bound['selected'],
                    'source_support_status': item.get('source_support_status'),
                    'positive_non_d_supported': item.get('positive_non_d_supported'),
                    'd_codes': item.get('d_codes'), 'historical_status_review': item.get('historical_status'),
                    'historical_comparison': item.get('historical_comparison'),
                    'actual_need_scope': item.get('actual_need_scope'),
                    'actual_sufficient_program': item.get('actual_sufficient_program'),
                    'positive_non_d_evidence': item.get('positive_non_d_evidence'),
                    'alternative_sufficient_routes': item.get('alternative_sufficient_routes'),
                    'uncertainty': item.get('uncertainty'),
                    'original_selection_disposition': prior.get('selection_disposition_raw'),
                    'original_final_fidelity_status': prior.get('final_target_fidelity_status'),
                    'original_final_S': prior.get('final_S_descriptive'),
                })
            for item in root['inventory_mapping']:
                mappings.append({
                    'root_id': root['root_id'], 'review_file': p.name,
                    'review_limit': 'READ_AND_WRITE_PROTOCOL_DEVIATION' if p.name == 'review_a.json' else 'SEE_REVIEW_DISCLOSURE',
                    **item,
                })
    assert len(slots) == 45 and len(mappings) == 83
    slots.sort(key=lambda x: (x['root_id'], x['blind_slot_id']))
    mappings.sort(key=lambda x: (x['root_id'], x['inventory_id']))
    write_csv_new(HERE / 'SLOT_TRACE.csv', slots)
    # Fields vary by reviewer; JSON retains the complete original structure.
    with (HERE / 'INVENTORY_TRACE.json').open('x') as f:
        json.dump(mappings, f, ensure_ascii=False, indent=2)
        f.write('\n')
    summary = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'slots': len(slots), 'inventory_entries': len(mappings),
        'slot_positive_non_d_counts': dict(Counter(str(x['positive_non_d_supported']) for x in slots)),
        'positive_slots_selected': dict(Counter(str(x['selected']) for x in slots if x['positive_non_d_supported'] is True)),
        'inventory_representation_counts': dict(Counter(x['representation_status'] for x in mappings)),
        'review_a_protocol_deviation': True,
        'not_original_S_and_not_causal': True,
        'inputs': [{'path': str(p), 'sha256': sha(p)} for p in
                   [HERE / 'BLIND_REVIEWS_FREEZE.json', HERE / 'PRIVATE_JOIN_MAP.json',
                    HERE.parent / 'execution_20261106_v1/results/SLOTS.csv']],
        'outputs': [{'path': str(p), 'sha256': sha(p)} for p in
                    [HERE / 'SLOT_TRACE.csv', HERE / 'INVENTORY_TRACE.json']],
    }
    with (HERE / 'TRACE_MANIFEST.json').open('x') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
