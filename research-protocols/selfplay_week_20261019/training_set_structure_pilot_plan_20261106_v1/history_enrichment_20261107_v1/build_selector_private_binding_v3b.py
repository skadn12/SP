#!/usr/bin/env python3
"""Check private candidate/root bindings without exposing them to selectors."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAN = HERE.parent
OUT = HERE / 'selector_safe_review_20261107_v3b' / 'CONTROLLER_PRIVATE_BINDING_AUDIT.json'
INVENTORY = PLAN / 'execution_20261106_v1/manifests/HISTORICAL_COVERAGE.json'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path):
    return json.loads(path.read_text(encoding='utf-8'))

def main():
    if OUT.exists():
        raise FileExistsError(f'refusing to overwrite {OUT}')
    inventory = load(INVENTORY)['records']
    expected_roots = {'cc-train-episode-' + n for n in ('138343', '14304', '162896', '191749', '225575', '25652', '75241', '77882')}
    bindings = []
    for group in ('R02', 'R03'):
        controller_path = HERE / f'CONTROLLER_ANNOTATION_AUDIT_{group}.json'
        selector_path = OUT.parent / f'SELECTOR_TUPLES_{group}_REVIEW.json'
        controller = load(controller_path)
        private = {r['history_item_id']: (r, r['root_id'], 'records') for r in controller['records']}
        for chunk in controller.get('root_chunks', []):
            for r in chunk.get('unknown_history_items', []):
                hid = r['history_item_id']
                if hid in private:
                    raise ValueError(f'duplicate private ID {hid}')
                private[hid] = (r, chunk['root_id'], 'root_chunks.parent_root_id')
        selector = load(selector_path)
        if set(private) != {r['history_item_id'] for r in selector['records']}:
            raise ValueError(f'private and selector ID sets differ: {group}')
        for row in selector['records']:
            hid = row['history_item_id']
            record, root, root_origin = private[hid]
            index = int(hid[1:]) - 1
            coverage = inventory[index]
            if root not in expected_roots or coverage['root_id'] != root or coverage['candidate_id'] != record['candidate_id']:
                raise ValueError(f'inventory candidate/root mismatch: {hid}')
            bindings.append({
                'history_item_id': hid,
                'coverage_global_index': index,
                'candidate_id': record['candidate_id'],
                'root_id': root,
                'root_binding_source': root_origin,
                'inventory_candidate_and_root_match': True,
                'controller_path': str(controller_path),
                'controller_sha256': sha(controller_path),
                'selector_sha256': sha(selector_path),
            })
    if len(bindings) != 77 or len({r['history_item_id'] for r in bindings}) != 77 or len({r['candidate_id'] for r in bindings}) != 77:
        raise ValueError('expected 77 unique IDs and candidate keys')
    result = {
        'schema_version': 'history_private_binding_audit_v1',
        'status': 'PRIVATE_ID_BINDINGS_VERIFIED_SEMANTIC_REVIEW_PENDING',
        'visibility': 'controller_only',
        'inventory_path': str(INVENTORY),
        'inventory_sha256': sha(INVENTORY),
        'row_count': len(bindings),
        'inherited_parent_root_count': sum(r['root_binding_source'] == 'root_chunks.parent_root_id' for r in bindings),
        'records': bindings,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'path': str(OUT), 'sha256': sha(OUT), 'rows': len(bindings), 'inherited_parent_roots': result['inherited_parent_root_count']}, indent=2))

if __name__ == '__main__':
    main()
