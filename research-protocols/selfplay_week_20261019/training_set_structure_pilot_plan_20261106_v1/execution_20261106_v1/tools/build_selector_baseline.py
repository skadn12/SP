#!/usr/bin/env python3
"""Build fixed coverage role inputs from inventory and independent reviews."""
import collections
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parents[1]
WEEK = HERE.parent.parent
FG = WEEK / 'training_set_rebuild_20261105_FG_exploration_run01'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def rows(value):
    if isinstance(value, list):
        yield from (r for r in value if isinstance(r, dict))
    elif isinstance(value, dict):
        if 'candidate_id' in value and 'question_type' in value:
            yield value
        for key in ('candidates', 'items', 'records'):
            if isinstance(value.get(key), list):
                yield from rows(value[key])


def display(row):
    result = {k: row[k] for k in ('candidate_id', 'root_id', 'question_type', 'question',
                                    'natural_information_need', 'cognitive_task', 'event_identity',
                                    'query_scope', 'query_pairs', 'answer_semantics') if k in row}
    if not result.get('question') and row.get('question_and_displayed_options'):
        result['question'] = row['question_and_displayed_options']
    for key in ('options', 'items'):
        if key in row:
            value = row[key]
            if isinstance(value, dict):
                result[key] = value
            elif isinstance(value, list):
                result[key] = [{k: x[k] for k in ('option_id', 'id', 'text') if k in x}
                               if isinstance(x, dict) else x for x in value]
    # Reference clues are citations, without stored correctness/Gold judgments.
    refs = []
    def collect(x):
        if isinstance(x, dict):
            if 'turn_ref' in x:
                refs.append({k: x[k] for k in ('turn_ref', 'quote', 'text', 'origin') if k in x})
            for y in x.values():
                collect(y)
        elif isinstance(x, list):
            for y in x:
                collect(y)
    for key in ('evidence', 'option_evidence', 'item_evidence'):
        collect(row.get(key))
    unique = {json.dumps(r, sort_keys=True, ensure_ascii=False): r for r in refs}
    result['source_evidence_clues'] = list(unique.values())
    return result


def main():
    coverage = HERE / 'manifests/HISTORICAL_COVERAGE.json'
    targeted = HERE / 'review/AUDIT_OPERATION_CODING_8ROOTS.json'
    sealed_path = HERE / 'review/HISTORICAL_SEALED135_OPERATION_AUDIT.json'
    assert sealed_path.is_file(), 'INDEPENDENT_SEALED135_REVIEW_REQUIRED'
    inventory, audit, sealed = read(coverage), read(targeted), read(sealed_path)
    freeze = read(HERE / 'manifests/EXECUTION_FREEZE.json')
    roots = [s['root_id'] for s in freeze['source_manifest']]
    by_root_audit = collections.defaultdict(list)
    for r in audit['records']:
        by_root_audit[r['root_id']].append(r)
    bindings = {r['review_packet_id']: r for r in read(FG / 'private_review_bindings.json')}
    projection_path = FG / 'private_semantic_reference/SEALED135_QUESTION_PROJECTION_ADDENDUM.json'
    projection = read(projection_path)['items']
    manifest = []
    for root in roots:
        ref_path = FG / 'private_semantic_reference' / ('REFERENCE_' + root + '.json')
        records = list(rows(read(ref_path)))
        records += [r for r in projection if r.get('root_id') == root]
        for packet in bindings.values():
            if packet['root_id'] == root:
                p = FG / 'blind_review_packets' / (packet['review_packet_id'] + '.json')
                r = dict(read(p)['candidate'])
                r['candidate_id'] = packet['candidate_id']
                records.append(r)
        # Include every named derived record bound by the repaired inventory,
        # retaining semantic variations and recording any unresolvable ID.
        for item in inventory['items']:
            if item.get('root_id') != root:
                continue
            artifact = item.get('artifact') or {}
            p = Path(artifact.get('path', '/does-not-exist'))
            if p.suffix == '.json' and p.is_file():
                records += [r for r in rows(read(p)) if r.get('candidate_id') == item['candidate_id']]
        grouped = collections.defaultdict(list)
        for record in records:
            c = display(record)
            c.setdefault('root_id', root)
            if not c.get('question') and not c.get('options') and not c.get('items'):
                continue
            key = json.dumps(c, sort_keys=True, ensure_ascii=False)
            if not any(x[0] == key for x in grouped[c.get('candidate_id')]):
                grouped[c.get('candidate_id')].append((key, c))
        result = []
        for cid in sorted({r['candidate_id'] for r in by_root_audit[root]} | set(grouped)):
            coding = [{k: r[k] for k in ('independent_operation_family', 'independent_relation_family',
                                         'confidence', 'reason') if k in r}
                      for r in by_root_audit[root] if r['candidate_id'] == cid]
            unique_codes = {json.dumps(r, sort_keys=True, ensure_ascii=False): r for r in coding}
            result.append({'historical_candidate_id': cid,
                           'semantic_representations': [c for _, c in grouped[cid]],
                           'independent_operation_review': list(unique_codes.values()),
                           'coverage_unknown': not bool(grouped[cid]),
                           'quality_status': 'Historical reference only; not quality approval or training admission.'})
        path = HERE / 'manifests/selector_baseline' / (root + '.json')
        write(path, {'root_id': root, 'records': result, 'unique_candidate_ids': len(result),
                     'coverage_policy': 'Unknown legacy entries do not prove absence. Compare natural information need, event/scope and actual sufficient answer operation, not exact label strings. This baseline is fixed across all four groups and never updated with pilot outputs.',
                     'inventory_sha256': sha(coverage), 'independent_root_audit_sha256': sha(targeted),
                     'sealed_operation_audit_sha256': sha(sealed_path),
                     'source_reference_sha256': sha(ref_path)})
        manifest.append({'root_id': root, 'path': str(path), 'sha256': sha(path),
                         'unique_candidate_ids': len(result)})
    write(HERE / 'manifests/SELECTOR_BASELINE_FREEZE.json', {
        'status': 'FROZEN_FOR_REMAINING_UNSTARTED_OPPORTUNITIES',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'inventory': {'path': str(coverage), 'sha256': sha(coverage)},
        'root_operation_audit': {'path': str(targeted), 'sha256': sha(targeted)},
        'sealed_operation_audit': {'path': str(sealed_path), 'sha256': sha(sealed_path)},
        'roots': manifest,
        'scope': 'Original pre-pilot historical corpus, repaired transparently after incomplete initial index. The six already-attempted opportunities retain timing/protocol deviation flags. No in-pilot outputs enter historical coverage.'})
    print(json.dumps({'roots': manifest, 'new_model_calls': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
