#!/usr/bin/env python3
"""Recheck provenance, whitelist, completeness and immutable public input bytes."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = Path('/root/autodl-tmp/gxy-selfplay')
V131 = BASE / 'offline-deliveries/h4_codex_historical_frontier_eval_v131'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    checks, errors = [], []
    def check(label, ok):
        checks.append(label)
        if not ok:
            errors.append(label)
    for row in read(HERE / 'INPUT_BINDINGS.json')['files']:
        p = Path(row['path'])
        check('input_binding:' + str(p), p.is_file() and p.stat().st_size == row['bytes'] and sha(p) == row['sha256'])
    frozen = read(HERE / 'PUBLIC_FREEZE.json')
    for row in frozen['files']:
        p = HERE / row['path']
        check('public_freeze:' + row['path'], p.is_file() and p.stat().st_size == row['bytes'] and sha(p) == row['sha256'])
    actual = {str(p.relative_to(HERE)) for d in ['public', 'pilot', 'sources'] for p in (HERE / d).rglob('*') if p.is_file()}
    check('public_no_unbound_file', actual == {r['path'] for r in frozen['files']})
    old = read(V131 / 'QUESTIONS.json')['candidates']
    target = {q['candidate_id']: q for q in old if q['question_type'] in {'multi_select', 'ordering'}}
    inv = read(HERE / 'INVENTORY.json')['questions']
    check('inventory_unique113', len(inv) == len({q['candidate_id'] for q in inv}) == 113)
    check('type_counts', Counter(q['question_type'] for q in inv) == {'multi_select': 51, 'ordering': 62})
    check('root_count56', len({q['root_id'] for q in inv}) == 56)
    check('candidate_set_equal', {q['candidate_id'] for q in inv} == set(target))
    srcindex = {s['root_id']: s for s in read(HERE / 'SOURCE_INDEX.json')['roots']}
    for q in inv:
        cid = q['candidate_id']
        check('lossless:' + cid, q['original_question_record'] == target[cid])
        check('pending_not_admitted:' + cid, q['content_status'] == 'PENDING_REVIEW' and q['training_admitted'] is False)
        check('train_namespace:' + cid, q['split'] == 'train' and q['split_namespace'] == 'gxy-selfplay:h4-round0-b0-k3-difficulty-v1')
        check('source_binding:' + cid, q['source_binding'] == srcindex[q['root_id']]['source_binding'])
        pub = {k: target[cid][k] for k in frozen['public_question_schema'] if k != 'source_binding'}
        pub['source_binding'] = q['source_binding']
        q['_expected_public'] = pub
    expected = {q['candidate_id']: q['_expected_public'] for q in inv}
    public = read(HERE / 'public/QUESTIONS.json')['questions']
    check('public113_unique', len(public) == len({q['candidate_id'] for q in public}) == 113)
    check('public_fields_projection', all(q == expected[q['candidate_id']] for q in public))
    chunks = read(HERE / 'public/CHUNK_INDEX.json')['chunks']
    flat, rootowners = [], {}
    for chunk in chunks:
        qs = read(chunk['binding']['path'])['questions']
        check('chunk_binding:' + chunk['chunk_id'], sha(chunk['binding']['path']) == chunk['binding']['sha256'])
        check('chunk_size:' + chunk['chunk_id'], len(qs) == chunk['question_count'] <= 12)
        check('chunk_projection:' + chunk['chunk_id'], all(q == expected[q['candidate_id']] for q in qs))
        rs = {q['root_id'] for q in qs}
        check('chunk_roots:' + chunk['chunk_id'], rs == set(chunk['root_ids']))
        for r in rs:
            check('root_not_split:' + r, r not in rootowners)
            rootowners[r] = chunk['chunk_id']
        flat.extend(q['candidate_id'] for q in qs)
    check('chunks_exactly113', len(flat) == len(set(flat)) == 113 and set(flat) == set(target))
    roots = sorted({q['root_id'] for q in public})
    pilot_roots = [r for r in roots if {q['question_type'] for q in public if q['root_id'] == r} == {'multi_select', 'ordering'}][:3]
    pilot_ids = [min(q['candidate_id'] for q in public if q['root_id'] == r and q['question_type'] == typ)
                 for r in pilot_roots for typ in ['multi_select', 'ordering']]
    check('pilot_rule6', read(HERE / 'pilot/PILOT_SELECTION.json')['candidate_ids'] == pilot_ids and len(pilot_ids) == 6)
    pilot = read(HERE / 'pilot/PUBLIC_QUESTIONS.json')['questions']
    check('pilot_public_projection', pilot == [expected[cid] for cid in pilot_ids])
    applicable = [q for q in inv if q['historical_review_lookup']['applicable_reviews']]
    check('historical_accept2_not_admission', len(applicable) == 2 and all(not q['training_admitted'] for q in applicable))
    for q in applicable:
        for r in q['historical_review_lookup']['applicable_reviews']:
            check('historical_review_binding:' + q['candidate_id'], sha(r['review_path']) == r['review_file_sha256'])
    result = {'status': 'PASS' if not errors else 'FAIL', 'check_count': len(checks),
              'errors': errors, 'error_count': len(errors),
              'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'semantic_review': False, 'independent_audit': False,
              'note': 'Mechanical recheck by inventory builder, no model calls or semantic admission.'}
    (HERE / 'VALIDATION_RECHECK.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result))
    raise SystemExit(1 if errors else 0)


if __name__ == '__main__':
    main()
