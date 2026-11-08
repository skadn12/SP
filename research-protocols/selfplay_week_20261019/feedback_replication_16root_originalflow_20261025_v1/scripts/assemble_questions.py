"""Assemble actual admissions and explicit failures; no semantic decisions."""
import argparse
import collections
import json
from pathlib import Path
from workflow_io import BASE, SLOTS, bind, load, now, save, candidate_errors


def key(row):
    return tuple(row[k] for k in ('root_id', 'arm', 'slot_id'))


def assemble(freeze=False):
    manifest = load(BASE / 'CONTINUATION_MANIFEST.json')
    roots = manifest['roots']
    selected = {r['root_id']: {k: r[k] for k in ('path', 'sha256', 'bytes')}
                for r in load(BASE / 'SOURCE_SELECTION.json')['roots']}
    admitted, histories, bindings = {}, collections.defaultdict(list), []
    for row in load(BASE / 'REUSED_SLOTS.json')['slots']:
        assert row['status'] == 'PASS'
        admitted[key(row)] = row
        histories[key(row)].append({'stage': 'reused_admission', 'candidate_id': row['candidate_id']})
    for path in sorted((BASE / 'reviews').glob('*/ADMISSION.json')):
        doc = load(path)
        bindings.append(bind(path))
        for row in doc['slots']:
            k = key(row)
            histories[k].append({'stage': 'current_independent_review', 'status': row['status'],
                                 'candidate_id': row['candidate_id'], 'artifact': str(path),
                                 'errors': row['review']['errors']})
            if row['status'] != 'PASS':
                continue
            if k in admitted:
                assert admitted[k]['candidate'] == row['candidate'], ('Conflicting PASS', k)
            else:
                admitted[k] = row
    explicit = {}
    for path in sorted((BASE / 'questions').glob('*/*/SLOT_FAILURES.json')):
        doc = load(path)
        rows = doc if isinstance(doc, list) else doc.get('slots', doc.get('failures', []))
        bindings.append(bind(path))
        for row in rows:
            k = (path.parent.parent.name, path.parent.name, row['slot_id'])
            explicit[k] = {'artifact': bind(path), 'record': row}
            histories[k].append({'stage': 'author_failure', **explicit[k]})
    # Parent recovery failures from the previously frozen run are retained where
    # this continuation neither admits nor explicitly revises their slot.
    previous = Path(manifest['previous16'])
    for row in load(previous / 'QUESTION_FREEZE.json')['slots']:
        k = key(row)
        if row['status'] != 'PASS':
            histories[k].insert(0, {'stage': 'previous_run', 'status': row['status'],
                                    'failure_reasons': row.get('failure_reasons', []),
                                    'artifact': str(previous / 'QUESTION_FREEZE.json')})
    planned = []
    for root in roots:
        for arm in ('F', 'G'):
            for slot in SLOTS:
                k = (root, arm, slot)
                if k in admitted:
                    row = admitted[k]
                    candidate = row['candidate']
                    assert not candidate_errors(candidate), (k, candidate_errors(candidate))
                    assert candidate['source_binding'] == selected[root], ('Source drift', k)
                    assert bind(candidate['source_binding']['path']) == candidate['source_binding']
                    assert row['review']['status'] == 'PASS' and not row['review']['errors']
                    for receipt in ('gold_receipt', 'review_receipt'):
                        bindings.append(bind(row['review'][receipt]))
                else:
                    status = ('SOURCE_BLOCKED' if root not in selected else
                              'FAIL' if k in explicit else 'PENDING')
                    row = {'root_id': root, 'arm': arm, 'slot_id': slot, 'status': status,
                           'failure': explicit.get(k), 'attempt_history': histories[k]}
                planned.append(row)
    assert len(planned) == 256 and len({key(s) for s in planned}) == 256
    candidates = [r['candidate'] for r in planned if r['status'] == 'PASS']
    # Detect only whole public-question duplicates, without excluding shared facts.
    public_keys = ('question_type', 'task_operation', 'question', 'options', 'items')
    duplicate_groups = collections.defaultdict(list)
    for c in candidates:
        signature = json.dumps({k: c[k] for k in public_keys if k in c}, ensure_ascii=False, sort_keys=True)
        duplicate_groups[signature].append(c['candidate_id'])
    duplicates = [v for v in duplicate_groups.values() if len(v) > 1]
    assert not duplicates, ('Whole-question duplicates require an explicit disposition', duplicates)
    progress = {'at': now(), 'planned': len(planned), 'statuses': dict(collections.Counter(r['status'] for r in planned)),
                'by_arm': {a: dict(collections.Counter(r['status'] for r in planned if r['arm'] == a)) for a in ('F','G')},
                'by_root': {r: dict(collections.Counter(s['status'] for s in planned if s['root_id'] == r)) for r in roots},
                'pending': [{k: s[k] for k in ('root_id', 'arm', 'slot_id')} for s in planned if s['status'] == 'PENDING'],
                'whole_question_duplicates': duplicates, 'reused_semantic_slots': 73,
                'new_semantic_slots': len(candidates) - 73}
    save(BASE / 'QUESTION_PROGRESS.json', progress)
    if freeze:
        assert not progress['pending'], 'Unresolved slots cannot be frozen'
        target = BASE / 'F_TARGET_REALIZATION_PRE_SOLVER.json'
        assert target.exists(), 'Prospective F target review required before Solver'
        assert not load(target).get('pending_roots'), 'Prospective target review still pending'
        assert not (BASE / 'QUESTION_FREEZE.json').exists(), 'No freeze overwrite'
        for path in (BASE / 'questions').glob('*/*/*FREEZE.json'):
            doc = load(path)
            for name in ('plan_binding', 'task_plans', 'reused_plan'):
                if name in doc:
                    assert bind(doc[name]['path']) == doc[name], ('Plan drift', path)
            bindings.append(bind(path))
        bindings += [bind(BASE / p) for p in ('SOURCE_SELECTION.json', 'SOURCE_FREEZE.json', 'REUSED_SLOTS.json',
                                             'CONTINUATION_MANIFEST.json', 'ANALYSIS_CONTINUATION_FREEZE.json',
                                             'ORIGINAL_ANALYSIS_PLAN.json', 'F_TARGET_REALIZATION_PRE_SOLVER.json')]
        unique = {b['path']: b for b in bindings}
        save(BASE / 'CANDIDATES_PRIVATE.json', {'candidates': candidates})
        unique[str(BASE / 'CANDIDATES_PRIVATE.json')] = bind(BASE / 'CANDIDATES_PRIVATE.json')
        save(BASE / 'QUESTION_FREEZE.json', {'at': now(), 'adaptive_recovery': True,
             'planned_slots': 256, 'bindings': list(unique.values()), 'slots': planned})
    print(json.dumps(progress, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    assemble(parser.parse_args().freeze)
