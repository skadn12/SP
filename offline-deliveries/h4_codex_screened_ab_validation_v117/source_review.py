"""Record a real cumulative-source read and explicit inherited constraints review.

The original eight scene checks retain their schema and meaning. The additional
constraint record attests a reader's judgments; Python checks only completeness
and bindings and never supplies a semantic PASS.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
BATCH = Path(__file__).resolve().parent
REVIEWER = os.environ.get('CODEX_REVIEWER', '/root')
CHECKS = {'time', 'location', 'actor_identity', 'plan_vs_completion',
          'cross_scene_continuity', 'semantic_repetition', 'naturalness', 'event_fidelity'}
STATUSES = {'PASS', 'WARNING', 'PASS_WITH_WARNING', 'FAIL'}


def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bind(path): return {'path': str(Path(path).resolve()), 'sha256': digest(path)}
def resolve(path): return Path(path) if Path(path).is_absolute() else BATCH / path


def folder(root, scene):
    assert scene in {f'R02:{route}:S{n:02d}' for route in 'AB' for n in range(1, 4)}
    return BATCH / 'roots' / ('root' + root.rsplit('-', 1)[-1]) / 'scenes' / scene.removeprefix('R02:').replace(':', '_')


def constraints_for(root):
    plan = read(BATCH / 'PRECALL_PLAN.json')
    row = next(r for r in plan['roots'] if r['root_id'] == root)
    path = resolve(row['continuation_constraints_path'])
    assert digest(path) == row['continuation_constraints_sha256'], 'CONSTRAINT_BINDING_CHANGED'
    document = read(path)
    assert document['schema_version'] == 'codex_screened_root_constraints_v1'
    assert document['root_id'] == root and document['baseline_gate_version'] == 'v116'
    assert document['continuation_constraints'] == row['continuation_constraints']
    assert document['baseline_gate_decision'] == row['baseline_gate_decision']
    assert document['baseline_gate_decision'] in {'PASS', 'WARNING'}
    authority = document['selection_authority']
    assert digest(resolve(authority['path'])) == authority['sha256']
    selected = next(r for r in read(resolve(authority['path']))['roots'] if r['root_id'] == root)
    assert selected['eligible'] is True and selected['selected'] is True
    assert selected['continuation_constraints'] == document['continuation_constraints']
    assert selected['decision'] == document['baseline_gate_decision']
    assert document['review_bindings'] == selected['reviews']
    assert document['baseline_binding']['sha256'] == selected['baseline_sha256']
    for item in [document['baseline_binding'], *document['review_bindings']]:
        assert digest(resolve(item['path'])) == item['sha256'], 'SCREEN_INPUT_CHANGED:' + item['path']
    assert document['baseline_binding']['sha256'] == row['baseline_sha256']
    return path, document


def verify_batch_gate(config=None):
    plan = read(BATCH / 'PRECALL_PLAN.json')
    binding = plan['baseline_gate']
    gate_path = resolve(binding['path'])
    assert digest(gate_path) == binding['sha256'], 'BASELINE_GATE_CHANGED'
    gate = read(gate_path)
    assert gate['screen_version'] == 'v116'
    assert gate['generation_root_replacement_allowed'] is False
    expected = [r['root_id'] for r in plan['roots']]
    assert gate['fixed_root_ids'] == expected
    authority = gate['selection_authority']
    assert digest(resolve(authority['path'])) == authority['sha256']
    assert [r['root_id'] for r in read(resolve(authority['path']))['roots']] == expected
    for item in gate['screen_bindings']:
        assert digest(resolve(item['path'])) == item['sha256'], 'SCREEN_DELIVERY_CHANGED:' + item['path']
    if config is not None:
        assert [r['root_id'] for r in config['roots']] == expected, 'SCREENED_ROOT_ORDER_CHANGED'
    return [bind(gate_path), bind(resolve(authority['path'])),
            *[bind(constraints_for(root)[0]) for root in expected]]


def validate_constraint_judgment(judgment, document, turn_refs):
    """Validate the supplied judgment's shape, not its semantic truth."""
    assert judgment['status'] in STATUSES
    assert isinstance(judgment['reason'], str) and judgment['reason'].strip()
    rows = judgment['constraint_results']
    constraints = document['continuation_constraints']
    assert isinstance(rows, list) and len(rows) == len(constraints), 'CONSTRAINT_COVERAGE_INCOMPLETE'
    for index, (constraint, row) in enumerate(zip(constraints, rows), 1):
        assert row['index'] == index and row['constraint'] == constraint, 'CONSTRAINT_ORDER_OR_TEXT_CHANGED'
        assert row['status'] in STATUSES
        assert isinstance(row['reason'], str) and row['reason'].strip()
        assert isinstance(row['evidence_turn_refs'], list) and row['evidence_turn_refs']
        assert len(row['evidence_turn_refs']) == len(set(row['evidence_turn_refs']))
        assert set(row['evidence_turn_refs']).issubset(turn_refs)
    assert isinstance(judgment['new_blocking_findings'], list)
    failed = any(row['status'] == 'FAIL' for row in rows) or bool(judgment['new_blocking_findings'])
    if failed:
        assert judgment['status'] == 'FAIL', 'NEW_CONSTRAINT_FAILURE_NOT_REFLECTED'
    if judgment['status'] == 'PASS':
        assert all(row['status'] == 'PASS' for row in rows)
    return failed


def verify_scene_constraints(root, scene, review):
    path, document = constraints_for(root)
    record = review['baseline_constraint_review']
    assert record['constraints_binding'] == bind(path)
    assert record['baseline_gate_version'] == 'v116'
    assert record['inherited_baseline_decision'] == document['baseline_gate_decision']
    source = read(folder(root, scene) / 'CUMULATIVE_SOURCE.json')
    failed = validate_constraint_judgment(record, document, {t['turn_ref'] for t in source['turns']})
    if review['decision'] == 'READY':
        assert not failed and record['status'] != 'FAIL', 'READY_WITH_NEW_CONSTRAINT_VIOLATION'
    return record


def context_path(root, scene, reviewer):
    identity = hashlib.sha256(reviewer.encode()).hexdigest()[:12]
    return Path('/tmp') / (BATCH.name + '-' + root.rsplit('-', 1)[-1] + '-' +
                            scene.replace(':', '_') + '-' + identity + '-read.json')


def start(root, scene, reviewer=REVIEWER):
    d = folder(root, scene); source = d / 'CUMULATIVE_SOURCE.json'; data = read(source)
    constraints_path, document = constraints_for(root)
    assert reviewer.strip() and reviewer != read(d / 'EXPANSION_DRAFT.json')['author']
    context = {'started_at_utc': now(), 'source_sha256': digest(source), 'reviewer': reviewer,
               'constraints_binding': bind(constraints_path)}
    path = context_path(root, scene, reviewer)
    if ':B:' in scene:
        reservation = next(r for r in read(BATCH / 'SOURCE_RESERVATIONS.json')['allocation']
                           if r['root_id'] == root and r['scene_ref'] == scene)
        event_path = resolve(reservation['source_item_path'])
        assert digest(event_path) == reservation['source_item_sha256']
        selection_path = BATCH / 'selection' / ('root' + root.rsplit('-', 1)[-1] + '.json')
        context.update(native_event_binding=bind(event_path), selection_binding=bind(selection_path))
    with path.open('x') as stream: json.dump(context, stream, ensure_ascii=False, indent=2)
    print('BASELINE_CONSTRAINTS_FULL', json.dumps(document, ensure_ascii=False))
    for turn in data['turns']:
        print(turn['turn_ref'] + ' | ' + turn['speaker'] + ': ' + turn['text'])
    print('DRAFT_NOTES', json.dumps({k: v for k, v in read(d / 'EXPANSION_DRAFT.json').items()
                                     if k != 'turns'}, ensure_ascii=False))
    if ':B:' in scene:
        print('NATIVE_EVENT_FULL', json.dumps(read(event_path), ensure_ascii=False))
        selected = next(a for a in read(selection_path)['allocations'] if a['scene_ref'] == scene)
        print('SELECTION', json.dumps({k: v for k, v in selected.items() if k != 'native_event'}, ensure_ascii=False))
    print('READ_CONTEXT', str(path))


def save_review(root, scene, judgment, reviewer=REVIEWER):
    d = folder(root, scene); source_path = d / 'CUMULATIVE_SOURCE.json'; source = read(source_path)
    context = read(context_path(root, scene, reviewer))
    assert context['reviewer'] == reviewer and digest(source_path) == context['source_sha256']
    assert reviewer.strip() and reviewer != read(d / 'EXPANSION_DRAFT.json')['author']
    constraints_path, document = constraints_for(root)
    assert context['constraints_binding'] == bind(constraints_path)
    supplied = read(judgment)
    assert set(supplied) == {'decision', 'checks', 'findings', 'baseline_constraint_review'}, 'JUDGMENT_FIELDS'
    assert set(supplied['checks']) == CHECKS
    assert supplied['decision'] in {'READY', 'HOLD'} and isinstance(supplied['findings'], list)
    for name, check in supplied['checks'].items():
        assert check['status'] in STATUSES and isinstance(check['reason'], str) and check['reason'].strip(), name
    constraint_review = supplied['baseline_constraint_review']
    assert set(constraint_review) == {'status', 'reason', 'constraint_results', 'new_blocking_findings'}
    constraint_review.update(constraints_binding=bind(constraints_path), baseline_gate_version='v116',
                             inherited_baseline_decision=document['baseline_gate_decision'])
    end = now()
    review = {'root_id': root, 'scene_ref': scene, 'source_sha256': digest(source_path),
              'reviewer': reviewer, 'review_role': 'non_author_complete_cumulative_source_and_native_event_reviewer',
              'backend': 'CURRENT_CODEX_SESSION', 'exact_model_id': None, 'external_semantic_certification': False,
              'started_at_utc': context['started_at_utc'], 'finished_at_utc': end, 'created_at_utc': end,
              'full_source_read': True, 'turn_refs_read': [t['turn_ref'] for t in source['turns']],
              **supplied, 'semantic_review_is_programmatic': False}
    verify_scene_constraints(root, scene, review)
    if review['decision'] == 'READY':
        assert all(c['status'] != 'FAIL' for c in review['checks'].values())
        assert not any(f.get('severity') == 'BLOCKING' for f in review['findings'])
    if ':B:' in scene:
        event_binding = context['native_event_binding']; selection_binding = context['selection_binding']
        assert digest(event_binding['path']) == event_binding['sha256']
        assert digest(selection_binding['path']) == selection_binding['sha256']
        event = read(event_binding['path'])
        reservation = next(r for r in read(BATCH / 'SOURCE_RESERVATIONS.json')['allocation']
                           if r['root_id'] == root and r['scene_ref'] == scene)
        assert resolve(reservation['source_item_path']).resolve() == Path(event_binding['path'])
        assert reservation['source_item_sha256'] == event_binding['sha256']
        review.update(native_event_full_read=True, native_event_path=event_binding['path'],
                      native_event_sha256=event_binding['sha256'],
                      native_turn_ids_read=[t['turn_id'] for t in event['turns']], source_id=reservation['source_id'],
                      selection_path=selection_binding['path'], selection_sha256=selection_binding['sha256'])
    with (d / 'SOURCE_REVIEW.json').open('x') as stream:
        json.dump(review, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(d / 'SOURCE_REVIEW.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['read', 'save'])
    parser.add_argument('root'); parser.add_argument('scene'); parser.add_argument('judgment', nargs='?')
    parser.add_argument('--reviewer', default=REVIEWER)
    args = parser.parse_args()
    if args.action == 'read': start(args.root, args.scene, args.reviewer)
    else:
        assert args.judgment, 'JUDGMENT_REQUIRED'
        save_review(args.root, args.scene, args.judgment, args.reviewer)
