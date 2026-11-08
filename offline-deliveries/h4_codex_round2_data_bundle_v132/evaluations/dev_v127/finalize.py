"""Seal the completed, independently audited DEV comparison without inference."""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDITS = [
    'INDEPENDENT_PREFLIGHT_AUDIT.json',
    'INDEPENDENT_FINAL_AUDIT.json',
    'INDEPENDENT_METRIC_AUDIT.json',
    'INDEPENDENT_EXECUTION_AUDIT.json',
]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main():
    assert not (HERE / 'FINAL_STATUS.json').exists(), 'ALREADY_FINALIZED'
    assert not (HERE / 'COMPLETE_MANIFEST.json').exists(), 'ALREADY_SEALED'
    status = read(HERE / 'RUN_STATUS.json')
    summary = read(HERE / 'PAIRED_SUMMARY.json')
    assert status['status'] == 'EVALUATION_REPORTED_AWAITING_INDEPENDENT_AUDIT'
    assert status['pid'] is None and status['active_process_exit_code'] == 0
    assert summary['status'] == 'COMPLETE' and summary['errors'] == []
    assert status['actual_calls'] == 1560 and status['test_evaluation'] is False

    # Recheck the union once, including external model/input bindings.
    bindings = {}
    for name in AUDITS:
        audit = read(HERE / 'evidence' / name)
        assert audit['status'] == 'PASS' and not audit.get('errors'), name
        assert not audit.get('inputs_changed_during_audit'), name
        for row in audit['audited_files']:
            previous = bindings.setdefault(row['path'], row['sha256'])
            assert previous == row['sha256'], 'CONFLICTING_AUDIT_BINDING:' + row['path']
    for path, expected in bindings.items():
        assert sha(path) == expected, 'AUDITED_FILE_CHANGED:' + path

    for side in ['baseline', 'new']:
        run = HERE / 'run' / side
        complete = read(run / 'COMPLETE.json')
        assert complete['status'] == 'COMPLETE' and complete['actual_call_count'] == 780
        assert not (run / 'FAILED.json').exists()
        rows = [read(path) for path in (run / 'responses').glob('*.json')]
        assert len(rows) == len({row['request_id'] for row in rows}) == 780
        assert all(row['evaluation_valid'] is True for row in rows)
    execution = read(HERE / 'evidence/INDEPENDENT_EXECUTION_AUDIT.json')
    assert execution['execution_checks']['controller_and_observed_children_exited'] is True

    created = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save(HERE / 'FINAL_STATUS.json', {
        'schema_version': 'v127_completed_dev_evaluation_v1',
        'status': 'DEV_EVALUATED_AND_INDEPENDENTLY_AUDITED',
        'created_at_utc': created,
        'models': {'baseline': 'M_clean_policy', 'new': 'M_v125_newonly'},
        'fixed_dev_roots': 15, 'covered_dev_roots': 7, 'uncovered_dev_roots': 8,
        'questions': 48, 'gold2345_multi_select': 27, 'gold6_controls': 7, 'ordering': 14,
        'actual_calls_per_model': 780, 'actual_calls_total': 1560,
        'invalid_missing_duplicate_calls': 0,
        'test_evaluation': False, 'automatic_retry': False,
        'selection_based_filtering': False,
        'source_training_artifacts_modified': False,
        'side_elapsed_seconds': status['side_elapsed_seconds'],
        'controller_elapsed_seconds': (
            datetime.datetime.fromisoformat(status['finished_at_utc']) -
            datetime.datetime.fromisoformat(status['started_at_utc'])).total_seconds(),
        'summary_binding': binding(HERE / 'PAIRED_SUMMARY.json'),
        'audit_bindings': [binding(HERE / 'evidence' / name) for name in AUDITS],
        'verified_audit_binding_union_count': len(bindings),
        'preflight_audit_attempt_history': (
            'Initial audit-only failure retained: historical absolute/relative seal paths. '
            'Unfrozen audit script corrected, then PASS before any inference; '
            'frozen evaluator, inputs and protocol unchanged. No inference rerun.'),
        'interpretation': (
            'Descriptive paired DEV result on 48 questions from 7 covered roots. '
            'Repeated permutations, rollouts and shared-root questions are correlated; '
            'this is not a significance or generalization claim. TEST not evaluated.'),
    })
    files = []
    for path in sorted(HERE.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        if path.name == 'COMPLETE_MANIFEST.json':
            continue
        row = binding(path)
        row['path'] = str(path.relative_to(HERE))
        files.append(row)
    manifest = {
        'schema_version': 'v127_complete_manifest_v1',
        'status': 'COMPLETE', 'created_at_utc': created,
        'scope': 'Frozen paired DEV evaluation, raw responses, models identities and independent audits.',
        'files': files, 'file_count': len(files),
        'total_bytes': sum(row['bytes'] for row in files),
        'excluded': ['COMPLETE_MANIFEST.json itself', '__pycache__'],
        'external_files': 'Bound and independently verified through preflight/model/final audit manifests.',
    }
    save(HERE / 'COMPLETE_MANIFEST.json', manifest)
    print(json.dumps({'status': 'COMPLETE', 'files': len(files),
                      'manifest_sha256': sha(HERE / 'COMPLETE_MANIFEST.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
