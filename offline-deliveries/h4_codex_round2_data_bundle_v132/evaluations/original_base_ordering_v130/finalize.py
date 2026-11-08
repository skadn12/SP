"""Verify completed audits and seal the base control; never run inference."""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'h4_codex_dev_eval_v127'
PRIOR_SEAL_SHA = '4fb446adcf81d8ce6db72accf78dd3e47b64c2e2548add3191561856e8389d61'


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
    summary = read(HERE / 'SUMMARY.json')
    run = read(HERE / 'RUN_STATUS.json')
    assert summary['status'] == 'COMPLETE' and not summary['errors']
    assert run['status'] == 'BASE_CONTROL_REPORTED_AWAITING_INDEPENDENT_AUDIT'
    assert run['pid'] is None and run['actual_new_calls'] == 168
    audit_paths = [HERE / 'evidence' / name for name in [
        'INDEPENDENT_PREFLIGHT_AUDIT.json',
        'INDEPENDENT_FINAL_AUDIT.json',
        'REPORT_ARTIFACT_AUDIT.json',
    ]]
    expected = {}
    for audit_path in audit_paths:
        audit = read(audit_path)
        assert audit['status'] == 'PASS' and not audit.get('errors'), str(audit_path)
        assert audit['audited_files'], 'AUDIT_WITHOUT_BINDINGS'
        for row in audit['audited_files']:
            path = Path(row['path'])
            assert path.is_absolute()
            previous = expected.setdefault(str(path), row['sha256'])
            assert previous == row['sha256'], 'CONFLICTING_AUDIT_BINDING:' + str(path)
    for path, expected_sha in expected.items():
        assert sha(path) == expected_sha, 'AUDITED_FILE_CHANGED:' + path
    assert sha(PRIOR / 'COMPLETE_MANIFEST.json') == PRIOR_SEAL_SHA
    for row in read(PRIOR / 'COMPLETE_MANIFEST.json')['files']:
        assert sha(PRIOR / row['path']) == row['sha256'], 'PRIOR_DELIVERY_CHANGED'
    for model in ['base', 'baseline', 'new']:
        stats = summary['models'][model]['summary']
        assert stats['planned_calls'] == stats['valid_calls'] == 168
        assert stats['invalid_or_missing_calls'] == 0
        assert stats['pair_counts']['unknown_pairs'] == 0
    gpu = subprocess.run([
        'nvidia-smi', '--query-gpu=name,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits',
    ], check=True, capture_output=True, text=True).stdout.strip()
    assert gpu and all(int(line.split(',')[-2].strip()) == 0 for line in gpu.splitlines()), 'GPU_NOT_RELEASED'
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save(HERE / 'RESOURCE_STATUS.json', {'checked_at_utc': now,
        'gpu_query': gpu, 'gpu_memory_released': True})
    save(HERE / 'FINAL_STATUS.json', {
        'schema_version': 'v130_completed_base_order_control_v1',
        'status': 'DEV_BASE_CONTROL_COMPLETED_AND_INDEPENDENTLY_AUDITED',
        'created_at_utc': now, 'questions': 14, 'fixed_dev_roots': 15,
        'covered_dev_roots': 7, 'uncovered_dev_roots': 8,
        'new_inference_calls': 168, 'reused_prior_calls': 336,
        'comparison_calls': 504, 'invalid_missing_duplicate_calls': 0,
        'test_evaluation': False, 'test_question_content_read_this_run': False,
        'new_training': False, 'new_question_generation': False,
        'automatic_retry': False, 'selection_based_filtering': False,
        'prior_sealed_artifacts_modified': False,
        'base_elapsed_seconds': run['base_elapsed_seconds'],
        'controller_elapsed_seconds': run['elapsed_seconds'],
        'results': {key: {
            'solver_model': value['solver_model'],
            'model_path': value['model_path'],
            'exact_correct': value['summary']['exact_correct'],
            'exact_denominator': value['summary']['planned_calls'],
            'exact_rate': value['summary']['exact_rate_fixed'],
            'pair_accuracy': value['summary']['metrics']['pair_accuracy']['mean'],
        } for key, value in summary['models'].items()},
        'summary_binding': binding(HERE / 'SUMMARY.json'),
        'audit_bindings': [binding(path) for path in audit_paths],
        'verified_unique_audit_binding_count': len(expected),
        'gpu_memory_released': True,
        'run_status_note': 'RUN_STATUS preserves the controller terminal state before independent final audit; FINAL_STATUS records completed independent audit and sealing.',
        'interpretation': 'Original base and first-trained model both have low exact ordering accuracy on this fixed template. This does not support first-training degradation of previously successful whole-order ability. No causal attribution or significance claim; new-model ceiling is specific to 14 correlated DEV questions from 7 roots.',
    })
    files = []
    for path in sorted(HERE.rglob('*')):
        if path.is_file():
            files.append({'path': str(path.relative_to(HERE)),
                          'bytes': path.stat().st_size, 'sha256': sha(path)})
    save(HERE / 'COMPLETE_MANIFEST.json', {
        'schema_version': 'v130_complete_manifest_v1', 'status': 'SEALED',
        'created_at_utc': now, 'file_count': len(files),
        'total_bytes': sum(row['bytes'] for row in files),
        'excludes': ['COMPLETE_MANIFEST.json'], 'files': files,
    })
    for row in files:
        assert sha(HERE / row['path']) == row['sha256'], 'SEAL_VERIFICATION_FAILED'
    print(json.dumps({'status': 'SEALED', 'files': len(files),
                      'manifest_sha256': sha(HERE / 'COMPLETE_MANIFEST.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
