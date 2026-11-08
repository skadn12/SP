"""Verify independently audited frontier results, then seal immutable delivery."""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
V127 = HERE.parent / 'h4_codex_dev_eval_v127'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
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
    summary, status = read(HERE / 'SUMMARY.json'), read(HERE / 'RUN_STATUS.json')
    assert summary['status'] == 'COMPLETE' and not summary['errors']
    assert set(summary['models']) == {'historical_m0', 'historical_clean', 'live_clean', 'live_new'}
    assert summary['population']['questions'] == 123
    assert summary['population']['train_roots'] == 58
    assert summary['population']['planned_calls_per_model'] == 1629
    assert status['status'] == 'REPORTED_AWAITING_INDEPENDENT_AUDIT'
    assert status['actual_new_calls'] == 3258 and status['pid'] is None
    for side in ['live_clean', 'live_new']:
        execution = read(HERE / (side + '_STATUS.json'))
        assert execution['status'] == 'COMPLETE' and execution['actual_calls'] == 1629
        assert execution['pid'] is None
    audit_names = ['INDEPENDENT_PREFLIGHT_AUDIT.json', 'QUALITY_PREFLIGHT_AUDIT.json',
                   'INDEPENDENT_FINAL_AUDIT.json', 'QUALITY_FINAL_AUDIT.json']
    audits = [HERE / 'evidence' / name for name in audit_names]
    hashes = {}
    for path in audits:
        audit = read(path)
        assert audit['status'] == 'PASS' and not audit.get('errors'), str(path)
        assert audit['audited_files']
        for row in audit['audited_files']:
            assert Path(row['path']).is_absolute()
            old = hashes.setdefault(row['path'], row['sha256'])
            assert old == row['sha256'], 'AUDIT_BINDING_CONFLICT:' + row['path']
    for path, digest in hashes.items():
        assert sha(path) == digest, 'AUDITED_FILE_CHANGED:' + path
    assert sha(V127 / 'COMPLETE_MANIFEST.json') == '4fb446adcf81d8ce6db72accf78dd3e47b64c2e2548add3191561856e8389d61'
    for row in read(V127 / 'COMPLETE_MANIFEST.json')['files']:
        assert sha(V127 / row['path']) == row['sha256'], 'V127_MODIFIED'
    for side, model in summary['models'].items():
        stats = model['summary']
        assert stats['planned_calls'] == stats['valid_calls'] == 1629
        assert stats['invalid_or_missing_calls'] == 0
        assert stats['ordering']['unknown_pairs'] == 0
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,utilization.gpu',
                          '--format=csv,noheader,nounits'],
                         check=True, capture_output=True, text=True).stdout.strip()
    assert gpu and all(int(row.split(',')[-2].strip()) == 0 for row in gpu.splitlines()), 'GPU_NOT_RELEASED'
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save(HERE / 'RESOURCE_STATUS.json', {'checked_at_utc': timestamp,
        'gpu_query': gpu, 'gpu_memory_released': True})
    save(HERE / 'FINAL_STATUS.json', {
        'schema_version': 'v131_historical_frontier_completed_v1',
        'status': 'HISTORICAL_TRAIN_FRONTIER_EVALUATED_AND_INDEPENDENTLY_AUDITED',
        'created_at_utc': timestamp, 'questions': 123, 'train_roots': 58,
        'question_types': {'ordering': 62, 'multi_select': 51, 'single_choice': 10},
        'new_calls': 3258, 'reused_historical_calls': 3258, 'comparison_calls': 6516,
        'new_calls_per_model': 1629, 'context_mode': 'full_context',
        'invalid_missing_duplicate_calls': 0, 'test_evaluation': False,
        'new_training': False, 'new_question_generation': False,
        'gold_modification': False, 'automatic_retry': False,
        'score_based_filtering_of_selected_set': False,
        'historical_models': ['M0', 'M_clean_policy'],
        'current_runtime_models': ['M_clean_policy', 'M_v125_newonly'],
        'runtime_bridge': 'Primary comparison re-evaluates M_clean and the new model under the same vLLM 0.21 runtime; historical 0.10.2 scores remain separate.',
        'elapsed_seconds': status['elapsed_seconds'],
        'side_elapsed_seconds': {side: read(HERE / (side + '_STATUS.json'))['elapsed_seconds'] for side in ['live_clean', 'live_new']},
        'results_per_question_equal_weight': {model: value['equal_question']
            for model, value in summary['strata']['all123']['models'].items()},
        'results_call_weighted': {model: value['pooled_calls']
            for model, value in summary['strata']['all123']['models'].items()},
        'summary_binding': binding(HERE / 'SUMMARY.json'),
        'audit_bindings': [binding(path) for path in audits],
        'verified_unique_audit_binding_count': len(hashes),
        'gpu_memory_released': True,
        'controller_status_note': 'RUN_STATUS retains the controller state before independent final audit; FINAL_STATUS records subsequent audit completion.',
        'limitations': ['Outcome-selected historical Train diagnostic; the old gain is partly guaranteed by selection.',
                       'Original questions/Gold/prompts/IDs/seeds were replayed without edits; this run does not certify historical Gold truth.',
                       'Same-version historical semantic accepts were located for 2 questions; 121 have no applicable semantic audit. Unknown does not mean valid.',
                       'Questions were not directly used in first-round training, while roots and related later training content can overlap.',
                       'No exact question or prompt overlap with the 363 new training records was found; 57/58 roots and 120/123 question roots overlap that set.',
                       'Repeated calls and shared-root questions are correlated; no significance or general DEV/TEST claim.'],
    })
    files = [{'path': str(path.relative_to(HERE)), 'bytes': path.stat().st_size,
              'sha256': sha(path)} for path in sorted(HERE.rglob('*')) if path.is_file()]
    save(HERE / 'COMPLETE_MANIFEST.json', {'schema_version': 'v131_complete_manifest_v1',
        'status': 'SEALED', 'created_at_utc': timestamp, 'file_count': len(files),
        'total_bytes': sum(row['bytes'] for row in files),
        'excludes': ['COMPLETE_MANIFEST.json'], 'files': files})
    for row in files:
        assert sha(HERE / row['path']) == row['sha256'], 'SEAL_VERIFICATION_FAILED'
    print(json.dumps({'status': 'SEALED', 'files': len(files),
                      'manifest_sha256': sha(HERE / 'COMPLETE_MANIFEST.json')}))


if __name__ == '__main__':
    main()
