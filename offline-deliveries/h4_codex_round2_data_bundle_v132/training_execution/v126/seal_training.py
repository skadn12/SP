"""Freeze completed training outputs after independent audit and final reporting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RUN = REPO / 'restart-n5-runs/h4_codex_combined_sft_v125'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def manifest(folder):
    return [{'path': str(p.relative_to(folder)), 'sha256': sha(p), 'size_bytes': p.stat().st_size}
            for p in sorted(folder.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
            and p.name != 'COMPLETE_MANIFEST.json']


def main():
    assert read(HERE / 'RUN_STATUS.json')['status'] == 'TRAINING_COMPLETE_AWAITING_EVALUATION'
    audit = read(HERE / 'evidence/INDEPENDENT_FINAL_TRAINING_AUDIT.json')
    assert audit['status'] == 'PASS' and audit['errors'] == []
    assert audit['training_record_count'] == 363 and audit['train_root_count'] == 68
    assert audit['replay_count'] == 0 and audit['dev_test_training_roots'] == []
    assert audit['inputs_changed_during_audit'] == []
    assert audit['training_results']['formal']['backward_calls'] == 726
    assert audit['training_results']['formal']['optimizer_steps'] == 182
    assert audit['audited_files']
    for row in audit['audited_files']:
        assert sha(Path(row['path'])) == row['sha256'], row['path']
    assert read(HERE / 'TRAINING_SUMMARY.json')['status'] == 'TRAINED_NOT_YET_SOLVER_EVALUATED'
    assert read(HERE / 'LIVE_PROCESS.json')['running'] is False
    pool = REPO / 'offline-deliveries/h4_codex_train_pool_v125'
    assert sha(pool / 'COMPLETE_MANIFEST.json') == read(HERE / 'EXECUTION_FREEZE.json')['source_package_manifest_sha256']
    for row in read(pool / 'COMPLETE_MANIFEST.json')['files']:
        assert sha(pool / row['path']) == row['sha256'], row['path']
    output_files = manifest(RUN)
    save(RUN / 'COMPLETE_MANIFEST.json', {'schema_version': 'v126_actual_training_outputs_v1',
         'status': 'TRAINING_COMPLETE_NOT_EVALUATED', 'file_count': len(output_files), 'files': output_files})
    save(HERE / 'MODEL_OUTPUT_BINDING.json', {'path': str(RUN),
         'complete_manifest_path': str(RUN / 'COMPLETE_MANIFEST.json'),
         'complete_manifest_sha256': sha(RUN / 'COMPLETE_MANIFEST.json'),
         'final_adapter_path': str(RUN / 'adapter'),
         'adapter_state_sha256': sha(RUN / 'adapter/adapter_model.pt'),
         'adapter_config_sha256': sha(RUN / 'adapter/adapter_config.json')})
    save(HERE / 'FINAL_STATUS.json', {'status': 'TRAINED_AND_AUDITED_AWAITING_DEV_EVALUATION',
         'training_complete': True, 'independent_audit': 'PASS', 'solver_evaluation_complete': False,
         'train_records': 363, 'train_roots': 68, 'replay': 0, 'epochs': 2,
         'backward_calls': 726, 'optimizer_steps': 182,
         'next_step': 'Run the paired frozen v15 protocol on v124 dev48 against original M_clean; keep test sealed.'})
    own_files = manifest(HERE)
    save(HERE / 'COMPLETE_MANIFEST.json', {'schema_version': 'v126_completed_training_delivery_v1',
         'batch_id': HERE.name, 'status': 'TRAINED_NOT_SOLVER_EVALUATED', 'file_count': len(own_files), 'files': own_files})
    print(json.dumps({'status': 'SEALED', 'delivery_files': len(own_files), 'model_output_files': len(output_files),
          'complete_manifest_sha256': sha(HERE / 'COMPLETE_MANIFEST.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
