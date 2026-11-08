"""Seal the combined data/report/model delivery after independent audit."""
from pathlib import Path
import datetime
import hashlib
import json
import struct

B = Path(__file__).resolve().parent

def read(p):
    return json.loads(p.read_text())

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def save(p, value):
    with p.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')

def main():
    assert not (B / 'COMPLETE_MANIFEST.json').exists()
    audit_path = B / 'evidence/INDEPENDENT_COMBINED_DATA_AUDIT.json'
    audit_sha_at_start = sha(audit_path)
    audit = read(audit_path)
    assert audit['status'] == 'PASS'
    for row in audit['audited_files']:
        assert sha(B / row['path']) == row['sha256'], row['path']
    for version in ['v132', 'v133', 'v134']:
        root = B / 'snapshots' / version
        manifest = read(root / 'COMPLETE_MANIFEST.json')
        for row in manifest['files']:
            p = root / row['path']
            assert p.is_file() and p.stat().st_size == row['bytes']
            assert sha(p) == row['sha256'], str(p)
    for folder in ['models', 'training_raw']:
        bindings = read(B / folder / 'ARTIFACT_BINDINGS.json')
        assert bindings['status'] == 'PASS'
        for row in bindings['files']:
            p = B / row['package_relative_path']
            assert p.is_file() and p.stat().st_size == row['size_bytes']
            assert sha(p) == row['sha256'], str(p)
    merged = B / 'models/round2_merged'
    index = read(merged / 'model.safetensors.index.json')['weight_map']
    tensors = {}
    for shard in sorted(set(index.values())):
        p = merged / shard
        with p.open('rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            assert 0 < n < 100 * 1024 * 1024
            header = json.loads(f.read(n))
        for key, value in header.items():
            if key == '__metadata__':
                continue
            assert key not in tensors
            assert index[key] == shard
            start, end = value['data_offsets']
            assert 0 <= start <= end <= p.stat().st_size - 8 - n
            tensors[key] = shard
    assert tensors == index
    stats = read(B / 'DATA_STATISTICS.json')
    assert stats['total_questions'] == 547 and stats['actual_training_records'] == 363
    assert stats['latest_added_evaluation_questions'] == 49
    assert {k: v['questions'] for k, v in stats['splits'].items()} == {'train': 363, 'dev': 83, 'test': 101}
    for name in ['README_ZH.md', 'MODEL_README_ZH.md', 'verify_bundle.py']:
        assert (B / name).is_file()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save(B / 'FINAL_STATUS.json', {
        'schema_version': 'v135_final_status_v1', 'status': 'COMPLETE',
        'created_at_utc': now, 'operation': 'Additive combined delivery of data, reports, weights, and raw training records',
        'total_questions': 547, 'train': 363, 'dev': 83, 'test': 101,
        'fixed_roots': 99, 'roots_with_questions': 95,
        'latest_added_evaluation_questions': 49,
        'latest_evaluation_live_calls': 2394,
        'new_training_runs': 0, 'new_inference_calls_during_packaging': 0,
        'whole_547_evaluated_after_training': False,
        'latest_49_used_in_training': False,
        'model_weights_included': ['second_round_merged', 'second_round_native_adapter', 'adapter_checkpoints_50_100_150'],
        'base_and_clean_weights_included': False,
        'formal_training_raw_logs_included': True,
        'historical_conflicts_resolved': False, 'test_has_historical_exposure': True,
        'old_packages_and_models_modified': False,
        'independent_data_audit': 'evidence/INDEPENDENT_COMBINED_DATA_AUDIT.json',
        'merged_shard_index_tensor_count_verified_without_loading': len(tensors),
    })
    files = sorted(p for p in B.rglob('*') if p.is_file())
    assert not any(p.is_symlink() for p in files)
    rows = [{'path': str(p.relative_to(B)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files]
    assert sha(audit_path) == audit_sha_at_start, 'Audit changed during sealing'
    assert next(r['sha256'] for r in rows if r['path'] == str(audit_path.relative_to(B))) == audit_sha_at_start
    save(B / 'COMPLETE_MANIFEST.json', {
        'schema_version': 'v135_complete_manifest_v1', 'status': 'SEALED_COMPLETE',
        'created_at_utc': now, 'scope': 'Every delivery file except this self-excluded manifest; includes all sealed snapshot files.',
        'file_count': len(rows), 'payload_bytes': sum(r['bytes'] for r in rows),
        'total_questions': 547, 'actual_training_records': 363,
        'files': rows,
    })
    print(json.dumps({'status': 'SEALED_COMPLETE', 'files': len(rows),
                      'payload_bytes': sum(r['bytes'] for r in rows),
                      'manifest_sha256': sha(B / 'COMPLETE_MANIFEST.json')}))

if __name__ == '__main__':
    main()
