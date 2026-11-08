"""Seal the final-adapter edition after a completed independent audit."""
from pathlib import Path
import datetime
import hashlib
import json

B = Path(__file__).resolve().parent

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read(p):
    return json.loads(p.read_text())

def save(p, x):
    with p.open('x') as f:
        json.dump(x, f, ensure_ascii=False, indent=2)
        f.write('\n')

def main():
    audit_path = B / 'evidence/INDEPENDENT_LIGHTWEIGHT_AUDIT.json'
    audit_sha = sha(audit_path)
    audit = read(audit_path)
    assert audit['status'] == 'PASS'
    for row in audit['audited_files']:
        assert sha(B / row['path']) == row['sha256'], row['path']
    for row in read(B / 'COPY_BINDINGS.json')['files']:
        p = B / row['path']
        assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], row['path']
    weights = [p for p in B.rglob('*') if p.is_file() and p.suffix in {'.pt', '.bin', '.safetensors', '.pth', '.ckpt', '.gguf'}]
    assert [str(p.relative_to(B)) for p in weights] == ['models/round2_native_adapter/adapter_model.pt']
    assert sha(weights[0]) == '0e89757df099234cc6a4f81dd1bfbbf869dd4a4425b6a8098f8c00a65a170ecc'
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save(B / 'FINAL_STATUS.json', {
        'status': 'COMPLETE', 'created_at_utc': now,
        'edition': 'FINAL_LORA_ONLY_WITH_ALL_DATA_AND_REPORTS',
        'total_questions': 547, 'actual_training_records': 363,
        'dev_questions': 83, 'test_questions': 101,
        'latest_evaluation_only_questions': 49, 'fixed_roots': 99,
        'base_weights_included': False, 'merged_weights_included': False,
        'intermediate_checkpoint_weights_included': False,
        'final_lora_included': True, 'final_lora_bytes': weights[0].stat().st_size,
        'new_training_runs': 0, 'new_inference_calls': 0,
        'independent_audit_sha256': audit_sha,
        'source_delivery_unchanged': True,
    })
    paths = sorted(p for p in B.rglob('*') if p.is_file())
    assert not any(p.is_symlink() for p in paths)
    rows = [{'path': str(p.relative_to(B)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in paths]
    assert sha(audit_path) == audit_sha, 'Audit changed during sealing'
    save(B / 'COMPLETE_MANIFEST.json', {
        'schema_version': 'v136_complete_manifest_v1', 'status': 'SEALED_COMPLETE',
        'created_at_utc': now, 'scope': 'All package files except this self-excluded manifest.',
        'file_count': len(rows), 'payload_bytes': sum(r['bytes'] for r in rows),
        'files': rows,
    })
    print(json.dumps({'status': 'SEALED_COMPLETE', 'files': len(rows),
                      'payload_bytes': sum(r['bytes'] for r in rows),
                      'manifest_sha256': sha(B / 'COMPLETE_MANIFEST.json')}))

if __name__ == '__main__':
    main()
