"""Make a final-adapter-only edition without editing the full delivery."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil

B = Path(__file__).resolve().parent
S = B.parent / 'h4_codex_round2_complete_bundle_v135'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read(p):
    return json.loads(p.read_text())

def save(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(x, f, ensure_ascii=False, indent=2)
        f.write('\n')

def main():
    source_manifest = S / 'COMPLETE_MANIFEST.json'
    manifest = read(source_manifest)
    assert manifest['status'] == 'SEALED_COMPLETE'
    rows = []
    kept_directories = {'data', 'snapshots', 'training_raw'}
    kept_top_files = {'DATA_STATISTICS.json', 'EVALUATION_INDEX.json',
                      'QUESTION_SOURCE_INDEX.json', 'ROOT_INDEX.csv',
                      'ROOT_INDEX.json', 'ROOT_INDEX.md', 'SOURCE_SNAPSHOTS.json'}
    for row in manifest['files'] + [{'path': 'COMPLETE_MANIFEST.json',
                                    'bytes': source_manifest.stat().st_size,
                                    'sha256': sha(source_manifest)}]:
        rel = Path(row['path'])
        src = S / rel
        direct = (rel.parts[0] in kept_directories or str(rel) in kept_top_files
                  or (rel.parts[:2] == ('models', 'round2_native_adapter')
                      and len(rel.parts) == 3))
        if direct:
            target = rel
        elif rel.parts[0] == 'models' and str(rel) != 'models/ARTIFACT_BINDINGS.json':
            continue
        else:
            target = Path('previous_delivery_metadata/v135') / rel
        assert src.stat().st_size == row['bytes'] and sha(src) == row['sha256'], str(src)
        dst = B / target
        dst.parent.mkdir(parents=True, exist_ok=True)
        assert not dst.exists(), str(dst)
        shutil.copy2(src, dst)
        assert sha(dst) == row['sha256'], str(dst)
        rows.append({'path': str(target), 'source_relative_path': str(rel),
                     'bytes': row['bytes'], 'sha256': row['sha256']})
    save(B / 'COPY_BINDINGS.json', {
        'schema_version': 'v136_source_copy_bindings_v1', 'status': 'PASS',
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_delivery': str(S), 'source_manifest_sha256': sha(source_manifest),
        'source_unchanged': True,
        'files': rows, 'file_count': len(rows),
        'omitted': ['models/round2_merged/', 'models/round2_native_adapter/checkpoint-*/'],
    })
    bindings = read(S / 'models/ARTIFACT_BINDINGS.json')
    models = [r for r in bindings['files']
              if Path(r['package_relative_path']).parts[:2] == ('models', 'round2_native_adapter')
              and len(Path(r['package_relative_path']).parts) == 3]
    save(B / 'models/ARTIFACT_BINDINGS.json', {
        'schema_version': 'v136_final_adapter_bindings_v1', 'status': 'PASS',
        'file_count': len(models), 'total_bytes': sum(r['size_bytes'] for r in models),
        'files': models,
        'note': 'Final step-182 native adapter and tokenizer only; no merged/Base/checkpoint weights.'})
    shutil.copy2(S / 'verify_bundle.py', B / 'verify_bundle.py')
    print(json.dumps({'status': 'COPIED', 'files': len(rows),
                      'bytes': sum(r['bytes'] for r in rows),
                      'final_adapter_files': len(models)}))

if __name__ == '__main__':
    main()
