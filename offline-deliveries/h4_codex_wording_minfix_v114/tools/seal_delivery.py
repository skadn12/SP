"""Seal the completed diagnostic, including post-call reviews and reports."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import stat

BATCH = Path(__file__).resolve().parent.parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(rows):
    for row in rows:
        assert sha(row['path']) == row['sha256'], row['path']


def main():
    target = BATCH / 'COMPLETE_MANIFEST.json'
    assert not target.exists()
    for name in ['INDEPENDENT_SOLVER_AUDIT.json', 'FINAL_INTEGRITY_CHECK.json',
                 'reviews/ACTUAL_PAIRING_REVIEW.json', 'reviews/SOLVER_COMPLETION_REVIEW.json',
                 'reviews/DELIVERY_REVIEW.json', 'reviews/NIRALA_EVIDENCE_ADDENDUM.json']:
        assert read(BATCH / name)['status'] == 'PASS', name
    assert read(BATCH / 'FINAL_STATUS.json')['status'] == 'COMPLETE_POSTHOC_WORDING_DIAGNOSTIC'
    for name in ['OLD_BATCH_PROTECTION.json', 'EXECUTION_FROZEN.json', 'evaluation/PAIRED_PREFLIGHT.json']:
        verify(read(BATCH / name)['files'])
    for name in ['reviews/ACTUAL_PAIRING_REVIEW.json', 'reviews/SOLVER_COMPLETION_REVIEW.json',
                 'reviews/DELIVERY_REVIEW.json', 'reviews/NIRALA_EVIDENCE_ADDENDUM.json',
                 'INDEPENDENT_SOLVER_AUDIT.json']:
        verify(read(BATCH / name)['input_bindings'])
    files = sorted(p for p in BATCH.rglob('*') if p.is_file())
    assert all('__pycache__' not in p.parts for p in files)
    rows = [{'path': str(p.resolve()), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files]
    with target.open('x') as f:
        json.dump({'status': 'SEALED_POSTHOC_WORDING_DIAGNOSTIC',
                   'created_at_utc': datetime.now(timezone.utc).isoformat(),
                   'files': rows, 'covered_files_excluding_this_manifest': len(rows),
                   'self_hash_excluded': True, 'fresh_calls': 72,
                   'training_started': False, 'training_pool_merged': False}, f,
                  ensure_ascii=False, indent=2)
        f.write('\n')
    verify(rows)
    assert {p.resolve() for p in BATCH.rglob('*') if p.is_file()} == {Path(r['path']) for r in rows} | {target.resolve()}
    for p in [*files, target]:
        p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    print(json.dumps({'status': 'SEALED', 'total_files': len(files) + 1,
                      'manifest_sha256': sha(target)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
