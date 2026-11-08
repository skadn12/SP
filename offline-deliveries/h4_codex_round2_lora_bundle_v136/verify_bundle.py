"""Verify delivery bytes or a plain tar archive without model inference."""
from pathlib import Path
import argparse
import hashlib
import json
import tarfile

B = Path(__file__).resolve().parent

def digest(f):
    h = hashlib.sha256()
    for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
        h.update(chunk)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    manifest_path = B / 'COMPLETE_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['status'] == 'SEALED_COMPLETE', manifest['status']
    expected = {r['path']: r for r in manifest['files']}
    assert len(expected) == manifest['file_count']
    for name in expected:
        p = Path(name)
        assert not p.is_absolute() and '..' not in p.parts, name
    if args.archive:
        with manifest_path.open('rb') as f:
            expected['COMPLETE_MANIFEST.json'] = {
                'bytes': manifest_path.stat().st_size, 'sha256': digest(f)}
        seen = set()
        with tarfile.open(args.archive, 'r:') as archive:
            for member in archive:
                if member.isdir():
                    continue
                assert member.isfile(), member.name
                name = str(Path(member.name).relative_to(B.name))
                assert name in expected and name not in seen, name
                row = expected[name]
                assert member.size == row['bytes'], name
                with archive.extractfile(member) as f:
                    assert digest(f) == row['sha256'], name
                seen.add(name)
        assert seen == set(expected), {'missing': sorted(set(expected) - seen)}
    else:
        actual = {str(p.relative_to(B)) for p in B.rglob('*') if p.is_file()
                  and p != manifest_path}
        assert actual == set(expected), {'missing': sorted(set(expected) - actual),
                                          'unbound': sorted(actual - set(expected))}
        for name, row in expected.items():
            p = B / name
            assert p.is_file() and p.stat().st_size == row['bytes'], name
            with p.open('rb') as f:
                assert digest(f) == row['sha256'], name
    print(json.dumps({'status': 'PASS', 'verified_files': len(expected),
                      'mode': 'archive' if args.archive else 'directory'}))

if __name__ == '__main__':
    main()
