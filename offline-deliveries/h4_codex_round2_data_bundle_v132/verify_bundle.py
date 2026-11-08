#!/usr/bin/env python3
"""Read-only verification of this package and, optionally, its tar.gz archive."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath


def digest(stream):
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(chunk)
    return h.hexdigest()


def verify(root, archive=None):
    manifest_path = root / 'COMPLETE_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    expected = {x['path']: x for x in manifest['files']}
    assert len(expected) == len(manifest['files']), 'duplicate manifest paths'
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    assert actual == set(expected) | {'COMPLETE_MANIFEST.json'}, 'missing or extra package files'
    for rel, entry in expected.items():
        assert '..' not in PurePosixPath(rel).parts and not rel.startswith('/'), rel
        p = root / rel
        assert not p.is_symlink(), rel
        assert p.stat().st_size == entry['bytes'], rel
        with p.open('rb') as f:
            assert digest(f) == entry['sha256'], rel
    result = {'status': 'PASS', 'verified_files': len(expected),
              'payload_bytes': sum(x['bytes'] for x in expected.values()),
              'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    if archive:
        with manifest_path.open('rb') as f:
            expected['COMPLETE_MANIFEST.json'] = {'bytes': manifest_path.stat().st_size, 'sha256': digest(f)}
        seen = set()
        with tarfile.open(archive, 'r:gz') as tf:
            for member in tf:
                path = PurePosixPath(member.name)
                assert path.parts[0] == root.name and '..' not in path.parts, member.name
                if member.isdir():
                    continue
                assert member.isfile(), member.name
                rel = str(PurePosixPath(*path.parts[1:]))
                assert rel in expected and rel not in seen, member.name
                entry = expected[rel]
                assert member.size == entry['bytes'], member.name
                f = tf.extractfile(member)
                assert digest(f) == entry['sha256'], member.name
                seen.add(rel)
        assert seen == set(expected), 'missing archive members'
        with archive.open('rb') as f:
            result.update(archive=str(archive), archive_sha256=digest(f), archive_verified_files=len(seen))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(Path(__file__).resolve().parent, args.archive), ensure_ascii=False))
