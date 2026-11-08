#!/usr/bin/env python3
"""Copy frozen evaluation evidence; preserve originals and verify recorded hashes."""
import hashlib
import json
import shutil
from pathlib import Path

REPO = Path('/root/autodl-tmp/gxy-selfplay')
OUT = Path(__file__).resolve().parents[1]
SPECS = [
    ('h4_codex_dev_eval_v127', 'evaluations/dev_v127', 'complete evaluation excluding Python caches'),
    ('h4_codex_dev_ceiling_review_v128', 'evaluations/ordering_ceiling_v128', 'complete diagnostic review excluding Python caches'),
    ('h4_codex_base_order_control_v130', 'evaluations/original_base_ordering_v130', 'complete evaluation excluding Python caches'),
    ('h4_codex_historical_frontier_eval_v131', 'evaluations/historical_round1_frontier_v131', 'historical round1 diagnostic questions; NOT second-round generated data'),
    ('h4_codex_training_v126', 'training_execution/v126', 'training execution metadata; smoke/adapter model weights and caches excluded'),
]


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    bindings = []
    seal_audits = []
    for name, dest, scope in SPECS:
        src = REPO / 'offline-deliveries' / name
        manifest = src / 'COMPLETE_MANIFEST.json'
        native = json.loads(manifest.read_text())
        checked = 0
        for entry in native['files']:
            p = src / entry['path']
            assert p.is_file(), str(p)
            assert sha(p) == entry['sha256'], str(p)
            if 'bytes' in entry:
                assert p.stat().st_size == entry['bytes'], str(p)
            checked += 1
        seal_audits.append({'source_directory': str(src), 'manifest_sha256': sha(manifest),
                            'recorded_files_checked': checked, 'status': 'PASS'})
        copied = []
        for p in sorted(src.rglob('*')):
            if not p.is_file():
                continue
            rel = p.relative_to(src)
            if '__pycache__' in rel.parts or (name.endswith('v126') and rel.parts[:2] == ('smoke', 'adapter')):
                continue
            assert p.suffix.lower() not in {'.pt', '.bin', '.safetensors', '.key'}, str(p)
            assert 'api_key' not in p.name.lower(), str(p)
            q = OUT / dest / rel
            q.parent.mkdir(parents=True, exist_ok=True)
            if q.exists():
                assert sha(q) == sha(p), str(q)
            else:
                shutil.copy2(p, q)
            assert sha(q) == sha(p), str(q)
            copied.append({'source': str(p), 'package_path': q.relative_to(OUT).as_posix(),
                           'bytes': q.stat().st_size, 'sha256': sha(q)})
        bindings.append({'source_directory': str(src), 'package_directory': dest,
                         'scope': scope, 'copied_files': copied})
    evidence = OUT / 'evidence'
    evidence.mkdir(exist_ok=True)
    (evidence / 'EVALUATION_SOURCE_BINDINGS.json').write_text(json.dumps({
        'schema_version': 'v132_evaluation_source_bindings_v1', 'status': 'PASS',
        'inference_performed': False, 'training_performed': False,
        'original_seal_audits': seal_audits, 'copies': bindings,
    }, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'copied_files': sum(len(x['copied_files']) for x in bindings),
                      'original_manifests_checked': len(seal_audits)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
