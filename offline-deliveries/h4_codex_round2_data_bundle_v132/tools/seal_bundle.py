#!/usr/bin/env python3
"""Seal a completed v132 package; do not edit frozen source deliveries."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write(name, obj):
    (ROOT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def main():
    assert not (ROOT / 'COMPLETE_MANIFEST.json').exists(), 'Package already sealed'
    for rel in ['evidence/INDEPENDENT_DATA_BUNDLE_AUDIT.json',
                'evidence/FEEDBACK_AND_DIFFICULTY_AUDIT.json',
                'evidence/SECOND_ROUND_TRANSITIONS.json',
                'evidence/EVALUATION_SOURCE_BINDINGS.json']:
        obj = json.loads((ROOT / rel).read_text())
        assert obj['status'] in {'PASS', 'COMPLETE'}, (rel, obj.get('status'))
    data = json.loads((ROOT / 'DATA_STATISTICS.json').read_text())
    write('FINAL_STATUS.json', {
        'schema_version': 'v132_final_status_v1', 'status': 'COMPLETE',
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'operation': 'Copy and organize existing second-round data and evaluation evidence',
        'original_files_moved_or_modified': False,
        'new_model_inference_calls': 0, 'new_training_runs': 0,
        'total_dialogue_roots': data['total_dialogue_roots'],
        'total_final_questions': data['total_final_questions'],
        'splits': {s: {k: data['splits'][s][k] for k in ['fixed_roots', 'roots_with_final_questions', 'final_questions', 'question_type_counts']}
                   for s in ['train', 'dev', 'test']},
        'data_entry': 'ROOT_INDEX.md', 'interpretation_entry': 'README_ZH.md',
        'scope_limitations': [
            'DEV/TEST fixed roots are not all fully expanded or admitted; retain unsuccessful root states.',
            'Historical first-round diagnostic questions are evaluation evidence, not second-round generated questions.',
            'DEV improvement does not establish a causal feedback contribution.',
            'Cross-round difficulty scores use differing formats/protocols/Gold quality; descriptive only.',
            'No complete second-round pre/post evaluation of all 363 actual training questions was located.'
        ],
    })
    files = []
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file():
            continue
        assert not p.is_symlink(), str(p)
        with p.open('rb') as f:
            h = hashlib.sha256()
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        files.append({'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size, 'sha256': h.hexdigest()})
    write('COMPLETE_MANIFEST.json', {
        'schema_version': 'v132_complete_manifest_v1', 'status': 'COMPLETE',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'All files below package root except this self-excluded manifest; archive and its checksum are adjacent.',
        'file_count': len(files), 'payload_bytes': sum(x['bytes'] for x in files), 'files': files,
    })
    print(json.dumps({'status': 'SEALED', 'files': len(files), 'payload_bytes': sum(x['bytes'] for x in files)}))


if __name__ == '__main__':
    main()
