"""Retain v2 integrity checks and verify one recorded original-raw relocation.

Only the exact (original absolute path, original SHA256) pair in the immutable
relocation evidence may resolve to its archived bytes. No missing/hash failure
is ignored. The directory move remains a reported procedure deviation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_content_v2 as previous
from review_scene_resume_v1 import norm


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def engine():
    master = HERE.parent
    proof = master / 'evidence/READ_DIRECTORY_RELOCATION_V1.json'
    frozen = json.loads((master / 'evidence/AUDIT_RELOCATION_V3_BINDINGS.json').read_text())
    for row in frozen['files']:
        assert sha(row['path']) == row['sha256'], 'V3_ADAPTER_BINDING_CHANGED:' + row['path']
    record = json.loads(proof.read_text())
    assert record['root_id'] == 'cc-train-episode-96462'
    assert record['semantic_normalization_equal'] is True
    for row in record['bindings']:
        assert sha(row['path']) == row['sha256'], 'RELOCATION_EVIDENCE_CHANGED'
    assert len(record['relocations']) == 1
    relocation = record['relocations'][0]
    original, archive = relocation['original_binding'], relocation['archived_binding']
    assert original['sha256'] == archive['sha256'] == sha(archive['path'])
    rejected = Path(archive['path']).parent
    current = Path(original['path']).parent
    failure = json.loads((rejected / 'SAVE_FAILURE.json').read_text())
    assert failure['raw_judgment_binding'] == original
    assert failure['error'] == 'CONSTRAINT_BLOCKING_FINDINGS_REQUIRED'
    assert norm(json.loads(Path(archive['path']).read_text())) == norm(json.loads(Path(original['path']).read_text()))
    ns = previous.engine()
    used = []

    def verify(row):
        if row == original:
            assert sha(archive['path']) == row['sha256'], 'RELOCATED_ORIGINAL_BYTES_CHANGED'
            used.append(dict(row))
        else:
            assert sha(row['path']) == row['sha256'], row['path']

    ns['verify'] = verify
    return ns, proof, used


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wave', choices=['validation', 'production', 'all'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    ns, proof, used = engine()
    ns['main'](a.wave, a.output.resolve())
    companion = a.output.with_name(a.output.stem + '_PROCEDURE_ADAPTER.json')
    with companion.open('x') as stream:
        json.dump({'status': 'PASS', 'wave': a.wave,
                   'relocation_evidence_binding': {'path': str(proof), 'sha256': sha(proof)},
                   'audit_binding': {'path': str(a.output.resolve()), 'sha256': sha(a.output)},
                   'resolved_exact_original_references': used,
                   'procedure_deviation_reported': bool(used),
                   'source_question_gold_or_original_review_changed': False,
                   'external_semantic_certification': False}, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
