"""Extend v3 with the separately retained 112812 original RAW relocation.

An exact original path and SHA may resolve only to the bound archive with the
same bytes. The original failure remains a procedure deviation. The replacement
judgment may differ only by qualifying the listed baseline turn references;
no semantic judgment, source, question, or Gold is repaired by this adapter.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_content_v3 as previous


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def validate_record(record):
    assert record['root_id'] == 'cc-train-episode-112812'
    for row in record['bindings']:
        assert sha(row['path']) == row['sha256'], 'RELOCATION_V4_EVIDENCE_CHANGED'
    relocation = record['relocation']
    original, archive = relocation['original_binding'], relocation['archived_binding']
    assert original['sha256'] == archive['sha256'] == sha(archive['path'])
    failure = json.loads(Path(record['failure_binding']['path']).read_text())
    assert failure['raw_judgment_binding'] == original
    assert failure['error'] == 'CONSTRAINT_EVIDENCE_REFS'
    baseline = json.loads(Path(record['baseline_binding']['path']).read_text())
    baseline_refs = {t['turn_ref'] for t in baseline['turns']}
    prefix = record['root_id'] + ':'
    old = json.loads(Path(archive['path']).read_text())
    current = json.loads(Path(original['path']).read_text())
    normalized = copy.deepcopy(old)
    edits = []
    results = normalized['baseline_constraint_review']['constraint_results']
    for i, result in enumerate(results):
        for j, ref in enumerate(result['evidence_turn_refs']):
            qualified = prefix + ref
            if qualified in baseline_refs:
                result['evidence_turn_refs'][j] = qualified
                edits.append({'constraint_result_index': i, 'evidence_ref_index': j,
                              'original': ref, 'qualified': qualified})
    assert edits == record['reference_qualifications'] and edits
    assert normalized == current, 'ONLY_BASELINE_REFERENCE_QUALIFICATION_ALLOWED'
    return original, archive


def engine():
    master = HERE.parent
    proof = master / 'evidence/READ_DIRECTORY_RELOCATION_V2.json'
    binding_path = master / 'evidence/AUDIT_RELOCATION_V4_BINDINGS.json'
    for row in json.loads(binding_path.read_text())['files']:
        assert sha(row['path']) == row['sha256'], 'V4_ADAPTER_BINDING_CHANGED'
    record = json.loads(proof.read_text())
    original, archive = validate_record(record)
    ns, previous_proof, previous_used = previous.engine()
    previous_verify = ns['verify']
    used = []

    def verify(row):
        if row == original:
            assert sha(archive['path']) == row['sha256'], 'RELOCATED_ORIGINAL_BYTES_CHANGED'
            used.append(dict(row))
        else:
            previous_verify(row)

    ns['verify'] = verify
    return ns, [previous_proof, proof, binding_path], previous_used, used


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wave', choices=['validation', 'production', 'all'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ns, proofs, previous_used, used = engine()
    ns['main'](args.wave, args.output.resolve())
    companion = args.output.with_name(args.output.stem + '_PROCEDURE_ADAPTER.json')
    with companion.open('x') as stream:
        json.dump({'status': 'PASS', 'wave': args.wave,
                   'relocation_evidence_bindings': [bind(p) for p in proofs],
                   'audit_adapter_binding': bind(__file__),
                   'audit_binding': bind(args.output),
                   'resolved_exact_original_references': previous_used + used,
                   'procedure_deviations_retained': ['cc-train-episode-96462', 'cc-train-episode-112812'],
                   'source_question_gold_or_original_review_changed': False,
                   'external_semantic_certification': False}, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
