"""Full-source, question-scoped workflow adapter; no model or network calls.

Uses the existing, difficulty-label-only V2.2.1 acceptance implementation.
This module never creates material approval or independently certifies QA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
BASE = HERE.parent
IMPLEMENTATION = BASE / 'base_qa_v2_2_1_acceptance_20261102/implementation'
sys.path[:0] = [str(IMPLEMENTATION), str(PROJECT / 'scripts/current_method')]
import base_qa_pipeline_v2_2_1 as pipeline

if Path(pipeline.__file__).resolve().parent != IMPLEMENTATION:
    raise ImportError('Use this adapter in a separate process from other pipeline versions')

ADMISSION = BASE / 'base_qa_v2_2_1_real_20261102/admission'
CONFIRMATION = BASE / 'base_qa_v2_2_1_real_20261102/CALLER_FORMAL_USE_CONFIRMATION.json'
WORKFLOW = 'full_source_question_evidence_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def binding(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def checked(record):
    actual = binding(record['path'])
    if any(actual[key] != record[key] for key in ('sha256', 'bytes') if key in record):
        raise ValueError('FILE_BINDING_DRIFT:' + record['path'])
    # Hash-bind every upstream artifact; parse JSON only where the caller needs
    # its object. Python implementation bindings are intentionally opaque here.
    return read(record['path']) if str(record['path']).lower().endswith('.json') else None


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def prepare_root(root_id, output):
    """Freeze every source turn; derive metadata from existing bound records."""
    authorization = read(CONFIRMATION)
    grant = next(row for row in authorization['roots'] if row['root_id'] == root_id)
    folder = ADMISSION / root_id
    source_path = folder / 'FORMAL_SOURCE_VIEW.json'
    scope_path = folder / 'SOURCE_SCOPE.json'
    receipt_path = folder / 'MATERIAL_ADMISSION.json'
    canonical_path = BASE / 'day3_production_v1/sources' / (root_id + '.json')
    expected = [(source_path, 'formal_source_view_sha256'),
                (scope_path, 'source_scope_sha256'),
                (receipt_path, 'admission_receipt_sha256'),
                (canonical_path, 'canonical_source_sha256')]
    for path, key in expected:
        if binding(path)['sha256'] != grant[key]:
            raise ValueError('CALLER_BINDING_DRIFT:' + key)
    source, scope, receipt = read(source_path), read(scope_path), read(receipt_path)
    canonical = read(canonical_path)
    # Metadata overlays are allowed; no dialogue text, identity, order or turn loss.
    identity = lambda s: [(t['turn_ref'], t['speaker'], t['text']) for t in s['turns']]
    if identity(source) != identity(canonical):
        raise ValueError('CANONICAL_DIALOGUE_CHANGED')
    if pipeline.v21.source_manifest(source, binding(source_path), receipt,
                                   binding(receipt_path), 'source-workflow-adapter') != scope:
        raise ValueError('UPSTREAM_SCOPE_REPLAY_MISMATCH')
    checked(receipt['independent_review_binding'])
    checked(receipt['authorization_binding'])
    by_ref = {r['turn_id']: r for r in receipt['turns']}
    scope_rows = {r['turn_id']: r for r in scope['turns']}
    ledger = []
    for turn in source['turns']:
        ref = turn['turn_ref']
        row, reviewed = by_ref[ref], scope_rows[ref]
        record = checked(row['review_record_binding'])
        context_only = record.get('scope_role') == 'CONTEXT_ONLY'
        legacy_status = row['review_status']
        risk = ('NOT_ASSESSED_IN_THIS_REVIEW_SCOPE' if context_only else
                'NO_RECORDED_BLOCKER' if legacy_status == 'PASS' else 'RECORDED_WARNING_OR_HOLD')
        ledger.append(dict(
            turn_id=ref, scene_id=turn.get('scene_ref') or turn.get('session_ref'),
            source_id=turn['source_id'], source_version=turn['source_version'],
            provenance_type=turn['provenance_type'], source_file_hash=binding(source_path)['sha256'],
            legacy_review_status=legacy_status, legacy_scope_role=record.get('scope_role'),
            content_risk=dict(status=risk, decision_basis=record.get('decision_basis'),
                              findings=record.get('findings', []),
                              limitations=receipt.get('review_findings', [])),
            authorization_status='ALLOWED_BY_EXISTING_RECORD' if reviewed['formal_qa_allowed'] else 'RESTRICTED_BY_EXISTING_RECORD',
            formal_evidence_allowed=reviewed['formal_qa_allowed'],
            review_record_binding=row['review_record_binding']))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, path in [('FULL_SOURCE.json', source_path), ('CANONICAL_SOURCE.json', canonical_path),
                       ('SOURCE_SCOPE.json', scope_path)]:
        with (output / name).open('xb') as handle:
            handle.write(path.read_bytes())
    write_new(output / 'TURN_LEDGER.json', dict(workflow=WORKFLOW, root_id=root_id, turns=ledger,
        interpretation='Legacy CONTEXT_ONLY/HOLD is a scope restriction, not a newly inferred content defect. No new approval.',
        source_review_mode=receipt.get('reviewer_mode'),
        reviewer_external_certification=receipt.get('reviewer_external_certification', False)))
    manifest = dict(workflow=WORKFLOW, root_id=root_id,
        full_source_turn_count=len(source['turns']),
        all_turns_formally_allowed_informational_only=scope['formal_qa_eligible'],
        files=[binding(output / name) for name in ('FULL_SOURCE.json', 'CANONICAL_SOURCE.json', 'SOURCE_SCOPE.json', 'TURN_LEDGER.json')],
        upstream=[binding(source_path), binding(scope_path), binding(receipt_path), binding(CONFIRMATION),
                  receipt['authorization_binding'], receipt['independent_review_binding']],
        pipeline_binding=binding(pipeline.__file__), new_material_approval=False)
    write_new(output / 'FULL_SOURCE_FREEZE.json', manifest)
    return manifest


def load_full_source(directory):
    directory = Path(directory)
    manifest = read(directory / 'FULL_SOURCE_FREEZE.json')
    if manifest.get('workflow') != WORKFLOW:
        raise ValueError('WORKFLOW_MISMATCH')
    for item in manifest['files'] + manifest['upstream'] + [manifest['pipeline_binding']]:
        checked(item)
    source, scope = read(directory / 'FULL_SOURCE.json'), read(directory / 'SOURCE_SCOPE.json')
    if len(source['turns']) != manifest['full_source_turn_count']:
        raise ValueError('SOURCE_TURN_COUNT_DRIFT')
    return source, scope


def evidence_closure(*documents, additional_dependencies=()):
    """Collect submitted evidence, bridges and distractor dependencies.

    Source-scan membership alone is visibility, not a supporting premise.
    Detecting an undeclared semantic dependency still requires full-source review.
    """
    refs = set(additional_dependencies)
    def walk(value):
        if isinstance(value, dict):
            if 'turn_ref' in value and 'quote' in value:
                refs.add(value['turn_ref'])
            refs.update(value.get('error_path_turn_refs', []))
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    for document in documents:
        walk(document)
        for node in document.get('dependency_graph', {}).get('nodes', []):
            if node.get('type') == 'fact':
                refs.update(node.get('turn_refs', []))
    if any(not isinstance(ref, str) or not ref.strip() for ref in refs):
        raise ValueError('EVIDENCE_REFS_INVALID')
    return sorted(refs)


def question_evidence_audit(source, scope, candidate, target, *, additional_dependencies=()):
    """Evidence permission and integrity, not Gold/necessity certification."""
    refs = evidence_closure(target, candidate, additional_dependencies=additional_dependencies)
    gate = pipeline._formal_scope_gate(source, scope, refs, candidate.get('author_id'), require_receipt=True)
    errors = list(gate['errors'])
    by_ref = {t['turn_ref']: t for t in source['turns']}
    if candidate.get('root_id') != source['root_id']:
        errors.append('CANDIDATE_ROOT_MISMATCH')
    if candidate.get('source_scope_sha256') != pipeline.fp(scope) or target.get('source_scope_sha256') != pipeline.fp(scope):
        errors.append('QUESTION_SCOPE_HASH_MISMATCH')
    def check_quotes(value):
        if isinstance(value, dict):
            if 'turn_ref' in value and 'quote' in value:
                turn, quote = by_ref.get(value['turn_ref']), value['quote']
                if (not turn or not isinstance(quote, str) or not quote.strip()
                        or quote not in turn['text'] or value.get('speaker') != turn['speaker']):
                    errors.append('EVIDENCE_QUOTE_OR_SPEAKER_MISMATCH:' + str(value['turn_ref']))
            for child in value.values():
                check_quotes(child)
        elif isinstance(value, list):
            for child in value:
                check_quotes(child)
    for document in (candidate, target):
        check_quotes(document)
    scan = candidate.get('query_path_audit', {}).get('source_scan', [])
    scanned = {ref for row in scan for ref in row.get('turn_refs', [])}
    if scanned != set(by_ref):
        errors.append('FULL_SOURCE_COUNTERPATH_SCAN_INCOMPLETE')
    return dict(workflow=WORKFLOW, candidate_id=candidate.get('candidate_id'),
        evidence_eligible=not errors, errors=sorted(set(errors)),
        evidence_turn_refs=refs, ineligible_turn_refs=gate['ineligible_turn_refs'],
        full_source_turn_count=len(source['turns']), full_source_scan_coverage=len(scanned & set(by_ref)),
        all_turns_formally_allowed_informational_only=scope.get('formal_qa_eligible'),
        content_quality='NOT_REASSESSED', certificate_accuracy='NOT_REASSESSED',
        difficulty_is_validity_veto=False, formal_qa_decision='NOT_GRANTED_BY_THIS_AUDIT')


def prepare_question_review(run_dir, candidate, target_freeze_path, discovery, frozen_source_dir,
                            *, additional_dependencies=(), **reviewer_context):
    """Actual authoring facade: complete source in the existing frozen packet."""
    source, scope = load_full_source(frozen_source_dir)
    target = read(target_freeze_path)['target']
    audit = question_evidence_audit(source, scope, candidate, target,
                                    additional_dependencies=additional_dependencies)
    if not audit['evidence_eligible']:
        raise ValueError('QUESTION_EVIDENCE_BLOCKED:' + ';'.join(audit['errors']))
    result = pipeline.prepare_frozen_review(run_dir, candidate, target_freeze_path,
                                            discovery, source, scope, **reviewer_context)
    packet = read(Path(run_dir) / 'PUBLIC_PACKET.json')['packet']
    if [t['turn_ref'] for t in packet['source']['turns']] != [t['turn_ref'] for t in source['turns']]:
        raise ValueError('REVIEWER_SOURCE_TRUNCATED')
    write_new(Path(run_dir) / 'QUESTION_EVIDENCE_AUDIT.json', audit)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-sources', type=Path, required=True)
    args = parser.parse_args()
    for short in ('103866', '104814'):
        root = 'cc-train-episode-' + short
        result = prepare_root(root, args.prepare_sources / root)
        print(root, result['full_source_turn_count'], 'turns frozen')


if __name__ == '__main__':
    main()
