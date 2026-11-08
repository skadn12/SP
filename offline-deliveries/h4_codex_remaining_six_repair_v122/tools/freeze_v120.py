"""Freeze v121 static runtime and inputs after independent engineering approval.

This cannot start generation or certify source/QA semantics. Prefix-specific
reads preceded this master freeze under their own immutable prefix protocol.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

B = Path(__file__).resolve().parents[1]
ESSENTIAL = ['scene_controller_v120.py', 'question_controller_v120.py', 'author_v120.py',
             'review_v4.py', 'review_provenance_gate_v2.py', 'scene_review_provenance_v1.py',
             'public_anchor_v1.py', 'run_guarded_questions_v2.py', 'prepare_v120.py',
             'prefix_review_v2.py', 'batch_prepare_v121.py', 'source_repair_v121.py', 'freeze_v120.py',
             'historical_fixed_import_v122.py']


def need(ok, why):
    if not ok: raise ValueError(why)
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p): return {'path': str(Path(p).resolve()), 'sha256': sha(p)}
def check(r): need(sha(r['path']) == r['sha256'], 'BOUND_FILE_CHANGED:' + r['path'])
def save(p, d):
    with Path(p).open('x') as f: json.dump(d, f, ensure_ascii=False, indent=2); f.write('\n')


def freeze(audit_path, receipt_path):
    need(not (B / 'MASTER_FREEZE.json').exists(), 'MASTER_ALREADY_FROZEN')
    audit, receipt = read(audit_path), read(receipt_path)
    need(audit.get('engineering_pass') is True and audit.get('all_known_blockers_resolved') is True,
         'INDEPENDENT_ENGINEERING_APPROVAL_REQUIRED')
    need(audit.get('production_generation_started') is False, 'PRODUCTION_ALREADY_STARTED')
    tool_rows = audit['frozen_candidate_tool_bindings']
    for r in tool_rows: check(r)
    approved = {r['path'] for r in tool_rows}
    need(all(str(B / 'tools' / n) in approved for n in ESSENTIAL), 'APPROVED_RUNTIME_BINDINGS_INCOMPLETE')
    need(receipt.get('pass') is True, 'FINAL_NATIVE_PREFLIGHT_REQUIRED')
    for r in receipt['files']: check(r)
    manifest = read(B / 'MANIFEST.json')
    scope = read(B / 'REPAIR_SCOPE.json')
    need(manifest['fixed_candidates'] == scope['fixed_candidate_count'] == 6, 'FIXED_REPAIR_DENOMINATOR')
    need([r['root_id'] for r in manifest['roots']] == [r['root_id'] for r in scope['roots']], 'FIXED_REPAIR_ROOTS')
    need(manifest['source_ready'] + manifest['source_hold'] == 6, 'SOURCE_DENOMINATOR')
    need(manifest['prefix_pass'] + manifest['prefix_hold'] == manifest['source_ready'], 'PREFIX_DENOMINATOR')
    need(not list(B.glob('units/*/selection.json')) and not list(B.glob('units/*/SOURCE_RESERVATIONS.json')),
         'MATCHING_MUST_NOT_PRECEDE_MASTER_FREEZE')
    need(not list(B.glob('units/*/evidence/MATCHING/*')) and not list(B.glob('units/*/evidence/AUTHOR_*')),
         'FORMAL_AUTHOR_READ_MUST_NOT_PRECEDE_MASTER_FREEZE')
    for name in ['SOURCE_PROTOCOL_FREEZE.json', 'PREFIX_PROTOCOL_FREEZE.json']:
        for r in read(B / name)['bindings']: check(r)
    historical = read(B / 'evidence/HISTORICAL_PROTECTION_PRE_GENERATION.json')
    need(historical['pass'] is True, 'HISTORICAL_PROTECTION_FAILED')
    for item in historical['checks']:
        check(item['inventory_binding'])
        for r in read(item['inventory_binding']['path'])['files']: check(r)
    for row in manifest['roots']:
        check(row['source_finalize_binding'])
        final = read(row['source_finalize_binding']['path'])
        check(final['source_output_binding'])
        source_output = read(final['source_output_binding']['path'])
        check(source_output['baseline_binding'])
        need(source_output['review_bindings'] == final['review_bindings'] and len(final['review_bindings']) == 2,
             'SOURCE_FINAL_REVIEW_BINDINGS')
        for binding in final['review_bindings']: check(binding)
        need((final['status'] == 'SOURCE_READY') == row['source_ready'], 'SOURCE_VERDICT_CHANGED')
        if not row['source_ready']:
            need(not row['prefix_reuse_approved'], 'SOURCE_HOLD_CANNOT_GENERATE')
            continue
        need(source_output['baseline_binding'] == bind(Path(row['unit_path']) / 'inputs/BASELINE.json'),
             'SOURCE_OUTPUT_BASELINE_PATH_OR_HASH')
        auditp = Path(row['prefix_review_binding']['path']); check(row['prefix_review_binding'])
        prefix = read(auditp)
        need(prefix['content_reuse_approved'] == row['prefix_reuse_approved'], 'PREFIX_VERDICT_CHANGED')
        if not row['prefix_reuse_approved']: continue
        unit = Path(row['unit_path'])
        imported = read(unit / 'PREFIX_IMPORT.json')
        retained = imported['retained_scene_refs']
        for draft in unit.glob('roots/*/scenes/*/EXPANSION_DRAFT.json'):
            need(read(draft)['scene_ref'] in retained, 'FORMAL_GENERATION_STARTED_TOO_EARLY')
    for p in (B / 'tools').glob('*.py'): compile(p.read_text(), str(p), 'exec')
    static = [p for p in B.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    external = []
    pool = read(B / 'evidence/SOURCE_POOL_FREEZE.json')
    external += [pool['source_pool_binding'], pool['legacy_binding'], *pool['consumption_files']]
    for item in historical['checks']: external.append(item['inventory_binding'])
    for r in external: check(r)
    paths = list(dict.fromkeys([p.resolve() for p in static] + [Path(r['path']).resolve() for r in external]))
    save(B / 'MASTER_FREEZE.json', {
        'schema_version': 'v122_master_freeze_v1',
        'status': 'FROZEN_BEFORE_V122_FORMAL_AUTHOR_MATCHING_AND_GENERATION',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'actual_executor': '/root',
        'files': [bind(p) for p in sorted(paths)],
        'independent_engineering_audit_binding': bind(audit_path),
        'native_preflight_receipt_binding': bind(receipt_path),
        'prefix_read_order': 'Prefix-specific roles/input/protocol freeze preceded independent prefix reads; master QA/matching runtime freeze follows engineering preflight.',
        'new_author_matching_started': False, 'formal_generation_started': False,
        'generation_roots': [r['root_id'] for r in manifest['roots'] if r['prefix_reuse_approved']],
        'remaining_scene_budget': sum(len(r['remaining_scene_refs']) for r in manifest['roots'] if r['prefix_reuse_approved']),
        'fixed_carried_scene_budget': sum(len(r.get('fixed_scene_import_refs', [])) for r in manifest['roots'] if r['prefix_reuse_approved']),
        'new_scene_budget': sum(len(r['remaining_scene_refs']) - len(r.get('fixed_scene_import_refs', [])) for r in manifest['roots'] if r['prefix_reuse_approved']),
        'new_b_event_budget': sum(len(r['new_b_scene_refs']) for r in manifest['roots'] if r['prefix_reuse_approved']),
        'input_freeze_after_matching_required': True, 'training_or_pool_merge': False,
        'identity_limit': 'Actual shared-filesystem executor/read assertions are auditable declarations, not cryptographic authentication.'})
    print(json.dumps({'status': 'MASTER_FROZEN', 'files': len(paths), 'generation_started': False}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args(); freeze(a.audit.resolve(), a.receipt.resolve())
