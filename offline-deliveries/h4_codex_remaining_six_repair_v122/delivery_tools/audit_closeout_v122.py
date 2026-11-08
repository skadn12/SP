"""Read-only integrity checks over the fixed six-root repair and old inventories."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from state_v122 import B, bind, classify, read


def main():
    errors, verified, roots, reserved = [], {}, [], []
    def need(ok, why):
        if not ok: errors.append(why)
    def verify(row):
        try:
            actual = bind(row['path']); need(actual == {k: row[k] for k in ('path', 'sha256')}, 'BINDING_CHANGED:' + row['path'])
            verified[actual['path']] = actual
        except (OSError, KeyError, TypeError) as exc: errors.append('INVALID_BINDING:' + str(exc))
    for name in ('SOURCE_PROTOCOL_FREEZE.json', 'PREFIX_PROTOCOL_FREEZE.json', 'MASTER_FREEZE.json', 'EVALUATION_TOOLS_FREEZE.json'):
        document = read(B / name); bindings = document.get('files', document.get('bindings', []))
        need(bool(bindings), 'EMPTY_FREEZE:' + name)
        for binding in bindings: verify(binding)
        if document.get('independent_audit'): verify(document['independent_audit'])
    scope, manifest = read(B / 'REPAIR_SCOPE.json'), read(B / 'MANIFEST.json')
    need(scope['fixed_candidate_count'] == len(manifest['roots']) == 6, 'FIXED_COHORT_NOT_SIX')
    need([r['root_id'] for r in scope['roots']] == [r['root_id'] for r in manifest['roots']], 'FIXED_ROOT_ORDER')
    consumed = set(read(B / 'evidence/SOURCE_POOL_FREEZE.json')['consumed_source_ids'])
    for row in manifest['roots']:
        root, unit = row['root_id'], Path(row['unit_path']); folder = unit / 'roots' / unit.name
        source = read(unit / 'source_repair/FINALIZE_RECEIPT.json')
        need(source.get('effective_source_status', source['source_status']) == row.get('effective_source_status', row['source_status']), root + ':EFFECTIVE_SOURCE_STATUS_CHANGED')
        need(source.get('local_patch_status', source['source_status']) == row.get('local_patch_status', row['source_status']), root + ':LOCAL_PATCH_STATUS_CHANGED')
        need(len(source['review_bindings']) == 2, root + ':TWO_SOURCE_REVIEWS_REQUIRED')
        for item in source['review_bindings']: verify(item)
        terminal = classify(row, require_terminal=True)
        if terminal['status'] in ('SOURCE_HOLD', 'PREFIX_HOLD'):
            need(not (unit / 'selection.json').exists(), root + ':MATCHED_AFTER_HOLD')
        else:
            reservations = read(unit / 'SOURCE_RESERVATIONS.json')
            new = [r['source_id'] for r in reservations['allocation'] if r['allocation_kind'] == 'NEW_PERMANENT_RESERVATION']
            reserved.extend(new); need(len(new) == reservations['new_consumption_count'], root + ':RESERVATION_COUNT')
        if terminal['status'] == 'FINALIZED':
            for action in ('COMPOSE', 'FINALIZE'):
                receipt = read(unit / 'evidence' / ('GUARDED_' + action + '_RECEIPT.json'))
                need(receipt['status'] == 'GUARDED_STAGE_COMPLETED' and receipt['gate']['pass'] is True and receipt['downstream_returncode'] == 0, root + ':GUARDED_' + action)
                for item in receipt['gate']['verified_file_bindings']: verify(item)
        for item in folder.glob('**/HISTORICAL_*_IMPORT.json'):
            document = read(item)
            if document.get('policy_binding'): verify(document['policy_binding'])
            if document.get('original_draft_binding'): verify(document['original_draft_binding'])
            if document.get('imported_draft_binding'): verify(document['imported_draft_binding'])
        roots.append(terminal)
    need(len(reserved) == len(set(reserved)), 'DUPLICATE_NEW_NATIVE_RESERVATION')
    need(not set(reserved) & consumed, 'HISTORICAL_NATIVE_EVENT_REUSED')
    historical = []
    historical_gate = read(B / 'evidence/HISTORICAL_PROTECTION_PRE_GENERATION.json')
    need(historical_gate['pass'] is True, 'HISTORICAL_PRE_GENERATION_CHECK_FAILED')
    for row in historical_gate['checks']:
        verify(row['inventory_binding']); inventory = read(row['inventory_binding']['path'])
        for item in inventory['files']: verify(item)
        historical.append({'inventory': row['inventory_binding'], 'files_checked': len(inventory['files'])})
    eval_scope = B / 'evaluations/paired_v1'
    bindings = read(eval_scope / 'ADAPTER_BINDINGS.json')
    for item in bindings['files']: verify(item)
    pool, candidates = read(eval_scope / 'EVALUATION_POOL.json'), read(B / 'delivery/TRAINING_CANDIDATES_V1.json')
    need(candidates['training_or_pool_merge'] is False, 'UNAUTHORIZED_TRAINING_OR_MERGE')
    need(all(q['question_type'] != 'multi_select' or len(q['gold']) != 6 for q in candidates['questions']), 'GOLD6_IN_CANDIDATES')
    need({q['candidate_id'] for q in candidates['questions']} <= {q['candidate_id'] for q in pool['questions']}, 'CANDIDATES_OUTSIDE_FROZEN_POOL')
    for side in ('BASE', 'TRAINED'): verify(read(eval_scope / (side + '_COMPLETE.json'))['results_binding'])
    report = {'schema_version': 'v122_post_run_integrity_audit_v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'actual_executor': '/root', 'pass': not errors, 'errors': errors, 'fixed_candidates': 6,
              'terminal_statuses': dict(Counter(r['status'] for r in roots)), 'roots': roots,
              'new_permanent_native_events': len(reserved), 'historical_checks': historical,
              'verified_file_count': len(verified), 'semantic_quality_certified_by_program': False,
              'execution_issue_bindings': [bind(p) for p in sorted((B / 'evidence').glob('*.json')) if any(s in p.name for s in ('DEVIATION', 'FAILURE', 'FALLTHROUGH'))],
              'absence_of_runtime_deviations_certified': False, 'training_or_pool_merge': False}
    with (B / 'delivery/POST_RUN_INTEGRITY_AUDIT_V1.json').open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('roots', 'historical_checks')}, ensure_ascii=False))
    if errors: raise SystemExit(1)


if __name__ == '__main__': main()
