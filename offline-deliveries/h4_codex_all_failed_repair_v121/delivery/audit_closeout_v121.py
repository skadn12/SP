"""Read-only closeout checks of the fixed cohort, lineage and saved stage gates."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sys

B = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(B / 'tools'))
import scene_controller_v120 as scene


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    errors, verified, roots, reserved = [], {}, [], []

    def need(ok, why):
        if not ok:
            errors.append(why)

    def verify(row):
        try:
            path = Path(row['path'])
            actual = binding(path)
            need(actual['sha256'] == row['sha256'], 'BINDING_CHANGED:' + str(path))
            verified[str(path.resolve())] = actual
        except (OSError, KeyError, TypeError) as exc:
            errors.append('INVALID_BINDING:' + str(row) + ':' + str(exc))

    for name in ('SOURCE_PROTOCOL_FREEZE.json', 'PREFIX_PROTOCOL_FREEZE.json', 'MASTER_FREEZE.json', 'EVALUATION_TOOLS_FREEZE.json', 'EVALUATION_TOOLS_FREEZE_V2.json', 'EVALUATION_TOOLS_FREEZE_V3.json'):
        document = read(B / name)
        rows = document.get('files', document.get('bindings', []))
        need(bool(rows), 'EMPTY_FREEZE:' + name)
        for item in rows:
            verify(item)
        if document.get('independent_audit'):
            verify(document['independent_audit'])

    manifest = read(B / 'MANIFEST.json')
    need(len(manifest['roots']) == 24, 'FIXED_COHORT_NOT_24')
    old_consumed = set(read(B / 'evidence/SOURCE_POOL_FREEZE.json')['consumed_source_ids'])
    for row in manifest['roots']:
        root = row['root_id']
        unit = Path(row['unit_path'])
        folder = unit / 'roots' / unit.name
        source = read(unit / 'source_repair/FINALIZE_RECEIPT.json')
        need(source['source_status'] == row['source_status'], root + ':SOURCE_STATUS_CHANGED')
        need(len(source['review_bindings']) == 2, root + ':TWO_SOURCE_REVIEWS_REQUIRED')
        for item in source['review_bindings']:
            verify(item)
        out = {'root_id': root, 'source_status': row['source_status']}
        if not row.get('source_ready', True):
            need(not (unit / 'selection.json').exists(), root + ':MATCHED_AFTER_SOURCE_HOLD')
            out['status'] = 'SOURCE_HOLD'
        elif not row['prefix_reuse_approved']:
            need(not (unit / 'selection.json').exists(), root + ':MATCHED_AFTER_PREFIX_HOLD')
            out['status'] = 'PREFIX_HOLD'
        else:
            state = scene.inspect_root(unit, scene.load_config(unit), root, record_reviews=False)
            out['scene_state'] = state
            reservations = read(unit / 'SOURCE_RESERVATIONS.json')
            new = [r['source_id'] for r in reservations['allocation'] if r['allocation_kind'] == 'NEW_PERMANENT_RESERVATION']
            reserved.extend(new)
            need(len(new) == reservations['new_consumption_count'], root + ':RESERVATION_COUNT')
            if state['root_stopped']:
                out['status'] = 'SCENE_HOLD'
            else:
                need(state['full_expansion_complete'], root + ':INCOMPLETE_SIX_SCENES')
                if (folder / 'QA_REVIEW_FAILURE.json').exists():
                    failure = read(folder / 'QA_REVIEW_FAILURE.json')
                    need(failure.get('terminal') is True and failure.get('no_second_semantic_judgment') is True,
                         root + ':INVALID_REVIEW_FAILURE_DISPOSITION')
                    need(not (folder / 'PRIVATE_GOLD.json').exists(), root + ':GOLD_AFTER_REVIEW_SCHEMA_HOLD')
                    out['status'] = 'QA_REVIEW_SCHEMA_HOLD'
                elif (folder / 'QA_AUTHOR_FAILURE.json').exists():
                    out['status'] = 'QA_FIRST_ATTEMPT_FAILED'
                else:
                    need((folder / 'FINAL_CROSSCHECK.json').is_file(), root + ':MISSING_FINAL_CROSSCHECK')
                    for action in ('COMPOSE', 'FINALIZE'):
                        receipt = read(unit / 'evidence' / ('GUARDED_' + action + '_RECEIPT.json'))
                        need(receipt['status'] == 'GUARDED_STAGE_COMPLETED' and receipt['gate']['pass']
                             and receipt['downstream_returncode'] == 0, root + ':GUARDED_' + action)
                        for item in receipt['gate']['verified_file_bindings']:
                            verify(item)
                    cross = read(folder / 'FINAL_CROSSCHECK.json')
                    passed = [q['candidate_id'] for q in cross['results'] if q['status'] == 'CONTENT_PASS']
                    need(len(set(passed)) == len(passed), root + ':DUPLICATE_CONTENT_PASS')
                    out.update(status='FINALIZED', content_pass=len(passed))
        roots.append(out)
    need(len(reserved) == len(set(reserved)), 'DUPLICATE_NEW_NATIVE_RESERVATION')
    need(not (set(reserved) & old_consumed), 'HISTORICAL_NATIVE_EVENT_REUSED')

    historical = []
    for row in read(B / 'evidence/HISTORICAL_PROTECTION_PRE_GENERATION.json')['checks']:
        verify(row['inventory_binding'])
        inventory = read(row['inventory_binding']['path'])
        for item in inventory['files']:
            verify(item)
        historical.append({'inventory': row['inventory_binding'], 'files_checked': len(inventory['files'])})

    pool = read(B / 'evaluations/paired_v1/EVALUATION_POOL.json')
    candidate = read(B / 'delivery/TRAINING_CANDIDATES_V1.json')
    need(candidate['training_or_pool_merge'] is False, 'UNAUTHORIZED_TRAINING_OR_MERGE')
    need(all(q['question_type'] != 'multi_select' or len(q['gold']) != 6 for q in candidate['questions']), 'GOLD6_IN_TRAINING_CANDIDATES')
    need(set(q['candidate_id'] for q in candidate['questions']) <= set(q['candidate_id'] for q in pool['questions']), 'CANDIDATES_OUTSIDE_FROZEN_POOL')
    for side in ('BASE', 'TRAINED'):
        complete = read(B / 'evaluations/paired_v1' / (side + '_COMPLETE.json'))
        verify(complete['results_binding'])
    report = {'schema_version': 'v121_post_run_integrity_audit_v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'actual_executor': '/root', 'pass': not errors, 'errors': errors, 'fixed_candidates': 24,
        'terminal_statuses': dict(Counter(r['status'] for r in roots)), 'roots': roots,
        'new_permanent_native_events': len(reserved), 'historical_checks': historical,
        'verified_file_count': len(verified), 'semantic_quality_certified_by_program': False,
        'execution_deviation_binding': binding(B / 'evidence/PARENT_COLLECTOR_FREEZE_DEVIATION_V1.json'),
        'review_preflight_execution_deviation_binding': binding(B / 'evidence/REVIEW_PREFLIGHT_FALLTHROUGH_B_V1.json'),
        'evaluation_prepare_compatibility_failure_binding': binding(B / 'evidence/EVALUATION_PREPARE_LABEL_ORDER_FAILURE_V1.json'),
        'evaluation_canonical_label_adapter_audit_binding': binding(B / 'evidence/EVALUATION_CANONICAL_LABEL_AUDIT_A_V1.json'),
        'absence_of_runtime_deviations_certified': False,
        'training_or_pool_merge': False}
    path = B / 'delivery/POST_RUN_INTEGRITY_AUDIT_V1.json'
    with path.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('roots', 'historical_checks')}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
