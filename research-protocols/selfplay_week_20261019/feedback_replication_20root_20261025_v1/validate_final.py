#!/usr/bin/env python3
"""Read-only final-artifact validator for the authorized 16+4-root replication.

Run --describe for the exact report/receipt schema. COMPLETED requires all 256
new semantic slots and 512 new returned/scored responses, six completed phases,
hash-bound report and Markdown, the original sealed manifest checked file by
file, and a distinct closeout audit with verdict PASS bound to the report hash.
Generation shortfalls may be valid PARTIAL reports; they never become Solver
errors or a COMPLETED experiment. This checker never starts a model or worker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
PHASES = [
    'frozen_design_and_inputs', 'shared_r03_source_generation_and_review',
    'isolated_fg_plans_questions_and_review', 'content_freeze_and_solver_preflight',
    'fixed_b1_solver_and_scoring', 'new16_and_combined20_analysis_and_delivery',
]
SLOTS = ['MC2', 'MC3', 'MC4', 'MC5', 'MC6', 'ORDER1', 'ORDER2', 'ORDER3']


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, code):
    if not condition:
        raise ValueError(code)


def bind(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def verify(binding, base):
    path = Path(binding['path'])
    if not path.is_absolute():
        path = Path(base) / path
    actual = bind(path)
    require(actual['sha256'] == binding['sha256'] and actual['bytes'] == binding['bytes'], 'BINDING_CHANGED:' + str(path))
    return path.resolve()


def numeric_root(root):
    return root.removeprefix('cc-train-episode-')


def design(base):
    base = Path(base).resolve()
    plan, config = read(base / 'PLAN.json'), read(base / 'CONFIG.json')
    new = [numeric_root(root) for root in plan['new_train_roots']]
    original = [numeric_root(root) for root in plan['original_train_roots']]
    require(len(new) == len(set(new)) == 16 and len(original) == len(set(original)) == 4 and not set(new) & set(original), 'FIXED_16_PLUS_4_ROOT_DESIGN_REQUIRED')
    require([numeric_root(root) for root in config['roots']] == new and [numeric_root(root) for root in config['original_roots']] == original, 'PLAN_CONFIG_ROOT_DRIFT')
    require(plan['phases'] == PHASES and plan['training'] is False and config['training'] is False and plan['rerun_original_four'] is False, 'PHASE_OR_NO_TRAINING_DESIGN_DRIFT')
    require(plan['max_seconds'] == 43200 and plan['max_worker_rounds'] == 8
            and plan['stop_new_generation_before_deadline_minutes'] == 45 and plan['reserve_closeout_minutes'] == 20, 'OVERNIGHT_BOUNDS_DRIFT')
    require(config['arms'] == ['F', 'G'] and config['slots_per_root_per_arm'] == 8 and config['display_variants'] == 2, 'FIXED_FG_SLOT_DISPLAY_DRIFT')
    counts = {'new_roots': 16, 'original_roots': 4, 'total_roots': 20,
              'planned_new_semantic': 256, 'planned_new_responses': 512,
              'planned_combined_semantic': 320, 'planned_combined_responses': 640}
    require(config['planned_semantic_questions'] == 256 and config['planned_solver_requests'] == 512
            and config['combined_planned_semantic_questions'] == 320 and config['combined_planned_solver_requests'] == 640, 'CONFIG_PLAN_COUNTS_DRIFT')
    for key in ('planned_new_semantic', 'planned_new_responses', 'planned_combined_semantic', 'planned_combined_responses'):
        require(plan[key] == counts[key], 'PLAN_COUNTS_DRIFT:' + key)
    return plan, config, new, original, counts


def verify_original(base, receipt_binding):
    """Recheck the frozen original manifest and every original file now."""
    base = Path(base).resolve()
    _, config, _, original, _ = design(base)
    receipt_path = verify(receipt_binding, base)
    require(receipt_path.is_relative_to(base), 'ORIGINAL_INTEGRITY_RECEIPT_MUST_BE_NEW_ARTIFACT')
    receipt = read(receipt_path)
    require(receipt['status'] == 'PASS' and receipt['unchanged'] is True, 'ORIGINAL_INTEGRITY_NOT_PASS')
    require(sorted(numeric_root(root) for root in receipt['original_root_ids']) == sorted(original), 'ORIGINAL_RECEIPT_ROOT_DRIFT')
    selection = read(base / 'SELECTION_FREEZE.json')
    frozen = selection['original_four_reused_read_only']['bindings']['FINAL_MANIFEST.json']
    require(receipt['original_manifest_binding'] == frozen, 'ORIGINAL_MANIFEST_RECEIPT_DRIFT')
    manifest_path = verify(frozen, base)
    original_base = Path(config['original_experiment']).resolve()
    require(manifest_path == original_base / 'FINAL_MANIFEST.json', 'ORIGINAL_MANIFEST_PATH_DRIFT')
    original_manifest = read(manifest_path)
    artifacts = original_manifest['files']
    require(artifacts and len(artifacts) == original_manifest['file_count'] == receipt['verified_artifacts'], 'ORIGINAL_MANIFEST_FILE_COUNT_DRIFT')
    paths = []
    for artifact in artifacts:
        path = verify(artifact, original_base)
        require(path.is_relative_to(original_base), 'ORIGINAL_MANIFEST_PATH_ESCAPE')
        paths.append(path)
    require(len(paths) == len(set(paths)), 'DUPLICATE_ORIGINAL_MANIFEST_PATH')
    return {'verified_artifacts': len(paths), 'receipt_path': str(receipt_path), 'original_manifest': frozen}


def verify_input_lock(base):
    base = Path(base).resolve()
    lock = read(base / 'INPUT_LOCK.json')
    bindings = lock.get('bindings', lock.get('artifacts', lock.get('files')))
    require(isinstance(bindings, list) and bindings, 'EMPTY_OR_UNSUPPORTED_INPUT_LOCK')
    paths = [verify(binding, base) for binding in bindings]
    require(len(paths) == len(set(paths)), 'DUPLICATE_INPUT_LOCK_PATH')
    for name in ('PLAN.json', 'CONFIG.json', 'SELECTION_FREEZE.json', 'solver_runner.py', 'WORKER_INSTRUCTIONS_ZH.md'):
        require((base / name).resolve() in paths, 'UNLOCKED_REQUIRED_INPUT:' + name)
    return len(paths)


def validate(base=BASE, *, require_completed=False):
    base = Path(base).resolve()
    result = {'valid': False, 'base_directory': str(base), 'model_or_worker_started': False}
    try:
        plan, config, new, original, expected = design(base)
        report_path, markdown_path = base / 'FINAL_REPORT.json', base / 'FINAL_REPORT_ZH.md'
        report = read(report_path)
        require(markdown_path.is_file() and markdown_path.stat().st_size > 0, 'MISSING_FINAL_REPORT_MARKDOWN')
        status = report['overall_status']
        require(status in ('COMPLETED', 'PARTIAL', 'BLOCKED'), 'INVALID_REPORT_STATUS')
        require(type(report['objective_completed']) is bool and report['objective_completed'] == (status == 'COMPLETED'), 'OBJECTIVE_STATUS_CONFLICT')
        require(report['training'] is False and report['rerun_original_four'] is False, 'REPORT_NO_TRAINING_OR_REUSE_DRIFT')
        for key, expected_value in expected.items():
            require(type(report['counts'][key]) is int and report['counts'][key] == expected_value, 'REPORT_PLAN_COUNT_DRIFT:' + key)
        for key, maximum in (('new_semantic_admitted', 256), ('new_responses_returned', 512), ('new_responses_scored', 512)):
            require(type(report['counts'][key]) is int and 0 <= report['counts'][key] <= maximum, 'INVALID_ACTUAL_REPORT_COUNT:' + key)
        require(set(report['phases']) == set(PHASES), 'REPORT_PHASE_SET_DRIFT')
        require(all(isinstance(report['phases'][phase].get('status'), str) for phase in PHASES), 'MISSING_PHASE_STATUS')
        manifest = read(base / 'FINAL_ARTIFACT_MANIFEST.json')
        artifacts = manifest['artifacts']
        require(isinstance(artifacts, list) and artifacts, 'EMPTY_FINAL_ARTIFACT_MANIFEST')
        bound = [verify(artifact, base) for artifact in artifacts]
        require(len(bound) == len(set(bound)), 'DUPLICATE_FINAL_ARTIFACT_PATH')
        bound = set(bound)
        for path in (report_path, markdown_path, base / 'PLAN.json', base / 'CONFIG.json', base / 'SELECTION_FREEZE.json', base / 'INPUT_LOCK.json'):
            require(path.resolve() in bound, 'UNBOUND_FINAL_INPUT:' + path.name)
        result['input_files_verified'] = verify_input_lock(base)
        old = verify_original(base, report['original_integrity_receipt'])
        require(Path(old['receipt_path']) in bound, 'UNBOUND_ORIGINAL_INTEGRITY_RECEIPT')
        # The report names the audit; the final manifest hashes both. Putting
        # audit.sha256 in the report would cycle with audit.report_sha256.
        audit_reference = report['independent_closeout_audit']
        require(isinstance(audit_reference, dict) and set(audit_reference) == {'path'}, 'CLOSEOUT_AUDIT_REFERENCE_MUST_BE_PATH_ONLY')
        audit_path = Path(audit_reference['path'])
        if not audit_path.is_absolute():
            audit_path = base / audit_path
        audit_path = audit_path.resolve()
        require(audit_path.is_relative_to(base) and audit_path in bound, 'UNBOUND_OR_EXTERNAL_CLOSEOUT_AUDIT')
        audit = read(audit_path)
        require(audit['report_sha256'] == bind(report_path)['sha256'], 'CLOSEOUT_AUDIT_REPORT_HASH_DRIFT')
        if status == 'COMPLETED':
            require(all(report['phases'][phase]['status'] == 'COMPLETED' for phase in PHASES), 'INCOMPLETE_PHASE_IN_COMPLETED_REPORT')
            require(audit['verdict'] == 'PASS' and audit.get('independent_of_report_author') is True,
                    'COMPLETED_REQUIRES_INDEPENDENT_PASS_AUDIT')
            require(report['counts']['new_semantic_admitted'] == 256 and report['counts']['new_responses_returned'] == report['counts']['new_responses_scored'] == 512, 'GENERATION_OR_RESPONSE_SHORTFALL_REQUIRES_PARTIAL')
            required = ['CANDIDATES_PRIVATE.json', 'CONTENT_REVIEW_AI.json', 'CONTENT_FREEZE.json',
                        'SOURCE_SELECTION_FINAL.json', 'execution/PREPARATION.json',
                        'execution/RAW_RESPONSE_FREEZE.json', 'execution/SCORES.json', 'execution/ANALYSIS_SUMMARY.json',
                        'REPLICATION_COMPARISON.json', 'REPLICATION_TABLES_ZH.md',
                        'REPLICATION_ROOT_DIFFERENCES.csv']
            for name in required:
                require((base / name).resolve() in bound, 'UNBOUND_COMPLETED_ARTIFACT:' + name)
            # The legacy analysis is retained for audit only. The dedicated
            # replication outputs supply the new16, original4, combined20
            # comparisons and must actually exist in the frozen deliverable.
            for name in ('REPLICATION_COMPARISON.json', 'REPLICATION_TABLES_ZH.md', 'REPLICATION_ROOT_DIFFERENCES.csv'):
                require((base / name).stat().st_size > 0, 'EMPTY_COMPLETED_REPLICATION_ANALYSIS:' + name)
            require(isinstance(read(base / 'REPLICATION_COMPARISON.json'), dict), 'INVALID_REPLICATION_COMPARISON_OBJECT')
            candidates = read(base / 'CANDIDATES_PRIVATE.json')
            require(len(candidates) == len({row['candidate_id'] for row in candidates}) == 256, 'COMPLETED_CANDIDATE_COUNT_DRIFT')
            expected_slots = {('cc-train-episode-' + root, slot, arm) for root in new for slot in SLOTS for arm in ('F', 'G')}
            require({(row['root_id'], row['slot_id'], row['arm']) for row in candidates} == expected_slots, 'COMPLETED_CANDIDATE_SLOT_DRIFT')
            reviews = read(base / 'CONTENT_REVIEW_AI.json')['reviews']
            require(len(reviews) == 256 and {row['candidate_id'] for row in reviews} == {row['candidate_id'] for row in candidates}
                    and all(row['decision'] == 'PASS' for row in reviews), 'COMPLETED_CONTENT_REVIEW_SHORTFALL')
            selected = read(base / 'SOURCE_SELECTION_FINAL.json')['roots']
            require(len(selected) == 16 and {numeric_root(row['root_id']) for row in selected} == set(new), 'COMPLETED_SOURCE_SHORTFALL')
            raw = read(base / 'execution/RAW_RESPONSE_FREEZE.json')
            require(raw['requests'] == raw['returned'] == len(raw['request_ids']) == len(set(raw['request_ids'])) == 512
                    and raw['failure'] is None and raw['submitted_ids'] == raw['request_ids'], 'COMPLETED_RAW_RESPONSE_SHORTFALL')
            require(len(raw['bindings']) == 512, 'COMPLETED_RAW_BINDING_COUNT_DRIFT')
            captured = []
            for binding in raw['bindings']:
                path = verify(binding, base)
                require(path in bound and path.parent == (base / 'execution/raw').resolve(), 'UNBOUND_OR_MISPLACED_RAW_RESPONSE')
                response = read(path)
                require(response['status'] == 'RETURNED', 'NONRETURNED_RESPONSE_IN_COMPLETED_REPORT')
                captured.append(response['request_id'])
            require(set(captured) == set(raw['request_ids']) and len(set(captured)) == 512, 'RAW_RESPONSE_ID_DRIFT')
            scores = read(base / 'execution/SCORES.json')['records']
            require(len(scores) == len({row['request_id'] for row in scores}) == 512
                    and {row['request_id'] for row in scores} == set(raw['request_ids']), 'COMPLETED_SCORE_COUNT_DRIFT')
            require({row['strict']['candidate_id'] for row in scores} == {row['candidate_id'] for row in candidates}, 'COMPLETED_SCORE_CANDIDATE_DRIFT')
            summary = read(base / 'execution/ANALYSIS_SUMMARY.json')['denominators']
            require(summary['planned_semantic_questions'] == summary['content_admitted_semantic_questions'] == 256
                    and summary['planned_solver_requests'] == summary['frozen_requests'] == summary['returned_requests'] == 512
                    and summary['held_or_unavailable_semantic_slots'] == 0, 'COMPLETED_ANALYSIS_DENOMINATOR_DRIFT')
        if require_completed:
            require(status == 'COMPLETED', 'REPORT_IS_NOT_COMPLETED')
        result.update(valid=True, overall_status=status, objective_completed=report['objective_completed'],
                      artifacts_verified=len(bound), original_artifacts_verified=old['verified_artifacts'], report=report)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    return result


def describe():
    return {
        'files': ['FINAL_REPORT.json', 'FINAL_REPORT_ZH.md', 'FINAL_ARTIFACT_MANIFEST.json'],
        'FINAL_REPORT.json': {
            'overall_status': 'COMPLETED | PARTIAL | BLOCKED', 'objective_completed': 'true iff COMPLETED',
            'training': False, 'rerun_original_four': False,
            'phases': {phase: {'status': 'COMPLETED | PARTIAL | BLOCKED | NOT_STARTED'} for phase in PHASES},
            'counts': {'new_roots': 16, 'original_roots': 4, 'total_roots': 20,
                       'planned_new_semantic': 256, 'planned_new_responses': 512,
                       'planned_combined_semantic': 320, 'planned_combined_responses': 640,
                       'new_semantic_admitted': 'integer 0..256', 'new_responses_returned': 'integer 0..512',
                       'new_responses_scored': 'integer 0..512'},
            'original_integrity_receipt': {'path': 'new receipt path', 'sha256': 'hash', 'bytes': 'integer'},
            'independent_closeout_audit': {'path': 'new audit path; path only to avoid report/audit hash cycle'},
        },
        'original_integrity_receipt_content': {'status': 'PASS', 'unchanged': True,
            'original_root_ids': 'four frozen root IDs, with or without cc-train-episode- prefix',
            'original_manifest_binding': 'exact SELECTION_FREEZE.original_four_reused_read_only.bindings[FINAL_MANIFEST.json]',
            'verified_artifacts': 'actual original manifest.files count after every hash check'},
        'independent_closeout_audit_content': {'verdict': 'PASS required for COMPLETED',
            'report_sha256': 'sha256 of exact FINAL_REPORT.json bytes',
            'independent_of_report_author': 'true required for COMPLETED'},
        'FINAL_ARTIFACT_MANIFEST.json': {'artifacts': '[{path,sha256,bytes}, ...]; hash report, Markdown, frozen inputs, receipts and actual outputs; exclude the manifest itself'},
        'completed_replication_analysis': ['REPLICATION_COMPARISON.json', 'REPLICATION_TABLES_ZH.md', 'REPLICATION_ROOT_DIFFERENCES.csv'],
        'INPUT_LOCK.json': {'bindings': '[{path,sha256,bytes}, ...]; required PLAN, CONFIG, SELECTION_FREEZE, solver_runner, WORKER_INSTRUCTIONS; no mutation'},
        'completed_rule': 'All six phases, 256 PASS candidates covering every planned slot, 16 final sources, 512 actual returned/scored responses, bound dedicated REPLICATION_COMPARISON/TABLES/ROOT_DIFFERENCES outputs, unchanged original manifest files, independent audit PASS. Any missing slot => PARTIAL.',
        'audit_hash_cycle': 'Report names audit path only. Freeze report, independently audit its hash, then create manifest hashing both report and audit. Never include audit hash in report.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--require-completed', action='store_true')
    parser.add_argument('--describe', action='store_true')
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(describe(), ensure_ascii=False, indent=2))
        return 0
    result = validate(args.base, require_completed=args.require_completed)
    print(json.dumps({key: value for key, value in result.items() if key != 'report'}, ensure_ascii=False, indent=2))
    return 0 if result['valid'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
