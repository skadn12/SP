"""Freeze completed production artifacts and publish a navigation pointer."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent


def read(p):
    return json.loads(Path(p).read_text())


def binding(p):
    p = Path(p).resolve()
    raw = p.read_bytes()
    return {'path': str(p), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def save(p, value):
    with Path(p).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main():
    assert not (HERE / 'FINAL_MANIFEST.json').exists(), 'ALREADY_FROZEN'
    pool = read(HERE / 'POOL_MANIFEST.json')
    collection = read(HERE / 'COLLECTION_AUDIT.json')
    completed = read(HERE / 'audit/COMPLETED_INTEGRITY.json')
    assert completed['status'] == 'PASS_EXECUTION_INTEGRITY'
    independent = read(HERE / 'audit/EXECUTION_CODE_REVIEW.json')
    assert independent['status'].startswith('PASS'), 'INDEPENDENT_CODE_REVIEW_REQUIRED'
    independent_execution = read(HERE / 'audit/INDEPENDENT_EXECUTION_AUDIT.json')
    assert independent_execution['status'].startswith('PASS'), 'INDEPENDENT_EXECUTION_REVIEW_REQUIRED'
    for audit in (independent, independent_execution):
        assert audit.get('bindings'), 'INDEPENDENT_AUDIT_BINDINGS_REQUIRED'
        for b in audit['bindings']:
            assert binding(b['path']) == b, 'INDEPENDENT_AUDIT_BINDING_DRIFT:' + b['path']
    summary = read(HERE / 'execution/ANALYSIS_SUMMARY.json')
    baseline = read(HERE / 'audit/HISTORICAL_BASELINE.json')
    for b in baseline['bindings']:
        assert binding(b['path']) == b, 'HISTORICAL_DRIFT:' + b['path']
    now = datetime.now(timezone.utc).isoformat()
    old_status_path = WEEK / 'CURRENT_STATUS.json'
    old_status = read(old_status_path)
    save(HERE / 'audit/CURRENT_STATUS_BEFORE_PRODUCTION.json', old_status)
    # The manifest binds completed batch outputs and historical source/code
    # references. Mutable week-level navigation files remain outside the seal.
    files = sorted(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                   and p.name != 'FINAL_MANIFEST.json')
    external_paths = {b['path'] for b in baseline['bindings']}
    preparation = read(HERE / 'execution/PREPARATION.json')
    external_paths.update(b['path'] for b in preparation['bindings'] if not Path(b['path']).is_relative_to(HERE))
    external_paths.update(b['path'] for b in read(HERE / 'audit/PRODUCTION_GATE_REGRESSION.json')['script_bindings'])
    manifest = {
        'schema_version': 'guarded_production_batch_final_v1',
        'created_at_utc': now,
        'status': 'PRODUCTION_BATCH_COMPLETE_CONTENT_CANDIDATES_FROZEN',
        'planned_slots': 70, 'selected_train_roots': 10,
        'submitted_candidates': collection['submitted_candidates'],
        'content_approved_unique_questions': pool['exported_unique_questions'],
        'question_types': pool['question_types'], 'mc_gold_histogram': pool['mc_gold_histogram'],
        'material_shortfalls': len(collection['material_shortfalls']),
        'initial_joint_pass_rate_planned': collection['initial_joint_pass_rate_planned'],
        'initial_joint_90_percent_scaleup_threshold_met': collection['initial_joint_90_percent_scaleup_threshold_met'],
        'solver_requests': summary['requests'], 'solver_valid': summary['valid'],
        'solver_conditions': summary['conditions'],
        'new_dialogue_turns': 0, 'new_training_runs': 0, 'dev_test_requests': 0,
        'solver_score_admission_filter': False, 'semantic_reason_release': False,
        'formal_weakness_or_feedback_causal_claim': False,
        'historical_baseline_bindings_unchanged': len(baseline['bindings']),
        'artifact_bindings': [binding(p) for p in files],
        'external_reference_bindings': [binding(p) for p in sorted(external_paths)],
    }
    save(HERE / 'FINAL_MANIFEST.json', manifest)
    pointer = {
        'schema_version': 'production_batch_latest_v1', 'created_at_utc': now,
        'status': manifest['status'], 'batch': HERE.name,
        'manifest': binding(HERE / 'FINAL_MANIFEST.json'),
        'report': binding(HERE / 'REPORT_ZH.md'),
        'pool': binding(HERE / 'TRAINING_POOL_CANDIDATES.jsonl'),
        'pool_manifest': binding(HERE / 'POOL_MANIFEST.json'),
        'examples': binding(HERE / 'NEW_QUESTIONS_AND_REASONS_ZH.md'),
        'selected_train_roots': 10, 'planned_slots': 70,
        'content_approved_unique_questions': pool['exported_unique_questions'],
        'question_types': pool['question_types'], 'solver_requests': summary['requests'],
        'initial_joint_90_percent_scaleup_threshold_met': collection['initial_joint_90_percent_scaleup_threshold_met'],
        'new_dialogue_turns': 0, 'new_training_runs': 0,
    }
    save(WEEK / 'PRODUCTION_BATCH_LATEST.json', pointer)
    next_action = ('Continue guarded question production on another prospectively fixed Train batch; '
                   'retain material shortages and independent content review. Do not infer formal weakness or causal feedback benefit.'
                   if collection['initial_joint_90_percent_scaleup_threshold_met'] else
                   'Keep content-approved questions; examine the frozen initial failures and material shortfalls before increasing batch size. '
                   'Preserve the 90 percent threshold and completed batch; no post-solver question rewriting.')
    updated = {**old_status, 'status': manifest['status'], 'updated_at_utc': now,
               'next_action': next_action, 'production_batch_latest': pointer,
               'status_history_snapshot_before_production': binding(HERE / 'audit/CURRENT_STATUS_BEFORE_PRODUCTION.json')}
    # CURRENT_STATUS is explicitly a mutable navigation file; the previous
    # version was copied into this batch and included in its manifest.
    with old_status_path.open('w', encoding='utf-8') as stream:
        json.dump(updated, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    for b in manifest['artifact_bindings'] + manifest['external_reference_bindings']:
        assert binding(b['path']) == b, 'FINAL_MANIFEST_BINDING_DRIFT:' + b['path']
    print(json.dumps({'status': manifest['status'], 'questions': pool['exported_unique_questions'],
                      'solver_requests': summary['requests'], 'bound_artifacts': len(files),
                      'scaleup_threshold_met': collection['initial_joint_90_percent_scaleup_threshold_met']}))


if __name__ == '__main__':
    main()
