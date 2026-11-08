#!/usr/bin/env python3
"""Bind independent semantic clearance and original-order finite selection."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
CHECKS = ('train_new_replay_basic_family', 'train_diagnostic_feedback_related_family',
          'historical_dev_test_v140_family', 'cross_root_cross_split_semantic_near_duplicate',
          'within_locked_eval_fact_overlap_and_revision')


def bind(path):
    data = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def main():
    audit_path = HERE / 'audit/INDEPENDENT_FAMILY_REVIEW.json'
    audit = json.loads(audit_path.read_text())
    reviewed = {r['plan_id']: r for r in audit['items']}
    selected_ids = set(audit['global_selection']['selected_plan_ids'])
    validation = json.loads((HERE / 'VALIDATION.json').read_text())
    assert validation['status'] == 'PASS_MECHANICAL_ONLY' and not validation['failures']
    lock_path = WEEK / 'day3_production_v1/evaluation/EVAL_SOURCE_FAMILY_LOCK.json'
    lock = json.loads(lock_path.read_text())
    slots, plans, bindings = {}, {}, []
    for group in ('dev_first', 'dev_rest', 'test'):
        path = HERE / 'screening' / (group + '.json')
        bindings.append(bind(path))
        for slot in json.loads(path.read_text())['slots']:
            slots[slot['slot_id']] = slot
            for plan in slot['plans']:
                plans[plan['plan_id']] = plan
    assert len(slots) == 60 and len(plans) == 107
    selected, dispositions = [], []
    for slot in lock['itemized_slot_family_ledger']:
        sid = slot['slot_id']
        if sid not in slots:
            continue
        row = slots[sid]
        eligible = []
        for plan in row['plans']:
            pid = plan['plan_id']
            if pid not in selected_ids:
                continue
            review = reviewed[pid]
            assert review['verdict'] == 'PASS'
            assert all(plan['family_checks'][k]['status'] == 'PASS' for k in CHECKS[:4])
            assert all(review['first_four_checks'][k]['status'] == 'PASS' for k in CHECKS[:4])
            eligible.append(plan)
        # Verify independent recommendation also obeys the finite eligible hash
        # rule for alternatives that do not conflict with earlier chosen facts.
        if eligible:
            assert len(eligible) == 1
            plan = eligible[0]
            all_local_pass = [p for p in row['plans'] if p['plan_id'] in reviewed
                              and reviewed[p['plan_id']]['verdict'] == 'PASS'
                              and all(p['family_checks'][k]['status'] == 'PASS' for k in CHECKS[:4])]
            assert plan['candidate_selection_key'] == min(p['candidate_selection_key'] for p in all_local_pass)
            copy = json.loads(json.dumps(plan))
            copy['family_checks'][CHECKS[4]] = {
                'status': 'PASS', 'compared_record_ids': sorted(plans),
                'exclusion_index_sha256': '42ca84587026069afbe084802565693894934b701fb112963cd9d59bf0d72e0c',
                'matched_or_related_record_ids': review.get('internal_eval_overlap_plan_ids', []),
                'evidence': {'independent_global_review': bind(audit_path)},
                'rationale': 'Reviewed complete finite plan inventory. Related later plans are retained and excluded; no necessary fact overlap with an earlier selected family in original DEV then TEST order.',
                'reviewer': 'root with independent score-free global family review',
            }
            selected.append({'slot_id': sid, 'root_id': slot['root_id'], 'split': slot['split'],
                             'question_type': slot['question_type'], 'design_target': slot['design_target'],
                             'source_sha256': slot['source_sha256'], 'selected_plan': copy})
            status = 'PENDING_CONTENT_GENERATION'
        elif any('EXCLUDED' in p.get('plan_disposition', '') for p in row['plans']) and all(
                'EXCLUDED' in p.get('plan_disposition', '') for p in row['plans']):
            status = 'FAMILY_EXCLUDED'
        else:
            status = 'CONTENT_HOLD'
        dispositions.append({'slot_id': sid, 'root_id': slot['root_id'], 'split': slot['split'],
                             'question_type': slot['question_type'], 'status': status,
                             'screening_disposition': row.get('slot_disposition'),
                             'inventoried_plan_ids': [p['plan_id'] for p in row['plans']],
                             'rationale': row['rationale']})
    assert {r['selected_plan']['plan_id'] for r in selected} == selected_ids
    queue = json.loads((WEEK / 'day3_eval_semantic_v1/COMMON_EVAL_WORK_QUEUE.json').read_text())
    for row in queue['retained_material_dispositions']:
        dispositions.append({**row, 'status': 'CONTENT_HOLD' if row['status'] == 'MATERIAL_AMBIGUOUS_HOLD'
                             else row['status'], 'original_material_status': row['status']})
    order = {s['slot_id']: i for i, s in enumerate(lock['itemized_slot_family_ledger'])}
    dispositions.sort(key=lambda r: order[r['slot_id']])
    assert len(dispositions) == len({r['slot_id'] for r in dispositions}) == 186
    report = {'schema_version': 'finite_common_eval_family_selection_v1',
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'FAMILY_SELECTION_COMPLETE_CONTENT_PENDING',
              'source_lock': bind(lock_path), 'screening_inputs': bindings,
              'mechanical_validation': bind(HERE / 'VALIDATION.json'), 'independent_review': bind(audit_path),
              'finite_inventory_slots': 60, 'finite_inventory_plans': 107,
              'selected': selected, 'all186_dispositions': dispositions,
              'counts': dict(Counter(r['status'] for r in dispositions)),
              'historical_role': 'REGRESSION_ONLY',
              'scope': 'Bounded finite screening and original-order selection only. Unresolved related historical mappings stay HOLD; no claim that sources contain no other distinct families.',
              'actual_public_questions': 0, 'new_solver_calls': 0, 'new_training_runs': 0,
              'day3_complete': False}
    out = HERE / 'FAMILY_SELECTION.json'
    assert not out.exists()
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'selected': len(selected), 'counts': report['counts']}))


if __name__ == '__main__':
    main()
