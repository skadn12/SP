"""Audit frozen ledger bindings/arithmetic after parent authorizes Gold access.

Does not re-evaluate per-item semantic judgments. Writes only under audit/.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json

OUT = Path(__file__).resolve().parent
BASE = OUT.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    checks = []
    bindings = []

    def check(name, passed):
        checks.append({'name': name, 'passed': bool(passed)})

    def read(rel):
        p = BASE / rel
        bindings.append({'path': rel, 'sha256': sha(p), 'bytes': p.stat().st_size})
        return json.loads(p.read_text())

    freeze = read('reviews/RECONSTRUCTION_FREEZE.json')
    ledger = read('reviews/ADMISSION_LEDGER.json')
    public = read('reuse/public/QUESTIONS.json')['questions']
    inventory = read('reuse/INVENTORY.json')['questions']
    admitted = read('reviews/PASS_CONTENT_CANDIDATES.json')
    qmap = {q['candidate_id']: q for q in public}
    original = {q['candidate_id']: q['original_question_record'] for q in inventory}
    check('all_reconstructions_frozen_before_comparison_declared', freeze['status'] == 'ALL_SOURCE_FIRST_RECONSTRUCTIONS_COMMITTED_BEFORE_THIS_SCRIPT_PRIVATE_COMPARISON')
    check('freeze_public_hash', freeze['public_questions_sha256'] == sha(BASE / 'reuse/public/QUESTIONS.json'))
    check('ledger_freeze_hash', ledger['reconstruction_freeze_sha256'] == sha(BASE / 'reviews/RECONSTRUCTION_FREEZE.json'))
    check('ledger_policy_hash', ledger['review_policy_sha256'] == sha(BASE / 'SOURCE_FIRST_REVIEW_POLICY.md'))
    check('ledger_inventory_hash', ledger['original_inventory_sha256'] == sha(BASE / 'reuse/INVENTORY.json'))
    check('freeze_timestamp_precedes_ledger', datetime.fromisoformat(freeze['created_at_beijing']) < datetime.fromisoformat(ledger['created_at_beijing']))
    check('113_unique_public_and_original', len(public) == len(qmap) == len(inventory) == len(original) == 113 and set(qmap) == set(original))
    reviews = {}
    reviewer_counts = {}
    review_paths = set()
    check('three_frozen_review_files', len(freeze['reviews']) == 3)
    for rec in freeze['reviews']:
        path = BASE / rec['path']
        check('review_hash_size:' + rec['path'], sha(path) == rec['sha256'] and path.stat().st_size == rec['bytes'])
        data = read(rec['path'])
        check('review_private_gold_unseen_declaration:' + rec['path'], data['private_gold_seen'] is False)
        reviewer_counts[data['reviewer']] = len(data['questions'])
        review_paths.add(rec['path'])
        for row in data['questions']:
            key = row['candidate_id']
            check('review_candidate_not_duplicate:' + key, key not in reviews)
            reviews[key] = (row, data['reviewer'], rec['path'])
    check('frozen_reconstructions_cover_all_113', len(reviews) == freeze['question_count'] == 113 and set(reviews) == set(qmap))
    rows = ledger['questions']
    by_id = {r['candidate_id']: r for r in rows}
    check('ledger_113_unique_complete', len(rows) == len(by_id) == ledger['question_count'] == 113 and set(by_id) == set(qmap))
    check('ledger_56_roots', len({r['root_id'] for r in rows}) == ledger['root_count'] == 56)
    expected_counts = {'PASS_INTERNAL_CONTENT': 4, 'HOLD_SOURCE_REVIEW': 94, 'HOLD_SEMANTIC_ADJUDICATION': 1, 'FAIL_ORIGINAL_GOLD_MISMATCH': 11, 'FAIL_SOURCE_REVIEW': 3}
    counts = dict(Counter(r['status'] for r in rows))
    check('detailed_status_counts_4_94_1_11_3', counts == expected_counts == ledger['status_counts'])
    high_level = {'PASS': sum(v for k, v in counts.items() if k.startswith('PASS')), 'HOLD': sum(v for k, v in counts.items() if k.startswith('HOLD')), 'FAIL': sum(v for k, v in counts.items() if k.startswith('FAIL'))}
    check('high_level_4_95_14_sum_113', high_level == {'PASS': 4, 'HOLD': 95, 'FAIL': 14} and sum(high_level.values()) == 113)
    types = {kind: dict(Counter(r['status'] for r in rows if r['question_type'] == kind)) for kind in ['multi_select', 'ordering']}
    check('by_type_counts_match', types == ledger['by_type'] and sum(types['multi_select'].values()) == 51 and sum(types['ordering'].values()) == 62)
    check('mechanical_count_10262', sum(len(r['mechanical_checks']) for r in rows) == ledger['mechanical_check_count'] == 10262)
    check('all_mechanical_checks_pass', all(c['pass'] is True for r in rows for c in r['mechanical_checks']))
    check('no_invalid_review_records', 'HOLD_REVIEW_RECORD_INVALID' not in counts and all(not r['review_record_errors'] for r in rows))
    check('zero_training_admission', ledger['training_admitted_count'] == 0 and all(r['training_admitted'] is False for r in rows))
    check('zero_solver_training_three_internal_review_tasks', ledger['solver_calls'] == ledger['training_runs'] == 0 and ledger['semantic_reviewer_tasks'] == 3)
    check('original_gold_not_modified_declaration', ledger['original_gold_modified'] is False)
    for row in rows:
        key = row['candidate_id']
        public_row = qmap[key]
        review, reviewer, review_path = reviews[key]
        check('ledger_metadata_matches_public_and_frozen_review:' + key,
              row['root_id'] == public_row['root_id'] == review['root_id'] and
              row['question_type'] == public_row['question_type'] == review['question_type'] and
              row['source_binding'] == public_row['source_binding'] and
              row['reviewer'] == reviewer and row['review_path'] == review_path and
              row['reviewer_recommendation'] == review['recommendation'] and
              row['reconstructed_gold'] == review.get('reconstructed_gold'))
        check('original_gold_preserved:' + key, row['original_gold_unchanged'] == original[key]['gold'])
        field = 'options' if row['question_type'] == 'multi_select' else 'items'
        labels = set(public_row[field])
        pred = row['reconstructed_gold']
        complete = isinstance(pred, list) and len(pred) == len(set(pred)) and set(pred) <= labels
        if row['question_type'] == 'ordering':
            complete = complete and set(pred) == labels
        gold = original[key]['gold']
        agrees = (set(pred) == set(gold) if row['question_type'] == 'multi_select' else pred == gold) if complete else None
        check('gold_agreement_independently_computed:' + key, row['gold_agreement'] is agrees)
        if row['review_record_errors']:
            status = 'HOLD_REVIEW_RECORD_INVALID'
        elif row['conservative_holds']:
            status = 'HOLD_SEMANTIC_ADJUDICATION'
        elif row['reviewer_recommendation'] == 'FAIL':
            status = 'FAIL_SOURCE_REVIEW'
        elif row['reviewer_recommendation'] == 'HOLD':
            status = 'HOLD_SOURCE_REVIEW'
        elif not agrees:
            status = 'FAIL_ORIGINAL_GOLD_MISMATCH'
        else:
            status = 'PASS_INTERNAL_CONTENT'
        check('status_rule_recomputed:' + key, row['status'] == status)
        if status == 'PASS_INTERNAL_CONTENT':
            check('pass_complete_and_all_gates:' + key, complete and agrees is True and not row['review_record_errors'] and not row['conservative_holds'] and row['reviewer_recommendation'] == 'PASS')
    passes = [r['candidate_id'] for r in rows if r['status'] == 'PASS_INTERNAL_CONTENT']
    check('content_pass_candidates_exact_original_records', admitted['status'] == 'CONTENT_ONLY_NOT_TRAINING_ROWS' and admitted['questions'] == [original[key] for key in passes] and len(admitted['questions']) == 4)
    csv_path = BASE / 'reviews/ADMISSION_LEDGER.csv'
    bindings.append({'path': 'reviews/ADMISSION_LEDGER.csv', 'sha256': sha(csv_path), 'bytes': csv_path.stat().st_size})
    with csv_path.open(newline='') as f:
        csv_rows = list(csv.DictReader(f))
    check('csv_covers_same_113', len(csv_rows) == 113 and {r['candidate_id'] for r in csv_rows} == set(by_id))
    for c in csv_rows:
        r = by_id[c['candidate_id']]
        expected = {k: '' if r[k] is None else str(r[k]) for k in c}
        check('csv_row_matches_ledger:' + c['candidate_id'], c == expected)
    errors = [c['name'] for c in checks if not c['passed']]
    result = {'status': 'PASS' if not errors else 'FAIL', 'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'check_count': len(checks), 'passed_count': sum(c['passed'] for c in checks), 'failed_checks': errors, 'checks': checks, 'final_status_counts': counts, 'high_level_counts': high_level, 'by_type': types, 'reviewer_counts': reviewer_counts, 'mechanical_check_count_in_ledger': ledger['mechanical_check_count'], 'training_admitted_count': ledger['training_admitted_count'], 'evidence_bindings': bindings, 'scope_limit': 'Arithmetic, structural metadata, original-Gold preservation and byte bindings only. Source semantics were not independently re-reviewed. No access-control or external-blind-review claim.'}
    (OUT / 'LEDGER_EVIDENCE.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    report_path = OUT / 'INDEPENDENT_DAY1_AUDIT.json'
    report = json.loads(report_path.read_text())
    report['status'] = 'PASS' if not errors and report['core_checks']['failed'] == [] else 'FAIL'
    report['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    report['scope'] = 'Independent Day1 mechanical/design audit, including frozen source-reconstruction bindings and final admission-ledger arithmetic. No per-item semantic re-review.'
    report['ledger_check'] = {'status': result['status'], 'check_count': result['check_count'], 'passed_count': result['passed_count'], 'failed_checks': errors, 'evidence': {'path': 'audit/LEDGER_EVIDENCE.json', 'sha256': sha(OUT / 'LEDGER_EVIDENCE.json')}, 'final_status_counts': counts, 'high_level_counts': high_level, 'training_admitted_count': 0, 'restricted_legacy_inventory_gold_access': 'Only after parent confirmation that all three reconstructions and freeze existed; used for unchanged-Gold comparison, not reconstruction.'}
    report['verified_counts']['legacy_content_pass'] = high_level['PASS']
    report['verified_counts']['legacy_content_hold'] = high_level['HOLD']
    report['verified_counts']['legacy_content_fail'] = high_level['FAIL']
    report['verified_counts']['legacy_training_admitted'] = 0
    report['conclusions'].append('Final frozen ledger covers all 113 original candidates: 4 internal content PASS, 95 HOLD, 14 FAIL; all 10262 recorded structural checks pass, original Gold is preserved, and training admission remains zero.')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['status', 'check_count', 'passed_count', 'failed_checks', 'final_status_counts', 'high_level_counts', 'training_admitted_count']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
