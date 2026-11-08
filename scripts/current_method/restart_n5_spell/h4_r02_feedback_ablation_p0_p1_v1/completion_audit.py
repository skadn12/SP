"""Supplementary completion evidence; never changes frozen inputs or metrics."""
from __future__ import annotations

import argparse
from pathlib import Path

from .freeze import DEFAULT, ARMS, ROOTS, load, rows, source, write
from .finalize import validate_review, require


def review_items(run: Path, complete=False):
    base = run / 'diagnostics/blind_review'
    checked = []
    for path in sorted((base / 'reviews').glob('*.json')):
        review = load(path)
        case = load(base / 'cases' / path.name)
        validate_review(review, case)
        require(review.get('review_status') == 'reviewed', 'review_not_finished')
        candidate = case['candidate']
        items = candidate['items'] if candidate['question_type'] == 'ordering' else candidate['options']
        expected = set(items) if isinstance(items, dict) else {item.get('item_id', item.get('option_id')) for item in items}
        checks = review['item_checks']
        actual = [item['item_id'] for item in checks]
        require(len(actual) == len(expected) and set(actual) == expected, 'review_item_coverage_mismatch:' + case['blind_id'])
        valid_refs = {turn['turn_ref'] for turn in case['transcript']['turns']}
        for item in checks:
            require(bool(item.get('support_status') or item.get('support')) and bool(item.get('notes')), 'review_item_judgment_missing')
            require(isinstance(item.get('evidence_refs'), list) and all(ref in valid_refs for ref in item['evidence_refs']), 'review_item_invalid_evidence_ref')
            state = item.get('support_status') or item.get('support')
            require(not state.startswith('supported') or bool(item['evidence_refs']), 'review_supported_item_missing_evidence')
        checked.append({'blind_id': case['blind_id'], 'root_id': case['root_id'], 'item_count': len(actual), 'review': source(path)})
    if complete:
        mapping = rows(base / 'arm_mapping_do_not_show_reviewers.jsonl')
        require({row['blind_id'] for row in mapping} == {row['blind_id'] for row in checked}, 'review_scope_incomplete')
    return {'status': 'PASS', 'complete_scope_required': complete, 'reviews_checked': len(checked),
            'items_checked': sum(row['item_count'] for row in checked), 'rows': checked,
            'meaning': 'Structural review completeness and reference existence only; no automatic semantic adjudication.'}


def final(run: Path):
    summary = load(run / 'reports/p1_comparison.json')
    require(summary['status'] == 'COMPLETE' and all(summary['completion_checks'].values()), 'p1_report_not_complete')
    require(summary['primary']['metric_complete'], 'six_root_primary_incomplete')
    quality = summary['quality_review']
    for citation in [quality['mapping_source']] + [citation for row in quality['review_sources']
                      for citation in (row['case_source'], row['review_source'])]:
        require(source(Path(citation['path']))['sha256'] == citation['sha256'], 'review_source_changed_since_comparison')
    budget = load(run / 'solver_budget_manifest.json')
    audit = load(run / 'audit/solver_denominator_audit.json')
    require(audit['total_requests'] == budget['all_models_both_arms_request_count'], 'final_solver_budget_mismatch')
    require(len(audit['rows']) == 6, 'three_models_two_arms_incomplete')
    require({(row['arm'], row['model']) for row in audit['rows']} == {
        (arm, model) for arm in ARMS for model in ('M0', 'M_clean_policy', 'M_inclusive_policy')}, 'solver_model_arm_scope_mismatch')
    for arm in ARMS:
        require(load(run / arm / 'solver/comparison.json')['status'] == 'COMPLETE', 'solver_arm_incomplete')
    result = {'status': 'PASS', 'review_item_audit': review_items(run, complete=True),
              'solver_requests_completed': audit['total_requests'], 'solver_budget': source(run / 'solver_budget_manifest.json'),
              'comparison': source(run / 'reports/p1_comparison.json'),
              'roots': list(ROOTS), 'arms': list(ARMS), 'stop_stage': 'P1',
              'scope_changed': False, 'generation_scoring_or_selection_changed': False}
    write(run / 'audit/p1_completion_audit.json', result)
    print({'status': result['status'], 'reviews': result['review_item_audit']['reviews_checked'],
           'solver_requests': result['solver_requests_completed']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT)
    parser.add_argument('--phase', choices=('reviews', 'final'), required=True)
    args = parser.parse_args()
    if args.phase == 'reviews':
        report = review_items(args.run)
        print({k: v for k, v in report.items() if k != 'rows'})
    else:
        final(args.run)
