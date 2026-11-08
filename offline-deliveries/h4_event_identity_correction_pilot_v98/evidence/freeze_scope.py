"""Freeze selected scope and code receipts before any model call."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

HOME = Path(__file__).resolve().parents[1]

def read(p): return json.loads(p.read_text())
def bind(p):
    p = p.resolve()
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
def write(p, value):
    with p.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')

def main():
    assert not (HOME / 'expansion_engine/runs').exists()
    for row in read(HOME / 'PRESERVATION_BEFORE.json')['files']:
        assert bind(Path(row['path']))['sha256'] == row['sha256'], row['path']
    for name in ('EXPANSION_OFFLINE_TESTS.log', 'ORDERING_MINFIX_TEST_RESULTS.log'):
        text = (HOME / 'evidence' / name).read_text()
        assert '\nOK\n' in text and 'FAILED (' not in text, name
    for name in ('CORRECTION_MINFIX_REPORT.json', 'EVENT_IDENTITY_MINFIX_REPORT.json',
                 'INVENTORY_INDEPENDENT_AUDIT.json', 'INTEGRATION_INDEPENDENT_REVIEW.json'):
        assert (HOME / 'evidence' / name).is_file(), name
    assert read(HOME / 'evidence/INTEGRATION_INDEPENDENT_REVIEW.json')['status'].startswith('PASS')
    for engine, receipt in [('expansion_engine', 'EXPANSION_MINFIX_RECEIPT.json'),
                            ('qa_engine', 'ORDERING_MINFIX_RECEIPT.json')]:
        files = list((HOME / engine).glob('*.py'))
        if engine == 'expansion_engine':
            files += [HOME / engine / x for x in ('POLICY.md', 'SOURCE_REGISTRY.json', 'SOURCE_SELECTION.json')]
        write(HOME / 'evidence' / receipt, {'version': 'v98_precall_code_receipt_v1',
              'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
              'files': [bind(p) for p in sorted(files)],
              'semantic_reliability_certified_by_receipt': False})
    selection = read(HOME / 'expansion_engine/SOURCE_SELECTION.json')
    write(HOME / 'PRECALL_PLAN.json', {
        'version': 'v98_precall_four_root_scope_v1',
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'candidate_roots': selection['roots'], 'source_ids': selection['source_ids'],
        'selection_timing': 'Before any v98 prepare, reservation or model call; no replacement after failure.',
        'selection_bias': selection['selection_bias'], 'roots_are_first_attempted_in_history': False,
        'scenes_per_root': 1, 'route': 'dataset_new_event', 'expansion_max_tokens': 4096,
        'max_card_extractions_per_source': 2, 'max_initial_state_attempts': 2,
        'max_planner_attempts': 2, 'max_event_review_attempts_per_plan': 2,
        'max_realizer_attempts': 2, 'max_scene_review_attempts_per_body': 2,
        'max_transport_retries_per_request': 1,
        'max_expansion_logical_requests': 64, 'max_expansion_http_attempts': 128,
        'qa_output_max_tokens': 6144, 'question_types': ['multi_select', 'ordering'],
        'gold_targets': [2, 3, 4, 5], 'fresh_parent_gold6_controls_per_ready_root': 1,
        'ordering_targets_per_ready_root': 2, 'max_qa_http': 12,
        'max_solver_calls_per_model': 456, 'independently_ready_new_sources_only': True,
        'historical_source_fallback': False, 'failure_replacement': False,
        'training_allowed': False, 'formal_training_admission': False,
        'prior_deliveries_immutable': True,
        'scope_minfix': 'Explicit event identity and old-event comparison; precise enum correction; verify targeted correction changed before full-validation acceptance.',
        'regression_cases_are_new_independent_yield': False,
        'frozen_inputs': [bind(HOME / 'evidence' / name) for name in (
            'ROOT_PREFLIGHT_INVENTORY.json', 'ROOT_PREFLIGHT_SELECTION.json',
            'INVENTORY_INDEPENDENT_AUDIT.json', 'CORRECTION_MINFIX_REPORT.json',
            'EVENT_IDENTITY_MINFIX_REPORT.json', 'INTEGRATION_INDEPENDENT_REVIEW.json',
            'EXPANSION_OFFLINE_TESTS.log',
            'ORDERING_MINFIX_TEST_RESULTS.log', 'EXPANSION_MINFIX_RECEIPT.json',
            'ORDERING_MINFIX_RECEIPT.json')]})
    print(json.dumps({'status': 'FROZEN_PRECALL', 'roots': selection['roots'],
                      'source_ids': selection['source_ids']}))

if __name__ == '__main__': main()
