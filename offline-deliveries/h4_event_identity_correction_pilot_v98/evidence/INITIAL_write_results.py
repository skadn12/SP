"""Publish actual completed experiment accounting after all independent audits."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

HOME = Path(__file__).resolve().parents[1]
RUN = HOME / 'expansion_engine/runs/four_roots_B_scene01'
def read(p): return json.loads(p.read_text())
def bind(p): return {'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
def write(p, obj):
    with p.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2); f.write('\n')

def main():
    plan = read(HOME / 'PRECALL_PLAN.json'); manifest = read(RUN / 'MANIFEST.json')
    stage = read(HOME / 'evidence/STAGE_RESULTS.json')
    transport = read(HOME / 'evidence/TRANSPORT_FEEDBACK_AUDIT.json')
    assert transport['status'] == 'PASS'
    rows = []; ready = []
    for row in stage['roots']:
        root = row['root_id']; short = root.rsplit('-', 1)[1]
        audit_path = HOME / f'evidence/root{short}_INDEPENDENT_SOURCE_AUDIT.json'
        audit = read(audit_path)
        assert audit['root_id'] == root
        assert row['model_accepted_scenes'] == row['model_accepted_new_turns'] == row['raw_generated_turns'] == 0
        assert not (HOME / f'sources/root{short}_SOURCE.json').exists()
        rows.append({'root_id': root, 'source_ids': row['source_ids'],
                     'historical_turns': row['input_turns'],
                     'structurally_accepted_cards': 1, 'initial_state_scans_passed': 1,
                     'planner_attempts': row['request_roles']['planner'],
                     'planner_identity_field_present': 0,
                     'terminal_failure': 'plan_rejected_after_retry',
                     'exact_failure_code': 'event_identity_fields_invalid',
                     'missing_path': 'scene.event_identity',
                     'event_review_requests': 0, 'realizer_requests': 0,
                     'raw_body_candidates': 0, 'raw_new_turns': 0,
                     'model_accepted_scenes': 0, 'model_accepted_new_turns': 0,
                     'materialized_source': None, 'independent_status': 'NO_ACCEPTED_NEW_SOURCE',
                     'ready_for_joint_qa': False, 'ready_new_turn_count': 0,
                     'independent_evidence': [bind(audit_path)]})
    write(HOME / 'evidence/READY_SOURCE_DECISIONS.json', {
        'version': 'v98_independent_source_decisions_v1', 'roots': rows,
        'ready_roots': [], 'ready_new_turns': 0, 'training_allowed': False,
        'reason': 'All four actual bounded Planner attempts failed structure; no realized new source exists.',
        'blindness': 'Reviewers authored portions of the fix and knew prior cases; individual exposure is disclosed. No blind semantic certification.'})
    totals = {**stage['totals'], 'historical_turns': sum(r['historical_turns'] for r in rows),
              'logical_requests': transport['summary']['logical_requests'],
              'http_attempts': transport['summary']['actual_http_attempts'],
              'http_200': transport['summary']['http_statuses']['200'],
              'planner_requests': stage['roles']['planner'], 'planner_identity_omissions': 8,
              'initial_state_requests': stage['roles']['state_extractor'],
              'initial_state_scans_passed': 4,
              'independent_ready_roots': 0, 'independent_ready_new_turns': 0,
              'qa_http_calls': 0, 'solver_calls_per_model': {'M0': 0, 'M_clean_policy': 0},
              'new_multiselect_questions': 0, 'fresh_gold6_controls': 0, 'new_ordering_questions': 0,
              'content_usable_questions': 0, 'new_selected_candidates': 0, 'training_jobs': 0}
    # The upstream collector's legacy label "usable_cards" means structural
    # acceptance only. Preserve it in its own artifact, use the accurate label here.
    totals['structurally_accepted_cards'] = totals.pop('usable_cards')
    results = {'version': 'v98_complete_actual_results_v1',
               'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'status': 'COMPLETE_WITH_ZERO_NEW_SOURCE_PLANNER_CONTRACT_FAILURE',
               'outcome': 'Four fresh source assignments ran to their frozen stopping condition; all eight Planner responses omitted required scene.event_identity. No body or QA was generated.',
               'run': str(RUN), 'totals': totals, 'roots': rows,
               'gold_distribution': {str(k): 0 for k in (2, 3, 4, 5, 6)},
               'roles': stage['roles'], 'usage': stage['usage'],
               'transport': transport['summary'],
               'feedback': stage['actual_previous_training_feedback'],
               'validation': {'expansion_offline_tests_passed': 157,
                              'ordering_offline_tests_passed': 12,
                              'inventory_independent_fixture_cases_passed': 15,
                              'independent_integration_bypasses_blocked': 2,
                              'checks_are_semantic_reliability_certificate': False},
               'actual_gate_coverage': {'card_enum_repair_triggered': False,
                    'planner_identity_shape_rejection': '8/8 responses rejected for missing field',
                    'whole_event_model_reviews': 0, 'scene_model_reviews': 0,
                    'event_or_scene_correction_progress_gate_requests': 0,
                    'semantic_fix_fresh_actual_effectiveness_measured': False},
               'source_inventory': {'before_occupied_identities': 96,
                    'newly_permanently_reserved_identities': 4, 'after_occupied_identities': 100,
                    'historical_conflicting_identities': 24, 'refresh_available_before': 88,
                    'refresh_available_after': 84, 'selected_sources_previously_unused': True,
                    'historical_global_uniqueness_restored': False,
                    'v96_selected_candidates_remain_provisional': True},
               'implementation_limit': 'Planner schema presentation was not fully unified: the old scene field list omits event_identity, the immediately following sentence adds it, while latest-state instructions still say exactly one extra field state_transition. Full new shapes, repeated final instructions and precise retry errors were present. These observations do not isolate a causal explanation for all omissions.',
               'selection': {'random': False, 'roots_first_attempted': False,
                    'bias': plan['selection_bias'], 'failure_replacement': False,
                    'regression_cases_counted_as_fresh_yield': False},
               'solver_accuracy': None, 'difficulty_comparison': None,
               'accuracy_unavailable_reason': 'Zero new questions and zero Solver calls; unavailable is not 0% accuracy.',
               'training_allowed': False, 'formal_training_admission': False,
               'semantic_reliability_certified': False, 'stable_production_ready': False,
               'evidence': {name: bind(HOME / name) for name in (
                   'PRECALL_PLAN.json', 'PILOT_PLAN.json', 'evidence/STAGE_RESULTS.json',
                   'evidence/TRANSPORT_FEEDBACK_AUDIT.json', 'evidence/ACTUAL_MINFIX_OBSERVATIONS.json',
                   'evidence/READY_SOURCE_DECISIONS.json', 'evidence/QA_SKIPPED_NO_READY_SOURCE.json',
                   'evidence/ACTUAL_QA_INPUT_AUDIT.json', 'evidence/FINAL_CODE_INPUT_INTEGRITY.json',
                   'evidence/INTEGRATION_INDEPENDENT_REVIEW.json')}}
    write(HOME / 'RESULTS.json', results)
    print(json.dumps({'status': results['status'], 'totals': totals}, ensure_ascii=False))
if __name__ == '__main__': main()
