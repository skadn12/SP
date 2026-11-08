"""Bind the two independently reviewed v95 roots and current QA budget scope."""
from pathlib import Path
import hashlib
import json

EXPECTED_ROOTS = ('cc-train-episode-25652', 'cc-train-episode-19708')
PLAN_VERSION = 'v96_two_ready_root_expansion_mixed_v1'
DELIVERIES = Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries')
UPSTREAM_RUN = (DELIVERIES / 'h4_controlled_six_root_expansion_v95'
                / 'expansion_engine/runs/six_roots_B_scene01')


def need(ok, reason):
    if not ok:
        raise ValueError('v96_scope:' + reason)


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def validate_roots(roots):
    need(isinstance(roots, list) and roots == list(EXPECTED_ROOTS),
         'exact_two_ready_roots_in_frozen_order_required')


def upstream_bindings():
    return {name: bind(UPSTREAM_RUN / name) for name in
            ('MANIFEST.json', 'REPORT.json', 'FINAL_AUDIT.json')}


def validate_plan(plan):
    need(plan.get('version') == PLAN_VERSION
         and type(plan.get('candidate_root_budget')) is int
         and plan['candidate_root_budget'] == 2
         and plan.get('source_split') == 'train'
         and plan.get('training_allowed') is False
         and plan.get('formal_training_admission') is False,
         'current_two_root_plan_required')
    validate_roots(plan.get('candidate_roots'))
    need(plan.get('upstream_expansion_run') == upstream_bindings(),
         'upstream_expansion_bindings_changed')
    need(plan.get('question_types') == ['multi_select', 'ordering']
         and plan.get('max_generation_http_total') == 6
         and plan.get('max_solver_calls_per_model_total') == 228,
         'current_mixed_budget_required')
    report = json.loads((UPSTREAM_RUN / 'REPORT.json').read_text())
    audit = json.loads((UPSTREAM_RUN / 'FINAL_AUDIT.json').read_text())
    need(audit.get('status') == 'PASS' and audit.get('failures') == []
         and audit.get('semantic_certified') is False,
         'upstream_structural_audit_required')
    accepted = {row['root_id'] for row in report['routes']
                if row['route'] == 'dataset_new_event' and row['accepted_scenes'] > 0}
    need(set(EXPECTED_ROOTS) <= accepted, 'selected_root_has_no_accepted_expansion')
    return plan
