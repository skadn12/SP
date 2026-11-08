"""Validate the preselected four-root expansion and its ordered READY QA subset."""
from pathlib import Path
import hashlib
import json
import re

PLAN_VERSION = 'v97_four_root_semantic_review_mixed_v1'
DELIVERIES = Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries')
PILOT = Path(__file__).resolve().parent
# Installed shard lives two directories below the delivery; engine lives one.
PILOT = PILOT.parent.parent if PILOT.parent.name == 'shards' else PILOT.parent
PLAN_PATH = PILOT / 'PILOT_PLAN.json'


def need(ok, reason):
    if not ok:
        raise ValueError('v97_scope:' + reason)


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def plan_data(plan=None):
    return json.loads(PLAN_PATH.read_text()) if plan is None else plan


def selected_roots(plan=None):
    rows = plan_data(plan).get('candidate_roots')
    need(isinstance(rows, list) and len(rows) == len(set(rows)) == 4
         and all(isinstance(root, str) and re.fullmatch(r'cc-train-episode-\d+', root) for root in rows),
         'four_unique_preselected_roots_required')
    return rows


def upstream_run(plan=None):
    bindings = plan_data(plan).get('upstream_expansion_run')
    need(isinstance(bindings, dict) and set(bindings) == {'MANIFEST.json', 'REPORT.json', 'FINAL_AUDIT.json'},
         'upstream_expansion_binding_schema')
    parents = {Path(row['path']).resolve().parent for row in bindings.values()}
    need(len(parents) == 1, 'upstream_expansion_bindings_not_one_run')
    run = parents.pop()
    need(run.name == 'four_roots_B_scene01' and run.parent.name == 'runs'
         and PILOT.resolve() in run.parents, 'upstream_expansion_run_wrong_scope')
    return run


def validate_roots(roots, plan=None, *, allow_empty=False):
    selected = selected_roots(plan)
    need(isinstance(roots, list) and (allow_empty or bool(roots))
         and roots == [root for root in selected if root in roots],
         'qa_requires_unique_ready_subset_in_frozen_order')


def upstream_bindings(plan=None):
    return {name: bind(upstream_run(plan) / name) for name in
            ('MANIFEST.json', 'REPORT.json', 'FINAL_AUDIT.json')}


def validate_plan(plan):
    need(plan.get('version') == PLAN_VERSION
         and type(plan.get('candidate_root_budget')) is int
         and plan['candidate_root_budget'] == 4
         and plan.get('source_split') == 'train'
         and plan.get('training_allowed') is False
         and plan.get('formal_training_admission') is False,
         'current_four_root_plan_required')
    selected = selected_roots(plan)
    need(plan.get('upstream_expansion_run') == upstream_bindings(plan),
         'upstream_expansion_bindings_changed')
    need(plan.get('question_types') == ['multi_select', 'ordering']
         and plan.get('max_generation_http_total') == 12
         and plan.get('max_solver_calls_per_model_total') == 456,
         'current_mixed_budget_required')
    run = upstream_run(plan)
    report = json.loads((run / 'REPORT.json').read_text())
    audit = json.loads((run / 'FINAL_AUDIT.json').read_text())
    manifest = json.loads((run / 'MANIFEST.json').read_text())
    need(audit.get('status') == 'PASS' and audit.get('failures') == []
         and audit.get('semantic_certified') is False,
         'upstream_structural_audit_required')
    need(manifest.get('roots') == selected, 'upstream_preselected_roots_changed')
    need({row['root_id'] for row in report['routes'] if row['route'] == 'dataset_new_event'} == set(selected),
         'upstream_outcome_coverage')
    return plan
