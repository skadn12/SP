#!/usr/bin/env python3
"""Append-only shape adaptation of authored plans; no semantic fields are authored here.

The frozen question driver is AST-cloned in memory. Only the work directory,
plan-receipt lookup, exact schema clarification/normalization, unique call IDs,
and guarded call dispatch change. Its author/review/revision policy is reused.
"""
from __future__ import annotations
import argparse
import ast
import concurrent.futures
import copy
import fcntl
import hashlib
import inspect
import json
import os
import time
from pathlib import Path
import sys
from typing import Any

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import question_driver as q

EXPECTED_DRIVER_SHA256 = '33ab6b7350bbaa1c9003d66f9c15a7c0c87fd855cf373ae3810664ab4701886d'
KIND = 'mechanical_saved_plan_schema_recovery'
ROOT = BASE / 'question_plan_continuation'


def work_dir(root: str, arm: str) -> Path:
    return ROOT / q.root_number(root) / arm


def original_dir(root: str, arm: str) -> Path:
    return BASE / 'question_authoring' / arm / 'roots' / q.root_number(root)


def normalize_plan_response(raw: Any, arm: str) -> tuple[Any, list[dict[str, Any]]]:
    """Conservative data-shape changes, retaining every changed authored value."""
    value = copy.deepcopy(raw)
    logs: list[dict[str, Any]] = []
    rows = q.extract_list(value, 'plans', 'task_plans')
    for plan in rows:
        slot = plan.get('slot_id')
        if slot not in q.SLOTS:
            continue
        # Feedback-bearing G content is never normalized into an authorized plan.
        if arm == 'G' and q._nested_material_keys(plan):
            continue
        def replace(field: str, new: Any, mapping: str) -> None:
            old = copy.deepcopy(plan[field])
            plan[field] = new
            logs.append({'slot_id': slot, 'field': field, 'mapping': mapping,
                         'authored_original_value': old, 'normalized_value': copy.deepcopy(new),
                         'semantic_acceptance': False})
        needed = plan.get('necessary_r03_premises')
        if (isinstance(needed, list) and needed
                and all(isinstance(row, dict) and isinstance(row.get('turn_refs'), list)
                        and row['turn_refs'] and all(isinstance(ref, str) for ref in row['turn_refs'])
                        and isinstance(row.get('removal_consequence'), str)
                        and row['removal_consequence'].strip() for row in needed)):
            refs = list(dict.fromkeys(ref for row in needed for ref in row['turn_refs']))
            replace('necessary_r03_premises', {'turn_refs': refs,
                    'removal_consequence': q.stable(needed)},
                    'list_to_union_refs_and_verbatim_canonical_authored_json')
        quality = plan.get('quality_acceptance')
        if isinstance(quality, dict) and quality:
            replace('quality_acceptance', q.stable(quality), 'object_to_verbatim_canonical_json_string')
        target = plan.get('feedback_target')
        if arm == 'F' and isinstance(target, dict) and target.get('target_status') == 'TARGET_PLANNED':
            original = target['target_status']
            target['target_status'] = 'PROSPECTIVE_PROBE'
            logs.append({'slot_id': slot, 'field': 'feedback_target.target_status',
                         'mapping': 'authored_prospective_metadata_enum_alias_only',
                         'authored_original_value': original, 'normalized_value': 'PROSPECTIVE_PROBE',
                         'semantic_acceptance': False})
        if arm == 'G':
            if plan.get('question_type') == q.SLOTS[slot][0]:
                old = plan.pop('question_type')
                logs.append({'slot_id': slot, 'field': 'question_type',
                             'mapping': 'remove_exact_redundant_fixed_slot_type',
                             'authored_original_value': old, 'normalized_value': None,
                             'semantic_acceptance': False})
            scope = plan.get('task_scope')
            if isinstance(scope, str) and scope.strip() and isinstance(plan.get('quality_acceptance'), str):
                old_quality = plan['quality_acceptance']
                new_quality = q.stable({'authored_quality_acceptance': old_quality,
                                       'authored_task_scope': scope})
                plan['quality_acceptance'] = new_quality
                plan.pop('task_scope')
                logs.append({'slot_id': slot, 'field': 'task_scope',
                             'mapping': 'preserve_authored_scope_inside_quality_acceptance_canonical_json',
                             'authored_original_value': scope, 'previous_quality_acceptance': old_quality,
                             'normalized_value': new_quality, 'semantic_acceptance': False})
    return value, logs


def validate_plans(raw: Any, arm: str, root: str, source: dict[str, Any],
                   feedback_ids: set[str] | None = None):
    normalized, changes = normalize_plan_response(raw, arm)
    plans, errors = q.validate_plans(normalized, arm, root, source, feedback_ids)
    q.publish(work_dir(root, arm) / 'MECHANICAL_PLAN_NORMALIZATION.json', {
        'schema_version': 'fgr20_mechanical_plan_shape_normalization_v1',
        'root_id': root, 'arm': arm, 'original_response_sha256': q.digest(raw),
        'normalized_response_sha256': q.digest(normalized), 'changes': changes,
        'normalized_response': normalized, 'strict_validation_errors': errors,
        'no_semantic_generation_or_acceptance': True,
        'raw_response_is_preserved': True})
    return plans, errors


def task_plan_prompt(*args: Any, **kwargs: Any) -> str:
    prompt = q.task_plan_prompt(*args, **kwargs)
    clarification = (
        'Exact output data types: necessary_r03_premises is one object '
        '{"turn_refs":["R03:..."],"removal_consequence":"nonempty explanation"}; '
        'quality_acceptance is one nonempty string. Use only the six requested plan fields '
        '(slot_id, task_operation, source_relations, necessary_r03_premises, r03_necessity, '
        'quality_acceptance). These are serialization requirements; source facts, fixed '
        'task operations and acceptance standards remain unchanged.\n')
    if args[0] == 'F':
        clarification += (
            'For this F assignment add feedback_target with the six fields already required '
            'above. target_status is exactly PROSPECTIVE_PROBE or TARGET_HOLD. Both are '
            'prospective metadata requiring later independent review, never observed outcomes.\n')
    prompt = prompt.replace('FROZEN_INPUTS:\n', clarification + 'FROZEN_INPUTS:\n', 1)
    if args[0] == 'G':
        q.validate_g_prompt(prompt, 'task_planner', args[1])
    return prompt


def plan_receipt(work: Path) -> dict[str, Any]:
    reference = work / 'ORIGINAL_PLAN_RECEIPT_REFERENCE.json'
    if reference.exists():
        value = q.read(reference)['original_plan_call_receipt']
        q.verify_binding(value)
        return q.read(value['path'])
    return q.read(work / 'TASK_PLAN_CALL.json')


def continuation_call_id(stage: str, arm: str, root: str, suffix: str = '') -> str:
    return q.call_id(stage, arm, root, '_'.join(x for x in ('plan_schema_01', suffix) if x))


def active_calls() -> list[dict[str, Any]]:
    rows = []
    for path in (BASE / 'process_registry').glob('*.json'):
        try:
            value = q.read(path)
            pid = value.get('pid')
            if not value.get('active') or not isinstance(pid, int) or pid <= 0:
                continue
            os.kill(pid, 0)
            if value.get('start_token') != q.cc.process_start(pid):
                continue
            stat = Path(f'/proc/{pid}/stat').read_text().split()
            if len(stat) > 2 and stat[2] != 'Z':
                rows.append({'pid': pid, 'path': str(path)})
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, json.JSONDecodeError):
            continue
    return rows


class GuardedCalls:
    def __getattr__(self, key: str) -> Any:
        return getattr(q.cc, key)

    def call(self, key: str, prompt: str, **kwargs: Any):
        # Parent scheduler allocates all external calls. This additional fail-closed
        # check prevents continuation workers starting while three real calls live.
        lock_path = ROOT / '.call_start.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            live = active_calls()
            result_path = BASE / 'cli_calls' / key / 'RESULT.json'
            if len(live) >= 3 and not result_path.exists():
                raise RuntimeError(f'Global semantic call capacity is full: {live}')
            # Serialize only call startup until its actual PID registry appears,
            # then allow the next continuation to use another global slot.
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = pool.submit(q.cc.call, key, prompt, **kwargs)
            while not future.done() and not any(
                    (BASE / 'process_registry').glob(key + '__attempt_*.json')):
                time.sleep(0.05)
        try:
            return future.result()
        finally:
            pool.shutdown(wait=True)


class Adapt(ast.NodeTransformer):
    def __init__(self, function_name: str):
        self.function_name = function_name
        self.changed: list[str] = []

    def visit_Assign(self, node: ast.Assign):
        if (self.function_name == 'process_root_arm' and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'work'):
            node.value = ast.parse('work_dir(root, arm)', mode='eval').body
            self.changed.append('work_dir_expression')
            return node
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if (self.function_name == 'process_root_arm' and isinstance(node.func, ast.Name)
                and node.func.id == 'read' and len(node.args) == 1
                and ast.unparse(node.args[0]) == "work / 'TASK_PLAN_CALL.json'"):
            self.changed.append('saved_plan_receipt_lookup')
            return ast.copy_location(ast.Call(func=ast.Name(id='plan_receipt', ctx=ast.Load()),
                                             args=[ast.Name(id='work', ctx=ast.Load())], keywords=[]), node)
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        replacements = {'call_id': 'continuation_call_id', 'review_batch': 'continuation_review_batch',
                        'task_plan_prompt': 'continuation_task_plan_prompt',
                        'validate_plans': 'continuation_validate_plans'}
        if isinstance(node.ctx, ast.Load) and node.id in replacements:
            self.changed.append(f'bound_function:{node.id}')
            node.id = replacements[node.id]
        return node


def build_driver():
    actual = hashlib.sha256(Path(q.__file__).read_bytes()).hexdigest()
    if actual != EXPECTED_DRIVER_SHA256:
        raise RuntimeError(f'Frozen question driver drift: {actual}')
    namespace = dict(vars(q))
    namespace.update(work_dir=work_dir, plan_receipt=plan_receipt,
        continuation_call_id=continuation_call_id, continuation_task_plan_prompt=task_plan_prompt,
        continuation_validate_plans=validate_plans, cc=GuardedCalls())
    transformations = []
    for func in (q.review_batch, q.process_root_arm):
        source = inspect.getsource(func)
        tree = ast.parse(source)
        adapter = Adapt(func.__name__)
        tree = ast.fix_missing_locations(adapter.visit(tree))
        exec(compile(tree, f'<plan_schema_clone:{func.__name__}>', 'exec'), namespace)
        transformations.append({'function': func.__name__, 'original_source_sha256':
            hashlib.sha256(source.encode()).hexdigest(), 'adapted_ast_sha256':
            hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
            'changes': adapter.changed})
        if func is q.review_batch:
            namespace['continuation_review_batch'] = namespace['review_batch']
    process = namespace['process_root_arm']
    return process, transformations


def prepare_saved(root: str, arm: str, source_row: dict[str, Any]) -> dict[str, Any]:
    old = original_dir(root, arm)
    work = work_dir(root, arm)
    if (work / 'TASK_PLANS_FREEZE.json').exists():
        frozen = q.read(work / 'TASK_PLANS_FREEZE.json')
        q.verify_binding(frozen['plans_binding'])
        return frozen
    if not old.exists() or not any(p.name != '.driver.lock' for p in old.iterdir()):
        # An original live plan request must finish; never create a second planner.
        if (BASE / 'cli_calls' / q.call_id('plan', arm, root)).exists():
            raise RuntimeError('Original plan call exists without saved terminal plan; cannot regenerate')
        return {'status': 'NEW_ROOT_NOT_STARTED', 'root_id': root, 'arm': arm}
    # A failed actual planner response consumes its sole plan attempt. Preserve
    # that terminal failure as a final empty surface, never restart planning.
    original_outcome_path = old / 'OUTCOME.json'
    plan_failure_path = BASE / 'cli_calls' / q.call_id('plan', arm, root) / 'SEMANTIC_FAILURE.json'
    if (original_outcome_path.exists() and plan_failure_path.exists()
            and not (old / 'TASK_PLAN_RESPONSE.json').exists()):
        original_outcome = q.read(original_outcome_path)
        if (original_outcome.get('status') != 'FAILED_STAGE'
                or original_outcome.get('candidate_ids')
                or original_outcome.get('source_binding') != source_row['binding']):
            raise RuntimeError('Consumed-plan failure disposition does not match saved outcome')
        failure = q.read(plan_failure_path)
        if failure.get('semantic_attempt_consumed') is not True:
            raise RuntimeError('Saved plan failure is not explicitly classified as consumed')
        terminal_binding = failure.get('terminal')
        if not isinstance(terminal_binding, dict):
            raise RuntimeError('Saved consumed plan failure lacks a real terminal binding')
        q.verify_binding(terminal_binding)
        terminal = q.read(terminal_binding['path'])
        for key in ('request', 'events', 'stderr', 'raw_response'):
            if isinstance(terminal.get(key), dict):
                q.verify_binding(terminal[key])
        q.publish(work / 'PLAN_FAILURE_DISPOSITION.json', {
            'schema_version': 'fgr20_consumed_plan_failure_disposition_v1',
            'root_id': root, 'arm': arm, 'status': 'PLAN_ATTEMPT_ALREADY_CONSUMED',
            'original_outcome_binding': q.binding(original_outcome_path),
            'semantic_failure_binding': q.binding(plan_failure_path),
            'actual_terminal_binding': terminal_binding,
            'actual_plan_call_artifacts': [q.binding(path) for path in
                sorted(plan_failure_path.parent.rglob('*')) if path.is_file()],
            'new_plan_calls': 0, 'new_candidate_calls': 0,
            'semantic_failure_was_not_reclassified_as_transport': True})
        q.publish(work / 'TASK_PLANS.json', {
            'schema_version': 'fgr20_task_plans_v1', 'root_id': root, 'arm': arm,
            'plans': [], 'errors': ['original_plan_semantic_attempt_consumed_no_accepted_plan'],
            'plan_frozen_before_candidate_draft': True})
        q.publish(work / 'TASK_PLANS_FREEZE.json', {
            'schema_version': 'fgr20_task_plan_freeze_v1', 'root_id': root, 'arm': arm,
            'status': 'NO_ACCEPTED_PLAN_ATTEMPT_ALREADY_CONSUMED',
            'plans_binding': q.binding(work / 'TASK_PLANS.json'),
            'failure_disposition_binding': q.binding(work / 'PLAN_FAILURE_DISPOSITION.json'),
            'errors': ['original_plan_semantic_attempt_consumed_no_accepted_plan'],
            'frozen_before_candidate_draft': True, 'plan_calls_repeated': 0})
        q.publish(work / 'SLOT_FAILURES.json', original_outcome.get('slot_failures', []))
        q.publish(work / 'OUTCOME.json', {**original_outcome,
            'original_outcome_binding': q.binding(original_outcome_path),
            'consumed_plan_failure_disposition_binding': q.binding(work / 'PLAN_FAILURE_DISPOSITION.json'),
            'plan_calls_repeated': 0, 'candidate_author_calls': 0})
        return {'status': 'PLAN_ATTEMPT_ALREADY_CONSUMED', 'no_semantic_calls_authorized': True}
    required = ['OUTCOME.json', 'TASK_PLANS.json', 'TASK_PLANS_FREEZE.json',
                'TASK_PLAN_RESPONSE.json', 'TASK_PLAN_CALL.json']
    if any(not (old / name).exists() for name in required):
        raise RuntimeError('Original root/arm stage is not terminal and fully saved')
    outcome = q.read(old / 'OUTCOME.json')
    if outcome.get('status') != 'FAILED_NO_PLANS' or outcome.get('candidate_ids'):
        raise RuntimeError('Saved-plan recovery is restricted to FAILED_NO_PLANS without candidate drafts')
    if outcome.get('source_binding') != source_row['binding']:
        raise RuntimeError('Original source binding differs')
    # A REQUEST consumes semantic budget even if no accepted response exists.
    prefixes = (q.call_id('author', arm, root), q.call_id('revision', arm, root))
    for call_dir in (BASE / 'cli_calls').glob('q20_*'):
        if any(call_dir.name.startswith(prefix) for prefix in prefixes) and any(call_dir.rglob('REQUEST.txt')):
            raise RuntimeError(f'Candidate/revision budget already requested: {call_dir.name}')
    old_freeze = q.read(old / 'TASK_PLANS_FREEZE.json')
    q.verify_binding(old_freeze['plans_binding'])
    q.verify_binding(old_freeze['call_receipt'])
    receipt = q.read(old / 'TASK_PLAN_CALL.json')
    for key in ('request', 'parsed_response', 'terminal_binding'):
        q.verify_binding(receipt[key])
    raw = q.read(old / 'TASK_PLAN_RESPONSE.json')
    if raw != q.read(receipt['parsed_response']['path']):
        raise RuntimeError('Saved planner response does not match actual receipt')
    if not receipt.get('backend', {}).get('thread_id') or receipt.get('response_received') is not True:
        raise RuntimeError('Saved plan has no actual completed fresh-session receipt')
    feedback, _, _, _ = q.feedback_material() if arm == 'F' else ({}, None, None, None)
    ids = {str(c.get('feedback_id')) for c in feedback.get('cards', []) if isinstance(c, dict)}
    plans, errors = validate_plans(raw, arm, root, source_row['source'], ids)
    q.publish(work / 'ORIGINAL_PLAN_RECEIPT_REFERENCE.json', {
        'original_plan_call_receipt': q.binding(old / 'TASK_PLAN_CALL.json'),
        'original_plan_response': q.binding(old / 'TASK_PLAN_RESPONSE.json'),
        'original_plan_freeze': q.binding(old / 'TASK_PLANS_FREEZE.json'),
        'original_outcome': q.binding(old / 'OUTCOME.json'),
        'plan_call_repeated': False, 'candidate_draft_calls_before_continuation': 0})
    q.publish(work / 'TASK_PLANS.json', plans)
    return q.publish(work / 'TASK_PLANS_FREEZE.json', {
        'schema_version': 'fgr20_task_plan_freeze_v1', 'root_id': root, 'arm': arm,
        'plans_binding': q.binding(work / 'TASK_PLANS.json'),
        'call_receipt': q.binding(old / 'TASK_PLAN_CALL.json'), 'errors': errors,
        'frozen_before_candidate_draft': True, 'original_plan_is_immutable': True,
        'normalization_binding': q.binding(work / 'MECHANICAL_PLAN_NORMALIZATION.json')})


def publish_ready(root: str, arm: str, outcome: dict[str, Any]) -> dict[str, Any]:
    work = work_dir(root, arm)
    old = original_dir(root, arm)
    # Failed stages still expose truthful empty final surfaces; raw failures stay.
    if not (work / 'CANDIDATES_FINAL.json').exists():
        q.publish(work / 'CANDIDATES_FINAL.json', [])
    if not (work / 'CONTENT_REVIEW_FINAL.json').exists():
        q.publish(work / 'CONTENT_REVIEW_FINAL.json', {'reviews': [], 'mechanical_audits': [],
            'all_reviews_ai_family_not_human': True})
    if not (work / 'SLOT_FAILURES.json').exists():
        q.publish(work / 'SLOT_FAILURES.json', outcome.get('slot_failures', []))
    if not (work / 'TASK_PLANS.json').exists():
        q.publish(work / 'TASK_PLANS.json', {'schema_version': 'fgr20_task_plans_v1',
            'root_id': root, 'arm': arm, 'plans': [],
            'errors': ['no_accepted_plan_response_or_stage_failed'],
            'plan_frozen_before_candidate_draft': True})
    if not (work / 'TASK_PLANS_FREEZE.json').exists():
        q.publish(work / 'TASK_PLANS_FREEZE.json', {'schema_version': 'fgr20_task_plan_freeze_v1',
            'root_id': root, 'arm': arm, 'status': 'NO_ACCEPTED_PLAN',
            'plans_binding': q.binding(work / 'TASK_PLANS.json'),
            'errors': ['no_accepted_plan_response_or_stage_failed'],
            'frozen_before_candidate_draft': True, 'no_plan_repeated': True})
    # The cloned driver's OUTCOME stays immutable. A separately bound integration
    # outcome adds the actual empty surfaces when a stage never reached assembly.
    integration_outcome = dict(outcome, final_candidates_binding=q.binding(work / 'CANDIDATES_FINAL.json'),
        review_binding=q.binding(work / 'CONTENT_REVIEW_FINAL.json'),
        slot_failures_binding=q.binding(work / 'SLOT_FAILURES.json'))
    q.publish(work / 'CONTINUATION_INTEGRATION_OUTCOME.json', integration_outcome)
    ready = {'schema_version': 'fgr20_question_continuation_finalizer_ready_v1',
        'root_id': root, 'arm': arm, 'continuation_kind': KIND,
        'original_work_dir': str(old), 'final_work_dir': str(work),
        'actual_plan_path': str(work / 'TASK_PLANS.json'),
        'actual_plan_freeze_path': str(work / 'TASK_PLANS_FREEZE.json'),
        'original_outcome_binding': q.binding(old / 'OUTCOME.json') if (old / 'OUTCOME.json').exists() else None,
        'final_outcome_binding': q.binding(work / 'CONTINUATION_INTEGRATION_OUTCOME.json'),
        'driver_outcome_binding': q.binding(work / 'OUTCOME.json'),
        'final_candidates_binding': q.binding(work / 'CANDIDATES_FINAL.json'),
        'final_review_binding': q.binding(work / 'CONTENT_REVIEW_FINAL.json'),
        'slot_failures_binding': q.binding(work / 'SLOT_FAILURES.json'),
        'normalization_binding': q.binding(work / 'MECHANICAL_PLAN_NORMALIZATION.json')
             if (work / 'MECHANICAL_PLAN_NORMALIZATION.json').exists() else None,
        'no_semantic_budget_expansion': True}
    q.publish(work / 'CONTINUATION_FINALIZER_READY.json', ready)
    return ready


def execute(root: str, arm: str, row: dict[str, Any], selection: Path,
            process: Any, transformations: list[dict[str, Any]], run: bool):
    work = work_dir(root, arm)
    work.mkdir(parents=True, exist_ok=True)
    with (work / '.continuation.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        q.publish(work / 'DRIVER_ADAPTATION_V2.json', {'schema_version': 'fgr20_question_plan_driver_adaptation_v1',
            'root_id': root, 'arm': arm, 'original_driver_binding': q.binding(q.__file__),
            'wrapper_binding': q.binding(__file__), 'transformations': transformations,
            'source_binding': row['binding'], 'original_files_untouched': True,
            'semantic_policy_changes': False, 'unique_call_namespace': 'plan_schema_01'})
        prepared = prepare_saved(root, arm, row)
        if (work / 'PLAN_FAILURE_DISPOSITION.json').exists():
            return publish_ready(root, arm, q.read(work / 'OUTCOME.json'))
        if not run:
            return {'root_id': root, 'arm': arm, 'preparation': prepared}
        if q.cc.stop_generation():
            raise RuntimeError('Generation cutoff: no fresh semantic continuation permitted')
        contract_path = BASE / 'question_authoring/COMMON_AUTHOR_CONTRACT_ZH.md'
        contract = contract_path.read_text()
        feedback, adapter, fb, ab = q.feedback_material() if arm == 'F' else (None, None, None, None)
        outcome = process(root, arm, row, selection, contract, q.binding(contract_path), feedback, adapter, fb, ab)
        return publish_ready(root, arm, outcome)


def self_check():
    process, transformations = build_driver()
    source = q.read(BASE / 'source_recovery/cc-train-episode-55788/SOURCE_FINAL.json')
    raw = q.read(BASE / 'question_authoring/F/roots/55788/TASK_PLAN_RESPONSE.json')
    normalized, changes = normalize_plan_response(raw, 'F')
    feedback, _, _, _ = q.feedback_material()
    ids = {str(c.get('feedback_id')) for c in feedback['cards']}
    plans, errors = q.validate_plans(normalized, 'F', 'cc-train-episode-55788', source, ids)
    assert not errors, errors
    assert len(plans['plans']) == 8 and len(changes) == 24
    assert raw == q.read(BASE / 'question_authoring/F/roots/55788/TASK_PLAN_RESPONSE.json')
    # G tests mechanically adapt copies of real authored data, never write them or
    # send these test objects to any semantic call.
    test_g = copy.deepcopy(raw)
    for plan in test_g['plans']:
        plan.pop('feedback_target', None)
        plan.pop('operation_class', None)
    gn, _ = normalize_plan_response(test_g, 'G')
    _, errors = q.validate_plans(gn, 'G', 'cc-train-episode-55788', source)
    assert not errors, errors
    test_g['plans'][0]['feedback_target'] = {'feedback_id': 'not_authorized'}
    bad, _ = normalize_plan_response(test_g, 'G')
    _, errors = q.validate_plans(bad, 'G', 'cc-train-episode-55788', source)
    assert any('unauthorized' in e for e in errors)
    wrong = copy.deepcopy(gn)
    wrong['plans'][0]['question_type'] = 'invalid_type'
    wn, _ = normalize_plan_response(wrong, 'G')
    _, errors = q.validate_plans(wn, 'G', 'cc-train-episode-55788', source)
    assert any('unauthorized_g_plan_fields' in e for e in errors)
    assert transformations[1]['changes'].count('work_dir_expression') == 1
    assert transformations[1]['changes'].count('saved_plan_receipt_lookup') == 1
    contract = (BASE / 'question_authoring/COMMON_AUTHOR_CONTRACT_ZH.md').read_text()
    gprompt = task_plan_prompt('G', 'cc-train-episode-55788',
        q.source_view({'source': source, 'binding': q.binding(BASE / 'source_recovery/cc-train-episode-55788/SOURCE_FINAL.json')}),
        contract, None, None)
    gclarification = gprompt.split('Exact output data types:', 1)[1].split('FROZEN_INPUTS:', 1)[0]
    assert all(token not in gclarification.lower() for token in ('feedback', 'for f', 'prospective_probe', 'target_hold'))
    assert continuation_call_id('author', 'F', '55788') == 'q20_author_f_55788_plan_schema_01'
    return {'status': 'PASS', 'real_saved_f_plans_valid_after_shape_adaptation': 8,
            'real_f_structural_changes': len(changes), 'semantic_calls': 0,
            'original_driver_sha256': EXPECTED_DRIVER_SHA256, 'transformations': transformations}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'self-check'])
    parser.add_argument('--root', action='append')
    parser.add_argument('--arm', action='append', choices=q.ARMS)
    parser.add_argument('--selection', default=str(BASE / 'SOURCE_SELECTION_FINAL.json'))
    parser.add_argument('--source-freeze', default=str(BASE / 'SOURCE_FREEZE.json'))
    parser.add_argument('--workers', type=int, default=1, choices=[1, 2, 3])
    args = parser.parse_args()
    if args.command == 'self-check':
        print(json.dumps(self_check(), ensure_ascii=False)); return 0
    if not args.root or not args.arm:
        parser.error('Explicit --root and --arm are required to avoid accidental semantic batches')
    selection = Path(args.selection).resolve()
    q.SOURCE_SELECTION_PATH = selection
    q.SOURCE_FREEZE_PATH = Path(args.source_freeze).resolve()
    _, rows = q.load_selection(selection)
    process, transformations = build_driver()
    jobs = [(q.root_id(root), arm) for root in args.root for arm in args.arm]
    def one(job):
        root, arm = job
        if root not in rows:
            raise RuntimeError(f'Root has no approved frozen source: {root}')
        return execute(root, arm, rows[root], selection, process, transformations, args.command == 'run')
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for value in pool.map(one, jobs):
            print(json.dumps(value, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
