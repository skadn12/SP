#!/usr/bin/env python3
"""Resume an actual saved initial draft after reversible A-F serialization repair.

Only options/items [{label,text}] -> {label:text} and item_analysis labelled
records -> labelled dictionaries are adapted. No question, text, Gold, evidence,
or semantic field is changed. The original author/planner are never called.
First semantic review and at most one substantive revision use fresh sessions.
"""
from __future__ import annotations
import argparse
import ast
import copy
import fcntl
import hashlib
import inspect
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Any
sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import question_driver as q
import question_plan_schema_continuation_v3 as v3

EXPECTED_V3_SHA256 = '083c968fad2c28cde34efd1b90f87ba97945020d9bc020d65b2e5f387c339b66'
ROOT = BASE / 'question_candidate_shape_continuation'
KIND = 'mechanical_saved_candidate_shape_recovery'
LABELS = set('ABCDEF')


def work_dir(root: str, arm: str) -> Path:
    return ROOT / q.root_number(root) / arm


def check_local(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    resolved.relative_to(BASE)
    return resolved


def normalizer(raw: Any) -> tuple[Any, list[dict[str, Any]]]:
    value = copy.deepcopy(raw)
    rows = q.extract_list(value, 'candidates', 'questions')
    changes = []
    for index, candidate in enumerate(rows):
        for field in ('options', 'items', 'item_analysis'):
            original = candidate.get(field)
            if not isinstance(original, list) or len(original) != 6:
                continue
            if (not all(isinstance(row, dict) and isinstance(row.get('label'), str) for row in original)
                    or len({row['label'] for row in original}) != 6
                    or {row['label'] for row in original} != LABELS):
                continue
            if field in ('options', 'items'):
                if not all(set(row) == {'label', 'text'} and isinstance(row['text'], str)
                           and row['text'].strip() for row in original):
                    continue
                normalized = {row['label']: row['text'] for row in original}
                mapping = 'exact_unique_A_F_label_text_list_to_dictionary'
            else:
                normalized = {row['label']: {k: copy.deepcopy(v) for k, v in row.items() if k != 'label'}
                              for row in original}
                mapping = 'exact_unique_A_F_labelled_analysis_list_to_dictionary_strip_label_only'
            candidate[field] = normalized
            changes.append({'candidate_index': index, 'slot_id': candidate.get('slot_id'),
                'field': field, 'mapping': mapping, 'original_label_sequence': [row['label'] for row in original],
                'authored_original_value': copy.deepcopy(original), 'normalized_value': copy.deepcopy(normalized),
                'semantics_authored_by_manager': False})
    # Demonstrate losslessness rather than merely assuming equivalent schemas.
    reversed_value = copy.deepcopy(value)
    reversed_rows = q.extract_list(reversed_value, 'candidates', 'questions')
    for change in reversed(changes):
        row = reversed_rows[change['candidate_index']]
        normalized = row[change['field']]
        if change['field'] in ('options', 'items'):
            restored = [{'label': label, 'text': normalized[label]} for label in change['original_label_sequence']]
        else:
            restored = [{'label': label, **copy.deepcopy(normalized[label])}
                        for label in change['original_label_sequence']]
        if restored != change['authored_original_value']:
            raise RuntimeError('Serialization transformation is not exactly reversible')
        row[change['field']] = restored
    if reversed_value != raw:
        raise RuntimeError('A non-serialization authored field changed')
    return value, changes


def request_payload(text: str) -> dict[str, Any]:
    marker = next((m for m in ('AUTHORIZED_INPUTS:\n', 'FROZEN_INPUTS:\n', 'REVIEW_INPUT:\n') if m in text), None)
    if marker is None:
        raise RuntimeError('Question call has no embedded authorized input marker')
    tail = text.split(marker, 1)[1]
    obj, end = json.JSONDecoder().raw_decode(tail)
    if not isinstance(obj, dict) or tail[end:].strip():
        raise RuntimeError('Malformed saved question call payload')
    return obj


def stage_requests(root: str, arm: str) -> dict[str, list[dict[str, Any]]]:
    result = {name: [] for name in ('plan', 'author', 'gold', 'verify', 'revision')}
    for directory in sorted((BASE / 'cli_calls').glob('q20_*')):
        match = re.fullmatch(r'q20_(plan|author|gold|verify|revision)_([fg])_(\d+)(?:_.*)?', directory.name)
        if not match:
            continue
        stage, token, number = match.groups()
        if token.upper() != arm or q.root_id(number) != root:
            continue
        for path in sorted(directory.glob('attempt_*/REQUEST.txt')):
            obj = request_payload(path.read_text())
            embedded = obj.get('assignment', {})
            actual_root = embedded.get('root_id') if embedded else obj.get('root_id')
            if actual_root != root:
                raise RuntimeError('Call directory and actual embedded root disagree')
            if embedded and embedded.get('arm') != arm:
                raise RuntimeError('Call directory and actual embedded arm disagree')
            result[stage].append({'call_id': directory.name, 'request_binding': q.binding(path),
                                  'slot_ids': embedded.get('slots') if embedded else None})
    return result


def inspect_inputs(root: str, arm: str, source: dict[str, Any], input_work: Path,
                   selection: Path) -> dict[str, Any]:
    if hashlib.sha256(Path(v3.__file__).read_bytes()).hexdigest() != EXPECTED_V3_SHA256:
        raise RuntimeError('Frozen V3 helper binding drift')
    v3.build_driver()  # Verify frozen base-driver bytes before inspecting content.
    for row in v3.active_calls():
        live = q.read(row['path'])
        if re.match(rf'q20_\w+_{arm.lower()}_{q.root_number(root)}(?:_|$)', live.get('call_id', '')):
            raise RuntimeError('Input root/arm still has a live call')
    required = ('OUTCOME.json', 'CONTINUATION_FINALIZER_READY.json', 'TASK_PLANS.json',
        'TASK_PLANS_FREEZE.json', 'CANDIDATES_DRAFT_RESPONSE.json', 'CANDIDATES_DRAFT_CALL.json',
        'CANDIDATES_DRAFT.json', 'CANDIDATES_FINAL.json', 'CONTENT_REVIEW_FINAL.json', 'SLOT_FAILURES.json')
    if any(not (input_work / name).is_file() for name in required):
        raise RuntimeError('Saved draft stage lacks terminal frozen inputs required for reversible recovery')
    outcome = q.read(input_work / 'OUTCOME.json')
    if (outcome.get('root_id'), outcome.get('arm'), outcome.get('source_binding')) != (root, arm, source['binding']):
        raise RuntimeError('Input saved outcome identity/source drift')
    ready = q.read(input_work / 'CONTINUATION_FINALIZER_READY.json')
    for field in ('final_outcome_binding', 'final_candidates_binding', 'final_review_binding', 'slot_failures_binding'):
        q.verify_binding(ready[field])
    reviews = q.read(input_work / 'CONTENT_REVIEW_FINAL.json').get('reviews', [])
    if any(review.get('decision') == 'PASS' for review in reviews):
        raise RuntimeError('This bounded whole-draft adapter cannot rereview or alter previously PASS slots')
    requests = stage_requests(root, arm)
    if requests['gold'] or requests['verify'] or requests['revision']:
        raise RuntimeError('Initial independent review or substantive revision was already requested; cannot replay it')
    author_ids = {r['call_id'] for r in requests['author']}
    if len(author_ids) != 1:
        raise RuntimeError('Exactly one actual initial-author call is required')
    receipt = q.read(input_work / 'CANDIDATES_DRAFT_CALL.json')
    for key in ('request', 'parsed_response', 'raw_response', 'terminal_binding'):
        q.verify_binding(receipt[key])
    actual_result = BASE / 'cli_calls' / next(iter(author_ids)) / 'RESULT.json'
    if not actual_result.exists() or q.read(actual_result) != receipt:
        raise RuntimeError('Saved initial-author receipt is not its real complete call result')
    if receipt.get('response_received') is not True or not receipt.get('backend', {}).get('thread_id'):
        raise RuntimeError('Initial author has no actual completed fresh-thread receipt')
    backend = receipt['backend']
    if any(backend.get(k) != v for k, v in {'model_override': None, 'model': 'gpt-6.1-sol',
                                           'model_provider': 'hua', 'effort': 'ultra'}.items()):
        raise RuntimeError('Saved actual initial backend differs from default research backend')
    raw = q.read(input_work / 'CANDIDATES_DRAFT_RESPONSE.json')
    if raw != q.read(receipt['parsed_response']['path']):
        raise RuntimeError('Saved author raw object differs from the actual parsed response')
    plan = q.read(input_work / 'TASK_PLANS.json')
    freeze = q.read(input_work / 'TASK_PLANS_FREEZE.json')
    q.verify_binding(freeze['plans_binding'])
    if freeze['plans_binding'] != q.binding(input_work / 'TASK_PLANS.json'):
        raise RuntimeError('Saved plan freeze does not bind actual saved plans')
    # These roots must have one valid preauthoring plan per assigned slot.
    feedback, _, _, _ = q.feedback_material() if arm == 'F' else ({}, None, None, None)
    ids = {str(c.get('feedback_id')) for c in feedback.get('cards', []) if isinstance(c, dict)}
    _, plan_errors = q.validate_plans(plan, arm, root, source['source'], ids)
    if plan_errors:
        raise RuntimeError('Candidate serialization adapter cannot repair or replan invalid slots')
    original_candidates, _, _ = q.normalize_candidates(raw, arm, root, backend['thread_id'],
        source, plan, q.binding(selection))
    if original_candidates != q.read(input_work / 'CANDIDATES_DRAFT.json'):
        raise RuntimeError('Original normalized draft does not match its saved author raw/metadata lineage')
    normalized_raw, transformations = normalizer(raw)
    if not transformations:
        raise RuntimeError('No reversible candidate serialization defect is present')
    candidates, normalization_failures, _ = q.normalize_candidates(normalized_raw, arm, root,
        backend['thread_id'], source, plan, q.binding(selection))
    shapes = [{'slot_id': c.get('slot_id'), 'basic_solver_shape': q.basic_solver_shape(c, source),
               'mechanical_audit': q.audit_candidate(c, selection)} for c in candidates]
    if not all(row['basic_solver_shape'] for row in shapes):
        raise RuntimeError('Defect is not bounded to the authorized reversible serialization mappings')
    paths = [input_work / name for name in required]
    for name in ('CONTENT_REVIEW_NORMALIZED.json', 'DRAFT_NORMALIZATION.json'):
        if (input_work / name).exists(): paths.append(input_work / name)
    return {'schema_version': 'fgr20_candidate_serialization_recovery_inputs_v1',
        'root_id': root, 'arm': arm, 'source_binding': source['binding'], 'input_work_dir': str(input_work),
        'input_bindings': [q.binding(path) for path in paths],
        'actual_stage_requests_before_recovery': requests,
        'initial_author_call_id': next(iter(author_ids)), 'actual_initial_author_thread_id': backend['thread_id'],
        'actual_initial_author_receipt_binding': q.binding(input_work / 'CANDIDATES_DRAFT_CALL.json'),
        'actual_plan_path': str(input_work / 'TASK_PLANS.json'),
        'actual_plan_freeze_path': str(input_work / 'TASK_PLANS_FREEZE.json'),
        'initial_author_call_repeated': False, 'content_revision_consumed_by_serialization': False,
        'serialization_transformations': transformations, 'normalized_basic_shape_and_mechanical_audits': shapes,
        'normalization_failures': normalization_failures, 'semantic_admission_asserted': False}


def call_id(stage: str, arm: str, root: str, suffix: str = '') -> str:
    return q.call_id(stage, arm, root, '_'.join(x for x in ('candidate_shape_01', suffix) if x))


class ActualPlanPaths(ast.NodeTransformer):
    def __init__(self): self.changes = []
    def visit_BinOp(self, node):
        if (isinstance(node.op, ast.Div) and isinstance(node.left, ast.Name) and node.left.id == 'work'
                and isinstance(node.right, ast.Constant) and node.right.value in ('TASK_PLANS.json', 'TASK_PLANS_FREEZE.json')):
            self.changes.append(node.right.value)
            return ast.copy_location(ast.Call(func=ast.Name(id='_actual_plan_file', ctx=ast.Load()),
                args=[ast.Constant(value=node.right.value)], keywords=[]), node)
        return self.generic_visit(node)


def build_process(root: str, arm: str, source: dict[str, Any], input_work: Path,
                  out: Path, selection: Path, mocked_review: bool = False):
    _, base_changes = v3.build_driver()
    namespace = dict(vars(q))
    tree = ast.parse(inspect.getsource(q.process_root_arm))
    first = v3.Adapt('process_root_arm')
    tree = first.visit(tree)
    second = ActualPlanPaths()
    tree = ast.fix_missing_locations(second.visit(tree))
    # Independently clone only the already frozen two-stage review function.
    review_tree = ast.parse(inspect.getsource(q.review_batch))
    review_adapter = v3.Adapt('review_batch')
    review_tree = ast.fix_missing_locations(review_adapter.visit(review_tree))
    saved_receipt = q.read(input_work / 'CANDIDATES_DRAFT_CALL.json')
    saved_raw = q.read(input_work / 'CANDIDATES_DRAFT_RESPONSE.json')
    reference = {'schema_version': 'fgr20_saved_draft_receipt_reference_v1',
        'original_candidates_draft_response': q.binding(input_work / 'CANDIDATES_DRAFT_RESPONSE.json'),
        'original_candidates_draft_call_receipt': q.binding(input_work / 'CANDIDATES_DRAFT_CALL.json'),
        'original_candidates_draft': q.binding(input_work / 'CANDIDATES_DRAFT.json'),
        'original_content_review_normalized': q.binding(input_work / 'CONTENT_REVIEW_NORMALIZED.json')
            if (input_work / 'CONTENT_REVIEW_NORMALIZED.json').exists() else None,
        'original_draft_normalization': q.binding(input_work / 'DRAFT_NORMALIZATION.json')
            if (input_work / 'DRAFT_NORMALIZATION.json').exists() else None,
        'initial_author_call_repeated': False, 'content_revision_consumed': False}
    real_dispatch = v3.GuardedCalls()
    trace = {'reused_actual_initial_author': 0, 'new_plan_calls': 0, 'new_author_calls': 0,
             'mock_first_review_reached': False}
    class SavedDraftCalls:
        def __getattr__(self, key): return getattr(q.cc, key)
        def stop_generation(self): return False if mocked_review else q.cc.stop_generation()
        def progress(self, *args, **kwargs):
            if not mocked_review: return q.cc.progress(*args, **kwargs)
        def call(self, key, prompt, **kwargs):
            if key.startswith('q20_plan_'):
                raise RuntimeError('Saved candidate recovery must never call the planner')
            if key.startswith('q20_author_'):
                if key != call_id('author', arm, root):
                    raise RuntimeError('Unexpected attempt to obtain another initial author')
                if hashlib.sha256(prompt.encode()).hexdigest() != saved_receipt['request']['sha256']:
                    raise RuntimeError('Reconstructed original-author prompt differs from its actual saved request')
                trace['reused_actual_initial_author'] += 1
                return copy.deepcopy(saved_raw), copy.deepcopy(saved_receipt)
            if mocked_review:
                raise RuntimeError('Mock adapter must never dispatch a semantic call')
            return real_dispatch.call(key, prompt, **kwargs)
    def publish(path, value):
        path = Path(path)
        if path == out / 'CANDIDATES_DRAFT_CALL.json':
            if value != saved_receipt: raise RuntimeError('Reused original author receipt drift')
            return q.publish(out / 'ORIGINAL_DRAFT_RECEIPT_REFERENCE.json', reference)
        return q.publish(path, value)
    def normalize(response, *args, **kwargs):
        revised = kwargs.get('revision_thread') or (args[6] if len(args) > 6 else None)
        normalized, changes = normalizer(response)
        stage = 'REVISION_01' if revised else 'INITIAL'
        q.publish(out / f'MECHANICAL_CANDIDATE_NORMALIZATION_{stage}.json', {
            'schema_version': 'fgr20_reversible_candidate_serialization_normalization_v1',
            'root_id': root, 'arm': arm, 'stage': stage,
            'authored_response_sha256': q.digest(response), 'normalized_response_sha256': q.digest(normalized),
            'changes': changes, 'exact_round_trip_verified': True,
            'question_text_gold_evidence_and_semantic_fields_changed': False,
            'no_extra_semantic_budget_consumed': True})
        return q.normalize_candidates(normalized, *args, **kwargs)
    def receipt_lookup(work):
        freeze = q.read(input_work / 'TASK_PLANS_FREEZE.json')
        receipt_binding = freeze.get('call_receipt')
        if not isinstance(receipt_binding, dict):
            reference_path = input_work / 'ORIGINAL_PLAN_RECEIPT_REFERENCE.json'
            receipt_binding = q.read(reference_path)['original_plan_call_receipt']
        q.verify_binding(receipt_binding)
        return q.read(receipt_binding['path'])
    def forbidden_planning(*args, **kwargs):
        raise RuntimeError('Saved-draft recovery may not invoke planning/plan normalization')
    namespace.update(work_dir=lambda rid, at: out, _saved_plan_receipt_lookup=receipt_lookup,
        _actual_plan_file=lambda name: input_work / name, continuation_call_id=call_id,
        continuation_task_plan_prompt=forbidden_planning, continuation_validate_plans=forbidden_planning,
        cc=SavedDraftCalls(), publish=publish, normalize_candidates=normalize)
    exec(compile(review_tree, '<candidate_shape_saved_review_clone>', 'exec'), namespace)
    namespace['continuation_review_batch'] = namespace['review_batch']
    if mocked_review:
        def stop_at_first_review(*args, **kwargs):
            trace['mock_first_review_reached'] = True
            raise RuntimeError('MOCK_FIRST_REVIEW_REACHED_WITHOUT_NEW_AUTHOR_OR_PLAN')
        namespace['continuation_review_batch'] = stop_at_first_review
    exec(compile(tree, '<candidate_shape_saved_process_clone>', 'exec'), namespace)
    changes = {'frozen_process_ast_adapter': first.changed,
               'actual_original_plan_path_expressions': second.changes,
               'review_adapter': review_adapter.changed,
               'adapted_process_ast_sha256': hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
               'prior_base_transformations': base_changes}
    return namespace['process_root_arm'], changes, trace


def publish_ready(root: str, arm: str, input_work: Path, out: Path, outcome: dict[str, Any]):
    for name, default in (('CANDIDATES_FINAL.json', []),
        ('CONTENT_REVIEW_FINAL.json', {'reviews': [], 'mechanical_audits': [], 'all_reviews_ai_family_not_human': True}),
        ('SLOT_FAILURES.json', outcome.get('slot_failures', []))):
        if not (out / name).exists(): q.publish(out / name, default)
    integrated = dict(outcome, final_candidates_binding=q.binding(out / 'CANDIDATES_FINAL.json'),
        review_binding=q.binding(out / 'CONTENT_REVIEW_FINAL.json'),
        slot_failures_binding=q.binding(out / 'SLOT_FAILURES.json'))
    q.publish(out / 'CONTINUATION_INTEGRATION_OUTCOME.json', integrated)
    row = {'schema_version': 'fgr20_question_continuation_finalizer_ready_v1',
        'root_id': root, 'arm': arm, 'continuation_kind': KIND,
        'original_work_dir': str(input_work),
        'main_original_work_dir': str(v3.original_dir(root, arm)), 'final_work_dir': str(out),
        'actual_plan_path': str(input_work / 'TASK_PLANS.json'),
        'actual_plan_freeze_path': str(input_work / 'TASK_PLANS_FREEZE.json'),
        'original_outcome_binding': q.binding(input_work / 'OUTCOME.json'),
        'supersedes_ready_binding': q.binding(input_work / 'CONTINUATION_FINALIZER_READY.json'),
        'final_outcome_binding': q.binding(out / 'CONTINUATION_INTEGRATION_OUTCOME.json'),
        'driver_outcome_binding': q.binding(out / 'OUTCOME.json'),
        'final_candidates_binding': q.binding(out / 'CANDIDATES_FINAL.json'),
        'final_review_binding': q.binding(out / 'CONTENT_REVIEW_FINAL.json'),
        'slot_failures_binding': q.binding(out / 'SLOT_FAILURES.json'),
        'original_draft_receipt_reference_binding': q.binding(out / 'ORIGINAL_DRAFT_RECEIPT_REFERENCE.json'),
        'normalization_binding': q.binding(out / 'MECHANICAL_CANDIDATE_NORMALIZATION_INITIAL.json'),
        'initial_author_call_repeated': False, 'mechanical_normalization_consumed_content_revision': False,
        'independent_fresh_semantic_review_required': True}
    q.publish(out / 'CONTINUATION_FINALIZER_READY.json', row)
    return row


def prepare(root, arm, source, input_work, selection):
    out = work_dir(root, arm)
    out.mkdir(parents=True, exist_ok=True)
    frozen_path = out / 'RECOVERY_INPUT_FREEZE.json'
    if frozen_path.exists():
        frozen = q.read(frozen_path)
        if frozen['input_work_dir'] != str(input_work) or frozen['source_binding'] != source['binding']:
            raise RuntimeError('Saved candidate recovery input identity drift')
        for value in frozen['input_bindings']: q.verify_binding(value)
        now = stage_requests(root, arm)
        for stage in ('gold', 'verify', 'revision'):
            allowed = {r['call_id'] for r in frozen['actual_stage_requests_before_recovery'][stage]}
            allowed.add(call_id(stage, arm, root, 'initial' if stage in ('gold', 'verify') else '01'))
            if stage in ('gold', 'verify'): allowed.add(call_id(stage, arm, root, 'revision_01'))
            if any(record['call_id'] not in allowed for record in now[stage]):
                raise RuntimeError('Additional review/revision requested outside this frozen continuation')
        return frozen
    frozen = inspect_inputs(root, arm, source, input_work, selection)
    q.publish(frozen_path, frozen)
    return frozen


def run(root, arm, source, input_work, selection, do_run):
    out = work_dir(root, arm)
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.continuation.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (out / 'CONTINUATION_FINALIZER_READY.json').exists():
            ready = q.read(out / 'CONTINUATION_FINALIZER_READY.json')
            if (ready.get('root_id'), ready.get('arm'), ready.get('continuation_kind'),
                    ready.get('original_work_dir'), ready.get('final_work_dir')) != (
                    root, arm, KIND, str(input_work), str(out)):
                raise RuntimeError('Cached candidate recovery identity drift')
            for key, value in ready.items():
                if key.endswith('_binding'):
                    q.verify_binding(value)
            q.verify_binding(q.read(out / 'DRIVER_ADAPTATION.json')['wrapper_binding'])
            return ready
        if do_run and (BASE / 'CONTENT_FREEZE.json').exists():
            raise RuntimeError('Content is already frozen; no fresh semantic review/revision may start')
        frozen = prepare(root, arm, source, input_work, selection)
        process, transformations, _ = build_process(root, arm, source, input_work, out, selection)
        q.publish(out / 'DRIVER_ADAPTATION.json', {'schema_version': 'fgr20_saved_candidate_shape_driver_adaptation_v1',
            'root_id': root, 'arm': arm, 'wrapper_binding': q.binding(__file__),
            'v3_helper_binding': q.binding(v3.__file__), 'frozen_driver_binding': q.binding(q.__file__),
            'transformations': transformations, 'input_freeze_binding': q.binding(out / 'RECOVERY_INPUT_FREEZE.json'),
            'initial_author_and_planner_calls_repeated': False, 'original_semantics_unchanged': True})
        if not do_run: return {'root_id': root, 'arm': arm, 'preparation_binding': q.binding(out / 'RECOVERY_INPUT_FREEZE.json')}
        if q.cc.stop_generation(): raise RuntimeError('Generation cutoff before fresh content review')
        contract_path = BASE / 'question_authoring/COMMON_AUTHOR_CONTRACT_ZH.md'
        feedback, adapter, fb, ab = q.feedback_material() if arm == 'F' else (None, None, None, None)
        outcome = process(root, arm, source, selection, contract_path.read_text(), q.binding(contract_path), feedback, adapter, fb, ab)
        return publish_ready(root, arm, input_work, out, outcome)


def self_check():
    root, arm = 'cc-train-episode-179186', 'F'
    input_work = BASE / 'question_plan_continuation/179186/F'
    selection = BASE / 'SOURCE_SELECTION_FINAL.json'
    q.SOURCE_FREEZE_PATH = BASE / 'SOURCE_FREEZE.json'
    _, sources = q.load_selection(selection)
    source = sources[root]
    frozen = inspect_inputs(root, arm, source, input_work, selection)
    assert len(frozen['serialization_transformations']) == 16
    assert all(r['basic_solver_shape'] and r['mechanical_audit']['mechanical_pass'] for r in frozen['normalized_basic_shape_and_mechanical_audits'])
    original = q.read(input_work / 'CANDIDATES_DRAFT_RESPONSE.json')
    normalized, transformations = normalizer(original)
    assert original == q.read(input_work / 'CANDIDATES_DRAFT_RESPONSE.json')
    # Invalid/duplicate labels or additional option fields must not be repaired.
    for defect in ('duplicate', 'extra'):
        bad = copy.deepcopy(original)
        choices = bad['candidates'][0]['options']
        if defect == 'duplicate': choices[0]['label'] = choices[1]['label']
        else: choices[0]['unapproved_extra'] = True
        changed, _ = normalizer(bad)
        assert changed['candidates'][0]['options'] == choices
    fixture = BASE / 'runtime_cache/engineering_candidate_shape_checks'
    fixture.mkdir(parents=True, exist_ok=True)
    before = sorted(str(path) for path in (BASE / 'cli_calls').glob('q20_*/attempt_*/REQUEST.txt'))
    with tempfile.TemporaryDirectory(prefix='saved_draft_', dir=fixture) as directory:
        out = Path(directory)
        process, changes, trace = build_process(root, arm, source, input_work, out, selection, True)
        cp = BASE / 'question_authoring/COMMON_AUTHOR_CONTRACT_ZH.md'
        feedback, adapter, fb, ab = q.feedback_material()
        outcome = process(root, arm, source, selection, cp.read_text(), q.binding(cp), feedback, adapter, fb, ab)
        assert outcome['status'] == 'FAILED_STAGE' and 'MOCK_FIRST_REVIEW_REACHED' in outcome.get('error', ''), outcome
        assert trace['reused_actual_initial_author'] == 1 and trace['mock_first_review_reached']
        assert not (out / 'CANDIDATES_DRAFT_CALL.json').exists()
        assert (out / 'ORIGINAL_DRAFT_RECEIPT_REFERENCE.json').exists()
        assert len(q.read(out / 'CANDIDATES_DRAFT.json')) == 8
    after = sorted(str(path) for path in (BASE / 'cli_calls').glob('q20_*/attempt_*/REQUEST.txt'))
    # Other live roots may legitimately begin calls during a concurrent self-check;
    # the tested root itself may never acquire any new request.
    assert [p for p in before if '_f_179186_' in p] == [p for p in after if '_f_179186_' in p]
    return {'status': 'PASS', 'semantic_calls_started': 0, 'real_saved_candidates_restored': 8,
        'reversible_serialization_field_changes': len(transformations), 'real_mechanical_gate_pass_count': 8,
        'saved_author_prompt_exactly_equal_original_request': True,
        'mock_reached_first_independent_review_without_new_author_or_planner': True,
        'no_new_standard_author_call_receipt_written': True, 'content_semantic_verdict': None,
        'transformations': changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inspect', 'prepare', 'run', 'self-check'])
    parser.add_argument('--root')
    parser.add_argument('--arm', choices=q.ARMS)
    parser.add_argument('--input-work')
    parser.add_argument('--selection', default=str(BASE / 'SOURCE_SELECTION_FINAL.json'))
    parser.add_argument('--source-freeze', default=str(BASE / 'SOURCE_FREEZE.json'))
    args = parser.parse_args()
    if args.command == 'self-check':
        print(json.dumps(self_check(), ensure_ascii=False)); return 0
    if not args.root or not args.arm or not args.input_work:
        parser.error('Explicit root, arm and input-work are required')
    root = q.root_id(args.root)
    selection = check_local(args.selection)
    q.SOURCE_FREEZE_PATH = check_local(args.source_freeze)
    _, sources = q.load_selection(selection)
    source = sources[root]
    input_work = check_local(args.input_work)
    result = inspect_inputs(root, args.arm, source, input_work, selection) if args.command == 'inspect' else run(root, args.arm, source, input_work, selection, args.command == 'run')
    print(json.dumps(result, ensure_ascii=False)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
