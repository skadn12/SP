#!/usr/bin/env python3
"""One unused content revision for missing or invalid initial draft slots.

Plan and initial author calls are never repeated. A real revision REQUEST
consumes its listed slots irrespective of JSON validity, completion or quality.
This coordinator creates no dialogue, question, answer key or semantic PASS.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import question_driver as q
import question_plan_schema_continuation as qs

SCHEMA_WRAPPER_SHA256 = 'ae7c583413379105e3c69128810893d1e53979bba04b5b41c0b2021146a6ac9e'
CONTINUATIONS = BASE / 'question_unfilled_continuation'


def payload(prompt: str) -> dict[str, Any]:
    marker = next((m for m in ('AUTHORIZED_INPUTS:\n', 'FROZEN_INPUTS:\n', 'REVIEW_INPUT:\n') if m in prompt), None)
    if marker is None:
        raise RuntimeError('Question call request has no recognized embedded input marker')
    tail = prompt.split(marker, 1)[1]
    obj, end = json.JSONDecoder().raw_decode(tail)
    if not isinstance(obj, dict) or tail[end:].strip():
        raise RuntimeError('Question call request is not one complete embedded JSON object')
    return obj


def assignment(prompt: str) -> dict[str, Any]:
    value = payload(prompt).get('assignment')
    if not isinstance(value, dict) or set(value) != {'root_id', 'arm', 'slots'}:
        raise RuntimeError('Author/revision request has malformed assignment')
    if value.get('arm') not in q.ARMS or not isinstance(value.get('slots'), list):
        raise RuntimeError('Author/revision request has malformed arm/slots')
    slots = value['slots']
    if len(set(slots)) != len(slots) or any(slot not in q.SLOTS for slot in slots):
        raise RuntimeError('Author/revision request has invalid or duplicate slot ids')
    return value


def actual_request_budget(root: str, arm: str) -> dict[str, Any]:
    authors: dict[str, list[dict[str, Any]]] = {}
    revisions: list[dict[str, Any]] = []
    used = set()
    for directory in sorted((BASE / 'cli_calls').glob('q20_*')):
        if directory.name.startswith('q20_author_'):
            kind = 'initial_author'
        elif directory.name.startswith('q20_revision_'):
            kind = 'revision'
        else:
            continue
        for path in sorted(directory.glob('attempt_*/REQUEST.txt')):
            request_assignment = assignment(path.read_text())
            if request_assignment['root_id'] != root or request_assignment['arm'] != arm:
                continue
            record = {'call_id': directory.name, 'request_binding': q.binding(path),
                      'requested_slot_ids': request_assignment['slots'],
                      'request_consumes_semantic_budget': True}
            if kind == 'initial_author':
                authors.setdefault(directory.name, []).append(record)
            else:
                revisions.append(record)
                used.update(request_assignment['slots'])
    return {'root_id': root, 'arm': arm, 'initial_author_calls': authors,
            'revision_requests': revisions, 'revision_requested_slots': sorted(used)}


def select_input(root: str, arm: str, supplied: str | None) -> Path:
    if supplied:
        path = Path(supplied).resolve()
        path.relative_to(BASE.resolve())
        return path
    normalized = qs.work_dir(root, arm)
    if (normalized / 'CONTINUATION_FINALIZER_READY.json').exists():
        return normalized
    return qs.original_dir(root, arm)


def actual_initial_author(budget: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    calls = budget['initial_author_calls']
    if len(calls) != 1:
        raise RuntimeError(f'Exactly one actual initial author call required; found {len(calls)}')
    key = next(iter(calls))
    directory = BASE / 'cli_calls' / key
    result = directory / 'RESULT.json'
    if result.exists():
        receipt = q.read(result)
        q.verify_binding(receipt['request'])
        q.verify_binding(receipt['terminal_binding'])
        terminal_path = Path(receipt['terminal_binding']['path'])
    else:
        terminals = sorted(directory.glob('attempt_*/TERMINAL.json'))
        if not terminals:
            raise RuntimeError('Interrupted initial START lacks terminal classification; fail closed')
        terminal_path = terminals[-1]
        receipt = q.read(terminal_path)
        q.verify_binding(receipt['request'])
        # The complete real terminal retains actual backend even for non-JSON output.
        q.classify_cli_events(Path(receipt['events']['path']), receipt)
    backend = receipt.get('backend', {})
    if not backend.get('thread_id') or backend.get('model_override') is not None:
        raise RuntimeError('Initial author lacks actual default-model fresh thread lineage')
    if receipt.get('response_received') is not True and not receipt.get('raw_response'):
        raise RuntimeError('Initial author has no actual semantic response; interrupted classification required')
    for field in ('raw_response', 'events'):
        if isinstance(receipt.get(field), dict):
            q.verify_binding(receipt[field])
    return backend['thread_id'], {'initial_author_call_id': key,
        'actual_initial_terminal_binding': q.binding(terminal_path),
        'initial_author_backend': backend,
        'raw_response_binding': receipt.get('raw_response'),
        'initial_author_response_received': receipt.get('response_received'),
        'initial_author_semantic_attempt_not_repeated': True}


def inspect_budget(root: str, arm: str, source: dict[str, Any], work: Path) -> dict[str, Any]:
    if not (work / 'OUTCOME.json').exists():
        raise RuntimeError('Input root/arm outcome is not terminal')
    old_outcome = q.read(work / 'OUTCOME.json')
    if old_outcome.get('root_id') != root or old_outcome.get('arm') != arm or old_outcome.get('source_binding') != source['binding']:
        raise RuntimeError('Input outcome identity/source binding drift')
    for live in qs.active_calls():
        data = q.read(live['path'])
        key = str(data.get('call_id', ''))
        if key.startswith('q20_') and f'_{arm.lower()}_{q.root_number(root)}' in key:
            raise RuntimeError(f'Root/arm still has an active semantic call: {key}')
    plan_path = work / 'TASK_PLANS.json'
    freeze_path = work / 'TASK_PLANS_FREEZE.json'
    frozen = q.read(freeze_path)
    q.verify_binding(frozen['plans_binding'])
    if isinstance(frozen.get('call_receipt'), dict):
        q.verify_binding(frozen['call_receipt'])
    if frozen['plans_binding'] != q.binding(plan_path):
        raise RuntimeError('Actual plan freeze does not bind actual plan')
    plans = q.read(plan_path)
    feedback, _, _, _ = q.feedback_material() if arm == 'F' else ({}, None, None, None)
    ids = {str(c.get('feedback_id')) for c in feedback.get('cards', []) if isinstance(c, dict)}
    strict, errors = q.validate_plans(plans, arm, root, source['source'], ids)
    bad = {err.rsplit(':', 1)[-1] for err in errors if err.rsplit(':', 1)[-1] in q.SLOTS}
    valid_plans = {row['slot_id']: row for row in strict['plans'] if row['slot_id'] not in bad}
    draft_path = work / 'CANDIDATES_DRAFT.json'
    drafts = q.read(draft_path) if draft_path.exists() else []
    if not isinstance(drafts, list):
        raise RuntimeError('Saved initial draft is not an actual candidate list')
    draft_by_slot = {c.get('slot_id'): c for c in drafts if isinstance(c, dict)}
    budget = actual_request_budget(root, arm)
    author_thread, lineage = actual_initial_author(budget)
    author_records = next(iter(budget['initial_author_calls'].values()))
    authored_slots = set(author_records[0]['requested_slot_ids'])
    if any(set(rec['requested_slot_ids']) != authored_slots for rec in author_records):
        raise RuntimeError('Transport attempts changed initial author assignment')
    already_used = set(budget['revision_requested_slots'])
    eligible = []
    dispositions = []
    for slot in q.SLOTS:
        initial = draft_by_slot.get(slot)
        if slot not in valid_plans:
            why = 'invalid_or_missing_frozen_plan_no_replan'
        elif slot not in authored_slots:
            why = 'initial_author_did_not_receive_slot_no_reassignment'
        elif slot in already_used:
            why = 'single_content_revision_already_requested'
        elif initial is not None and q.basic_solver_shape(initial, source):
            why = 'initial_candidate_basic_shape_usable_not_unfilled'
        else:
            why = 'eligible_missing_or_basic_shape_invalid_initial_draft'
            eligible.append(slot)
        dispositions.append({'slot_id': slot, 'disposition': why,
                             'initial_candidate_present': initial is not None})
    if not (work / 'CANDIDATES_FINAL.json').exists() and any(
            q.basic_solver_shape(c, source) for c in drafts):
        raise RuntimeError('Untouched usable initial drafts lack final surfaces; outside unfilled-slot recovery scope')
    evidence_paths = [work / name for name in ('OUTCOME.json', 'TASK_PLANS.json', 'TASK_PLANS_FREEZE.json',
        'CANDIDATES_DRAFT_RESPONSE.json', 'CANDIDATES_DRAFT.json', 'DRAFT_NORMALIZATION.json',
        'CANDIDATES_FINAL.json', 'CONTENT_REVIEW_FINAL.json', 'SLOT_FAILURES.json') if (work / name).exists()]
    for path in evidence_paths:
        if path.name == 'CANDIDATES_FINAL.json':
            finals = q.read(path)
            for c in finals:
                if c.get('slot_id') in eligible and c != draft_by_slot.get(c.get('slot_id')):
                    raise RuntimeError('Eligible slot already changed after its initial draft')
    return {**budget, 'schema_version': 'fgr20_unfilled_slot_revision_budget_v1',
        'source_binding': source['binding'], 'original_work_dir': str(work),
        'actual_plan_path': str(plan_path), 'actual_plan_freeze_path': str(freeze_path),
        'eligible_slot_ids': eligible, 'slot_dispositions': dispositions,
        'strict_frozen_plan_errors': errors, 'initial_author_thread_id': author_thread,
        'initial_author_lineage': lineage, 'input_bindings': [q.binding(p) for p in evidence_paths],
        'one_revision_per_slot_total': True, 'plan_calls_repeated': 0,
        'initial_author_calls_repeated': 0}


def unused_call_id(stage: str, arm: str, root: str, suffix: str = '') -> str:
    return q.call_id(stage, arm, root, '_'.join(x for x in ('unused_content_revision_01', suffix) if x))


def revision_prompt(root: str, arm: str, source: dict[str, Any], plans: dict[str, Any],
                    eligible: list[str], budget: dict[str, Any], work: Path) -> str:
    contract = (BASE / 'question_authoring/COMMON_AUTHOR_CONTRACT_ZH.md').read_text()
    selected = {**plans, 'plans': [p for p in plans['plans'] if p['slot_id'] in eligible]}
    feedback, adapter, _, _ = q.feedback_material() if arm == 'F' else (None, None, None, None)
    ordinary = q.author_prompt(arm, root, q.source_view(source), contract, selected, feedback, adapter)
    base, _ = ordinary.split('AUTHORIZED_INPUTS:\n', 1)
    embedded = payload(ordinary)
    # These are engineering failure records, not fabricated question placeholders.
    failures = q.read(work / 'SLOT_FAILURES.json') if (work / 'SLOT_FAILURES.json').exists() else []
    initial_path = work / 'CANDIDATES_DRAFT.json'
    initial = q.read(initial_path) if initial_path.exists() else []
    embedded['revision_cases'] = [{'slot_id': slot,
        'actual_initial_candidate': next((c for c in initial if c.get('slot_id') == slot), None),
        'saved_failure_records': [f for f in failures if f.get('slot_id') == slot],
        'budget_disposition': next(d for d in budget['slot_dispositions'] if d['slot_id'] == slot)}
        for slot in eligible]
    rules = ('This is the single permitted content revision for these slots. The actual initial '
        'author request was consumed; its response was missing, non-JSON or mechanically invalid. '
        'Do not repeat planning or invent a replacement plan. Use each supplied exact frozen plan '
        'and the complete authorized source. Return replacements only for assignment.slots; '
        'do not add or modify any other slot. A null actual_initial_candidate means none was '
        'parseable, not an authored placeholder. All source-support, Gold and R03 checks remain '
        'subject to fresh independent blind reconstruction and a separate content verifier.\n')
    prompt = base + rules + 'AUTHORIZED_INPUTS:\n' + q.prompt_json(embedded)
    if arm == 'G':
        q.validate_g_prompt(prompt, 'candidate_reviser', root)
    return prompt


def run_one(root: str, arm: str, source: dict[str, Any], selection: Path, input_work: Path):
    out = CONTINUATIONS / q.root_number(root) / arm
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.continuation.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (out / 'CONTINUATION_FINALIZER_READY.json').exists():
            ready = q.read(out / 'CONTINUATION_FINALIZER_READY.json')
            for key in ('final_outcome_binding', 'final_candidates_binding', 'final_review_binding', 'slot_failures_binding'):
                q.verify_binding(ready[key])
            return ready
        if hashlib.sha256(Path(qs.__file__).read_bytes()).hexdigest() != SCHEMA_WRAPPER_SHA256:
            raise RuntimeError('Frozen schema-wrapper binding drift')
        audit_path = out / 'REVISION_BUDGET_FREEZE.json'
        if audit_path.exists():
            audit = q.read(audit_path)
            for b in audit['input_bindings']:
                q.verify_binding(b)
            if audit['source_binding'] != source['binding'] or audit['original_work_dir'] != str(input_work):
                raise RuntimeError('Frozen unused-revision inputs drift')
            now = actual_request_budget(root, arm)
            allowed_calls = {r['call_id'] for r in audit['revision_requests']} | {unused_call_id('revision', arm, root)}
            if any(r['call_id'] not in allowed_calls for r in now['revision_requests']):
                raise RuntimeError('An additional revision was requested after this budget freeze')
        else:
            audit = inspect_budget(root, arm, source, input_work)
            if not audit['eligible_slot_ids']:
                raise RuntimeError('No untouched missing/invalid initial-draft slots remain; no call authorized')
            q.publish(audit_path, audit)
        eligible = audit['eligible_slot_ids']
        plans = q.read(audit['actual_plan_path'])
        all_candidates = q.read(input_work / 'CANDIDATES_FINAL.json') if (input_work / 'CANDIDATES_FINAL.json').exists() else []
        if not all_candidates and (input_work / 'CANDIDATES_DRAFT.json').exists():
            all_candidates = q.read(input_work / 'CANDIDATES_DRAFT.json')
        review_obj = q.read(input_work / 'CONTENT_REVIEW_FINAL.json') if (input_work / 'CONTENT_REVIEW_FINAL.json').exists() else {'reviews': []}
        old_reviews = {r['candidate_id']: r for r in review_obj.get('reviews', [])}
        current = {c['slot_id']: c for c in all_candidates}
        reviews = dict(old_reviews)
        failures = q.read(input_work / 'SLOT_FAILURES.json') if (input_work / 'SLOT_FAILURES.json').exists() else []
        outcome = {'schema_version': 'fgr20_question_root_arm_v1', 'root_id': root, 'arm': arm,
            'source_binding': source['binding'], 'status': 'PARTIAL', 'calls': [],
            'candidate_ids': [], 'slot_failures': failures, 'max_content_revisions': 1,
            'continuation_kind': 'single_unused_content_revision', 'eligible_slot_ids': eligible,
            'initial_author_lineage': audit['initial_author_lineage'], 'budget_freeze_binding': q.binding(audit_path)}
        process, transformations = qs.build_driver()
        namespace = process.__globals__
        namespace['continuation_call_id'] = unused_call_id
        reviewer = namespace['continuation_review_batch']
        calls = namespace['cc']
        q.publish(out / 'DRIVER_ADAPTATION.json', {'frozen_driver_binding': q.binding(q.__file__),
            'schema_wrapper_binding': q.binding(qs.__file__), 'wrapper_binding': q.binding(__file__),
            'review_ast_transformations': transformations[0], 'input_work_dir': str(input_work),
            'no_plan_or_initial_author_regeneration': True, 'only_eligible_slots_may_change': eligible})
        try:
            if q.cc.stop_generation():
                raise RuntimeError('Generation cutoff before unused revision')
            prompt = revision_prompt(root, arm, source, plans, eligible, audit, input_work)
            raw, receipt = calls.call(unused_call_id('revision', arm, root), prompt)
            if receipt['backend']['thread_id'] == audit['initial_author_thread_id']:
                raise RuntimeError('Reviser reused the original author thread')
            q.publish(out / 'CANDIDATES_REVISION_01_RESPONSE.json', raw)
            q.publish(out / 'CANDIDATES_REVISION_01_CALL.json', receipt)
            outcome['calls'].append(receipt)
            selected_plans = {**plans, 'plans': [p for p in plans['plans'] if p['slot_id'] in eligible]}
            revisions, bad, normalization = q.normalize_candidates(raw, arm, root,
                audit['initial_author_thread_id'], source, selected_plans,
                q.binding(selection), receipt['backend']['thread_id'])
            q.publish(out / 'CANDIDATES_REVISION_01.json', revisions)
            q.publish(out / 'REVISION_NORMALIZATION.json', {'changes': normalization,
                'revision_author_thread_id': receipt['backend']['thread_id'],
                'candidate_lineage_author_thread_id': audit['initial_author_thread_id']})
            for item in bad:
                if item.get('slot_id') in eligible or item.get('slot_id') is None:
                    failures.append(q.slot_failure(item.get('slot_id') or 'UNASSIGNED', item['reason'], root, arm, 'unused_revision'))
            changed = [c for c in revisions if c['slot_id'] in eligible and q.basic_solver_shape(c, source)]
            for slot in eligible:
                replacement = next((c for c in changed if c['slot_id'] == slot), None)
                if replacement is not None:
                    current[slot] = replacement
                else:
                    failures.append(q.slot_failure(slot, 'single unused revision missing or basic-shape invalid', root, arm, 'unused_revision'))
            if changed:
                _, new_reviews, review_calls, _ = reviewer(root, changed, source, selection, out, 'unused_revision_01')
                reviews.update(new_reviews)
                outcome['calls'].extend(c for c in review_calls.values() if c)
        except Exception as exc:
            q.publish(out / 'UNUSED_REVISION_FAILURE.json', {'error': f'{type(exc).__name__}: {exc}',
                'single_revision_call_id': unused_call_id('revision', arm, root),
                'no_semantic_retry_or_second_revision': True})
            for slot in eligible:
                failures.append(q.slot_failure(slot, f'unused revision incomplete: {type(exc).__name__}: {exc}', root, arm, 'unused_revision'))
        final = [current[slot] for slot in q.SLOTS if slot in current]
        final_reviews, audits = [], []
        for c in final:
            cid = c['candidate_id']
            review = reviews.get(cid)
            if review is None:
                review = q.normalized_review(c, None, None, None, source['source'])
            # Never mutate a saved PASS or any unchanged review. Missing reviewer
            # information can only create HOLD through the frozen normalizer.
            if c['slot_id'] not in eligible and cid in old_reviews:
                if review != old_reviews[cid]:
                    raise RuntimeError('Untouched saved review changed')
            final_reviews.append(review)
            audits.append(q.audit_candidate(c, selection))
            if c['slot_id'] in eligible and (review.get('decision') != 'PASS' or not audits[-1].get('mechanical_pass')):
                failures.append(q.slot_failure(c['slot_id'], 'unused revision not admitted by independent review/mechanical gate', root, arm, 'content_review'))
        q.publish(out / 'CANDIDATES_FINAL.json', final)
        q.publish(out / 'CONTENT_REVIEW_FINAL.json', {'reviews': final_reviews, 'mechanical_audits': audits,
            'all_reviews_ai_family_not_human': True})
        q.publish(out / 'SLOT_FAILURES.json', failures)
        outcome.update(status='COMPLETED_WITH_HOLES' if failures or len(final) != 8 or any(r.get('decision') != 'PASS' for r in final_reviews) else 'COMPLETED',
            candidate_ids=[c['candidate_id'] for c in final], slot_failures=failures,
            candidate_bindings=[{'candidate_id': c['candidate_id'], 'candidate_sha256': q.gate.fingerprint(c)} for c in final],
            final_candidates_binding=q.binding(out / 'CANDIDATES_FINAL.json'),
            review_binding=q.binding(out / 'CONTENT_REVIEW_FINAL.json'),
            slot_failures_binding=q.binding(out / 'SLOT_FAILURES.json'),
            plan_freeze_binding=q.binding(audit['actual_plan_freeze_path']))
        q.publish(out / 'OUTCOME.json', outcome)
        prior_ready = input_work / 'CONTINUATION_FINALIZER_READY.json'
        ready = {'schema_version': 'fgr20_question_continuation_finalizer_ready_v1',
            'root_id': root, 'arm': arm, 'continuation_kind': 'single_unused_content_revision',
            'original_work_dir': str(input_work), 'main_original_work_dir': str(qs.original_dir(root, arm)),
            'final_work_dir': str(out), 'actual_plan_path': audit['actual_plan_path'],
            'actual_plan_freeze_path': audit['actual_plan_freeze_path'],
            'original_outcome_binding': q.binding(input_work / 'OUTCOME.json'),
            'final_outcome_binding': q.binding(out / 'OUTCOME.json'),
            'final_candidates_binding': q.binding(out / 'CANDIDATES_FINAL.json'),
            'final_review_binding': q.binding(out / 'CONTENT_REVIEW_FINAL.json'),
            'slot_failures_binding': q.binding(out / 'SLOT_FAILURES.json'),
            'eligible_slot_ids': eligible, 'budget_freeze_binding': q.binding(audit_path),
            'supersedes_ready_binding': q.binding(prior_ready) if prior_ready.exists() else None,
            'final_candidates_are_full_merged_root_arm_collection': True}
        q.publish(out / 'CONTINUATION_FINALIZER_READY.json', ready)
        return ready


def self_check():
    assert hashlib.sha256(Path(qs.__file__).read_bytes()).hexdigest() == SCHEMA_WRAPPER_SHA256
    _, transformations = qs.build_driver()
    root = 'cc-train-episode-55788'
    p = 'Exact test metadata only.\nAUTHORIZED_INPUTS:\n' + q.prompt_json({'assignment': {'root_id': root, 'arm': 'G', 'slots': ['MC2', 'ORDER2']}})
    assert assignment(p)['slots'] == ['MC2', 'ORDER2']
    try:
        assignment(p + ' trailing invalid')
        raise AssertionError('Trailing request content accepted')
    except RuntimeError:
        pass
    for bad_slots in (['MC2', 'MC2'], ['UNKNOWN']):
        try:
            assignment('AUTHORIZED_INPUTS:\n' + q.prompt_json({'assignment': {'root_id': root, 'arm': 'G', 'slots': bad_slots}}))
            raise AssertionError('Invalid slots accepted')
        except RuntimeError:
            pass
    assert unused_call_id('revision', 'F', root) == 'q20_revision_f_55788_unused_content_revision_01'
    return {'status': 'PASS', 'semantic_calls': 0, 'budget_rule': 'actual_revision_REQUEST_assignment_slots',
            'schema_wrapper_sha256': SCHEMA_WRAPPER_SHA256, 'review_ast': transformations[0]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inspect', 'run', 'self-check'])
    parser.add_argument('--root')
    parser.add_argument('--arm', choices=q.ARMS)
    parser.add_argument('--work-dir')
    parser.add_argument('--selection', default=str(BASE / 'SOURCE_SELECTION_FINAL.json'))
    parser.add_argument('--source-freeze', default=str(BASE / 'SOURCE_FREEZE.json'))
    args = parser.parse_args()
    if args.command == 'self-check':
        print(json.dumps(self_check(), ensure_ascii=False)); return 0
    if not args.root or not args.arm:
        parser.error('Explicit root and arm are required')
    root = q.root_id(args.root)
    selection = Path(args.selection).resolve()
    q.SOURCE_SELECTION_PATH = selection
    q.SOURCE_FREEZE_PATH = Path(args.source_freeze).resolve()
    _, sources = q.load_selection(selection)
    work = select_input(root, args.arm, args.work_dir)
    if root not in sources:
        raise RuntimeError('No approved shared source for this planned root')
    result = inspect_budget(root, args.arm, sources[root], work) if args.command == 'inspect' else run_one(root, args.arm, sources[root], selection, work)
    print(json.dumps(result, ensure_ascii=False)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
