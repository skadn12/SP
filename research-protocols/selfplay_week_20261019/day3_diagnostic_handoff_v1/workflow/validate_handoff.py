#!/usr/bin/env python3
"""Validate frozen diagnostic handoff artifacts; does not judge semantic truth.

Usage: python workflow/validate_handoff.py --state HANDOFF_STATE.json --output VALIDATION.json
Exit 0: structurally complete, including complete negative/invalid records.
Exit 2: structural/integrity failure. Exit 3: --require-ready requested but gate is NO_GO.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

REQUIRED = ('accepted_questions', 'execution_freeze', 'direct_baseline_binding',
            'coverage_plan', 'responses', 'semantic_reviews', 'feedback_snapshot')
REVIEW_RESULTS = {'PASS', 'FAIL', 'HOLD', 'NOT_APPLICABLE'}
DISPOSITIONS = {'usable_local', 'negative_evidence', 'invalid_diagnostic',
                'review_hold', 'no_supported_weakness'}

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def read_rows(path):
    if path.suffix == '.jsonl':
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    obj = json.loads(path.read_text())
    if isinstance(obj, list): return obj
    for k in ('questions', 'items', 'responses', 'reviews', 'records'):
        if isinstance(obj.get(k), list): return obj[k]
    raise ValueError(f'No row list in {path}')

def validate(state_path):
    state_path = Path(state_path).resolve()
    failures, warnings, checks = [], [], 0
    def check(cond, code, detail=''):
        nonlocal checks
        checks += 1
        if not cond: failures.append({'code': code, 'detail': str(detail)})
        return bool(cond)
    def bind(binding, name):
        if not check(isinstance(binding, dict), 'BINDING_MISSING', name): return None
        value = binding.get('path')
        if not check(isinstance(value, str) and value, 'BINDING_PATH', name): return None
        path = Path(value)
        if not path.is_absolute(): path = state_path.parent / path
        path = path.resolve()
        if not check(path.is_file(), 'BINDING_FILE_MISSING', str(path)): return None
        if not check(binding.get('sha256') == sha(path), 'BINDING_HASH_MISMATCH', name): return None
        if 'bytes' in binding: check(binding['bytes'] == path.stat().st_size, 'BINDING_SIZE_MISMATCH', name)
        return path
    state = json.loads(state_path.read_text())
    artifacts = state.get('artifacts', {})
    paths = {k: bind(artifacts.get(k), k) for k in REQUIRED}
    for k, v in artifacts.items():
        if k not in REQUIRED: bind(v, k)
    if not all(paths.values()):
        return {'status': 'FAIL', 'checks': checks, 'failures': failures, 'warnings': warnings, 'gates': {}}
    accepted = read_rows(paths['accepted_questions'])
    questions = {q.get('candidate_id'): q for q in accepted}
    check(len(questions) == len(accepted) and None not in questions, 'ACCEPTED_UNIQUE_IDS')
    expected = state.get('expected_question_count', 23)
    check(len(questions) == expected, 'ACCEPTED_COUNT', len(questions))
    check(state.get('prospective_all_question_diagnostic_rule') is True, 'PROSPECTIVE_RULE_DECLARATION')
    check(state.get('question_admission_changed_by_diagnostic') is False, 'QUESTION_CONTENT_ADMISSION_PRESERVED')
    check(state.get('primary_evaluation') == 'direct', 'DIRECT_REMAINS_PRIMARY')
    baseline = json.loads(paths['direct_baseline_binding'].read_text())
    check(baseline.get('calls') == state.get('expected_direct_calls', 342), 'DIRECT_BASELINE_CALL_COUNT')
    check(baseline.get('unchanged') is True, 'DIRECT_BASELINE_UNCHANGED')
    for i, b in enumerate(baseline.get('bindings', [])): bind(b, f'direct_baseline[{i}]')
    check(bool(baseline.get('bindings')), 'DIRECT_BASELINE_BINDINGS_REQUIRED')
    coverage = json.loads(paths['coverage_plan'].read_text())
    items = coverage.get('items', [])
    by_id = {x.get('candidate_id'): x for x in items}
    requests = {x.get('request_id'): x for x in items}
    check(set(by_id) == set(questions), 'FULL_ACCEPTED_COVERAGE')
    check(len(by_id) == len(items) == len(requests) and None not in requests, 'ONE_FIXED_REQUEST_PER_QUESTION')
    max_calls = coverage.get('max_solver_calls')
    check(isinstance(max_calls, int) and not isinstance(max_calls, bool) and len(items) <= max_calls <= 23, 'FIXED_CALL_BUDGET', max_calls)
    check(coverage.get('automatic_retries') is False, 'NO_AUTOMATIC_RETRIES')
    for cid, item in by_id.items():
        q = questions.get(cid, {})
        mapping = item.get('display_to_canonical', {})
        labels = set(q.get('options' if q.get('question_type') == 'multi_select' else 'items', {}))
        check(bool(labels) and set(mapping) == labels == set(mapping.values()), 'DISPLAY_BIJECTION', cid)
        check(item.get('root_id') == q.get('root_id'), 'ROOT_ID_MATCH', cid)
        check(bool(item.get('fact_family_id')), 'FACT_FAMILY_ID_REQUIRED', cid)
    response_rows = read_rows(paths['responses'])
    response_by_req = {r.get('request_id'): r for r in response_rows}
    check(len(response_rows) == len(response_by_req) == len(items), 'RESPONSE_SLOT_DENOMINATOR')
    check(set(response_by_req) == set(requests), 'RESPONSE_REQUEST_COVERAGE')
    valid_ids, invalid_ids, quote_count = set(), set(), 0
    quote_mismatch_ids, quote_mismatches = set(), []
    source_cache = {}
    for reqid, r in response_by_req.items():
        item = requests.get(reqid, {})
        cid = item.get('candidate_id')
        check(r.get('candidate_id') == cid, 'RESPONSE_QUESTION_MATCH', reqid)
        check(isinstance(r.get('status'), str) and bool(r['status']), 'TERMINAL_STATUS_REQUIRED', reqid)
        check(isinstance(r.get('raw_output'), str), 'RAW_OUTPUT_REQUIRED', reqid)
        if r.get('status') == 'RETURNED': check(bool(r.get('raw_output')), 'RETURNED_OUTPUT_EMPTY', reqid)
        bind(r.get('raw_response_binding'), f'raw_response:{reqid}')
        check(isinstance(r.get('diagnostic_valid'), bool), 'DIAGNOSTIC_VALID_BOOLEAN', reqid)
        if not r.get('diagnostic_valid'):
            invalid_ids.add(cid)
            check(bool(r.get('invalid_reason')), 'INVALID_REASON_REQUIRED', reqid)
            continue
        valid_ids.add(cid)
        p = r.get('parsed')
        if not check(isinstance(p, dict), 'PARSED_DIAGNOSTIC_REQUIRED', reqid): continue
        q = questions.get(cid, {})
        labels = set(q.get('options' if q.get('question_type') == 'multi_select' else 'items', {}))
        judgments = p.get('item_judgments', [])
        check(isinstance(judgments, list), 'ITEM_JUDGMENTS_LIST', cid)
        if not isinstance(judgments, list): judgments = []
        jlabels = [j.get('label') for j in judgments]
        check(len(jlabels) == len(labels) and set(jlabels) == labels, 'JUDGMENT_ALL_ITEMS_ONCE', cid)
        answer = p.get('final_answer')
        check(isinstance(answer, list) and bool(answer) and len(answer) == len(set(answer)) and set(answer) <= labels, 'FINAL_ANSWER_DISPLAY_LABELS', cid)
        if q.get('question_type') == 'ordering' and isinstance(answer, list): check(set(answer) == labels, 'ORDERING_FULL_PERMUTATION', cid)
        source_path = Path(q.get('source_path', ''))
        if not source_path.is_absolute(): source_path = paths['accepted_questions'].parent / source_path
        source_key = str(source_path.resolve())
        if source_key not in source_cache:
            sp = bind({'path': str(source_path), 'sha256': q.get('source_sha256')}, f'source:{cid}')
            if sp: source_cache[source_key] = json.loads(sp.read_text()).get('turns', [])
        turns = source_cache.get(source_key, [])
        for j in judgments:
            check(isinstance(j.get('judgment'), str) and bool(j['judgment'].strip()), 'ITEM_JUDGMENT_REQUIRED', f'{cid}:{j.get("label")}')
            check(isinstance(j.get('rationale'), str) and bool(j['rationale'].strip()), 'ITEM_RATIONALE_REQUIRED', f'{cid}:{j.get("label")}')
            evidence = j.get('evidence', [])
            check(isinstance(evidence, list), 'ITEM_EVIDENCE_LIST', cid)
            if not evidence: check(j.get('evidence_status') in {'no_support_found', 'not_applicable'}, 'EMPTY_EVIDENCE_EXPLICIT_STATUS', f'{cid}:{j.get("label")}')
            for e in evidence if isinstance(evidence, list) else []:
                quote_count += 1
                turn = e.get('turn')
                if isinstance(turn, int) and not isinstance(turn, bool): matches = turns[turn-1:turn] if 1 <= turn <= len(turns) else []
                else: matches = [t for t in turns if t.get('turn_ref') == turn]
                quote = e.get('quote')
                is_match = len(matches) == 1 and isinstance(quote, str) and bool(quote) and quote in matches[0].get('text', '')
                # A fabricated/mislocated model citation is a real diagnostic outcome,
                # not corrupt research data. Require the review to detect it below.
                checks += 1
                if not is_match:
                    quote_mismatch_ids.add(cid)
                    quote_mismatches.append({'candidate_id': cid, 'label': j.get('label'), 'turn': turn, 'quote': quote})
    reviews = read_rows(paths['semantic_reviews'])
    rb = {r.get('request_id'): r for r in reviews}
    check(len(rb) == len(reviews) == len(items) and set(rb) == set(requests), 'SEMANTIC_REVIEW_COVERAGE')
    review_pass_ids, review_hold_ids, reviewed_negative_ids = set(), set(), set()
    for reqid, review in rb.items():
        cid = requests.get(reqid, {}).get('candidate_id')
        check(review.get('candidate_id') == cid, 'REVIEW_QUESTION_MATCH', reqid)
        status = review.get('review_status')
        check(status in {'PASS', 'FAIL', 'HOLD', 'NOT_REVIEWABLE'}, 'REVIEW_STATUS_ENUM', reqid)
        check(review.get('training_use') is False, 'RATIONALE_NOT_TRAINING_LABEL', reqid)
        check(isinstance(review.get('limitations'), list), 'REVIEW_LIMITATIONS_REQUIRED', reqid)
        for f in ('quote_check', 'semantic_support_check', 'judgment_answer_consistency_check'):
            check(review.get(f) in REVIEW_RESULTS, 'REVIEW_DIMENSION_ENUM', f'{reqid}:{f}')
        if status == 'PASS':
            check(cid in valid_ids and all(review.get(f) == 'PASS' for f in ('quote_check', 'semantic_support_check', 'judgment_answer_consistency_check')), 'REVIEW_PASS_ALL_DIMENSIONS', cid)
            review_pass_ids.add(cid)
        elif status == 'HOLD': review_hold_ids.add(cid)
        elif status == 'FAIL':
            # FAIL here denotes an audited model-semantic error, not failure to
            # perform the review. It can support a narrowly phrased negative card.
            check(cid in valid_ids and bool(review.get('source_verified_error')), 'NEGATIVE_REVIEW_SOURCE_VERIFIED', cid)
            reviewed_negative_ids.add(cid)
        if cid in quote_mismatch_ids:
            check(review.get('quote_check') == 'FAIL', 'QUOTE_MISMATCH_MUST_BE_REVIEWED_FAIL', cid)
    feedback = json.loads(paths['feedback_snapshot'].read_text())
    check(bool(feedback.get('version')), 'FEEDBACK_VERSION_REQUIRED')
    check('parent_version' in feedback, 'FEEDBACK_PARENT_VERSION_EXPLICIT')
    check(feedback.get('faithful_internal_trace_claim') is False, 'NO_INTERNAL_TRACE_CLAIM')
    check(feedback.get('unverified_rationale_training_target') is False, 'NO_UNVERIFIED_TRAINING_TARGET')
    entries = feedback.get('entries', [])
    eb = {e.get('feedback_id'): e for e in entries}
    check(len(eb) == len(entries) and None not in eb, 'FEEDBACK_IDS_UNIQUE')
    reproduced = []
    for fid, e in eb.items():
        check(e.get('status') in {'hypothesis', 'reproduced', 'improving', 'recurred', 'negative', 'supported_negative', 'unknown'}, 'FEEDBACK_STATUS_ENUM', fid)
        check(e.get('version') == feedback.get('version'), 'FEEDBACK_ENTRY_VERSION', fid)
        cids = e.get('evidence_candidate_ids', [])
        check(isinstance(cids, list) and bool(cids) and set(cids) <= set(questions), 'FEEDBACK_EVIDENCE_IDS', fid)
        check(bool(e.get('operational_claim')), 'OPERATIONAL_CLAIM_REQUIRED', fid)
        check(isinstance(e.get('competing_explanations'), list), 'COMPETING_EXPLANATIONS_REQUIRED', fid)
        check(bool(e.get('falsification_condition')), 'FALSIFICATION_REQUIRED', fid)
        if e.get('status') == 'supported_negative':
            check(bool(e.get('source_verified_operational_feedback')) and set(cids) <= reviewed_negative_ids, 'NEGATIVE_FEEDBACK_REVIEWED_SCOPE', fid)
        if e.get('status') == 'reproduced':
            proof = e.get('reproduction_evidence', {})
            ok = (len(set(proof.get('distinct_train_roots', []))) >= 3 and len(set(proof.get('distinct_families', []))) >= 3 and proof.get('same_frozen_solver') is True and proof.get('qualified_source_reviewed_failures') is True and proof.get('related_predeclared_controlled_probe_support') is True and bool(proof.get('probe_bindings')) and bool(e.get('competing_explanations')))
            check(ok, 'ORIGINAL_REPRODUCED_GATE', fid)
            for i, b in enumerate(proof.get('probe_bindings', [])): bind(b, f'probe:{fid}:{i}')
            if ok: reproduced.append(fid)
    dispositions = feedback.get('question_dispositions', [])
    db = {d.get('candidate_id'): d for d in dispositions}
    check(len(db) == len(dispositions) == len(items) and set(db) == set(questions), 'FEEDBACK_ALL_QUESTION_DISPOSITIONS')
    usable_ids, isolated_ids = set(), set()
    for cid, d in db.items():
        check(d.get('status') in DISPOSITIONS, 'DISPOSITION_ENUM', cid)
        check(bool(d.get('reason')), 'DISPOSITION_REASON', cid)
        check(isinstance(d.get('feedback_ids'), list) and set(d.get('feedback_ids', [])) <= set(eb), 'DISPOSITION_FEEDBACK_IDS', cid)
        if d.get('status') == 'usable_local':
            check(cid in review_pass_ids and bool(d.get('feedback_ids')), 'USABLE_REQUIRES_VALID_SEMANTICS_AND_FEEDBACK', cid)
            usable_ids.add(cid)
        if d.get('status') == 'negative_evidence' and d.get('feedback_ids'):
            check(cid in reviewed_negative_ids and d.get('source_verified_operational_feedback') is True, 'NEGATIVE_DISPOSITION_VERIFIED_SCOPE', cid)
            usable_ids.add(cid)
        if d.get('status') in {'invalid_diagnostic', 'review_hold'}:
            check(d.get('excluded_from_usable_feedback') is True, 'INVALID_OR_HOLD_EXPLICITLY_ISOLATED', cid)
            if d.get('excluded_from_usable_feedback') is True: isolated_ids.add(cid)
        if cid in invalid_ids: check(d.get('status') == 'invalid_diagnostic', 'INVALID_EXPLICITLY_NOT_READY', cid)
        if cid in review_hold_ids: check(d.get('status') == 'review_hold', 'HOLD_EXPLICITLY_NOT_READY', cid)
    handoff = feedback.get('generation_handoff', {})
    closed_claim = handoff.get('feedback_loop_closed', False)
    actual_inputs = handoff.get('actual_input_bindings', [])
    target_reviews = handoff.get('target_realization_review_bindings', [])
    for i, b in enumerate(actual_inputs): bind(b, f'challenger_actual_input:{i}')
    for i, b in enumerate(target_reviews): bind(b, f'target_realization_review:{i}')
    if closed_claim: check(bool(actual_inputs) and bool(target_reviews) and bool(handoff.get('feedback_to_plan_to_candidate')), 'CLOSED_LOOP_REAL_CONSUMPTION_REQUIRED')
    collection_complete = set(response_by_req) == set(requests) and len(response_rows) == len(items)
    semantic_complete = set(rb) == set(requests) and len(reviews) == len(items)
    snapshot_complete = set(db) == set(questions) and len(dispositions) == len(items)
    blocked = (invalid_ids | review_hold_ids) - isolated_ids
    local_ready = not failures and collection_complete and semantic_complete and snapshot_complete and bool(usable_ids) and not blocked
    formal_go = local_ready and bool(reproduced) and closed_claim and bool(handoff.get('formal_target_realization_gate_passed'))
    if invalid_ids: warnings.append('Invalid diagnostics are preserved; their source questions remain content-admitted and direct scores remain valid.')
    if not closed_claim: warnings.append('Diagnostic/feedback snapshot does not yet prove actual downstream consumption; feedback_loop_closed=false.')
    if not reproduced: warnings.append('No newly verified reproduced weakness; formal B gate remains NO_GO.')
    return {'schema_version': 'diagnostic_handoff_validation_v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'state_path': str(state_path), 'state_sha256': sha(state_path),
            'status': 'FAIL' if failures else ('PASS' if local_ready else 'PASS_WITH_NO_GO'),
            'checks': checks, 'failures': failures, 'warnings': warnings,
            'counts': {'accepted_questions': len(questions), 'fixed_requests': len(items), 'responses': len(response_rows),
                       'valid_diagnostics': len(valid_ids), 'invalid_diagnostics': len(invalid_ids), 'semantic_pass': len(review_pass_ids),
                       'semantic_hold': len(review_hold_ids), 'reviewed_negative': len(reviewed_negative_ids),
                       'usable_local_questions': len(usable_ids), 'isolated_invalid_or_hold': len(isolated_ids),
                       'verbatim_quotes_checked': quote_count, 'quote_mismatch_count': len(quote_mismatches)},
            'quote_mismatches': quote_mismatches,
            'gates': {'diagnostic_collection_complete': collection_complete, 'semantic_review_complete': semantic_complete,
                      'feedback_snapshot_complete': snapshot_complete, 'local_feedback_ready': local_ready,
                      'local_feedback_ready_scope': sorted(usable_ids) if local_ready else [],
                      'all_diagnostics_valid': not invalid_ids,
                      'feedback_loop_closed': bool(closed_claim and not failures), 'formal_feedback_B_go': bool(formal_go)},
            'question_content_admission_changed': False, 'primary_evaluation': 'direct',
            'limits': ['Integrity and declared semantic judgments are checked; this program does not infer semantic truth or faithful internal reasoning.',
                       'Formal reproduced evidence must still be semantically audited; a declaration or one diagnostic per question cannot replace the original probes.']}

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--state', required=True)
    ap.add_argument('--output')
    ap.add_argument('--require-ready', action='store_true')
    args = ap.parse_args()
    try: result = validate(args.state)
    except Exception as exc: result = {'status': 'FAIL', 'failures': [{'code': 'UNHANDLED_INVALID_ARTIFACT', 'detail': f'{type(exc).__name__}: {exc}'}], 'gates': {}}
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + chr(10)
    if args.output:
        out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(rendered)
    print(rendered, end='')
    if result['status'] == 'FAIL': return 2
    if args.require_ready and not result.get('gates', {}).get('local_feedback_ready'): return 3
    return 0

if __name__ == '__main__': raise SystemExit(main())
