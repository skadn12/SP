"""Offline, sequential review accounting for the six frozen Miner requests.

Semantic judgments are supplied by an attributed reviewer. Code only preserves
raw output, checks review coverage and prerequisites, and computes denominators.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from ..core import atomic_json, atomic_text, sha256_file
from ..simple_append_v1.live_runtime import load_json_object_strict
from ..h4_r02_miner_binding_v2.transfer import derive_all_intent_evidence, validate_bindings
from .constants import ROOTS, ARMS, SCHEDULE, VERSION, root_path

REVIEWER = 'Codex direct transcript review'
SEMANTIC = {'PASS', 'FAIL', 'UNDETERMINED'}
NOT_EVALUATED = 'NOT_EVALUATED_STRUCTURAL_OR_MISSING'
TRANSFER_CRITERIA = ('old_question_focus', 'feedback_basis', 'added_distinction', 'intent_alignment', 'scope')
C_ARM = 'C_P2_2_TRANSFER'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else deepcopy(default)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).is_file() else []


def write_jsonl(path, rows):
    atomic_text(Path(path), ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows))


def key(row):
    return row.get('root_id'), row.get('arm_id'), row.get('intent_index')


def refs_valid(refs):
    return isinstance(refs, list) and all(isinstance(ref, str) and ref for ref in refs)


def evidence_metrics(derived, expanded):
    """Exact original-chain metrics; unknown values are null, with no cap."""
    empty = {'evidence_ref_count': None, 'distinct_scene_count': None,
        'evidence_span': {'min_position': None, 'max_position': None, 'width': None},
        'distinct_event_count': None, 'scene_mapping_methods': [],
        'event_policy': 'No reliable frozen event map supplied; no semantic event inference.',
        'admission_thresholds': None}
    if not isinstance(derived, dict) or not derived.get('usable'):
        return empty
    refs = {r['turn_ref'] for r in derived.get('derived_current_evidence', [])}
    turns = expanded.get('turns', [])
    selected = [(i, t) for i, t in enumerate(turns, 1) if t.get('turn_ref') in refs]
    if not refs or len(selected) != len(refs):
        return empty
    methods, scenes = [], []
    for _, turn in selected:
        if turn.get('scene_ref'):
            method, value = 'frozen_scene_ref', turn['scene_ref']
        elif turn.get('session_ref'):
            method, value = 'frozen_session_ref', turn['session_ref']
        elif ':T' in str(turn.get('turn_ref', '')):
            method, value = 'explicit_turn_ref_prefix_before_:T', turn['turn_ref'].rsplit(':T', 1)[0]
        else:
            method, value = 'UNMAPPABLE', None
        methods.append(method); scenes.append(value)
    positions = [i for i, _ in selected]
    return {**empty, 'evidence_ref_count': len(refs),
        'distinct_scene_count': None if None in scenes else len(set(scenes)),
        'evidence_span': {'min_position': min(positions), 'max_position': max(positions),
                          'width': max(positions) - min(positions) + 1},
        'scene_mapping_methods': sorted(set(methods))}


def collect_group(run, arm_id, root_id):
    """Keep planned missing slots and every extra raw item without repairing it."""
    folder = root_path(run, arm_id, root_id)
    source = run / 'inputs/shared' / root_id
    expanded = read(source / 'expanded_dialogue_clean.json', {})
    cards = read(source / 'observation_cards.json', {})
    validation = read(folder / 'intents.json', {}).get('validation', {})
    normalized = validation.get('intents', [])
    contracts = read(folder / 'reasoning_contracts.json', [])
    evidence = read(folder / 'derived_intent_evidence.json', {})
    bindings = read(folder / 'binding_validation.json', {}) if arm_id == C_ARM else {}
    raw_path = folder / 'raw/responses' / (root_id + '-intent-miner.json.txt')
    raw, parse_error = None, None
    if raw_path.is_file():
        try:
            raw = load_json_object_strict(raw_path.read_text())
        except (ValueError, json.JSONDecodeError) as error:
            parse_error = type(error).__name__ + ':' + str(error)
    raw_items = raw.get('intents', []) if isinstance(raw, dict) and isinstance(raw.get('intents'), list) else []
    checks = {'miner_artifacts_present': all((folder / n).is_file() for n in
        ('intents.json', 'reasoning_contracts.json', 'derived_intent_evidence.json')),
        'raw_response_recorded': raw_path.is_file()}
    if checks['miner_artifacts_present'] and expanded:
        try:
            checks['evidence_reconstruction'] = evidence == derive_all_intent_evidence(validation, contracts,
                expanded=expanded, root_id=root_id, transcript_sha256=sha256_file(source / 'expanded_dialogue_clean.json'))
            checks['no_evidence_integrity_errors'] = not evidence.get('integrity_errors')
            if arm_id == C_ARM:
                checks['binding_reconstruction'] = bindings == validate_bindings(raw, validation, contracts,
                    expanded=expanded, root_id=root_id, cards=cards)
            else:
                checks['binding_processing_not_applicable'] = not any((folder / n).exists() for n in
                    ('binding_validation.json', 'feedback_bindings.json', 'guided_brief_preview.json'))
        except Exception as error:
            checks['reconstruction_error'] = False
            checks['reconstruction_error_detail'] = type(error).__name__ + ':' + str(error)
    items = []
    derived_by_index = {d.get('intent_index'): d for d in evidence.get('items', [])}
    for index in range(max(3, len(raw_items), len(normalized))):
        intent = normalized[index] if index < len(normalized) else None
        derived = derived_by_index.get(index, {})
        identifier = intent.get('intent_id') if isinstance(intent, dict) else None
        raw_intent = raw_items[index] if index < len(raw_items) else None
        if identifier is None and isinstance(raw_intent, dict):
            identifier = raw_intent.get('intent_id')
        matching = [b for b in bindings.get('bindings', []) if identifier is not None and b.get('intent_id') == identifier]
        selected = matching[0] if len(matching) == 1 and matching[0].get('mechanical_valid') else None
        binding_state = (bindings.get('by_intent', {}).get(identifier, {}).get('binding_state',
            'BINDING_MISSING' if not matching else 'BINDING_INVALID') if arm_id == C_ARM else 'NOT_APPLICABLE')
        structural = bool(isinstance(intent, dict) and intent.get('valid') and derived.get('usable'))
        item = {'root_id': root_id, 'arm_id': arm_id, 'intent_index': index,
            'intent_id': identifier, 'planned_slot': index < 3, 'returned': index < len(raw_items),
            'structurally_valid': structural, 'raw_intent': deepcopy(raw_intent),
            'original_intent': deepcopy(intent), 'derived_evidence': deepcopy(derived),
            'intent_errors': (intent or {}).get('errors', []),
            'response_errors': list(validation.get('errors') or []),
            'parse_error': parse_error, 'binding_state': binding_state,
            'binding_mechanical_valid': bool(selected), 'binding_indices': [b['binding_index'] for b in matching],
            'raw_bindings': [deepcopy(b.get('binding')) for b in matching],
            'binding_errors': [b.get('errors', []) for b in matching],
            'proposed_transfer': any(b.get('status') == 'PROPOSED_TRANSFER' for b in matching),
            'valid_proposed_transfer': bool(selected and selected.get('status') == 'PROPOSED_TRANSFER'),
            'model_declared_no_transfer': any(b.get('status') == 'NO_SUPPORTED_TRANSFER' for b in matching),
            'valid_no_transfer': bool(selected and selected.get('status') == 'NO_SUPPORTED_TRANSFER'),
            'current_refs': [e['turn_ref'] for e in derived.get('derived_current_evidence', [])],
            'raw_response_path': str(raw_path), 'frozen_transcript_path': str(source / 'expanded_dialogue_clean.json'),
            'metrics': evidence_metrics(derived, expanded)}
        items.append(item)
    covered = {i for item in items for i in item['binding_indices']}
    orphan_bindings = [deepcopy(b) for b in bindings.get('bindings', []) if b['binding_index'] not in covered]
    summary = {'root_id': root_id, 'arm_id': arm_id, 'planned_slots': 3,
        'returned': min(len(raw_items), 3), 'returned_total': len(raw_items),
        'extra_outputs': max(0, len(raw_items) - 3),
        'structurally_valid': sum(i['structurally_valid'] for i in items if i['planned_slot']),
        'structurally_valid_all_returned': sum(i['structurally_valid'] for i in items),
        'missing': max(0, 3 - len(raw_items)), 'response_schema_valid': bool(validation.get('valid')),
        'response_errors': list(validation.get('errors') or []), 'parse_error': parse_error,
        'binding_processing': 'V2_UNCHANGED' if arm_id == C_ARM else 'NOT_APPLICABLE',
        'raw_bindings_envelope': deepcopy(raw.get('feedback_bindings')) if isinstance(raw, dict) else None,
        'orphan_binding_rows': orphan_bindings, 'binding_envelope_errors': bindings.get('envelope_errors', []),
        'checks': checks}
    return summary, items


def call_accounting(run):
    state_records = []
    for path in sorted(run.rglob('ledger/requests/*.json')):
        try:
            state = read(path, {})
        except (ValueError, OSError) as error:
            state = {'status': 'UNREADABLE', 'read_error': str(error)}
        state_records.append((path, state))
    transport = jsonl(run / 'audit/transport_events.jsonl')
    call_events = jsonl(run / 'audit/call_events.jsonl')
    def summarize(records, events, logical_events):
        states = [s for _, s in records]
        completed = [s for s in states if s.get('status') == 'completed']
        usage = Counter()
        for s in completed:
            usage.update({k: v for k, v in (s.get('response_metadata', {}).get('usage') or {}).items()
                          if isinstance(v, (int, float)) and not isinstance(v, bool)})
        failures = sum((p.parents[1] / 'observed_failures' / p.name).is_file()
                       for p, s in records if s.get('status') != 'completed')
        outcomes = [e for e in events if e.get('event') == 'logical_call_transport_outcome']
        return {'logical_calls': len(states), 'completed_calls': len(completed),
            'observed_failed_calls': failures, 'unknown_reservations': len(states) - len(completed) - failures,
            'unknown_billing_status_calls': len(states) - len(completed),
            'http_attempts': sum(e.get('event') == 'http_attempt_started' for e in events),
            'http_retries': sum(max(0, e.get('transport_attempt_count', 0) - 1) for e in outcomes),
            'finish_reasons': dict(Counter(str(s.get('response_metadata', {}).get('finish_reason', 'unknown')) for s in completed)),
            'usage': dict(usage), 'completed_without_usage': sum(not s.get('response_metadata', {}).get('usage') for s in completed),
            'cache_hits': sum(e.get('event') == 'completed_cache_reused_no_send' for e in logical_events),
            'known_failure_replays': sum(e.get('event') == 'known_failure_replayed_no_send' for e in logical_events)}
    groups = []
    known_paths = set()
    for schedule in SCHEDULE:
        arm, root = schedule['arm_id'], schedule['root_id']
        expected_folder = root_path(run, arm, root) / 'ledger/requests'
        records = [(p, s) for p, s in state_records if p.parent == expected_folder]
        known_paths.update(p for p, _ in records)
        match = lambda e: e.get('root_id') == root and e.get('arm_id', e.get('arm')) == arm
        groups.append({**schedule, **summarize(records, [e for e in transport if match(e)], [e for e in call_events if match(e)])})
    unexpected = [str(p) for p, _ in state_records if p not in known_paths]
    non_miner = [str(p) for p, s in state_records if s.get('request_id') != s.get('request_metadata', {}).get('root_id', '') + '-intent-miner']
    return {'planned_logical_calls': 6, 'maximum_logical_calls': 6,
        **summarize(state_records, transport, call_events), 'groups': groups,
        'unexpected_ledger_paths': unexpected, 'non_miner_request_paths': non_miner,
        'writer_calls': sum('-generator-' in str(p) for p, _ in state_records),
        'repair_calls': sum('-repair-' in str(p) for p, _ in state_records),
        'solver_memory_training_other_calls': len(non_miner),
        'monetary_cost': None, 'monetary_cost_note': 'No provider billing amount supplied; no price inferred.'}


def mechanical_report(run):
    run = Path(run)
    errors = []
    try:
        from .run import verify_frozen
        verify_frozen(run)
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        errors.append('frozen_integrity:' + str(error))
    try:
        from .runtime import audit_reservations
        audit_reservations(run)
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        errors.append('ledger_scope_or_reservation_integrity:' + str(error))
    accounting = call_accounting(run)
    pairing = {(row.get('arm_id'), row.get('root_id')): row for row in jsonl(run / 'pairing_manifest.jsonl')}
    groups, items = [], []
    for schedule in SCHEDULE:
        group, output = collect_group(run, schedule['arm_id'], schedule['root_id'])
        folder = root_path(run, group['arm_id'], group['root_id'])
        prompt = folder / 'raw/prompts' / (group['root_id'] + '-intent-miner.txt')
        paired = pairing.get((group['arm_id'], group['root_id']), {})
        group['checks']['frozen_prompt_exact_hash'] = prompt.is_file() and sha256_file(prompt) == paired.get('prompt_sha256')
        try:
            from .runtime import verify_checkpoint
            group['checks']['completed_artifact_checkpoint'] = verify_checkpoint(folder)
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            group['checks']['completed_artifact_checkpoint'] = False
            group['checks']['checkpoint_error_detail'] = str(error)
        group['calls'] = next(c for c in accounting['groups'] if c['arm_id'] == group['arm_id'] and c['root_id'] == group['root_id'])
        groups.append(group); items.extend(output)
        errors.extend(group['arm_id'] + ':' + group['root_id'] + ':' + k for k, v in group['checks'].items() if v is False)
    if accounting['logical_calls'] != 6 or any(g['logical_calls'] != 1 for g in accounting['groups']):
        errors.append('six_scheduled_reservations_incomplete_or_violated')
    if accounting['unexpected_ledger_paths'] or accounting['non_miner_request_paths']:
        errors.append('unexpected_request_scope')
    forbidden = [str(p) for p in (run / 'arms').rglob('*') if p.is_file() and
        (p.name in {'candidate_pool.json', 'candidate_generation.json', 'candidates_valid.json'} or '-generator-' in p.name or '-repair-' in p.name)]
    if forbidden:
        errors.append('forbidden_non_miner_artifacts')
    result = {'version': VERSION, 'status': 'PASS' if not errors else 'FAIL', 'errors': errors,
        'groups': groups, 'items': items, 'accounting': accounting, 'forbidden_files': forbidden,
        'mechanical_not_semantic': True}
    atomic_json(run / 'audit/call_accounting.json', accounting)
    atomic_json(run / 'audit/mechanical_report.json', result)
    write_jsonl(run / 'reports/evidence_width_metrics.jsonl', [
        {k: item[k] for k in ('root_id', 'arm_id', 'intent_index', 'intent_id', 'planned_slot', 'structurally_valid')} | item['metrics'] for item in items])
    return result


def assemble_intent_reviews(items, authored):
    """Fill only structural/missing states; never assign semantic judgments."""
    errors, records = [], []
    expected = {key(i) for i in items}
    by_key = {}
    for row in authored:
        k = key(row)
        if k not in expected:
            errors.append('unknown_review_item:' + repr(k))
        if k in by_key:
            errors.append('duplicate_review_item:' + repr(k))
        by_key[k] = row
    for item in items:
        k = key(item); row = by_key.get(k)
        if not item['structurally_valid']:
            if row is not None and row.get('judgment') != NOT_EVALUATED:
                errors.append('semantic_judgment_on_structurally_unusable_item:' + repr(k))
            row = row or {**{f: item[f] for f in ('root_id', 'arm_id', 'intent_index', 'intent_id')},
                'judgment': NOT_EVALUATED, 'reason': 'No original structurally valid, usable intent evidence for this slot; semantic review not performed.',
                'current_refs': [], 'reviewer': REVIEWER, 'record_origin': 'PROGRAM_STRUCTURAL_STATUS_ONLY'}
        elif row is None:
            errors.append('missing_semantic_review:' + repr(k)); continue
        if row.get('reviewer') != REVIEWER or not isinstance(row.get('reason'), str) or not row['reason'].strip() or not refs_valid(row.get('current_refs')):
            errors.append('review_schema_or_attribution:' + repr(k))
        if item['structurally_valid'] and row.get('judgment') not in SEMANTIC:
            errors.append('review_judgment_invalid:' + repr(k))
        records.append({**row, 'planned_slot': item['planned_slot']})
    return records, errors


def freeze_intent_review(run, mechanical=None):
    """Finish layer one and lock its exact bytes before any layer-two judgment."""
    run = Path(run)
    mechanical = mechanical or mechanical_report(run)
    path, lock = run / 'reports/intent_evidence_review.jsonl', run / 'audit/intent_review_freeze.json'
    if lock.is_file():
        if not path.is_file() or read(lock)['sha256'] != sha256_file(path):
            raise ValueError('INTENT_REVIEW_CHANGED_AFTER_FREEZE')
        rows, errors = assemble_intent_reviews(mechanical['items'], jsonl(path))
        if errors:
            raise ValueError('INTENT_REVIEW_COVERAGE:' + repr(errors))
        return read(lock)
    if jsonl(run / 'reports/transfer_judgments.jsonl') or jsonl(run / 'reports/transfer_review.jsonl'):
        raise ValueError('TRANSFER_REVIEW_PRECEDES_INTENT_REVIEW_FREEZE')
    rows, errors = assemble_intent_reviews(mechanical['items'], jsonl(path))
    if errors:
        raise ValueError('INTENT_REVIEW_COVERAGE:' + repr(errors))
    write_jsonl(path, rows)
    value = {'version': VERSION, 'path': str(path), 'sha256': sha256_file(path),
        'review_count': len(rows), 'planned_slots': 18, 'at_utc': now(),
        'policy_sha256': sha256_file(run / 'review_policy.json'), 'immutable_before_stage_2': True}
    atomic_json(lock, value)
    return value


def assemble_transfer_reviews(items, base_reviews, authored, orphan_bindings=()):
    """A/B=N/A; C semantics require a base PASS and valid proposed binding."""
    base = {key(r): r for r in base_reviews}; semantic = {}
    errors, rows, consumed = [], [], set()
    for row in authored:
        k = key(row)
        if k in semantic:
            errors.append('duplicate_transfer_review:' + repr(k))
        semantic[k] = row
    for item in items:
        k = key(item); first = base.get(k, {}); supplied = semantic.get(k)
        common = {field: deepcopy(item[field]) for field in ('root_id', 'arm_id', 'intent_index', 'intent_id', 'planned_slot',
            'binding_state', 'binding_indices', 'binding_errors', 'raw_bindings', 'proposed_transfer',
            'valid_proposed_transfer', 'model_declared_no_transfer', 'valid_no_transfer')}
        common.update(base_judgment=first.get('judgment', NOT_EVALUATED), reviewer=REVIEWER)
        if item['arm_id'] != C_ARM:
            judgment, reason = 'NOT_APPLICABLE', 'A/B have no binding schema or transfer task.'
        elif first.get('judgment') != 'PASS':
            judgment, reason = 'NOT_EVALUATED_INPUT_INTENT_NOT_PASS', 'Layer one is not PASS; no transfer content judgment was performed.'
        elif not item['binding_mechanical_valid']:
            judgment, reason = 'NOT_EVALUATED_BINDING_INVALID', 'Missing or invalid binding; this is not model-declared refusal.'
        elif item['binding_state'] == 'NO_SUPPORTED_TRANSFER':
            judgment, reason = 'DECLINED_TRANSFER', 'The model explicitly declined transfer with valid empty fields. No content transfer success is counted; reasonableness may be discussed in the authored narrative.'
        elif item['valid_proposed_transfer']:
            if supplied is None:
                errors.append('missing_transfer_semantic_review:' + repr(k))
                judgment, reason = 'REVIEW_MISSING', 'Eligible proposed transfer still requires attributed offline review.'
            else:
                consumed.add(k)
                if supplied.get('judgment') not in SEMANTIC or supplied.get('reviewer') != REVIEWER or not supplied.get('reason') or not refs_valid(supplied.get('current_refs')) or not isinstance(supplied.get('historical_observation_keys'), list):
                    errors.append('transfer_review_schema:' + repr(k))
                criteria = supplied.get('criteria', {})
                for criterion in TRANSFER_CRITERIA:
                    check = criteria.get(criterion, {})
                    if check.get('judgment') not in SEMANTIC or not check.get('reason') or not refs_valid(check.get('current_refs')) or not isinstance(check.get('historical_observation_keys'), list):
                        errors.append('transfer_criterion:' + repr(k) + ':' + criterion)
                if supplied.get('judgment') == 'PASS' and any(criteria.get(c, {}).get('judgment') != 'PASS' for c in TRANSFER_CRITERIA):
                    errors.append('transfer_pass_without_all_criteria_pass:' + repr(k))
                rows.append({**common, **deepcopy(supplied), 'record_origin': 'AUTHORED_SEMANTIC_REVIEW'})
                continue
        else:
            judgment, reason = 'NOT_EVALUATED_BINDING_INVALID', 'No valid proposed transfer is associated with this intent.'
        if supplied is not None:
            consumed.add(k); errors.append('forbidden_transfer_judgment_before_prerequisites:' + repr(k))
        rows.append({**common, 'judgment': judgment, 'reason': reason,
                     'record_origin': 'PROGRAM_PREREQUISITE_STATUS_ONLY'})
    for orphan in orphan_bindings:
        rows.append({**deepcopy(orphan), 'intent_index': None, 'planned_slot': False,
            'judgment': 'NOT_EVALUATED_BINDING_INVALID', 'base_judgment': 'NO_ASSOCIATED_ORIGINAL_INTENT',
            'reason': 'Unknown or unassociated raw binding; retained without semantic review or planned-slot credit.',
            'reviewer': REVIEWER, 'record_origin': 'PROGRAM_PREREQUISITE_STATUS_ONLY'})
    for k in set(semantic) - consumed:
        errors.append('unknown_transfer_review_item:' + repr(k))
    return rows, errors


def summarize_layers(items, base_reviews, transfer_reviews):
    base = {key(r): r for r in base_reviews}; transfers = {key(r): r for r in transfer_reviews if r.get('intent_index') is not None}
    groups = []
    def summary(selected, planned_slots):
        planned = [i for i in selected if i['planned_slot']]
        counts = Counter(base.get(key(i), {}).get('judgment', 'REVIEW_MISSING') for i in planned)
        result = {'planned_slots': planned_slots, 'returned': sum(i['returned'] for i in planned),
            'returned_total': sum(i['returned'] for i in selected), 'extra_outputs': sum(not i['planned_slot'] for i in selected),
            'structurally_valid': sum(i['structurally_valid'] for i in planned),
            'semantic_pass': counts['PASS'], 'semantic_fail': counts['FAIL'], 'semantic_undetermined': counts['UNDETERMINED'],
            'not_evaluated': counts[NOT_EVALUATED], 'missing': sum(not i['returned'] for i in planned),
            'review_missing': counts['REVIEW_MISSING'], 'planned_pass_yield': counts['PASS'] / planned_slots,
            'reviewable_pass_fraction': counts['PASS'] / sum(i['structurally_valid'] for i in planned) if any(i['structurally_valid'] for i in planned) else None,
            'extra_output_reviews': [{**{f: i[f] for f in ('root_id', 'arm_id', 'intent_index', 'intent_id')}, 'judgment': base.get(key(i), {}).get('judgment', 'REVIEW_MISSING')} for i in selected if not i['planned_slot']]}
        citems = [i for i in planned if i['arm_id'] == C_ARM]
        if not citems:
            result['transfer'] = {'status': 'NOT_APPLICABLE'}
        else:
            tc = Counter(transfers.get(key(i), {}).get('judgment', 'REVIEW_MISSING') for i in citems)
            eligible = sum(base.get(key(i), {}).get('judgment') == 'PASS' and i['valid_proposed_transfer'] for i in citems)
            result['transfer'] = {'planned_slots': len(citems), 'base_pass': sum(base.get(key(i), {}).get('judgment') == 'PASS' for i in citems),
                'proposed': sum(i['proposed_transfer'] for i in citems), 'structurally_valid_proposed': sum(i['valid_proposed_transfer'] for i in citems),
                'eligible_transfer_reviews': eligible, 'transfer_pass': tc['PASS'], 'transfer_fail': tc['FAIL'], 'transfer_undetermined': tc['UNDETERMINED'],
                'model_declared_no_transfer': sum(i['model_declared_no_transfer'] for i in citems),
                'valid_declined_regardless_of_base': sum(i['valid_no_transfer'] for i in citems),
                'declined_after_base_pass': tc['DECLINED_TRANSFER'],
                'not_evaluated_base': tc['NOT_EVALUATED_INPUT_INTENT_NOT_PASS'], 'not_evaluated_binding': tc['NOT_EVALUATED_BINDING_INVALID'],
                'review_missing': tc['REVIEW_MISSING'], 'planned_transfer_pass_yield': tc['PASS'] / len(citems),
                'eligible_transfer_pass_fraction': tc['PASS'] / eligible if eligible else None}
        return result
    for schedule in SCHEDULE:
        selected = [i for i in items if i['root_id'] == schedule['root_id'] and i['arm_id'] == schedule['arm_id']]
        groups.append({**schedule, **summary(selected, 3)})
    arms = [{ 'arm_id': arm, **summary([i for i in items if i['arm_id'] == arm], 6)} for arm in ARMS]
    paired = []
    for root in ROOTS:
        values = {g['arm_id']: g['planned_pass_yield'] for g in groups if g['root_id'] == root}
        passed = {g['arm_id']: g['semantic_pass'] for g in groups if g['root_id'] == root}
        paired.append({'root_id': root, 'planned_pass_yield': values,
            'B_minus_A': (passed[ARMS[1]] - passed[ARMS[0]]) / 3, 'C_minus_B': (passed[ARMS[2]] - passed[ARMS[1]]) / 3,
            'C_minus_A': (passed[ARMS[2]] - passed[ARMS[0]]) / 3, 'paired_unit': 'root',
            'intent_ids_are_not_semantically_paired': True})
    return {'groups': groups, 'arms': arms, 'root_paired': paired,
        'two_roots_equal_weight': {k: sum(p[k] for p in paired) / len(ROOTS) for k in ('B_minus_A', 'C_minus_B', 'C_minus_A')},
        'statistical_scope': 'Two roots, one generation per root/arm; descriptive planned-slot yield only, no significance or causal-effect claim.'}


def preservation(run):
    histories = []
    for label in ('P1', 'V1_RUN', 'V2_RUN'):
        baseline = read(run / f'audit/{label}_inventory_before.json', {})
        folder = Path(baseline.get('root', '/nonexistent'))
        if 'files' not in baseline or not folder.is_dir():
            histories.append({'history': label, 'status': 'FAIL', 'reason': 'history baseline or directory missing'}); continue
        current = {str(p.relative_to(folder)): sha256_file(p) for p in sorted(folder.rglob('*')) if p.is_file()}
        old = baseline['files']
        histories.append({'history': label, 'root': str(folder), 'status': 'PASS' if current == old else 'FAIL',
            'before_files': len(old), 'after_files': len(current),
            'changed_or_missing': [k for k in old if current.get(k) != old[k]], 'added': sorted(set(current) - set(old))})
    result = {'status': 'PASS' if all(h['status'] == 'PASS' for h in histories) else 'FAIL', 'histories': histories, 'at_utc': now()}
    atomic_json(run / 'audit/history_preservation.json', result)
    return result


def prepare_review_views(run):
    mechanical = mechanical_report(Path(run))
    views = []
    mapping = []
    for number, item in enumerate(mechanical['items'], 1):
        view_id = f'V{number:03d}'
        views.append({'view_id': view_id, 'raw_intent': item['raw_intent'], 'original_intent': item['original_intent'],
            'derived_evidence': item['derived_evidence'], 'structurally_valid': item['structurally_valid'],
            'complete_transcript': read(item['frozen_transcript_path'], {})})
        mapping.append({'view_id': view_id, **{f: item[f] for f in ('root_id', 'arm_id', 'intent_index', 'intent_id', 'planned_slot')}})
    write_jsonl(Path(run) / 'reports/intent_review_views.jsonl', views)
    write_jsonl(Path(run) / 'audit/intent_review_view_mapping.jsonl', mapping)
    return mechanical


def final_report(run):
    run = Path(run)
    mechanical = mechanical_report(run); history = preservation(run)
    errors = []
    base, stage1_errors = assemble_intent_reviews(mechanical['items'], jsonl(run / 'reports/intent_evidence_review.jsonl'))
    errors.extend(stage1_errors)
    try:
        freeze_intent_review(run, mechanical)
        base = jsonl(run / 'reports/intent_evidence_review.jsonl')
    except (ValueError, OSError, KeyError) as error:
        errors.append(str(error))
    orphans = [{'root_id': g['root_id'], 'arm_id': g['arm_id'], 'raw_binding': b, 'binding_indices': [b['binding_index']]}
               for g in mechanical['groups'] for b in g['orphan_binding_rows']]
    if not errors:
        transfer, stage2_errors = assemble_transfer_reviews(mechanical['items'], base,
            jsonl(run / 'reports/transfer_judgments.jsonl'), orphans)
        errors.extend(stage2_errors)
        write_jsonl(run / 'reports/transfer_review.jsonl', transfer)
    else:
        transfer = []
    summary = summarize_layers(mechanical['items'], base, transfer)
    atomic_json(run / 'reports/root_paired_comparison.json', summary)
    required = ('test_report', 'prompt_pairing_report', 'shared_path_parity', 'preflight_report', 'token_report')
    checks = {name: read(run / 'audit' / (name + '.json'), {}).get('status') == 'PASS' for name in required}
    checks.update(mechanical=mechanical['status'] == 'PASS', history_preservation=history['status'] == 'PASS', review_coverage_and_order=not errors)
    findings = read(run / 'reports/diagnostic_findings.json', {})
    if not findings.get('reason') or not findings.get('DIAGNOSTIC_FINDINGS'):
        errors.append('authored_diagnostic_findings_missing')
    engineering = 'PASS' if all(checks.values()) else 'FAIL'
    result = {'version': VERSION, 'status': 'COMPLETED' if not errors else 'BLOCKED',
        'ENGINEERING_COMPARABILITY': engineering, 'DIAGNOSTIC_FINDINGS': findings.get('DIAGNOSTIC_FINDINGS', 'INCONCLUSIVE'),
        'diagnostic_reason': findings.get('reason', 'Authored diagnostic findings are missing.'),
        'checks': checks, 'review_errors': errors, 'summary': summary, 'accounting': mechanical['accounting'],
        'history_preservation': history, 'reviewer': REVIEWER, 'at_utc': now()}
    atomic_json(run / 'reports/diagnostic_summary.json', result)
    lines = ['# 两 root × 三臂 Miner 定位报告', '', f'ENGINEERING_COMPARABILITY：{engineering}。',
        f"DIAGNOSTIC_FINDINGS：{result['DIAGNOSTIC_FINDINGS']}。{result['diagnostic_reason']}", '',
        '| root | arm | 调用/完成 | returned/3 | 原有效 | PASS/FAIL/UNDETERMINED/未评 | 额外输出 |',
        '|---|---|---:|---:|---:|---|---:|']
    for group in summary['groups']:
        calls = next(g for g in mechanical['accounting']['groups'] if g['root_id'] == group['root_id'] and g['arm_id'] == group['arm_id'])
        lines.append(f"| {group['root_id']} | {group['arm_id']} | {calls['logical_calls']}/{calls['completed_calls']} | {group['returned']}/3 | {group['structurally_valid']} | {group['semantic_pass']}/{group['semantic_fail']}/{group['semantic_undetermined']}/{group['not_evaluated']} | {group['extra_outputs']} |")
    lines += ['', '第一层为全部 planned slots 与额外输出的 Codex direct transcript review；结构非法/缺失记未评。第一层文件在第二层前锁定，程序不判断证据语义。', '', '各臂计划分母为 6：']
    for arm in summary['arms']:
        lines.append(f"- {arm['arm_id']}：PASS {arm['semantic_pass']}/6；可审内容 PASS 比例 {arm['reviewable_pass_fraction']}；transfer {json.dumps(arm['transfer'], ensure_ascii=False)}")
    narrative = run / 'reports/diagnostic_narrative.md'
    if narrative.is_file():
        lines += ['', narrative.read_text()]
    lines += ['', 'root 配对差值（计划 slot PASS 产出率）：', '```json', json.dumps(summary['root_paired'], ensure_ascii=False, indent=2), '```',
        '两 root 等权描述：' + json.dumps(summary['two_roots_equal_weight'], ensure_ascii=False),
        '证据宽度逐项见 evidence_width_metrics.jsonl；scene 不等于 event。无可靠冻结 event map 时 event_count=null，未知 span=null；没有数量阈值、自动裁剪或语义筛选。',
        '调用与费用：' + json.dumps(mechanical['accounting'], ensure_ascii=False),
        'token 预检及实际 usage 见 audit/token_report.json；各请求输出上限相同不代表费用或解释负担相同。',
        'B−A 对应共同政策下整份观测包；C−B 对应完整迁移任务模块，不能单独归因为数字、解释负担或文本长度。两 root 各一次生成，18 intents 不是 18 个独立配对样本；不作显著性或稳定因果结论。',
        f"历史保全：{history['status']}；审阅错误：{errors}。",
        '本次 Writer、repair、Solver、Memory、训练和额外审核 API 均未调用。六次定位实验到此停止；后续仅给建议，不自动修 Miner、修改反馈、扩展运行或恢复 Writer。']
    atomic_text(run / 'reports/miner_three_arm_diagnostic.md', '\n\n'.join(lines) + '\n')
    atomic_text(run / 'RUN_STATUS.md', f"Report: {result['status']}\nENGINEERING_COMPARABILITY: {engineering}\nDIAGNOSTIC_FINDINGS: {result['DIAGNOSTIC_FINDINGS']}\nSTOPPED; no Writer/repair/Solver/Memory/training calls.\n")
    paths = ['execution_contract.json', 'implementation_manifest.json', 'pairing_manifest.jsonl', 'schedule.json', 'review_policy.json', 'input_path_audit.json',
        'audit/test_report.json', 'audit/prompt_pairing_report.json', 'audit/shared_path_parity.json', 'audit/preflight_report.json', 'audit/token_report.json',
        'audit/call_accounting.json', 'audit/history_preservation.json', 'audit/intent_review_freeze.json', 'audit/mechanical_report.json',
        'reports/intent_evidence_review.jsonl', 'reports/transfer_review.jsonl', 'reports/evidence_width_metrics.jsonl',
        'reports/root_paired_comparison.json', 'reports/diagnostic_summary.json', 'reports/miner_three_arm_diagnostic.md', 'RUN_STATUS.md']
    atomic_text(run / 'ARTIFACT_INDEX.md', '# Three-arm diagnostic artifacts\n\n' + ''.join(f'- [{p}]({p})\n' for p in paths) +
        '\nShared frozen inputs: inputs/shared/<root>/. Real raw and ledgers: arms/<arm>/qa/shards/shard-00/roots/<root>/. No QA candidates were generated.\n')
    return result
