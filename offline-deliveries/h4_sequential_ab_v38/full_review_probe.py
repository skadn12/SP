"""Two bounded full Reviewer calls over an immutable v37 scene.

The second case changes one required plan action only. Neither case generates
or admits dialogue. Prepare/verify/audit are offline; only live calls Qwen.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import fcntl
import json
from pathlib import Path

import runner as r

RUNTIME = Path(__file__).resolve().parent
ORIGIN = RUNTIME.parent / 'h4_sequential_ab_v37/experiments/fixed_soda_a3_01'
SOURCE_REQUEST = ORIGIN / 'requests/007-cc-train-episode-104814-B-S01-review-1-1'
VERSION = 'v38_fixed_full_review_state_scope_1'
HISTORY_SHA256 = 'fe7a1ab35ed3efb4fad16c0d81501e6b80d8cd3615ef344de465bc4dfa63e997'
EXECUTION_REQUIREMENT = 'James admits he is hungry, then actually eats and finishes the found sandwich during this scene.'
CASE_NAMES = ('original_decision_scene', 'execution_required_fixture')
POLICY = {'version': VERSION, 'model': r.MODEL, 'logical_call_max': 2, 'http_max': 4,
    'automatic_transport_retries': 1, 'minimum_http_gap_after_response_seconds': 35.0,
    'schema_retries': 0, 'semantic_retries': 0, 'new_scenes': 0,
    'new_source_consumption': 0, 'production_admission': False, 'training_pool_eligible': False,
    'question_calls': 0, 'solver_calls': 0, 'training_calls': 0, 'semantic_certified': False}


def _freeze(source, destination):
    data = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as out:
        out.write(data)
    return {'original_path': str(source), 'frozen_path': str(destination), 'sha256': r.base.digest(data)}


def _context(payload):
    return json.loads(payload['messages'][0]['content'].split('\nREVIEW INPUT DATA:\n', 1)[1])


def source_cases():
    original = _context(r.base.read(SOURCE_REQUEST / 'REQUEST.json'))
    card = r.base.read(ORIGIN / 'FROZEN_CARD.json')
    r.base.require(len(original['branch_history']) == 132 and
        r.base.digest(original['branch_history']) == HISTORY_SHA256, 'fixed_history_changed')
    cases = []
    for name in CASE_NAMES:
        context = deepcopy(original)
        if name == CASE_NAMES[1]:
            context['scene_plan']['developments'][2] = EXECUTION_REQUIREMENT
        cases.append({'name': name, 'inputs': context, 'card': card,
            'original_inputs_sha256': r.base.digest(original),
            'diagnostic_fixture_only': name == CASE_NAMES[1],
            'changed_paths': ['scene_plan.developments[2]'] if name == CASE_NAMES[1] else [],
            'manual_semantic_checks': ['summary_does_not_claim_consumption',
                'agreement_completion_not_rollout_execution',
                'real_installation_and_test_completion_preserved',
                'consideration_card_step_not_rejected_for_absent_eating'],
            'completion_positive_controls': [
                {'fact_id': row['fact_id'], 'statement': row['statement'],
                 'evidence_turn_refs': [span['turn_ref'] for span in row['evidence']]}
                for row in context['latest_state']['facts'] if row.get('fact_id') in ('f3', 'f4')],
            'overall_pass_required': False})
    return cases


def make_request(case):
    c = case['inputs']
    payload, proof = r.review_request(c['root_id'], c['route'], c['scene_plan'],
        c['branch_history'], c['new_scene'], c['other_branch_added_turns_for_review_only'],
        card=case['card'], latest_state=c['latest_state'], event_state=c['event_state'],
        purpose_input=c.get('purpose_input'))
    r.base.require(_context(payload) == c, 'full_review_context_changed')
    proof.update(probe_version=VERSION, case_name=case['name'], case_sha256=r.base.digest(case),
        original_inputs_sha256=case['original_inputs_sha256'], changed_paths=case['changed_paths'],
        diagnostic_fixture_only=case['diagnostic_fixture_only'], semantic_certified=False)
    return payload, proof


def evaluate(case, raw):
    c = case['inputs']
    checked = r.realization_review.validate_review(raw, c['root_id'], c['route'],
        c['scene_plan'], c['new_scene'], dataset_card=case['card'], latest_state=c['latest_state'],
        event_state=c['event_state'], history=c['branch_history'],
        require_actor_checks=True, require_goal_checks=True)
    parsed = checked.get('raw_parsed') or {}
    developments = {str(row.get('development_index')): row for row in parsed.get('checks', [])
        if isinstance(row, dict)}
    cards = {str(row.get('kind')) + ':' + str(row.get('id')): row
        for row in parsed.get('card_checks', []) if isinstance(row, dict)}
    d3 = developments.get('3', {})
    actor_rows = d3.get('actor_checks', [])
    observed = {'development_3_realized': d3.get('realized'),
        'development_3_actor_satisfied': [row.get('satisfied') for row in actor_rows if isinstance(row, dict)],
        'card_s1_realized': cards.get('step:s1', {}).get('realized')}
    negative_rejected = (checked['schema_valid'] and d3.get('realized') is False and
        bool(actor_rows) and any(row.get('satisfied') is False for row in actor_rows if isinstance(row, dict)))
    return {'status': checked['status'], 'schema_valid': checked['schema_valid'],
        'observed': observed, 'execution_fixture_rejected': negative_rejected if case['diagnostic_fixture_only'] else None,
        'completion_positive_controls': deepcopy(case['completion_positive_controls']),
        'completion_positive_controls_source': 'same_history.latest_state.f3_f4',
        'review': checked, 'raw_response_sha256': r.base.digest(raw), 'semantic_certified': False,
        'manual_semantic_review_required': True, 'overall_pass_required': False}


def _error_result(error):
    return {'status': 'TRANSPORT_OR_LOCAL_ERROR', 'schema_valid': False, 'reason': str(error),
        'semantic_certified': False, 'manual_semantic_review_required': True, 'overall_pass_required': False}


def _report(rows, logical, http):
    return {'status': 'COMPLETE_REQUIRES_INDEPENDENT_SEMANTIC_REVIEW',
        'cases': rows, 'logical_calls': logical, 'http_attempts': http,
        'new_scenes': 0, 'production_admission': False, 'training_pool_eligible': False,
        'semantic_certified': False}


def prepare(run):
    r.base.require(not run.exists() and run.parent == RUNTIME / 'probes', 'probe_path_invalid_or_exists')
    r.base.require(r.MIN_HTTP_GAP_SECONDS == 35 and r.TRANSPORT_RETRIES == 1, 'transport_policy_changed')
    cases = source_cases()
    previews = [make_request(case) for case in cases]
    run.mkdir(parents=True)
    sources = [SOURCE_REQUEST / name for name in ('REQUEST.json', 'RESPONSE.json')]
    sources += [ORIGIN / name for name in ('FROZEN_CARD.json', 'FROZEN_SOURCE.json', 'MANIFEST.json')]
    frozen = [_freeze(path, run / 'frozen_sources' / path.relative_to(ORIGIN)) for path in sources]
    for case, (payload, proof) in zip(cases, previews):
        r.base.write(run / 'cases' / (case['name'] + '.json'), case)
        r.base.write(run / 'previews' / (case['name'] + '.json'), {'payload': payload, 'proof': proof})
    r.base.write(run / 'SOURCE_SNAPSHOT.json', {'files': frozen})
    profile, transport, tokenizer = r.transport_inputs()
    r.base.write(run / 'provider_profile.json', profile)
    code = set(RUNTIME.glob('*.py')) | set(r.LEGACY.glob('*.py')) | set(transport)
    snapshots = [_freeze(path, run / 'execution_code' / f'{i:03d}_{path.name}')
        for i, path in enumerate(sorted(code))]
    r.base.write(run / 'EXECUTION_CODE_SNAPSHOT.json', {'files': snapshots})
    paths = set(sources) | code | set(tokenizer) | {RUNTIME / 'POLICY.md'}
    paths.update(p for p in run.rglob('*') if p.is_file())
    r.base.write(run / 'MANIFEST.json', {**POLICY, 'prepared_at_utc': r.now(),
        'files': r.base.bindings(sorted(paths))})
    return {'status': 'PREPARED', 'logical_call_max': 2, 'http_max': 4, 'http_calls': 0}


def verify(run):
    manifest = r.base.read(run / 'MANIFEST.json')
    for key, value in POLICY.items():
        r.base.require(manifest.get(key) == value, 'manifest_policy_changed:' + key)
    r.base.verify_bindings(manifest['files'])
    for case in source_cases():
        r.base.require(r.base.read(run / 'cases' / (case['name'] + '.json')) == case, 'frozen_case_changed')
        payload, proof = make_request(case)
        r.base.require(r.base.read(run / 'previews' / (case['name'] + '.json')) ==
            {'payload': payload, 'proof': proof}, 'request_preview_changed')
    return manifest


def live(run):
    manifest = verify(run)
    r.base.require(not (run / 'LIVE_STARTED.json').exists(), 'probe_cannot_restart')
    r.LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with r.LOCK_PATH.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r.base.write(run / 'LIVE_STARTED.json', {'manifest_sha256': r.base.digest(manifest), 'started_at_utc': r.now()})
        caller, rows = r.Caller(run), []
        for number, name in enumerate(CASE_NAMES, 1):
            case = r.base.read(run / 'cases' / (name + '.json'))
            payload, proof = make_request(case)
            directory = run / 'requests' / f'{number:03d}-full-review-{name}'
            try:
                raw, actual = caller.call('full-review-' + name, payload, proof)
                r.base.require(actual == directory, 'request_directory_changed')
                result = evaluate(case, raw)
            except (RuntimeError, ValueError) as error:
                result = _error_result(error)
            r.base.write(directory / 'EVALUATION.json', result)
            rows.append({'case': name, 'status': result['status'], 'schema_valid': result['schema_valid'],
                'evaluation_sha256': r.base.digest(result)})
            print(json.dumps(rows[-1]), flush=True)
        report = _report(rows, caller.request_counter, caller.http_counter)
        report['completed_at_utc'] = r.now()
        r.base.write(run / 'REPORT.json', report)
        verify(run)
        return report


def audit(run):
    failures = []
    def check(ok, code):
        if not ok: failures.append(code)
    try:
        manifest = verify(run)
        directories = sorted(p for p in (run / 'requests').glob('*') if p.is_dir())
        started = (run / 'LIVE_STARTED.json').exists()
        check(len(directories) == (2 if started else 0), 'logical_request_inventory_invalid')
        if started:
            check(r.base.read(run / 'LIVE_STARTED.json')['manifest_sha256'] == r.base.digest(manifest), 'live_manifest_changed')
        http_count, offline, rows, previous_end = 0, False, [], None
        for number, directory in enumerate(directories, 1):
            name = CASE_NAMES[number - 1]
            case = r.base.read(run / 'cases' / (name + '.json'))
            payload, proof = make_request(case)
            check(directory.name == f'{number:03d}-full-review-{name}', 'request_label_changed')
            check(r.base.read(directory / 'REQUEST.json') == payload, 'request_replay_changed')
            check(r.base.read(directory / 'RESERVATION.json') == {'request_number': number, 'http_max': 4,
                'max_attempts': 2, 'retryable_statuses': [503, 504], 'status': 'RESERVED'}, 'reservation_changed')
            http = sorted(directory.glob('HTTP_ATTEMPT*.json'))
            check(len(http) <= 2, 'http_request_budget_exceeded')
            for index, path in enumerate(http, 1):
                http_count += 1
                entry = r.base.read(path)
                result = r.base.read(directory / f'HTTP_RESULT_{index:02d}.json')
                throttle = r.base.read(directory / f'THROTTLE_{index:02d}.json')
                check(entry.get('request_attempt') == index and entry.get('http_attempt_number') == http_count
                    and entry.get('actual_payload_equals_frozen') is True
                    and entry.get('payload_sha256') == r.base.digest(payload)
                    and entry.get('requested_model') == r.MODEL, 'http_payload_or_sequence_changed')
                check(throttle.get('minimum_gap_after_previous_response_seconds') == 35, 'http_gap_policy_changed')
                began = datetime.fromisoformat(throttle['http_started_at_utc'])
                ended = datetime.fromisoformat(result['http_completed_at_utc'])
                check(ended >= began, 'http_completion_before_start')
                if previous_end is not None:
                    check((began - previous_end).total_seconds() >= 34.9, 'http_gap_under_minimum')
                if index > 1:
                    prior = r.base.read(directory / f'HTTP_RESULT_{index-1:02d}.json')
                    check(entry.get('same_payload_retry') is True and prior.get('outcome') == 'error'
                        and prior.get('http_status') in (503, 504), 'http_retry_policy_changed')
                previous_end = ended
            if (directory / 'PROOF.json').exists():
                recorded = r.base.read(directory / 'PROOF.json')
                check(all(recorded.get(k) == v for k, v in proof.items()) and
                    recorded.get('payload_sha256') == r.base.digest(payload), 'request_proof_changed')
            if (directory / 'RESPONSE.json').exists():
                check(bool(http) and (directory / 'PROOF.json').exists(), 'response_without_http_or_proof')
                response = r.base.read(directory / 'RESPONSE.json')
                metadata = response['metadata']
                offline = offline or metadata.get('offline_test_fake_transport') is True
                check(metadata.get('returned_model') == r.MODEL and metadata.get('finish_reason') == 'stop', 'response_model_or_completion_changed')
                check(r.base.read(directory / f'HTTP_RESPONSE_{len(http):02d}.json') ==
                    {'text': response['text'], 'metadata': metadata}, 'response_not_final_http_body')
                final = r.base.read(directory / f'HTTP_RESULT_{len(http):02d}.json')
                check(final.get('outcome') == 'response' and final.get('http_status') in (None, 200), 'successful_response_transport_changed')
                expected = evaluate(case, response['text'])
            else:
                local, transport = directory / 'LOCAL_ERROR.json', directory / 'ERROR.json'
                check(local.exists() or transport.exists(), 'error_receipt_missing')
                reason = r.base.read(local)['error_code'] if local.exists() else 'provider_request_failed_after_retry_policy'
                expected = _error_result(reason)
            check(r.base.read(directory / 'EVALUATION.json') == expected, 'evaluation_replay_changed')
            rows.append({'case': name, 'status': expected['status'], 'schema_valid': expected['schema_valid'],
                'evaluation_sha256': r.base.digest(expected)})
        check(http_count <= 4, 'global_http_budget_exceeded')
        if started:
            report = r.base.read(run / 'REPORT.json')
            check({k: v for k, v in report.items() if k != 'completed_at_utc'} == _report(rows, 2, http_count), 'report_replay_changed')
        result = {'status': 'PASS' if not failures else 'FAIL', 'failures': failures,
            'logical_calls': len(directories), 'http_attempts': http_count,
            'offline_test_fake_transport': offline, 'semantic_certified': False, 'new_scenes': 0,
            'production_admission': False, 'training_pool_eligible': False, 'audited_at_utc': r.now()}
    except (ValueError, KeyError, TypeError, IndexError, FileNotFoundError) as error:
        result = {'status': 'FAIL', 'failures': failures + ['audit_exception:' + str(error)], 'semantic_certified': False}
    target, index = run / 'FULL_REVIEW_AUDIT.json', 2
    while target.exists():
        target = run / f'FULL_REVIEW_AUDIT_{index:03d}.json'; index += 1
    r.base.write(target, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'verify', 'live', 'audit'))
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    result = globals()[args.action](args.run.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get('status') == 'FAIL': raise SystemExit(1)
