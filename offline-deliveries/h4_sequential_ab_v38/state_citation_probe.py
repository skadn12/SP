"""One bounded Qwen citation patch of the immutable v36 failed state snapshot.

No source allocation, scene generation, training admission or full-state rewrite.
Prepare, verify and audit are offline; only live invokes the model once.
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
ORIGIN = RUNTIME.parent / 'h4_sequential_ab_v36/experiments/fixed_soda_a3_01'
SOURCE_REQUEST = ORIGIN / 'requests/002-cc-train-episode-104814-B-S01-state-1'
VERSION = 'v37_fixed_state_citation_probe_1'
LABEL = 'fixed-v36-state-citation-patch'
REQUIRED_PATHS = ['facts[1].evidence[0].quote', 'facts[4].evidence[0].turn_ref',
    'facts[4].evidence[1].turn_ref', 'facts[5].evidence[0].turn_ref', 'facts[5].evidence[1].turn_ref']
HISTORY_SHA256 = 'fe7a1ab35ed3efb4fad16c0d81501e6b80d8cd3615ef344de465bc4dfa63e997'
POLICY = {'version': VERSION, 'model': r.MODEL, 'logical_call_max': 1, 'http_max': 2,
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


def source_case():
    request = r.base.read(SOURCE_REQUEST / 'REQUEST.json')
    context = json.loads(request['messages'][0]['content'].split('\nSTATE_INPUT_JSON\n')[1])
    raw = r.base.read(SOURCE_REQUEST / 'RESPONSE.json')['text']
    card = r.base.read(ORIGIN / 'FROZEN_CARD.json')
    r.base.require(len(context['cumulative_history']) == 132 and
        r.base.digest(context['cumulative_history']) == HISTORY_SHA256, 'fixed_history_changed')
    r.base.require(context['dataset_cards'] == [r.planner_contract.mapping_context(card)['card_guidance']],
        'fixed_card_projection_changed')
    try:
        r.state_contract.validate_state(raw, context['root_id'], context['route'],
            context['scene_id'], context['cumulative_history'])
    except r.state_contract.StateValidationError as error:
        recorded = r.base.read(SOURCE_REQUEST / 'VALIDATION.json')
        r.base.require(recorded == {'status': 'REJECT', 'reason': str(error), 'errors': error.errors},
            'fixed_validation_changed')
        repair = r.state_contract.build_state_repair(error, raw, context['cumulative_history'],
            require_local_citation_patch=True)
    else:
        raise ValueError('fixed_response_no_longer_failed')
    r.base.require(repair['repair_mode'] == r.state_contract.STATE_CITATION_PATCH_VERSION and
        repair['required_paths'] == REQUIRED_PATHS and not repair['unrepairable_failures'],
        'fixed_patch_scope_changed')
    return {'original_context': context, 'card': card, 'previous_raw': raw, 'repair': repair}


def make_request(case):
    c = case['original_context']
    payload = r.state_contract.build_state_request(c['root_id'], c['route'], c['scene_id'],
        c['cumulative_history'], [case['card']], repair=case['repair'], purpose_input=c['purpose_input'])
    sent = json.loads(payload['messages'][0]['content'].split('\nSTATE_INPUT_JSON\n')[1])
    r.base.require({k: v for k, v in sent.items() if k != 'targeted_repair'} == c,
        'original_context_not_preserved')
    r.base.require(sent['targeted_repair'] == case['repair'], 'patch_spec_changed')
    proof = {'probe_version': VERSION, 'role': 'fixed_state_citation_patch',
        'root_id': c['root_id'], 'route': c['route'], 'scene_id': c['scene_id'],
        'history_sha256': r.base.digest(c['cumulative_history']), 'history_turns': 132,
        'previous_raw_sha256': r.base.digest(case['previous_raw']),
        'repair_spec_sha256': r.base.digest(case['repair']), 'case_sha256': r.base.digest(case),
        'required_paths': REQUIRED_PATHS, 'full_state_rewrite_allowed': False,
        'semantic_certified': False}
    return payload, proof


def evaluate(case, raw):
    c = case['original_context']
    result = {'previous_raw_sha256': r.base.digest(case['previous_raw']),
        'raw_patch_sha256': r.base.digest(raw), 'repair_spec_sha256': r.base.digest(case['repair']),
        'required_paths': REQUIRED_PATHS, 'semantic_certified': False}
    try:
        candidate = r.state_contract.merge_state_repair(case['previous_raw'], raw,
            case['repair'], c['cumulative_history'])
        snapshot = r.state_contract.validate_state(candidate, c['root_id'], c['route'],
            c['scene_id'], c['cumulative_history'])
    except (ValueError, TypeError, KeyError) as error:
        return {**result, 'status': 'REJECT', 'reason': str(error),
            'errors': deepcopy(getattr(error, 'errors', []))}
    return {**result, 'status': 'CITATION_STRUCTURE_PASS', 'candidate': snapshot,
        'candidate_sha256': r.base.digest(snapshot), 'changed_paths': REQUIRED_PATHS}


def prepare(run):
    r.base.require(not run.exists() and run.parent == RUNTIME / 'probes', 'probe_path_invalid_or_exists')
    r.base.require(r.MIN_HTTP_GAP_SECONDS == 35 and r.TRANSPORT_RETRIES == 1, 'transport_policy_changed')
    case = source_case()
    payload, proof = make_request(case)
    run.mkdir(parents=True)
    source_files = [SOURCE_REQUEST / name for name in ('REQUEST.json', 'RESPONSE.json', 'VALIDATION.json')]
    source_files += [ORIGIN / name for name in ('FROZEN_CARD.json', 'FROZEN_SOURCE.json', 'MANIFEST.json')]
    sources = [_freeze(path, run / 'frozen_sources' / path.relative_to(ORIGIN)) for path in source_files]
    r.base.write(run / 'CASE.json', case)
    r.base.write(run / 'REQUEST_PREVIEW.json', {'payload': payload, 'proof': proof})
    r.base.write(run / 'SOURCE_SNAPSHOT.json', {'files': sources})
    profile, transport, tokenizer = r.transport_inputs()
    r.base.write(run / 'provider_profile.json', profile)
    code = set(RUNTIME.glob('*.py')) | set(r.LEGACY.glob('*.py')) | set(transport)
    snapshots = [_freeze(path, run / 'execution_code' / f'{i:03d}_{path.name}')
        for i, path in enumerate(sorted(code))]
    r.base.write(run / 'EXECUTION_CODE_SNAPSHOT.json', {'files': snapshots})
    paths = set(source_files) | code | set(tokenizer) | {RUNTIME / 'POLICY.md'}
    paths.update(p for p in run.rglob('*') if p.is_file())
    r.base.write(run / 'MANIFEST.json', {**POLICY, 'prepared_at_utc': r.now(),
        'files': r.base.bindings(sorted(paths))})
    return {'status': 'PREPARED', 'logical_call_max': 1, 'http_max': 2, 'http_calls': 0}


def verify(run):
    manifest = r.base.read(run / 'MANIFEST.json')
    for key, value in POLICY.items():
        r.base.require(manifest.get(key) == value, 'manifest_policy_changed:' + key)
    r.base.verify_bindings(manifest['files'])
    case = source_case()
    r.base.require(r.base.read(run / 'CASE.json') == case, 'frozen_case_changed')
    payload, proof = make_request(case)
    r.base.require(r.base.read(run / 'REQUEST_PREVIEW.json') == {'payload': payload, 'proof': proof},
        'request_preview_changed')
    return manifest


def _error_result(error):
    return {'status': 'TRANSPORT_OR_LOCAL_ERROR', 'reason': str(error), 'semantic_certified': False}


def live(run):
    manifest = verify(run)
    r.base.require(not (run / 'LIVE_STARTED.json').exists(), 'probe_cannot_restart')
    r.LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with r.LOCK_PATH.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r.base.write(run / 'LIVE_STARTED.json', {'manifest_sha256': r.base.digest(manifest), 'started_at_utc': r.now()})
        caller = r.Caller(run)
        case = r.base.read(run / 'CASE.json')
        payload, proof = make_request(case)
        directory = run / 'requests' / ('001-' + LABEL)
        try:
            raw, actual = caller.call(LABEL, payload, proof)
            r.base.require(actual == directory, 'request_directory_changed')
            result = evaluate(case, raw)
        except (RuntimeError, ValueError) as error:
            result = _error_result(error)
        r.base.write(directory / 'EVALUATION.json', result)
        report = {'status': result['status'], 'evaluation_sha256': r.base.digest(result),
            'logical_calls': caller.request_counter, 'http_attempts': caller.http_counter,
            'new_scenes': 0, 'production_admission': False, 'training_pool_eligible': False,
            'semantic_certified': False, 'completed_at_utc': r.now()}
        r.base.write(run / 'REPORT.json', report)
        verify(run)
        return report


def audit(run):
    failures = []
    def check(ok, code):
        if not ok: failures.append(code)
    try:
        manifest = verify(run)
        case = r.base.read(run / 'CASE.json')
        payload, proof = make_request(case)
        directories = sorted(p for p in (run / 'requests').glob('*') if p.is_dir())
        started = (run / 'LIVE_STARTED.json').exists()
        check(len(directories) == int(started), 'logical_request_inventory_invalid')
        if started:
            check(r.base.read(run / 'LIVE_STARTED.json')['manifest_sha256'] == r.base.digest(manifest),
                'live_manifest_changed')
        http_count, offline, status = 0, False, 'PREPARED_NO_LIVE_CALLS'
        for directory in directories:
            check(directory.name == '001-' + LABEL, 'request_label_changed')
            check(r.base.read(directory / 'REQUEST.json') == payload, 'request_replay_changed')
            reservation = r.base.read(directory / 'RESERVATION.json')
            check(reservation == {'request_number': 1, 'http_max': 2, 'max_attempts': 2,
                'retryable_statuses': [503, 504], 'status': 'RESERVED'}, 'reservation_changed')
            http = sorted(directory.glob('HTTP_ATTEMPT*.json'))
            check(len(http) <= 2, 'http_budget_exceeded')
            http_count += len(http)
            previous_end = None
            for index, path in enumerate(http, 1):
                entry = r.base.read(path)
                result = r.base.read(directory / f'HTTP_RESULT_{index:02d}.json')
                throttle = r.base.read(directory / f'THROTTLE_{index:02d}.json')
                check(entry.get('request_attempt') == index and entry.get('http_attempt_number') == index
                    and entry.get('actual_payload_equals_frozen') is True
                    and entry.get('payload_sha256') == r.base.digest(payload)
                    and entry.get('requested_model') == r.MODEL, 'http_payload_or_sequence_changed')
                check(throttle.get('minimum_gap_after_previous_response_seconds') == 35,
                    'http_gap_policy_changed')
                began = datetime.fromisoformat(throttle['http_started_at_utc'])
                ended = datetime.fromisoformat(result['http_completed_at_utc'])
                check(ended >= began, 'http_completion_before_start')
                if index > 1:
                    prior = r.base.read(directory / f'HTTP_RESULT_{index-1:02d}.json')
                    check(entry.get('same_payload_retry') is True and prior.get('outcome') == 'error'
                        and prior.get('http_status') in (503, 504), 'http_retry_policy_changed')
                    check((began - previous_end).total_seconds() >= 34.9, 'http_gap_under_minimum')
                previous_end = ended
            if (directory / 'PROOF.json').exists():
                recorded = r.base.read(directory / 'PROOF.json')
                check(all(recorded.get(k) == v for k, v in proof.items()) and
                    recorded.get('payload_sha256') == r.base.digest(payload), 'request_proof_changed')
            if (directory / 'RESPONSE.json').exists():
                check(bool(http) and (directory / 'PROOF.json').exists(), 'response_without_http_or_proof')
                response = r.base.read(directory / 'RESPONSE.json')
                metadata = response['metadata']
                offline = metadata.get('offline_test_fake_transport') is True
                check(metadata.get('returned_model') == r.MODEL and metadata.get('finish_reason') == 'stop',
                    'response_model_or_completion_changed')
                check(r.base.read(directory / f'HTTP_RESPONSE_{len(http):02d}.json') ==
                    {'text': response['text'], 'metadata': metadata}, 'response_not_final_http_body')
                final_http = r.base.read(directory / f'HTTP_RESULT_{len(http):02d}.json')
                check(final_http.get('outcome') == 'response' and final_http.get('http_status') in (None, 200),
                    'successful_response_transport_changed')
                expected = evaluate(case, response['text'])
            else:
                local = directory / 'LOCAL_ERROR.json'
                transport = directory / 'ERROR.json'
                check(local.exists() or transport.exists(), 'error_receipt_missing')
                reason = r.base.read(local)['error_code'] if local.exists() else 'provider_request_failed_after_retry_policy'
                expected = _error_result(reason)
            check(r.base.read(directory / 'EVALUATION.json') == expected, 'merge_evaluation_replay_changed')
            status = expected['status']
            report = r.base.read(run / 'REPORT.json')
            check(report.get('status') == status and report.get('evaluation_sha256') == r.base.digest(expected)
                and report.get('logical_calls') == 1 and report.get('http_attempts') == http_count,
                'report_replay_changed')
            check(all(report.get(k) == v for k, v in {'new_scenes': 0, 'production_admission': False,
                'training_pool_eligible': False, 'semantic_certified': False}.items()), 'report_scope_changed')
        result = {'status': 'PASS' if not failures else 'FAIL', 'failures': failures,
            'generation_status': status, 'logical_calls': len(directories), 'http_attempts': http_count,
            'offline_test_fake_transport': offline, 'semantic_certified': False, 'new_scenes': 0,
            'production_admission': False, 'training_pool_eligible': False, 'audited_at_utc': r.now()}
    except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
        result = {'status': 'FAIL', 'failures': failures + ['audit_exception:' + str(error)],
            'semantic_certified': False}
    target, index = run / 'STATE_CITATION_AUDIT.json', 2
    while target.exists():
        target = run / f'STATE_CITATION_AUDIT_{index:03d}.json'
        index += 1
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
