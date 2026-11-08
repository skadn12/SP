"""Three frozen source-card mapping diagnostics; prepare/audit are offline.

Only live invokes Qwen. Each case permits one logical request and at most one
same-payload 503/504 transport retry. There is no semantic or schema retry.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

RUNTIME = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNTIME))
import runner as r
import card_grounding as grounding

DELIVERIES = RUNTIME.parent
PARENT = DELIVERIES / 'h4_sequential_ab_v35/probes/frozen_contract_comparison_01'
VERSION = 'v36_frozen_mapping_only_probe_1'
CASES = (
    ('case_01', 'fresh_v35_s02', {'step:s1': 'revise'}),
    ('case_02', 'fresh_v35_s03', {'step:s1': 'revise', 'step:s2': 'revise',
        'step:s3': 'compatible', 'step:s4': 'compatible',
        'dependency:1': 'revise', 'dependency:2': 'revise', 'dependency:3': 'compatible'}),
    ('case_03', 'control_v28_s01_independent', {'step:s1': 'compatible',
        'step:s2': 'compatible', 'step:s3': 'compatible',
        'dependency:1': 'compatible', 'dependency:2': 'compatible'}),
)
PROMPT = """Review ONLY the supplied frozen plan's selected source-card mappings.
The original plan and its development links are immutable. Do not rewrite the
plan, move links, repair missing work, or generate dialogue. Full cumulative
accepted history is supplied to identify participants and constrain identity;
this task does not evaluate old goals, novelty, chronology, or overall B success.

Return one JSON object with exactly root_id, route, scene_id,
card_actor_bindings, card_plan_checks. For each required source actor choose the
participant represented by that role in the actual plan. Return exactly one
{actor_id,participant} binding for each required_actor_id. This is your mapping
proposal, not evidence that the participant performs the source action. A
participant may be mentioned without speaking. Do not infer that a reporting
speaker performed the reported action. Preserve actor/action/object/state.

card_plan_checks contains exactly one row per selected_card_checks entry:
{kind,id,verdict,reason,correction,evidence_development_indices}.
verdict is compatible or revise. reason is a concrete comparison with current
linked work. correction is empty for compatible; for revise state the required
change without treating that change as already present. Evidence indices are
1-based integers; use [] if the required action is absent. For a compatible
step cite at least one actual linked development. Only that step's mapped
developments may demonstrate its action. Other parts of the plan can reveal a
contradiction but cannot fill missing linked work. Sharing a theme/object,
a merely related action, or another participant's action cannot satisfy the
selected actor's action. Renaming participants and
adapting the setting are allowed if the selected interaction is preserved.

For a dependency assess its two selected endpoint actions and the relation
between them in the current plan. Both endpoint steps must be compatible for
the dependency to be compatible. Cite evidence covering each endpoint; the
permitted indices are supplied in dependency_mappings. If an endpoint is
missing, incorrect, or would only exist after a proposed correction, revise
the current dependency. Do not assume a correction has happened.

Return only the JSON object. No historical scanner verdicts or expected
answers are supplied. This diagnostic does not authorize training admission.
"""


def _freeze(source, destination):
    source, destination = Path(source), Path(destination)
    data = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        r.base.require(destination.read_bytes() == data, 'frozen_copy_changed')
    else:
        with destination.open('xb') as out:
            out.write(data)
    return {'original_path': str(source), 'frozen_path': str(destination), 'sha256': r.base.digest(data)}


def source_cases():
    cases = []
    for name, parent_name, expected in CASES:
        source = PARENT / 'cases' / (parent_name + '.json')
        original = r.base.read(source)
        c = deepcopy(original['inputs'])
        r.base.require(c['history'] == original['original_context']['cumulative_history'] and
            r.base.digest(c['history']) == original['original_history_sha256'], 'source_full_history_mismatch')
        for binding in original['source_bindings']:
            for key in ('original_path', 'frozen_path'):
                r.base.require(r.base.digest(Path(binding[key]).read_bytes()) == binding['sha256'],
                               'parent_source_binding_changed:' + binding[key])
        case = {'name': name, 'inputs': c, 'expected_checks': expected,
                'expected_status': 'REJECT' if 'revise' in expected.values() else 'PASS',
                'original_history_sha256': r.base.digest(c['history']),
                'original_scene_plan_sha256': r.base.digest(c['scene_plan']),
                'parent_case': str(source), 'parent_case_sha256': r.base.digest(source.read_bytes()),
                'source_bindings': original['source_bindings'], 'semantic_certified': False}
        cases.append(case)
    return cases


def input_mapping(case):
    """Dereference original links before any model actor bindings exist."""
    c = case['inputs']
    checks = r.event_contract.expected_card_plan_checks(c['scene_plan'], [c['dataset_card']])
    actors, steps, dependencies = set(), {}, []
    for row in checks:
        if row['kind'] == 'step':
            actors.add(row['source']['actor_id'])
            steps[row['id']] = {'kind': 'step', 'id': row['id'],
                'development_indices': [d['development_index'] for d in row['linked_developments']]}
        else:
            actors.update(row['source'][side + '_step']['actor_id'] for side in ('before', 'after'))
    for row in checks:
        if row['kind'] == 'dependency':
            source = row['source']
            dependencies.append({'kind': 'dependency', 'id': row['id'],
                'before_step_id': source['before'], 'after_step_id': source['after'],
                'before_development_indices': steps.get(source['before'], {}).get('development_indices', []),
                'after_development_indices': steps.get(source['after'], {}).get('development_indices', []),
                'development_indices': [d['development_index'] for d in row['linked_developments']]})
    return {'required_actor_ids': sorted(actors), 'steps': list(steps.values()),
            'dependencies': dependencies, 'selected_card_checks': checks}


def make_request(case):
    c = case['inputs']
    scene, card = c['scene_plan'], c['dataset_card']
    # This dereferences immutable links, never chooses a participant or verdict.
    dereferenced = input_mapping(case)
    context = {'root_id': c['root_id'], 'route': c['route'], 'scene_id': scene['scene_id'],
        'scene_plan': deepcopy(scene), 'dataset_card': deepcopy(card),
        'cumulative_history': deepcopy(c['history']),
        'required_actor_ids': dereferenced['required_actor_ids'],
        'selected_card_checks': dereferenced['selected_card_checks'],
        'dependency_mappings': dereferenced['dependencies']}
    payload = {'model': r.MODEL, 'temperature': .25, 'max_tokens': 4096, 'enable_thinking': False,
        'response_format': {'type': 'json_object'}, 'messages': [{'role': 'user', 'content':
        PROMPT + '\nMAPPING_INPUT_JSON\n' + json.dumps(context, ensure_ascii=False, separators=(',', ':'))}]}
    proof = {'probe_version': VERSION, 'case': case['name'], 'root_id': c['root_id'],
        'route': c['route'], 'scene_id': scene['scene_id'], 'role': 'frozen_mapping_only_diagnostic',
        'history_sha256': r.base.digest(c['history']), 'history_turns': len(c['history']),
        'scene_plan_sha256': r.base.digest(scene), 'dataset_card_sha256': r.base.digest(card),
        'frozen_inputs_sha256': r.base.digest(c), 'expected_answers_withheld': True,
        'original_plan_immutable': True, 'bindings_generated_by_model': True,
        'schema_retry': False, 'semantic_retry': False, 'semantic_certified': False}
    return payload, proof


def evaluate(case, raw):
    c = case['inputs']
    errors = []
    try:
        candidate = r.parse_object(raw)
        fields = {'root_id', 'route', 'scene_id', 'card_actor_bindings', 'card_plan_checks'}
        if set(candidate) != fields:
            errors.append({'code': 'response_fields_invalid', 'allowed': sorted(fields)})
        for key, expected in (('root_id', c['root_id']), ('route', c['route']),
                              ('scene_id', c['scene_plan']['scene_id'])):
            if candidate.get(key) != expected:
                errors.append({'code': key + '_mismatch'})
        projected = deepcopy(c['scene_plan'])
        projected['card_actor_bindings'] = deepcopy(candidate.get('card_actor_bindings'))
        errors.extend(grounding.collect_mapping_grounding_errors(projected, c['dataset_card']))
        errors.extend(r.event_contract.validate_card_plan_checks(candidate.get('card_plan_checks'), c['route'],
            projected, [c['dataset_card']], required=True, require_grounding=True))
        # Never infer a model verdict; this merely summarizes returned values.
        judgments = {str(row.get('kind')) + ':' + str(row.get('id')): row.get('verdict')
                     for row in candidate.get('card_plan_checks', []) if isinstance(row, dict)}
        status = 'FORMAT_ERROR' if errors else ('REJECT' if 'revise' in judgments.values() else 'PASS')
        return {'status': status, 'schema_valid': not errors, 'errors': errors, 'candidate': candidate,
            'checks': judgments, 'test_projection_sha256': r.base.digest(projected),
            'original_scene_plan_sha256': r.base.digest(c['scene_plan']), 'semantic_certified': False}
    except (ValueError, TypeError, KeyError) as exc:
        return {'status': 'FORMAT_ERROR', 'schema_valid': False, 'error': str(exc),
                'errors': errors, 'checks': {}, 'semantic_certified': False}


def summary(case, checked, directory, transport_error=None):
    return {'case': case['name'], 'status': checked['status'], 'schema_valid': checked['schema_valid'],
        'expected_match': checked['schema_valid'] and checked.get('checks') == case['expected_checks'],
        'expected_status': case['expected_status'], 'expected_checks': case['expected_checks'],
        'observed_checks': checked.get('checks', {}), 'request': str(directory),
        'transport_error': transport_error, 'evaluation_sha256': r.base.digest(checked),
        'semantic_certified': False}


def prepare(run):
    r.base.require(not run.exists(), 'probe_already_exists')
    r.base.require(run.parent == RUNTIME / 'probes', 'probe_must_be_in_v36_probes')
    r.base.require(r.MIN_HTTP_GAP_SECONDS >= 35 and r.TRANSPORT_RETRIES == 1, 'transport_policy_changed')
    run.mkdir(parents=True)
    cases = source_cases()
    sources = {Path(case['parent_case']) for case in cases}
    for case in cases:
        sources.update(Path(row[key]) for row in case['source_bindings'] for key in ('original_path', 'frozen_path'))
    origin = DELIVERIES / 'h4_sequential_ab_v34/runs/inherited_a3_b3_01'
    sources.update(origin / filename for filename in ('dataset_cards_result.json', 'dataset_sources.json', 'MANIFEST.json'))
    refresh = DELIVERIES / 'h4_external_event_source_refresh_v1'
    sources.update(refresh / filename for filename in ('soda_train_sample.json', 'multiwoz_train_sample.json'))
    frozen = [_freeze(path, run / 'frozen_sources' / path.relative_to(DELIVERIES)) for path in sorted(sources)]
    case_paths = []
    for case in cases:
        path = run / 'cases' / (case['name'] + '.json')
        r.base.write(path, case); case_paths.append(str(path))
        payload, proof = make_request(case)
        r.base.write(run / 'previews' / (case['name'] + '.json'), {'payload': payload, 'proof': proof})
    profile, transport, tokenizer = r.transport_inputs()
    r.base.write(run / 'provider_profile.json', profile)
    code = {Path(__file__).resolve(), Path(r.__file__).resolve(), Path(r.base.__file__).resolve(),
            Path(grounding.__file__).resolve(), *map(Path, transport)}
    roots = {RUNTIME.resolve(), r.LEGACY.resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.is_file() and path.suffix == '.py' and any(path.is_relative_to(root) for root in roots):
                code.add(path)
    snapshots = [_freeze(path, run / 'execution_code' / f'{i:03d}_{path.name}') for i, path in enumerate(sorted(code))]
    r.base.write(run / 'EXECUTION_CODE_SNAPSHOT.json', {'files': snapshots})
    r.base.write(run / 'SOURCE_SNAPSHOT.json', {'files': frozen})
    paths = set(code) | set(tokenizer) | {RUNTIME / 'SOURCE_REVIEW.md', RUNTIME / 'POLICY.md'}
    paths.update(Path(row[key]) for row in frozen for key in ('original_path', 'frozen_path'))
    paths.update(run.rglob('*.json')); paths.update(Path(row['frozen_path']) for row in snapshots)
    manifest = {'version': VERSION, 'model': r.MODEL, 'prepared_at_utc': r.now(), 'cases': case_paths,
        'logical_call_max': 3, 'http_max': 6, 'automatic_transport_retries': 1,
        'minimum_http_gap_after_response_seconds': r.MIN_HTTP_GAP_SECONDS,
        'schema_retries': 0, 'semantic_retries': 0, 'source_events_reassigned': False,
        'production_admission': False, 'new_scene_generation': False,
        'question_calls': 0, 'solver_calls': 0, 'training_calls': 0,
        'semantic_certified': False, 'files': r.base.bindings(sorted(paths))}
    r.base.write(run / 'MANIFEST.json', manifest)
    return {'status': 'PREPARED', 'cases': 3, 'logical_call_max': 3, 'http_max': 6, 'http_calls': 0}


def verify(run):
    manifest = r.base.read(run / 'MANIFEST.json')
    expected = {'version': VERSION, 'model': r.MODEL, 'logical_call_max': 3, 'http_max': 6,
        'automatic_transport_retries': 1, 'minimum_http_gap_after_response_seconds': r.MIN_HTTP_GAP_SECONDS,
        'schema_retries': 0, 'semantic_retries': 0, 'source_events_reassigned': False,
        'production_admission': False, 'new_scene_generation': False,
        'question_calls': 0, 'solver_calls': 0, 'training_calls': 0, 'semantic_certified': False}
    for key, value in expected.items():
        r.base.require(manifest.get(key) == value, 'manifest_policy_changed:' + key)
    r.base.verify_bindings(manifest['files'])
    r.base.require([r.base.read(path)['name'] for path in manifest['cases']] == [row[0] for row in CASES],
                   'case_inventory_changed')
    for path in manifest['cases']:
        case = r.base.read(path); c = case['inputs']
        parent = r.base.read(case['parent_case'])
        r.base.require(c == parent['inputs'], 'original_inputs_changed')
        r.base.require(r.base.digest(c['history']) == case['original_history_sha256'] and
            r.base.digest(c['scene_plan']) == case['original_scene_plan_sha256'], 'original_history_or_plan_changed')
        payload, proof = make_request(case)
        r.base.require(r.base.read(run / 'previews' / (case['name'] + '.json')) == {'payload': payload, 'proof': proof},
                       'request_preview_changed')
    return manifest


def cost(run):
    token_fields = ('prompt_tokens', 'completion_tokens', 'total_tokens')
    totals = {key: 0 for key in token_fields}
    bodies = sorted((run / 'requests').glob('*/HTTP_RESPONSE_*.json'))
    missing = []
    for path in bodies:
        metadata = r.base.read(path).get('metadata', {})
        usage = metadata.get('usage', {})
        if not all(type(usage.get(key)) is int for key in token_fields):
            missing.append(str(path)); continue
        for key in token_fields:
            totals[key] += usage[key]
    return {'provider_reported_tokens': totals, 'response_bodies': len(bodies),
            'usage_missing_response_paths': missing, 'monetary_cost': None,
            'monetary_cost_reason': 'No verified provider price is bound; token usage is reported without inventing a price.'}


def live(run):
    manifest = verify(run)
    r.base.require(not (run / 'LIVE_STARTED.json').exists(), 'probe_cannot_restart')
    r.base.write(run / 'LIVE_STARTED.json', {'started_at_utc': r.now(), 'manifest_sha256': r.base.digest(manifest)})
    caller, rows = r.Caller(run), []
    for path in manifest['cases']:
        case = r.base.read(path)
        payload, proof = make_request(case)
        directory = run / 'requests' / f"{caller.request_counter + 1:03d}-mapping-{case['name']}"
        transport_error = None
        try:
            raw, actual = caller.call('mapping-' + case['name'], payload, proof)
            r.base.require(actual == directory, 'caller_request_directory_mismatch')
            checked = evaluate(case, raw)
        except (RuntimeError, ValueError) as exc:
            transport_error = str(exc)
            checked = {'status': 'ERROR', 'schema_valid': False, 'error': transport_error,
                       'checks': {}, 'semantic_certified': False}
        r.base.write(directory / 'EVALUATION.json', checked)
        row = summary(case, checked, directory, transport_error)
        r.base.write(run / 'results' / (case['name'] + '.json'), row)
        rows.append(row)
        print(json.dumps({key: row[key] for key in ('case', 'status', 'schema_valid', 'expected_match')}), flush=True)
    verify(run)
    report = {'status': 'PASS' if all(row['expected_match'] for row in rows) else 'REGRESSION_NOT_ESTABLISHED',
        'cases': rows, 'logical_calls': caller.request_counter, 'http_attempts': caller.http_counter,
        'cost': cost(run), 'new_scenes': 0, 'production_admission': False, 'semantic_certified': False,
        'completed_at_utc': r.now()}
    r.base.write(run / 'REPORT.json', report)
    return report


def audit(run):
    manifest = verify(run)
    failures, rows, inventory = [], [], []
    def check(ok, code):
        if not ok:
            failures.append(code)
    for path in manifest['cases']:
        case = r.base.read(path)
        result_path = run / 'results' / (case['name'] + '.json')
        if not result_path.exists():
            check(not (run / 'LIVE_STARTED.json').exists(), 'case_result_missing:' + case['name'])
            continue
        row = r.base.read(result_path); directory = Path(row['request'])
        inventory.append(directory.resolve())
        check(directory.resolve().parent == (run / 'requests').resolve(), 'request_outside_probe')
        payload, proof = make_request(case)
        check(r.base.read(directory / 'REQUEST.json') == payload, 'request_replay_mismatch:' + case['name'])
        if (directory / 'RESPONSE.json').exists():
            recorded = r.base.read(directory / 'PROOF.json')
            check(all(recorded.get(k) == v for k, v in proof.items()) and
                recorded.get('payload_sha256') == r.base.digest(payload), 'proof_replay_mismatch:' + case['name'])
            response = r.base.read(directory / 'RESPONSE.json')
            checked = evaluate(case, response['text'])
        else:
            check((directory / 'ERROR.json').exists() or (directory / 'LOCAL_ERROR.json').exists(), 'missing_error_receipt')
            checked = {'status': 'ERROR', 'schema_valid': False, 'error': row['transport_error'],
                       'checks': {}, 'semantic_certified': False}
        check(r.base.read(directory / 'EVALUATION.json') == checked, 'evaluation_replay_mismatch:' + case['name'])
        replay = summary(case, checked, directory, row.get('transport_error'))
        check(replay == row, 'result_replay_mismatch:' + case['name']); rows.append(replay)
    actual = sorted(p.resolve() for p in (run / 'requests').glob('*') if p.is_dir())
    check(sorted(inventory) == actual and len(actual) <= 3, 'logical_request_inventory_invalid')
    attempts = list((run / 'requests').glob('*/HTTP_ATTEMPT*.json'))
    numbers = [r.base.read(p).get('http_attempt_number') for p in attempts]
    check(len(attempts) <= 6 and all(type(n) is int for n in numbers) and
          sorted(numbers) == list(range(1, len(numbers) + 1)), 'http_budget_or_sequence_invalid')
    offline_test = False
    for directory in actual:
        payload = r.base.read(directory / 'REQUEST.json')
        http = sorted(directory.glob('HTTP_ATTEMPT*.json'))
        check(len(http) <= 2, 'transport_budget_exceeded')
        check(sorted(r.base.read(p).get('request_attempt', 0) for p in http) == list(range(1, len(http) + 1)),
              'request_http_sequence_invalid')
        for p in http:
            entry = r.base.read(p); number = entry['request_attempt']
            check(entry.get('payload_sha256') == r.base.digest(payload) and
                entry.get('actual_payload_equals_frozen') is True and entry.get('requested_model') == r.MODEL,
                'http_payload_or_model_mismatch')
            throttle = r.base.read(directory / f'THROTTLE_{number:02d}.json')
            check(throttle.get('minimum_gap_after_previous_response_seconds') == r.MIN_HTTP_GAP_SECONDS and
                type(throttle.get('waited_seconds')) in (int, float) and throttle['waited_seconds'] >= 0 and
                isinstance(throttle.get('http_started_at_utc'), str), 'http_throttle_receipt_invalid')
            if number > 1:
                previous = r.base.read(directory / f'HTTP_RESULT_{number-1:02d}.json')
                check(entry.get('same_payload_retry') is True and previous.get('outcome') == 'error' and
                      previous.get('http_status') in (503, 504), 'retry_policy_mismatch')
        if (directory / 'RESPONSE.json').exists():
            response = r.base.read(directory / 'RESPONSE.json'); metadata = response.get('metadata', {})
            offline_test |= metadata.get('offline_test_fake_transport') is True
            check(metadata.get('returned_model') == r.MODEL and metadata.get('finish_reason') == 'stop',
                  'response_model_or_finish_invalid')
            check(bool(http), 'response_without_http')
            final = r.base.read(directory / f'HTTP_RESPONSE_{len(http):02d}.json')
            result = r.base.read(directory / f'HTTP_RESULT_{len(http):02d}.json')
            check(response.get('text') == final.get('text') and metadata == final.get('metadata') and
                result.get('outcome') == 'response' and result.get('http_status') in (None, 200),
                'response_not_final_http_body')
    regression = 'PREPARED_NO_LIVE_CALLS' if not actual else (
        'PASS' if len(rows) == 3 and all(row['expected_match'] for row in rows) else 'REGRESSION_NOT_ESTABLISHED')
    if (run / 'REPORT.json').exists():
        report = r.base.read(run / 'REPORT.json')
        check(report.get('cases') == rows and report.get('status') == regression, 'report_replay_mismatch')
        check(report.get('logical_calls') == len(actual) and report.get('http_attempts') == len(attempts), 'report_counts_mismatch')
        check(report.get('cost') == cost(run), 'cost_replay_mismatch')
        check(report.get('new_scenes') == 0 and report.get('production_admission') is False and
              report.get('semantic_certified') is False, 'report_scope_mismatch')
    result = {'status': 'PASS' if not failures else 'FAIL', 'failures': failures,
        'regression_status': regression, 'cases': rows, 'logical_calls': len(actual),
        'http_attempts': len(attempts), 'cost': cost(run), 'offline_test_fake_transport': offline_test,
        'new_scenes': 0, 'production_admission': False, 'semantic_certified': False,
        'audited_at_utc': r.now()}
    output = run / 'MAPPING_AUDIT.json'; index = 2
    while output.exists():
        output = run / f'MAPPING_AUDIT_{index:03d}.json'; index += 1
    r.base.write(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'live', 'audit'])
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    result = globals()[args.command](args.run.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command == 'audit' and result['status'] != 'PASS':
        raise SystemExit(1)
