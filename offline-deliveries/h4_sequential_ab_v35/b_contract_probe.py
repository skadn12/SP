"""Frozen B scanner contract comparison: 8 cases, at most 11 logical calls.

prepare and audit are offline. Only live invokes the existing Qwen Caller.
No new source reservation, dialogue, question, Solver or training output.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

DELIVERIES = Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries')
RUNTIME = DELIVERIES / 'h4_sequential_ab_v35'
sys.path.insert(0, str(RUNTIME))
import runner as r
import scanner_patch

_legacy_path = DELIVERIES / 'h4_sequential_ab_v34/event_state.py'
_spec = importlib.util.spec_from_file_location('frozen_v34_event_contract', _legacy_path)
legacy = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = legacy
_spec.loader.exec_module(legacy)

VERSION = 'v35_frozen_scanner_contract_comparison_1'
ORIGIN = DELIVERIES / 'h4_sequential_ab_v34/runs/inherited_a3_b3_01'
CONTROLS = DELIVERIES / 'h4_sequential_ab_v34/probes/frozen_parent_goal_repair_01'
SLOTS = (('S01', '006', '007'), ('S02', '012', '013'), ('S03', '018', '019'))
CASES = tuple((f'legacy_repair_{slot.lower()}', 1) for slot, _, _ in SLOTS) + tuple(
    (f'fresh_v35_{slot.lower()}', 2) for slot, _, _ in SLOTS) + (
    ('control_v32_s02_card_old_goal', 1), ('control_v28_s01_independent', 1))
LOGICAL_MAX = sum(limit for _, limit in CASES)


def _decode(payload):
    marker = '\nEVENT_STATE_INPUT_JSON\n'
    content = payload['messages'][0]['content']
    r.base.require(content.count(marker) == 1, 'historical_prompt_marker_invalid')
    return json.JSONDecoder().raw_decode(content.split(marker, 1)[1])[0]


def _directory(prefix):
    found = list((ORIGIN / 'requests').glob(prefix + '-*'))
    r.base.require(len(found) == 1, 'historical_request_not_unique:' + prefix)
    return found[0]


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


def _source_cases():
    cards = {row['card_id']: row for row in r.base.read(ORIGIN / 'dataset_cards_result.json')['accepted_cards']}
    actual = []
    for slot, planner_prefix, scanner_prefix in SLOTS:
        planner, scanner = _directory(planner_prefix), _directory(scanner_prefix)
        plan = r.base.read(planner / 'CANDIDATE_PLAN.json')['scenes'][0]
        context = _decode(r.base.read(scanner / 'REQUEST.json'))
        proof = r.base.read(scanner / 'PROOF.json')
        history = context['cumulative_history']
        r.base.require(r.base.digest(plan) == proof['scene_plan_sha256'], 'original_full_plan_hash_mismatch')
        r.base.require(r.event_contract.scan_plan_context(plan) == context['scene_plan'], 'original_plan_projection_mismatch')
        r.base.require(r.base.digest(history) == proof['history_sha256'] and len(history) == 132,
                       'original_complete_history_mismatch')
        card = cards[plan['dataset_card_id']]
        r.base.require(r.planner_contract.mapping_context(card)['card_guidance'] == context['dataset_cards'][0],
                       'original_card_context_mismatch')
        inputs = {'root_id': context['root_id'], 'route': context['route'], 'scene_plan': plan,
                  'history': history, 'dataset_card': card, 'latest_state': context['latest_state'],
                  'purpose_input': context['purpose_input']}
        paths = [planner / name for name in ('REQUEST.json', 'RESPONSE.json', 'CANDIDATE_PLAN.json',
                                             'PLAN_CITATION_PROVENANCE.json', 'PROOF.json', 'VALIDATION.json')]
        paths += [scanner / name for name in ('REQUEST.json', 'RESPONSE.json', 'PROOF.json', 'VALIDATION.json')]
        raw = r.base.read(scanner / 'RESPONSE.json')['text']
        actual.append((slot, inputs, context, raw, paths, str(scanner)))
    cases = []
    for slot, inputs, context, raw, paths, scanner in actual:
        cases.append(({'name': f'legacy_repair_{slot.lower()}', 'kind': 'legacy_repair', 'logical_call_max': 1,
                       'expected_status': 'PASS' if slot == 'S02' else 'REJECT',
                       'expectation_scope': 'format_and_exact_preservation_only',
                       'inputs': deepcopy(inputs), 'original_context': deepcopy(context), 'previous_raw': raw,
                       'origin_request': scanner}, paths))
    for slot, inputs, context, raw, paths, scanner in actual:
        expected = {'card_revise_steps': ['s1'], 'old_goal_blocker': slot == 'S01'}
        if slot == 'S01':
            expected.update(card_revise_steps=['s1', 's2', 's3', 's4', 's5', 's6'],
                            card_revise_dependencies=[1, 2, 3])
        if slot == 'S03':
            expected.update(card_revise_steps=['s1', 's2'], card_revise_dependencies=[1, 2],
                            card_compatible_steps=['s3', 's4'], card_compatible_dependencies=[3])
        cases.append(({'name': f'fresh_v35_{slot.lower()}', 'kind': 'fresh_scan', 'logical_call_max': 2,
                       'expected_status': 'REJECT', 'expected_dimensions': expected,
                       'expectation_scope': 'predeclared_separate_card_and_goal_dimensions',
                       'inputs': deepcopy(inputs), 'original_context': deepcopy(context), 'previous_raw': raw,
                       'origin_request': scanner}, paths))
    for name, source_name, expected_status, dimensions in (
            ('control_v32_s02_card_old_goal', 'event_s02_card_replacement_old_goal', 'REJECT',
             {'card_revise_steps': ['s1', 's2', 's3', 's4', 's5', 's6'],
              'card_revise_dependencies': [1, 2, 3], 'old_goal_blocker': True}),
            ('control_v28_s01_independent', 'event_s01_same_theme_independent_control', 'PASS',
             {'all_cards_compatible': True, 'old_goal_blocker': False})):
        source = CONTROLS / 'cases' / (source_name + '.json')
        original = r.base.read(source)
        inputs = {key: deepcopy(original['inputs'][key]) for key in
                  ('root_id', 'route', 'scene_plan', 'history', 'dataset_card', 'latest_state', 'purpose_input')}
        case = {'name': name, 'kind': 'fresh_scan', 'logical_call_max': 1, 'expected_status': expected_status,
                'expected_dimensions': dimensions, 'expectation_scope': 'known_control_not_held_out',
                'inputs': inputs, 'original_context': original['original_context'],
                'previous_raw': original['previous_raw'], 'origin_request': original['origin_request']}
        paths = [source] + [Path(row['original_path']) for row in original['source_bindings']]
        cases.append((case, paths))
    return cases


def _contract(case):
    return legacy if case['kind'] == 'legacy_repair' else r.event_contract


def _evaluate(case, candidate):
    c = case['inputs']
    contract = _contract(case)
    try:
        checked = contract.validate(candidate, c['root_id'], c['route'], c['scene_plan'], c['history'],
            require_goal_checks=True, dataset_cards=[c['dataset_card']], require_card_checks=True,
            require_parent_goal_checks=case['kind'] == 'legacy_repair')
    except (ValueError, TypeError) as error:
        return {'status': 'FORMAT_ERROR', 'schema_valid': False, 'error': str(error),
                'errors': deepcopy(getattr(error, 'errors', [])), 'semantic_certified': False}
    return {'status': 'REJECT' if contract.requires_replan(checked) else 'PASS',
            'schema_valid': True, 'validated': checked, 'semantic_certified': False}


def _format_repair(case, candidate, checked):
    r.base.require(checked.get('schema_valid') is False, 'semantic_result_must_not_be_retried')
    if case['kind'] == 'legacy_repair':
        spec = scanner_patch.build_spec(candidate, checked['errors'], source_contract='legacy_v34',
                                       target_contract='legacy_v34')
    else:
        error = r.event_contract.EventStateValidationError(checked['errors'])
        spec = r.event_contract.build_local_repair(error, candidate, history=case['inputs']['history'])
    r.base.require(not spec.get('unrepairable_failures'), 'scanner_patch_unrepairable:' +
                   json.dumps(spec.get('unrepairable_failures'), ensure_ascii=False))
    return spec


def _initial(case):
    if case['kind'] != 'legacy_repair':
        return None, None
    previous = case['previous_raw']
    return _format_repair(case, previous, _evaluate(case, previous)), previous


def _make_request(case, *, repair=None):
    c = case['inputs']
    if case['kind'] != 'legacy_repair':
        payload = r.event_contract.build_request(c['root_id'], c['route'], c['scene_plan'], c['history'],
            [c['dataset_card']], c['latest_state'], repair=repair, purpose_input=c['purpose_input'],
            require_goal_checks=True, require_card_checks=True, require_parent_goal_checks=False)
        sent = _decode(payload)
        r.base.require(sent['scene_plan'] == r.event_contract.scan_plan_context(c['scene_plan']) and
                       sent['cumulative_history'] == c['history'], 'fresh_complete_input_changed')
    else:
        r.base.require(repair is not None, 'legacy_repair_requires_spec')
        context = {'root_id': c['root_id'], 'route': c['route'], 'scene_id': c['scene_plan']['scene_id'],
                   'scene_plan': legacy.scan_plan_context(c['scene_plan']), 'cumulative_history': c['history'],
                   'dataset_cards': [r.planner_contract.mapping_context(c['dataset_card'])['card_guidance']],
                   'latest_state': c['latest_state'], 'purpose_input': c['purpose_input']}
        payload = {'model': r.MODEL, 'temperature': .25, 'max_tokens': 4096, 'enable_thinking': False,
                   'response_format': {'type': 'json_object'},
                   'messages': [{'role': 'user', 'content': scanner_patch.build_prompt(repair)
                        + '\nEVENT_STATE_INPUT_JSON\n' + json.dumps(context, ensure_ascii=False, separators=(',', ':'))}]}
    proof = {'probe_version': VERSION, 'regression_case': case['name'], 'kind': case['kind'],
             'root_id': c['root_id'], 'route': c['route'], 'scene_id': c['scene_plan']['scene_id'],
             'role': 'frozen_scanner_format_patch' if repair is not None else 'frozen_single_contract_scan',
             'history_sha256': r.base.digest(c['history']), 'history_turns': len(c['history']),
             'scene_plan_sha256': r.base.digest(c['scene_plan']), 'frozen_inputs_sha256': r.base.digest(c),
             'expected_verdict_withheld_from_model': True, 'prior_response_withheld_from_initial_judgment': repair is None,
             'schema_repair_only': repair is not None, 'repair_spec_sha256': r.base.digest(repair),
             'require_parent_goal_checks': case['kind'] == 'legacy_repair',
             'semantic_certified': False}
    return payload, proof


def _merge(case, previous, raw, repair):
    if case['kind'] == 'legacy_repair':
        return scanner_patch.apply_patch(previous, raw, repair)
    merged = r.event_contract.merge_event_repair(previous, raw, repair, case['inputs']['history'])
    r.event_contract.validate_repair_preservation(previous, merged, repair)
    return merged


def _dimensions(checked):
    validated = checked.get('validated') or {}
    role = validated.get('route_fulfillment') or {}
    cards = validated.get('card_plan_checks') or []
    goals = role.get('history_goal_checks') or []
    parent = validated.get('parent_goal_checks') or []
    return {'card_revise_steps': [row['id'] for row in cards if row.get('kind') == 'step' and row.get('verdict') == 'revise'],
            'card_revise_dependencies': [row['id'] for row in cards if row.get('kind') == 'dependency' and row.get('verdict') == 'revise'],
            'card_compatible_steps': [row['id'] for row in cards if row.get('kind') == 'step' and row.get('verdict') == 'compatible'],
            'card_compatible_dependencies': [row['id'] for row in cards if row.get('kind') == 'dependency' and row.get('verdict') == 'compatible'],
            'all_cards_compatible': bool(cards) and all(row.get('verdict') == 'compatible' for row in cards),
            'old_goal_blocker': bool(any(row.get('relation') in ('serves_existing_goal', 'uncertain') for row in goals)
                or any(row.get('functional_relation') in ('executes_or_prepares', 'verifies_or_follows_up', 'uncertain') for row in parent)
                or role.get('serves_existing_event') is True
                or role.get('relation_to_history') in ('continuation', 'old_event_substep')),
            'history_goal_comparisons': deepcopy(goals)}


def _expected_match(case, checked):
    if checked.get('schema_valid') is not True or checked.get('status') != case['expected_status']:
        return False
    if case['kind'] == 'legacy_repair':
        # apply_patch has independently enforced exact operation coverage and
        # frozen-value preservation. No semantic correctness assertion is made.
        return True
    observed = _dimensions(checked)
    for key, expected in case['expected_dimensions'].items():
        actual = observed[key]
        if isinstance(expected, list):
            if not set(expected) <= set(actual):
                return False
        elif expected != actual:
            return False
    return True


def _summary(case, checked, requests, *, transport_error=None, repair_error=None):
    return {'case': case['name'], 'kind': case['kind'], 'expected_status': case['expected_status'],
            'expectation_scope': case['expectation_scope'], 'status': checked.get('status', 'ERROR'),
            'schema_valid': checked.get('schema_valid', False), 'expected_match': _expected_match(case, checked),
            'dimensions': _dimensions(checked), 'requests': requests, 'logical_calls': len(requests),
            'transport_error': transport_error, 'repair_error': repair_error,
            'history_turns': len(case['inputs']['history']), 'history_sha256': r.base.digest(case['inputs']['history']),
            'result_sha256': r.base.digest(checked), 'semantic_certified': False}


def _parent_provenance(case, candidate, checked):
    if case['kind'] != 'legacy_repair' or checked.get('schema_valid') is not True:
        return None
    return legacy.parent_goal_decision_provenance(r.parse_object(candidate), case['inputs']['history'])


def prepare(run):
    r.base.require(not run.exists(), 'probe_already_exists')
    r.base.require(run.parent == RUNTIME / 'probes', 'probe_must_be_in_v35_probes')
    r.base.require(r.MIN_HTTP_GAP_SECONDS >= 35, 'probe_gap_below_existing_policy')
    run.mkdir(parents=True)
    sources, cases = [], []
    common_paths = [ORIGIN / name for name in ('dataset_cards_result.json', 'dataset_sources.json',
        'MANIFEST.json', 'REPORT.json', 'FINAL_AUDIT.json', 'INHERITED_A.json')]
    common_paths += list((ORIGIN / 'requests').glob('00[123]-*/RESPONSE.json'))
    for path in common_paths:
        r.base.require(path.is_file(), 'origin_binding_missing:' + str(path))
        sources.append(_freeze(path, run / 'frozen_sources' / path.relative_to(DELIVERIES)))
    for case, paths in _source_cases():
        receipts = [_freeze(path, run / 'frozen_sources' / path.relative_to(DELIVERIES)) for path in sorted(set(paths))]
        case['source_bindings'] = receipts
        sources.extend(row for row in receipts if row not in sources)
        case['original_history_sha256'] = r.base.digest(case['inputs']['history'])
        case['historical_baseline_evaluation'] = _legacy_baseline(case)
        repair, previous = _initial(case)
        case['initial_repair'] = repair
        case['initial_previous_sha256'] = r.base.digest(previous)
        case_path = run / 'cases' / (case['name'] + '.json')
        r.base.write(case_path, case)
        payload, proof = _make_request(case, repair=repair)
        r.base.write(run / 'previews' / (case['name'] + '.json'), {'payload': payload, 'proof': proof})
        cases.append(str(case_path))
    profile, transport, tokenizer = r.transport_inputs()
    r.base.write(run / 'provider_profile.json', profile)
    code = {Path(__file__).resolve(), Path(r.__file__).resolve(), Path(r.base.__file__).resolve(),
            Path(scanner_patch.__file__).resolve(), _legacy_path, *map(Path, transport)}
    runtime_roots = {RUNTIME.resolve(), r.LEGACY.resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.is_file() and path.suffix == '.py' and any(path.is_relative_to(root) for root in runtime_roots):
                code.add(path)
    snapshots = [_freeze(path, run / 'execution_code' / f'{i:03d}_{path.name}') for i, path in enumerate(sorted(code))]
    r.base.write(run / 'EXECUTION_CODE_SNAPSHOT.json', {'files': snapshots})
    paths = list(code) + list(tokenizer) + [RUNTIME / 'POLICY.md', RUNTIME / 'COMPARISON_DESIGN.md']
    paths += [Path(row[key]) for row in sources for key in ('original_path', 'frozen_path')]
    paths += list(run.rglob('*.json')) + [Path(row['frozen_path']) for row in snapshots]
    manifest = {'version': VERSION, 'model': r.MODEL, 'prepared_at_utc': r.now(), 'cases': cases,
        'logical_call_max': LOGICAL_MAX, 'case_logical_limits': dict(CASES),
        'http_max': LOGICAL_MAX * (r.TRANSPORT_RETRIES + 1), 'automatic_transport_retries': r.TRANSPORT_RETRIES,
        'minimum_http_gap_after_response_seconds': r.MIN_HTTP_GAP_SECONDS,
        'historical_baseline_is_recorded_not_rerun': True, 'paired_same_plan_source_history': True,
        'source_events_reassigned': False, 'historical_records_modified': False, 'new_scene_generation': False,
        'question_calls': 0, 'solver_calls': 0, 'training_calls': 0, 'semantic_certified': False,
        'files': r.base.bindings(sorted(set(paths)))}
    r.base.write(run / 'MANIFEST.json', manifest)
    return {'status': 'PREPARED', 'cases': len(cases), 'logical_call_max': LOGICAL_MAX,
            'http_max': manifest['http_max'], 'http_calls': 0, 'new_scenes': 0}


def _legacy_baseline(case):
    copied = deepcopy(case)
    copied['kind'] = 'legacy_repair'
    return _evaluate(copied, copied['previous_raw'])


def verify(run):
    manifest = r.base.read(run / 'MANIFEST.json')
    r.base.require(manifest['version'] == VERSION and manifest['model'] == r.MODEL, 'wrong_probe_version_or_model')
    r.base.require(manifest['logical_call_max'] == LOGICAL_MAX and manifest['case_logical_limits'] == dict(CASES),
                   'logical_budget_changed')
    r.base.require(manifest['http_max'] == LOGICAL_MAX * (r.TRANSPORT_RETRIES + 1), 'http_budget_changed')
    r.base.require(manifest['minimum_http_gap_after_response_seconds'] == r.MIN_HTTP_GAP_SECONDS, 'http_gap_changed')
    for key, expected in {'source_events_reassigned': False, 'historical_records_modified': False,
            'new_scene_generation': False, 'paired_same_plan_source_history': True,
            'historical_baseline_is_recorded_not_rerun': True,
            'question_calls': 0, 'solver_calls': 0, 'training_calls': 0, 'semantic_certified': False}.items():
        r.base.require(manifest.get(key) == expected, 'probe_scope_changed:' + key)
    r.base.verify_bindings(manifest['files'])
    for row in r.base.read(run / 'EXECUTION_CODE_SNAPSHOT.json')['files']:
        r.base.require(r.base.digest(Path(row['frozen_path']).read_bytes()) == row['sha256'], 'execution_snapshot_changed')
    r.base.require([r.base.read(path)['name'] for path in manifest['cases']] == [name for name, _ in CASES],
                   'case_inventory_changed')
    for path in manifest['cases']:
        case = r.base.read(path)
        history = case['inputs']['history']
        r.base.require(history == case['original_context']['cumulative_history'] and
                       r.base.digest(history) == case['original_history_sha256'], 'complete_case_history_changed')
        r.base.require(_legacy_baseline(case) == case['historical_baseline_evaluation'], 'baseline_replay_mismatch')
        repair, previous = _initial(case)
        r.base.require(repair == case['initial_repair'] and r.base.digest(previous) == case['initial_previous_sha256'],
                       'initial_repair_replay_mismatch')
        payload, proof = _make_request(case, repair=repair)
        r.base.require(r.base.read(run / 'previews' / (case['name'] + '.json')) == {'payload': payload, 'proof': proof},
                       'frozen_preview_replay_mismatch:' + case['name'])
    return manifest


def live(run):
    manifest = verify(run)
    r.base.require(not (run / "LIVE_STARTED.json").exists(), "probe_cannot_restart")
    r.base.write(run / "LIVE_STARTED.json", {"started_at_utc": r.now(), "manifest_sha256": r.base.digest(manifest)})
    caller, rows = r.Caller(run), []
    for path in manifest["cases"]:
        case = r.base.read(path)
        repair, previous = _initial(case)
        requests, checked, transport_error, repair_error = [], {}, None, None
        for attempt in range(1, case["logical_call_max"] + 1):
            payload, proof = _make_request(case, repair=repair)
            proof.update(logical_attempt=attempt, previous_candidate_sha256=r.base.digest(previous))
            before = caller.request_counter
            label = f"B-contract-{case['name']}-{attempt}"
            try:
                raw, directory = caller.call(label, payload, proof)
            except (RuntimeError, ValueError) as error:
                transport_error = str(error)
                if caller.request_counter > before:
                    requests.append(str(run / "requests" / f"{caller.request_counter:03d}-{label}"))
                checked = {"status": "ERROR", "schema_valid": False, "error": str(error), "semantic_certified": False}
                break
            requests.append(str(directory))
            r.base.write(directory / "REGRESSION_STEP.json", {"case": case["name"], "logical_attempt": attempt,
                "repair": repair, "previous_candidate_sha256": r.base.digest(previous)})
            try:
                candidate = raw if repair is None else _merge(case, previous, raw, repair)
            except (ValueError, TypeError) as error:
                checked = {"status": "FORMAT_ERROR", "schema_valid": False, "error": str(error), "semantic_certified": False}
                r.base.write(directory / "EVALUATION.json", checked)
                break
            r.base.write(directory / "CANDIDATE.json", candidate)
            checked = _evaluate(case, candidate)
            r.base.write(directory / "EVALUATION.json", checked)
            provenance = _parent_provenance(case, candidate, checked)
            if provenance is not None:
                r.base.write(directory / "PARENT_GOAL_DECISION_PROVENANCE.json", provenance)
            if checked.get("schema_valid") is not False:
                break
            if attempt < case["logical_call_max"]:
                try:
                    repair = _format_repair(case, candidate, checked)
                except (ValueError, TypeError) as error:
                    repair_error = str(error)
                    r.base.write(directory / "FORMAT_REPAIR_ERROR.json", {"error": repair_error})
                    break
                previous = candidate
                r.base.write(directory / "FORMAT_REPAIR_SPEC.json", repair)
        row = _summary(case, checked, requests, transport_error=transport_error, repair_error=repair_error)
        rows.append(row)
        r.base.write(run / "results" / (case["name"] + ".json"), row)
        print(json.dumps({key: row[key] for key in ('case', 'kind', 'status', 'schema_valid',
            'expected_match', 'logical_calls')}, ensure_ascii=False), flush=True)
    verify(run)
    report = {"status": "PASS" if all(row["expected_match"] for row in rows) else "REGRESSION_NOT_ESTABLISHED",
              "cases": rows, "logical_calls": caller.request_counter, "http_attempts": caller.http_counter,
              "new_scenes": 0, "question_calls": 0, "solver_calls": 0, "training_calls": 0,
              "semantic_certified": False, "completed_at_utc": r.now()}
    r.base.write(run / "REPORT.json", report)
    return report


def audit(run):
    manifest = verify(run)
    failures, rows, inventory = [], [], []
    def check(condition, code):
        if not condition:
            failures.append(code)
    for path in manifest["cases"]:
        case = r.base.read(path)
        result_path = run / "results" / (case["name"] + ".json")
        if not result_path.exists():
            rows.append({"case": case["name"], "status": "NO_LIVE_RESULT", "expected_match": False})
            check(not (run / "LIVE_STARTED.json").exists(), "case_result_missing:" + case["name"])
            continue
        row = r.base.read(result_path)
        directories = [Path(p) for p in row["requests"]]
        check(1 <= len(directories) <= case["logical_call_max"], "case_request_budget_invalid:" + case["name"])
        repair, previous = _initial(case)
        checked, repair_error = {}, None
        for attempt, directory in enumerate(directories, 1):
            inventory.append(directory.resolve())
            check(directory.resolve().parent == (run / "requests").resolve(), "request_outside_probe")
            payload, proof = _make_request(case, repair=repair)
            proof.update(logical_attempt=attempt, previous_candidate_sha256=r.base.digest(previous))
            check(r.base.read(directory / "REQUEST.json") == payload, "request_replay_mismatch:" + directory.name)
            if not (directory / "RESPONSE.json").exists():
                check((directory / "ERROR.json").exists() or (directory / "LOCAL_ERROR.json").exists(),
                      "missing_transport_or_local_error:" + directory.name)
                checked = {"status": "ERROR", "schema_valid": False, "error": row["transport_error"],
                           "semantic_certified": False}
                check(attempt == len(directories), "request_after_transport_failure")
                break
            actual_proof = r.base.read(directory / "PROOF.json")
            check(all(actual_proof.get(k) == v for k, v in proof.items()), "proof_replay_mismatch:" + directory.name)
            check(actual_proof.get("payload_sha256") == r.base.digest(payload), "proof_payload_hash_mismatch:" + directory.name)
            raw = r.base.read(directory / "RESPONSE.json")["text"]
            check(r.base.read(directory / "REGRESSION_STEP.json") == {
                "case": case["name"], "logical_attempt": attempt, "repair": repair,
                "previous_candidate_sha256": r.base.digest(previous)}, "regression_step_mismatch:" + directory.name)
            try:
                candidate = raw if repair is None else _merge(case, previous, raw, repair)
            except (ValueError, TypeError) as error:
                checked = {"status": "FORMAT_ERROR", "schema_valid": False, "error": str(error), "semantic_certified": False}
                check(r.base.read(directory / "EVALUATION.json") == checked, "merge_failure_replay_mismatch:" + directory.name)
                check(attempt == len(directories), "request_after_failed_merge")
                break
            check(r.base.read(directory / "CANDIDATE.json") == candidate, "candidate_replay_mismatch:" + directory.name)
            checked = _evaluate(case, candidate)
            check(r.base.read(directory / "EVALUATION.json") == checked, "evaluation_replay_mismatch:" + directory.name)
            provenance = _parent_provenance(case, candidate, checked)
            if provenance is not None:
                check(r.base.read(directory / "PARENT_GOAL_DECISION_PROVENANCE.json") == provenance,
                      "parent_goal_provenance_replay_mismatch:" + directory.name)
            if checked.get("schema_valid") is not False:
                check(attempt == len(directories), "semantic_result_retried:" + directory.name)
                break
            if attempt < case["logical_call_max"]:
                try:
                    next_repair = _format_repair(case, candidate, checked)
                except (ValueError, TypeError) as error:
                    repair_error = str(error)
                    check(r.base.read(directory / "FORMAT_REPAIR_ERROR.json") == {"error": repair_error},
                          "format_repair_error_replay_mismatch")
                    check(attempt == len(directories), "request_after_repair_spec_error")
                    break
                check(r.base.read(directory / "FORMAT_REPAIR_SPEC.json") == next_repair, "repair_spec_replay_mismatch")
                repair, previous = next_repair, candidate
        replay_row = _summary(case, checked, row["requests"], transport_error=row.get("transport_error"), repair_error=repair_error)
        check(replay_row == row, "case_result_replay_mismatch:" + case["name"])
        rows.append(replay_row)
    actual = sorted(path.resolve() for path in (run / "requests").glob("*") if path.is_dir())
    check(sorted(inventory) == actual, "unexpected_or_missing_request_directory")
    check(len(actual) <= manifest["logical_call_max"], "logical_budget_exceeded")
    attempts = sorted((run / "requests").glob("*/HTTP_ATTEMPT*.json"))
    check(len(attempts) <= manifest["http_max"], "http_budget_exceeded")
    global_numbers = [r.base.read(path).get("http_attempt_number") for path in attempts]
    check(all(type(number) is int for number in global_numbers)
          and sorted(global_numbers) == list(range(1, len(attempts) + 1)),
          "global_http_attempt_sequence_invalid")
    for directory in actual:
        payload = r.base.read(directory / "REQUEST.json")
        http = sorted(directory.glob("HTTP_ATTEMPT*.json"))
        check(len(http) <= r.TRANSPORT_RETRIES + 1, "transport_retry_budget_exceeded:" + directory.name)
        check(sorted(r.base.read(path).get("request_attempt", 0) for path in http) == list(range(1, len(http) + 1)),
              "request_http_attempt_sequence_invalid:" + directory.name)
        for path in http:
            entry = r.base.read(path)
            check(entry.get("payload_sha256") == r.base.digest(payload)
                  and entry.get("actual_payload_equals_frozen") is True and entry.get("requested_model") == r.MODEL,
                  "http_payload_or_model_mismatch:" + str(path))
            number = entry.get("request_attempt", 1)
            throttle = r.base.read(directory / f"THROTTLE_{number:02d}.json")
            check(throttle.get("minimum_gap_after_previous_response_seconds") == r.MIN_HTTP_GAP_SECONDS
                  and type(throttle.get("waited_seconds")) in (int, float)
                  and throttle["waited_seconds"] >= 0 and isinstance(throttle.get("http_started_at_utc"), str),
                  "http_throttle_record_invalid:" + directory.name)
            if number > 1:
                previous_result = r.base.read(directory / f"HTTP_RESULT_{number-1:02d}.json")
                check(entry.get("same_payload_retry") is True and previous_result.get("outcome") == "error"
                      and previous_result.get("http_status") in (503, 504), "transport_retry_policy_mismatch:" + directory.name)
        if (directory / "RESPONSE.json").exists():
            response = r.base.read(directory / "RESPONSE.json")
            metadata = response.get("metadata", {})
            check(metadata.get("returned_model") == r.MODEL and metadata.get("finish_reason") == "stop",
                  "response_model_or_finish_mismatch:" + directory.name)
            check(bool(http), "response_without_http_attempt:" + directory.name)
            final_response_path = directory / f"HTTP_RESPONSE_{len(http):02d}.json"
            final_result_path = directory / f"HTTP_RESULT_{len(http):02d}.json"
            check(final_response_path.exists() and final_result_path.exists(),
                  "response_http_artifact_missing:" + directory.name)
            if final_response_path.exists() and final_result_path.exists():
                final_response, final_result = r.base.read(final_response_path), r.base.read(final_result_path)
                check(response.get("text") == final_response.get("text")
                      and metadata == final_response.get("metadata"), "response_not_final_http_body:" + directory.name)
                check(final_result.get("outcome") == "response" and final_result.get("http_status") in (None, 200),
                      "final_http_result_not_response:" + directory.name)
    if (run / "REPORT.json").exists():
        report = r.base.read(run / "REPORT.json")
        check(report.get("cases") == rows, "report_case_replay_mismatch")
        expected_status = "PASS" if all(row.get("expected_match") for row in rows) else "REGRESSION_NOT_ESTABLISHED"
        check(report.get("status") == expected_status, "report_regression_status_mismatch")
        check(report.get("http_attempts") == len(attempts) and report.get("logical_calls") == len(actual),
              "report_call_count_mismatch")
        for key in ("new_scenes", "question_calls", "solver_calls", "training_calls"):
            check(report.get(key) == 0, "forbidden_generation_count:" + key)
    result = {"status": "PASS" if not failures else "FAIL", "run": str(run), "failures": failures,
              "regression_status": "PREPARED_NO_LIVE_CALLS" if not actual else (
                  "PASS" if len(rows) == len(CASES) and all(row.get("expected_match") for row in rows)
                  else "REGRESSION_NOT_ESTABLISHED"),
              "cases": rows, "logical_calls": len(actual), "http_attempts": len(attempts),
              "new_scenes": 0, "question_calls": 0, "solver_calls": 0, "training_calls": 0,
              "semantic_certified": False, "audited_at_utc": r.now()}
    output, index = run / "B_CONTRACT_AUDIT.json", 2
    while output.exists():
        output = run / f"B_CONTRACT_AUDIT_{index:03d}.json"
        index += 1
    r.base.write(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "live", "audit"])
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    result = globals()[args.command](args.run.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command == "audit" and result["status"] != "PASS":
        raise SystemExit(1)
