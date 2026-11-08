"""Independent stdlib raw-artifact audit for v101 historical replay and bounded continuation.

No engine imports or provider calls. PASS attests bindings, preservation and budgets,
not semantic reliability, fresh yield or paired causal effects.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parents[1]
RUN = BASE / "regression"
OLD = BASE.parent / "h4_bound_card_review_feedback_v100"
OLD_RUN = OLD / "regression"
LEGACY = BASE.parent / "h4_r02_dualroute_feedback_v1"
ROOT = "cc-train-episode-19053"
PROJECTION_FIELDS = ("card_id", "card_sha256", "guidance_level", "actors", "steps", "dependencies")
LEAF_FIELDS = {"actors": ("role",), "steps": ("action", "state"), "dependencies": ("relation",)}
NESTED_FIELDS = {"actors": ("actor_id", "role"), "steps": ("step_id", "actor_id", "action", "state"),
                 "dependencies": ("before", "after", "relation")}
IDENTITIES = {"segments": [("segment_id",)], "timeline": [("constraint_id",)],
 "plan_checks": [("development_index",)], "checks": [("development_index",)],
 "card_checks": [("card_id",), ("kind", "id")], "step_checks": [("step_id",)],
 "dependency_checks": [("dependency_index",)], "consistency_checks": [("category",), ("check_id",)],
 "state_checks": [("fact_id",)], "event_checks": [("constraint_id",)], "roles": [("actor_id",)]}
EVIDENCE_KEYS = {"evidence", "plan_evidence", "history_evidence", "request_evidence", "acceptance_evidence", "source_paths"}
COMPARISON_KEYS = {"purpose", "start_state", "end_state", "roles", "current_trigger", "nearest_old_event",
                   "request_acceptance", "experience_state", "source_binding"}
WRAPPER_KEYS = {"comparison", "plan_evidence", "reason", "verdict"}
MISSING = object()


def read(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value: Any, *, ascii_only: bool = False) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=ascii_only, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def check(condition: bool, code: str, failures: list[dict[str, Any]], **detail: Any) -> None:
    if not condition:
        failures.append({"code": code, **detail})


def request_context(payload: dict[str, Any], *, event: bool) -> dict[str, Any]:
    marker = "\nEVENT_STATE_INPUT_JSON\n" if event else "\nINPUT_JSON\n"
    messages = payload["messages"]
    assert len(messages) == 1 and messages[0]["role"] == "user", "request_message_shape_invalid"
    text = messages[0]["content"]
    assert text.count(marker) == 1, "request_context_marker_count_invalid"
    value = json.JSONDecoder().raw_decode(text.split(marker, 1)[1])[0]
    assert isinstance(value, dict), "request_context_not_object"
    return value


def card_contract(card: dict[str, Any]) -> dict[str, Any]:
    projection = {field: card.get(field) for field in PROJECTION_FIELDS}
    for collection, fields in NESTED_FIELDS.items():
        rows = card.get(collection)
        projection[collection] = [{field: row[field] for field in fields if field in row}
                                  for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    leaves = []
    for name, fields in LEAF_FIELDS.items():
        rows = projection.get(name)
        for index, row in enumerate(rows if isinstance(rows, list) else []):
            if not isinstance(row, dict):
                continue
            for field in fields:
                value = row.get(field)
                if isinstance(value, str) and value.strip():
                    leaves.append({"source_path": f"/{name}/{index}/{field}", "value": value})
    return {
        "source_projection": projection,
        "source_binding": {"card_id": projection["card_id"], "card_sha256": projection["card_sha256"],
                           "projection_sha256": digest(projection, ascii_only=True)},
        "source_leaves": leaves, "allowed_source_paths": [row["source_path"] for row in leaves],
    }


def audit_event_context(ctx: dict[str, Any], original: dict[str, Any], plan: dict[str, Any],
                        failures: list[dict[str, Any]], **label: Any) -> None:
    check(ctx.get("root_id") == original["root_id"] and ctx.get("route") == original["route"]
          and ctx.get("scene_id") == "S01", "event_identity_changed", failures, **label)
    check(ctx.get("cumulative_history") == original["cumulative_history"],
          "event_full_history_changed", failures, **label)
    check(ctx.get("latest_state") == original["latest_state"], "event_state_changed", failures, **label)
    check(ctx.get("scene_plan") == plan["scenes"][0], "event_plan_not_expected_candidate", failures, **label)
    expected = [card_contract(card) for card in original["dataset_cards"]]
    observed = ctx.get("card_source_contracts")
    check(isinstance(observed, list) and len(observed) == len(expected),
          "bound_contract_count_invalid", failures, **label)
    if isinstance(observed, list):
        for index, contract in enumerate(expected):
            actual = observed[index] if index < len(observed) else {}
            for field, value in contract.items():
                check(isinstance(actual, dict) and actual.get(field) == value,
                      "bound_card_source_contract_changed", failures, index=index, field=field, **label)


def audit_review_output(value: dict[str, Any], original: dict[str, Any],
                        failures: list[dict[str, Any]], **label: Any) -> None:
    cards = {card["card_id"]: card for card in original["dataset_cards"]}
    observed_cards = value.get("card_checks", [])
    check(isinstance(observed_cards, list) and
          [row.get("card_id") for row in observed_cards if isinstance(row, dict)] == list(cards),
          "validated_review_card_coverage_changed", failures, **label)
    for row in observed_cards if isinstance(observed_cards, list) else []:
        card = cards.get(row.get("card_id")) if isinstance(row, dict) else None
        if card is None:
            continue
        expected = card_contract(card)
        comparison = row.get("function_check", {}).get("comparison", {})
        check(comparison.get("source_binding") == expected["source_binding"],
              "validated_review_source_binding_changed", failures, **label)
        for dimension in ("purpose", "start_state", "end_state", "request_acceptance", "experience_state"):
            pair = comparison.get(dimension)
            paths = pair.get("source_paths") if isinstance(pair, dict) else None
            check(isinstance(paths, list) and len(paths) == len(set(paths))
                  and all(isinstance(path, str) and path in expected["allowed_source_paths"] for path in paths),
                  "validated_review_source_paths_invalid", failures, dimension=dimension, **label)
            if dimension in {"purpose", "start_state", "end_state"}:
                check(isinstance(pair, dict) and set(pair) == {"source_paths", "actual"}
                      and isinstance(pair.get("actual"), str) and bool(pair["actual"].strip()),
                      "validated_review_has_free_source_text_or_invalid_actual", failures,
                      dimension=dimension, **label)

def binding(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def historical_request(prefix):
    matches = list((OLD_RUN / "requests").glob(prefix + "-*"))
    assert len(matches) == 1, "historical_request_not_unique"
    return matches[0]


def row_identity(collection, row):
    for keys in IDENTITIES.get(collection, []):
        if isinstance(row, dict) and all(key in row for key in keys):
            return {"identity": {key: deepcopy(row[key]) for key in keys}}
    return None


def at(value, path):
    for token in path:
        if isinstance(token, dict):
            matches = [row for row in value if isinstance(row, dict)
                       and all(row.get(key, MISSING) == expected for key, expected in token['identity'].items())] \
                if isinstance(value, list) else []
            value = matches[0] if len(matches) == 1 else MISSING
        elif isinstance(token, str):
            value = value.get(token, MISSING) if isinstance(value, dict) else MISSING
        else:
            value = value[token] if isinstance(value, list) and 0 <= token < len(value) else MISSING
        if value is MISSING:
            break
    return value


def obligations(value):
    protected, evidence = [], []
    def walk(node, path, collection=None):
        if isinstance(node, dict):
            if node.get('verdict') in ('revise', 'contradicted'):
                protected.append({'path': path + ['verdict'], 'value': node['verdict'], 'kind': 'explicit_negative'})
            for key, expected in [('realized', False), ('new_event_or_progress', False),
                                  ('consistent_with_history', False), ('duplicates_existing_event', True)]:
                if key in node and node[key] is expected:
                    protected.append({'path': path + [key], 'value': expected, 'kind': 'explicit_negative'})
            if path and path[-1] == 'request_acceptance' and node.get('required') is True:
                protected.append({'path': path + ['required'], 'value': True, 'kind': 'declared_applicability'})
            for key, child in node.items():
                if key in EVIDENCE_KEYS and isinstance(child, list) and child:
                    evidence.append({'path': path + [key], 'minimum_count': len(child)})
                walk(child, path + [key], key)
        elif isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, path + [row_identity(collection, child) or index])
    walk(value, [])
    return protected, evidence


def replay_audit(failures):
    first, second = historical_request('001'), historical_request('002')
    first_raw, second_raw = read(first / 'RESPONSE.json')['text'], read(second / 'RESPONSE.json')['text']
    previous, current = json.loads(first_raw), json.loads(second_raw)
    repair = request_context(read(second / 'REQUEST.json'), event=True)['targeted_repair']
    old_errors = read(first / 'VALIDATION.json')['errors']
    check(repair.get('previous_raw') == first_raw and repair.get('errors') == old_errors,
          'historical_retry_not_exact_first_response_and_errors', failures)
    check(not any(error.get('code', '').startswith('event_state_history_absence_not_') for error in old_errors),
          'unexpected_contract_negative_exemption_requires_auditor_update', failures)
    replay, receipt = read(RUN / 'REPLAY_REVIEW.json'), read(RUN / 'REPLAY_VALIDATION.json')
    check(replay == current, 'replayed_review_not_exact_historical_raw', failures)
    old_full = read(second / 'FULL_VALIDATION_BEFORE_CORRECTION_GATE.json')
    check(old_full.get('status') == 'PASS' and old_full.get('requires_replan') is True
          and old_full.get('event_state_sha256') == digest(current),
          'historical_review_not_original_full_validator_pass', failures)
    check(read(second / 'VALIDATION.json').get('status') == 'REJECT'
          and read(second / 'VALIDATION.json').get('reason') == 'correction_protected_judgment_erased',
          'historical_failure_classification_changed', failures)
    for key, path in [('first_raw', first / 'RESPONSE.json'), ('second_raw', second / 'RESPONSE.json'),
                      ('original_reject', second / 'VALIDATION.json')]:
        check(receipt.get(key) == binding(path), 'replay_origin_binding_wrong', failures, field=key)
    check(receipt.get('status') == 'PASS' and receipt.get('requires_replan') is True
          and receipt.get('full_validation_pass') is True and receipt.get('semantic_certified') is False
          and 'zero new model calls' in receipt.get('classification', ''),
          'replay_receipt_overclaims_or_missing_validation', failures)
    correction = receipt['correction_progress']
    check(correction.get('status') == 'PASS' and correction.get('errors') == []
          and correction.get('full_validation_required') is True and correction.get('semantic_certified') is False
          and correction.get('previous_document_sha256') == digest(previous, ascii_only=True)
          and correction.get('response_document_sha256') == digest(current, ascii_only=True),
          'replay_correction_not_bound_to_original_documents', failures)
    protected, evidence = obligations(previous)
    check(correction.get('protected_judgment_count') == len(protected) == 7
          and correction.get('evidence_obligation_count') == len(evidence) == 21
          and correction.get('contract_invalid_negative_owner_exemptions') == [],
          'replay_obligations_reduced_or_exempted', failures)
    old_node = previous['card_checks'][0]['function_check']
    card_id = previous['card_checks'][0]['card_id']
    prefix = ['card_checks', {'identity': {'card_id': card_id}}, 'function_check']
    destination = prefix + ['comparison']
    check(set(old_node) == COMPARISON_KEYS and 'comparison' not in old_node
          and old_node == current['card_checks'][0]['function_check']['comparison'],
          'replay_not_exact_flattened_comparison_relocation', failures)
    wrapper_error = next(row for row in old_errors if row['path'] == 'card_checks[0].function_check')
    missing_error = next(row for row in old_errors if row['path'] == 'card_checks[0].function_check.comparison')
    check(wrapper_error.get('code') == 'event_state_fields_invalid'
          and set(wrapper_error.get('allowed', [])) == WRAPPER_KEYS
          and missing_error.get('code') == 'event_identity_fields_invalid'
          and missing_error.get('actual', MISSING) is None
          and set(missing_error.get('allowed', [])) == COMPARISON_KEYS,
          'replay_relocation_lacks_exact_original_validator_authorization', failures)
    mappings = []
    for kind, rows in [('protected_judgment', protected), ('evidence_obligation', evidence)]:
        for row in rows:
            old_path = row['path']
            new_path = destination + old_path[len(prefix):] if old_path[:len(prefix)] == prefix else old_path
            if new_path != old_path:
                mappings.append({'obligation_type': kind, 'old_path': old_path, 'new_path': new_path,
                                 **{key: value for key, value in row.items() if key != 'path'}})
            actual = at(current, new_path)
            if kind == 'protected_judgment':
                check(type(actual) is type(row['value']) and actual == row['value'],
                      'replay_protected_value_not_preserved', failures, path=new_path)
            else:
                check(isinstance(actual, list) and len(actual) >= row['minimum_count'],
                      'replay_evidence_obligation_not_preserved', failures, path=new_path)
    expected_relocation = {'rule': 'exact_flattened_bound_card_function_comparison',
        'original_path': 'card_checks[0].function_check',
        'destination_path': 'card_checks[0].function_check.comparison',
        'original_prefix': prefix, 'destination_prefix': destination,
        'trigger_errors': [wrapper_error, missing_error], 'obligation_mappings': mappings}
    check(len(mappings) == 6 and correction.get('wrapper_relocations') == [expected_relocation],
          'replay_relocation_receipt_not_exact_authorized_obligations', failures)
    expected_targets = [{'code': row['code'], 'path': row['path'], 'changed': True} for row in old_errors]
    check(correction.get('targets') == expected_targets, 'replay_error_targets_changed_or_not_fixed', failures)
    feedback = read(RUN / 'SEMANTIC_REPLAN_FEEDBACK.json')
    negatives = [row for row in old_full['plan_diagnostics'] if row.get('requires_replan') is True]
    # Independently tie the seven binding findings to raw checks. No lexical or semantic classifier.
    raw_findings = []
    for card in current['card_checks']:
        if card['function_check']['verdict'] == 'revise':
            raw_findings.append({**deepcopy(card['function_check']), 'card_id': card['card_id'],
                                 'check_kind': 'card_event_function', 'requires_replan': True, 'evidence_bases': {}})
        for collection, kind in [('step_checks', 'card_step'), ('dependency_checks', 'card_dependency')]:
            for row in card[collection]:
                if row['verdict'] == 'revise':
                    raw_findings.append({**deepcopy(row), 'card_id': card['card_id'], 'check_kind': kind,
                                         'requires_replan': True, 'evidence_bases': {}})
    check(negatives == raw_findings and len(negatives) == 7,
          'historical_binding_diagnostics_not_all_raw_card_negatives', failures)
    data = read(RUN / 'INPUT.json')
    check(feedback.get('failure') == 'event_state_requires_replan'
          and feedback.get('previous_response') == data['initial_plan']
          and feedback.get('plan_checks') == negatives,
          'semantic_feedback_changed_or_dropped_binding_negatives', failures)
    audit_review_output(replay, data['context'], failures, stage='historical_replay')
    return {'raw_first_sha256': sha(first / 'RESPONSE.json'), 'raw_second_sha256': sha(second / 'RESPONSE.json'),
            'replayed_review_sha256': sha(RUN / 'REPLAY_REVIEW.json'),
            'protected_judgments': len(protected), 'evidence_obligations': len(evidence),
            'authorized_path_mappings': len(mappings), 'binding_negative_findings': len(negatives),
            'new_model_calls': 0, 'historical_full_validation_receipt_sha256': sha(second / 'FULL_VALIDATION_BEFORE_CORRECTION_GATE.json')}


def audit_planner_context(payload, original, snapshot, feedback, failures, *, initial, previous_raw=None, errors=None):
    ctx = request_context(payload, event=False)
    check(ctx.get('event_state') == snapshot, 'planner_snapshot_not_exact_replayed_review', failures)
    common = dict(ctx); common.pop('event_state', None); common.pop('targeted_repair', None)
    check(common == original, 'planner_original_context_changed', failures)
    repaired = ctx.get('targeted_repair')
    if initial:
        check(repaired == feedback, 'planner_first_feedback_not_exact_binding_negative_artifact', failures)
    else:
        check(isinstance(repaired, dict) and repaired.get('previous_response') == previous_raw
              and repaired.get('errors') == errors, 'planner_structural_retry_not_actual_prior_failure', failures)
    frozen_feedback = (RUN / 'feedback' / ROOT / 'dataset_new_event__planner.txt').read_text(encoding='utf-8')
    check(payload['messages'][0]['content'].count(frozen_feedback) == 1,
          'planner_frozen_training_feedback_changed', failures)


def frozen_audit(manifest, failures, *, live):
    expected = {'version': 'v101_wrapper_feedback_continuation_v1', 'root_id': ROOT,
                'max_logical_requests': 4, 'http_max': 8, 'max_planner_attempts': 2,
                'max_event_review_attempts': 2, 'max_tokens': 4096, 'fresh_source_count': 0,
                'training_allowed': False, 'prior_phase_logical_requests': 3, 'prior_phase_http_attempts': 3,
                'joint_max_logical_requests': 12, 'joint_max_http_attempts': 24}
    for key, value in expected.items():
        check(manifest.get(key) == value, 'manifest_value_changed', failures,
              field=key, expected=value, actual=manifest.get(key))
    files = manifest.get('files', []); bindings = {row['path']: row['sha256'] for row in files}
    check(len(bindings) == len(files), 'duplicate_manifest_path', failures)
    changed = [path for path, expected_sha in bindings.items() if not Path(path).is_file() or sha(path) != expected_sha]
    check(not changed, 'manifest_frozen_file_changed', failures, paths=changed)
    required = {BASE / 'evidence/regression_driver.py', BASE / 'PRECALL_PLAN_ZH.md',
                BASE / 'PRESERVATION_BEFORE.json', RUN / 'provider_profile.json', RUN / 'CAPACITY_PREFLIGHT.json',
                RUN / 'REPLAY_VALIDATION.json', RUN / 'REPLAY_REVIEW.json', RUN / 'SEMANTIC_REPLAN_FEEDBACK.json',
                RUN / 'INPUT.json', OLD_RUN / 'inputs/19053.json', OLD_RUN / 'REPORT.json', OLD / 'COMPLETE_MANIFEST.json'}
    required.update((BASE / 'expansion_engine').glob('*.py')); required.update(LEGACY.glob('*.py'))
    required.update(BASE / 'expansion_engine' / name for name in ['SOURCE_REGISTRY.json', 'SOURCE_SELECTION.json'])
    for prefix in ('001', '002'):
        required.update(historical_request(prefix) / name for name in ['REQUEST.json', 'PROOF.json', 'RESPONSE.json', 'VALIDATION.json'])
    required.add(Path(manifest['prepared_request']['path']))
    missing = sorted(str(path) for path in required if str(path.resolve()) not in bindings)
    check(not missing, 'required_dependency_not_frozen', failures, paths=missing)
    preservation = read(BASE / 'PRESERVATION_BEFORE.json')
    historical_changed = [row['path'] for row in preservation['files'] if not Path(row['path']).is_file()
                          or sha(row['path']) != row['sha256']]
    check(not historical_changed, 'historical_protected_file_changed', failures, paths=historical_changed)
    protected = {row['path']: row['sha256'] for row in preservation['files']}
    old_seal = read(OLD / 'COMPLETE_MANIFEST.json')
    # The sealed v100 inventory and every listed member must remain bound by preservation.
    check(str((OLD / 'COMPLETE_MANIFEST.json').resolve()) in protected,
          'v100_sealed_manifest_not_preserved', failures)
    for row in old_seal['files']:
        path = Path(row['path']); path = path if path.is_absolute() else OLD / path
        check(protected.get(str(path.resolve())) == sha(path), 'v100_sealed_member_not_preserved', failures, path=str(path))
    qa_files = list((BASE / 'qa_engine').glob('*.py'))
    qa_changed = [path.name for path in qa_files if not (OLD / 'qa_engine' / path.name).is_file()
                  or sha(path) != sha(OLD / 'qa_engine' / path.name)]
    check(len(qa_files) == 15 and not qa_changed, 'qa_engine_changed_outside_scope', failures, paths=qa_changed)
    check(sha(BASE / 'expansion_engine/realization_review.py') == sha(OLD / 'expansion_engine/realization_review.py'),
          'scene_review_changed_outside_scope', failures)
    data = read(RUN / 'INPUT.json'); original = data['context']
    check(data == read(OLD_RUN / 'inputs/19053.json') and data.get('exposed_regression_only') is True,
          'historical_full_input_changed', failures)
    for folder in ['feedback', 'sources']:
        for old_path in (OLD_RUN / folder).rglob('*'):
            if old_path.is_file():
                path = RUN / folder / old_path.relative_to(OLD_RUN / folder)
                check(path.is_file() and sha(path) == sha(old_path), 'historical_archive_copy_changed', failures, path=str(path))
    check(read(RUN / 'sources' / f'{ROOT}.json')['turns'] == original['cumulative_history'],
          'source_history_not_exact_complete_input', failures)
    prior = read(OLD_RUN / 'REPORT.json')
    check(prior.get('logical_requests') == manifest['prior_phase_logical_requests']
          and prior.get('http_attempts') == manifest['prior_phase_http_attempts'], 'prior_budget_accounting_changed', failures)
    prepared_binding = manifest['prepared_request']; prepared_path = Path(prepared_binding['path'])
    check(prepared_binding == binding(prepared_path) and bindings.get(str(prepared_path.resolve())) == prepared_binding['sha256'],
          'prepared_request_binding_wrong', failures)
    prepared = read(prepared_path); payload, proof = prepared['payload'], prepared['proof']
    snapshot, feedback = read(RUN / 'REPLAY_REVIEW.json'), read(RUN / 'SEMANTIC_REPLAN_FEEDBACK.json')
    audit_planner_context(payload, original, snapshot, feedback, failures, initial=True)
    check(payload.get('max_tokens') == 4096 and proof.get('payload_sha256') == digest(payload)
          and proof.get('role') == 'planner' and proof.get('root_id') == ROOT
          and proof.get('history_sha256') == digest(original['cumulative_history'])
          and proof.get('latest_state_sha256') == digest(original['latest_state'])
          and proof.get('event_state_sha256') == digest(snapshot), 'prepared_planner_proof_wrong', failures)
    capacity = read(RUN / 'CAPACITY_PREFLIGHT.json')
    check(capacity.get('status') == 'PASS' and capacity.get('network_calls') == 0
          and capacity.get('prepared_request_sha256') == prepared_binding['sha256']
          and capacity.get('capacity', {}).get('status') == 'PASS'
          and capacity.get('capacity', {}).get('truncation') is False, 'capacity_preflight_not_complete_offline_pass', failures)
    forbidden = [str(RUN / name) for name in ('SOURCE_RESERVATIONS.json', 'UNIQUE_EVENT_ALLOCATION.json',
                 'dataset_cards_result.json', 'shards', 'source_reviews', 'materialized_sources') if (RUN / name).exists()]
    check(not forbidden, 'regression_contains_new_source_artifacts', failures, paths=forbidden)
    if not live:
        check(not (RUN / 'LIVE_STARTED.json').exists() and not (RUN / 'requests').exists(),
              'preflight_already_has_live_artifacts', failures)
    return {'frozen_file_count': len(files), 'historical_protected_file_count': len(preservation['files']),
            'v100_sealed_members': len(old_seal['files']), 'unchanged_qa_files': len(qa_files),
            'history_turns': len(original['cumulative_history']), 'prepared_request_sha256': prepared_binding['sha256'],
            'replay': replay_audit(failures)}
def live_audit(manifest, failures):
    directories = sorted(path for path in (RUN / 'requests').glob('*') if path.is_dir())
    check(len(directories) <= 4, 'logical_request_budget_exceeded', failures)
    original = read(RUN / 'INPUT.json')['context']; initial_plan = read(RUN / 'INPUT.json')['initial_plan']
    snapshot = read(RUN / 'REPLAY_REVIEW.json'); feedback = read(RUN / 'SEMANTIC_REPLAN_FEEDBACK.json')
    stages, roles, http_numbers, request_rows, token_rows = [], Counter(), [], [], []
    planner_dirs, review_dirs = [], []; candidate = None; prior_planner_raw = None; prior_planner_errors = None
    prior_review_common = None; prior_review_raw = None; prior_review_errors = None
    for number, directory in enumerate(directories, 1):
        check(directory.name.startswith(f'{number:03d}-'), 'logical_request_sequence_invalid', failures, request=directory.name)
        payload, proof = read(directory / 'REQUEST.json'), read(directory / 'PROOF.json')
        stage, role = proof.get('regression_stage'), proof.get('role'); stages.append(stage); roles[role] += 1
        check(stage in ['semantic_replan', 'post_replan_event_review']
              and role == ('planner' if stage == 'semantic_replan' else 'event_state_extractor'),
              'unexpected_stage_or_role', failures, request=directory.name)
        check(proof.get('root_id') == proof.get('regression_root') == ROOT
              and proof.get('route') == 'dataset_new_event' and proof.get('regression_only') is True
              and proof.get('fresh_source') is False and proof.get('historical_review_replay_sha256') == sha(RUN / 'REPLAY_REVIEW.json'),
              'request_exposure_or_replay_binding_wrong', failures, request=directory.name)
        check(proof.get('payload_sha256') == digest(payload) and payload.get('max_tokens') == 4096
              and proof.get('history_sha256') == digest(original['cumulative_history'])
              and proof.get('latest_state_sha256') == digest(original['latest_state'])
              and proof.get('capacity', {}).get('status') == 'PASS'
              and proof.get('capacity', {}).get('truncation') is False,
              'request_hash_budget_history_or_capacity_wrong', failures, request=directory.name)
        validation = read(directory / 'VALIDATION.json') if (directory / 'VALIDATION.json').exists() else None
        if role == 'planner':
            planner_dirs.append(directory)
            audit_planner_context(payload, original, snapshot, feedback, failures, initial=len(planner_dirs) == 1,
                                  previous_raw=prior_planner_raw, errors=prior_planner_errors)
            check(proof.get('event_state_sha256') == digest(snapshot), 'planner_snapshot_hash_wrong', failures)
            if len(planner_dirs) == 1:
                prepared = read(manifest['prepared_request']['path'])
                check(payload == prepared['payload'], 'first_live_planner_not_frozen_prepared_payload', failures)
                check(all(proof.get(key) == value for key, value in prepared['proof'].items()),
                      'first_live_planner_prepared_proof_changed', failures)
            else:
                check(candidate is None and prior_planner_errors is not None,
                      'planner_retry_without_actual_structural_rejection', failures)
            if (directory / 'CANDIDATE_PLAN.json').exists():
                candidate = read(directory / 'CANDIDATE_PLAN.json')
                check(validation is not None and validation.get('status') == 'PASS'
                      and json.loads(read(directory / 'RESPONSE.json')['text']) == candidate,
                      'candidate_not_actual_validated_raw', failures)
            if (directory / 'RESPONSE.json').exists():
                prior_planner_raw = read(directory / 'RESPONSE.json')['text']
            prior_planner_errors = validation.get('errors') if validation is not None and validation.get('status') == 'REJECT' else None
        else:
            review_dirs.append(directory)
            check(candidate is not None and bool(candidate.get('scenes')), 'post_review_without_valid_nonempty_candidate', failures)
            ctx = request_context(payload, event=True)
            if candidate:
                audit_event_context(ctx, original, candidate, failures, request=directory.name)
                check(proof.get('scene_plan_sha256') == digest(candidate['scenes'][0]) and proof.get('plan_attempt') == 1,
                      'post_review_plan_proof_wrong', failures)
            if len(review_dirs) == 1:
                check('targeted_repair' not in ctx, 'first_post_review_has_repair', failures)
            else:
                repair = ctx.get('targeted_repair')
                check(isinstance(repair, dict) and repair.get('previous_raw') == prior_review_raw
                      and repair.get('errors') == prior_review_errors, 'review_retry_not_actual_prior_failure', failures)
            common = dict(ctx); common.pop('targeted_repair', None)
            if prior_review_common is not None:
                check(common == prior_review_common, 'review_retry_changed_frozen_context', failures)
            prior_review_common = common
            if (directory / 'RESPONSE.json').exists():
                prior_review_raw = read(directory / 'RESPONSE.json')['text']
            prior_review_errors = validation.get('errors') if validation is not None and validation.get('status') == 'REJECT' else None
        attempts = sorted(directory.glob('HTTP_ATTEMPT*.json')); statuses = []
        check(1 <= len(attempts) <= 2, 'http_attempt_count_invalid', failures, request=directory.name)
        for index, path in enumerate(attempts, 1):
            attempt = read(path); http_numbers.append(attempt.get('http_attempt_number'))
            check(attempt.get('payload_sha256') == digest(payload) and attempt.get('actual_payload_equals_frozen') is True
                  and attempt.get('request_attempt') == index, 'actual_http_request_not_frozen', failures, request=directory.name)
            result = read(directory / f'HTTP_RESULT_{index:02d}.json'); statuses.append(result.get('http_status'))
            if index > 1:
                check(statuses[-2] in [503, 504], 'unallowed_transport_retry', failures, request=directory.name)
            response_path = directory / f'HTTP_RESPONSE_{index:02d}.json'
            if response_path.exists():
                body = read(response_path); token_rows.append(body.get('metadata', {}))
        if (directory / 'RESPONSE.json').exists():
            response = read(directory / 'RESPONSE.json')
            check(response == {**read(directory / f'HTTP_RESPONSE_{len(attempts):02d}.json'), 'attempts': response.get('attempts')},
                  'success_raw_not_last_actual_http_body', failures, request=directory.name)
            check(response.get('metadata', {}).get('finish_reason') == 'stop', 'incomplete_success_response', failures)
        request_rows.append({'request': directory.name, 'stage': stage, 'role': role,
                             'http_attempts': len(attempts), 'http_statuses': statuses,
                             'validation': validation.get('status') if validation else None})
    check(len(planner_dirs) <= 2 and len(review_dirs) <= 2
          and stages == sorted(stages, key=lambda stage: ['semantic_replan', 'post_replan_event_review'].index(stage)),
          'stage_budget_or_order_invalid', failures)
    check(sorted(http_numbers) == list(range(1, len(http_numbers) + 1)) and len(http_numbers) <= 8,
          'http_sequence_or_budget_invalid', failures)
    report = read(RUN / 'REPORT.json'); result = report.get('result', {})
    check(report.get('status') == 'COMPLETE' and result.get('root_id') == ROOT
          and result.get('historical_replay') == 'PASS, not a new model review', 'report_identity_or_replay_claim_wrong', failures)
    check(report.get('logical_requests') == len(directories) and report.get('http_attempts') == len(http_numbers)
          and report.get('joint_logical_requests') == 3 + len(directories)
          and report.get('joint_http_attempts') == 3 + len(http_numbers)
          and report['joint_logical_requests'] <= 7 <= 12 and report['joint_http_attempts'] <= 11 <= 24,
          'new_or_joint_accounting_wrong', failures)
    for key in ['new_sources', 'new_dialogue_turns', 'ready_sources', 'new_questions', 'solver_calls', 'training_jobs']:
        check(report.get(key) == 0, 'regression_claims_production_output', failures, field=key)
    check(report.get('semantic_certified') is False, 'report_overclaims_semantic_reliability', failures)
    actual_planner_rows = []
    for index, path in enumerate(planner_dirs, 1):
        if (path / 'VALIDATION.json').exists():
            actual_planner_rows.append({'attempt': index, 'directory': str(path), 'validation': read(path / 'VALIDATION.json')})
    check(result.get('planner_attempts') == actual_planner_rows, 'report_planner_attempts_not_actual_validations', failures)
    post_summary = result.get('post_replan_review'); post_path = RUN / 'POST_REPLAN_REVIEW.json'
    post = read(post_path) if post_path.exists() else None
    if isinstance(post_summary, dict):
        check(post_summary.get('logical_requests') == len(review_dirs)
              and post_summary.get('structurally_valid') is (post is not None), 'post_review_summary_not_actual', failures)
        if post is not None:
            audit_review_output(post, original, failures, stage='post_replan')
            last = review_dirs[-1]; validation = read(last / 'VALIDATION.json')
            check(post == json.loads(read(last / 'RESPONSE.json')['text']) and validation.get('status') == 'PASS'
                  and validation.get('event_state_sha256') == digest(post)
                  and validation.get('requires_replan') == post_summary.get('requires_replan'),
                  'post_review_not_actual_last_validated_raw', failures)
    else:
        check(not post_path.exists(), 'unreported_post_review_artifact', failures)
    check(result.get('endpoint_review_without_replan') is
          (isinstance(post_summary, dict) and post_summary.get('structurally_valid') is True
           and post_summary.get('requires_replan') is False), 'report_endpoint_not_supported_by_actual_review', failures)
    prior_tokens = read(OLD / 'evidence/REGRESSION_LIVE_AUDIT.json')['summary']['live']['provider_tokens']
    tokens = {key: sum(row.get(key) or 0 for row in token_rows) for key in ['input_tokens', 'output_tokens', 'total_tokens']}
    return {'logical_requests': len(directories), 'http_attempts': len(http_numbers), 'role_counts': dict(roles),
            'joint_logical_requests': len(directories) + 3, 'joint_http_attempts': len(http_numbers) + 3,
            'requests': request_rows, 'provider_tokens': tokens,
            'joint_provider_tokens': {key: tokens[key] + prior_tokens[key] for key in tokens},
            'candidate_changed_from_historical_plan': candidate != initial_plan if candidate is not None else None,
            'candidate_sha256': digest(candidate) if candidate is not None else None,
            'endpoint_review_without_replan': result.get('endpoint_review_without_replan'), 'report_sha256': sha(RUN / 'REPORT.json')}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=['preflight', 'live'], required=True)
    parser.add_argument('--output', type=Path); args = parser.parse_args()
    failures, summary = [], {}; manifest_path = RUN / 'REGRESSION_MANIFEST.json'
    try:
        manifest = read(manifest_path); summary['frozen_inputs'] = frozen_audit(manifest, failures, live=args.phase == 'live')
        if args.phase == 'live': summary['live'] = live_audit(manifest, failures)
    except Exception as error:
        failures.append({'code': 'audit_exception', 'error': f'{type(error).__name__}:{error}'})
    output = {'version': 'v101_independent_regression_audit_v1', 'phase': args.phase,
              'status': 'PASS' if not failures else 'FAIL', 'manifest_path': str(manifest_path),
              'manifest_sha256': sha(manifest_path) if manifest_path.is_file() else None,
              'audited_at_utc': datetime.now(timezone.utc).isoformat(), 'audit_script_sha256': sha(__file__),
              'summary': summary, 'failures': failures, 'provider_calls_by_auditor': 0,
              'auditor_imports_engine': False, 'semantic_certified': False,
              'classification': 'Exact exposed historical replay plus treatment-only continuation; no fresh yield or paired causal claim.'}
    path = args.output or BASE / 'evidence' / f'REGRESSION_{args.phase.upper()}_AUDIT.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps({'phase': args.phase, 'status': output['status'], 'failures': failures,
                      'manifest_sha256': output['manifest_sha256'], 'output': str(path)}, ensure_ascii=False))
    raise SystemExit(0 if not failures else 1)


if __name__ == '__main__': main()
