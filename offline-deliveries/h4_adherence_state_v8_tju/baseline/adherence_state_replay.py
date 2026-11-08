"""Two-call frozen-scene replay of the target-state adherence prompt.

Preparation never reads credentials or constructs an HTTP client. Execution
uses the shared bounded runtime and sends only the two frozen adherence slots;
continuation eligibility is materialized locally and is never executed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys

from ..core import atomic_json, atomic_text
from ..live_backends import load_challenger_config
from ..h4_writer_model_comparison_v1.transport import ProfileClient, build_payload
from .development_runtime import (
    BoundedDevelopmentCaller, DevelopmentContentError, DevelopmentRuntimeError,
    capacity_check,
)
from .development_validate import check_freeze, cost_summary, now, public_profile, sha, tokenizer_from
from .realizer_adherence import (
    build_adherence_auditor_input, build_adherence_auditor_prompt,
    validate_and_materialize_adherence_audit,
)
from .scene_continuation import build_scene_continuation_request


DEFAULT_SOURCE = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_gate_veto_ablation_v1_live_1')
DEFAULT_RUN = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_adherence_state_v8_replay_live_1')
CASES = ('cc-train-episode-122711__HARD', 'cc-train-episode-122711__ADVISORY')
MAX_TOKENS = 4096
INPUT_MARKER = '\nAUDITOR INPUT\n'
OUTPUT_MARKER = '\n\nOUTPUT ONLY VALID JSON.'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def input_text(prompt):
    if prompt.count(INPUT_MARKER) != 1:
        raise ValueError('one_auditor_input_block_required')
    body = prompt.split(INPUT_MARKER, 1)[1]
    if body.count(OUTPUT_MARKER) != 1:
        raise ValueError('one_auditor_output_boundary_required')
    return body.split(OUTPUT_MARKER, 1)[0]


def verify_request_change(old_request, new_payload, old_prompt, new_prompt, auditor_input):
    """Only user prompt rules may change; the complete serialized input stays exact."""
    old_payload = deepcopy(old_request['payload'])
    old_messages = old_payload.pop('messages')
    new_parameters = deepcopy(new_payload)
    new_messages = new_parameters.pop('messages')
    if old_payload != new_parameters:
        raise ValueError('source_request_parameters_changed')
    if old_messages != [{'role': 'user', 'content': old_prompt}]:
        raise ValueError('source_prompt_and_actual_messages_differ')
    if new_messages != [{'role': 'user', 'content': new_prompt}]:
        raise ValueError('new_messages_role_or_structure_changed')
    old_input, new_input = input_text(old_prompt), input_text(new_prompt)
    if old_input != new_input or json.loads(new_input) != auditor_input:
        raise ValueError('frozen_auditor_input_changed')
    return {'parameters_identical': True, 'messages_role_order_identical': True,
            'auditor_input_bytes_identical': True, 'prompt_rules_only_change': True}


class AdherenceStateCaller(BoundedDevelopmentCaller):
    MAX_HTTP_ATTEMPTS = 2
    MAX_CALLS_PER_ROOT = 1

    def __init__(self, run_root, profile, tokenizer, *, http_max=2, per_root_max=1, **kwargs):
        if http_max != 2 or per_root_max != 1:
            raise DevelopmentRuntimeError('fixed_adherence_replay_budget_required')
        super().__init__(run_root, profile, tokenizer, http_max=2, per_root_max=1, **kwargs)

    def _validate_slot(self, root_id, stage):
        if root_id not in CASES:
            raise DevelopmentRuntimeError('adherence_replay_case_not_allowed')
        if stage != 'adherence':
            raise DevelopmentRuntimeError('adherence_replay_stage_not_allowed')


def _dependencies():
    """Freeze imported project dependencies, excluding tests and unrelated history."""
    scripts = Path(__file__).resolve().parents[3]
    paths = {Path(__file__).resolve()}
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if (path.is_relative_to(scripts) and path.suffix == '.py'
                    and not path.name.startswith('test_')):
                paths.add(path)
    return paths


def prepare(run_root=DEFAULT_RUN, *, source_run=DEFAULT_SOURCE, tokenizer=None):
    run_root, source_run = Path(run_root), Path(source_run)
    if run_root.exists():
        raise FileExistsError('run_exists_no_overwrite_or_automatic_resume')
    source_manifest_path = source_run / 'freeze_manifest.json'
    source_manifest = read(source_manifest_path)
    profile_path = source_run / 'frozen/provider_profile.json'
    profile = read(profile_path)  # Public frozen profile; never open config.json here.
    if source_manifest['public_profile'] != profile:
        raise ValueError('source_public_profile_mismatch')
    if (profile['model'] != 'Qwen/Qwen3-14B'
            or profile['endpoint'] != 'https://api.siliconflow.cn/v1/chat/completions'
            or profile['request_parameters'] != {'temperature': 0.25, 'max_tokens': 4096,
                                                  'response_format': {'type': 'json_object'}}
            or profile['output_limit'] != MAX_TOKENS
            or profile.get('http_retry_limit') != 0
            or any(profile.get(k) is not False for k in ('redirects', 'ssl_fallback', 'parameter_fallback'))):
        raise ValueError('source_adherence_service_contract_changed')
    tokenizer_path = Path(source_manifest['tokenizer_path'])
    tokenizer = tokenizer if tokenizer is not None else tokenizer_from(tokenizer_path)
    prepared, sources = [], {source_manifest_path, profile_path}
    for case in CASES:
        directory = source_run / 'cases' / case
        paths = {'plan.json': directory / 'planner/materialized_plan.json',
                 'child.json': directory / 'child_dialogue.json',
                 'auditor_input.json': directory / 'realizer_adherence/input.json',
                 'source_prompt.txt': directory / 'realizer_adherence/prompt.txt',
                 'source_request.json': source_run / 'execution/calls' / case / 'adherence/request.json'}
        plan, child = read(paths['plan.json']), read(paths['child.json'])
        auditor_input = read(paths['auditor_input.json'])
        if build_adherence_auditor_input(plan=plan, child=child) != auditor_input:
            raise ValueError('source_plan_child_and_auditor_input_disagree')
        prompt = build_adherence_auditor_prompt(auditor_input=auditor_input)
        old_prompt = paths['source_prompt.txt'].read_text(encoding='utf-8')
        request = read(paths['source_request.json'])
        if request['endpoint'] != profile['endpoint']:
            raise ValueError('source_endpoint_changed')
        payload = build_payload(profile, prompt['prompt'])
        identity = verify_request_change(request, payload, old_prompt, prompt['prompt'], auditor_input)
        capacity = capacity_check(profile, tokenizer, payload['messages'], MAX_TOKENS)
        if capacity['status'] != 'PASS':
            raise ValueError('adherence_replay_context_or_output_limit_exceeded')
        prepared.append((case, paths, prompt, payload, identity, capacity))
        sources.update(paths.values())
    run_root.mkdir(parents=True)
    atomic_json(run_root / 'frozen/provider_profile.json', profile)
    capacities = []
    for case, paths, prompt, payload, identity, capacity in prepared:
        destination = run_root / 'frozen' / case
        destination.mkdir(parents=True)
        for name, path in paths.items():
            shutil.copyfile(path, destination / name)
        atomic_json(destination / 'prompt.json', prompt)
        atomic_text(destination / 'prompt.txt', prompt['prompt'])
        atomic_json(destination / 'payload.json', payload)
        atomic_json(destination / 'identity.json', identity)
        atomic_json(destination / 'capacity.json', capacity)
        capacities.append({'case_id': case, **capacity})
    sources.update(p for p in tokenizer_path.iterdir() if p.is_file())
    sources.update(_dependencies())
    sources.update(p for p in (run_root / 'frozen').rglob('*') if p.is_file())
    manifest = {
        'version': 'adherence_state_v8_frozen_replay_v1', 'status': 'PREPARED',
        'prepared_at': now(), 'source_run': str(source_run.resolve()),
        'config_path': source_manifest['config_path'], 'tokenizer_path': str(tokenizer_path),
        'case_order': list(CASES), 'public_profile': profile, 'stage': 'adherence',
        'maximum_http_attempts': 2, 'maximum_logical_calls': 2,
        'maximum_calls_per_case': 1, 'maximum_output_tokens_per_call': MAX_TOKENS,
        'maximum_output_tokens_total': 2 * MAX_TOKENS, 'capacities': capacities,
        'estimated_input_tokens_total': sum(c['input_tokens'] for c in capacities),
        'automatic_retries': 0, 'automatic_resume': False, 'parameter_fallback': False,
        'verified_cost': 'UNKNOWN', 'external_calls': 0,
        'interpretation': 'Same frozen full inputs; revised prompt rules only. Historical outputs '
                          'are diagnostic references, not a fresh randomized prompt comparison. '
                          'Program contract validity is not semantic correctness.',
        'files': [{'path': str(p.resolve()), 'sha256': sha(p)} for p in sorted(sources)],
    }
    atomic_json(run_root / 'freeze_manifest.json', manifest)
    atomic_json(run_root / 'run_status.json', {'status': 'PREPARED', 'external_calls': 0})
    return manifest


def execute(run_root=DEFAULT_RUN, *, http_budget, client_factory=None, tokenizer=None):
    if type(http_budget) is not int or http_budget != 2:
        raise ValueError('explicit_http_budget_two_required')
    run_root = Path(run_root)
    manifest = read(run_root / 'freeze_manifest.json')
    check_freeze(manifest)
    if (manifest['case_order'] != list(CASES) or manifest['stage'] != 'adherence'
            or manifest['maximum_http_attempts'] != 2
            or manifest['maximum_calls_per_case'] != 1
            or manifest['maximum_output_tokens_per_call'] != MAX_TOKENS):
        raise ValueError('frozen_replay_contract_changed')
    if (run_root / 'execution').exists():
        raise FileExistsError('execution_exists_no_automatic_resume')
    profile = manifest['public_profile']
    if client_factory is None:
        if public_profile(manifest['config_path']) != profile:
            raise ValueError('public_provider_configuration_changed')
        credential = (os.environ.get(profile['api_key_env'])
                      or load_challenger_config(Path(manifest['config_path'])).api_key)
        def client_factory(stage_profile, *, on_http_attempt):
            return ProfileClient(stage_profile, credential=credential, on_http_attempt=on_http_attempt)
    tokenizer = tokenizer if tokenizer is not None else tokenizer_from(manifest['tokenizer_path'])
    active = [None]
    def before_stage(case, stage, prompt, maximum):
        check_freeze(manifest)
        frozen = run_root / 'frozen' / case
        if (case != active[0] or stage != 'adherence' or maximum != MAX_TOKENS
                or prompt != read(frozen / 'prompt.json')
                or build_payload(profile, prompt['prompt']) != read(frozen / 'payload.json')):
            raise ValueError('call_outside_frozen_adherence_schedule')
    outcomes = []
    with AdherenceStateCaller(run_root / 'execution', profile, tokenizer,
            client_factory=client_factory, before_stage=before_stage) as caller:
        for case in CASES:
            if caller.state['status'] == 'STOPPED':
                break
            active[0] = case
            frozen, target = run_root / 'frozen' / case, run_root / 'cases' / case
            outcome = {'case_id': case, 'semantic_quality': 'PENDING_CODEX_OFFLINE_DIAGNOSIS',
                       'continuation_calls': 0, 'other_stage_calls': 0}
            try:
                value = caller.call(case, 'adherence', read(frozen / 'prompt.json'), MAX_TOKENS)
                plan, child = read(frozen / 'plan.json'), read(frozen / 'child.json')
                validation = validate_and_materialize_adherence_audit(value, plan=plan, child=child)
                atomic_json(target / 'output.json', value)
                atomic_json(target / 'validation.json', validation)
                outcome['contract_valid'] = validation['valid']
                if validation['valid']:
                    audit = validation['audit']
                    atomic_json(target / 'audit.json', audit)
                    request = build_scene_continuation_request(plan=plan, child=child, adherence_audit=audit)
                    atomic_json(target / 'continuation_eligibility_only.json', request)
                    outcome.update(status='COMPLETED', scene_count=len(audit['scenes']),
                        eligible_continuation_scene_count=request['eligible_scene_count'])
                else:
                    outcome.update(status='CONTENT_OR_CONTRACT_FAILURE', errors=validation['errors'])
            except DevelopmentContentError as error:
                outcome.update(status='CONTENT_OR_CONTRACT_FAILURE', errors=[error.status])
            except DevelopmentRuntimeError as error:
                outcome.update(status='TECHNICAL_FAILURE_STOPPED', errors=[error.status])
            except Exception:
                caller._stop('local_replay_processing_failure')
                outcome.update(status='LOCAL_FAILURE_STOPPED', errors=['local_replay_processing_failure'])
            outcome['cost'] = cost_summary([r for r in caller.records if r['root_id'] == case])
            atomic_json(target / 'outcome.json', outcome)
            outcomes.append(outcome)
        summary = {'status': 'STOPPED_TECHNICAL' if caller.state['status'] == 'STOPPED' else 'FINISHED',
            'completed_at': now(), 'outcomes': outcomes, 'cost': cost_summary(caller.records),
            'unsent_cases': [c for c in CASES if c not in {
                r['root_id'] for r in caller.records if r['http_attempt_count'] > 0}],
            'continuation_calls': 0, 'other_stage_calls': 0,
            'semantic_quality': 'PENDING_CODEX_OFFLINE_DIAGNOSIS'}
        atomic_json(run_root / 'run_status.json', summary)
        atomic_json(run_root / 'generation_summary.json', summary)
        atomic_json(run_root / 'reports/cost.json', summary['cost'])
        atomic_text(run_root / 'reports/RUN_RESULT.md', '\n'.join([
            '# Frozen adherence replay', '', f"Status: {summary['status']}",
            f"HTTP attempts: {summary['cost']['http_attempts']}; retries: {summary['cost']['retries']}.",
            'Only adherence was called. Continuation eligibility was computed locally; no continuation was sent.',
            'Contract validity is a structural result, not semantic PASS. Content requires Codex offline diagnosis.',
            'Historical results are diagnostic references; this is not a fresh paired causal estimate.',
            f"Cost: {json.dumps(summary['cost'], ensure_ascii=False)}", '']))
        if caller.state['status'] != 'STOPPED':
            caller.state.update(status='FINISHED', finished_at=now())
            caller._save()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'execute'))
    parser.add_argument('--run-root', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--source-run', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--http-budget', type=int)
    args = parser.parse_args()
    if args.action == 'execute' and args.http_budget != 2:
        parser.error('execute requires --http-budget 2')
    result = (prepare(args.run_root, source_run=args.source_run) if args.action == 'prepare'
              else execute(args.run_root, http_budget=args.http_budget))
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
