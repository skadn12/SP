"""Bounded dated trial with the frozen SiliconFlow Qwen3-14B Writer profile."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import difflib
from pathlib import Path

import writer_trial as dated

MODEL_SOURCE = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_writer_model_comparison_v1_live_1')
MODEL_PROFILE = MODEL_SOURCE / 'profiles.json'
RUN = MODEL_SOURCE.parent / 'h4_dated_transcript_qwen14b_v1_prepare_1'
ARM = 'DATED_QWEN14B_TRANSCRIPT_ONLY'
PARAMETERS = {'temperature': 0.1, 'max_tokens': 8192, 'enable_thinking': False,
              'response_format': {'type': 'json_object'}}
read = dated.read


def prepare():
    if (RUN / 'executions').exists():
        raise RuntimeError('REFUSE_PREPARE_STARTED_RUN')
    if not RUN.exists():
        dated.prepare(RUN)
    qwen = deepcopy(read(MODEL_PROFILE)['QWEN14B_TEMPORAL'])
    assert qwen['request_parameters'] == PARAMETERS
    assert qwen['provider'] == 'siliconflow' and qwen['model'] == 'Qwen/Qwen3-14B'
    assert qwen['endpoint'] == 'https://api.siliconflow.cn/v1/chat/completions'
    qwen['status'] = 'READY_USER_REQUESTED_DATED_QWEN_TRIAL'
    qwen['source_profile_sha256'] = dated.record(MODEL_PROFILE)['sha256']
    qwen['source_profile'] = str(MODEL_PROFILE)
    dated.atomic_json(RUN / 'profile.json', qwen)
    schedule = read(RUN / 'schedule.json')
    for row in schedule:
        row['arm_id'] = ARM
    dated.atomic_json(RUN / 'schedule.json', schedule)
    manifests, identity, capacities, additional_sources = [], [], [], [MODEL_PROFILE]
    with dated.date_adapter():
        for row in schedule:
            root, slot = row['root_id'], row['slot_id']
            folder = RUN / 'inputs/TRANSCRIPT_ONLY' / root
            expanded = read(folder / 'expanded.json')
            contract = read(folder / 'contracts.json')[row['raw_index']]
            spec = dated.guided.build_prompt(expanded, contract, ordering_policy=dated.runtime.POLICY)
            location = RUN / 'policy_snapshot/primary_requests/TRANSCRIPT_ONLY' / root / slot
            prompt = (location / 'prompt.txt').read_text()
            assert prompt == spec['prompt']
            old = MODEL_SOURCE / 'policy_snapshot/primary_requests/QWEN14B_TEMPORAL' / root / slot
            original_payload = read(old / 'request.json')['payload']
            assert {k: v for k, v in original_payload.items() if k not in ('model', 'messages')} == PARAMETERS
            assert original_payload['model'] == qwen['model']
            stripped = '\n'.join(line for line in prompt.split('\n') if not line.startswith('[Conversation date:'))
            assert stripped == (old / 'prompt.txt').read_text()
            payload = dated.build_payload(qwen, prompt)
            dated.atomic_json(location / 'request.json', {**row, 'payload': payload})
            additional_sources.extend((old / 'request.json', old / 'prompt.txt'))
            manifests.append({**row, 'prompt_sha256': spec['prompt_sha256'],
                'messages_sha256': dated.sha256_bytes(dated.canonical_json(payload['messages']).encode()),
                'requested_counts': spec['expected_counts']})
            school_prompt = dated.DEFAULT_RUN / location.relative_to(RUN) / 'prompt.txt'
            identity.append({**row, 'dates_removed_equals_old_qwen_prompt': True,
                'qwen_parameters_equal_historical_payload': True, 'header_count': prompt.count('[Conversation date:'),
                'school_dated_prompt_unchanged': prompt == school_prompt.read_text(),
                'old_prompt_sha256': dated.record(old / 'prompt.txt')['sha256'],
                'dated_prompt_sha256': spec['prompt_sha256']})
            capacity = dated._capacity(qwen, payload['messages'], 8192, f'{root}:{slot}')
            assert capacity['status'] == 'PASS', capacity
            capacities.append({**capacity, **row})
    dated.runtime.write_rows(RUN / 'slot_call_manifest.jsonl', manifests)
    dated.atomic_json(RUN / 'audit/prompt_identity.json', identity)
    dated.atomic_json(RUN / 'audit/token_budget.json', {
        'budget': dated.BUDGET, 'requests': capacities,
        'primary_input_tokens_local_template': sum(c['input_tokens'] for c in capacities),
        'primary_output_ceiling': 6 * 8192, 'all_output_ceiling': 12 * 8192,
        'repair_input_tokens': 'DYNAMIC_SAME_SOURCE_PLUS_OWN_PRIMARY; validated before each request',
        'actual_usage_and_money': 'UNKNOWN_UNTIL_RESPONSE', 'pricing': 'UNKNOWN',
        'capacity_basis': 'Existing frozen provider evidence and local tokenizer, not a new service probe',
        'no_truncation': True})
    contract = read(RUN / 'execution_contract.json')
    contract.update(model=qwen['model'], provider=qwen['provider'], arm_id=ARM,
        requested_reasoning={'enable_thinking': False}, actual_reasoning_adoption='UNKNOWN',
        real_execution_authorized=True, authorization_record='audit/scope_authorization.json',
        unknown_service_capacity_reasoning_and_pricing=False, pricing='UNKNOWN')
    dated.atomic_json(RUN / 'execution_contract.json', contract)
    dated.atomic_json(RUN / 'audit/scope_authorization.json', {
        'source': 'User requests SiliconFlow Qwen3-14B with the previous default configuration to try the same dated data after the school-service timeout.',
        'scope': dated.SCOPE, 'budget': dated.BUDGET, 'arm_id': ARM,
        'primary_per_slot': 1, 'repair_per_primary_max': 1, 'automatic_retries': 0,
        'parameter_fallback': False, 'no_smoke_calls': True, 'stop_on_technical_failure': True,
        'old_tju_attempt_retained_separately': True, 'old_budget_reused': False,
        'memory_calls': 0, 'extra_model_roles': [], 'stop_after_reports': True})
    import qwen_runtime
    source_runtime = Path(dated.runtime.__file__).resolve()
    local_runtime = Path(qwen_runtime.__file__).resolve()
    diff = ''.join(difflib.unified_diff(source_runtime.read_text().splitlines(keepends=True),
        local_runtime.read_text().splitlines(keepends=True), fromfile=str(source_runtime), tofile=str(local_runtime)))
    dated.atomic_text(RUN / 'audit/qwen_runtime.diff', diff)
    dated.atomic_json(RUN / 'audit/runtime_adapter_identity.json', {
        'source_runtime': dated.record(source_runtime), 'local_runtime': dated.record(local_runtime),
        'generator_reused': True, 'ledger_reused': True,
        'changes': ['explicit Qwen provider/parameter guard', 'existing env/challenger_json credential resolution',
                    'requested disabled reasoning audit; actual adoption UNKNOWN'],
        'diff': 'audit/qwen_runtime.diff'})
    manifest_path = RUN / 'audit/frozen_manifest.json'
    manifest = read(manifest_path)
    additional_sources += [Path(qwen['token_accounting']['evidence_path'])]
    additional_sources += [p for p in Path(qwen['token_accounting']['local_tokenizer_path']).iterdir() if p.is_file()]
    original_manifest = read(MODEL_SOURCE / 'audit/frozen_manifest.json')
    originals = {r['path']: r for group in ('files', 'source_files', 'code') for r in original_manifest.get(group, [])}
    for path in additional_sources:
        old = originals.get(str(path))
        if old:
            assert dated.record(path)['sha256'] == old['sha256'], path
    sources = {r['path']: r for r in manifest['source_files']}
    sources.update({str(p): dated.record(p) for p in additional_sources})
    files = [RUN / r['path'] for r in manifest['files']]
    files.extend(RUN / f for f in ('audit/scope_authorization.json', 'audit/qwen_runtime.diff', 'audit/runtime_adapter_identity.json'))
    code = {r['path']: r for r in manifest['code']}
    code.update({str(p): dated.record(p) for p in (Path(__file__).resolve(), local_runtime)})
    manifest.update(version='h4_dated_transcript_qwen14b_v1', files=[dated.record(p, RUN) for p in sorted(set(files))],
        source_files=list(sources.values()), code=list(code.values()))
    dated.atomic_json(manifest_path, manifest)
    return {'status': 'PREPARED', 'model': qwen['model'], 'budget': dated.BUDGET,
        'capacity': 'PASS', 'identity': dated.verify(RUN), 'external_calls': 0}


@contextmanager
def qwen_executor():
    import qwen_runtime
    original = dated.runtime.execute
    dated.runtime.execute = qwen_runtime.execute
    try:
        yield qwen_runtime.execute
    finally:
        dated.runtime.execute = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'preflight', 'mock', 'live'))
    parser.add_argument('--authorize-primary', type=int)
    parser.add_argument('--authorize-repair-max', type=int)
    parser.add_argument('--authorize-http-max', type=int)
    args = parser.parse_args()
    if args.phase == 'live':
        if (args.authorize_primary, args.authorize_repair_max, args.authorize_http_max) != (6, 6, 12):
            parser.error('Required current trial budget: 6 primary / 6 repair / 12 HTTP')
        with qwen_executor() as execute, dated.date_adapter():
            result = execute(RUN, dated.SCOPE, authorized=True, allow_unknown_service=False)
    else:
        with dated.forbid_network():
            if args.phase == 'prepare':
                result = prepare()
            elif args.phase == 'mock':
                with qwen_executor():
                    result = dated.mock(RUN)
            else:
                result = dated.verify(RUN)
    print(__import__('json').dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
