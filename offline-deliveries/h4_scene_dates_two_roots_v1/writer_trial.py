"""Prepare or execute a bounded dated-input trial through the existing Writer."""
from contextlib import contextmanager
from copy import deepcopy
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

REPO = Path('/root/autodl-tmp/gxy-selfplay/current_method_github')
sys.path.insert(0, str(REPO))
from scripts.current_method.restart_n5_spell.core import atomic_json, atomic_text, canonical_json, sha256_bytes
from scripts.current_method.restart_n5_spell.h4_qa_mining_hybrid_generation_v1 import contract_guided_generator as guided
from scripts.current_method.restart_n5_spell.h4_writer_partial_memory_v1 import runtime
from scripts.current_method.restart_n5_spell.h4_writer_partial_memory_v1.prepare import code_closure, record, verify
from scripts.current_method.restart_n5_spell.h4_writer_model_comparison_v1.prepare import _capacity
from scripts.current_method.restart_n5_spell.h4_writer_model_comparison_v1.transport import build_payload
from scripts.current_method.restart_n5_spell.h4_r02_ordering_temporal_contract_v1.dependencies import forbid_network

DATA = Path(__file__).resolve().parent
SOURCE = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_writer_partial_memory_v1_prepare_1')
DEFAULT_RUN = SOURCE.parent / 'h4_dated_transcript_writer_v1_prepare_1'
SCOPE = 'dated_transcript'
ARM = 'DATED_TRANSCRIPT_ONLY'
MODE = 'TRANSCRIPT_ONLY'
BUDGET = {'primary': 6, 'repair': 6, 'http': 12}
ROOTS = ('cc-train-episode-198480', 'cc-train-episode-122711')
read = runtime.read


@contextmanager
def date_adapter():
    """Change only session rendering and register this run's smaller call cap."""
    original = guided._dialogue
    if SCOPE in runtime.BUDGETS:
        raise RuntimeError('DATE_SCOPE_ALREADY_REGISTERED')

    def render(expanded):
        segments = expanded.get('conversation_date_segments')
        if segments is None:
            return original(expanded)
        turns = expanded['turns']
        chunks, cursor = [], 0
        for segment in segments:
            start, stop = segment['first_turn_index_1_based'] - 1, segment['last_turn_index_1_based']
            if start != cursor or stop <= start or stop > len(turns):
                raise ValueError('DATE_SEGMENT_COVERAGE_INVALID')
            members = turns[start:stop]
            if (members[0]['turn_ref'], members[-1]['turn_ref']) != (segment['first_turn_ref'], segment['last_turn_ref']):
                raise ValueError('DATE_SEGMENT_REFERENCE_MISMATCH')
            chunks.append(f'[Conversation date: {segment["conversation_date"]} | Session: {segment["segment_id"]}]')
            chunks.append(original({'turns': members}))
            cursor = stop
        if cursor != len(turns):
            raise ValueError('DATE_SEGMENT_TAIL_MISSING')
        return '\n'.join(chunks)

    guided._dialogue = render
    runtime.BUDGETS[SCOPE] = dict(BUDGET)
    try:
        yield
    finally:
        guided._dialogue = original
        del runtime.BUDGETS[SCOPE]


def prepare(run):
    if run.exists():
        raise ValueError('NEW_RUN_DIRECTORY_REQUIRED; use preflight for an existing preparation')
    original_manifest = read(SOURCE / 'audit/frozen_manifest.json')
    original_records = {str((SOURCE / r['path']).resolve()): r for r in original_manifest['files']}
    sources, outputs = {}, {}

    def source(path, *, frozen=False):
        row = record(path)
        if frozen and original_records.get(str(path.resolve()), {}).get('sha256') != row['sha256']:
            raise ValueError('ORIGINAL_INPUT_DRIFT:' + str(path))
        sources[row['path']] = row
        return read(path)

    profile = source(SOURCE / 'profile.json', frozen=True)
    profile['status'] = 'WAITING_FOR_DATED_TRIAL_BUDGET_CONFIRMATION'
    outputs['profile.json'] = profile
    outputs['ordering_policy.json'] = source(SOURCE / 'ordering_policy.json', frozen=True)
    outputs['admission_policy.json'] = source(SOURCE / 'admission_policy.json', frozen=True)
    outputs['execution_contract.json'] = {
        'scope': SCOPE, 'budget': BUDGET, 'per_primary_repair_max': 1,
        'real_execution_authorized': False, 'automatic_retries': 0,
        'parameter_fallback': False, 'stop_on_technical_failure': True,
        'no_automatic_resume': True, 'source_mode': MODE, 'memory_calls': 0,
        'question_quota': {'single_choice': 2, 'multi_select': 1, 'ordering': 1},
        'evaluation_focus': 'ordering', 'default_model_upgrade': False,
        'historical_control': 'Previous TRANSCRIPT_ONLY is diagnostic only; no concurrent randomized control.',
        'unknown_service_capacity_reasoning_and_pricing': True,
    }
    source(DATA / 'date_plan.json')
    date_validation = source(DATA / 'validation.json')
    dated_identities = {r['root_id']: r for r in date_validation['roots']}
    schedule, manifests, identity, capacity, date_refs = [], [], [], [], []
    prompts = {}
    with date_adapter():
        for root in ROOTS:
            expanded = source(SOURCE / 'inputs' / MODE / root / 'expanded.json', frozen=True)
            contracts = source(SOURCE / 'inputs' / MODE / root / 'contracts.json', frozen=True)
            dated = source(DATA / root / 'dated_dialogue.json')
            dated_record = next(r for r in dated_identities[root]['files'] if r['path'].endswith('.json'))
            if record(DATA / root / 'dated_dialogue.json')['sha256'] != dated_record['sha256']:
                raise ValueError('DATED_INPUT_DRIFT')
            if dated['turns'] != expanded['turns'] or [c['intent_id'] for c in contracts] != ['intent_1', 'intent_2', 'intent_3']:
                raise ValueError('TURN_OR_INTENT_MISMATCH')
            expanded = deepcopy(expanded)
            expanded['conversation_date_segments'] = dated['segments']
            outputs[f'inputs/{MODE}/{root}/expanded.json'] = expanded
            outputs[f'inputs/{MODE}/{root}/contracts.json'] = contracts
            for segment in dated['segments']:
                for turn in dated['turns'][segment['first_turn_index_1_based']-1:segment['last_turn_index_1_based']]:
                    date_refs.append({'root_id': root, 'turn_ref': turn['turn_ref'],
                        'segment_id': segment['segment_id'], 'conversation_date': segment['conversation_date'],
                        'date_origin': dated['date_origin'], 'event_date_implied': False})
            for index, contract in enumerate(contracts):
                spec = guided.build_prompt(expanded, contract, ordering_policy=runtime.POLICY)
                old = SOURCE / 'policy_snapshot/primary_requests' / MODE / root / contract['intent_id'] / 'prompt.txt'
                old_row = record(old)
                if original_records[str(old.resolve())]['sha256'] != old_row['sha256']:
                    raise ValueError('BASELINE_PROMPT_DRIFT')
                sources[old_row['path']] = old_row
                stripped = '\n'.join(line for line in spec['prompt'].split('\n') if not line.startswith('[Conversation date:'))
                if stripped != old.read_text() or spec['prompt'].count('[Conversation date:') != len(dated['segments']):
                    raise ValueError('PROMPT_HAS_NON_DATE_CHANGES')
                payload = build_payload(profile, spec['prompt'])
                row = {'scope': SCOPE, 'arm_id': ARM, 'source_mode': MODE, 'root_id': root,
                    'slot_id': contract['intent_id'], 'raw_index': index, 'order': len(schedule)+1}
                schedule.append(row)
                location = f'policy_snapshot/primary_requests/{MODE}/{root}/{contract["intent_id"]}'
                outputs[location + '/request.json'] = {**row, 'payload': payload}
                prompts[location + '/prompt.txt'] = spec['prompt']
                manifests.append({**row, 'prompt_sha256': spec['prompt_sha256'],
                    'messages_sha256': sha256_bytes(canonical_json(payload['messages']).encode()),
                    'requested_counts': spec['expected_counts']})
                identity.append({**row, 'dates_removed_equals_old_prompt': True,
                    'all_turn_objects_equal': True, 'same_contract_and_parameters': True,
                    'header_count': len(dated['segments']), 'old_prompt_sha256': old_row['sha256'],
                    'dated_prompt_sha256': spec['prompt_sha256']})
                capacity.append({**_capacity(profile, payload['messages'], 8192, str(row['order'])),
                    **row, 'messages_utf8_bytes': len(canonical_json(payload['messages']).encode()),
                    'content_characters': sum(len(m['content']) for m in payload['messages'])})
    outputs['schedule.json'] = schedule
    outputs['audit/prompt_identity.json'] = identity
    outputs['audit/turn_date_mapping.json'] = date_refs
    outputs['audit/token_budget.json'] = {
        'budget': BUDGET, 'requests': capacity, 'input_tokens': 'UNKNOWN',
        'reasoning_tokens': 'UNKNOWN', 'verified_money': 'UNKNOWN',
        'primary_requested_output_ceiling': 6 * 8192,
        'maximum_requested_output_ceiling_with_repair': 12 * 8192,
        'repair_input': 'Own primary output plus same full dated source and repair template',
        'service_capacity_verified': False, 'no_truncation': True,
    }
    for name, value in outputs.items():
        atomic_json(run / name, value)
    for name, prompt in prompts.items():
        atomic_text(run / name, prompt)
    runtime.write_rows(run / 'slot_call_manifest.jsonl', manifests)
    files = [run / name for name in (*outputs, *prompts, 'slot_call_manifest.jsonl')]
    atomic_json(run / 'audit/frozen_manifest.json', {
        'version': 'h4_dated_transcript_writer_v1', 'files': [record(p, run) for p in sorted(files)],
        'source_files': list(sources.values()),
        'code': [record(p) for p in sorted(set(code_closure() + [Path(__file__).resolve()]))],
        'credentials_in_manifest': False,
    })
    return {'status': 'OFFLINE_PREPARED_WAITING_FOR_BUDGET_CONFIRMATION',
        'primary_slots': len(schedule), 'budget': BUDGET, 'external_calls': 0, 'identity': verify(run)}


def mock(run):
    from scripts.current_method.restart_n5_spell.backends import BackendResponse
    from scripts.current_method.restart_n5_spell.h4_writer_partial_memory_v1.mock import synthetic_candidates
    results = []
    verify(run)
    with forbid_network(), date_adapter():
        for scenario in ('success', 'repair_limit', 'timeout_stop', 'nonterminal_stop', 'bad_json_repair'):
            with tempfile.TemporaryDirectory(prefix='h4_dates_mock_') as directory:
                target = Path(directory) / 'run'
                shutil.copytree(run, target, ignore=shutil.ignore_patterns('executions'))
                sent = []

                class Client:
                    def __init__(self, row, profile, callback):
                        self.row, self.profile, self.callback = row, profile, callback
                        self.config = SimpleNamespace(model=profile['model'], base_url=profile['base_url'], temperature=0.1)

                    def complete(self, *, prompt, **kwargs):
                        self.callback(build_payload(self.profile, prompt))
                        stage = 'repair' if 'REPAIR THE FOLLOWING BATCH' in prompt else 'primary'
                        sent.append((self.row['order'], stage))
                        expanded = read(target / 'inputs' / MODE / self.row['root_id'] / 'expanded.json')
                        assert prompt.count('[Conversation date:') == len(expanded['conversation_date_segments'])
                        if scenario == 'timeout_stop':
                            raise TimeoutError('synthetic offline timeout')
                        raw = synthetic_candidates(expanded, self.row['slot_id'], stage=stage)
                        if scenario == 'repair_limit':
                            raw['candidates'][-1]['temporal_relations'] = []
                        text = 'invalid JSON fixture' if scenario == 'bad_json_repair' and stage == 'primary' else json.dumps(raw)
                        return BackendResponse(text, 'OFFLINE_FIXTURE', {
                            'finish_reason': 'length' if scenario == 'nonterminal_stop' else 'stop',
                            'actual_model': 'OFFLINE_FIXTURE', 'input_tokens': None,
                            'output_tokens': None, 'reasoning_tokens': None, 'usage': None,
                            'http_attempt_count': 1, 'transport_attempt_count': 1, 'transport_retry_count': 0})

                try:
                    summary = runtime.execute(target, SCOPE, authorized=True, allow_unknown_service=True,
                        client_factory=lambda row, profile, callback: Client(row, profile, callback))
                except runtime.ExecutionStopped:
                    if scenario not in ('timeout_stop', 'nonterminal_stop'):
                        raise
                    summary = read(target / 'executions' / SCOPE / 'reports/summary.json')
                count, accepted = {'success': (6, 24), 'repair_limit': (12, 18),
                    'timeout_stop': (1, 0), 'nonterminal_stop': (1, 0), 'bad_json_repair': (12, 24)}[scenario]
                assert len(sent) == summary['http_attempts'] == count
                assert summary['program_accepted'] == accepted
                assert len(set(sent)) == len(sent)
                results.append({'scenario': scenario, 'status': 'PASS', 'synthetic_attempts': count,
                    'program_accepted': accepted, 'external_calls': 0})
    result = {'status': 'PASS', 'scenarios': results, 'external_calls': 0, 'identity': verify(run)}
    atomic_json(run / 'audit/mock_report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'preflight', 'mock', 'live'))
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--authorize-primary', type=int)
    parser.add_argument('--authorize-repair-max', type=int)
    parser.add_argument('--authorize-http-max', type=int)
    parser.add_argument('--allow-unknown-service', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    if args.phase == 'live':
        if (args.authorize_primary, args.authorize_repair_max, args.authorize_http_max) != (6, 6, 12):
            parser.error('live requires explicit current-trial authorization: 6 primary / 6 repair / 12 HTTP')
        if not args.allow_unknown_service:
            parser.error('service capacity, parameter adoption, reasoning and pricing remain UNKNOWN')
        with date_adapter():
            result = runtime.execute(run, SCOPE, authorized=True, allow_unknown_service=True)
    else:
        with forbid_network():
            result = prepare(run) if args.phase == 'prepare' else mock(run) if args.phase == 'mock' else verify(run)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
