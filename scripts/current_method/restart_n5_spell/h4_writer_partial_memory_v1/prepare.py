"""Local-only input preparation and identity checks for the shared Writer."""
from __future__ import annotations

import ast
import copy
import json
import shutil
from pathlib import Path

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from ..h4_writer_model_comparison_v1.runtime import read, jsonl
from ..h4_writer_model_comparison_v1.transport import build_payload, _reject_inline_credentials
from ..h4_writer_model_comparison_v1.prepare import _capacity
from ..h4_round1_r03_memory_readonly_trial_preflight_v2.memory_readonly_dialogue_adapter import (
    normalize_historical_snapshot, memory_prompt_projection,
)
from ..h4_qa_mining_hybrid_generation_v1.contract_guided_generator import build_prompt
from .sources import SOURCE_MODES, TRANSCRIPT_ONLY, MEMORY_ONLY, source_envelope, projection_sha256, generic_memory_contract, source_identity
from .admission import POLICY as ADMISSION_POLICY

PROJECT = Path('/root/autodl-tmp/gxy-selfplay')
SOURCE = PROJECT / 'restart-n5-runs/h4_writer_model_comparison_v1_live_1'
DEFAULT_RUN = PROJECT / 'restart-n5-runs/h4_writer_partial_memory_v1_prepare_1'
ROOTS = ('cc-train-episode-198480', 'cc-train-episode-122711')
ORDERING = {'version': 'ordering_temporal_contract_v1', 'node_count': 4}
BUDGETS = {'memory_assist': {'primary': 12, 'repair_max': 12, 'http_max': 24},
           'memory_only': {'primary': 2, 'repair_max': 2, 'http_max': 4}}


def record(path, run=None):
    path = Path(path).resolve()
    return {'path': str(path.relative_to(run)) if run and path.is_relative_to(run) else str(path),
            'sha256': sha256_file(path), 'size': path.stat().st_size}


def code_closure():
    """Follow local AST imports from this small entry, excluding unrelated history."""
    package = Path(__file__).resolve().parent
    repo = package.parents[3]
    queue = list(package.glob('*.py'))
    seen = set()
    while queue:
        path = queue.pop().resolve()
        if path in seen or not path.is_file() or path.name.startswith('test_'):
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level:
                base = path.parent
                for _ in range(node.level - 1):
                    base = base.parent
                module = base.joinpath(*(node.module or '').split('.'))
                candidates = [module.with_suffix('.py'), module / '__init__.py']
                candidates.extend(module / (alias.name + '.py') for alias in node.names)
                for candidate in candidates:
                    if candidate.is_file() and candidate.is_relative_to(repo):
                        queue.append(candidate)
    return sorted(seen)


def verify(run):
    run = Path(run).resolve()
    manifest = read(run / 'audit/frozen_manifest.json')
    count = 0
    for group in ('files', 'source_files', 'code'):
        for row in manifest[group]:
            path = Path(row['path'])
            if not path.is_absolute():
                path = run / path
            if not path.is_file() or sha256_file(path) != row['sha256']:
                raise ValueError('FROZEN_FILE_DRIFT:' + str(path))
            count += 1
    return {'status': 'PASS', 'verified_files': count, 'external_calls': 0}


def project_snapshot(snapshot):
    """Reuse stable historical normalization; retain exact person/event strings."""
    projection = memory_prompt_projection(normalize_historical_snapshot(snapshot))
    items, mappings = [], []
    for row in projection['buckets']:
        if row['bucket_type'] not in ('event', 'person'):
            continue
        content = row['content'] if isinstance(row['content'], list) else [row['content']]
        for index, text in enumerate(content):
            if not isinstance(text, str) or not text.strip():
                raise ValueError('MEMORY_CONTENT_NOT_PLAIN_TEXT')
            ref = f'MEM_ITEM_{len(items)+1:03d}'
            items.append({'memory_ref': ref, 'bucket_type': row['bucket_type'], 'key': row['key'], 'content': text})
            mappings.append({'memory_ref': ref, 'bucket_type': row['bucket_type'], 'key': row['key'],
                'normalized_content_index': index, 'source_turn_refs': 'UNKNOWN', 'transcript_support': 'UNDETERMINED'})
    return items, mappings


def prepare(run=DEFAULT_RUN, *, source=SOURCE, findings=None):
    run, source = Path(run).resolve(), Path(source).resolve()
    if (run == source or source in run.parents or
            any((run / 'executions').glob('*/audit/execution_started.json'))):
        raise ValueError('REFUSE_OVERWRITE_SOURCE_OR_STARTED_RUN')
    run.mkdir(parents=True, exist_ok=True)
    findings = Path(findings or run / 'memory_source_findings.json')
    found = read(findings)
    if findings.resolve() != run / 'memory_source_findings.json':
        shutil.copyfile(findings, run / 'memory_source_findings.json')
    source_rows = {r['path']: r for r in read(source / 'audit/frozen_manifest.json')['files']}
    dependencies = {}
    def copy_frozen(path, target, *, require_source_freeze=False, expected=None):
        path, target = Path(path).resolve(), Path(target)
        item = record(path)
        old = source_rows.get(str(path))
        if require_source_freeze and (not old or old['sha256'] != item['sha256']):
            raise ValueError('ORIGINAL_WRITER_INPUT_DRIFT:' + str(path))
        if expected and item['sha256'] != expected:
            raise ValueError('MEMORY_SOURCE_IDENTITY_DRIFT:' + str(path))
        dependencies[str(path)] = item
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    profile = copy.deepcopy(read(source / 'profiles.json')['STRONG_TEMPORAL'])
    _reject_inline_credentials(profile)
    profile.pop('user_service_exception', None)
    profile['waived_configuration'] = []
    profile['unresolved'] = list(profile.get('unknown_configuration', []))
    profile['status'] = 'WAITING_FOR_NEW_SCOPE_BUDGET_AUTHORIZATION'
    atomic_json(run / 'profile.json', profile)
    dependencies[str(source / 'profiles.json')] = record(source / 'profiles.json')
    atomic_json(run / 'admission_policy.json', ADMISSION_POLICY)
    atomic_json(run / 'ordering_policy.json', ORDERING)
    schedule, requests, coverage, capacity, pairing = [], [], [], [], []
    for root in ROOTS:
        f = found['roots'][root]
        if f['status'] != 'LAWFUL_R01_LOCAL_SNAPSHOT' or not all(f['checks'].values()):
            raise ValueError('R1_MEMORY_BLOCKED:' + root)
        snapshot_path = run / 'inputs/memory' / root / 'snapshot.json'
        copy_frozen(f['snapshot_path'], snapshot_path, expected=f['raw_file_sha256'])
        snapshot = read(snapshot_path)
        if snapshot['root_id'] != root or snapshot['snapshot_sha256'] != f['snapshot_internal_sha256']:
            raise ValueError('SNAPSHOT_ROOT_OR_INTERNAL_SHA_MISMATCH')
        internal = {k: v for k, v in snapshot.items() if k != 'snapshot_sha256'}
        if sha256_bytes(canonical_json(internal).encode()) != snapshot['snapshot_sha256']:
            raise ValueError('MEMORY_INTERNAL_HASH_MISMATCH')
        for index, item in enumerate(f['evidence']):
            copy_frozen(item['path'], run / 'inputs/memory' / root / 'provenance' / f'{index:02d}_{Path(item["path"]).name}', expected=item['sha256'])
        items, sidecar = project_snapshot(snapshot)
        atomic_json(snapshot_path.with_name('projection.json'), items)
        atomic_json(snapshot_path.with_name('source_mapping.json'), sidecar)
        coverage.append({**f, 'projection_item_count': len(items), 'projection_sha256': projection_sha256(items)})
        shared = source / 'inputs/shared' / root
        for name in ('expanded_dialogue_clean.json', 'contracts.json'):
            copy_frozen(shared / name, run / 'inputs/original' / root / name, require_source_freeze=True)
        expanded, contracts = read(shared / 'expanded_dialogue_clean.json'), read(shared / 'contracts.json')
        for mode in SOURCE_MODES:
            envelope = source_envelope(expanded, mode, **({} if mode == TRANSCRIPT_ONLY else {
                'memory_items': items, 'snapshot_sha256': f['raw_file_sha256'], 'projection_sha256_value': projection_sha256(items)}))
            mode_contracts = [generic_memory_contract()] if mode == MEMORY_ONLY else contracts
            target = run / 'inputs' / mode / root
            atomic_json(target / 'expanded.json', envelope)
            atomic_json(target / 'contracts.json', mode_contracts)
            for index, contract in enumerate(mode_contracts):
                spec = build_prompt(envelope, contract, ordering_policy=ORDERING)
                slot = {'scope': 'memory_only' if mode == MEMORY_ONLY else 'memory_assist',
                    'arm_id': mode, 'source_mode': mode, 'root_id': root, 'slot_id': contract['intent_id'], 'raw_index': index}
                schedule.append(slot)
                prompt = spec['prompt']
                if mode != MEMORY_ONLY:
                    old_path = source / 'policy_snapshot/primary_requests/STRONG_TEMPORAL' / root / contract['intent_id'] / 'prompt.txt'
                    old_prompt = old_path.read_text()
                    dependencies[str(old_path)] = record(old_path)
                    if (mode == TRANSCRIPT_ONLY and prompt != old_prompt) or (mode != TRANSCRIPT_ONLY and not prompt.startswith(old_prompt + '\n')):
                        raise ValueError('ORIGINAL_PROMPT_NOT_PRESERVED')
                    pairing.append({**slot, 'original_prompt_sha256': sha256_file(old_path),
                        'original_body_exact': True, 'difference': 'NONE' if mode == TRANSCRIPT_ONLY else 'FROZEN_MEMORY_NAVIGATION_SUFFIX_ONLY'})
                payload = build_payload(profile, prompt)
                location = run / 'policy_snapshot/primary_requests' / mode / root / contract['intent_id']
                identity = {**slot, 'source_identity': source_identity(envelope),
                    'expanded_file_sha256': sha256_file(target / 'expanded.json'),
                    'contract_file_sha256': sha256_file(target / 'contracts.json'),
                    'prompt_sha256': sha256_bytes(prompt.encode()),
                    'messages_sha256': sha256_bytes(canonical_json(payload['messages']).encode()),
                    'requested_counts': {'single_choice': 2, 'multi_select': 1, 'ordering': 1}}
                atomic_text(location / 'prompt.txt', prompt)
                atomic_json(location / 'request.json', {**identity, 'payload': payload})
                requests.append(identity)
                budget = _capacity(profile, payload['messages'], 8192, '::'.join((mode, root, contract['intent_id'])))
                budget.update(scope=slot['scope'], source_mode=mode, root_id=root, slot_id=contract['intent_id'],
                    complete_messages_utf8_bytes=len(canonical_json(payload['messages']).encode()),
                    complete_messages_characters=sum(len(m['content']) for m in payload['messages']),
                    real_model_input_tokens='UNKNOWN', reasoning_tokens='UNKNOWN', verified_money='UNKNOWN')
                capacity.append(budget)
    # Paired alternating order is fixed before any output; generic Memory slots are separate.
    schedule.sort(key=lambda r: (r['scope'] == 'memory_only', ROOTS.index(r['root_id']), r['raw_index'], SOURCE_MODES.index(r['source_mode'])))
    for scope in BUDGETS:
        for number, row in enumerate([s for s in schedule if s['scope'] == scope], 1):
            row['order'] = number
    atomic_json(run / 'schedule.json', schedule)
    jsonl(run / 'slot_call_manifest.jsonl', requests)
    atomic_json(run / 'execution_contract.json', {'budgets': BUDGETS, 'per_primary_repair_max': 1,
        'model': profile['model'], 'admission_policy': 'partial_v1', 'ordering_policy': ORDERING,
        'seed_policy': profile['seed_policy'], 'transport_retries': 0, 'memory_api_calls': 0,
        'real_execution_authorized': False, 'previous_service_exception_reused': False,
        'technical_failure_policy': 'STOP_NO_RESEND_NO_REPAIR', 'no_automatic_resume': True,
        'memory_only_abstention': 'empty candidates or valid INSUFFICIENT_EVIDENCE abstentions do not repair waived types',
        'memory_only_task': '2SC+1MS+1ORDER; generic slot per root; separate feasibility test'})
    atomic_json(run / 'audit/memory_source_audit.json', coverage)
    atomic_json(run / 'audit/prompt_identity.json', pairing)
    atomic_json(run / 'audit/token_budget.json', {'requests': capacity, 'by_scope': {
        scope: {**values, 'max_logical_calls': values['http_max'],
            'primary_requested_output_tokens': values['primary'] * 8192,
            'maximum_requested_output_tokens_including_repair': values['http_max'] * 8192,
            'actual_input_tokens': 'UNKNOWN', 'repair_input_tokens': 'DYNAMIC_OWN_PRIMARY_PLUS_FROZEN_SOURCE',
            'reasoning_reserve': 'UNKNOWN', 'service_capacity_verified': False, 'verified_cost': 'UNKNOWN'}
        for scope, values in BUDGETS.items()}, 'no_truncation': True,
        'qualification': 'Full messages measured in bytes/characters; no verified service tokenizer or context/reasoning limit. Output ceilings are requested token budgets, not verified costs or capacity.'})
    freeze(run, source_files=list(dependencies.values()))
    return {'status': 'OFFLINE_PREPARED_WAITING_FOR_AUTHORIZATION', 'primary_slots': len(schedule),
            'budgets': BUDGETS, 'external_calls': 0, 'identity': verify(run)}


def freeze(run, *, source_files=None):
    run = Path(run).resolve()
    if source_files is None:
        source_files = read(run / 'audit/frozen_manifest.json')['source_files']
    files = list((run / 'inputs').rglob('*')) + list((run / 'policy_snapshot').rglob('*'))
    files += [run / name for name in ('profile.json', 'schedule.json', 'admission_policy.json',
        'ordering_policy.json', 'execution_contract.json', 'memory_source_findings.json', 'slot_call_manifest.jsonl',
        'audit/memory_source_audit.json', 'audit/prompt_identity.json', 'audit/token_budget.json')]
    atomic_json(run / 'audit/frozen_manifest.json', {'version': 'h4_writer_partial_memory_v1',
        'files': [record(p, run) for p in sorted(set(files)) if p.is_file()],
        'source_files': source_files, 'code': [record(p) for p in code_closure()], 'credentials_in_manifest': False})
