"""Offline prepare, bounded three-root live acceptance, and no-API report."""
from __future__ import annotations

import argparse
from dataclasses import replace
import difflib
import json
from pathlib import Path
import re
import subprocess
import sys

from ..core import atomic_json, atomic_text, sha256_file
from ..h4_qa_multiroot_k3_t08_probe_v1.generation import generate_root_qa, MINIMUM_COUNTS
from ..h4_qa_mining_probe_v1.intent_miner import build_intent_prompt
from ..live_backends import load_challenger_config, OpenAICompatibleChallengerClient, RATE_LIMIT_RETRY_LIMIT
from .common_sources import COMMON_BASE_VERSION, P1, prepare_sources, write_frozen
from .runtime import (FIRST_ROOTS, MODE, PLAN_TOTAL, LIMIT_TOTAL, LIMIT_PER_ROOT,
    BoundedLedger, GuardrailStop, ObservedClient, exclusive_run, root_path,
    read, now, audit_reservations, checkpoint_root, verify_checkpoint)
from .transfer import TransferHooks, TRANSFER_VERSION, revise_miner_prompt, model_cards

CODE = Path(__file__).resolve().parent
METHOD = CODE.parent
REPO = METHOD.parents[2]
CONFIG = Path('/root/autodl-tmp/gxy-selfplay/config.json')
DEFAULT_RUN = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_r02_feedback_transfer_p2_2_acceptance_v1_live_1')
WORK_ORDER = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_r02_feedback_transfer_p2_2_handoff_v1/CODEX_WORK_ORDER.md')
REPORT_VERSION = 'h4_r02_feedback_transfer_p2_2_report_v1'
TEMPLATE = P1 / 'inputs/feedback_compiler_snapshots/feedback_guidance_v1.md'


def record(path):
    return {'path': str(path), 'sha256': sha256_file(path)}


def generic_templates():
    text = TEMPLATE.read_text(encoding='utf-8')
    def section(heading):
        match = re.search(r'## ' + re.escape(heading) + r'[^\n]*\n+```text\n(.*?)\n```', text, re.S)
        if not match:
            raise GuardrailStop('legacy_template_section_drift')
        return match.group(1)
    return section('QA Writer 附加段'), section('Intent Miner 附加段')


def make_hooks(run, root_id, mode=MODE):
    folder = run / 'inputs' / root_id
    cards, history = read(folder / 'observation_cards.json'), read(folder / 'history_only.json')
    generic, legacy = generic_templates()
    # The legacy arm is built offline from the same filtered root observations.
    # Its old policy stays intents-only; no P2 envelope/brief is appended there.
    payload = {'root_id': root_id, 'feedback_status': cards['feedback_status'],
               'observation_cards': model_cards(cards, root_id=root_id)}
    legacy = legacy.replace('{INTENT_FEEDBACK_JSON}', json.dumps(payload, ensure_ascii=False, indent=2))
    return TransferHooks(mode, cards, history, generic, legacy_intent_suffix=legacy)


def source_code_identity(run):
    initial = read(run / 'audit/initial_code_identity.json')['rows']
    paths = {Path(row['path']) for row in initial}
    paths.update(path for path in CODE.rglob('*') if path.is_file() and '__pycache__' not in path.parts)
    paths.add(METHOD / 'h4_r02_feedback_ablation_p0_p1_v1/qa_run.py')
    return [record(path) for path in sorted(paths)]


def validate_original_changes(run):
    allowed = METHOD / 'h4_qa_multiroot_k3_t08_probe_v1/generation.py'
    changed = [row['path'] for row in read(run / 'audit/initial_code_identity.json')['rows']
               if sha256_file(Path(row['path'])) != row['sha256']]
    if any(Path(path) != allowed for path in changed):
        raise GuardrailStop('unexpected_original_code_change:' + ','.join(changed))
    return changed


def freeze_inputs(run):
    paths = []
    for folder in ('inputs', 'common_base', 'policy_snapshot'):
        paths.extend(path for path in (run / folder).rglob('*') if path.is_file())
    paths += [run / 'execution_contract.json', run / 'input_path_audit.json',
              run / 'implementation_manifest.json', run / 'audit/test_report.json',
              run / 'audit/hook_disabled_parity.json', run / 'audit/token_identity.json']
    paths.extend(path for path in (run / 'audit/service_evidence').rglob('*') if path.is_file())
    return [record(path) for path in sorted(set(paths))]


def verify_frozen(run):
    frozen = read(run / 'audit/frozen_manifest.json')
    if frozen['version'] != TRANSFER_VERSION or frozen['root_order'] != list(FIRST_ROOTS):
        raise GuardrailStop('frozen_scope_drift')
    for row in frozen['code'] + frozen['inputs'] + frozen['sources']:
        path = Path(row['path'])
        if not path.is_file() or sha256_file(path) != row['sha256']:
            raise GuardrailStop('frozen_source_code_or_input_drift:' + str(path))
    if source_code_identity(run) != frozen['code']:
        raise GuardrailStop('generation_code_inventory_drift')
    if sha256_file(CONFIG) != frozen['config']['sha256']:
        raise GuardrailStop('configuration_file_drift')
    return frozen


def test_phase(run):
    command = [sys.executable, '-m', 'unittest', 'discover', '-s', str(CODE), '-p', 'test_*.py', '-t', str(REPO), '-v']
    before = source_code_identity(run)
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    output = result.stdout + result.stderr
    atomic_text(run / 'audit/test_output.txt', output)
    count = re.search(r'Ran (\d+) tests?', output)
    report = {'status': 'PASS' if result.returncode == 0 and before == source_code_identity(run) else 'FAIL',
              'command': command, 'cwd': str(REPO), 'exit_code': result.returncode,
              'tests_run': int(count.group(1)) if count else None,
              'at_utc': now(), 'log': record(run / 'audit/test_output.txt'),
              'program_tests_only': True, 'live_calls': 0, 'code': before}
    atomic_json(run / 'audit/test_report.json', report)
    hook_lines = [line for line in output.splitlines() if 'test_generation_hooks.' in line]
    atomic_json(run / 'audit/hook_disabled_parity.json', {
        'status': 'PASS' if report['status'] == 'PASS' and hook_lines else 'FAIL',
        'test_log': report['log'], 'test_results': hook_lines,
        'original_snapshot': str(run / 'audit/original_code_snapshot/h4_qa_multiroot_k3_t08_probe_v1/generation.py'),
        'method': 'Full mocked original snapshot and current generator comparison; no paid calls.'})
    return report


def prepare(run):
    if (run / 'audit/frozen_manifest.json').is_file():
        verify_frozen(run)
        return read(run / 'audit/preflight_report.json')
    if list((run / 'qa').glob('**/ledger/requests/*.json')):
        raise GuardrailStop('live_states_without_freeze')
    sources = prepare_sources(run)
    changed = validate_original_changes(run)
    config = replace(load_challenger_config(CONFIG), temperature=0.1, enable_thinking=False,
                     timeout_seconds=max(180, int(load_challenger_config(CONFIG).timeout_seconds)))
    write_frozen(run / 'input_path_audit.json', {'status': sources['status'],
        'sources': sources['path_audit'], 'frozen_inputs': sources['inputs'],
        'transcript_policy': 'Exact P1 frozen bytes; no cleaning or preparation transform.'})
    contract = {'version': TRANSFER_VERSION, 'baseline_label': 'P1_LEGACY_FEEDBACK_TRANSFER_BASELINE',
        'common_base_version': COMMON_BASE_VERSION, 'mode': MODE, 'roots': list(FIRST_ROOTS),
        'planned_logical_calls': PLAN_TOTAL, 'maximum_logical_calls': LIMIT_TOTAL,
        'per_root_limit': LIMIT_PER_ROOT, 'maximum_miner_per_root': 1,
        'maximum_primary_per_root': 6, 'maximum_repair_per_root': 6,
        'generation': {'model': config.model, 'base_url': config.base_url, 'temperature': config.temperature,
          'enable_thinking': config.enable_thinking, 'seed': 'not_sent_by_original_client',
          'miner_max_output_tokens': 4096, 'writer_max_output_tokens': 8192,
          'intent_count': 3, 'guided_counts': MINIMUM_COUNTS,
          'free_counts': {key: count * 2 for key, count in MINIMUM_COUNTS.items()},
          'guided_free_requested_ratio': [1, 2], 'minimum_counts_each_mode': MINIMUM_COUNTS,
          'timeout_seconds': config.timeout_seconds,
          'repair': 'one_original_whole_batch_replacement_only_if_below_minimum',
          'transport_retry_policy': 'original_client_unchanged', 'http_429_retry_limit': RATE_LIMIT_RETRY_LIMIT},
        'context_reserve_tokens': 512, 'solver_calls_allowed': 0, 'memory_calls_allowed': 0,
        'no_semantic_admission': True, 'stop_after_first_acceptance': True,
        'work_order': record(WORK_ORDER), 'configuration': record(CONFIG)}
    write_frozen(run / 'execution_contract.json', contract)
    for path in sorted(CODE.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            write_frozen(run / 'policy_snapshot/code' / path.relative_to(CODE), raw=path.read_bytes())
    write_frozen(run / 'policy_snapshot/feedback_guidance_v1.md', raw=TEMPLATE.read_bytes())
    write_frozen(run / 'policy_snapshot/CODEX_WORK_ORDER.md', raw=WORK_ORDER.read_bytes())
    write_frozen(run / 'policy_snapshot/original_generation_with_optional_hooks.py',
                 raw=(METHOD / 'h4_qa_multiroot_k3_t08_probe_v1/generation.py').read_bytes())
    revisions = []
    miners = []
    for root_id in FIRST_ROOTS:
        folder = run / 'inputs' / root_id
        if not (folder / 'expanded_dialogue_clean.json').is_file():
            continue
        expanded = read(folder / 'expanded_dialogue_clean.json')
        original_spec = build_intent_prompt(expanded)
        revised = revise_miner_prompt(original_spec)
        for mode in ('LEGACY_COMMON_BASE', MODE):
            adapted = make_hooks(run, root_id, mode).miner_prompt(original_spec, expanded=expanded,
                        root_id=root_id, root_dir=run / 'common_base/offline_prompts' / mode / root_id)
            write_frozen(run / 'common_base/offline_prompts' / mode / root_id / 'miner_prompt.txt', raw=adapted['prompt'].encode())
            if mode == MODE:
                miners.append((root_id, adapted['prompt'], expanded))
        revisions.append({'root_id': root_id, 'old_prompt_sha256': original_spec['prompt_sha256'],
            'common_prompt_sha256': revised['prompt_sha256'],
            'diff': ''.join(difflib.unified_diff(original_spec['prompt'].splitlines(True), revised['prompt'].splitlines(True),
                                              fromfile='original_default', tofile=COMMON_BASE_VERSION))})
    write_frozen(run / 'common_base/miner_prompt_revision.json', {
        'version': COMMON_BASE_VERSION, 'changes': 'Concrete example theme and fixed evidence positions replaced with field/type schema only.',
        'original_default_unchanged': True, 'revisions': revisions})
    tests = test_phase(run)
    from .token_preflight import TokenPreflight, prepare_tokenizer
    prepare_tokenizer(run, config)
    token = TokenPreflight(run)
    token_results, token_errors = [], []
    for root_id, prompt, expanded in miners:
        try:
            token_results.append(token.check(prompt=prompt, max_output_tokens=4096,
                request_id=f'{root_id}-intent-miner', components=token_components(run, root_id, prompt, expanded)))
        except BaseException as error:
            token_errors.append({'root_id': root_id, 'reason': f'{type(error).__name__}:{error}'})
    checks = {'source_eligibility': sources['status'] == 'PASS', 'tests': tests['status'] == 'PASS',
        'model_identity': config.model == 'Qwen/Qwen3-14B', 'miner_token_preflight': len(token_results) == 3 and not token_errors,
        'only_authorized_original_file_changed': True, 'exact_fixed_scope': len(miners) == 3}
    report = {'status': 'PASS' if all(checks.values()) else 'BLOCKED', 'at_utc': now(),
              'checks': checks, 'token_errors': token_errors, 'miner_checks': token_results,
              'live_calls': 0, 'writer_preflight': 'Required immediately before each real Writer/repair reservation.'}
    atomic_json(run / 'audit/preflight_report.json', report)
    implementation = {'baseline_label': 'P1_LEGACY_FEEDBACK_TRANSFER_BASELINE', 'common_base_version': COMMON_BASE_VERSION,
        'transfer_version': TRANSFER_VERSION, 'report_version': REPORT_VERSION,
        'modes': ['ORIGINAL_HOOK_DISABLED', 'LEGACY_COMMON_BASE', MODE], 'live_mode': MODE,
        'original_files_changed': changed, 'same_original_generator': True, 'new_code': str(CODE),
        'prior_workspace_snapshot': record(run / 'audit/initial_code_identity.json'),
        'work_order': record(WORK_ORDER), 'no_planner_miner_default_or_sorting_change': True}
    write_frozen(run / 'implementation_manifest.json', implementation)
    if report['status'] == 'PASS':
        write_frozen(run / 'audit/frozen_manifest.json', {'version': TRANSFER_VERSION, 'root_order': list(FIRST_ROOTS),
            'code': source_code_identity(run), 'inputs': freeze_inputs(run), 'sources': sources['path_audit'],
            'config': record(CONFIG), 'preflight': record(run / 'audit/preflight_report.json')})
    atomic_text(run / 'RUN_STATUS.md', f"P2-2 first acceptance\n\nPrepare: {report['status']}\nLive: NOT_RUN\nReport: NOT_RUN\nNo Solver / Memory calls.\n")
    return report


def token_components(run, root_id, prompt, expanded, request_id=None):
    numbered = 'NUMBERED AUTHORITATIVE DIALOGUE:' in prompt
    transcript = '\n'.join((f'{i} | ' if numbered else '') + f"{row.get('turn_ref')} | {row.get('speaker')} | {row.get('text')}"
                           for i, row in enumerate(expanded.get('turns') or [], 1))
    cards = read(run / 'inputs' / root_id / 'observation_cards.json')
    feedback = json.dumps(model_cards(cards, root_id=root_id), ensure_ascii=False, indent=2)
    brief, binding = '', ''
    if request_id:
        audit_path = root_path(run, root_id) / 'writer_briefs/requests' / f'{request_id}.json'
        if audit_path.is_file():
            entry = read(Path(read(audit_path)['brief_path']))
            brief = entry['rendered_suffix']
            feedback = json.dumps(entry['observation_cards'], ensure_ascii=False, indent=2) if 'observation_cards' in entry else ''
            if entry.get('binding'):
                binding = json.dumps(entry['binding'], ensure_ascii=False, indent=2)
    template = prompt.replace(transcript, '', 1)
    if brief:
        template = template.removesuffix(brief)
    return {'transcript': transcript, 'feedback': feedback, 'binding': binding, 'brief': brief, 'template': template}


def live(run):
    prior_stop = run / 'audit/execution_stop.json'
    if prior_stop.is_file() and read(prior_stop).get('phase') == 'live':
        raise GuardrailStop('first_acceptance_already_stopped:' + read(prior_stop)['reason'])
    frozen = verify_frozen(run)
    preflight = read(run / 'audit/preflight_report.json')
    if preflight['status'] != 'PASS' or record(run / 'audit/preflight_report.json') != frozen['preflight']:
        raise GuardrailStop('preflight_not_passed_or_drift')
    audit_reservations(run)
    from .token_preflight import TokenPreflight
    token = TokenPreflight(run)
    original_config = load_challenger_config(CONFIG)
    config = replace(original_config, temperature=0.1, enable_thinking=False,
                     timeout_seconds=max(180, int(original_config.timeout_seconds)))
    if config.model != 'Qwen/Qwen3-14B':
        raise GuardrailStop('model_identity_drift')
    delegate = OpenAICompatibleChallengerClient(config)
    for root_id in FIRST_ROOTS:
        root = root_path(run, root_id)
        verify_frozen(run)
        if verify_checkpoint(root):
            print(json.dumps({'root_id': root_id, 'status': 'verified_completed_root_reused'}), flush=True)
            continue
        expanded = read(run / 'inputs' / root_id / 'expanded_dialogue_clean.json')
        hooks = make_hooks(run, root_id)
        def before(request_id, metadata):
            verify_frozen(run)
            prompt = (root / 'raw/prompts' / f'{request_id}.txt').read_text()
            token.check(prompt=prompt, max_output_tokens=metadata['max_tokens'], request_id=request_id,
                        components=token_components(run, root_id, prompt, expanded, request_id))
        ledger = BoundedLedger(root, sha256_file(run / 'audit/frozen_manifest.json'), run, root_id, before_request=before, after_response=lambda request_id, response: token.verify_usage(request_id, response.metadata.get('usage')))
        client = ObservedClient(delegate, run / 'audit/transport_events.jsonl', MODE, root_id)
        print(json.dumps({'root_id': root_id, 'status': 'started', 'at_utc': now()}), flush=True)
        result = generate_root_qa(expanded=expanded, root_dir=root, root_id=root_id, api_config=CONFIG,
                                  ledger=ledger, client=client, hooks=hooks)
        result.update(assigned_mode=MODE, solver_calls=0, memory_calls=0, semantic_admission_applied=False)
        atomic_json(root / 'qa_generation_report.json', result)
        checkpoint_root(root)
        print(json.dumps({**result, 'at_utc': now()}, ensure_ascii=False), flush=True)
        from .reporting import mechanical_report
        mechanical_report(run)
    atomic_json(run / 'audit/live_completion.json', {'status': 'COMPLETED', 'at_utc': now(), 'root_order': list(FIRST_ROOTS)})
    from .reporting import mechanical_report
    return mechanical_report(run)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'live', 'report'), required=True)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    args = parser.parse_args(argv)
    run = args.run.resolve()
    if run == P1 or P1 in run.parents:
        raise RuntimeError('P1_is_read_only')
    run.mkdir(parents=True, exist_ok=True)
    try:
        with exclusive_run(run):
            if args.phase == 'prepare':
                result = prepare(run)
            elif args.phase == 'live':
                result = live(run)
            else:
                from .reporting import final_report
                result = final_report(run)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0 if result.get('status') not in {'BLOCKED', 'FAIL', 'FAILED'} else 2
    except (GuardrailStop, Exception) as error:
        stop_path = run / 'audit/execution_stop.json'
        if stop_path.is_file():
            stop_path = run / 'audit' / ('execution_stop_attempt_' + now().replace(':', '-') + '.json')
        atomic_json(stop_path, {'status': 'BLOCKED', 'at_utc': now(),
            'phase': args.phase, 'reason': str(error), 'automatic_resend': False})
        print('P2-2 stopped: ' + str(error), flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
