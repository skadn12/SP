#!/usr/bin/env python3
"""Independent, CPU-only audit of the frozen Day 5 training execution.

Does not launch the runner, load a model onto a GPU, evaluate Solver, or write
training artifacts. The optional tensor inspection loads only custom LoRA
state dictionaries on CPU. A missing execution result remains PENDING.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

REPO = Path('/root/autodl-tmp/gxy-selfplay')
WEEK = REPO / 'research-protocols/selfplay_week_20261019'
D4 = WEEK / 'day4_schedule_v1'
RUNS = REPO / 'restart-n5-runs'
HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def lr_at_step(step, total_steps):
    warmup = max(1, int(total_steps * .05))
    if step < warmup:
        return 1e-4 * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 1e-4 * (.1 + .9 * .5 * (1. + math.cos(math.pi * progress)))


class Audit:
    def __init__(self):
        self.checks = []
        self.evidence = {}
        self.pending = []

    def check(self, name, condition, detail=None):
        self.checks.append({'check': name, 'pass': bool(condition), 'detail': detail})

    def check_binding(self, name, expected):
        actual = binding(expected['path'])
        self.check(name, actual == expected, {'expected': expected, 'actual': actual})
        return actual

    def guarded(self, name, function):
        try:
            return function()
        except Exception as exc:
            self.check(name, False, f'{type(exc).__name__}: {exc}')
            return None


def audit_static(audit):
    runtime = read(D4 / 'runtime/RUNTIME_BINDING.json')
    budget = read(D4 / 'TRAINING_BUDGET_FREEZE.json')
    policy = read(D4 / 'EXECUTION_POLICY.json')
    audit.evidence['frozen_runtime_binding'] = binding(D4 / 'runtime/RUNTIME_BINDING.json')
    for key in ['runtime', 'native_trainer', 'execution_policy', 'model_manifest',
                'preflight_A', 'preflight_B', 'negative_checks', 'independent_code_review']:
        audit.check_binding('frozen_' + key, runtime[key])
    audit.check_binding('frozen_budget_runtime_binding', budget['runtime_binding'])
    audit.check_binding('frozen_budget_evaluation_lock', budget['evaluation_lock'])
    for arm in ['A', 'B']:
        audit.check_binding('frozen_train_' + arm, runtime['schedule'][arm]['train_binding'])
        rows = jsonl(runtime['schedule'][arm]['train_binding']['path'])
        audit.check('train_only_' + arm, all(r['split'] == 'train' and r['arm'] == arm for r in rows))
        audit.check('two_epochs_112_questions_' + arm,
                    len(rows) == 224 and {r['epoch'] for r in rows} == {1, 2}
                    and all(len([r for r in rows if r['epoch'] == e]) == 112 for e in [1, 2])
                    and len({r['candidate_id'] for r in rows}) == 112)
    source = Path(runtime['runtime']['path']).read_text()
    tree = ast.parse(source)
    run_arm = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_arm')
    calls = [ast.unparse(n.func) for n in ast.walk(run_arm) if isinstance(n, ast.Call)]
    audit.check('fresh_base_adapter_optimizer_each_arm_code',
                calls.count('AutoModelForCausalLM.from_pretrained') == 1
                and calls.count('native.load_initial_adapter') == 1
                and calls.count('torch.optim.AdamW') == 1
                and 'native.load_initial_adapter(model, initial, args)' in source
                and 'initial = identity[\'S0_M_clean_adapter\'][\'directory\']' in source)
    audit.check('fixed_endpoint_no_eval_selection_code',
                'native.save_adapter(model, out / \'adapter\', args, step, {\'fixed_final_endpoint\': True})' in source
                and not any('evaluate' in c or 'solver' in c.lower() for c in calls)
                and budget['checkpoint'] == {'fixed_final_optimizer_step': 56,
                    'score_selection': False, 'intermediate_test': False})
    manifest = read(runtime['model_manifest']['path'])
    identities = {}
    for role in ['base', 'S0_M_clean_adapter']:
        identities[role] = []
        for frozen in manifest['models'][role]['files']:
            identities[role].append(audit.check_binding('model_' + role + '_' + Path(frozen['path']).name, frozen))
    audit.evidence['actual_initial_model_files'] = identities
    audit.evidence['initialization_evidence_limit'] = (
        'Independent fresh initialization is established from the hash-bound executed '
        'runner and adapter config; no pre-update tensor snapshot was recorded by this runner.')
    return runtime, budget, policy, manifest


def audit_common(audit, label, report, runtime):
    expected = {'runtime': runtime['runtime'], 'native_trainer': runtime['native_trainer'],
                'policy': runtime['execution_policy'], 'model_binding_manifest': runtime['model_manifest']}
    for key, value in expected.items():
        audit.check(label + '_' + key, report[key] == value)
    env = runtime['environment']
    for key in ['versions', 'python', 'executable', 'python_binary', 'torch_cuda_build']:
        audit.check(label + '_environment_' + key, report[key] == env[key])
    audit.check(label + '_no_runtime_shuffle_truncation',
                report['shuffle'] is False and report['permutation'] is False
                and report['runtime_epochs'] == 1 and report['truncation'] is False)


def audit_telemetry(audit, label, result, schedule, smoke=False):
    audit.check_binding(label + '_telemetry_digest', result['telemetry'])
    events = jsonl(result['telemetry']['path'])
    rows = schedule['rows']
    if smoke:
        rows = sorted(rows, key=lambda row: (-row['full_input_tokens'], row['index']))[:4]
    microsteps, optsteps = len(rows), len(rows) // 4
    audit.check(label + '_length_steps', len(events) == microsteps
                and result['microsteps'] == microsteps and result['optimizer_steps'] == optsteps)
    audit.check(label + '_event_schedule_identity', len(events) == len(rows) and all(
        e['microstep'] == i + 1 and e['arm'] == schedule['arm']
        and all(e[k] == r[k] for k in ['candidate_id', 'epoch', 'schedule_key'])
        and e['input_tokens'] == r['full_input_tokens']
        and e['supervised_tokens'] == r['supervised_tokens']
        for i, (e, r) in enumerate(zip(events, rows))))
    audit.check(label + '_finite_losses', all(math.isfinite(e['loss']) for e in events))
    updates = [e for e in events if 'optimizer_step' in e]
    audit.check(label + '_native_accumulation_updates', len(updates) == optsteps and all(
        e['microstep'] == (i + 1) * 4 and e['optimizer_step'] == i + 1
        and math.isfinite(e['gradient_norm_before_clip']) and e['gradient_norm_before_clip'] > 0
        and e['updated_parameters_finite'] is True
        and math.isclose(e['learning_rate'], lr_at_step(i, optsteps), rel_tol=1e-12)
        for i, e in enumerate(updates)))
    audit.check(label + '_mean_loss_matches', bool(events) and math.isclose(
        sum(e['loss'] for e in events) / len(events), result['mean_loss'], rel_tol=1e-12))
    audit.check(label + '_monotonic_time_memory', bool(events) and all(
        e['wall_seconds'] > 0 and e['peak_allocated_bytes'] > 0
        and e['peak_reserved_bytes'] >= e['peak_allocated_bytes']
        for e in events) and all(events[i]['wall_seconds'] <= events[i + 1]['wall_seconds']
                                    for i in range(len(events) - 1)))
    audit.check(label + '_result_time_memory', bool(events)
                and result['wall_seconds'] >= events[-1]['wall_seconds']
                and result['peak_allocated_bytes'] >= max(e['peak_allocated_bytes'] for e in events)
                and result['peak_reserved_bytes'] >= max(e['peak_reserved_bytes'] for e in events))
    audit.evidence[label] = {
        'telemetry': result['telemetry'], 'microsteps': len(events), 'optimizer_steps': len(updates),
        'input_tokens': sum(e['input_tokens'] for e in events),
        'supervised_tokens': sum(e['supervised_tokens'] for e in events),
        'mean_loss': result['mean_loss'], 'first_loss': events[0]['loss'] if events else None,
        'last_loss': events[-1]['loss'] if events else None,
        'minimum_loss': min((e['loss'] for e in events), default=None),
        'maximum_loss': max((e['loss'] for e in events), default=None),
        'first_28_mean_loss': sum(e['loss'] for e in events[:28]) / min(len(events), 28) if events else None,
        'last_28_mean_loss': sum(e['loss'] for e in events[-28:]) / min(len(events), 28) if events else None,
        'first_learning_rate': updates[0]['learning_rate'] if updates else None,
        'final_learning_rate': updates[-1]['learning_rate'] if updates else None,
        'maximum_gradient_norm': max((e['gradient_norm_before_clip'] for e in updates), default=None),
        'wall_seconds': result['wall_seconds'], 'peak_allocated_bytes': result['peak_allocated_bytes'],
        'peak_reserved_bytes': result['peak_reserved_bytes']}


def audit_smoke(audit, path, runtime):
    smoke = read(path)
    audit.evidence['resource_smoke_result'] = binding(path)
    audit_common(audit, 'smoke', smoke, runtime)
    audit.check('smoke_isolation_mode', smoke['status'] == 'PASS_ISOLATED_RESOURCE_SMOKE'
                and smoke['formal_training_runs'] == 0 and smoke['formal_artifact_saved'] is False)
    schedules = [smoke['schedule'], smoke['peer_schedule']]
    audit.check('smoke_both_arms', {r['arm'] for r in smoke['results']} == {'A', 'B'}
                and len(smoke['results']) == 2)
    for result in smoke['results']:
        arm = result['arm']
        schedule = next(s for s in schedules if s['arm'] == arm)
        audit.check('smoke_' + arm + '_schedule_binding', all(schedule[k] == v
                    for k, v in runtime['schedule'][arm].items()))
        audit.check('smoke_' + arm + '_no_formal_artifact', result['status'] == 'PASS'
                    and result['mode'] == 'isolated_resource_smoke'
                    and result['formal_model_artifact_saved'] is False)
        audit_telemetry(audit, 'smoke_' + arm, result, schedule, smoke=True)
        audit.check('smoke_' + arm + '_no_adapter_saved', not (
            Path(result['telemetry']['path']).parent / 'adapter').exists())
    return smoke


def audit_training(audit, arm, path, smoke_path, runtime):
    report = read(path)
    audit.evidence['formal_' + arm + '_result'] = binding(path)
    audit_common(audit, 'formal_' + arm, report, runtime)
    audit.check_binding('formal_' + arm + '_acceptance_digest', report['day5_acceptance'])
    audit.check('formal_' + arm + '_same_smoke_acceptance',
                report['day5_acceptance'] == binding(smoke_path))
    audit.check('formal_' + arm + '_schedule_binding', all(report['schedule'][k] == v
                for k, v in runtime['schedule'][arm].items()))
    result = report['result']
    audit.check('formal_' + arm + '_mode', report['status'] == 'PASS_FORMAL_FIXED_ENDPOINT'
                and result['status'] == 'PASS' and result['arm'] == arm
                and result['mode'] == 'formal_fixed_endpoint' and result['formal_model_artifact_saved'] is True)
    audit_telemetry(audit, 'formal_' + arm, result, report['schedule'])
    out = Path(result['telemetry']['path']).parent
    config = read(out / 'adapter/adapter_config.json')
    audit.check('formal_' + arm + '_fixed_endpoint_adapter_config',
                config['step'] == 56 and config['model'] == runtime['initialization']['base']
                and config['init_adapter_dir'] == runtime['initialization']['custom_adapter']
                and config['metrics'] == {'fixed_final_endpoint': True}
                and config['lora_rank'] == 16 and config['lora_alpha'] == 32
                and config['lora_dropout'] == .05 and config['max_length'] == 12288)
    files = sorted((out / 'adapter').iterdir())
    audit.evidence['formal_' + arm + '_adapter_files'] = [binding(p) for p in files if p.is_file()]
    audit.check('formal_' + arm + '_only_final_adapter_no_intermediate_checkpoints',
                {'adapter', 'telemetry.jsonl'} <= {p.name for p in out.iterdir()}
                <= {'adapter', 'telemetry.jsonl', 'merged_model'})
    return out / 'adapter/adapter_model.pt'


def audit_tensors(audit, manifest, final_paths):
    import torch
    torch.set_num_threads(2)
    initial_path = Path(manifest['models']['S0_M_clean_adapter']['directory']) / 'adapter_model.pt'
    initial = torch.load(initial_path, map_location='cpu', weights_only=True)
    previous = None
    for arm, path in final_paths.items():
        state = torch.load(path, map_location='cpu', weights_only=True)
        audit.check('formal_' + arm + '_adapter_tensor_keys', state.keys() == initial.keys()
                    and all('.lora_' in k and state[k].shape == initial[k].shape for k in state))
        finite = all(bool(torch.isfinite(t).all()) for t in state.values())
        changes = [k for k in initial if not torch.equal(initial[k], state[k])]
        audit.check('formal_' + arm + '_adapter_tensors_finite_and_changed', finite and bool(changes))
        audit.evidence['formal_' + arm + '_cpu_tensor_audit'] = {
            'state_dict_tensors': len(state), 'parameters': sum(t.numel() for t in state.values()),
            'changed_tensors_vs_initial': len(changes), 'all_finite': finite,
            'maximum_absolute_change_vs_initial': max(
                float((state[k].float() - initial[k].float()).abs().max()) for k in initial)}
        if previous is not None:
            audit.check('A_B_final_tensor_states_differ',
                        any(not torch.equal(previous[k], state[k]) for k in previous))
        previous = state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-result', type=Path, default=RUNS / 'day5_schedule_resource_smoke_v1_RESULT.json')
    parser.add_argument('--A-result', type=Path, default=RUNS / 'day5_A1_schedule_v1_RESULT.json')
    parser.add_argument('--B-result', type=Path, default=RUNS / 'day5_B1_schedule_v1_RESULT.json')
    parser.add_argument('--output', type=Path, default=HERE / 'INDEPENDENT_TRAINING_AUDIT.json')
    args = parser.parse_args()
    audit = Audit()
    static = audit.guarded('static_audit_completed', lambda: audit_static(audit))
    final_paths = {}
    if static:
        runtime, budget, policy, manifest = static
        if args.smoke_result.exists():
            audit.guarded('smoke_audit_completed', lambda: audit_smoke(audit, args.smoke_result, runtime))
        else:
            audit.pending.append(str(args.smoke_result))
        for arm, path in [('A', args.A_result), ('B', args.B_result)]:
            if path.exists() and args.smoke_result.exists():
                value = audit.guarded('formal_' + arm + '_audit_completed',
                    lambda arm=arm, path=path: audit_training(audit, arm, path, args.smoke_result, runtime))
                if value:
                    final_paths[arm] = value
            else:
                audit.pending.append(str(path))
        if final_paths:
            audit.guarded('cpu_adapter_tensor_audit_completed', lambda: audit_tensors(audit, manifest, final_paths))
        if len(final_paths) == 2:
            audit.check('independent_output_directories', final_paths['A'].parent != final_paths['B'].parent)
    failures = [c for c in audit.checks if not c['pass']]
    status = 'FAIL' if failures else 'PENDING_REAL_OUTPUTS' if audit.pending else 'PASS_INDEPENDENT_TRAINING_AUDIT'
    report = {'schema_version': 'day5_independent_training_audit_v1',
              'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'status': status,
              'audit_script': binding(__file__), 'checks': audit.checks,
              'checks_count': len(audit.checks), 'failures_count': len(failures),
              'pending_execution_outputs': audit.pending, 'evidence': audit.evidence,
              'scope': 'CPU-only independent execution-integrity review; no Solver or GPU calls',
              'limitations': ['Training integrity and finite optimization do not prove evaluation improvement.',
                             'No claim of feedback benefit or unseen-root generalization follows from this audit.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': status, 'checks': len(audit.checks),
                      'failures': len(failures), 'output': str(args.output)}))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
