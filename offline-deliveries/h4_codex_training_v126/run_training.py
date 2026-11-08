"""Run sealed-data preflight, a worst-length smoke, then exact-plan continuation."""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
POOL = REPO / 'offline-deliveries/h4_codex_train_pool_v125'
FORMAL = REPO / 'restart-n5-runs/h4_codex_combined_sft_v125'
SMOKE = HERE / 'smoke'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + '.pending')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    temp.replace(p)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def replace_arg(argv, name, value):
    result = list(argv)
    result[result.index(name) + 1] = str(value)
    return result


def run_process(argv, *, cwd, log, env=None):
    with Path(log).open('x') as output:
        process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT)
        while process.poll() is None:
            time.sleep(10)
            status = {'pid': process.pid, 'running': process.poll() is None, 'checked_at_utc': utc(), 'log': str(log)}
            try:
                status['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
            except Exception as exc:
                status['gpu_monitor_error'] = type(exc).__name__
            write(HERE / 'LIVE_PROCESS.json', status)
        return process.returncode


def validate_training(folder, expected_micro, expected_updates):
    audit = read(folder / 'gradient_audit.json')
    assert audit['all_losses_finite'] and audit['all_gradients_finite_nonzero']
    assert audit['backward_calls'] == expected_micro and audit['optimizer_steps'] == expected_updates
    config = read(folder / 'adapter/adapter_config.json')
    assert config['step'] == expected_updates and config['max_length'] == 12288
    assert config['init_adapter_dir'] == str(REPO / 'restart-n5-runs/h4_round0_b0_formal_training_v1_live_2/clean_adapter')
    state = folder / 'adapter/adapter_model.pt'
    assert state.is_file() and sha(state) != sha(Path(config['init_adapter_dir']) / 'adapter_model.pt')
    events = [json.loads(line) for line in (folder / 'telemetry.jsonl').read_text().splitlines()]
    backwards = [e for e in events if e['event'] == 'backward_complete']
    assert len(backwards) == expected_micro and all(math.isfinite(e['unscaled_loss']) for e in backwards)
    losses = [e['unscaled_loss'] for e in backwards]
    return {'status': 'PASS', 'backward_calls': expected_micro, 'optimizer_steps': expected_updates,
            'all_losses_finite': True, 'all_gradients_finite_nonzero': True,
            'mean_training_loss': sum(losses) / len(losses), 'first_25_microbatch_mean_loss': sum(losses[:25]) / len(losses[:25]),
            'last_25_microbatch_mean_loss': sum(losses[-25:]) / len(losses[-25:]),
            'peak_allocated_bytes': max(e['peak_allocated_bytes'] for e in backwards),
            'peak_reserved_bytes': max(e['peak_reserved_bytes'] for e in backwards),
            'adapter_config_sha256': sha(folder / 'adapter/adapter_config.json'), 'adapter_state_sha256': sha(state),
            'gradient_audit_sha256': sha(folder / 'gradient_audit.json'), 'telemetry_sha256': sha(folder / 'telemetry.jsonl')}


def main():
    assert not (HERE / 'RUN_STATUS.json').exists(), 'Existing run status: do not silently repeat training'
    assert not FORMAL.exists(), 'Formal output already exists'
    status = {'status': 'PREFLIGHT_RUNNING', 'started_at_utc': utc(), 'training_started': False,
              'source_package_manifest_sha256': sha(POOL / 'COMPLETE_MANIFEST.json')}
    write(HERE / 'RUN_STATUS.json', status)
    try:
        rc = run_process([sys.executable, str(POOL / 'preflight.py')], cwd=REPO, log=HERE / 'preflight.log')
        if rc:
            raise RuntimeError(f'PREFLIGHT_FAILED:{rc}')
        plan = read(POOL / 'TRAINING_PLAN.json')
        records = [json.loads(x) for x in (POOL / 'data/TRAIN_NEW_ONLY.jsonl').read_text().splitlines()]
        lineage = [json.loads(x) for x in (POOL / 'data/QUESTION_LINEAGE.jsonl').read_text().splitlines()]
        by_id = {r['candidate_id']: r for r in records}
        # Two longest examples per question type, fixed without model feedback.
        chosen = []
        for qtype in ['multi_select', 'ordering']:
            chosen.extend(sorted((r for r in lineage if r['question_type'] == qtype), key=lambda r: (-r['input_token_count'], r['candidate_id']))[:2])
        SMOKE.mkdir()
        smoke_file = SMOKE / 'TRAIN_SMOKE.jsonl'
        smoke_file.write_text(''.join(json.dumps(by_id[r['candidate_id']], ensure_ascii=False) + '\n' for r in chosen))
        base_argv = list(plan['argv'])
        # Replace only the launcher with audited telemetry instrumentation.
        # All trainer arguments for the formal run remain exactly frozen.
        native_module = base_argv[2]
        instrumented = [sys.executable, str(HERE / 'training_instrumentation.py'), *base_argv[3:]]
        smoke_argv = replace_arg(instrumented, '--train-file', smoke_file)
        smoke_argv = replace_arg(smoke_argv, '--output-dir', SMOKE / 'adapter')
        smoke_argv = replace_arg(smoke_argv, '--epochs', '1.0') + ['--eval-steps', '1']
        runtime = {'schema_version': 'v126_training_execution_freeze_v1', 'frozen_at_utc': utc(),
                   'source_package_manifest_sha256': status['source_package_manifest_sha256'],
                   'training_plan_sha256': sha(POOL / 'TRAINING_PLAN.json'), 'training_data_sha256': sha(POOL / 'data/TRAIN_NEW_ONLY.jsonl'),
                   'native_runner_module': native_module, 'telemetry_launcher_binding': {'path': str(HERE / 'training_instrumentation.py'), 'sha256': sha(HERE / 'training_instrumentation.py')},
                   'controller_binding': {'path': str(Path(__file__).resolve()), 'sha256': sha(Path(__file__).resolve())},
                   'smoke_argv': smoke_argv, 'formal_argv': instrumented, 'formal_original_argv': base_argv,
                   'smoke_candidates': [{'candidate_id': r['candidate_id'], 'input_tokens': r['input_token_count']} for r in chosen],
                   'smoke_training_not_reused': True, 'test_evaluation': False,
                   'telemetry_note': 'Native audit callbacks retained; extra callbacks record unscaled loss, timing and peak GPU memory without changing gradients or optimizer.',
                   'inherited_trainer_details': {'raw_step1_loss_log_divisor_bias': 'Native first print is 1/25 of true four-microbatch mean; use telemetry instead.',
                                                'final_accumulation': '726 microsteps / 4 -> 182 updates. Last two microbatches retain native divisor4; no algorithm patch.',
                                                'epoch_boundary': 'Native accumulation crosses epoch boundary.'}}
        assert instrumented[2:] == base_argv[3:]
        write(HERE / 'EXECUTION_FREEZE.json', runtime)
        status.update(status='SMOKE_TRAINING_RUNNING', training_started=True, smoke_started_at_utc=utc())
        write(HERE / 'RUN_STATUS.json', status)
        env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
        env['SMOKE_GRADIENT_AUDIT_PATH'] = str(SMOKE / 'gradient_audit.json')
        env['V126_TELEMETRY_PATH'] = str(SMOKE / 'telemetry.jsonl')
        before = time.monotonic()
        rc = run_process(smoke_argv, cwd=plan['cwd'], log=SMOKE / 'train.log', env=env)
        if rc:
            raise RuntimeError(f'SMOKE_FAILED:{rc}')
        result = validate_training(SMOKE, 4, 1)
        result['elapsed_seconds'] = time.monotonic() - before
        result['smoke_only_not_final_model'] = True
        write(SMOKE / 'RESULT.json', result)
        # The full run reloads original M_clean; smoke optimizer/state are unused.
        FORMAL.mkdir()
        write(FORMAL / 'LAUNCH_BINDING.json', runtime)
        env['SMOKE_GRADIENT_AUDIT_PATH'] = str(FORMAL / 'gradient_audit.json')
        env['V126_TELEMETRY_PATH'] = str(FORMAL / 'telemetry.jsonl')
        status.update(status='FORMAL_TRAINING_RUNNING', smoke_status='PASS', formal_started_at_utc=utc(), formal_output=str(FORMAL))
        write(HERE / 'RUN_STATUS.json', status)
        before = time.monotonic()
        rc = run_process(instrumented, cwd=plan['cwd'], log=FORMAL / 'train.log', env=env)
        if rc:
            raise RuntimeError(f'FORMAL_TRAINING_FAILED:{rc}')
        result = validate_training(FORMAL, 726, 182)
        result.update(elapsed_seconds=time.monotonic() - before, training_records=363, root_count=68, epochs=2, replay_count=0,
                      evaluation_started=False, loss_note='Training loss measures fit, not Solver accuracy or heldout performance.')
        write(FORMAL / 'TRAINING_RESULT.json', result)
        status.update(status='TRAINING_COMPLETE_AWAITING_EVALUATION', finished_at_utc=utc(), formal_status='PASS', result=result)
        write(HERE / 'RUN_STATUS.json', status)
        print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
        return 0
    except BaseException as exc:
        status.update(status='STOPPED_WITH_ERROR', error_type=type(exc).__name__, error=str(exc), stopped_at_utc=utc())
        write(HERE / 'RUN_STATUS.json', status)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
