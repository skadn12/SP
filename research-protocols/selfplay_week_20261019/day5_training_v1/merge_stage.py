"""Run a frozen CPU merge and retain the exact final-adapter lineage."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def binding(path):
    p = Path(path).resolve()
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return {'path': str(p), 'sha256': h.hexdigest(), 'bytes': p.stat().st_size}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ('A1', 'B1'):
        raise SystemExit('usage: merge_stage.py A1|B1')
    role = sys.argv[1]
    plan_path = HERE / 'MERGE_PLAN.json'
    plan = json.loads(plan_path.read_text())
    for key in ('merge_script', 'merge_python_binary', 'base_manifest'):
        if binding(plan[key]['path']) != plan[key]:
            raise RuntimeError('MERGE_INPUT_CHANGED:' + key)
    stage = plan['stages'][role]
    report = json.loads(Path(stage['training_result']).read_text())
    config_path = Path(stage['final_adapter']) / 'adapter_config.json'
    config = json.loads(config_path.read_text())
    if report['status'] != 'PASS_FORMAL_FIXED_ENDPOINT' or report['result']['optimizer_steps'] != 56:
        raise RuntimeError('FINAL_STEP56_TRAINING_REQUIRED')
    if config.get('step') != 56:
        raise RuntimeError('ADAPTER_STEP_MISMATCH')
    if Path(stage['output_dir']).exists():
        raise RuntimeError('REFUSE_TO_OVERWRITE_MERGE')
    inputs = [binding(stage['training_result']), binding(config_path),
              binding(Path(stage['final_adapter']) / 'adapter_model.pt')]
    start = HERE / 'merges' / (role + '_START.json')
    save(start, {'model_role': role, 'started_at_utc': now(),
                 'plan': binding(plan_path), 'launcher': binding(__file__),
                 'inputs': inputs, 'argv': stage['argv'], 'environment': stage['environment']})
    log = HERE / 'merges' / (role + '.log')
    with log.open('x') as stream:
        result = subprocess.run(stage['argv'], stdout=stream, stderr=subprocess.STDOUT,
                                env={**os.environ, **stage['environment'], 'PYTHONUNBUFFERED': '1'})
    terminal = HERE / 'merges' / (role + '_TERMINAL.json')
    output = Path(stage['output_dir'])
    save(terminal, {'model_role': role, 'completed_at_utc': now(),
                    'exit_code': result.returncode, 'log': binding(log),
                    'files': [binding(p) for p in sorted(output.iterdir()) if p.is_file()]
                    if output.exists() else []})
    if result.returncode:
        raise SystemExit(result.returncode)
    metadata = json.loads((output / 'merged_adapter_info.json').read_text())
    if (metadata['base_model'] != '/root/autodl-tmp/Qwen3-4B'
            or metadata['adapter_dir'] != stage['final_adapter']
            or metadata['merge_info'] != {'merged_linear_modules': 252, 'skipped_missing_pairs': []}):
        raise RuntimeError('MERGE_LINEAGE_OR_COVERAGE_FAILED')
    for ref in inputs:
        if binding(ref['path']) != ref:
            raise RuntimeError('ADAPTER_CHANGED_DURING_MERGE')
    proof = HERE / 'evaluation/endpoint_proofs' / (role + '.json')
    save(proof, {'model_role': role, 'model_directory': str(output),
                 'fixed_final_optimizer_step': 56, 'completed_optimizer_steps': 56,
                 'training_completed': True, 'independent_initialization': 'S0',
                 'merge_base': '/root/autodl-tmp/Qwen3-4B',
                 'adapter_representation': 'FULL_UPDATED_S0_CUSTOM_LORA',
                 'evidence_bindings': inputs + [binding(start), binding(terminal),
                    binding(output / 'merged_adapter_info.json'), binding(plan_path), binding(__file__)]})
    print(json.dumps({'status': 'CPU_MERGE_COMPLETE', 'model_role': role,
                      'output_directory': str(output), 'endpoint_proof': str(proof)}), flush=True)


if __name__ == '__main__':
    main()
