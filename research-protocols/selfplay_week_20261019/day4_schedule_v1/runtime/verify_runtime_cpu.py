"""Exercise fail-closed preflight checks against temporary mutated schedules."""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('schedule_runtime_under_test', HERE / 'schedule_runner.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
policy = json.loads(runtime.POLICY.read_text())
native = runtime.native_module()
tokenizer = native.AutoTokenizer.from_pretrained('/root/autodl-tmp/Qwen3-4B', local_files_only=True,
                                               trust_remote_code=False)
source = HERE.parent / 'TRAIN_POOL_A.jsonl'
rows = [json.loads(line) for line in source.read_text().splitlines()]
results = []


def check(name, changed, expected_error, hash_override=None):
    with tempfile.TemporaryDirectory(prefix='day4_runtime_negative_') as directory:
        path = Path(directory) / 'mutated.jsonl'
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in changed))
        try:
            runtime.inspect_schedule(path, hash_override or runtime.sha(path), 'A', tokenizer, native, policy)
        except AssertionError as error:
            actual = str(error)
            assert actual == expected_error, (name, actual, expected_error)
            results.append({'case': name, 'expected_error': expected_error, 'status': 'PASS_REJECTED'})
        else:
            raise AssertionError('MUTATION_INCORRECTLY_ACCEPTED:' + name)


check('changed_training_file', rows, 'TRAIN_FILE_BINDING_CHANGED', '0' * 64)
changed = copy.deepcopy(rows)
changed[0]['expected_token_ids_sha256'] = '0' * 64
check('builder_token_digest_mismatch', changed, 'BUILDER_TOKEN_DIGEST_MISMATCH')
changed = copy.deepcopy(rows)
changed[0]['ordinal'] = 2
check('changed_ordinal', changed, 'SPLIT_OR_ORDINAL_MISMATCH')
changed = copy.deepcopy(rows)
changed[0]['messages'][1]['content'] += ' marker' * 13000
check('overlength_must_not_left_truncate', changed, 'OVER_LENGTH_NO_TRUNCATION_ALLOWED')
changed = copy.deepcopy(rows)
changed[0], changed[1] = changed[1], changed[0]
changed[0]['ordinal'], changed[1]['ordinal'] = 1, 2
check('changed_hash_order', changed, 'SCHEDULE_HASH_ORDER_MISMATCH')
summary_a = json.loads((HERE / 'PREFLIGHT_A.json').read_text())['schedule']
summary_b = json.loads((HERE / 'PREFLIGHT_B.json').read_text())['schedule']
common = {k: 'same-frozen-identity' for k in ['runtime', 'native_trainer', 'policy',
          'model_binding_manifest', 'versions', 'python', 'executable', 'python_binary', 'torch_cuda_build']}
common['schedule'] = summary_a
acceptance = {**common, 'status': 'PASS_ISOLATED_RESOURCE_SMOKE',
              'formal_training_runs': 0, 'formal_artifact_saved': False,
              'peer_schedule': summary_b, 'results': []}


def check_acceptance(name, mutated, expected_error):
    try:
        runtime.validate_acceptance(mutated, common)
    except AssertionError as error:
        assert str(error) == expected_error, (name, str(error), expected_error)
        results.append({'case': name, 'expected_error': expected_error, 'status': 'PASS_REJECTED'})
    else:
        raise AssertionError('INCOMPLETE_SMOKE_ACCEPTED:' + name)


check_acceptance('empty_smoke_results', acceptance, 'BOTH_SMOKE_RESULTS_REQUIRED')
changed = copy.deepcopy(acceptance)
changed['peer_schedule'] = summary_a
check_acceptance('duplicated_arm_smoke_schedules', changed, 'BOTH_SMOKE_SCHEDULES_REQUIRED')
changed = copy.deepcopy(acceptance)
changed['versions'] = 'changed-version'
check_acceptance('changed_runtime_versions', changed, 'RESOURCE_ACCEPTANCE_BINDING_CHANGED:versions')
changed = copy.deepcopy(acceptance)
changed['results'] = [{'arm': arm, 'status': 'PASS', 'mode': 'isolated_resource_smoke',
                       'microsteps': 0, 'optimizer_steps': 0} for arm in ['A', 'B']]
check_acceptance('missing_smoke_accumulation', changed, 'SMOKE_ACCUMULATION_EVIDENCE_INCOMPLETE')
optimized = subprocess.run([sys.executable, '-O', str(HERE / 'schedule_runner.py'), '--help'],
                           capture_output=True, text=True)
assert optimized.returncode != 0 and 'OPTIMIZED_PYTHON_NOT_ALLOWED' in optimized.stderr
results.append({'case': 'optimized_python_cannot_disable_assertions', 'status': 'PASS_REJECTED'})
runtime.write_new(HERE / 'NEGATIVE_PREFLIGHT_CHECKS_V2.json', {
    'schema_version': 'day4_runtime_cpu_negative_checks_v1', 'status': 'PASS',
    'runtime': runtime.binding(HERE / 'schedule_runner.py'),
    'test_script': runtime.binding(__file__), 'source_schedule': runtime.binding(source),
    'cases': results, 'model_weights_loaded': False, 'gpu_calls': 0, 'training_runs': 0,
    'scope': 'CPU tokenization, file identity and schedule rejection checks; no GPU/backpropagation claim'})
print(json.dumps({'status': 'PASS', 'cases': len(results)}))
