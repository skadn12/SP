"""Independently verify a completed v126 run, using CPU and frozen artifacts only.

Run with the project's Python after the controller has completed. This script
never imports the trainer, loads base-model weights, performs inference, or
starts training. It writes one audit report outside the sealed v125 package.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parent.parent
POOL = REPO / 'offline-deliveries/h4_codex_train_pool_v125'
FORMAL = REPO / 'restart-n5-runs/h4_codex_combined_sft_v125'
SMOKE = HERE / 'smoke'
INITIAL = REPO / 'restart-n5-runs/h4_round0_b0_formal_training_v1_live_2/clean_adapter'
EXPECTED_POOL_SHA = '63ff60a7b4761a8b27fac747a21c7379fff1e64c195c21ef934e2ff601f5f6b6'
EXPECTED_DATA_SHA = '3ad675fd36212aef3636d21fa19f1e45f3144a1351b66b369acaa118eb5ba43b'
EXPECTED_PLAN_SHA = 'c0891bda67e4833d0c777b9f64276f3e3c85fb11ebd6b379d062088dc0a055b6'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def objects_in_log(text):
    decoder = json.JSONDecoder()
    position = 0
    objects = []
    while position < len(text):
        index = text.find('{', position)
        if index < 0:
            break
        try:
            value, length = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            position = index + 1
        else:
            if isinstance(value, dict):
                objects.append(value)
            position = index + length
    return objects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'evidence/INDEPENDENT_FINAL_TRAINING_AUDIT.json')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(POOL) or output.is_relative_to(FORMAL) or output.is_relative_to(SMOKE):
        raise RuntimeError('Audit output must not alter sealed data or trainer outputs')
    status_path = HERE / 'RUN_STATUS.json'
    status = json.loads(status_path.read_text())
    if status.get('status') != 'TRAINING_COMPLETE_AWAITING_EVALUATION':
        print(json.dumps({'status': 'PENDING', 'controller_status': status.get('status'),
                          'reason': 'Final PASS requires completed formal training; no report written.'}))
        return 2

    errors, tracked = [], {}
    def require(condition, code, **details):
        if not condition:
            errors.append({'code': code, **details})
    def track(path):
        path = Path(path).resolve()
        value = sha(path)
        if str(path) in tracked:
            require(tracked[str(path)] == value, 'INPUT_CHANGED_DURING_AUDIT', path=str(path))
        tracked[str(path)] = value
        return value
    def read(path):
        track(path)
        return json.loads(Path(path).read_text())
    def readl(path):
        track(path)
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    def finite(value):
        return isinstance(value, (int, float)) and math.isfinite(value)
    def arg(argv, key):
        require(argv.count(key) == 1, 'ARGUMENT_MISSING_OR_DUPLICATED', argument=key)
        return argv[argv.index(key) + 1]
    def replace(argv, key, value):
        result = list(argv)
        result[result.index(key) + 1] = str(value)
        return result

    track(__file__)
    track(status_path)
    freeze = read(HERE / 'EXECUTION_FREEZE.json')
    require(track(POOL / 'COMPLETE_MANIFEST.json') == EXPECTED_POOL_SHA == freeze['source_package_manifest_sha256'], 'POOL_SEAL_CHANGED')
    seal = read(POOL / 'COMPLETE_MANIFEST.json')
    for entry in seal['files']:
        path = POOL / entry['path']
        require(path.is_file() and track(path) == entry['sha256'], 'SEALED_POOL_FILE_CHANGED', path=str(path))
    actual_pool_files = {str(p.relative_to(POOL)) for p in POOL.rglob('*') if p.is_file()
                         and '__pycache__' not in p.parts and p.name != 'COMPLETE_MANIFEST.json'}
    require(actual_pool_files == {entry['path'] for entry in seal['files']}, 'POOL_FILE_SET_CHANGED')
    external = read(POOL / 'INPUT_FREEZE.json')
    require(len(external['files']) == len({entry['path'] for entry in external['files']}) == 1102, 'EXTERNAL_BINDING_COUNT')
    for entry in external['files']:
        path = Path(entry['path'])
        require(path.is_absolute() and not path.is_relative_to('/tmp') and path.is_file()
                and track(path) == entry['sha256'], 'EXTERNAL_BINDING_CHANGED', path=str(path))

    plan = read(POOL / 'TRAINING_PLAN.json')
    protocol = read(POOL / 'PROTOCOL.json')
    require(track(POOL / 'TRAINING_PLAN.json') == EXPECTED_PLAN_SHA == freeze['training_plan_sha256'], 'TRAINING_PLAN_CHANGED')
    require(track(POOL / 'data/TRAIN_NEW_ONLY.jsonl') == EXPECTED_DATA_SHA == freeze['training_data_sha256'], 'TRAINING_DATA_CHANGED')
    require(plan['training_started'] is False and plan['status'] == 'PREPARED_NOT_RUN', 'SEALED_PLAN_MUTATED')
    require(freeze['formal_original_argv'] == plan['argv'], 'ORIGINAL_PLAN_ARGV_CHANGED')
    require(freeze['formal_argv'] == [plan['argv'][0], str(HERE / 'training_instrumentation.py'), *plan['argv'][3:]], 'FORMAL_ARGUMENTS_CHANGED')
    require(freeze['native_runner_module'] == plan['argv'][2], 'NATIVE_RUNNER_CHANGED')
    for key in ['controller_binding', 'telemetry_launcher_binding']:
        require(track(freeze[key]['path']) == freeze[key]['sha256'], 'RUNTIME_CODE_CHANGED', binding=key)
    formal_argv, smoke_argv = freeze['formal_argv'], freeze['smoke_argv']
    expected_smoke = replace(formal_argv, '--train-file', SMOKE / 'TRAIN_SMOKE.jsonl')
    expected_smoke = replace(expected_smoke, '--output-dir', SMOKE / 'adapter')
    expected_smoke = replace(expected_smoke, '--epochs', '1.0') + ['--eval-steps', '1']
    require(smoke_argv == expected_smoke, 'UNREVIEWED_SMOKE_ARGUMENT_DIFFERENCE')
    for argv, folder, epochs, records in [(formal_argv, FORMAL, '2.0', 363), (smoke_argv, SMOKE, '1.0', 4)]:
        require(arg(argv, '--init-adapter-dir') == str(INITIAL), 'WRONG_INITIAL_ADAPTER', folder=str(folder))
        require(arg(argv, '--output-dir') == str(folder / 'adapter'), 'OUTPUT_DIRECTORY_MIXUP', folder=str(folder))
        require(arg(argv, '--epochs') == epochs and arg(argv, '--batch-size') == '1'
                and arg(argv, '--grad-accum') == '4' and arg(argv, '--max-length') == '12288', 'TRAINING_SHAPE_CHANGED', folder=str(folder))
        require('--valid-file' not in argv and '--max-train-samples' not in argv, 'TRAINING_FILTER_OR_HELDOUT_ARGUMENT')
        require(len(readl(arg(argv, '--train-file'))) == records, 'WRONG_TRAINING_FILE_COUNT', folder=str(folder))
    require(arg(formal_argv, '--train-file') == str(POOL / 'data/TRAIN_NEW_ONLY.jsonl'), 'FORMAL_USES_SMOKE_OR_OTHER_DATA')
    require(freeze['smoke_training_not_reused'] is True and freeze['test_evaluation'] is False, 'SMOKE_REUSE_OR_EVALUATION_POLICY')
    require(read(FORMAL / 'LAUNCH_BINDING.json') == freeze, 'FORMAL_LAUNCH_FREEZE_MISMATCH')

    rows = readl(POOL / 'data/TRAIN_NEW_ONLY.jsonl')
    lineage = readl(POOL / 'data/QUESTION_LINEAGE.jsonl')
    by_id = {row['candidate_id']: row for row in rows}
    lmap = {row['candidate_id']: row for row in lineage}
    raw = read(POOL / 'data/RAW_SELECTED_QUESTIONS.json')['questions']
    original = [q for entry in protocol['inputs'] for q in read(entry['path'])['questions']]
    require(raw == original, 'RAW_CANDIDATE_SET_NOT_LOSSLESS')
    require(len(rows) == len(by_id) == len(lineage) == len(lmap) == len(raw) == 363, 'TRAINING_RECORD_CARDINALITY')
    require(set(by_id) == set(lmap) == {q['candidate_id'] for q in raw}, 'TRAINING_MEMBER_SET_CHANGED')
    heldout = {row['root_id'] for row in read(protocol['split']['heldout_population_binding']['path'])['roots']}
    splitmap = {row['root_id']: row['split'] for row in readl(protocol['split']['authority']['path'])}
    roots = {row['root_id'] for row in rows}
    require(len(roots) == 68 and len(heldout) == 31 and roots.isdisjoint(heldout), 'ROOT_SPLIT_LEAKAGE')
    require(all(row['split'] == 'train' and splitmap[row['root_id']] == 'train'
                and row['selection_role'] == 'new_selected' for row in rows), 'REPLAY_OR_NONTRAIN_RECORD')
    require(protocol['replay']['count'] == 0, 'REPLAY_POLICY_CHANGED')
    for row in rows:
        require(canonical_sha(row['messages']) == lmap[row['candidate_id']]['messages_sha256'], 'TRAINING_MESSAGES_CHANGED', candidate=row['candidate_id'])
    chosen = []
    for qtype in ['multi_select', 'ordering']:
        chosen.extend(sorted((row for row in lineage if row['question_type'] == qtype),
                             key=lambda row: (-row['input_token_count'], row['candidate_id']))[:2])
    require(freeze['smoke_candidates'] == [{'candidate_id': row['candidate_id'], 'input_tokens': row['input_token_count']} for row in chosen], 'SMOKE_NOT_LONGEST_TWO_PER_TYPE')
    require(readl(SMOKE / 'TRAIN_SMOKE.jsonl') == [by_id[row['candidate_id']] for row in chosen], 'SMOKE_ROWS_NOT_EXACT_SUBSET')
    preflight_log = HERE / 'preflight.log'
    track(preflight_log)
    require('PASS: 363 new questions / 68 train roots / 0 replay / 0 truncated / heldouts excluded' in preflight_log.read_text(), 'PREFLIGHT_PASS_MISSING')

    initial_config = read(INITIAL / 'adapter_config.json')
    initial_state_sha = track(INITIAL / 'adapter_model.pt')
    require(track(INITIAL / 'adapter_model.pt') != track(SMOKE / 'adapter/adapter_model.pt'), 'SMOKE_ADAPTER_UNCHANGED')
    require(track(INITIAL / 'adapter_model.pt') != track(FORMAL / 'adapter/adapter_model.pt'), 'FORMAL_ADAPTER_UNCHANGED')
    require(track(SMOKE / 'adapter/adapter_model.pt') != track(FORMAL / 'adapter/adapter_model.pt'), 'FORMAL_ADAPTER_EQUALS_SMOKE')

    training_results = {}
    for label, folder, expected_micro, expected_updates, count in [('smoke', SMOKE, 4, 1, 4), ('formal', FORMAL, 726, 182, 363)]:
        audit = read(folder / 'gradient_audit.json')
        config = read(folder / 'adapter/adapter_config.json')
        events = readl(folder / 'telemetry.jsonl')
        backward = [event for event in events if event['event'] == 'backward_complete']
        clips = [event for event in events if event['event'] == 'gradient_clip_complete']
        native_backwards, native_gradients = audit['backward_records'], audit['gradient_records']
        require(audit['backward_calls'] == len(native_backwards) == len(backward) == expected_micro, 'BACKWARD_COUNT', run=label)
        require(audit['optimizer_steps'] == len(native_gradients) == len(clips) == expected_updates, 'UPDATE_COUNT', run=label)
        require(audit['all_losses_finite'] is True and audit['all_gradients_finite_nonzero'] is True, 'NATIVE_AUDIT_FAILED', run=label)
        require([row['micro_step'] for row in backward] == list(range(1, expected_micro + 1)), 'TELEMETRY_MICRO_SEQUENCE', run=label)
        require([row['micro_step'] for row in native_backwards] == list(range(1, expected_micro + 1)), 'NATIVE_MICRO_SEQUENCE', run=label)
        require([row['step'] for row in clips] == list(range(1, expected_updates + 1)), 'TELEMETRY_UPDATE_SEQUENCE', run=label)
        require([row['optimizer_step'] for row in native_gradients] == list(range(1, expected_updates + 1)), 'NATIVE_UPDATE_SEQUENCE', run=label)
        require([row['micro_step'] for row in clips] == [min(i * 4, expected_micro) for i in range(1, expected_updates + 1)], 'UPDATE_ACCUMULATION_TIMING', run=label)
        expected_event_sequence = []
        for micro in range(1, expected_micro + 1):
            expected_event_sequence.append('backward_complete')
            if micro % 4 == 0 or micro == expected_micro:
                expected_event_sequence.append('gradient_clip_complete')
        require([event['event'] for event in events] == expected_event_sequence, 'EVENT_ORDER_OR_EXTRA_EVENTS', run=label)
        for native, telemetry in zip(native_backwards, backward):
            require(native['loss_finite'] is True and finite(native['scaled_loss']) and finite(telemetry['unscaled_loss'])
                    and math.isclose(telemetry['unscaled_loss'], 4 * native['scaled_loss'], rel_tol=1e-7, abs_tol=1e-10), 'LOSS_NOT_FINITE_OR_MISMATCH', run=label, micro=telemetry['micro_step'])
        for native, telemetry in zip(native_gradients, clips):
            require(native['gradient_finite_before_clip'] is True and native['gradient_nonzero_elements'] > 0
                    and finite(native['gradient_l2_norm_before_clip']) and native['gradient_l2_norm_before_clip'] > 0
                    and finite(native['clipped_norm']) and native['clipped_norm'] > 0
                    and finite(telemetry['gradient_norm_before_clip']) and telemetry['gradient_norm_before_clip'] > 0
                    and math.isclose(native['clipped_norm'], telemetry['gradient_norm_before_clip'], rel_tol=1e-6), 'GRADIENT_NOT_FINITE_NONZERO_OR_MISMATCH', run=label, step=telemetry['step'])
        require(config['step'] == expected_updates and config['max_length'] == 12288 and config['init_adapter_dir'] == str(INITIAL), 'SAVED_ADAPTER_WRONG_STEP_LENGTH_OR_INITIALIZER', run=label)
        for key in ['model', 'tokenizer', 'lora_rank', 'lora_alpha', 'lora_dropout', 'target_modules']:
            require(config[key] == initial_config[key], 'SAVED_ADAPTER_IDENTITY_OR_LORA_CHANGED', run=label, field=key)
        log_path = folder / 'train.log'
        track(log_path)
        log_objects = objects_in_log(log_path.read_text())
        starts = [obj for obj in log_objects if 'train_records' in obj]
        finishes = [obj for obj in log_objects if obj.get('finished') is True]
        require(len(starts) == 1 and len(finishes) == 1, 'START_OR_FINISH_LOG_MISSING_OR_DUPLICATED', run=label)
        if len(starts) == 1:
            start = starts[0]
            require(start['train_records'] == count and start['valid_records'] == 0 and start['total_update_steps'] == expected_updates, 'NATIVE_DATA_OR_UPDATE_COUNT', run=label)
            require(start['init_adapter'] == {'init_adapter_dir': str(INITIAL), 'init_adapter_config': initial_config}, 'NATIVE_INITIAL_ADAPTER_NOT_EXACT_MCLEAN', run=label)
            require(start['trainable_parameters'] == 33030144 and start['target_modules_replaced'] == 252, 'LORA_PARAMETER_COUNT_CHANGED', run=label)
        if len(finishes) == 1:
            require(finishes[0]['global_step'] == expected_updates and finishes[0]['final_valid_loss'] is None, 'FINAL_STEP_OR_HELDOUT_EVALUATION', run=label)
        losses = [event['unscaled_loss'] for event in backward]
        training_results[label] = {'backward_calls': len(backward), 'optimizer_steps': len(clips),
                                   'all_losses_finite': all(finite(value) for value in losses),
                                   'mean_loss': sum(losses) / len(losses), 'first_25_mean_loss': sum(losses[:25]) / len(losses[:25]),
                                   'last_25_mean_loss': sum(losses[-25:]) / len(losses[-25:]),
                                   'adapter_state_sha256': track(folder / 'adapter/adapter_model.pt'),
                                   'peak_allocated_bytes': max(event['peak_allocated_bytes'] for event in backward),
                                   'peak_reserved_bytes': max(event['peak_reserved_bytes'] for event in backward)}
    formal_result = read(FORMAL / 'TRAINING_RESULT.json')
    smoke_result = read(SMOKE / 'RESULT.json')
    require(smoke_result['status'] == 'PASS' and smoke_result['smoke_only_not_final_model'] is True, 'SMOKE_RESULT_NOT_PASS_OR_MISLABELED')
    require(formal_result['status'] == 'PASS' and formal_result['training_records'] == 363 and formal_result['root_count'] == 68
            and formal_result['epochs'] == 2 and formal_result['replay_count'] == 0 and formal_result['evaluation_started'] is False, 'FORMAL_RESULT_MISMATCH')
    require(status['result'] == formal_result and status['formal_status'] == 'PASS' and status['smoke_status'] == 'PASS', 'CONTROLLER_FINAL_RESULT_MISMATCH')

    # Loading the three small LoRA states on CPU verifies the files contain
    # finite, shape-compatible tensors and a real numerical parameter update.
    # Base-model weights are never loaded, and no CUDA operation is performed.
    import torch
    states = {}
    for label, directory in [('initial', INITIAL), ('smoke', SMOKE / 'adapter'), ('formal', FORMAL / 'adapter')]:
        state = torch.load(directory / 'adapter_model.pt', map_location='cpu', weights_only=True)
        require(isinstance(state, dict) and bool(state), 'ADAPTER_STATE_NOT_MAPPING', run=label)
        require(all(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and bool(torch.isfinite(value).all()) for value in state.values()), 'ADAPTER_STATE_NOT_FINITE_CPU_TENSORS', run=label)
        states[label] = state
    tensor_results = {}
    for label in ['smoke', 'formal']:
        require(set(states[label]) == set(states['initial']), 'ADAPTER_TENSOR_KEYS_CHANGED', run=label)
        changed_tensors = 0
        changed_elements = 0
        for key, value in states[label].items():
            initial = states['initial'][key]
            require(value.shape == initial.shape and value.dtype == initial.dtype, 'ADAPTER_TENSOR_SHAPE_OR_DTYPE_CHANGED', run=label, tensor=key)
            if value.shape == initial.shape:
                count = int(torch.count_nonzero(value != initial))
                changed_elements += count
                changed_tensors += count > 0
        require(changed_tensors > 0 and changed_elements > 0, 'ADAPTER_NUMERICALLY_UNCHANGED', run=label)
        tensor_results[label] = {'changed_tensor_count': changed_tensors, 'changed_element_count': changed_elements,
                                 'state_tensor_count': len(states[label]), 'all_finite': True}
    changed_inputs = [path for path, digest in tracked.items() if sha(path) != digest]
    require(not changed_inputs, 'AUDITED_INPUT_CHANGED', paths=changed_inputs)
    report = {'schema_version': 'v126_independent_final_training_audit_v1', 'status': 'PASS' if not errors else 'FAIL',
              'errors': errors, 'controller_status': status['status'], 'source_package_manifest_sha256': EXPECTED_POOL_SHA,
              'sealed_pool_file_count': len(seal['files']), 'external_frozen_reference_count': len(external['files']),
              'training_data_sha256': EXPECTED_DATA_SHA, 'training_record_count': len(rows), 'train_root_count': len(roots),
              'dev_test_training_roots': sorted(roots & heldout), 'replay_count': 0,
              'smoke_candidates': freeze['smoke_candidates'], 'smoke_formal_original_initializer_verified': True,
              'initial_adapter_state_sha256': initial_state_sha, 'training_results': training_results,
              'adapter_tensor_verification_cpu': tensor_results, 'inputs_changed_during_audit': changed_inputs,
              'limitations': ['Training loss is not Solver accuracy or heldout quality.',
                              'No per-example IDs were instrumented at runtime. Complete dataset use is supported by frozen exact 363-row data, native loader/start count, unchanged no-filter two-epoch loop and 726 backward calls; this is not an independent per-example execution trace.',
                              'Native accumulation crosses epochs and divides the final two microbatches by four; the audit verifies that frozen behavior without modifying it.',
                              'CPU-only artifact verification; no generation, inference, evaluation, optimizer step or GPU allocation performed.'],
              'audited_files': [{'path': path, 'sha256': digest} for path, digest in sorted(tracked.items())]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'errors': errors, 'report': str(output), 'report_sha256': sha(output),
                      'training_results': training_results, 'adapter_tensor_verification_cpu': tensor_results}, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
