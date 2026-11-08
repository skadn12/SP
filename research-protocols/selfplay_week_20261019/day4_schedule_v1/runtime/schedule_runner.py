"""Replay a frozen, pre-expanded schedule using the historical custom LoRA code.

The default operation is a CPU-only preflight. Formal execution requires the
separately recorded Day 5 resource smoke acceptance. No model is loaded by
preflight, and no historical trainer source is modified.
"""
from __future__ import annotations

if not __debug__:
    raise RuntimeError('OPTIMIZED_PYTHON_NOT_ALLOWED')

import argparse
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path('/root/autodl-tmp/gxy-selfplay')
WEEK = REPO / 'research-protocols/selfplay_week_20261019'
HERE = Path(__file__).resolve().parent
POLICY = HERE.parent / 'EXECUTION_POLICY.json'
NATIVE = REPO / 'current_method_github/train_qwen3_lora_sft.py'
NATIVE_SHA256 = '918a45232e4287a26f46a5b4eb48175476a82136f5c3e202c94bf0d8cecb87dc'
BINDINGS = WEEK / 'day1/MODEL_BINDINGS.json'
TARGETS = 'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def binding(path):
    p = Path(path).resolve()
    return {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def write_new(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def native_module():
    assert sha(NATIVE) == NATIVE_SHA256, 'NATIVE_TRAINER_SOURCE_CHANGED'
    spec = importlib.util.spec_from_file_location('day4_bound_native_trainer', NATIVE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-file', required=True)
    parser.add_argument('--arm', choices=['A', 'B'], required=True)
    parser.add_argument('--expected-train-sha256', required=True)
    parser.add_argument('--expected-policy-sha256', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--resource-smoke', action='store_true')
    parser.add_argument('--peer-train-file')
    parser.add_argument('--expected-peer-sha256')
    parser.add_argument('--execute-training', action='store_true')
    parser.add_argument('--day5-acceptance')
    return parser.parse_args()


def checked_policy(args):
    assert sha(POLICY) == args.expected_policy_sha256, 'POLICY_BINDING_CHANGED'
    policy = json.loads(POLICY.read_text())
    s, b, t = policy['serialization'], policy['budget'], policy['training_schedule']
    assert s['preexpanded_displays'] is True
    assert s['epochs'] == 2 and s['runtime_epochs_over_preexpanded_data'] == 1
    assert s['runtime_shuffle'] is False and s['runtime_permutation'] is False
    assert s['truncation'] is False and s['full_input_max_tokens'] == 12288
    assert b['microbatch'] == 1 and b['gradient_accumulation'] == 4
    expected = {'lr': 1e-4, 'min_lr_ratio': .1, 'warmup_ratio': .05,
                'weight_decay': 0., 'max_grad_norm': 1., 'lora_rank': 16,
                'lora_alpha': 32, 'lora_dropout': .05}
    assert all(t[k] == v for k, v in expected.items()), 'HYPERPARAMETER_CHANGED'
    return policy


def model_identity(verify_all=False):
    models = json.loads(BINDINGS.read_text())['models']
    required = {k: models[k] for k in ['base', 'S0_M_clean_adapter']}
    tokenizer_files = {'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'}
    checked = []
    for role, item in required.items():
        for frozen in item['files']:
            if verify_all or (role == 'base' and Path(frozen['path']).name in tokenizer_files):
                actual = binding(frozen['path'])
                assert actual['sha256'] == frozen['sha256'], 'MODEL_OR_TOKENIZER_CHANGED'
                assert actual['bytes'] == frozen['bytes'], 'MODEL_OR_TOKENIZER_SIZE_CHANGED'
                checked.append({'role': role, **actual})
    return required, checked


def inspect_schedule(path, expected_hash, arm, tokenizer, native, policy):
    assert sha(path) == expected_hash, 'TRAIN_FILE_BINDING_CHANGED'
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    assert rows and len(rows) % 4 == 0, 'EMPTY_OR_PARTIAL_ACCUMULATION'
    max_length = policy['serialization']['full_input_max_tokens']
    features, measured, seen = [], [], set()
    previous_epoch = 1
    for index, row in enumerate(rows):
        assert all(k in row for k in ['messages', 'candidate_id', 'epoch', 'arm', 'category', 'schedule_key',
                                      'split', 'ordinal', 'expected_full_input_tokens',
                                      'expected_prompt_tokens', 'expected_supervised_tokens',
                                      'expected_token_ids_sha256'])
        assert row['arm'] == arm and row['epoch'] in [1, 2]
        assert row['split'] == 'train' and row['ordinal'] == index + 1, 'SPLIT_OR_ORDINAL_MISMATCH'
        assert row['epoch'] >= previous_epoch, 'EPOCH_ORDER_CHANGED'
        previous_epoch = row['epoch']
        key = (row['candidate_id'], row['epoch'])
        assert key not in seen, 'REPEATED_QUESTION_IN_EPOCH'
        seen.add(key)
        messages = row['messages']
        assert [m['role'] for m in messages] == ['system', 'user', 'assistant']
        answer = messages[-1]['content']
        assert answer.startswith('<box>') and answer.endswith('</box>')
        assert answer.count('<box>') == answer.count('</box>') == 1
        prompt = native.render_chat(tokenizer, messages[:-1], True)
        full = native.render_chat(tokenizer, messages, False)
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        full_ids = tokenizer(full, add_special_tokens=False).input_ids
        if tokenizer.eos_token_id is not None and (not full_ids or full_ids[-1] != tokenizer.eos_token_id):
            full_ids.append(tokenizer.eos_token_id)
        assert len(full_ids) <= max_length, 'OVER_LENGTH_NO_TRUNCATION_ALLOWED'
        assert full_ids[:len(prompt_ids)] == prompt_ids, 'CHAT_PROMPT_NOT_FULL_PREFIX'
        feature = native.tokenize_record(tokenizer, row, max_length)
        assert feature['input_ids'] == full_ids, 'NATIVE_TOKENIZATION_MISMATCH'
        supervised = sum(v != -100 for v in feature['labels'])
        assert supervised == len(full_ids) - min(len(prompt_ids), len(full_ids)) > 0
        item = {'index': index, 'candidate_id': row['candidate_id'], 'epoch': row['epoch'],
                'arm': arm, 'category': row['category'], 'schedule_key': row['schedule_key'],
                'prompt_tokens': len(prompt_ids), 'full_input_tokens': len(full_ids),
                'supervised_tokens': supervised, 'token_ids_sha256': canonical_sha(full_ids),
                'messages_sha256': canonical_sha(messages)}
        assert row['expected_full_input_tokens'] == len(full_ids), 'BUILDER_FULL_TOKEN_COUNT_MISMATCH'
        assert row['expected_prompt_tokens'] == len(prompt_ids), 'BUILDER_PROMPT_TOKEN_COUNT_MISMATCH'
        assert row['expected_supervised_tokens'] == supervised, 'BUILDER_SUPERVISED_TOKEN_COUNT_MISMATCH'
        assert row['expected_token_ids_sha256'] == canonical_sha(full_ids), 'BUILDER_TOKEN_DIGEST_MISMATCH'
        features.append(feature)
        measured.append(item)
    ids1 = {r['candidate_id'] for r in rows if r['epoch'] == 1}
    ids2 = {r['candidate_id'] for r in rows if r['epoch'] == 2}
    assert ids1 == ids2 and len(ids1) * 2 == len(rows), 'TWO_EPOCH_MEMBERSHIP_MISMATCH'
    for epoch in [1, 2]:
        part = [r for r in rows if r['epoch'] == epoch]
        assert len(part) % 4 == 0, 'EPOCH_PARTIAL_ACCUMULATION'
        seed = policy['serialization']['training_seed']
        expected = sorted(part, key=lambda r: (hashlib.sha256(
            (str(seed) + '|' + str(epoch) + '|' + r['schedule_key']).encode()).hexdigest(), r['schedule_key']))
        assert part == expected, 'SCHEDULE_HASH_ORDER_MISMATCH'
    summary = {'arm': arm, 'train_binding': binding(path), 'unique_questions': len(ids1),
               'microsteps': len(rows), 'optimizer_steps': len(rows) // 4,
               'full_input_tokens': sum(r['full_input_tokens'] for r in measured),
               'supervised_tokens': sum(r['supervised_tokens'] for r in measured),
               'maximum_input_tokens': max(r['full_input_tokens'] for r in measured),
               'rows': measured}
    return rows, features, summary


def run_arm(native, tokenizer, features, measured, arm, out, policy, smoke=False):
    """Only the schedule loop is new; all LoRA/loss/LR primitives are inherited."""
    import torch
    from transformers import AutoModelForCausalLM
    identity, _ = model_identity(verify_all=True)
    base = identity['base']['directory']
    initial = identity['S0_M_clean_adapter']['directory']
    assert torch.cuda.is_available(), 'CUDA_REQUIRED_FOR_RESOURCE_SMOKE_AND_FORMAL_TRAINING'
    random.seed(policy['serialization']['training_seed'])
    torch.manual_seed(policy['serialization']['training_seed'])
    args = SimpleNamespace(model=base, tokenizer=base, lora_rank=16, lora_alpha=32,
                           lora_dropout=.05, target_modules=TARGETS, max_length=12288,
                           init_adapter_dir=initial)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda')
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,
                                                trust_remote_code=True, local_files_only=True)
    native.freeze_base_model(model)
    replaced = native.inject_lora(model, set(TARGETS.split(',')), 16, 32, .05)
    assert replaced == 252, 'LORA_MODULE_COUNT_MISMATCH'
    native.load_initial_adapter(model, initial, args)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.to(device)
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.)
    optimizer.zero_grad(set_to_none=True)
    steps = len(features) // 4
    assert len(features) % 4 == 0 and steps > 0
    telemetry = out / 'telemetry.jsonl'
    losses, step = [], 0
    for i, (feature, row) in enumerate(zip(features, measured)):
        batch = native.collate([feature], tokenizer.pad_token_id, device)
        loss = model(**batch).loss
        assert bool(torch.isfinite(loss.detach()).all()), 'NONFINITE_LOSS'
        value = float(loss.detach().cpu())
        (loss / 4).backward()
        losses.append(value)
        event = {'microstep': i + 1, 'arm': arm, 'candidate_id': row['candidate_id'],
                 'epoch': row['epoch'], 'schedule_key': row['schedule_key'], 'loss': value,
                 'input_tokens': row['full_input_tokens'], 'supervised_tokens': row['supervised_tokens']}
        if (i + 1) % 4 == 0:
            grads = [p.grad for p in parameters if p.grad is not None]
            assert grads and all(bool(torch.isfinite(g).all()) for g in grads), 'NONFINITE_GRADIENT'
            assert any(bool(torch.count_nonzero(g)) for g in grads), 'ALL_ZERO_GRADIENT'
            lr = native.lr_at_step(step, steps, 1e-4, .1, .05)
            native.set_lr(optimizer, lr)
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            assert bool(torch.isfinite(norm)), 'NONFINITE_GRADIENT_NORM'
            optimizer.step()
            assert all(bool(torch.isfinite(p.detach()).all()) for p in parameters), 'NONFINITE_UPDATED_PARAMETER'
            optimizer.zero_grad(set_to_none=True)
            step += 1
            event.update(optimizer_step=step, learning_rate=lr,
                         gradient_norm_before_clip=float(norm.detach().cpu()),
                         updated_parameters_finite=True)
        torch.cuda.synchronize()
        event.update(wall_seconds=time.monotonic() - started,
                     peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                     peak_reserved_bytes=torch.cuda.max_memory_reserved())
        with telemetry.open('a') as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + '\n')
    assert step == steps
    if not smoke:
        native.save_adapter(model, out / 'adapter', args, step, {'fixed_final_endpoint': True})
        tokenizer.save_pretrained(out / 'adapter')
    result = {'status': 'PASS', 'mode': 'isolated_resource_smoke' if smoke else 'formal_fixed_endpoint',
              'arm': arm, 'microsteps': len(features), 'optimizer_steps': step,
              'mean_loss': sum(losses) / len(losses),
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
              'wall_seconds': time.monotonic() - started,
              'telemetry': binding(telemetry), 'formal_model_artifact_saved': not smoke}
    del loss, batch, model, optimizer, parameters, grads
    torch.cuda.empty_cache()
    return result


def validate_acceptance(acceptance, common):
    """Require complete, compatible evidence from both isolated smoke arms."""
    assert acceptance['status'] == 'PASS_ISOLATED_RESOURCE_SMOKE'
    for field in ['runtime', 'native_trainer', 'policy', 'model_binding_manifest',
                  'versions', 'python', 'executable', 'python_binary', 'torch_cuda_build']:
        assert acceptance[field] == common[field], 'RESOURCE_ACCEPTANCE_BINDING_CHANGED:' + field
    assert acceptance['formal_training_runs'] == 0
    assert acceptance['formal_artifact_saved'] is False
    schedules = [acceptance['schedule'], acceptance['peer_schedule']]
    assert len(schedules) == 2 and {s['arm'] for s in schedules} == {'A', 'B'}, 'BOTH_SMOKE_SCHEDULES_REQUIRED'
    assert any(s['train_binding'] == common['schedule']['train_binding'] for s in schedules)
    for schedule in schedules:
        assert binding(schedule['train_binding']['path']) == schedule['train_binding'], 'SMOKE_DATA_CHANGED'
    results = acceptance['results']
    assert len(results) == 2 and {r['arm'] for r in results} == {'A', 'B'}, 'BOTH_SMOKE_RESULTS_REQUIRED'
    for result in results:
        assert result['status'] == 'PASS' and result['mode'] == 'isolated_resource_smoke'
        assert result['microsteps'] == 4 and result['optimizer_steps'] == 1, 'SMOKE_ACCUMULATION_EVIDENCE_INCOMPLETE'
        assert result['formal_model_artifact_saved'] is False, 'SMOKE_FORMAL_ARTIFACT_NOT_ALLOWED'
        assert binding(result['telemetry']['path']) == result['telemetry'], 'SMOKE_TELEMETRY_CHANGED'
        events = [json.loads(line) for line in Path(result['telemetry']['path']).read_text().splitlines()]
        assert len(events) == 4 and [e['microstep'] for e in events] == [1, 2, 3, 4]
        schedule = next(s for s in schedules if s['arm'] == result['arm'])
        longest = sorted(schedule['rows'], key=lambda row: (-row['full_input_tokens'], row['index']))[:4]
        for event, row in zip(events, longest):
            assert event['arm'] == result['arm']
            assert all(event[k] == row[k] for k in ['candidate_id', 'epoch', 'schedule_key'])
            assert event['input_tokens'] == row['full_input_tokens']
            assert event['supervised_tokens'] == row['supervised_tokens']
            assert math.isfinite(event['loss'])
        updates = [e for e in events if 'optimizer_step' in e]
        assert len(updates) == 1 and updates[0]['microstep'] == 4 and updates[0]['optimizer_step'] == 1
        assert math.isfinite(updates[0]['gradient_norm_before_clip']) and updates[0]['gradient_norm_before_clip'] > 0
        assert updates[0]['updated_parameters_finite'] is True


def main():
    args = parse_args()
    assert sum([args.preflight_only, args.resource_smoke, args.execute_training]) == 1, 'EXPLICIT_MODE_REQUIRED'
    assert not Path(args.report).exists(), 'REFUSE_TO_OVERWRITE_REPORT'
    policy = checked_policy(args)
    identity, checked = model_identity()
    native = native_module()
    tokenizer = native.AutoTokenizer.from_pretrained(identity['base']['directory'],
                                                    trust_remote_code=False, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows, features, summary = inspect_schedule(args.train_file, args.expected_train_sha256,
                                               args.arm, tokenizer, native, policy)
    common = {'schema_version': 'day4_schedule_runtime_v1',
              'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'runtime': binding(__file__), 'native_trainer': binding(NATIVE),
              'policy': binding(POLICY), 'model_binding_manifest': binding(BINDINGS),
              'tokenizer_files_verified': checked,
              'versions': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'tokenizers', 'safetensors']},
              'python': sys.version, 'executable': sys.executable,
              'python_binary': binding(sys.executable), 'torch_cuda_build': native.torch.version.cuda,
              'schedule': summary, 'shuffle': False, 'permutation': False,
              'runtime_epochs': 1, 'truncation': False}
    if args.preflight_only:
        write_new(args.report, {**common, 'status': 'PASS_CPU_PREFLIGHT', 'model_weights_loaded': False,
                                'training_runs': 0, 'resource_smoke_passed': False,
                                'formal_training_execution_permitted': False})
    elif args.resource_smoke:
        assert args.peer_train_file and args.expected_peer_sha256, 'BOTH_ARMS_REQUIRED_FOR_SMOKE'
        peer_arm = 'B' if args.arm == 'A' else 'A'
        _, peer_features, peer_summary = inspect_schedule(args.peer_train_file, args.expected_peer_sha256,
                                                        peer_arm, tokenizer, native, policy)
        assert [r['schedule_key'] for r in summary['rows']] == [r['schedule_key'] for r in peer_summary['rows']]
        assert summary['microsteps'] == peer_summary['microsteps']
        out = Path(args.output_dir)
        assert not out.exists(), 'REFUSE_TO_OVERWRITE_SMOKE'
        out.mkdir(parents=True)
        results = []
        for arm, fs, sm in [(args.arm, features, summary), (peer_arm, peer_features, peer_summary)]:
            longest = sorted(range(len(fs)), key=lambda i: (-sm['rows'][i]['full_input_tokens'], i))[:4]
            assert len(longest) == 4
            results.append(run_arm(native, tokenizer, [fs[i] for i in longest],
                                   [sm['rows'][i] for i in longest], arm, out / arm, policy, smoke=True))
        write_new(args.report, {**common, 'status': 'PASS_ISOLATED_RESOURCE_SMOKE',
                                'peer_schedule': peer_summary, 'results': results,
                                'formal_training_runs': 0, 'formal_artifact_saved': False})
    else:
        assert args.day5_acceptance, 'DAY5_RESOURCE_ACCEPTANCE_REQUIRED'
        acceptance = json.loads(Path(args.day5_acceptance).read_text())
        validate_acceptance(acceptance, common)
        result = run_arm(native, tokenizer, features, summary['rows'], args.arm,
                         args.output_dir, policy)
        write_new(args.report, {**common, 'status': 'PASS_FORMAL_FIXED_ENDPOINT',
                                'day5_acceptance': binding(args.day5_acceptance), 'result': result})
    print(json.dumps({'status': 'COMPLETE', 'report': str(Path(args.report).resolve())}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
