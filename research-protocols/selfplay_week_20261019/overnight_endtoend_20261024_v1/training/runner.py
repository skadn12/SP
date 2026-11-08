"""OET1 isolated B1 + fresh custom LoRA, exact resume, and shared DEV answers.

CPU preflight is default. GPU commands require --execute-gpu, acquire one common
flock, preserve fixed schedules, and never read Test or change old experiments.
"""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import importlib.metadata
import math
import random
import shutil
import time
import traceback
import uuid
from types import SimpleNamespace
from common import *


def versions():
    out = {'python': sys.version}
    for name in ('torch', 'transformers', 'tokenizers', 'wandb'):
        try: out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: out[name] = None
    return out


def verify_schedule(path, arm):
    frozen = read(path)
    require(frozen['status'] == 'FROZEN_CPU_PREFLIGHT_PASS', 'SCHEDULE_NOT_FROZEN')
    verify([frozen['pool_freeze'], frozen['protocol'], frozen['evaluation_freeze']] + frozen['code_bindings'] + frozen['model_bindings'] + frozen['tokenizer_bindings'])
    require(frozen['model_directory'] == str(MODEL), 'WRONG_STARTING_CHECKPOINT')
    item = frozen['arms'][arm]; verify([item['schedule']])
    rows = lines(item['schedule']['path'])
    require(len(rows) == item['microsteps'] == frozen['fixed_final_step'] * 4 and frozen['runtime_epochs'] == 1, 'FROZEN_BUDGET_DRIFT')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(frozen['tokenizer_directory'], local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    features = []
    seen = collections.Counter()
    for i, row in enumerate(rows):
        require(row['arm'] == arm and row['split'] == 'train' and row['ordinal'] == i + 1, 'SCHEDULE_IDENTITY_DRIFT')
        require(row['slot'] != 'MC6', 'MC6_TRAINING_FORBIDDEN')
        feature, measured = tokenize(tokenizer, row['messages'], frozen['max_length'])
        require(all(row[k] == v for k, v in measured.items()), 'FROZEN_TOKEN_MEASUREMENT_DRIFT')
        seen[row['candidate_id'], row['epoch'], row['variant_id']] += 1
        features.append(feature)
    require(all(v == 1 for v in seen.values()), 'DUPLICATE_SCHEDULE_EXPOSURE')
    require(len({r['candidate_id'] for r in rows}) == item['questions'], 'QUESTION_COUNT_DRIFT')
    for cid in {r['candidate_id'] for r in rows}:
        part = [r for r in rows if r['candidate_id'] == cid]
        require({(r['epoch'], r['variant_id']) for r in part} == {(e, f'perm_{v:02d}') for e in (1, 2) for v in (0, 1)}, 'TWO_DISPLAY_TWO_EPOCH_REQUIRED')
    require(sum(len(f['input_ids']) for f in features) == item['full_input_tokens'], 'FULL_TOKEN_BUDGET_DRIFT')
    return frozen, tokenizer, rows, features


def eval_inputs(path, scope):
    frozen = read(path)
    require(frozen['status'] == 'FROZEN_BEFORE_TRAINING', 'EVALUATION_NOT_FROZEN')
    verify([frozen['protocol'], frozen['wrapper'], frozen['final_requests'], frozen['periodic_requests'], frozen['private_gold'], frozen['tokenization_checks']] + frozen['input_bindings'])
    rows = lines(frozen['final_requests' if scope == 'final' else 'periodic_requests']['path'])
    require(len(rows) == frozen['final_request_count' if scope == 'final' else 'periodic_request_count'], 'EVALUATION_DENOMINATOR_DRIFT')
    private = {r['candidate_id']: r for r in lines(frozen['private_gold']['path'])}
    for r in rows:
        require(r['split'] == 'dev', 'NON_DEV_IN_EVALUATION')
        verify([r['prompt_binding']])
    return frozen, rows, private


def rng_state(torch):
    return {'python': random.getstate(), 'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def set_rng(torch, state):
    random.setstate(state['python']); torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda']: torch.cuda.set_rng_state_all([s.cpu() for s in state['torch_cuda']])


def seed_all(torch, seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def lora_state(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if '.lora_' in name and p.requires_grad}


def load_lora(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    require(not [x for x in missing if '.lora_' in x] and not unexpected, 'ADAPTER_STATE_LOAD_MISMATCH')


def init_model(frozen, trainable, adapter_path=None):
    import torch
    from transformers import AutoModelForCausalLM
    require(torch.cuda.is_available(), 'CUDA_REQUIRED')
    require(frozen['model_directory'] == str(MODEL), 'WRONG_B1_BASE')
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), torch_dtype=torch.bfloat16,
                    local_files_only=True, trust_remote_code=True, attn_implementation='sdpa')
    n = native(); n.freeze_base_model(model)
    if trainable or adapter_path:
        t = frozen['training']
        replaced = n.inject_lora(model, set(TARGETS.split(',')), t['rank'], t['alpha'], t['dropout'])
        require(replaced == 252, 'EXPECTED_252_LORA_LINEAR_TARGETS')
        if adapter_path:
            load_lora(model, torch.load(adapter_path, map_location='cpu', weights_only=True))
    if trainable:
        model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    model.config.use_cache = False
    model.to(torch.device('cuda'))
    model.train(trainable)
    return model


def generate_answer(model, tokenizer, ids):
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    require(len(ids) + 32 <= 16384, 'GENERATION_NO_TRUNCATION_CAP')
    input_ids = torch.tensor([ids], device='cuda', dtype=torch.long)
    prompt_length = len(ids)
    class StopOnClosedBox(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return '</box>' in tokenizer.decode(input_ids[0, prompt_length:].tolist(), skip_special_tokens=True)
    with torch.no_grad():
        generated = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                    do_sample=False, num_beams=1, max_new_tokens=32, repetition_penalty=1., length_penalty=1.,
                    no_repeat_ngram_size=0, use_cache=True, pad_token_id=tokenizer.pad_token_id,
                    stopping_criteria=StoppingCriteriaList([StopOnClosedBox()]))
    generated_ids = generated[0, prompt_length:].detach().cpu().tolist()
    return tokenizer.decode(generated_ids, skip_special_tokens=True), generated_ids


def evaluate_model(model, tokenizer, evaluation_freeze, scope, output, identity, step=None):
    """Save every raw once; resume only missing requests after an engineering interruption."""
    import torch
    frozen, requests, private = eval_inputs(evaluation_freeze, scope)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    identity_digest = canonical_sha(identity)
    run_meta = {'schema_version': 'oet1_actual_answer_eval_v1', 'identity': identity, 'identity_sha256': identity_digest,
                'evaluation_freeze': binding(evaluation_freeze), 'scope': scope, 'step': step,
                'decoding': frozen['decoding'], 'request_count': len(requests), 'created_at_utc': now(), 'runtime_versions': versions()}
    meta_path = output / 'RUN_MANIFEST.json'
    if meta_path.exists():
        old = read(meta_path)
        require(old['identity_sha256'] == identity_digest and old['evaluation_freeze'] == binding(evaluation_freeze) and old['scope'] == scope and old['step'] == step and old['runtime_versions'] == versions(), 'EVAL_RESUME_IDENTITY_DRIFT')
    else: save(meta_path, run_meta)
    raw_path = output / 'RAW.jsonl'
    raw_rows = lines(raw_path) if raw_path.exists() else []
    already = {r['request_id']: r for r in raw_rows}
    require(len(already) == len(raw_rows) and set(already) <= {r['request_id'] for r in requests}, 'EVAL_RAW_DUPLICATE_OR_UNKNOWN_REQUEST')
    for r in requests:
        if r['request_id'] in already:
            require(already[r['request_id']]['prompt_sha256'] == r['prompt_sha256'], 'EVAL_EXISTING_PROMPT_DRIFT')
    rng = rng_state(torch); was_training = model.training; use_cache = model.config.use_cache
    started = time.monotonic()
    try:
        model.eval(); model.config.use_cache = True
        # Validation never changes the subsequent LoRA dropout/random stream.
        seed_all(torch, frozen['decoding']['seed'])
        for index, request in enumerate(requests):
            if request['request_id'] in already: continue
            prompt = Path(request['prompt_file']).read_text()
            ids = tokenizer(prompt, add_special_tokens=False).input_ids
            require(len(ids) == request['prompt_tokens'] and canonical_sha(ids) == request['prompt_token_ids_sha256'], 'EVAL_TOKEN_DRIFT')
            require(len(ids) + 32 <= 16384, 'EVAL_NO_TRUNCATION_CAP')
            prompt_length = len(ids)
            tick = time.monotonic()
            raw, generated_ids = generate_answer(model, tokenizer, ids)
            row = {'request_id': request['request_id'], 'prompt_sha256': request['prompt_sha256'],
                   'raw_output': raw, 'output_token_ids': generated_ids, 'prompt_tokens': prompt_length,
                   'output_tokens': len(generated_ids), 'wall_seconds': time.monotonic() - tick,
                   'completed_at_utc': now(), 'identity_sha256': identity_digest}
            append(raw_path, row); already[row['request_id']] = row
            print(json.dumps({'event': 'eval_progress', 'scope': scope, 'step': step, 'completed': len(already), 'planned': len(requests), 'wall_seconds': time.monotonic() - started}), flush=True)
    finally:
        set_rng(torch, rng); model.config.use_cache = use_cache; model.train(was_training)
    require(len(already) == len(requests), 'EVAL_INCOMPLETE')
    scored = [score_one(already[r['request_id']]['raw_output'], r, private[r['candidate_id']]) for r in requests]
    scores_path = output / 'SCORES.jsonl'
    summary_path = output / 'SUMMARY.json'
    summary = {'status': 'COMPLETED', 'scope': scope, 'optimizer_step': step, 'identity': identity,
               'evaluation_freeze': binding(evaluation_freeze), 'run_manifest': binding(meta_path), 'raw': binding(raw_path),
               'metrics': aggregate(scored), 'no_dev_checkpoint_selection': True,
               'historically_exposed_dev': True, 'test_requests': 0}
    if scores_path.exists():
        require(lines(scores_path) == scored, 'EXISTING_EVAL_SCORE_DRIFT')
    else: save_lines(scores_path, scored)
    summary['scores'] = binding(scores_path)
    if summary_path.exists(): require(read(summary_path) == summary, 'EXISTING_EVAL_SUMMARY_DRIFT')
    else: save(summary_path, summary)
    return summary


def save_checkpoint(path, model, optimizer, state, torch):
    require(shutil.disk_usage(Path(path).parent).free > 600 * 1024 * 1024, 'INSUFFICIENT_DURABLE_DISK_FOR_ATOMIC_RESUME')
    payload = {'adapter': lora_state(model), 'optimizer': optimizer.state_dict(), 'rng': rng_state(torch), **state}
    temp = Path(path).with_name(Path(path).name + '.tmp.' + str(os.getpid()))
    with temp.open('xb') as f:
        torch.save(payload, f); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def wandb_start(out, arm, config, run_id):
    try:
        import wandb
    except ImportError:
        return None, {'status': 'SDK_UNAVAILABLE_LOCAL_LOGS_ONLY', 'synced': False}
    try:
        directory = Path(out) / 'wandb_offline'; directory.mkdir(exist_ok=True)
        run = wandb.init(project='gxy-selfplay', group='oet1-20261024-exploratory', id=run_id, name=arm,
                         mode='offline', dir=str(directory), config=config, reinit=True,
                         settings=wandb.Settings(start_method='thread'))
        return run, {'status': 'OFFLINE', 'synced': False, 'run_id': run.id, 'directory': str(directory)}
    except Exception as exc:
        return None, {'status': 'OFFLINE_SDK_INITIALIZATION_FAILED_LOCAL_LOGS_PRESERVED', 'synced': False, 'error_type': type(exc).__name__, 'error': str(exc)}


def wandb_log_preserving_rng(run, values):
    import torch
    snapshot = rng_state(torch)
    try: run.log(values)
    finally: set_rng(torch, snapshot)


def train(args, frozen, tokenizer, rows, features):
    import torch
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    require(not (out / 'RESULT.json').exists(), 'TRAINING_ALREADY_COMPLETED_DO_NOT_RERUN')
    seed = frozen['training']['seed']; seed_all(torch, seed)
    model = init_model(frozen, trainable=True)
    initial_state_sha = canonical_sha({name: hashlib.sha256(t.numpy().tobytes()).hexdigest() for name, t in lora_state(model).items()})
    parameters = [p for p in model.parameters() if p.requires_grad]
    t = frozen['training']; n = native()
    optimizer = torch.optim.AdamW(parameters, lr=t['lr'], weight_decay=t['weight_decay'])
    lock_binding = binding(args.schedule_freeze)
    state = {'schema_version': 'oet1_exact_resume_v1', 'schedule_freeze': lock_binding, 'arm': args.arm,
             'step': 0, 'cursor': 0, 'evaluated_steps': [], 'initial_lora_sha256': initial_state_sha,
             'scheduler': {'type': 'native_cosine_stateless', 'next_step': 0, 'fixed_total_steps': frozen['fixed_final_step'],
                           'base_lr': t['lr'], 'min_lr_ratio': t['min_lr_ratio'], 'warmup_ratio': t['warmup_ratio']},
             'training_losses_sum': 0., 'training_microsteps_count': 0,
             'committed_optimizer_metrics': [],
             'runtime_versions': versions(), 'created_at_utc': now()}
    checkpoint = out / 'resume_state.pt'
    if args.resume:
        require(checkpoint.is_file(), 'RESUME_CHECKPOINT_REQUIRED')
        loaded = torch.load(checkpoint, map_location='cpu', weights_only=False)
        require(loaded['schedule_freeze'] == lock_binding and loaded['arm'] == args.arm and loaded['runtime_versions'] == versions(), 'RESUME_IDENTITY_MISMATCH')
        require(loaded['initial_lora_sha256'] == initial_state_sha, 'RESUME_INITIAL_LORA_DIFFERENT')
        require(loaded['cursor'] == loaded['step'] * 4 <= len(features), 'RESUME_CURSOR_INVALID')
        load_lora(model, loaded.pop('adapter')); optimizer.load_state_dict(loaded.pop('optimizer'))
        set_rng(torch, loaded.pop('rng')); state = loaded
    else:
        require(not checkpoint.exists(), 'USE_RESUME_FOR_EXISTING_CHECKPOINT')
        save_checkpoint(checkpoint, model, optimizer, state, torch)
    attempt = 'attempt_' + now().replace(':', '').replace('.', '') + '_' + uuid.uuid4().hex[:6]
    telemetry = out / (attempt + '_telemetry.jsonl')
    optimizer_log = out / (attempt + '_optimizer.jsonl')
    config = {'arm': args.arm, 'schedule_freeze': lock_binding, 'training': t, 'max_length': frozen['max_length'], 'versions': versions()}
    logger_rng = rng_state(torch)
    wandb_run, wandb_info = wandb_start(out, args.arm, config, 'oet1-' + args.arm.lower() + '-' + attempt[-6:])
    set_rng(torch, logger_rng)
    save(out / (attempt + '_STARTED.json'), {'status': 'STARTED', 'state_step': state['step'], 'cursor': state['cursor'], 'config': config, 'wandb': wandb_info, 'started_at_utc': now()})
    started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    identity = {'logical_model': 'B1_' + args.arm, 'schedule_freeze': lock_binding,
                'starting_checkpoint_bindings': frozen['model_bindings'], 'arm': args.arm}
    def periodic():
        step = state['step']
        if step in state['evaluated_steps']: return
        endpoint_identity = {**identity, 'optimizer_step': step, 'adapter_content_sha256': canonical_sha({name: hashlib.sha256(v.numpy().tobytes()).hexdigest() for name, v in lora_state(model).items()})}
        summary = evaluate_model(model, tokenizer, Path(frozen['evaluation_freeze']['path']), 'periodic', out / 'dev' / f'step_{step:04d}', endpoint_identity, step)
        state['evaluated_steps'].append(step)
        save_checkpoint(checkpoint, model, optimizer, state, torch)
        if wandb_run:
            for task, values in summary['metrics']['by_task'].items():
                wandb_log_preserving_rng(wandb_run, {f'dev/{task}/exact': values['exact'], 'optimizer_step': step})
    try:
        if state['step'] % 20 == 0 or state['step'] == frozen['fixed_final_step']: periodic()
        optimizer.zero_grad(set_to_none=True)
        step_losses = []; tokens_in_step = 0; supervised_in_step = 0
        for i in range(state['cursor'], len(features)):
            batch = n.collate([features[i]], tokenizer.pad_token_id, torch.device('cuda'))
            tick = time.monotonic(); output = model(**batch); loss = output.loss
            require(bool(torch.isfinite(loss.detach()).all()), 'NONFINITE_LOSS')
            value = float(loss.detach().cpu()); (loss / 4).backward()
            del output, loss, batch
            step_losses.append(value); tokens_in_step += len(features[i]['input_ids']); supervised_in_step += rows[i]['expected_supervised_tokens']
            event = {'attempt': attempt, 'arm': args.arm, 'microstep': i + 1, 'candidate_id': rows[i]['candidate_id'],
                     'epoch': rows[i]['epoch'], 'variant_id': rows[i]['variant_id'], 'loss': value,
                     'input_tokens': len(features[i]['input_ids']), 'supervised_tokens': rows[i]['expected_supervised_tokens'],
                     'wall_seconds': time.monotonic() - started, 'microstep_seconds': time.monotonic() - tick}
            state['training_losses_sum'] += value; state['training_microsteps_count'] += 1
            if (i + 1) % 4 == 0:
                grads = [p.grad for p in parameters if p.grad is not None]
                require(grads and all(bool(torch.isfinite(g).all()) for g in grads), 'NONFINITE_GRADIENT')
                require(any(bool(torch.count_nonzero(g)) for g in grads), 'ALL_ZERO_GRADIENT')
                lr = n.lr_at_step(state['step'], frozen['fixed_final_step'], t['lr'], t['min_lr_ratio'], t['warmup_ratio'])
                n.set_lr(optimizer, lr)
                norm = torch.nn.utils.clip_grad_norm_(parameters, t['max_grad_norm'])
                require(bool(torch.isfinite(norm)), 'NONFINITE_GRADIENT_NORM')
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                require(all(bool(torch.isfinite(p.detach()).all()) for p in parameters), 'NONFINITE_UPDATED_PARAMETERS')
                state['step'] += 1; state['cursor'] = i + 1; state['scheduler']['next_step'] = state['step']
                event.update(optimizer_step=state['step'], learning_rate=lr, gradient_norm_before_clip=float(norm.cpu()), gradient_finite=True,
                             updated_parameters_finite=True, optimizer_step_mean_loss=sum(step_losses) / len(step_losses),
                             step_input_tokens=tokens_in_step, step_supervised_tokens=supervised_in_step)
                step_losses = []; tokens_in_step = supervised_in_step = 0
                torch.cuda.synchronize()
                event.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                             current_allocated_bytes=torch.cuda.memory_allocated(), current_reserved_bytes=torch.cuda.memory_reserved())
                # The event and optimizer state commit together; a kill before the
                # append cannot lose step metrics from a resumable checkpoint.
                state['committed_optimizer_metrics'].append(dict(event))
                save_checkpoint(checkpoint, model, optimizer, state, torch)
                append(optimizer_log, event)
                if wandb_run: wandb_log_preserving_rng(wandb_run, {k: v for k, v in event.items() if isinstance(v, (int, float, bool))})
                print(json.dumps({'event': 'optimizer_step', **event}), flush=True)
            append(telemetry, event)
            if (i + 1) % 4 == 0 and (state['step'] % 20 == 0 or state['step'] == frozen['fixed_final_step']): periodic()
        require(state['step'] == frozen['fixed_final_step'] and state['cursor'] == len(features), 'FIXED_FINAL_BUDGET_NOT_REACHED')
        expected_eval = {0, frozen['fixed_final_step']} | set(range(20, frozen['fixed_final_step'] + 1, 20))
        require(set(state['evaluated_steps']) == expected_eval, 'PERIODIC_ACTUAL_ANSWER_DEV_MISSING')
        adapter_dir = out / 'adapter'
        args_native = SimpleNamespace(model=str(MODEL), tokenizer=str(TOKENIZER), lora_rank=t['rank'], lora_alpha=t['alpha'],
                        lora_dropout=t['dropout'], target_modules=TARGETS, max_length=frozen['max_length'], init_adapter_dir=None)
        if adapter_dir.exists():
            receipt = read(adapter_dir / 'SEALED_ADAPTER.json')
            verify([receipt['adapter'], receipt['config']])
            require(receipt['schedule_freeze'] == lock_binding and receipt['step'] == state['step'], 'EXISTING_FINAL_ADAPTER_IDENTITY_DRIFT')
            actual_state = torch.load(adapter_dir / 'adapter_model.pt', map_location='cpu', weights_only=True)
            desired_state = lora_state(model)
            require(set(actual_state) == set(desired_state) and all(torch.equal(actual_state[k], desired_state[k]) for k in actual_state), 'EXISTING_FINAL_ADAPTER_WEIGHT_DRIFT')
        else:
            pending = out / ('adapter.pending.' + uuid.uuid4().hex)
            n.save_adapter(model, pending, args_native, state['step'], {'fixed_final_endpoint': True, 'no_dev_selection': True, 'arm': args.arm})
            # Record final paths before a single directory rename seals the pair.
            sealed = {'status': 'SEALED', 'step': state['step'], 'schedule_freeze': lock_binding,
                      'adapter': {**binding(pending / 'adapter_model.pt'), 'path': str((adapter_dir / 'adapter_model.pt').resolve())},
                      'config': {**binding(pending / 'adapter_config.json'), 'path': str((adapter_dir / 'adapter_config.json').resolve())}}
            save(pending / 'SEALED_ADAPTER.json', sealed)
            for artifact in pending.iterdir():
                if artifact.is_file():
                    with artifact.open('rb') as stream: os.fsync(stream.fileno())
            os.rename(pending, adapter_dir)
            directory_fd = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(directory_fd)
            finally: os.close(directory_fd)
        require(len(state['committed_optimizer_metrics']) == state['step'] and
                [r['optimizer_step'] for r in state['committed_optimizer_metrics']] == list(range(1, state['step'] + 1)), 'COMMITTED_OPTIMIZER_METRICS_INCOMPLETE')
        committed_metrics = out / 'COMMITTED_OPTIMIZER_METRICS.json'
        save(committed_metrics, {'status': 'COMPLETED', 'schedule_freeze': lock_binding,
                                'metrics': state['committed_optimizer_metrics']}, replace=True)
        result = {'schema_version': 'oet1_training_result_v1', 'status': 'COMPLETED', 'arm': args.arm,
                  'logical_model': 'B1_' + args.arm, 'fixed_final_optimizer_step': frozen['fixed_final_step'],
                  'completed_optimizer_steps': state['step'], 'completed_microsteps': state['cursor'],
                  'mean_train_loss': state['training_losses_sum'] / state['training_microsteps_count'],
                  'starting_checkpoint': str(MODEL), 'initial_lora_sha256': initial_state_sha, 'new_independent_lora': True,
                  'adapter_path': str(adapter_dir / 'adapter_model.pt'), 'adapter': binding(adapter_dir / 'adapter_model.pt'),
                  'adapter_config': binding(adapter_dir / 'adapter_config.json'), 'resume_state': binding(checkpoint),
                  'sealed_adapter_receipt': binding(adapter_dir / 'SEALED_ADAPTER.json'),
                  'committed_optimizer_metrics': binding(committed_metrics),
                  'schedule_freeze': lock_binding, 'evaluation_freeze': frozen['evaluation_freeze'],
                  'dev_steps': sorted(state['evaluated_steps']), 'dev_summaries': [binding(out / 'dev' / f'step_{s:04d}' / 'SUMMARY.json') for s in sorted(state['evaluated_steps'])],
                  'telemetry_files': [binding(p) for p in sorted(out.glob('attempt_*_telemetry.jsonl'))],
                  'optimizer_metrics_files': [binding(p) for p in sorted(out.glob('attempt_*_optimizer.jsonl'))],
                  'source_truncation': False, 'input_tokens': frozen['arms'][args.arm]['full_input_tokens'],
                  'supervised_tokens': frozen['arms'][args.arm]['supervised_tokens'], 'runtime_versions': versions(),
                  'input_token_relative_difference': frozen['input_token_relative_difference'],
                  'formal_token_fairness_within_5pct': frozen['formal_token_fairness_within_5pct'],
                  'wandb': wandb_info, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'attempt_wall_seconds': time.monotonic() - started,
                  'completed_at_utc': now(), 'checkpoint_selection': 'FIXED_FINAL_NO_DEV_SELECTION'}
        save(out / 'RESULT.json', result)
        print(json.dumps({'status': 'COMPLETED', 'arm': args.arm, 'result': binding(out / 'RESULT.json')}), flush=True)
    except Exception as exc:
        save(out / (attempt + '_TERMINAL.json'), {'status': 'FAILED_ENGINEERING_OR_RUNTIME', 'error_type': type(exc).__name__,
             'error': str(exc), 'traceback': traceback.format_exc(), 'step': state['step'], 'last_atomic_checkpoint': binding(checkpoint), 'ended_at_utc': now()})
        raise
    finally:
        if wandb_run: wandb_run.finish()


def smoke(args, frozen, tokenizer, rows, features):
    import torch
    require(not (args.output / 'SMOKE_RESULT.json').exists(), 'SMOKE_ALREADY_COMPLETED')
    args.output.mkdir(parents=True, exist_ok=True)
    seed_all(torch, frozen['training']['seed']); model = init_model(frozen, trainable=True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=frozen['training']['lr'], weight_decay=0.)
    longest = max(range(len(features)), key=lambda i: len(features[i]['input_ids']))
    started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    loss = model(**native().collate([features[longest]], tokenizer.pad_token_id, torch.device('cuda'))).loss
    require(bool(torch.isfinite(loss.detach()).all()), 'SMOKE_NONFINITE_LOSS')
    loss.backward(); norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
    require(bool(torch.isfinite(norm)), 'SMOKE_NONFINITE_GRADIENT')
    optimizer.step(); torch.cuda.synchronize()
    require(all(bool(torch.isfinite(p.detach()).all()) for p in parameters), 'SMOKE_NONFINITE_UPDATED_PARAMETER')
    save(args.output / 'SMOKE_RESULT.json', {'status': 'PASS', 'mode': 'isolated_longest_full_sample_forward_backward_optimizer',
         'schedule_freeze': binding(args.schedule_freeze), 'arm': args.arm, 'microstep_index': longest,
         'full_input_tokens': len(features[longest]['input_ids']), 'loss': float(loss.detach().cpu()),
         'gradient_norm_before_clip': float(norm.cpu()), 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
         'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'wall_seconds': time.monotonic() - started,
         'model_discarded_after_smoke': True, 'formal_training_steps': 0, 'runtime_versions': versions(), 'completed_at_utc': now()})


def evaluate_endpoint(args, frozen, tokenizer):
    import torch
    require(args.role in ('B1', 'B1_GENERIC', 'B1_TARGETED'), 'UNKNOWN_EVAL_ROLE')
    seed_all(torch, frozen['training']['seed'])
    identity = {'logical_model': args.role, 'base_model_directory': str(MODEL), 'base_model_bindings': frozen['model_bindings'],
                'schedule_freeze': binding(args.schedule_freeze)}
    adapter_path = None
    if args.role != 'B1':
        require(args.training_result is not None, 'TRAINING_RESULT_REQUIRED_FOR_ENDPOINT')
        r = read(args.training_result)
        require(r['status'] == 'COMPLETED' and r['logical_model'] == args.role and r['schedule_freeze'] == binding(args.schedule_freeze), 'ENDPOINT_NOT_FIXED_FINAL_MATCH')
        require(r['completed_optimizer_steps'] == r['fixed_final_optimizer_step'] == frozen['fixed_final_step'], 'ENDPOINT_WRONG_STEP')
        verify([r['adapter'], r['adapter_config']]); adapter_path = r['adapter']['path']
        identity.update(adapter=r['adapter'], training_result=binding(args.training_result))
    model = init_model(frozen, trainable=False, adapter_path=adapter_path)
    evaluate_model(model, tokenizer, Path(frozen['evaluation_freeze']['path']), 'final', args.output, identity)
    print(json.dumps({'status': 'COMPLETED', 'model': args.role, 'summary': binding(args.output / 'SUMMARY.json')}), flush=True)


def training_fit(args, frozen, tokenizer, rows):
    """Fixed endpoint fitting diagnostic; never pooled with DEV regression."""
    import torch
    require(args.training_result is not None, 'TRAINING_RESULT_REQUIRED_FOR_FIT')
    result = read(args.training_result)
    require(result['status'] == 'COMPLETED' and result['arm'] == args.arm and
            result['schedule_freeze'] == binding(args.schedule_freeze), 'FIT_ENDPOINT_IDENTITY_MISMATCH')
    require(result['completed_optimizer_steps'] == frozen['fixed_final_step'], 'FIT_REQUIRES_FIXED_FINAL')
    verify([result['adapter'], result['adapter_config']])
    seed_all(torch, frozen['training']['seed'])
    model = init_model(frozen, trainable=False, adapter_path=result['adapter']['path'])
    model.eval(); model.config.use_cache = True
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    selected = [r for r in rows if r['epoch'] == 1]
    identity = {'logical_model': result['logical_model'], 'adapter': result['adapter'],
                'training_result': binding(args.training_result), 'schedule_freeze': binding(args.schedule_freeze)}
    manifest = {'schema_version': 'oet1_training_fit_diagnostic_v1', 'scope': 'TRAIN_FIT_SEPARATE_FROM_DEV',
                'identity': identity, 'request_count': len(selected), 'decoding': read(frozen['evaluation_freeze']['path'])['decoding']}
    manifest_path = out / 'RUN_MANIFEST.json'
    if manifest_path.exists(): require(read(manifest_path) == manifest, 'FIT_RESUME_MANIFEST_CHANGED')
    else: save(manifest_path, manifest)
    raw_path = out / 'RAW.jsonl'
    raw_rows = lines(raw_path) if raw_path.exists() else []
    completed = {r['request_id']: r for r in raw_rows}
    require(len(completed) == len(raw_rows), 'FIT_DUPLICATE_RAW')
    request_ids = {r['candidate_id'] + '::' + r['variant_id'] for r in selected}
    require(set(completed) <= request_ids, 'FIT_UNKNOWN_EXISTING_REQUEST')
    scored = []
    for i, row in enumerate(selected):
        request_id = row['candidate_id'] + '::' + row['variant_id']
        prompt = native().render_chat(tokenizer, row['messages'][:-1], True)
        ids = tokenizer(prompt, add_special_tokens=False).input_ids
        require(len(ids) == row['expected_prompt_tokens'], 'FIT_FULL_PROMPT_DRIFT')
        ph = hashlib.sha256(prompt.encode()).hexdigest()
        if request_id not in completed:
            raw, generated_ids = generate_answer(model, tokenizer, ids)
            value = {'request_id': request_id, 'prompt_sha256': ph, 'raw_output': raw,
                     'output_token_ids': generated_ids, 'completed_at_utc': now()}
            append(raw_path, value); completed[request_id] = value
        require(completed[request_id]['prompt_sha256'] == ph, 'FIT_EXISTING_PROMPT_DRIFT')
        request = {k: row[k] for k in ('candidate_id', 'root_id', 'variant_id', 'question_type', 'display_to_content')}
        request.update(request_id=request_id, cohort_id=row['historical_exposure'])
        private = {'gold_content_ids': row['gold_content_ids'], 'reporting_role': 'TRAIN_FIT_PRIMARY'}
        scored.append(score_one(completed[request_id]['raw_output'], request, private))
        print(json.dumps({'event': 'training_fit_progress', 'arm': args.arm, 'completed': i + 1, 'planned': len(selected)}), flush=True)
    scores_path = out / 'SCORES.jsonl'
    if scores_path.exists(): require(lines(scores_path) == scored, 'FIT_EXISTING_SCORES_CHANGED')
    else: save_lines(scores_path, scored)
    summary = {'status': 'COMPLETED', 'scope': 'TRAIN_FIT_SEPARATE_FROM_DEV', 'identity': identity,
               'run_manifest': binding(manifest_path), 'raw': binding(raw_path), 'scores': binding(scores_path),
               'metrics': aggregate(scored), 'new_vs_historical_layers': 'metrics.by_cohort',
               'interpretation': 'Training fit is not evidence of generalization or feedback benefit'}
    summary_path = out / 'SUMMARY.json'
    if summary_path.exists(): require(read(summary_path) == summary, 'FIT_EXISTING_SUMMARY_CHANGED')
    else: save(summary_path, summary)
    print(json.dumps({'status': 'COMPLETED', 'training_fit_summary': binding(summary_path)}), flush=True)


def compare(args):
    summaries = {role: read(path) for role, path in zip(('B1', 'B1_GENERIC', 'B1_TARGETED'), args.summaries)}
    freezes = [canonical_sha(s['evaluation_freeze']) for s in summaries.values()]
    require(len(set(freezes)) == 1 and all(s['status'] == 'COMPLETED' and s['scope'] == 'final' for s in summaries.values()), 'COMMON_FINAL_EVALUATION_REQUIRED')
    for role, summary in summaries.items():
        require(summary['identity']['logical_model'] == role, 'COMMON_MODEL_ROLE_MISMATCH')
        verify([summary['raw'], summary['scores'], summary['run_manifest'], summary['evaluation_freeze']])
    scored = {role: lines(s['scores']['path']) for role, s in summaries.items()}
    ids = [{r['request_id'] for r in values} for values in scored.values()]
    require(ids[0] == ids[1] == ids[2], 'COMMON_REQUESTS_MISMATCH')
    roots = summaries['B1']['metrics']['by_root']
    paired = {root: {role: summaries[role]['metrics']['by_root'][root] for role in summaries} for root in roots}
    deltas = {}
    for typ in ('MC2_5_PRIMARY', 'ORDERING', 'MC6_CONTROL'):
        vals = {role: s['metrics']['by_task'].get(typ) for role, s in summaries.items()}
        if all(vals.values()):
            deltas[typ] = {'GENERIC_minus_B1_exact': vals['B1_GENERIC']['exact'] - vals['B1']['exact'],
                           'TARGETED_minus_B1_exact': vals['B1_TARGETED']['exact'] - vals['B1']['exact'],
                           'TARGETED_minus_GENERIC_exact': vals['B1_TARGETED']['exact'] - vals['B1_GENERIC']['exact']}
    g, t = read(args.training_results[0]), read(args.training_results[1])
    require(g['arm'] == 'GENERIC' and t['arm'] == 'TARGETED' and g['initial_lora_sha256'] == t['initial_lora_sha256'], 'PAIRED_INITIAL_LORA_MISMATCH')
    require(g['schedule_freeze'] == t['schedule_freeze'] and g['completed_optimizer_steps'] == t['completed_optimizer_steps'] and g['supervised_tokens'] == t['supervised_tokens'], 'PAIRED_TRAINING_BUDGET_MISMATCH')
    save(args.output, {'schema_version': 'oet1_common_three_model_evaluation_v1', 'status': 'COMPLETED', 'models': list(summaries),
         'evaluation_freeze': summaries['B1']['evaluation_freeze'], 'summaries': {role: binding(path) for role, path in zip(summaries, args.summaries)},
         'training_results': {'GENERIC': binding(args.training_results[0]), 'TARGETED': binding(args.training_results[1])},
         'metrics': {role: s['metrics'] for role, s in summaries.items()}, 'paired_by_root': paired, 'descriptive_deltas': deltas,
         'initial_lora_equal': True, 'fixed_final_endpoint': True, 'historically_exposed_dev': True, 'test_requests': 0,
         'scope': 'held out from this generation/training; historically exposed DEV regression; single-seed exploratory paired comparison',
         'causal_or_unseen_root_claim': False, 'created_at_utc': now()})
    print(json.dumps({'status': 'COMPLETED', 'common_summary': binding(args.output)}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('preflight', 'smoke', 'train', 'evaluate', 'fit', 'compare'))
    p.add_argument('--schedule-freeze', type=Path, default=HERE / 'schedule/SCHEDULE_FREEZE.json')
    p.add_argument('--arm', choices=('GENERIC', 'TARGETED'), default='GENERIC')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute-gpu', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke-result', type=Path)
    p.add_argument('--role', choices=('B1', 'B1_GENERIC', 'B1_TARGETED'))
    p.add_argument('--training-result', type=Path)
    p.add_argument('--summaries', nargs=3, type=Path)
    p.add_argument('--training-results', nargs=2, type=Path)
    p.add_argument('--gpu-lock', type=Path, default=Path('/tmp/gxy-selfplay-gpu.lock'))
    args = p.parse_args()
    if args.action == 'compare': return compare(args)
    frozen, tokenizer, rows, features = verify_schedule(args.schedule_freeze, args.arm)
    eval_inputs(Path(frozen['evaluation_freeze']['path']), 'final')
    if args.action == 'preflight':
        save(args.output, {'status': 'PASS', 'mode': 'CPU_ONLY_ALL_RECORDS_REAL_TOKENIZATION', 'arm': args.arm,
             'schedule_freeze': binding(args.schedule_freeze), 'questions': frozen['arms'][args.arm]['questions'],
             'microsteps': len(rows), 'fixed_optimizer_steps': frozen['fixed_final_step'],
             'max_full_input_tokens': max(len(f['input_ids']) for f in features), 'source_truncation': False,
             'runtime_versions': versions(), 'checked_at_utc': now()})
        print(json.dumps({'status': 'PASS', 'report': binding(args.output)})); return
    require(args.execute_gpu, 'GPU_EXECUTION_REQUIRES_EXPLICIT_FLAG')
    if args.action in ('train', 'smoke') and not args.resume:
        require(not (TASK / 'FINALIZE_REQUEST').exists(), 'FINALIZE_REQUEST_STOPS_NEW_TRAINING')
    if args.action == 'train':
        require(args.smoke_result is not None, 'LONGEST_SAMPLE_SMOKE_REQUIRED')
        smoke_receipt = read(args.smoke_result)
        require(smoke_receipt['status'] == 'PASS' and smoke_receipt['schedule_freeze'] == binding(args.schedule_freeze), 'SMOKE_NOT_BOUND_TO_SCHEDULE')
        longest_both = max(v['max_full_input_tokens'] for v in frozen['arms'].values())
        require(smoke_receipt['full_input_tokens'] >= longest_both, 'SMOKE_MUST_COVER_LONGEST_SAMPLE_ACROSS_BOTH_ARMS')
    args.gpu_lock.parent.mkdir(parents=True, exist_ok=True)
    with args.gpu_lock.open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'event': 'gpu_lock_acquired', 'path': str(args.gpu_lock), 'pid': os.getpid(), 'at_utc': now()}), flush=True)
        try:
            if args.action == 'train': train(args, frozen, tokenizer, rows, features)
            elif args.action == 'smoke': smoke(args, frozen, tokenizer, rows, features)
            elif args.action == 'fit': training_fit(args, frozen, tokenizer, rows)
            else: evaluate_endpoint(args, frozen, tokenizer)
        except Exception as exc:
            args.output.mkdir(parents=True, exist_ok=True)
            save(args.output / ('EXECUTION_FAILURE_' + uuid.uuid4().hex + '.json'),
                 {'status': 'FAILED_ENGINEERING_OR_RUNTIME', 'action': args.action, 'arm': args.arm,
                  'schedule_freeze': binding(args.schedule_freeze), 'error_type': type(exc).__name__,
                  'error': str(exc), 'traceback': traceback.format_exc(), 'ended_at_utc': now()})
            raise

if __name__ == '__main__': main()
