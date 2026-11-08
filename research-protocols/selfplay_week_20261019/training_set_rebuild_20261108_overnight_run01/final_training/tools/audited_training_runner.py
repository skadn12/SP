#!/usr/bin/env python3
"""Instrument the immutable native trainer, W&B, and fixed validation callback.

The optimizer, data order, loss, native checkpoint writes and model recipe are
unchanged. Additional checkpoint files are audit snapshots. Evaluation saves and
restores model mode plus Python/Torch/CUDA RNG state. Formal runs require online
W&B and step-0/every-20/final evaluation; a disposable smoke is explicitly separate.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import traceback

from checkpoint_health import validate_checkpoint
from training_contract import (INIT, INIT_SHA, TRAINER, TRAINER_SHA, atomic_json,
    bind, final_manifest, load, now, read, sha, verify_binding, write_once)


class Audit:
    def __init__(self, run_dir, phase, expected_backward, expected_updates, validation, callback, manifest_path):
        self.run_dir = Path(run_dir).resolve()
        self.phase = phase
        self.expected_backward = expected_backward
        self.expected_updates = expected_updates
        self.validation = validation
        self.callback = callback
        self.manifest_path = manifest_path
        self.backward_calls = self.optimizer_steps = 0
        self.gradient_records = []
        self.checkpoint_records = []
        self.evaluations = []
        self.failures = []
        self.pending_gradient = None
        self.tokenizer = self.model = self.native_args = None
        self.wandb = None
        self.wandb_sdk = None
        self.native = None
        self.window_loss_sum = 0.0
        self.window_loss_count = 0
        self.event_path = self.run_dir/'gradient_audit.jsonl'
        self.summary_path = self.run_dir/'gradient_audit.json'

    def event(self, kind, **fields):
        record = dict(event=kind, recorded_at_utc=now(), **fields)
        with self.event_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        return record

    def log(self, values):
        if self.wandb is not None:
            # Monitoring must not consume the native trainer's shuffle/dropout RNG.
            import torch
            py_rng=random.getstate()
            torch_rng=torch.random.get_rng_state()
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            try:
                self.wandb.log(values)
            finally:
                random.setstate(py_rng)
                torch.random.set_rng_state(torch_rng)
                if cuda_rng is not None: torch.cuda.set_rng_state_all(cuda_rng)

    def start_wandb(self):
        if self.phase == 'smoke':
            return
        assert os.environ.get('WANDB_MODE') == 'online', 'WANDB_ONLINE_REQUIRED'
        import wandb
        assert bool(wandb.api.api_key), 'WANDB_ONLINE_AUTHENTICATION_REQUIRED'
        run = wandb.init(project=os.environ.get('WANDB_PROJECT', 'gxy-selfplay'),
                         entity=os.environ.get('WANDB_ENTITY') or None,
                         name=self.run_dir.name, id=self.run_dir.name, resume='never',
                         dir=str(self.run_dir/'wandb'), mode='online', sync_tensorboard=False,
                         settings=wandb.Settings(init_timeout=60),
                         config=dict(final_manifest=bind(self.manifest_path),
                                     validation_manifest=bind(self.validation), callback=bind(self.callback),
                                     native_trainer_sha256=TRAINER_SHA, initial_adapter_sha256=INIT_SHA,
                                     expected_backward_calls=self.expected_backward,
                                     expected_optimizer_updates=self.expected_updates,
                                     tensorboard_enabled=False, validation_every_optimizer_steps=20))
        assert run is not None and run.settings.mode == 'online' and run.url, 'WANDB_ONLINE_INIT_FAILED'
        self.wandb = run
        self.wandb_sdk = wandb
        run.define_metric('train/micro_step')
        run.define_metric('train/*', step_metric='train/micro_step')
        run.define_metric('optimizer/step')
        run.define_metric('optimizer/*', step_metric='optimizer/step')
        run.define_metric('validation/optimizer_step')
        run.define_metric('validation/*', step_metric='validation/optimizer_step')
        write_once(self.run_dir/'WANDB_RUN.json', dict(status='ONLINE', run_id=run.id, url=run.url,
                   project=run.project, entity=run.entity, tensorboard_enabled=False, created_at_utc=now()))

    def evaluate(self, step, reason):
        if self.phase != 'formal':
            return
        existing = next((r for r in self.evaluations if r['step'] == step), None)
        if existing:
            self.event('evaluation_reused', step=step, reason=reason, evaluation=existing['result_binding'])
            return
        import torch
        assert self.model is not None and self.tokenizer is not None
        module = load('fixed_training_validation_callback', self.callback)
        callback_before, manifest_before = bind(self.callback), bind(self.validation)
        prior_mode = self.model.training
        py_rng = random.getstate()
        torch_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        cuda_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        config_cache = getattr(self.model.config, 'use_cache', None) if hasattr(self.model, 'config') else None
        def parameter_versions():
            return {name:(id(p),p._version,id(p.grad) if p.grad is not None else None,
                          p.grad._version if p.grad is not None else None)
                    for name,p in self.model.named_parameters()}
        versions=parameter_versions()
        output = self.run_dir/'validation'/f'step-{step:06d}'
        assert not output.exists(), 'VALIDATION_OUTPUT_ALREADY_EXISTS'
        output.mkdir(parents=True)
        self.event('evaluation_started', step=step, reason=reason)
        try:
            self.model.eval()
            with torch.no_grad():
                result = module.evaluate(model=self.model, tokenizer=self.tokenizer, step=step,
                                         output_dir=output, validation_manifest=Path(self.validation))
            assert isinstance(result, dict) and result.get('raw_cot_saved') is True, 'VALIDATION_RAW_COT_REQUIRED'
            assert isinstance(result.get('evaluated_records'), int) and result['evaluated_records'] > 0
            metrics = result.get('metrics')
            assert isinstance(metrics, dict) and metrics, 'VALIDATION_METRICS_REQUIRED'
            assert any('loss' in key.lower() for key in metrics), 'VALIDATION_LOSS_REQUIRED'
            assert any(any(word in key.lower() for word in ('exact', 'accuracy', 'mean_j', 'task_score')) for key in metrics), 'VALIDATION_TASK_METRIC_REQUIRED'
            for key, value in metrics.items():
                assert isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), ('NONFINITE_VALIDATION_METRIC', key)
            artifacts = result.get('artifacts')
            assert isinstance(artifacts, list) and artifacts, 'VALIDATION_ARTIFACTS_REQUIRED'
            for artifact in artifacts:
                artifact_path = Path(artifact['path']).resolve()
                assert artifact_path.is_relative_to(output.resolve()), 'VALIDATION_ARTIFACT_OUTSIDE_OUTPUT'
                verify_binding(artifact)
            assert bind(self.callback) == callback_before and bind(self.validation) == manifest_before, 'VALIDATION_CONTRACT_CHANGED'
            assert parameter_versions()==versions, 'VALIDATION_MUTATED_MODEL_PARAMETERS_OR_TRAINING_GRADIENTS'
            envelope = dict(status='PASS', step=step, reason=reason, recorded_at_utc=now(),
                            callback_binding=callback_before, validation_manifest_binding=manifest_before, **result)
            write_once(output/'VALIDATION_RESULT.json', envelope)
            record = dict(step=step, reason=reason, result_binding=bind(output/'VALIDATION_RESULT.json'), metrics=metrics)
            self.evaluations.append(record)
            self.event('evaluation_completed', **record)
            self.log({'validation/optimizer_step':step,
                      **{'validation/'+k:v for k,v in metrics.items()},
                      'validation/evaluated_records':result['evaluated_records']})
            artifact = self.wandb_sdk.Artifact(name=f'{self.run_dir.name}-validation-step-{step:06d}',
                type='fixed-validation', metadata=dict(step=step, metrics=metrics,
                    validation_manifest_sha256=manifest_before['sha256'], raw_cot_preserved=True))
            for item in artifacts:
                artifact.add_file(item['path'], name=Path(item['path']).relative_to(output).as_posix())
            artifact.add_file(str(output/'VALIDATION_RESULT.json'), name='VALIDATION_RESULT.json')
            self.wandb.log_artifact(artifact)
        except BaseException as exc:
            write_once(output/'VALIDATION_FAILURE.json', dict(status='FAILED', step=step, reason=reason,
                       error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc()))
            raise
        finally:
            self.model.train(prior_mode)
            if hasattr(self.model,'config') and hasattr(self.model.config,'use_cache'):
                self.model.config.use_cache=config_cache
            random.setstate(py_rng)
            torch.random.set_rng_state(torch_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            self.event('evaluation_state_restored', step=step, model_training=prior_mode,
                       rng_state_restored=True, cuda_peak_bytes=cuda_peak)

    def install(self, native):
        import torch
        self.native = native
        original_backward = torch.Tensor.backward
        original_clip = torch.nn.utils.clip_grad_norm_
        original_step = torch.optim.AdamW.step
        original_load = native.load_initial_adapter
        original_tokenize = native.tokenize_record
        original_count = native.trainable_parameters
        original_save = native.save_adapter

        def audited_tokenize(tokenizer, record, max_length):
            self.tokenizer = tokenizer
            messages = record['messages']
            assert [m['role'] for m in messages] == ['system','user','assistant']
            full = tokenizer(native.render_chat(tokenizer, messages, False), add_special_tokens=False).input_ids
            if tokenizer.eos_token_id is not None and (not full or full[-1] != tokenizer.eos_token_id): full.append(tokenizer.eos_token_id)
            prompt = tokenizer(native.render_chat(tokenizer, messages[:-1], True), add_special_tokens=False).input_ids
            assert len(full) <= max_length and full[:len(prompt)] == prompt and len(full)>len(prompt), 'RUNTIME_TOKENIZATION_GATE_FAILED'
            feature = original_tokenize(tokenizer, record, max_length)
            assert feature['input_ids'] == full
            assert feature['labels'] == [-100]*len(prompt)+full[len(prompt):], 'RUNTIME_ASSISTANT_MASK_DRIFT'
            return feature

        def audited_load(model, adapter_dir, args):
            assert Path(adapter_dir).resolve() == INIT.resolve() and sha(INIT/'adapter_model.pt') == INIT_SHA
            self.model, self.native_args = model, args
            return original_load(model, adapter_dir, args)

        def audited_count(model):
            result = original_count(model)
            self.model = model
            self.evaluate(0, 'before_training')
            return result

        def audited_backward(tensor, *args, **kwargs):
            assert tensor.numel() == 1, 'LOSS_MUST_BE_SCALAR'
            value = float(tensor.detach().cpu())
            assert math.isfinite(value), 'NONFINITE_LOSS'
            result = original_backward(tensor, *args, **kwargs)
            self.backward_calls += 1
            self.window_loss_sum += value*self.native_args.grad_accum
            self.window_loss_count += 1
            self.event('backward', micro_step=self.backward_calls, scaled_loss=value, loss_finite=True)
            self.log({'train/micro_step':self.backward_calls, 'train/scaled_loss':value,
                      'train/loss':value*self.native_args.grad_accum})
            return result

        def audited_clip(parameters, max_norm, *args, **kwargs):
            params = list(parameters)
            gradients = [p.grad.detach() for p in params if p.grad is not None]
            finite = bool(gradients) and all(bool(torch.isfinite(g).all().item()) for g in gradients)
            nonzero = sum(int(torch.count_nonzero(g).item()) for g in gradients)
            norm = float(torch.sqrt(sum(torch.sum(g.float()**2) for g in gradients)).item()) if gradients else 0.
            assert finite and nonzero>0 and math.isfinite(norm) and norm>0, 'GRADIENT_NOT_FINITE_NONZERO'
            result = original_clip(params, max_norm, *args, **kwargs)
            clipped = float(result.detach().cpu()) if torch.is_tensor(result) else float(result)
            assert math.isfinite(clipped) and all(bool(torch.isfinite(g).all().item()) for g in gradients), 'CLIPPED_GRADIENT_NOT_FINITE'
            self.pending_gradient = dict(optimizer_step=self.optimizer_steps+1, micro_step=self.backward_calls,
                gradient_finite_before_clip=finite, gradient_nonzero_elements=nonzero,
                gradient_l2_norm_before_clip=norm, clipped_norm=clipped)
            self.gradient_records.append(self.pending_gradient)
            self.event('gradient_before_step', **self.pending_gradient)
            return result

        def audited_save(model, output_dir, args, step, metrics):
            result = original_save(model, output_dir, args, step, metrics)
            health = validate_checkpoint(output_dir, step)
            self.checkpoint_records.append(health)
            write_once(Path(output_dir)/'CHECKPOINT_HEALTH.json', health)
            self.event('checkpoint_saved_and_verified', checkpoint=str(output_dir), step=step,
                       health_binding=bind(Path(output_dir)/'CHECKPOINT_HEALTH.json'))
            return result

        def audited_step(optimizer, *args, **kwargs):
            assert self.pending_gradient is not None and self.pending_gradient['optimizer_step']==self.optimizer_steps+1, 'OPTIMIZER_STEP_WITHOUT_GRADIENT_AUDIT'
            result = original_step(optimizer, *args, **kwargs)
            self.optimizer_steps += 1
            finite = all(bool(torch.isfinite(p.detach()).all().item()) for group in optimizer.param_groups for p in group['params'])
            assert finite, 'NONFINITE_PARAMETERS_AFTER_OPTIMIZER_STEP'
            record = self.pending_gradient
            self.pending_gradient = None
            self.event('optimizer_step', optimizer_step=self.optimizer_steps, micro_step=self.backward_calls,
                       learning_rates=[g['lr'] for g in optimizer.param_groups], parameters_finite=True)
            self.log({'optimizer/step':self.optimizer_steps, 'optimizer/lr':optimizer.param_groups[0]['lr'],
                      'optimizer/gradient_l2_norm':record['gradient_l2_norm_before_clip'],
                      'optimizer/gradient_nonzero_elements':record['gradient_nonzero_elements'],
                      'optimizer/parameters_finite':1, 'optimizer/micro_step':self.backward_calls})
            if self.optimizer_steps == 1 or self.optimizer_steps % 20 == 0 or self.optimizer_steps == self.expected_updates:
                window_mean = self.window_loss_sum/self.window_loss_count
                self.event('training_loss_window', optimizer_step=self.optimizer_steps,
                    micro_step=self.backward_calls, mean_unscaled_loss=window_mean,
                    backward_calls_in_window=self.window_loss_count)
                self.log({'train/micro_step':self.backward_calls, 'train/window_loss':window_mean,
                          'train/window_backward_calls':self.window_loss_count})
                self.window_loss_sum = 0.0
                self.window_loss_count = 0
            if self.optimizer_steps == 1:
                path = self.run_dir/'audit_checkpoints'/'step-000001'
                audited_save(self.model, path, self.native_args, 1, {'audit_only_snapshot':True})
                write_once(self.run_dir/'TRAINING_HEALTHY.json', dict(status='HEALTHY', recorded_at_utc=now(),
                    health_scope='first actual optimizer update and separately validated audit checkpoint',
                    optimizer_steps=1, backward_calls=self.backward_calls, all_losses_so_far_finite=True,
                    all_gradients_so_far_finite_nonzero=True, checkpoint_health=bind(path/'CHECKPOINT_HEALTH.json'),
                    wandb_online=self.wandb is not None, tensorboard_enabled=False,
                    overnight_cancelled=True, final_training_completion_not_implied=True))
            if self.optimizer_steps % 20 == 0:
                self.evaluate(self.optimizer_steps, 'fixed_20_step_interval')
            return result

        torch.Tensor.backward = audited_backward
        torch.nn.utils.clip_grad_norm_ = audited_clip
        torch.optim.AdamW.step = audited_step
        native.tokenize_record = audited_tokenize
        native.load_initial_adapter = audited_load
        native.trainable_parameters = audited_count
        native.save_adapter = audited_save

    def finish(self, success):
        report = dict(schema='full_training_gradient_audit_v2', phase=self.phase,
            status='PASS' if success else 'FAILED', recorded_at_utc=now(),
            backward_calls=self.backward_calls, optimizer_steps=self.optimizer_steps,
            expected_backward_calls=self.expected_backward, expected_optimizer_steps=self.expected_updates,
            all_losses_finite=not self.failures and self.backward_calls>0,
            all_gradients_finite_nonzero=bool(self.gradient_records) and all(r['gradient_finite_before_clip'] and r['gradient_nonzero_elements']>0 for r in self.gradient_records),
            gradient_records=self.gradient_records, checkpoint_records=self.checkpoint_records,
            evaluations=self.evaluations, failures=self.failures,
            event_ledger=bind(self.event_path) if self.event_path.exists() else None,
            native_trainer_sha256=sha(TRAINER), native_trainer_unchanged=sha(TRAINER)==TRAINER_SHA,
            tensorboard_enabled=False, wandb_online=self.wandb is not None)
        atomic_json(self.summary_path, report)
        if self.wandb is not None:
            self.wandb.summary['training_audit_status'] = report['status']
            self.wandb.summary['actual_optimizer_steps'] = self.optimizer_steps
            self.wandb.summary['actual_backward_calls'] = self.backward_calls
            self.wandb.finish(exit_code=0 if success else 1)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--phase', choices=('formal','smoke'), required=True)
    parser.add_argument('--expected-backward', type=int, required=True)
    parser.add_argument('--expected-updates', type=int, required=True)
    parser.add_argument('--validation-manifest', type=Path)
    parser.add_argument('--validation-callback', type=Path)
    parser.add_argument('--final-manifest', type=Path)
    args, native_args = parser.parse_known_args()
    if native_args and native_args[0]=='--': native_args=native_args[1:]
    assert sha(TRAINER)==TRAINER_SHA
    if args.phase == 'formal':
        manifest = final_manifest()
        assert args.final_manifest is not None and args.validation_manifest and args.validation_callback
        assert args.expected_backward==manifest['expected_backward_calls'] and args.expected_updates==manifest['expected_optimizer_updates']
        assert args.validation_manifest.is_file() and args.validation_callback.is_file()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    audit = Audit(args.run_dir, args.phase, args.expected_backward, args.expected_updates,
                  args.validation_manifest, args.validation_callback, args.final_manifest)
    success=False
    try:
        audit.start_wandb()
        native=load('audited_native_training_immutable', TRAINER)
        audit.install(native)
        sys.argv=[str(TRAINER)]+native_args
        code=native.main()
        assert code==0
        assert audit.backward_calls==args.expected_backward and audit.optimizer_steps==args.expected_updates, 'ACTUAL_TRAINING_BUDGET_MISMATCH'
        audit.evaluate(audit.optimizer_steps, 'final_fixed_endpoint')
        if args.phase=='formal':
            expected=[0]+list(range(20,args.expected_updates+1,20))
            if args.expected_updates%20: expected.append(args.expected_updates)
            assert [r['step'] for r in audit.evaluations]==expected, 'VALIDATION_SCHEDULE_INCOMPLETE'
        validate_checkpoint(args.run_dir/'adapter', args.expected_updates)
        success=True
        return 0
    except BaseException as exc:
        audit.failures.append(dict(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc()))
        raise
    finally:
        report=audit.finish(success)
        if success:
            write_once(args.run_dir/'TRAINING_COMPLETED.json', dict(status='COMPLETED', completed_at_utc=now(),
                       audit_binding=bind(audit.summary_path), actual_optimizer_steps=audit.optimizer_steps,
                       actual_backward_calls=audit.backward_calls, evaluations=[r['step'] for r in audit.evaluations],
                       checkpoint=bind(args.run_dir/'adapter'/'adapter_model.pt'), overnight_cancelled=True))


if __name__=='__main__':
    raise SystemExit(main())
