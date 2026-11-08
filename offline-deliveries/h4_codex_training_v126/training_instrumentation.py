"""Add per-backward timing/loss/memory telemetry to the unchanged native runner."""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path

NATIVE = Path('/root/autodl-tmp/gxy-selfplay/current_method_github')
sys.path.insert(0, str(NATIVE))
runner = importlib.import_module('scripts.current_method.restart_n5_spell.h4_round0_b0_training_smoke_v1.train_smoke_runner')
import torch


def main():
    telemetry_path = Path(os.environ['V126_TELEMETRY_PATH'])
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    if telemetry_path.exists():
        raise RuntimeError('REFUSE_TO_OVERWRITE_TELEMETRY')
    grad_accum = int(sys.argv[sys.argv.index('--grad-accum') + 1])
    backward = runner.original_backward
    clip = runner.original_clip
    started = time.monotonic()
    last_backward = [None]
    losses = []
    steps = [0]

    def emit(row):
        with telemetry_path.open('a') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    def live_backward(loss, *args, **kwargs):
        begin = time.monotonic()
        result = backward(loss, *args, **kwargs)
        torch.cuda.synchronize()
        end = time.monotonic()
        raw_loss = float(loss.detach().cpu()) * grad_accum
        losses.append(raw_loss)
        emit({'event': 'backward_complete', 'micro_step': len(losses),
              'unscaled_loss': raw_loss, 'elapsed_seconds': end - started,
              'backward_seconds': end - begin,
              'since_previous_backward_seconds': None if last_backward[0] is None else end - last_backward[0],
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
        last_backward[0] = end
        return result

    def live_clip(parameters, max_norm, *args, **kwargs):
        result = clip(parameters, max_norm, *args, **kwargs)
        steps[0] += 1
        emit({'event': 'gradient_clip_complete', 'step': steps[0], 'micro_step': len(losses),
              'gradient_norm_before_clip': float(result.detach().cpu()),
              'mean_loss_last_microbatches': sum(losses[-grad_accum:]) / len(losses[-grad_accum:]),
              'elapsed_seconds': time.monotonic() - started})
        return result

    # The native audit still checks finite losses, finite/nonzero gradients and
    # writes its original gradient_audit.json. These callbacks do not alter loss,
    # gradient scaling, clipping, optimizer, data order, or checkpoint saving.
    runner.original_backward = live_backward
    runner.original_clip = live_clip
    return runner.main()


if __name__ == '__main__':
    raise SystemExit(main())
