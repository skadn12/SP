"""Transparent instrumentation launcher around the existing SFT trainer."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch


gradient_records: list[dict[str, object]] = []
backward_records: list[dict[str, object]] = []
original_clip = torch.nn.utils.clip_grad_norm_
original_backward = torch.Tensor.backward


def audited_clip(parameters, max_norm, *args, **kwargs):
    params = list(parameters)
    grads = [param.grad.detach() for param in params if param.grad is not None]
    finite = all(bool(torch.isfinite(grad).all().item()) for grad in grads)
    nonzero = sum(int(torch.count_nonzero(grad).item()) for grad in grads)
    norm = float(torch.sqrt(sum(torch.sum(grad.float() ** 2) for grad in grads)).item()) if grads else 0.0
    if not finite or nonzero == 0 or not (norm > 0.0):
        raise RuntimeError(f"gradient_audit_failed:finite={finite}:nonzero={nonzero}:norm={norm}")
    result = original_clip(params, max_norm, *args, **kwargs)
    gradient_records.append({
        "optimizer_step": len(gradient_records) + 1,
        "gradient_finite_before_clip": finite,
        "gradient_nonzero_elements": nonzero,
        "gradient_l2_norm_before_clip": norm,
        "clipped_norm": float(result.detach().cpu()) if torch.is_tensor(result) else float(result),
    })
    return result


def audited_backward(self, *args, **kwargs):
    finite = bool(torch.isfinite(self.detach()).all().item())
    value = float(self.detach().cpu())
    if not finite:
        raise RuntimeError(f"loss_not_finite:{value}")
    backward_records.append({"micro_step": len(backward_records) + 1, "loss_finite": finite, "scaled_loss": value})
    return original_backward(self, *args, **kwargs)


def main() -> int:
    import train_qwen3_lora_sft as trainer

    torch.nn.utils.clip_grad_norm_ = audited_clip
    torch.Tensor.backward = audited_backward
    try:
        return int(trainer.main())
    finally:
        path = Path(os.environ["SMOKE_GRADIENT_AUDIT_PATH"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "gradient_records": gradient_records,
            "backward_records": backward_records,
            "optimizer_steps": len(gradient_records),
            "backward_calls": len(backward_records),
            "all_losses_finite": all(bool(row["loss_finite"]) for row in backward_records),
            "all_gradients_finite_nonzero": all(bool(row["gradient_finite_before_clip"]) and int(row["gradient_nonzero_elements"]) > 0 for row in gradient_records),
        }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
