#!/usr/bin/env bash
set -euo pipefail

BASELINE_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR=/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_b0_baseline135_training_20261108_run01
DATA=/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/training_set_rebuild_20261105_F_supplement_run01/repaired_training_export_v1/TRAIN.jsonl
MODEL=/root/autodl-tmp/Qwen3-4B
TRAINER=/root/autodl-tmp/gxy-selfplay/current_method_github/train_qwen3_lora_sft.py
INIT=/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_b0_formal_training_v1_live_2/clean_adapter

test "$(sha256sum "$DATA" | awk '{print $1}')" = 46acd3d1725c8af4fea264a5cad2dbdc5b36e604a75ff04f5ea2db1b4b2b50e1
test "$(sha256sum "$TRAINER" | awk '{print $1}')" = 918a45232e4287a26f46a5b4eb48175476a82136f5c3e202c94bf0d8cecb87dc
test "$(sha256sum "$INIT/adapter_model.pt" | awk '{print $1}')" = 92b5706e2e2e4201e1aeff91bad8137e9662fc95ce96e9b71a2475e9097a31bf
test ! -e "$RUN_DIR"
mkdir -p "$RUN_DIR/tmp" "$RUN_DIR/cache" "$RUN_DIR/adapter"

if pgrep -af 'train_qwen3_lora_sft.py|vllm|run_formal.py' >/dev/null; then
  echo 'refusing launch: another training or Solver process is active' >&2
  exit 2
fi

export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export TMPDIR="$RUN_DIR/tmp" XDG_CACHE_HOME="$RUN_DIR/cache" HF_HOME="$RUN_DIR/cache/huggingface"
export TRANSFORMERS_CACHE="$RUN_DIR/cache/huggingface/transformers" TORCH_HOME="$RUN_DIR/cache/torch"
date -u +%FT%TZ > "$RUN_DIR/LAUNCH_STARTED_UTC"
exec /root/miniconda3/bin/python "$TRAINER" \
  --model "$MODEL" --tokenizer "$MODEL" --train-file "$DATA" \
  --output-dir "$RUN_DIR/adapter" --init-adapter-dir "$INIT" \
  --max-length 12288 --epochs 2.0 --batch-size 1 --grad-accum 4 \
  --lr 0.0001 --min-lr-ratio 0.1 --warmup-ratio 0.05 --weight-decay 0.0 \
  --max-grad-norm 1.0 --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05 \
  --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --seed 20261018 --eval-steps 25 --save-steps 50 \
  2>&1 | tee "$RUN_DIR/train.log"
