#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/gxy-selfplay

export PYTHONPATH=scripts
export VLLM_USE_FLASHINFER_SAMPLER=0

STATE=current_method_runs/r3_phase1_single_choice_difficulty_state.json
RUN=current_method_runs/current_method_r3_phase1_single_choice_gate200

if [ -d "$RUN" ]; then
  echo "[ERROR] RUN already exists: $RUN"
  echo "If you really want to overwrite it, run:"
  echo "  rm -rf $RUN"
  exit 1
fi

mkdir -p current_method_runs

.venv-local4b/bin/python scripts/run_auto_graph_expansion_round.py \
  --run-name current_method_r3_phase1_single_choice_gate200 \
  --run-root $RUN \
  --round-id current_method_r3_long_append_closedloop \
  --round-number 3 \
  --parent-round-id r2 \
  --parent-round-number 2 \
  --parent-mode round_dataset \
  --parent-dir eval_sets/graph_expansion_challenger_v1/r2_auto_full_sample0_5_complete \
  --fixed-solver-name solver_r2_ctx9216_full \
  --generation-backend native_current_method \
  --native-challenger-mode graph_conditioned_api_dialogue \
  --native-curriculum-mode r3_long_easy_v2 \
  --closed-loop-difficulty \
  --difficulty-state-path $STATE \
  --generation-concurrency 4 \
  --max-attempts-per-sample 4 \
  --max-llm-retries 1 \
  --max-api-failures 200 \
  --n-train 200 \
  --run-k5 \
  --train-only \
  --frontier-splits train \
  --forbid-frontier-from-dev-heldout \
  --solver-model outputs/qwen3_4b_lora_r2_frontier_plus_clean_replay_s0_s2_from_r1_e1_ctx9216_full_merged \
  --solver-adapter outputs/qwen3_4b_lora_r2_frontier_plus_clean_replay_s0_s2_from_r1_e1_ctx9216_full \
  --parent-context-tail-turns 12 \
  --parent-context-relevant-turns 24 \
  --dialogue-api-parent-token-budget 2048 \
  --graph-expansion-max-tokens 2048 \
  --challenger-max-tokens 2048 \
  --llm-timeout 120 \
  --checkpoint-every 1
