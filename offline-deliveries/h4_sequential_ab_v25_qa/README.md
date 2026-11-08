# v25 sequential question QA adapter

This independent adapter consumes an audited v25 cumulative root context (`frozen base -> inherited accepted A -> accepted B`) and reuses the v17 question contract and v15 full-only evaluation/selection policy. Historical v24/v24_qa code and pools are unchanged.

The upstream run must already contain `MERGE_MANIFEST.json`, `FINAL_AUDIT.json` (`PASS`, complete or explicit shortfall), `MANIFEST.json`, and `INHERITED_A.json`. The adapter checks inherited A's original/local bytes, provenance hash, unchanged history prefix, B-only new suffix, and consistent scene counts. These are artifact-binding checks; they add no semantic novelty or difficulty gate. An audited explicit shortfall, including zero new B, remains representable.

Run the commands below from this directory after the upstream expansion finishes. Replace `<completed-v25-run>` with that expansion's run directory. The generation and evaluation directories must be new. No command below has been executed against a real v25 run as part of adapter preparation.

```bash
cd /root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_sequential_ab_v25_qa
QA_PY=/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python

"$QA_PY" generate.py prepare \
  --run runs/root104814_slots2_01_generation \
  --merge-manifest ../h4_sequential_ab_v25/runs/<completed-v25-run>/MERGE_MANIFEST.json \
  --roots cc-train-episode-104814 --slots 2
"$QA_PY" generate.py verify --run runs/root104814_slots2_01_generation
"$QA_PY" generate.py generate --run runs/root104814_slots2_01_generation

"$QA_PY" evaluate_and_pool.py preflight \
  --generation runs/root104814_slots2_01_generation \
  --run runs/root104814_slots2_01_evaluation
"$QA_PY" evaluate_and_pool.py verify --run runs/root104814_slots2_01_evaluation
"$QA_PY" evaluate_and_pool.py live --run runs/root104814_slots2_01_evaluation
"$QA_PY" evaluate_and_pool.py finalize --run runs/root104814_slots2_01_evaluation
```

`generate` calls Qwen for two candidates of each question type: four candidate slots, normally four API calls (two generation calls plus two independent ordering reviews), at most six calls including mechanical repairs. There is a 35-second interval after every completed API call before the next. Candidate slots are not a selected quota.

The default prior inventory remains the frozen 30-question pool at `h4_root_merged_context_v17/runs/merged_latest_02/training_pool/TRAINING_POOL.jsonl`. Deduplication rejects only exact entire six-statement sets; supported paraphrases and partial overlap remain allowed. A and B appear in one full source context. Current Solver feedback does not enter generation.

Evaluation uses the exact audited `M_clean_policy` checkpoint at `/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_common_dev_eval_v1_live_3/clean_merged_model`, with the established tokenizer at `/root/autodl-tmp/Qwen3-4B`. The proven `.venv-local4b` interpreter is required. Two multi-select questions require 36 full answers (6 permutations × 3 answers each); two ordering questions require 24 (12 × 1), totaling at most 60. Selection remains `complete_valid_full_set and 0 < full_partial_mean < 1`. No ground calls or training occur. The new immutable pool is written to `runs/root104814_slots2_01_evaluation/training_pool/TRAINING_POOL.jsonl`, preserving original question text, full source, complete Gold, and original ordering Gold.

Offline validation:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest -q test_generate_adapter test_pipeline
```

All 18 tests pass: 15 inherited adapter/selection tests, including full generation→evaluation→pool control flow with transport/model boundaries mocked, and three inherited-A provenance/count/prefix/shortfall checks. The integration test blocks sockets and uses only temporary synthetic artifacts.

Read-only readiness check (2026-10-10): `.venv-local4b` imports PyTorch 2.11.0+cu130, Transformers 5.9.0 and vLLM 0.21.0 successfully. The fixed model audit and all 18 checkpoint/tokenizer file bindings passed exact hashing in 5.67 seconds; no CUDA context or model was loaded. GPU query showed NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB total, 0 MiB used. GPU availability can change before execution.

For this one-root/four-candidate QA probe, allow approximately 5–10 minutes once v25's final audit passes, conditional on ordinary Qwen latency and few repairs. The 35-second pacing alone contributes 105 seconds for four calls or 175 seconds for six. The historical v17 run used this exact checkpoint for 510 full answers in approximately 147 seconds from STARTED to COMPLETE filesystem timestamps (127 seconds between first/last answer writes). A 60-answer run is therefore expected to take roughly 35–90 seconds for model startup and inference, plus hash checks and pool assembly. These are estimates from prior artifacts, not timings from a new v25 run; the final number of selected questions may be zero through four.
