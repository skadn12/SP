# Round-0 B0 Formal Training

This route serially trains `M_clean_policy` and `M_inclusive_policy` from the
immutable `h4_round0_b0_matched_training_v1_live_2` JSONL files. It reuses the
verified custom `train_qwen3_lora_sft.py` recipe, uses `--max-length 8192`, and
does not run Dev/Test evaluation, Solver, external APIs, or semantic auditing.

The output run rejects an existing directory and records input, base-model,
recipe, log, gradient, adapter, and report hashes. With 386 records, two
epochs, batch size 1, and gradient accumulation 4, the trainer performs 772
backward calls and 193 optimizer updates; the trainer's informational
`total_update_steps` field reports 194 because it rounds per-epoch steps before
multiplying, so the final `global_step` is the authoritative update count.
