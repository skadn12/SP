# Accepted Stage2 Solver inference protocol audit

The accepted solver was evaluated by the legacy solver evaluator with the base tokenizer at
`/root/autodl-tmp/Qwen3-4B`, not the merged-model tokenizer. Its prompt rendering calls
`apply_chat_template(..., tokenize=False, add_generation_prompt=True, enable_thinking=False)`.

For multi-sample evaluation the recorded evaluator defaults are temperature `0.7` and top-p
`0.9`. It supplies stop sequences `"\n\n"`, `"\n"`, and `"</answer>"`. The accepted Stage2
post-train evaluation records an effective max model length of `16384`. Its single-choice format
instruction requests exactly one option letter without explanation; the legacy structured cap was
8 tokens. The isolated S1-v2 smoke preserves this protocol but uses the explicitly requested
32-token cap as a small format margin.

Primary sources inspected:

- `scripts/old_method/run_old_improved_gate200.py` (`render_prompt`,
  `effective_solver_max_new_tokens`, `evaluate_with_vllm`)
- `base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/training/reports/post_train_eval_summary.json`
- the corresponding `training/reports/eval_frontier_train_recovery.log`
