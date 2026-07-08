# Stage2 Accepted Solver Audit

- created_at: 2026-07-03 21:06:53
- accepted solver symlink: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/accepted_models/solver_base_restart_stage2_merged`
- accepted solver target: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/training/models/merged`
- frozen: true
- freeze record: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/reports/stage2_accepted_solver_freeze_record.json`
- no Stage3 smoke/mining/training was started.
- tokenizer: `/root/autodl-tmp/Qwen3-4B`
- max_model_len: 16384
- evidence_window_used_as_prompt: false
- r4/r5/r1 data/model used: false

## Freeze Status

Stage2 accepted solver is treated as frozen from this audit onward. The model target has 14 files and was not modified by these supplemental evals; outputs were written only under experiment reports/eval paths.

## Accepted Stage2 Summary

- train frontier k1/k5_soft: 0.846154 / 0.828205
- heldout frontier k1/k5_soft: 0.533333 / 0.526667
- clean replay k1/k5_soft: 1.0 / 1.0
- candidate post k1/k5_soft: 0.587174 / 0.587575
- stage1 baseline on same candidate pool k1/k5_soft: 0.496994 / 0.503006
- candidate delta k1: 0.09018

## Supplemental Regression Guards

| Guard | k1 | k5_soft | bad_question | overflow | truncated | empty_prediction | parsed_failure |
|---|---:|---:|---:|---:|---:|---:|---:|
| old_failed_heldout_regression_guard | 0.533333 | 0.573333 | 0 | 0 | 0 | 0 | 0 |
| stage1_frontier_heldout_regression_guard | 0.636364 | 0.663636 | 4 | 0 | 0 | 0 | 0 |
| stage1_clean_replay_regression_guard | 0.9 | 0.926667 | 0 | 0 | 0 | 0 | 0 |

## Comparison To Stage1 Original Eval

- Stage1 original heldout recovery k1/k5_soft: 0.681818 / 0.618182; stage2 accepted on same heldout k1 delta: -0.045454
- Stage1 original clean replay recovery k1/k5_soft: 0.966667 / 0.953333; stage2 accepted on same clean replay k1 delta: -0.066667

## Risk Notes

- Stage1 clean replay regression guard is below 0.95 retention target: k1=0.900; inspect before Stage3.

## Key Files

- training summary: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/training/reports/training_summary.json`
- post-train eval summary: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/training/reports/post_train_eval_summary.json`
- old failed heldout guard summary: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/training/eval/old_failed_heldout_regression_guard/reports/verify_summary.json`
- stage1 heldout guard summary: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/training/eval/stage1_frontier_heldout_regression_guard/reports/verify_summary.json`
- stage1 clean replay guard summary: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/training/eval/stage1_clean_replay_regression_guard/reports/verify_summary.json`
- accepted record: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/accepted_models/solver_base_restart_stage2_accepted_record.json`

## Decision

Stage2 accepted solver remains accepted and frozen. Do not enter Stage3 until a human explicitly confirms Stage3 smoke50.
