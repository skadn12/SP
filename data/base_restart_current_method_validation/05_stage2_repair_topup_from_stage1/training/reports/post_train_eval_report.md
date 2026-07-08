# Stage2 Repair Post-Train Eval Report

- status: success
- init model: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/accepted_models/solver_base_restart_stage1_merged`
- merged model: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/training/models/merged`
- tokenizer: `/root/autodl-tmp/Qwen3-4B`
- max_model_len: 16384
- evidence_window_used_as_prompt: false
- r4/r5/r1 data/model used: false

## Train Mix
- combined_frontier_pool_count: 108
- frontier_train_count: 78
- frontier_heldout_count: 30
- clean_replay_count: 30
- prior_failed_heldout_in_train_count: 0
- heldout_overlap_check: `{'dialogue_hash_overlap': 0, 'question_id_overlap': 0, 'sample_id_overlap': 0, 'total_overlap': 0}`
- max_sft_token_length: 12950
- over_max_length_count: 0

## Metrics
- train frontier k1/k5_soft: 0.846154 / 0.828205
- heldout frontier k1/k5_soft: 0.533333 / 0.526667
- clean replay k1/k5_soft: 1.0 / 1.0
- combined candidate post k1/k5_soft: 0.587174 / 0.587575
- stage1 baseline k1/k5_soft: 0.496994 / 0.503006
- candidate_pool_delta_k1: 0.09018
- format guard k1/k5_soft: 0.636364 / 0.663636
- overflow_count: 0
- truncated_count: 0
- empty_prediction_count: 0
- parsed_failure_count: 0

## Pass Criteria
- pass_criteria: `{'train_frontier_recovery_k1_gte_0_80': True, 'heldout_frontier_recovery_k1_gte_0_50': True, 'clean_replay_recovery_k1_gte_0_95': True, 'combined_candidate_pool_post_k1_gt_stage1_baseline': True, 'combined_candidate_pool_delta_gte_0_05': True, 'overflow_count_zero': True, 'truncated_count_zero': True, 'parsed_failure_count_zero': True, 'empty_prediction_count_zero': True, 'train_mix_audit_passed': True, 'evidence_window_prompt_false': True}`

## Decision
- accepted: true
- stop: true; no stage3 was started.
