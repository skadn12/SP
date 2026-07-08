# base_restart_current_method_validation_smoke50 Report

- method: current_method
- fixed_solver: solver_base_qwen3_4b
- generated_count: 50
- valid_count: 50
- clean_count: 50
- solver_exact_accuracy_k1: 0.42
- solver_soft_accuracy_k5: 0.408
- frontier_count: 6
- frontier_yield: 0.12
- strict_frontier_count: 6
- boundary_frontier_count: 6
- frontier_pool_count: 8
- largest_type_share: 0.666667
- largest_skill_share: 0.5
- dev_or_heldout_frontier_count: 0
- policy_created_by: native_current_method_graph_conditioned_api_dialogue
- policy_validation_pass: True
- policy_fallback_used: False
- length_guard_target_delta: 4000
- length_guard_parent_baseline_tokens: 7753
- length_guard_baseline_mode: max_actual_parent_and_configured_baseline
- parent_dialogue_tokens_mean: 6940.24
- final_dialogue_tokens_mean: 12156.92
- length_delta_tokens_mean: 5216.68
- length_guard_effective_parent_tokens_mean: 7753.0
- length_guard_effective_delta_tokens_mean: 4403.92
- length_guard_fallback_used: False
- fallback_delta: 3000
- emergency_floor: 2500
- rows_below_4000_delta: 0
- anchor_leakage_count: 0
- solver_visible_meta_leak_count: 0
- recommendation: viable_gate

## Soft Buckets

- 0/5: 27
- 1/5: 1
- 2/5: 0
- 3/5: 4
- 4/5: 1
- 5/5: 17

## Frontier By Question Type

- single_choice: 4
- multi_select: 2

## Frontier By Attack Skill

- conditional_plan: 3
- delayed_activation: 3

## Single Choice Split

- hard_single_choice_count: 33
- ordinary_single_choice_count: 7
- hard_single_choice_bucket_distribution: {'0/5': 18, '5/5': 12, '3/5': 2, '1/5': 1}
- ordinary_single_choice_bucket_distribution: {'5/5': 4, '0/5': 2, '3/5': 1}
- hard_single_choice_boundary_count: 3
- hard_single_choice_strict_frontier_count: 3
- hard_single_choice_near_miss_count: 0
- hard_single_choice_partial_credit_mean: 0.406061
- hard_single_choice_five_of_five_share: 0.363636
- hard_single_choice_zero_of_five_share: 0.545455
- ordinary_single_choice_five_of_five_share: 0.571429
- ordinary_single_choice_zero_of_five_share: 0.285714
