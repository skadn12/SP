# stage2_guarded_mining300_bonus60_after_recharge Report

- method: current_method
- fixed_solver: solver_base_restart_stage1_merged
- generated_count: 60
- valid_count: 60
- clean_count: 60
- solver_exact_accuracy_k1: 0.533333
- solver_soft_accuracy_k5: 0.516667
- frontier_count: 13
- frontier_yield: 0.216667
- strict_frontier_count: 13
- boundary_frontier_count: 13
- frontier_pool_count: 14
- largest_type_share: 0.923077
- largest_skill_share: 0.615385
- dev_or_heldout_frontier_count: 0
- policy_created_by: native_current_method_graph_conditioned_api_dialogue
- policy_validation_pass: True
- policy_fallback_used: False
- length_guard_target_delta: 4000
- length_guard_parent_baseline_tokens: 7753
- length_guard_baseline_mode: max_actual_parent_and_configured_baseline
- parent_dialogue_tokens_mean: 6936.5
- final_dialogue_tokens_mean: 12153.7667
- length_delta_tokens_mean: 5217.2667
- length_guard_effective_parent_tokens_mean: 7753.0
- length_guard_effective_delta_tokens_mean: 4400.7667
- length_guard_fallback_used: False
- fallback_delta: 0
- emergency_floor: 0
- rows_below_4000_delta: 0
- anchor_leakage_count: 0
- solver_visible_meta_leak_count: 0
- recommendation: viable_gate

## Soft Buckets

- 0/5: 23
- 1/5: 4
- 2/5: 1
- 3/5: 3
- 4/5: 5
- 5/5: 24

## Frontier By Question Type

- single_choice: 12
- multi_select: 1

## Frontier By Attack Skill

- conditional_plan: 8
- delayed_activation: 5

## Single Choice Split

- hard_single_choice_count: 33
- ordinary_single_choice_count: 0
- hard_single_choice_bucket_distribution: {'5/5': 11, '0/5': 10, '1/5': 4, '4/5': 4, '3/5': 3, '2/5': 1}
- ordinary_single_choice_bucket_distribution: {}
- hard_single_choice_boundary_count: 12
- hard_single_choice_strict_frontier_count: 12
- hard_single_choice_near_miss_count: 0
- hard_single_choice_partial_credit_mean: 0.521212
- hard_single_choice_five_of_five_share: 0.333333
- hard_single_choice_zero_of_five_share: 0.30303
- ordinary_single_choice_five_of_five_share: 0.0
- ordinary_single_choice_zero_of_five_share: 0.0
