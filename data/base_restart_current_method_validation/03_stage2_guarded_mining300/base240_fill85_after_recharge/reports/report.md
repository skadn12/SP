# stage2_guarded_mining300_base240_fill85_after_recharge Report

- method: current_method
- fixed_solver: solver_base_restart_stage1_merged
- generated_count: 80
- valid_count: 80
- clean_count: 80
- solver_exact_accuracy_k1: 0.575
- solver_soft_accuracy_k5: 0.5475
- frontier_count: 21
- frontier_yield: 0.2625
- strict_frontier_count: 21
- boundary_frontier_count: 21
- frontier_pool_count: 24
- largest_type_share: 0.666667
- largest_skill_share: 0.47619
- dev_or_heldout_frontier_count: 0
- policy_created_by: native_current_method_graph_conditioned_api_dialogue
- policy_validation_pass: True
- policy_fallback_used: False
- length_guard_target_delta: 4000
- length_guard_parent_baseline_tokens: 7753
- length_guard_baseline_mode: max_actual_parent_and_configured_baseline
- parent_dialogue_tokens_mean: 6905.45
- final_dialogue_tokens_mean: 12158.975
- length_delta_tokens_mean: 5253.525
- length_guard_effective_parent_tokens_mean: 7753.0
- length_guard_effective_delta_tokens_mean: 4405.975
- length_guard_fallback_used: False
- fallback_delta: 0
- emergency_floor: 0
- rows_below_4000_delta: 0
- anchor_leakage_count: 0
- solver_visible_meta_leak_count: 0
- recommendation: viable_gate

## Soft Buckets

- 0/5: 25
- 1/5: 6
- 2/5: 5
- 3/5: 7
- 4/5: 3
- 5/5: 34

## Frontier By Question Type

- single_choice: 14
- multi_select: 7

## Frontier By Attack Skill

- conditional_plan: 10
- delayed_activation: 10
- permission_visibility_scope: 1

## Single Choice Split

- hard_single_choice_count: 42
- ordinary_single_choice_count: 6
- hard_single_choice_bucket_distribution: {'5/5': 20, '0/5': 10, '3/5': 6, '2/5': 3, '4/5': 2, '1/5': 1}
- ordinary_single_choice_bucket_distribution: {'5/5': 4, '4/5': 1, '3/5': 1}
- hard_single_choice_boundary_count: 12
- hard_single_choice_strict_frontier_count: 12
- hard_single_choice_near_miss_count: 0
- hard_single_choice_partial_credit_mean: 0.633333
- hard_single_choice_five_of_five_share: 0.47619
- hard_single_choice_zero_of_five_share: 0.238095
- ordinary_single_choice_five_of_five_share: 0.666667
- ordinary_single_choice_zero_of_five_share: 0.0
