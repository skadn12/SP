# stage2_guarded_mining300_base240 Report

- method: current_method
- fixed_solver: solver_base_restart_stage1_merged
- generated_count: 155
- valid_count: 155
- clean_count: 155
- solver_exact_accuracy_k1: 0.496774
- solver_soft_accuracy_k5: 0.492903
- frontier_count: 37
- frontier_yield: 0.23871
- strict_frontier_count: 37
- boundary_frontier_count: 37
- frontier_pool_count: 39
- largest_type_share: 0.702703
- largest_skill_share: 0.567568
- dev_or_heldout_frontier_count: 0
- policy_created_by: native_current_method_graph_conditioned_api_dialogue
- policy_validation_pass: True
- policy_fallback_used: False
- length_guard_target_delta: 4000
- length_guard_parent_baseline_tokens: 7753
- length_guard_baseline_mode: max_actual_parent_and_configured_baseline
- parent_dialogue_tokens_mean: 6907.9677
- final_dialogue_tokens_mean: 12160.4323
- length_delta_tokens_mean: 5252.4645
- length_guard_effective_parent_tokens_mean: 7753.0
- length_guard_effective_delta_tokens_mean: 4407.4323
- length_guard_fallback_used: False
- fallback_delta: 0
- emergency_floor: 0
- rows_below_4000_delta: 0
- anchor_leakage_count: 0
- solver_visible_meta_leak_count: 0
- recommendation: viable_gate

## Soft Buckets

- 0/5: 61
- 1/5: 6
- 2/5: 9
- 3/5: 15
- 4/5: 7
- 5/5: 57

## Frontier By Question Type

- single_choice: 26
- multi_select: 11

## Frontier By Attack Skill

- conditional_plan: 21
- delayed_activation: 16

## Single Choice Split

- hard_single_choice_count: 85
- ordinary_single_choice_count: 8
- hard_single_choice_bucket_distribution: {'5/5': 31, '0/5': 30, '3/5': 11, '4/5': 6, '2/5': 4, '1/5': 3}
- ordinary_single_choice_bucket_distribution: {'5/5': 6, '1/5': 1, '2/5': 1}
- hard_single_choice_boundary_count: 24
- hard_single_choice_strict_frontier_count: 24
- hard_single_choice_near_miss_count: 0
- hard_single_choice_partial_credit_mean: 0.524706
- hard_single_choice_five_of_five_share: 0.364706
- hard_single_choice_zero_of_five_share: 0.352941
- ordinary_single_choice_five_of_five_share: 0.75
- ordinary_single_choice_zero_of_five_share: 0.0
