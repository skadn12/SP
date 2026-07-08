# base_restart_current_method_validation Mining Report
- candidate_budget: 500
- base_count: 350
- bonus_count: 150
- generated_count: 500
- valid_count: 500
- clean_count: 500
- valid_rate: 1.0
- clean_rate: 1.0
- frontier_pool_count: 66
- boundary_frontier_pool_count: 43
- near_miss_pool_count: 23
- overall_zero_share: 0.508
- hard_single_choice_zero_share: 0.442897
- multi_select_zero_share: 0.878378
- ordering_zero_share: 0.777778
- rows_below_4000_delta: 0
- anchor_leakage_count: 0
- solver_visible_meta_leak_count: 0
- bad_question_count: 170
- overflow_count: 0
- truncated_count: 0
- mining_pass: False

## qtype_distribution
- multi_select: 74
- ordering: 27
- single_choice: 399

## frontier_pool_qtype_distribution
- multi_select: 23
- ordering: 8
- single_choice: 35

## bonus allocation
- candidate_budget: 150
- hard_single_choice: 120
- multi_select_conditional_plan: 20
- ordering_timeline_fork_merge: 10
- base_qtype_zero_share: {'single_choice': 0.414286, 'multi_select': 0.886792, 'ordering': 0.705882, 'hard_single_choice': 0.445833}
- guardrail: multi_select and ordering zero_share exceeded 0.60 in base350, so bonus is capped and h0-only for those qtypes.
- safe_buckets: ['hard_single_choice templates', 'multi_select|conditional_plan', 'ordering|timeline_fork_merge']

## pass criteria
- valid_rate_ge_0.95: True
- clean_rate_ge_0.95: True
- rows_below_4000_delta_eq_0: True
- anchor_leakage_count_eq_0: True
- solver_visible_meta_leak_count_eq_0: True
- frontier_pool_count_ge_80: False
- overflow_count_eq_0: True
- truncated_count_eq_0: True

## blockers
- frontier_pool_count below required 80
