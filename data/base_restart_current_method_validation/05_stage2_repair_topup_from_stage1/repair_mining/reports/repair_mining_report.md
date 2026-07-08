# Stage2 Repair Top-up Mining Report

- status: gate_failed_frontier_insufficient
- generated/valid/clean: 119/119/119
- strict_frontier_count: 15
- combined_strict_frontier_count: 88
- near_miss_zero_partial_count: 15
- frontier_plus_zero_partial_near_miss_count: 30
- qtype_distribution: `{'multi_select': 29, 'single_choice': 90}`
- qtype_zero_share: `{'multi_select': 0.517241, 'single_choice': 0.488889}`
- hard_single_choice_template_distribution: `{'latest_valid_state': 19, 'minimal_difference': 16, 'sequence_as_single_choice': 44, 'override_sensitive': 5, 'guaranteed_vs_possible': 6}`
- api_reject_counts: `{'parse_failure': 6, 'NativeGraphDialogueReject': 6, 'meta_leak': 4, 'dialogue_alignment': 1, 'base_validation_anti_shortcut': 2, 'easy_v2_attack_skill_cap': 1}`
- api_graph_403_count: 0
- rows_below_4000_delta: 0
- anchor_leakage_count: 0
- solver_visible_meta_leak_count: 0
- eval_context_overflow_count: 0
- truncated_count: 0
- prompt_token_max: 12845
- evidence_window_used_as_prompt: false

## Gate
- repair_gate: `{'generated_gte_100': True, 'valid_clean_rate_gte_095': True, 'rows_below_4000_delta_eq_0': True, 'anchor_leakage_eq_0': True, 'solver_visible_meta_leak_eq_0': True, 'eval_context_overflow_eq_0': True, 'truncated_eq_0': True, 'strict_repair_frontier_gte_27_for_combined_100': False, 'combined_strict_frontier_count': 88, 'combined_strict_frontier_gte_100': False}`
- train_mix_allowed: false
- blocker: strict repair frontier insufficient for larger heldout repair train_mix; do not train

## Files
- candidate_pool_repair: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/repair_mining/candidate_pool_repair.jsonl`
- frontier_pool_repair: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/repair_mining/frontier_pool_repair.jsonl`
- near_miss_zero_partial_pool_repair: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/repair_mining/near_miss_zero_partial_pool_repair.jsonl`
- adaptive_repair_summary: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/repair_mining/adaptive_repair_summary.json`
- report: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/repair_mining/reports/repair_mining_report.md`
- eval_snapshot: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/repair_mining/eval119/samples_snapshot.jsonl`
- eval_summary: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/repair_mining/eval119/reports/verify_summary.json`
