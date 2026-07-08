# Stage2 Repair Train Mix Audit Report

- status: audit_passed
- combined_frontier_pool_count: 108
- frontier_train_count: 78
- frontier_heldout_count: 30
- clean_replay_count: 30
- train_mix_count: 108
- clean_replay_share: 0.277778
- prior_failed_heldout_reserved_count: 15
- prior_failed_heldout_in_train_count: 0
- near_miss_aux_pool_count: 131
- heldout_overlap_check: `{'sample_id_overlap': 0, 'question_id_overlap': 0, 'dialogue_hash_overlap': 0, 'total_overlap': 0}`
- tokenizer_path: `/root/autodl-tmp/Qwen3-4B`
- max_sft_token_length: 12950
- over_max_length_count: 0
- evidence_window_used_as_prompt: false
- train_mix_pass: True

## Frontier Train
- qtype_distribution: `{'single_choice': 64, 'multi_select': 14}`
- bucket_distribution: `{'1/5': 23, '3/5': 20, '2/5': 18, '4/5': 17}`
- hard_single_choice_template_distribution: `{'minimal_difference': 18, 'sequence_as_single_choice': 17, 'latest_valid_state': 13, 'guaranteed_vs_possible': 8, 'override_sensitive': 4, 'none': 4}`
- source_phase_distribution: `{'base': 24, 'base_fill_after_recharge': 17, 'stage2_repair_boundary_topup_from_stage1': 15, 'stage2_repair_topup_from_stage1': 12, 'bonus_after_recharge_ordering_frozen': 10}`

## Frontier Heldout
- qtype_distribution: `{'single_choice': 21, 'multi_select': 9}`
- bucket_distribution: `{'2/5': 10, '3/5': 9, '1/5': 7, '4/5': 4}`
- hard_single_choice_template_distribution: `{'sequence_as_single_choice': 11, 'minimal_difference': 7, 'latest_valid_state': 2, 'guaranteed_vs_possible': 1}`
- source_phase_distribution: `{'base': 13, 'base_fill_after_recharge': 6, 'stage2_repair_boundary_topup_from_stage1': 5, 'bonus_after_recharge_ordering_frozen': 3, 'stage2_repair_topup_from_stage1': 3}`

## Clean Replay
- qtype_distribution: `{'single_choice': 22, 'multi_select': 6, 'ordering': 2}`
- bucket_distribution: `{'5/5': 30}`
- notes: `['ordering 5/5 clean candidates are scarce: available=2; selected=2']`

## Pass Criteria
- pass_criteria: `{'combined_frontier_pool_count_gte_100': True, 'frontier_train_count_gte_65': True, 'frontier_heldout_count_gte_30': True, 'clean_replay_count_25_to_35': True, 'prior_failed_heldout_not_in_train': True, 'heldout_overlap_zero': True, 'over_max_length_zero': True, 'max_sft_token_length_lte_16384': True, 'evidence_window_prompt_false': True}`

## Files
- frontier_train: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/train_mix/frontier_train.jsonl`
- frontier_heldout: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/train_mix/frontier_heldout.jsonl`
- clean_replay: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/train_mix/clean_replay.jsonl`
- near_miss_aux_pool: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/train_mix/near_miss_aux_pool.jsonl`
- excluded_prior_heldout: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/train_mix/excluded_prior_heldout.jsonl`
- excluded_train_heldout_key_conflicts: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/train_mix/excluded_train_heldout_key_conflicts.jsonl`
- train_mix: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/train_mix/train_mix.jsonl`
- sft_train: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/train_mix/sft_train.jsonl`
- train_mix_summary: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/train_mix/train_mix_summary.json`
- train_mix_report: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/train_mix/reports/train_mix_report.md`
