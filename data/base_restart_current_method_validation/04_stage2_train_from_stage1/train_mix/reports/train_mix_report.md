# Stage2 Train Mix Audit Report

- status: audit_passed
- frontier_pool_count: 73
- frontier_train_count: 58
- frontier_heldout_count: 15
- clean_replay_count: 23
- near_miss_aux_pool_count: 101
- train_mix_count: 81
- clean_replay_share: 0.283951
- train_mix_pass: True
- tokenizer_path: `/root/autodl-tmp/Qwen3-4B`
- max_sft_token_length: 12882
- over_max_length_count: 0
- evidence_window_used_as_prompt: false
- heldout_overlap_check: `{'sample_id_overlap': 0, 'question_id_overlap': 0, 'dialogue_hash_overlap': 0, 'total_overlap': 0}`
- ordering_count: 12
- ordering_zero_share_note: ordering_zero_share=0.833333; ordering is too hard, not oversampled; ordering kept at natural strict-frontier proportion and only 5/5 clean replay if available

## Frontier Train
- qtype_distribution: `{'single_choice': 43, 'multi_select': 15}`
- bucket_distribution: `{'3/5': 19, '1/5': 15, '4/5': 13, '2/5': 11}`
- hard_single_choice_template_distribution: `{'minimal_difference': 15, 'latest_valid_state': 8, 'guaranteed_vs_possible': 6, 'sequence_as_single_choice': 6, 'override_sensitive': 4, 'none': 4}`

## Frontier Heldout
- qtype_distribution: `{'single_choice': 11, 'multi_select': 4}`
- bucket_distribution: `{'3/5': 6, '2/5': 4, '1/5': 3, '4/5': 2}`
- hard_single_choice_template_distribution: `{'sequence_as_single_choice': 6, 'minimal_difference': 3, 'latest_valid_state': 1, 'guaranteed_vs_possible': 1}`

## Clean Replay
- qtype_distribution: `{'single_choice': 15, 'multi_select': 6, 'ordering': 2}`
- bucket_distribution: `{'5/5': 23}`
- notes: `['ordering 5/5 clean candidates are scarce: available=2; selected=2']`

## Pass Criteria
- pass_criteria: `{'frontier_pool_count_eq_73': True, 'frontier_train_count_gte_55': True, 'frontier_heldout_count_gte_14': True, 'clean_replay_count_20_to_25': True, 'near_miss_aux_pool_count_eq_101': True, 'heldout_overlap_zero': True, 'over_max_length_zero': True, 'max_sft_token_length_lte_16384': True, 'evidence_window_prompt_false': True}`

## Files
- frontier_train: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/04_stage2_train_from_stage1/train_mix/frontier_train.jsonl`
- frontier_heldout: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/04_stage2_train_from_stage1/train_mix/frontier_heldout.jsonl`
- clean_replay: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/04_stage2_train_from_stage1/train_mix/clean_replay.jsonl`
- near_miss_aux_pool: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/04_stage2_train_from_stage1/train_mix/near_miss_aux_pool.jsonl`
- train_mix: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/04_stage2_train_from_stage1/train_mix/train_mix.jsonl`
- sft_train: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/04_stage2_train_from_stage1/train_mix/sft_train.jsonl`
- train_mix_summary: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/04_stage2_train_from_stage1/train_mix/train_mix_summary.json`
- train_mix_report: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/04_stage2_train_from_stage1/train_mix/reports/train_mix_report.md`
