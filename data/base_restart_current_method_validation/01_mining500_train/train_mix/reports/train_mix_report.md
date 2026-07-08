# Train Mix Audit Report

- combined_frontier_pool_count: 106
- frontier_train_count: 84
- frontier_heldout_count: 22
- clean_replay_count: 30
- train_mix_count: 114
- clean_replay_share: 0.263158
- train_mix_pass: True
- tokenizer_path: `/root/autodl-tmp/Qwen3-4B`
- max_sft_token_length: 12873
- over_max_length_count: 0
- evidence_window_used_as_prompt: false
- heldout_overlap_check: `{'sample_id_overlap': 0, 'question_id_overlap': 0, 'dialogue_hash_overlap': 0, 'total_overlap': 0}`

## Frontier Train
- source_distribution: `{'original_mining500': 52, 'topup': 32}`
- qtype_distribution: `{'single_choice': 60, 'multi_select': 18, 'ordering': 6}`
- frontier_type_distribution: `{'boundary': 66, 'near_miss': 18}`
- hard_single_choice_template_distribution: `{'minimal_difference': 21, 'latest_valid_state': 15, 'sequence_as_single_choice': 15, 'guaranteed_vs_possible': 3, 'override_sensitive': 3, 'none': 3}`

## Frontier Heldout
- source_distribution: `{'original_mining500': 14, 'topup': 8}`
- qtype_distribution: `{'single_choice': 15, 'multi_select': 5, 'ordering': 2}`
- frontier_type_distribution: `{'boundary': 17, 'near_miss': 5}`
- hard_single_choice_template_distribution: `{'minimal_difference': 5, 'sequence_as_single_choice': 4, 'latest_valid_state': 3, 'guaranteed_vs_possible': 1, 'override_sensitive': 1, 'none': 1}`

## Clean Replay
- source_distribution: `{'original_mining500': 19, 'topup': 11}`
- qtype_distribution: `{'single_choice': 23, 'multi_select': 5, 'ordering': 2}`
- hard_single_choice_template_distribution: `{'latest_valid_state': 6, 'sequence_as_single_choice': 6, 'minimal_difference': 4, 'guaranteed_vs_possible': 3, 'override_sensitive': 2, 'none': 2}`
- notes: `['ordering 5/5 clean candidates are scarce: available=2; selected=2']`

## Files
- frontier_train: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/01_mining500_train/train_mix/frontier_train.jsonl`
- frontier_heldout: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/01_mining500_train/train_mix/frontier_heldout.jsonl`
- clean_replay: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/01_mining500_train/train_mix/clean_replay.jsonl`
- train_mix: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/01_mining500_train/train_mix/train_mix.jsonl`
- sft_train: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/01_mining500_train/train_mix/sft_train.jsonl`
- train_mix_summary: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/01_mining500_train/train_mix/train_mix_summary.json`
- train_mix_report: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/01_mining500_train/train_mix/reports/train_mix_report.md`
