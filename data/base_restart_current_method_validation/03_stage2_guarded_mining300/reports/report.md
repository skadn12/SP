# Stage2 Guarded Mining300 Resume Report

## Decision

- status: PASS
- recommendation: 可以人工确认是否进入 stage2 train_mix
- training_status: not_started
- train_mix_status: not_constructed
- next_round_status: not_started

## Inputs

- fixed_solver: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/accepted_models/solver_base_restart_stage1_merged`
- tokenizer_path: `/root/autodl-tmp/Qwen3-4B`
- seed_parent_path: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/seed_parents/seed_parents.jsonl`
- evidence_window_used_as_prompt: false
- r4_r5_r1_data_or_model_used: false

## Counts

- candidate_budget: 300
- base_count: 240
- bonus_count: 60
- generated_count: 300
- valid_count: 300
- clean_count: 300
- valid_rate: 1.0
- clean_rate: 1.0
- frontier_pool_count: 73
- strict_frontier_count: 73
- near_miss_count: 101
- frontier_plus_near_miss_count: 174

## Quality Gates

- rows_below_4000_delta: 0
- anchor_leakage_count: 0
- solver_visible_meta_leak_count: 0
- eval_context_overflow_count: 0
- truncated_count: 0
- bad_question_count: 51
- empty_prediction_count: 0
- parsed_failure_count: 0
- prompt_token_max: 12790
- final_dialogue_token_max: 12258

## Distribution

- qtype_distribution: {'multi_select': 111, 'ordering': 12, 'single_choice': 177}
- qtype_bucket_distribution: {'multi_select': {'0/5': 49, '1/5': 7, '2/5': 6, '3/5': 4, '4/5': 2, '5/5': 43}, 'ordering': {'0/5': 10, '5/5': 2}, 'single_choice': {'0/5': 50, '1/5': 11, '2/5': 9, '3/5': 21, '4/5': 13, '5/5': 73}}
- frontier_pool_qtype_distribution: {'multi_select': 19, 'single_choice': 54}
- hard_single_choice_template_distribution: {'guaranteed_vs_possible': 28, 'latest_valid_state': 34, 'minimal_difference': 44, 'override_sensitive': 24, 'sequence_as_single_choice': 33}
- overall_zero_share: 0.363333
- hard_single_choice_zero_share: 0.306748
- multi_select_zero_share: 0.441441
- ordering_zero_share: 0.833333
- ordering_sample_count: 12
- ordering_bonus_remained_frozen: True

## API 403

- api_graph_403_count_prior_to_recharge: 60
- api_graph_403_count_after_recharge_resume: 0
- api_graph_403_recheck_after_recharge_status: recovered_minimal_graph_chat_completion_200
- api_failure_reason_counts_after_candidate_subruns: {'NativeGraphDialogueReject': 29, 'api_dialogue': 5, 'api_graph': 202, 'base_validation_anti_shortcut': 10, 'dialogue_alignment': 1, 'meta_leak': 35, 'parse_failure': 32, 'slot_skipped_attempt_cap': 29, 'timeout': 2}

## Phase Summaries

- base240: generated=155, clean=155, frontier=37, near_miss=54, qtypes={'multi_select': 54, 'ordering': 8, 'single_choice': 93}
- base240_fill85_after_recharge: generated=80, clean=80, frontier=21, near_miss=31, qtypes={'multi_select': 28, 'ordering': 4, 'single_choice': 48}
- base240_fill5_after_recharge: generated=5, clean=5, frontier=2, near_miss=0, qtypes={'multi_select': 2, 'single_choice': 3}
- bonus60_after_recharge: generated=60, clean=60, frontier=13, near_miss=16, qtypes={'multi_select': 27, 'single_choice': 33}

## Files

- candidate_pool: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/03_stage2_guarded_mining300/candidate_pool.jsonl`
- frontier_pool: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/03_stage2_guarded_mining300/frontier_pool.jsonl`
- adaptive_smoke_summary: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/03_stage2_guarded_mining300/adaptive_smoke_summary.json`
- report: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/03_stage2_guarded_mining300/reports/report.md`
- previous candidate backup: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/03_stage2_guarded_mining300/candidate_pool.jsonl.before_recharge_resume`
- previous frontier backup: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/03_stage2_guarded_mining300/frontier_pool.jsonl.before_recharge_resume`

No training, no train_mix construction, and no automatic next round were run.
