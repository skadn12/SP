# Top-up Combined Mining Report

- experiment: `base_restart_current_method_validation`
- top-up mode: hard single-choice focused current_method mining
- original frontier preserved: 66 rows
- top-up candidates: 400
- top-up frontier: 40
- combined candidates: 900
- combined frontier_pool_count: 106
- combined_frontier_above_100: true
- top-up valid_count: 400
- top-up clean_count: 400
- top-up frontier_yield: 0.1
- top-up rows_below_4000_delta: 0
- top-up anchor_leakage_count: 0
- top-up solver_visible_meta_leak_count: 0
- top-up qtype_distribution: `{'single_choice': 400}`
- top-up frontier_qtype_distribution: `{'single_choice': 40}`
- combined frontier_qtype_distribution: `{'multi_select': 23, 'ordering': 8, 'single_choice': 75}`
- top-up template_distribution: `{'latest_valid_state': 125, 'minimal_difference': 130, 'sequence_as_single_choice': 145}`
- top-up frontier_bucket_distribution: `{'1/5': 12, '2/5': 8, '3/5': 11, '4/5': 9}`
- sample_id_overlap: `{'topup_vs_original_candidate': 0, 'topup_frontier_vs_original_candidate': 0, 'topup_frontier_vs_original_frontier': 0, 'train_safety_note': 'No sample_id collisions were found; original 66 frontier rows are preserved unchanged in the combined copy.'}`

## Files

- combined candidate pool: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/01_mining500_train/mining/topup_combined_candidate_pool.jsonl`
- combined frontier pool: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/01_mining500_train/mining/topup_combined_frontier_pool.jsonl`
- combined summary: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/01_mining500_train/mining/topup_combined_summary.json`
- top-up run report: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/01_mining500_train/mining/topup_hard_sc350_resume/reports/report.md`
- original frontier pool kept unchanged: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/01_mining500_train/mining/frontier_pool.jsonl`

## Notes

This top-up used from-base fixed solver eval with `/root/autodl-tmp/Qwen3-4B`, tokenizer `/root/autodl-tmp/Qwen3-4B`, and 16384 context. It did not use evidence-window prompts, r4 parents, r4 train mix, r4/probe solver, or r1 legacy data.
