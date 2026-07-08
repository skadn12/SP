# Stage2 Heldout Failure Diagnosis

This is a diagnosis-only step. No mining, training, train_mix rebuild, or stage3 run was executed.

## Headline
- heldout_n: 15
- post_train heldout k1/k5_soft: 0.333333 / 0.48
- gate failure: ['heldout frontier recovery below 0.50']
- bad_question / true_too_hard / near_miss: 4 / 1 / 3
- overflow/truncated/prompt_token_max: 0 / 0 / 12709

## Same Heldout: Stage1 Mining Eval -> Stage2 Post-Train
- stage1 k1 at mining: 0.733333
- stage2 post-train k1: 0.333333
- stage1 mean exact-count: 2.466667
- stage2 mean exact-count: 2.4
- improved/same/degraded by exact-count: 7/0/8
- k1 gained/lost: 2/8
- soft-good-but-k1-bad rows (>=4/5 but first sample wrong): 2
- zero-bucket rows: 5
- transitions: {'3/5->2/5': 2, '2/5->4/5': 2, '3/5->4/5': 2, '3/5->0/5': 1, '4/5->5/5': 2, '1/5->0/5': 3, '2/5->5/5': 1, '3/5->1/5': 1, '2/5->0/5': 1}

## Group Stats
### question_type
- single_choice: n=11, stage2_k1=0.363636, stage2_mean_exact=2.272727, buckets={'4/5': 3, '0/5': 4, '2/5': 1, '5/5': 2, '1/5': 1}, transitions={'3/5->4/5': 2, '3/5->0/5': 1, '3/5->2/5': 1, '2/5->4/5': 1, '4/5->5/5': 2, '1/5->0/5': 2, '3/5->1/5': 1, '2/5->0/5': 1}
- multi_select: n=4, stage2_k1=0.25, stage2_mean_exact=2.75, buckets={'2/5': 1, '4/5': 1, '5/5': 1, '0/5': 1}, transitions={'3/5->2/5': 1, '2/5->4/5': 1, '2/5->5/5': 1, '1/5->0/5': 1}

### template_or_skill
- sequence_as_single_choice: n=6, stage2_k1=0.166667, stage2_mean_exact=0.833333, buckets={'0/5': 4, '1/5': 1, '4/5': 1}, transitions={'3/5->0/5': 1, '1/5->0/5': 2, '3/5->1/5': 1, '2/5->0/5': 1, '3/5->4/5': 1}
- conditional_plan: n=4, stage2_k1=0.25, stage2_mean_exact=2.75, buckets={'2/5': 1, '4/5': 1, '5/5': 1, '0/5': 1}, transitions={'3/5->2/5': 1, '2/5->4/5': 1, '2/5->5/5': 1, '1/5->0/5': 1}
- minimal_difference: n=3, stage2_k1=0.666667, stage2_mean_exact=4.333333, buckets={'4/5': 2, '5/5': 1}, transitions={'3/5->4/5': 1, '2/5->4/5': 1, '4/5->5/5': 1}
- guaranteed_vs_possible: n=1, stage2_k1=1.0, stage2_mean_exact=5.0, buckets={'5/5': 1}, transitions={'4/5->5/5': 1}
- latest_valid_state: n=1, stage2_k1=0.0, stage2_mean_exact=2.0, buckets={'2/5': 1}, transitions={'3/5->2/5': 1}

### source_id
- locomo_raw_sample4_chunk001: n=3, stage2_k1=0.333333, stage2_mean_exact=2.0, buckets={'2/5': 1, '4/5': 1, '0/5': 1}, transitions={'3/5->2/5': 1, '2/5->4/5': 1, '1/5->0/5': 1}
- locomo_raw_sample1_chunk001: n=2, stage2_k1=0.0, stage2_mean_exact=0.0, buckets={'0/5': 2}, transitions={'1/5->0/5': 1, '2/5->0/5': 1}
- locomo_raw_sample2_chunk000: n=2, stage2_k1=0.0, stage2_mean_exact=0.5, buckets={'0/5': 1, '1/5': 1}, transitions={'3/5->0/5': 1, '3/5->1/5': 1}
- locomo_raw_sample3_chunk000: n=2, stage2_k1=1.0, stage2_mean_exact=4.5, buckets={'4/5': 1, '5/5': 1}, transitions={'3/5->4/5': 1, '4/5->5/5': 1}
- locomo_raw_sample0_chunk000: n=1, stage2_k1=0.0, stage2_mean_exact=4.0, buckets={'4/5': 1}, transitions={'3/5->4/5': 1}
- locomo_raw_sample2_chunk002: n=1, stage2_k1=1.0, stage2_mean_exact=5.0, buckets={'5/5': 1}, transitions={'4/5->5/5': 1}
- locomo_raw_sample3_chunk001: n=1, stage2_k1=0.0, stage2_mean_exact=2.0, buckets={'2/5': 1}, transitions={'3/5->2/5': 1}
- locomo_raw_sample3_chunk002: n=1, stage2_k1=1.0, stage2_mean_exact=5.0, buckets={'5/5': 1}, transitions={'2/5->5/5': 1}
- locomo_raw_sample4_chunk002: n=1, stage2_k1=0.0, stage2_mean_exact=0.0, buckets={'0/5': 1}, transitions={'1/5->0/5': 1}
- locomo_raw_sample5_chunk000: n=1, stage2_k1=0.0, stage2_mean_exact=4.0, buckets={'4/5': 1}, transitions={'2/5->4/5': 1}

### source_subrun
- base240: n=8, stage2_k1=0.25, stage2_mean_exact=2.625, buckets={'2/5': 2, '4/5': 3, '0/5': 2, '5/5': 1}, transitions={'3/5->2/5': 2, '2/5->4/5': 2, '3/5->4/5': 1, '3/5->0/5': 1, '4/5->5/5': 1, '1/5->0/5': 1}
- base240_fill85_after_recharge: n=4, stage2_k1=0.25, stage2_mean_exact=1.5, buckets={'5/5': 1, '1/5': 1, '0/5': 2}, transitions={'2/5->5/5': 1, '3/5->1/5': 1, '2/5->0/5': 1, '1/5->0/5': 1}
- bonus60_after_recharge: n=3, stage2_k1=0.666667, stage2_mean_exact=3.0, buckets={'4/5': 1, '5/5': 1, '0/5': 1}, transitions={'3/5->4/5': 1, '4/5->5/5': 1, '1/5->0/5': 1}

## Notable Rows
- zero_bucket_sample_ids: ['native_current_method_graph_dialogue_0113_145d72ad92', 'native_current_method_graph_dialogue_0111_cd5b8ff3f7', 'native_current_method_graph_dialogue_0046_1c7cc5d04a', 'native_current_method_graph_dialogue_0038_7320227b8c', 'native_current_method_graph_dialogue_0033_780fd675ab']
- soft_good_but_k1_bad_sample_ids: ['native_current_method_graph_dialogue_0012_215c677c26', 'native_current_method_graph_dialogue_0063_4487fc4ac6']
- degraded_sample_ids: ['native_current_method_graph_dialogue_0100_f6cc64e595', 'native_current_method_graph_dialogue_0113_145d72ad92', 'native_current_method_graph_dialogue_0116_d447322209', 'native_current_method_graph_dialogue_0111_cd5b8ff3f7', 'native_current_method_graph_dialogue_0023_9d9c3c3ef7', 'native_current_method_graph_dialogue_0046_1c7cc5d04a', 'native_current_method_graph_dialogue_0038_7320227b8c', 'native_current_method_graph_dialogue_0033_780fd675ab']

## Recommendation
- heldout set is only 15 rows, so k1 is high-variance, but the observed heldout k1 is still below the acceptance gate
- candidate_pool delta is positive, so the training recipe is not globally broken
- heldout degradation/zero buckets show insufficient generalization on heldout frontier, especially the listed templates/skills
- ordering should remain frozen/capped; it appears only in clean replay for this train_mix and remains known-hard from mining

Suggested next guarded plan:
- Keep the current failed stage2 model as diagnostic only, not accepted
- Do not train on frontier_heldout
- Mine a small repair/top-up pool against stage1 accepted solver, with ordering bonus frozen and focus on failed hard_single_choice templates plus multi_select conditional_plan h0/h1
- Rebuild a larger stratified train_mix with heldout >=30 before another stage2 training attempt
- Optionally lower epochs or add stronger clean/frontier replay balance if new diagnosis still shows overfit

## Files
- json: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/04_stage2_train_from_stage1/diagnostics/heldout_failure_diagnosis.json`
- rows: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/04_stage2_train_from_stage1/diagnostics/heldout_failure_rows.jsonl`
