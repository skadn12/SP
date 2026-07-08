# Stage2 Smoke50 with Stage1 Solver

## Scope

This is a smoke-only stage2 check for `base_restart_current_method_validation`. It uses the accepted stage1 from-base solver as the fixed solver and the validation seed parents from chronological LoCoMo chunks. No mining, no training, no train mix construction, and no next round were run.

- stage1 solver path: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/accepted_models/solver_base_restart_stage1_merged`
- stage1 solver realpath: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/01_mining500_train/training/models/merged`
- seed parent path: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/seed_parents/seed_parents.jsonl`
- tokenizer: `/root/autodl-tmp/Qwen3-4B`
- max_model_len/max_length: `16384`
- generation backend: `native_current_method`
- native challenger mode: `graph_conditioned_api_dialogue`
- evidence_window_used_as_prompt: `false`
- r4/r5/r1 data or model used: `false`

## Curriculum

- target: ordinary_single_choice=0.05, hard_single_choice=0.60, multi_select=0.25, ordering=0.10
- actual qtype_distribution: `{"multi_select": 13, "ordering": 5, "single_choice": 32}`
- actual difficulty_subtype_distribution: `{"hard_single_choice": 29, "ordinary": 18, "ordinary_single_choice": 3}`
- hard_single_choice_template_distribution: `{"guaranteed_vs_possible": 5, "latest_valid_state": 5, "minimal_difference": 7, "override_sensitive": 5, "sequence_as_single_choice": 7}`

## Quality Gates

| metric | value |
| --- | ---: |
| generated_count | 50 |
| valid_count | 50 |
| clean_count | 50 |
| valid_rate | 1.000000 |
| clean_rate | 1.000000 |
| rows_below_4000_delta | 0 |
| anchor_leakage_count | 0 |
| solver_visible_meta_leak_count | 0 |
| eval_context_overflow_count | 0 |
| truncated_count | 0 |
| bad_question_count | 9 |
| empty_prediction_count | 0 |
| parsed_failure_count | 0 |
| prompt_token_mean | 12400.0 |
| prompt_token_max | 12788 |
| final_dialogue_token_mean | 12155.92 |
| final_dialogue_token_max | 12208 |

## Frontier Metrics

| metric | value |
| --- | ---: |
| frontier_count | 4 |
| near_miss_count | 19 |
| frontier_plus_near_miss_count | 23 |
| strict_frontier_count | 4 |
| boundary_frontier_count | 16 |
| frontier_pool_count | 17 |
| boundary_plus_near_miss_count | 17 |
| overall_zero_share | 0.340000 |
| hard_single_choice_zero_share | 0.310345 |
| multi_select_zero_share | 0.307692 |
| ordering_zero_share | 0.800000 |

qtype_bucket_distribution:

```json
{
  "multi_select": {
    "0/5": 4,
    "1/5": 2,
    "3/5": 2,
    "5/5": 5
  },
  "ordering": {
    "0/5": 4,
    "2/5": 1
  },
  "single_choice": {
    "0/5": 9,
    "1/5": 3,
    "2/5": 5,
    "3/5": 2,
    "4/5": 1,
    "5/5": 12
  }
}
```

soft_buckets:

```json
{
  "0/5": 17,
  "1/5": 5,
  "2/5": 6,
  "3/5": 4,
  "4/5": 1,
  "5/5": 17
}
```

## API And Resume

- API failure/retry count: `13`
- API failure reasons: `{"NativeGraphDialogueReject": 1, "base_validation_anti_shortcut": 2, "meta_leak": 10}`
- API timeout count: `2`
- generation proposal parse failures/retries: `34`
- resume_count: `0`

## Decision

- quality_gate_pass: `true`
- frontier_gate_pass: `true` (frontier_plus_near_miss_count>=10)
- stage2_smoke_pass: `true`
- recommend_stage2_mining: `manual_consider_guarded`

Ordering is currently very hard for the stage1 solver (`ordering_zero_share=0.800000`). Do not train immediately on this smoke result; if humans approve stage2 mining, keep ordering capped/guarded and inspect the generated ordering cases.

Confirmed: no automatic mining, no training, and no next round were run.
