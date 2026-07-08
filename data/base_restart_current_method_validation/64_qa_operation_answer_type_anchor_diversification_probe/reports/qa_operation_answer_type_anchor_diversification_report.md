# QA Operation / Answer Type Anchor Diversification Probe

- Source: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/01_mining500_train/mining/topup_combined_frontier_pool.jsonl`
- Clean samples: 240
- Fixed solver: `solver_base_restart_stage2_accepted`
- Frontier rule: true frontier is k5 bucket 1/5-4/5 only; k1 is diagnostic only.

## Direct Answers

1. QA shell expansion: yes. Generated operations are {'temporal_state_update': 60, 'bridge': 36, 'comparison': 36, 'visibility': 34, 'count': 26, 'cause_consequence': 24, 'relation_lookup': 24}.
2. temporal_state_update share: 25.00%; it is capped as one operation family, not counted as four schemas.
3. Clean-friendly answer types: [('Event', 123), ('YesNo', 38), ('Number', 34), ('ObjectOrCommonNoun', 32), ('DateOrTime', 7), ('Person', 6)].
4. Boundary-rich operation/answer_type pairs: [('comparison|Event', 15), ('temporal_state_update|Event', 11), ('bridge|Event', 9), ('count|Number', 9), ('visibility|YesNo', 4), ('cause_consequence|Event', 3), ('relation_lookup|ObjectOrCommonNoun', 2), ('cause_consequence|ObjectOrCommonNoun', 1)].
5. Too-easy pairs: [('visibility|YesNo', 29), ('relation_lookup|ObjectOrCommonNoun', 16), ('count|Number', 11), ('bridge|Event', 10), ('temporal_state_update|Number', 8), ('temporal_state_update|Event', 7), ('cause_consequence|Event', 7), ('temporal_state_update|ObjectOrCommonNoun', 7)].
6. Too-hard/bad pairs: [('temporal_state_update|Event', 19), ('bridge|Event', 17), ('comparison|Event', 15), ('cause_consequence|Event', 8), ('count|Number', 6), ('visibility|YesNo', 1), ('relation_lookup|ObjectOrCommonNoun', 1), ('temporal_state_update|ObjectOrCommonNoun', 1)]; bad diagnostics are in the summary JSON.
7. Next trainmix plan: worth planning, but not executed here.

## Overall Bucket

```json
{
  "clean_count": 240,
  "k1": 0.616667,
  "k1_correct": 148,
  "k1_seen": 240,
  "k5_soft": 0.600833,
  "0/5": 68,
  "1/5": 16,
  "2/5": 11,
  "3/5": 13,
  "4/5": 16,
  "5/5": 116,
  "boundary": 56,
  "easy": 116,
  "hard0": 68,
  "bad_question": 0,
  "parser": 0,
  "leakage": 0,
  "grounding": 0,
  "overflow": 0,
  "truncated": 0
}
```

## By QA Operation

| bucket | clean_count | k1 | k5_soft | 0/5 | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | boundary | easy | hard0 | bad_question | parser | leakage | grounding | overflow | truncated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temporal_state_update | 60 | 0.6 | 0.593333 | 20 | 0 | 5 | 1 | 5 | 29 | 11 | 29 | 20 | 0 | 0 | 0 | 0 | 0 | 0 |
| bridge | 36 | 0.388889 | 0.388889 | 17 | 3 | 2 | 3 | 1 | 10 | 9 | 10 | 17 | 0 | 0 | 0 | 0 | 0 | 0 |
| comparison | 36 | 0.305556 | 0.305556 | 15 | 8 | 2 | 6 | 0 | 5 | 16 | 5 | 15 | 0 | 0 | 0 | 0 | 0 | 0 |
| visibility | 34 | 0.941176 | 0.929412 | 1 | 1 | 0 | 0 | 3 | 29 | 4 | 29 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| count | 26 | 0.692308 | 0.592308 | 6 | 3 | 2 | 1 | 3 | 11 | 9 | 11 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| cause_consequence | 24 | 0.625 | 0.625 | 8 | 0 | 0 | 1 | 3 | 12 | 4 | 12 | 8 | 0 | 0 | 0 | 0 | 0 | 0 |
| relation_lookup | 24 | 0.916667 | 0.9 | 1 | 1 | 0 | 1 | 1 | 20 | 3 | 20 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |

## By Answer Type

| bucket | clean_count | k1 | k5_soft | 0/5 | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | boundary | easy | hard0 | bad_question | parser | leakage | grounding | overflow | truncated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Event | 123 | 0.357724 | 0.360976 | 59 | 11 | 9 | 9 | 9 | 26 | 38 | 26 | 59 | 0 | 0 | 0 | 0 | 0 | 0 |
| YesNo | 38 | 0.947368 | 0.926316 | 1 | 1 | 0 | 1 | 3 | 32 | 5 | 32 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| Number | 34 | 0.764706 | 0.688235 | 6 | 3 | 2 | 1 | 3 | 19 | 9 | 19 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| ObjectOrCommonNoun | 32 | 0.90625 | 0.8875 | 2 | 1 | 0 | 2 | 0 | 27 | 3 | 27 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| DateOrTime | 7 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Person | 6 | 1 | 0.966667 | 0 | 0 | 0 | 0 | 1 | 5 | 1 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## By Operation Answer Type Pair

| bucket | clean_count | k1 | k5_soft | 0/5 | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | boundary | easy | hard0 | bad_question | parser | leakage | grounding | overflow | truncated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temporal_state_update|Event | 37 | 0.378378 | 0.367568 | 19 | 0 | 5 | 1 | 5 | 7 | 11 | 7 | 19 | 0 | 0 | 0 | 0 | 0 | 0 |
| bridge|Event | 36 | 0.388889 | 0.388889 | 17 | 3 | 2 | 3 | 1 | 10 | 9 | 10 | 17 | 0 | 0 | 0 | 0 | 0 | 0 |
| visibility|YesNo | 34 | 0.941176 | 0.929412 | 1 | 1 | 0 | 0 | 3 | 29 | 4 | 29 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| comparison|Event | 32 | 0.21875 | 0.23125 | 15 | 8 | 2 | 5 | 0 | 2 | 15 | 2 | 15 | 0 | 0 | 0 | 0 | 0 | 0 |
| count|Number | 26 | 0.692308 | 0.592308 | 6 | 3 | 2 | 1 | 3 | 11 | 9 | 11 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| relation_lookup|ObjectOrCommonNoun | 19 | 0.894737 | 0.884211 | 1 | 1 | 0 | 1 | 0 | 16 | 2 | 16 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| cause_consequence|Event | 18 | 0.5 | 0.522222 | 8 | 0 | 0 | 0 | 3 | 7 | 3 | 7 | 8 | 0 | 0 | 0 | 0 | 0 | 0 |
| temporal_state_update|Number | 8 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| temporal_state_update|ObjectOrCommonNoun | 8 | 0.875 | 0.875 | 1 | 0 | 0 | 0 | 0 | 7 | 0 | 7 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| temporal_state_update|DateOrTime | 7 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cause_consequence|ObjectOrCommonNoun | 5 | 1 | 0.92 | 0 | 0 | 0 | 1 | 0 | 4 | 1 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| relation_lookup|Person | 5 | 1 | 0.96 | 0 | 0 | 0 | 0 | 1 | 4 | 1 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| comparison|YesNo | 4 | 1 | 0.9 | 0 | 0 | 0 | 1 | 0 | 3 | 1 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cause_consequence|Person | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## By Parent ID

| bucket | clean_count | k1 | k5_soft | 0/5 | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | boundary | easy | hard0 | bad_question | parser | leakage | grounding | overflow | truncated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| locomo_raw_sample2_chunk000 | 21 | 0.666667 | 0.619048 | 5 | 1 | 2 | 1 | 3 | 9 | 7 | 9 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| locomo_raw_sample5_chunk000 | 19 | 0.736842 | 0.705263 | 2 | 2 | 1 | 3 | 1 | 10 | 7 | 10 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| locomo_raw_sample0_chunk001 | 18 | 0.666667 | 0.655556 | 6 | 0 | 0 | 0 | 1 | 11 | 1 | 11 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| locomo_raw_sample3_chunk002 | 18 | 0.777778 | 0.744444 | 1 | 4 | 0 | 1 | 0 | 12 | 5 | 12 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| locomo_raw_sample4_chunk002 | 17 | 0.588235 | 0.564706 | 6 | 1 | 1 | 0 | 0 | 9 | 2 | 9 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| locomo_raw_sample1_chunk000 | 16 | 0.5625 | 0.6 | 6 | 0 | 0 | 1 | 0 | 9 | 1 | 9 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| locomo_raw_sample2_chunk001 | 16 | 0.875 | 0.8625 | 1 | 0 | 1 | 1 | 1 | 12 | 3 | 12 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| locomo_raw_sample4_chunk000 | 16 | 0.4375 | 0.4875 | 7 | 1 | 0 | 1 | 0 | 7 | 2 | 7 | 7 | 0 | 0 | 0 | 0 | 0 | 0 |
| locomo_raw_sample5_chunk002 | 15 | 0.533333 | 0.453333 | 7 | 1 | 0 | 0 | 2 | 5 | 3 | 5 | 7 | 0 | 0 | 0 | 0 | 0 | 0 |
| locomo_raw_sample0_chunk000 | 14 | 0.642857 | 0.585714 | 5 | 0 | 1 | 0 | 1 | 7 | 2 | 7 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| locomo_raw_sample3_chunk001 | 14 | 0.714286 | 0.7 | 1 | 2 | 0 | 3 | 2 | 6 | 7 | 6 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| locomo_raw_sample1_chunk001 | 13 | 0.461538 | 0.538462 | 4 | 0 | 2 | 1 | 2 | 4 | 5 | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| locomo_raw_sample4_chunk001 | 13 | 0.384615 | 0.307692 | 6 | 2 | 2 | 0 | 1 | 2 | 5 | 2 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| locomo_raw_sample5_chunk001 | 13 | 0.538462 | 0.461538 | 5 | 1 | 1 | 1 | 1 | 4 | 4 | 4 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| locomo_raw_sample3_chunk000 | 9 | 0.444444 | 0.533333 | 4 | 0 | 0 | 0 | 1 | 4 | 1 | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| locomo_raw_sample2_chunk002 | 8 | 0.625 | 0.65 | 2 | 1 | 0 | 0 | 0 | 5 | 1 | 5 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |

## Diagnostics

```json
{
  "qa_operation_counts": {
    "temporal_state_update": 60,
    "bridge": 36,
    "comparison": 36,
    "visibility": 34,
    "count": 26,
    "cause_consequence": 24,
    "relation_lookup": 24
  },
  "qa_subtype_counts": {
    "entity_to_event": 36,
    "earlier_event": 32,
    "entity_relation": 24,
    "final_state": 20,
    "before_after": 19,
    "ordering": 18,
    "scope_person_yesno": 18,
    "next_event_after": 18,
    "scope_yesno": 16,
    "entity_anchor_count": 14,
    "explicit_count_lookup": 12,
    "cause_lookup": 6,
    "before_after_yesno": 4,
    "still_true": 3
  },
  "answer_type_counts": {
    "Event": 123,
    "YesNo": 38,
    "Number": 34,
    "ObjectOrCommonNoun": 32,
    "DateOrTime": 7,
    "Person": 6
  },
  "operation_answer_type_matrix": {
    "temporal_state_update": {
      "Event": 37,
      "Number": 8,
      "ObjectOrCommonNoun": 8,
      "DateOrTime": 7
    },
    "bridge": {
      "Event": 36
    },
    "comparison": {
      "Event": 32,
      "YesNo": 4
    },
    "visibility": {
      "YesNo": 34
    },
    "count": {
      "Number": 26
    },
    "cause_consequence": {
      "Event": 18,
      "ObjectOrCommonNoun": 5,
      "Person": 1
    },
    "relation_lookup": {
      "ObjectOrCommonNoun": 19,
      "Person": 5
    }
  },
  "anchor_query_plan_valid_count": 240,
  "anchor_query_plan_fail_count": 0,
  "missing_anchor_count": 0,
  "unresolved_anchor_count": 0,
  "answer_type_mismatch_count": 0,
  "distractor_answer_type_mismatch_count": 0,
  "duplicate_option_count": 0,
  "near_duplicate_option_count": 0,
  "placeholder_option_count": 0,
  "state_shell_leak_count": 0,
  "final_state_shell_overuse_count": 0,
  "option_surface_risk_count": 0,
  "bad_question_count": 0,
  "parser_failure_count": 0,
  "leakage_count": 0,
  "grounding_failure_count": 0,
  "overflow_count": 0,
  "truncated_count": 0,
  "solver_summary": {
    "method": "current_method",
    "fixed_solver": "solver_base_restart_stage2_accepted",
    "model": "/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/accepted_models/solver_base_restart_stage2_merged",
    "adapter_recorded": null,
    "k1_backend": "vllm",
    "k5_backend": "vllm",
    "eval_context_overflow_count": 0,
    "prompt_debug_files": {
      "k1": [
        "/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/64_qa_operation_answer_type_anchor_diversification_probe/reports/solver_prompt_debug_sample_1.txt",
        "/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/64_qa_operation_answer_type_anchor_diversification_probe/reports/solver_prompt_debug_sample_2.txt",
        "/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/64_qa_operation_answer_type_anchor_diversification_probe/reports/solver_prompt_debug_sample_3.txt"
      ],
      "k5": [
        "/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/64_qa_operation_answer_type_anchor_diversification_probe/reports/solver_prompt_debug_k5/solver_prompt_debug_sample_1.txt",
        "/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/64_qa_operation_answer_type_anchor_diversification_probe/reports/solver_prompt_debug_k5/solver_prompt_debug_sample_2.txt",
        "/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/64_qa_operation_answer_type_anchor_diversification_probe/reports/solver_prompt_debug_k5/solver_prompt_debug_sample_3.txt"
      ]
    },
    "solver_exact_accuracy_k1": 0.616667,
    "solver_soft_accuracy_k5": 0.600833,
    "k1": {
      "n": 240,
      "exact_correct": 148,
      "accuracy": 0.616667,
      "diagnostics": {
        "eval_context_overflow_count": 0,
        "truncated_count": 0,
        "answer_format_instruction_missing_count": 0,
        "prompt_token_estimate_mean": 12309.871,
        "prompt_token_estimate_max": 12415,
        "prompt_char_len_mean": 50801.1,
        "prompt_char_len_max": 52610,
        "max_new_tokens_mean": 8.0,
        "max_new_tokens_max": 8
      },
      "by_question_type": {
        "single_choice": {
          "n": 240,
          "mean_soft_accuracy": 0.616667,
          "mean_exact_count": 0.616667,
          "k": 1
        }
      },
      "by_attack_skill": {
        "temporal_state_update": {
          "n": 60,
          "mean_soft_accuracy": 0.6,
          "mean_exact_count": 0.6,
          "k": 1
        },
        "bridge": {
          "n": 36,
          "mean_soft_accuracy": 0.388889,
          "mean_exact_count": 0.388889,
          "k": 1
        },
        "comparison": {
          "n": 36,
          "mean_soft_accuracy": 0.305556,
          "mean_exact_count": 0.305556,
          "k": 1
        },
        "visibility": {
          "n": 34,
          "mean_soft_accuracy": 0.941176,
          "mean_exact_count": 0.941176,
          "k": 1
        },
        "count": {
          "n": 26,
          "mean_soft_accuracy": 0.692308,
          "mean_exact_count": 0.692308,
          "k": 1
        },
        "cause_consequence": {
          "n": 24,
          "mean_soft_accuracy": 0.625,
          "mean_exact_count": 0.625,
          "k": 1
        },
        "relation_lookup": {
          "n": 24,
          "mean_soft_accuracy": 0.916667,
          "mean_exact_count": 0.916667,
          "k": 1
        }
      }
    },
    "k5": {
      "n": 240,
      "mean_soft_accuracy": 0.600833,
      "soft_buckets": {
        "0/5": 68,
        "1/5": 16,
        "2/5": 11,
        "3/5": 13,
        "4/5": 16,
        "5/5": 116
      },
      "diagnostics": {
        "eval_context_overflow_count": 0,
        "truncated_count": 0,
        "answer_format_instruction_missing_count": 0,
        "prompt_token_estimate_mean": 12309.871,
        "prompt_token_estimate_max": 12415,
        "prompt_char_len_mean": 50801.1,
        "prompt_char_len_max": 52610,
        "max_new_tokens_mean": 8.0,
        "max_new_tokens_max": 8
      },
      "by_question_type": {
        "single_choice": {
          "n": 240,
          "mean_soft_accuracy": 0.600833,
          "mean_exact_count": 3.004167,
          "k": 5
        }
      },
      "by_attack_skill": {
        "temporal_state_update": {
          "n": 60,
          "mean_soft_accuracy": 0.593333,
          "mean_exact_count": 2.966667,
          "k": 5
        },
        "bridge": {
          "n": 36,
          "mean_soft_accuracy": 0.388889,
          "mean_exact_count": 1.944444,
          "k": 5
        },
        "comparison": {
          "n": 36,
          "mean_soft_accuracy": 0.305556,
          "mean_exact_count": 1.527778,
          "k": 5
        },
        "visibility": {
          "n": 34,
          "mean_soft_accuracy": 0.929412,
          "mean_exact_count": 4.647059,
          "k": 5
        },
        "count": {
          "n": 26,
          "mean_soft_accuracy": 0.592308,
          "mean_exact_count": 2.961538,
          "k": 5
        },
        "cause_consequence": {
          "n": 24,
          "mean_soft_accuracy": 0.625,
          "mean_exact_count": 3.125,
          "k": 5
        },
        "relation_lookup": {
          "n": 24,
          "mean_soft_accuracy": 0.9,
          "mean_exact_count": 4.5,
          "k": 5
        }
      }
    }
  },
  "append_existing": true,
  "append_existing_rows": 120,
  "append_new_target": 120,
  "append_new_count": 120,
  "append_existing_identity_keys": 120,
  "append_existing_candidate_skip": 76,
  "resume_target_mode": "new"
}
```
