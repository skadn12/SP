# Stage1 Clean Replay Regression Error Analysis

- created_at: 2026-07-03 21:17:58
- evaluated set: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/01_mining500_train/train_mix/clean_replay.jsonl`
- stage2 accepted eval: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/training/eval/stage1_clean_replay_regression_guard/generation/verify_k5_eval.jsonl`
- stage1 reference eval: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/01_mining500_train/training/eval/clean_replay_recovery/generation/verify_k5_eval.jsonl`
- k1_wrong_count: 3 / 30
- qtype_distribution: `{'single_choice': 3}`
- attack_skill_distribution: `{'conditional_plan': 2, 'delayed_activation': 1}`
- hard_single_choice_template_distribution: `{'override_sensitive': 1, 'sequence_as_single_choice': 2}`
- answer_format_issue: false
- automated_bad_question_or_ambiguous: false

## Findings

- All 3 k1 misses are `single_choice` / `hard_single_choice`; none are multi-select or ordering k1 failures.
- Two are `sequence_as_single_choice`; one is `override_sensitive`.
- All wrong first predictions were parsed cleanly as option letters, so this is not a parser or answer-format failure.
- Source validation passes on all 3 rows with no warnings, and guard summary has `bad_question_count=0`.
- One row is clearly inherited from Stage1, one is mostly sampling noise/soft improvement, and one is the likely Stage2 repair preference-migration case.

## Error Rows

| sample_id | qtype/skill/template | gold | stage2 pred0 | stage2 bucket | stage1 pred0 | stage1 bucket | diagnosis |
|---|---|---|---|---|---|---|---|
| native_current_method_graph_dialogue_0071_f938eeb78d | single_choice/conditional_plan/override_sensitive | ['C'] | A | 1/5 | A | 1/5 | Not Stage2-specific: stage1 accepted also failed this sample 1/5. Valid parsed letters; likely brittle hard single-choice clean replay item rather than repair-induced regression. |
| native_current_method_graph_dialogue_0377_2225dc3aa9 | single_choice/conditional_plan/sequence_as_single_choice | ['C'] | D | 4/5 | C | 3/5 | Weak/no migration evidence: stage2 k1 sampled the distractor D, but k5 improved from stage1 3/5 to stage2 4/5. The option texts contain repeated temporal item labels, making this sequence item brittle though validator passes. |
| native_current_method_graph_dialogue_0375_d51d9075cd | single_choice/delayed_activation/sequence_as_single_choice | ['A'] | C | 1/5 | A | 5/5 | Strongest migration signal: stage1 was 5/5 on A, stage2 became 1/5 favoring C. Same hard single-choice sequence template family targeted by repair mining; likely decision-boundary/preference shift from Stage2 repair. |

## Recommendation

- Do not treat this as a format/parser blocker.
- Before Stage3, consider adding a small retention guard or replay emphasis for hard single-choice `sequence_as_single_choice`, especially delayed_activation cases like `native_current_method_graph_dialogue_0375_d51d9075cd`.
- The Stage1 clean replay k1=0.90 should remain a retention risk note, but only one of the three errors looks like a strong Stage2-induced regression.
