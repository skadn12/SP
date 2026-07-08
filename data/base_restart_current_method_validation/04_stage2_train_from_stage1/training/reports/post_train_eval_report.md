# Stage2 Train From Stage1 Post-Train Eval

- status: failed
- stage2_success: False
- init/fixed previous solver: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/accepted_models/solver_base_restart_stage1_merged`
- trained merged model: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/04_stage2_train_from_stage1/training/models/merged`
- tokenizer: `/root/autodl-tmp/Qwen3-4B`
- max_model_len: 16384
- evidence_window_used_as_prompt: false

## Train Mix Audit
- frontier_pool/frontier_train/frontier_heldout: 73/58/15
- clean_replay/train_mix/near_miss_aux: 23/81/101
- heldout_overlap_total: 0
- max_sft_token_length: 12882
- over_max_length_count: 0

## Eval Metrics
- train frontier k1/k5_soft: 0.913793 / 0.9
- heldout frontier k1/k5_soft: 0.333333 / 0.48
- clean replay k1/k5_soft: 1.0 / 0.982609
- candidate_pool post k1/k5_soft: 0.626667 / 0.63
- candidate_pool stage1 baseline k1/k5_soft: 0.523333 / 0.524667
- candidate_pool delta k1: 0.103334
- phase guard k1/k5_soft: 0.772727 / 0.709091
- overflow/truncated: 0 / 0
- empty_prediction_count / parsed_failure_count: 0 / 0
- aggregate bad_question_count: 116

## Gate Decision
- FAIL: stage2 solver is not accepted.
- failure_reasons: ['heldout frontier recovery below 0.50']
- accepted solver symlink was not created.
- Stop confirmation: no automatic stage3.
