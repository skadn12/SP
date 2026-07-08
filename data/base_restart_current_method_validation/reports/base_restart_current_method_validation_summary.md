# base_restart_current_method_validation Summary

Updated: 2026-07-01T12:46:30Z

## 1. Experiment Goal
Validate whether the existing current_method pipeline can restart from base Qwen3-4B, rebuild LoCoMo raw seed graph/state-chain parents, generate/evaluate/mine/train/evaluate one first-stage solver round.

## 2. Scope Assertions
- This is from-base current_method validation.
- This is not r5.
- This is not r1 legacy.
- This is not r4 continuation.
- No r4 parent, r4 mining/train data, r4 adapter/merged model, probe197 solver, or r1/r2 solver was used for training/init.
- No evidence-window prompt was used; prompts are full-dialogue current_method MCQ format.
- Base/init/eval tokenizer: `/root/autodl-tmp/Qwen3-4B`; max_length/max_model_len: 16384.

## 3. LoCoMo Raw Path
- raw_path: /root/autodl-tmp/gxy-selfplay/locomo10.json
- raw sample_count: 10
- sample0-5 usable: [True, True, True, True, True, True]
- contains speaker: True
- contains turn id: True
- contains session/time info: True

## 4. Seed Adapter Result
- adapter_status: success
- seed parent count: 16
- raw sample count used: 6
- chunk strategy: locomo_session_chronological_contiguous_context_budget
- chunk token target: 6000-7000 hard_max=7500
- output: /root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/seed_parents/seed_parents.jsonl

## 5. 00_smoke50_check Result
- generated/valid/clean: 50/50/50
- valid_rate / clean_rate: 100.0% / 100.0%
- frontier_plus_near_miss_count: 17
- rows_below_4000_delta: 0
- anchor_leakage_count: 0
- solver_visible_meta_leak_count: 0
- status: passed

## 6. 01_mining500_train Mining Result
- original mining generated/valid/clean: 500/500/500
- original frontier_pool_count: 66
- original status: stopped before training because frontier_pool_count < 80
- top-up candidate_count: 400
- top-up frontier_count: 40
- combined_candidate_count: 900
- combined_frontier_pool_count: 106
- combined_frontier qtype distribution: {'multi_select': 23, 'ordering': 8, 'single_choice': 75}
- combined_frontier label distribution: {'boundary_frontier': 83, 'near_miss': 23}
- top-up length/leakage: rows_below_4000_delta=0, anchor=0, meta=0
- status: top-up objective passed; combined frontier exceeded training gate

## 7. Train Mix Result
- status: audit_passed
- combined_frontier_pool_count: 106
- frontier_train_count: 84
- frontier_heldout_count: 22
- clean_replay_count: 30
- train_mix_count: 114
- clean_replay_share: 0.263158
- heldout_overlap_check: {'dialogue_hash_overlap': 0, 'question_id_overlap': 0, 'sample_id_overlap': 0, 'total_overlap': 0}
- max_sft_token_length: 12873
- over_max_length_count: 0
- evidence_window_used_as_prompt: False

## 8. Training Result
- status: success
- init_model: /root/autodl-tmp/Qwen3-4B
- tokenizer_path: /root/autodl-tmp/Qwen3-4B
- max_length: 16384
- train_mix_count: 114
- global_step: 57
- trainable_parameters: 33030144
- adapter_dir: /root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/01_mining500_train/training/models/adapter
- merged_dir: /root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/01_mining500_train/training/models/merged

## 9. Post-Train Eval Result
- train frontier k1 / k5_soft: 0.821429 / 0.804762
- heldout frontier k1 / k5_soft: 0.681818 / 0.618182
- clean replay k1 / k5_soft: 0.966667 / 0.953333
- combined candidate post k1 / k5_soft: 0.573333 / 0.568889
- combined candidate base baseline k1 / k5_soft: 0.464444 / 0.463333
- candidate k1 delta post-base: 0.108889
- phase guard: {'used': True, 'name': 'phase19_r3_single_choice_guard', 'samples_path': '/root/autodl-tmp/gxy-selfplay/current_method_runs/current_method_r4_from_probe197_mining500_train_v1/eval_sets/phase19_r3_small.jsonl', 'read_only_eval_guard': True, 'k1': 0.48, 'k5_soft': 0.452, 'n': 50}
- bad_question_count: 309
- empty_prediction_count: 0
- parsed_failure_count: 0
- overflow_count: 0
- truncated_count: 0
- max_model_len: 16384
- tokenizer_path: /root/autodl-tmp/Qwen3-4B

## 10. First-Round Validation Decision
Passed. The first from-base current_method training validation succeeded against the requested gates: train k1 >= 0.80, heldout k1 >= 0.50, clean replay k1 >= 0.95, candidate-pool post k1 clearly above base baseline, and overflow/truncated counts are zero.

## 11. Next Round Recommendation
Recommend entering a next round only after manual review of this report and accepted model. This run stopped after stage1 acceptance and did not auto-run another mining/training round.

## 12. Blockers / Risks
- No blocking failure remains for stage1 validation.
- Original mining500 frontier count was only 66, so the accepted run depends on the hard_single_choice top-up to reach 106 combined frontier rows.
- Clean replay ordering coverage was scarce: only 2 ordering 5/5 clean candidates were available and selected.
- Phase19 single_choice guard is usable but modest (k1=0.48, k5_soft=0.452); treat it as a format sanity guard, not as evidence of broad phase19 recovery.
- Candidate-pool bad_question_count remains high in diagnostic categorization because the pool intentionally includes difficult frontier and failed-base items; this did not trip length or parser gates.

## 13. Key Files
- raw_schema_summary: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/raw_audit/raw_schema_summary.json`
- seed_parents: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/seed_parents/seed_parents.jsonl`
- preprocess_summary: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/preprocess_locomo_current_method/preprocess_summary.json`
- smoke_report: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/00_smoke50_check/reports/report.md`
- smoke_summary: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/00_smoke50_check/adaptive_smoke_summary.json`
- combined_candidate_pool: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/01_mining500_train/mining/topup_combined_candidate_pool.jsonl`
- combined_frontier_pool: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/01_mining500_train/mining/topup_combined_frontier_pool.jsonl`
- topup_combined_summary: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/01_mining500_train/mining/topup_combined_summary.json`
- train_mix_report: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/01_mining500_train/train_mix/reports/train_mix_report.md`
- train_mix_summary: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/01_mining500_train/train_mix/train_mix_summary.json`
- sft_train: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/01_mining500_train/train_mix/sft_train.jsonl`
- adapter: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/01_mining500_train/training/models/adapter`
- merged: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/01_mining500_train/training/models/merged`
- training_summary: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/01_mining500_train/training/reports/training_summary.json`
- post_train_eval_summary: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/01_mining500_train/training/reports/post_train_eval_summary.json`
- accepted_symlink: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/accepted_models/solver_base_restart_stage1_merged`
- accepted target resolved: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/01_mining500_train/training/models/merged`

## Stage2 Smoke50 with Stage1 Solver

1. Stage2 target: smoke-only check that `graph_conditioned_api_dialogue` can still find new frontier or near-miss examples against the accepted stage1 from-base solver. No mining or training was requested or run.
2. Fixed solver path: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/accepted_models/solver_base_restart_stage1_merged` (resolved to `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/01_mining500_train/training/models/merged`).
3. Seed parent path: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/seed_parents/seed_parents.jsonl`. These are the validation LoCoMo raw chronological context-budgeted full-dialogue chunk parents, not evidence windows.
4. Qtype curriculum target: ordinary_single_choice=0.05, hard_single_choice=0.60, multi_select=0.25, ordering=0.10. Actual qtype distribution: `{"multi_select": 13, "ordering": 5, "single_choice": 32}`.
5. Generated/valid/clean: 50/50/50 with valid_rate=1.000000, clean_rate=1.000000.
6. Length/leakage gate: rows_below_4000_delta=0, anchor_leakage_count=0, solver_visible_meta_leak_count=0, eval_context_overflow_count=0, truncated_count=0.
7. Frontier/near-miss: frontier_count=4, strict_frontier_count=4, boundary_frontier_count=16, near_miss_count=19, frontier_plus_near_miss_count=23.
8. Qtype bucket distribution: `{"multi_select": {"0/5": 4, "1/5": 2, "3/5": 2, "5/5": 5}, "ordering": {"0/5": 4, "2/5": 1}, "single_choice": {"0/5": 9, "1/5": 3, "2/5": 5, "3/5": 2, "4/5": 1, "5/5": 12}}`. Zero shares: overall=0.340000, hard_single_choice=0.310345, multi_select=0.307692, ordering=0.800000.
9. Smoke decision: passed. Quality gate passed and frontier gate passed via `frontier_plus_near_miss_count>=10`.
10. Stage2 mining recommendation: `manual_consider_guarded`. The smoke suggests challenger can still produce useful near-boundary material, but ordering is too hard at this setting (`ordering_zero_share=0.800000`), so ordering should stay capped/guarded if humans approve mining.
11. Blockers / risks: no schema, length, leakage, parser, overflow, or truncation blocker. Main risk is ordering difficulty; do not train directly from this smoke result.
12. Stop confirmation: no automatic mining, no training, no train_mix construction, and no next round were run. Evidence-window prompt used: `false`. r4/r5/r1 data/model used: `false`.

Stage2 key files:
- samples: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/02_stage2_smoke50_with_stage1_solver/samples.jsonl`
- valid_samples: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/02_stage2_smoke50_with_stage1_solver/valid_samples.jsonl`
- clean_samples: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/02_stage2_smoke50_with_stage1_solver/clean_samples.jsonl`
- adaptive_smoke_summary: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/02_stage2_smoke50_with_stage1_solver/adaptive_smoke_summary.json`
- report: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/02_stage2_smoke50_with_stage1_solver/reports/report.md`
- log: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/logs/02_stage2_smoke50_with_stage1_solver.log`

## Stage2 Guarded Mining300

1. Goal: run guarded stage2 mining with the accepted stage1 solver, `candidate_budget=300`, base/bonus plan 240/60, and no training.
2. Fixed solver: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/accepted_models/solver_base_restart_stage1_merged`; tokenizer `/root/autodl-tmp/Qwen3-4B`; max_model_len=16384.
3. Parent source: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/seed_parents/seed_parents.jsonl`; no evidence-window prompt; no r4/r5/r1 data or model used.
4. Status: failed/incomplete due to external graph-expansion API 403. Base240 accepted/evaluated 155 rows; fill85 accepted 0 rows and logged 60 rejects, all `api_graph_403`; bonus60 was not started.
5. Accepted-row quality: valid/clean=155/155, rows_below_4000_delta=0, anchor_leakage_count=0, solver_visible_meta_leak_count=0, eval_context_overflow_count=0, truncated_count=0.
6. Accepted-row mining metrics: frontier_pool_count=37, frontier_count=37, near_miss_count=54, frontier_plus_near_miss_count=91, bad_question_count=31.
7. Qtype distribution: `{"multi_select": 54, "ordering": 8, "single_choice": 93}`. Zero shares: hard_single_choice=0.352941, multi_select=0.444444, ordering=0.875000.
8. Gate decision: not passed. `generated_valid_clean_ge_300_285_285=false`; `frontier_pool_count_ge_60=false`; `frontier_plus_near_miss_count_ge_90=true`; length/leakage/context gates passed.
9. Blockers: API 403 prevented candidate_budget=300; strict frontier/frontier_pool insufficient; near-miss is plentiful but boundary/strict frontier is too small; ordering remains too hard and should stay frozen from bonus.
10. Recommendation: stop here, do not train, do not construct train_mix. Resume guarded mining only after the graph-expansion API 403 issue is resolved.
11. Stop confirmation: no train_mix, no training, no next round.

Stage2 guarded mining files:
- candidate_pool: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/03_stage2_guarded_mining300/candidate_pool.jsonl`
- frontier_pool: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/03_stage2_guarded_mining300/frontier_pool.jsonl`
- adaptive summary: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/03_stage2_guarded_mining300/adaptive_smoke_summary.json`
- report: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/03_stage2_guarded_mining300/reports/report.md`
- base240 report: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/03_stage2_guarded_mining300/base240/reports/report.md`
- fill85 rejects: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/03_stage2_guarded_mining300/base240_fill85/graph_conditioned_api_dialogue_rejects.jsonl`

## Stage2 Guarded Mining300 API 403 Diagnosis

1. Purpose: diagnose whether the `api_graph_403` blocker recovered before resuming guarded mining.
2. Non-sensitive config: base_url=`https://api.siliconflow.cn/v1`, model=`Qwen/Qwen3-14B`, api_key_present=`true`, api_key_length=51; key was not printed.
3. `/models` probe: status=200, configured_model_listed=`true`.
4. Minimal graph-style chat completion: status=403, response=`{"code":30001,"message":"Sorry, your account balance is insufficient","data":null}`.
5. Diagnosis: external API quota/balance blocker. The key can access model metadata and the configured model is listed, but chat completion is blocked before generation.
6. Decision: stop. Guarded mining was not resumed; bonus60 was not started; no train_mix, no training, and no next round.
7. Diagnosis report: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/03_stage2_guarded_mining300/reports/api_403_diagnosis.md`.

## Stage2 Guarded Mining300 Resume

- Goal: resume the incomplete stage2 guarded mining after API 403/balance recovery; mine/evaluate only.
- Fixed solver: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/accepted_models/solver_base_restart_stage1_merged`
- Seed parent path: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/seed_parents/seed_parents.jsonl`
- Curriculum: base240 plus guarded bonus60; ordering bonus frozen; bonus only hard_single_choice/multi_select.
- generated/valid/clean: 300/300/300
- frontier_pool_count: 73
- near_miss_count: 101
- frontier_plus_near_miss_count: 174
- qtype_distribution: {'multi_select': 111, 'ordering': 12, 'single_choice': 177}
- qtype_bucket_distribution: {'multi_select': {'0/5': 49, '1/5': 7, '2/5': 6, '3/5': 4, '4/5': 2, '5/5': 43}, 'ordering': {'0/5': 10, '5/5': 2}, 'single_choice': {'0/5': 50, '1/5': 11, '2/5': 9, '3/5': 21, '4/5': 13, '5/5': 73}}
- rows_below_4000_delta / anchor_leakage / solver_visible_meta_leak / overflow / truncated: 0 / 0 / 0 / 0 / 0
- api_graph_403 prior/after recharge resume: 60 / 0
- Smoke/mining gate: PASS
- Recommendation: 可以人工确认是否进入 stage2 train_mix.
- Confirmation: no train_mix, no training, no automatic next round.
- Key files: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/03_stage2_guarded_mining300/candidate_pool.jsonl`, `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/03_stage2_guarded_mining300/frontier_pool.jsonl`, `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/03_stage2_guarded_mining300/adaptive_smoke_summary.json`, `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/03_stage2_guarded_mining300/reports/report.md`

## Stage2 Train From Stage1

1. Goal: construct audited stage2 train_mix from strict stage2 frontier, train from the accepted stage1 solver, then evaluate without automatically entering stage3.
2. Train init model: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/accepted_models/solver_base_restart_stage1_merged`; tokenizer `/root/autodl-tmp/Qwen3-4B`; max_length/max_model_len=16384.
3. Train mix audit: frontier_pool=73, frontier_train=58, frontier_heldout=15, clean_replay=23, train_mix=81, near_miss_aux_pool=101.
4. Audit gates: heldout_overlap_total=0, max_sft_token_length=12882, over_max_length_count=0, evidence_window_used_as_prompt=false.
5. Training: completed LoRA SFT for 2.0 epochs from stage1 accepted solver, merged successfully to `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/04_stage2_train_from_stage1/training/models/merged`.
6. Post-train eval: train frontier k1/k5_soft=0.913793 / 0.9; heldout frontier k1/k5_soft=0.333333 / 0.48; clean replay k1/k5_soft=1.0 / 0.982609.
7. Candidate pool: stage2 post k1/k5_soft=0.626667 / 0.63; stage1 baseline k1/k5_soft=0.523333 / 0.524667; delta_k1=0.103334.
8. Format/phase guard: validation stage1 frontier-heldout guard k1/k5_soft=0.772727 / 0.709091.
9. Length/parser gates: overflow_count=0, truncated_count=0, empty_prediction_count=0, parsed_failure_count=0.
10. Stage2 decision: FAIL. Failure reasons: ['heldout frontier recovery below 0.50'].
11. Accepted stage2 solver: not created because gate failed.
12. Stop confirmation: no automatic stage3, no extra mining, no evidence-window prompt, no r4/r5/r1 data/model, and no current_method_runs mainline output used.

Stage2 train/eval key files:
- train_mix_summary: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/04_stage2_train_from_stage1/train_mix/train_mix_summary.json`
- train_mix_report: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/04_stage2_train_from_stage1/train_mix/reports/train_mix_report.md`
- adapter: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/04_stage2_train_from_stage1/training/models/adapter`
- merged: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/04_stage2_train_from_stage1/training/models/merged`
- training_summary: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/04_stage2_train_from_stage1/training/reports/training_summary.json`
- post_train_eval_summary: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/04_stage2_train_from_stage1/training/reports/post_train_eval_summary.json`
- post_train_eval_report: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/04_stage2_train_from_stage1/training/reports/post_train_eval_report.md`

## Stage2 Heldout Failure Diagnosis

- Purpose: diagnosis-only follow-up after stage2 train gate failure; no mining, no training, no train_mix rebuild, and no stage3 were run.
- Heldout post-train k1/k5_soft: 0.333333 / 0.48 on n=15.
- Same heldout stage1 mining k1 -> stage2 post k1: 0.733333 -> 0.333333.
- Same heldout mean exact-count stage1 -> stage2: 2.466667 -> 2.4.
- Improved/same/degraded by exact-count: 7/0/8; k1 gained/lost: 2/8.
- Soft-good-but-k1-bad rows (>=4/5 but first prediction wrong): 2; zero-bucket rows: 5.
- Main interpretation: candidate_pool improved globally, but heldout frontier generalization is unstable and below gate; stage2 model remains diagnostic only, not accepted.
- Recommended next action: do not retrain the same mix and do not train on heldout. If continuing, run a human-approved guarded repair/top-up mining focused on failed hard_single_choice templates and multi_select conditional_plan h0/h1, with ordering still frozen/capped, then rebuild a larger stratified train_mix with heldout >=30.
- Files: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/04_stage2_train_from_stage1/diagnostics/heldout_failure_diagnosis.json`, `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/04_stage2_train_from_stage1/diagnostics/heldout_failure_diagnosis.md`, `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/04_stage2_train_from_stage1/diagnostics/heldout_failure_rows.jsonl`.

## Stage2 Repair Top-up Training from Stage1 Solver

1. 目标：修复 `04_stage2_train_from_stage1` 的 heldout frontier recovery 失败，只使用 stage1 accepted solver 初始化，先做 guarded repair/top-up mining，再构造更大的 heldout train_mix 并训练。
2. 明确边界：这是 `base_restart_current_method_validation` 的 stage2 repair；不是 r5；不是 r4 continuation；不是 r1 legacy；没有使用 evidence-window prompt；没有自动 stage3。
3. repair/top-up mining：repair119 strict frontier=15，boundary top-up 80/80 valid/clean 且 strict frontier=20；与原始 stage2 frontier 73 合并后 strict frontier=108。
4. train_mix 审计：frontier_train=78，frontier_heldout=30，clean_replay=30，train_mix=108，prior_failed_heldout_in_train=0，heldout_overlap=0，max_sft_token_length=12950，over_max_length_count=0。
5. 训练：init model=`/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/accepted_models/solver_base_restart_stage1_merged`，merged model=`/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/training/models/merged`，tokenizer=`/root/autodl-tmp/Qwen3-4B`，max_length/max_model_len=16384。
6. post-train eval：train frontier k1/k5_soft=0.846154 / 0.828205；heldout frontier k1/k5_soft=0.533333 / 0.526667；clean replay k1/k5_soft=1.0 / 1.0。
7. candidate_pool 对比：repair-stage2 post k1/k5_soft=0.587174 / 0.587575；stage1 baseline k1/k5_soft=0.496994 / 0.503006；delta k1=0.09018。
8. length/parser gate：overflow_count=0，truncated_count=0，empty_prediction_count=0，parsed_failure_count=0。
9. format guard：stage1 frontier heldout guard k1/k5_soft=0.636364 / 0.663636。
10. 结论：stage2 repair 通过；accepted solver symlink=`/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/accepted_models/solver_base_restart_stage2_merged`。建议停止在此，人工确认是否进入 stage3 smoke/mining；本次没有自动启动 stage3。

关键文件：
- train_mix summary: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/train_mix/train_mix_summary.json`
- training summary: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/training/reports/training_summary.json`
- post-train eval summary: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/training/reports/post_train_eval_summary.json`
- post-train eval report: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/training/reports/post_train_eval_report.md`
- accepted record: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/accepted_models/solver_base_restart_stage2_accepted_record.json`

## Stage3 Smoke50 with Stage2 Accepted Solver

1. stage3 目标：人工批准后，只验证 frozen Stage2 accepted solver 面对 current_method challenger 是否仍有 frontier / near-miss；不做 mining、train_mix、training 或下一轮。
2. fixed solver: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/accepted_models/solver_base_restart_stage2_merged`
3. seed parent: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/seed_parents/seed_parents.jsonl`
4. tokenizer/max_model_len: `/root/autodl-tmp/Qwen3-4B` / `16384`
5. generation/eval: generated/valid/clean = 50/50/50; k1=0.56, k5_soft=0.536
6. actual qtype distribution: `{'multi_select': 12, 'single_choice': 33, 'ordering': 5}`; bucket distribution: `{'multi_select': {'0/5': 8, '2/5': 1, '5/5': 3}, 'single_choice': {'3/5': 3, '5/5': 16, '2/5': 2, '0/5': 6, '4/5': 5, '1/5': 1}, 'ordering': {'0/5': 4, '3/5': 1}}`
7. length/leakage gate: rows_below_4000_delta=0, anchor_leakage_count=0, solver_visible_meta_leak_count=0, eval_context_overflow_count=0, truncated_count=0
8. frontier signal: frontier_count=4, near_miss_count=17, frontier_plus_near_miss_count=21, strict_frontier_count=4, boundary_frontier_count=13
9. zero shares: overall=0.36, hard_single_choice=0.2, multi_select=0.666667, ordering=0.8
10. sequence_as_single_choice / delayed_activation clean retention guard: sequence_all k1/k5=0.833333/0.833333; exact delayed_activation k1/k5=0.666667/0.666667; known migration sample `native_current_method_graph_dialogue_0375_d51d9075cd` remains 0/5 with pred0=C vs gold=['A'].
11. smoke decision: stage3_smoke_pass=true; recommend_stage3_mining_manual_only=true.
12. risk/blocker: retention_guard_pass=false, ordering_zero_share remains high at 0.8, multi_select_zero_share=0.666667; any Stage3 mining should keep ordering guarded and include clean-retention protection.
13. execution note: final smoke50 was assembled as 49 main samples plus 1 clean fill-one sample due a last-slot easy_v2 quota/resume cap edge; duplicate_sample_id_count=0.
14. confirmation: evidence_window_used_as_prompt=false; r4/r5/r1 data/model used=false; no mining, no train_mix, no training, no Stage4/next round started.

Key files:
- Stage3 run: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/06_stage3_smoke50_with_stage2_solver`
- adaptive summary: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/06_stage3_smoke50_with_stage2_solver/adaptive_smoke_summary.json`
- report: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/06_stage3_smoke50_with_stage2_solver/reports/report.md`
- fill audit: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/06_stage3_smoke50_with_stage2_solver/reports/stage3_smoke50_fill_combination_audit.json`
- retention report: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/06_stage3_smoke50_with_stage2_solver/retention_guards/sequence_delayed_activation_clean_retention_report.md`

## Stage3 Guarded Mining300 with Stage2 Accepted Solver

1. 目标：在人工批准后，只用 frozen Stage2 accepted solver 进行 guarded mining300 生成与评估探针；不构造 train_mix、不训练、不自动进入下一轮。
2. fixed solver: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/accepted_models/solver_base_restart_stage2_merged`
3. seed parent: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/seed_parents/seed_parents.jsonl`
4. tokenizer/max_model_len: `/root/autodl-tmp/Qwen3-4B` / `16384`
5. generation: base240 完成 240/240/240；bonus 主 run 在 59/60 卡住最后 multi_select slot，保留 59/59/59；随后用 `bonus60_fill1` 在 ordering=0、sequence_as_single_choice=0 的 retention guard 下补 1 条 multi_select。最终 generated/valid/clean=300/300/300，duplicate sample_id=0。
6. guard 执行：ordering bonus remained frozen=true；sequence_as_single_choice_bonus_count=0；delayed_activation_bonus_count=3；evidence_window_used_as_prompt=false；r4/r5/r1 data/model used=false。
7. combined eval: k1/k5_soft=0.546667 / 0.563333；soft buckets=`{'0/5': 110, '1/5': 10, '2/5': 9, '3/5': 12, '4/5': 14, '5/5': 145}`。
8. frontier signal: strict frontier_pool_count=6，boundary_frontier_count=45，near_miss_count=75，frontier_plus_near_miss_count=81；frontier_pool qtype distribution=`{'multi_select': 2, 'ordering': 2, 'single_choice': 2}`。
9. qtype distribution: `{'single_choice': 213, 'ordering': 12, 'multi_select': 75}`；zero shares: overall=0.366667，hard_single_choice=0.28，multi_select=0.573333，ordering=0.833333。
10. length/leakage/parser gate: rows_below_4000_delta=0，anchor_leakage_count=0，solver_visible_meta_leak_count=0，eval_context_overflow_count=0，truncated_count=0，empty_prediction_count=0，parsed_failure_count=0；prompt_token_mean/max=12376.82 / 12795。
11. retention guard: sequence_as_single_choice_all k1/k5_soft=0.833333 / 0.833333；exact delayed_activation k1/k5_soft=0.666667 / 0.666667；known migration sample `native_current_method_graph_dialogue_0375_d51d9075cd` remains 0/5 with pred0=C vs gold=['A']。
12. decision: quality gate passed, but strict frontier_pool_count is too low for training; mining_gate_pass=false。Stage3 guarded mining stops here. Do not construct train_mix and do not train automatically.

Stage3 guarded mining key files:
- run root: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/07_stage3_guarded_mining300`
- candidate_pool: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/07_stage3_guarded_mining300/candidate_pool.jsonl`
- frontier_pool: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/07_stage3_guarded_mining300/frontier_pool.jsonl`
- adaptive summary: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/07_stage3_guarded_mining300/adaptive_smoke_summary.json`
- report: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/07_stage3_guarded_mining300/reports/report.md`
- retention report: `/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/07_stage3_guarded_mining300/retention_guards/sequence_delayed_activation_clean_retention_report.md`
