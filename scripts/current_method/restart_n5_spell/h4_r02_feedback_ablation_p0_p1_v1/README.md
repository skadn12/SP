# Frozen R02 empirical QA feedback P0/P1

This experiment uses six fixed Parent+R01+R02 transcripts, with fresh Miner and original Guided/Free generation in both arms. Only single_choice/multi_select history can supply active feedback. Existing generation builders, normalization, minimum-count admission, repair replacement, permutation and Solver logic are reused. Guided:Free requested ratio remains 1:2. There are no Memory calls, expansion calls, training, P2 bindings or second semantic/dedup gates.

Working directory: `/root/autodl-tmp/gxy-selfplay/current_method_github`.
Python: `/root/miniconda3/bin/python`.
Module prefix: `scripts.current_method.restart_n5_spell.h4_r02_feedback_ablation_p0_p1_v1`.
This run: `/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_r02_feedback_ablation_p0_p1_v1_live_1`.

Reproduction uses a new output directory, or a verifiable continuation of the same frozen run. Never use force to resample completed API calls. `freeze.initial` checks original successful attempts and saves the shared cleaned transcript; `feedback.compile_feedback` verifies source SHA and selects with the original frontier ranking after the ordering mask and four confirmed historical Gold defects. Historical ambiguity remains explicitly recorded for two chosen seeds.

Execution sequence:

1. Save current code/source hashes and the existing dirty working tree status; see this run's `audit/initial_code_identity.json` and `audit/original_code_snapshot`.
2. `python -m <prefix>.freeze --phase initial --run <run>`
3. `python -m <prefix>.feedback --output <run>`
4. `python -m <prefix>.qa_run --phase prepare --run <run>`
5. Run the `test_feedback`, `test_qa_run`, `test_analysis` unittest modules and retain their log. The initial P0 suite contains 23 tests; strengthened denominator and blind-whitelist review brings the final suite to 26. Verify all three model identities and the full 14-source control protocol as recorded in `audit/solver_identity_protocol_p0.json`.
6. `python -m <prefix>.prompt_audit --run <run>`
7. `python -m <prefix>.freeze --phase pairing --run <run>` followed by `python -m <prefix>.preflight --run <run>`.
8. `python -u -m <prefix>.qa_run --phase live --run <run>`; this requires P0 PASS, verifies frozen generation code, and enforces 13 logical calls per root/arm and 156 in total. Original HTTP retries remain owned by the client. Completed calls reuse their exact recorded responses. Unknown orphan reservations stop automatically.
9. `python -m <prefix>.prompt_audit --run <run> --require-complete`; then `python -m <prefix>.analysis --phase review --run <run>` to create the predeclared hash-selected blinded sample.
10. For each arm run the existing `restart-n5-runs/prepare_aligned_r02_solver_eval.py --source <run>/<arm>/qa --output <run>/<arm>/solver --control /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round1_r03_r1_aligned_feedback_r02_solver_eval_live_1`. This freezes the exact candidate-dependent request budget. Do not use the old runner's prepare/all phases.
11. With an idle GPU, run `python -u -m scripts.current_method.restart_n5_spell.h4_round1_r03_difficulty_followup_v1.run --phase live --output <run>/<arm>/solver` sequentially for both arms. It evaluates Base, Clean and Inclusive using the unchanged control protocol and ledger recovery.
12. Finish direct blinded content reading, all 52 old-ordering diagnostics, and `python -m <prefix>.analysis --phase compare --run <run>`. Retain original generated Gold and every program-valid candidate in all results.

Primary descriptive metric is Base full-context candidate accuracy for SC+MS, averaged within root then equally across all six roots. Missing groups are null. Ordering scores are diagnostic only. Full/no context, source, type, candidate/request/root/type macro metrics and paired differences have separate denominators. One generation over six roots gives exploratory evidence only. Negative on-minus-off is not itself proof of useful difficulty or feedback efficacy.

Actual generation seed policy is `not_sent_by_original_client`. Monetary API prices are not supplied by the provider; actual returned token counts, logical calls, transport attempts, timeouts and unknown billing are retained without invented monetary totals.

## Final report and continuation

The additive report validation must be recorded before Solver outcomes exist:
`python -m <prefix>.finalize --phase record-revision --run <run>`.
This saves the actual `analysis.py`, `finalize.py` and associated test identities separately from the original P0 freeze. The current run retains earlier snapshots and records later completion-only supplements under `audit/completion_reporting_supplement*.json`; none changes metrics, generation, selection or scoring.

After QA finishes, `python -m <prefix>.p1 --phase prepare --run <run>` performs the complete staged prompt audit, freezes the 72-case maximum blind sample, invokes the original parameterized Solver adapter for each arm and records the exact candidate-dependent request budget. Then `--phase live` calls only the original runner's live phase sequentially for both arms. Never launch a second copy while one is active.

When all six model evaluations and all selected reviews have finished:

1. `python -m <prefix>.p1 --phase compare --run <run>`
2. `python -m <prefix>.completion_audit --phase final --run <run>`
3. `python -m <prefix>.conclusions --run <run>`
4. `python -m <prefix>.status --run <run>`

The completion audit verifies every reviewed option/item, supported-item references, exact six arm/model coverage and the review/case/mapping hashes used by the comparison. The interpretation reports actual root differences, context dependence, sampled quality, historical-relation correspondence, duplicates and costs. It is descriptive and does not add statistical tests or change candidate admission. P1 ends here.
