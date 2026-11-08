# Two-root × three-arm Miner-only diagnostic

Fixed roots: 198480 and 122711. A_OFF has no active cards; B_OBSERVATION_ONLY receives the frozen eligible cards. Both output only original intents. C_P2_2_TRANSFER receives exactly B's cards and reuses the v2 binding schema/instructions. All arms use the same formal `generate_root_intents`, canonicalizer, original evidence contract and v2 deterministic derivation. No shared source changes or new evidence selection strategy are introduced.

The single common sentence fix is versioned as `h4_r02_miner_explanation_scope_fix_v1`. Prompt-pairing checks prove A/B differ only in active payload, B/C only in transfer schema/instructions, and C/archived-v2 only in the declared sentence. The full transcript, history projection and fixed policy are preserved. Arm identities and schedule are not sent to the model.

Each root/arm owns its original request ID and isolated ledger under `arms/<arm>/qa/shards/shard-00/roots/<root>/`. Global identity is `(arm_id, root_id, request_id)` and token accounting uses the corresponding namespaced ID. Schedule is fixed: 198480 A/B/C, then 122711 C/B/A. Limits are 1/group, 3/root, 2/arm, 6 total logical calls including failures. Exact cache and known failure replay never resend; unknown reservations stop paid work. Original HTTP retries are retained and recorded separately. No other role is scheduled.

```bash
cd /root/autodl-tmp/gxy-selfplay/current_method_github
/root/miniconda3/bin/python -B -m scripts.current_method.restart_n5_spell.h4_r02_miner_three_arm_diagnostic_v1.run --phase prepare --run /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_r02_miner_three_arm_diagnostic_v1_live_1
/root/miniconda3/bin/python -B -m scripts.current_method.restart_n5_spell.h4_r02_miner_three_arm_diagnostic_v1.run --phase live --run /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_r02_miner_three_arm_diagnostic_v1_live_1
/root/miniconda3/bin/python -B -m scripts.current_method.restart_n5_spell.h4_r02_miner_three_arm_diagnostic_v1.run --phase report --run /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_r02_miner_three_arm_diagnostic_v1_live_1
```

Prepare requires prechange workspace/code/history inventories and review policy, copies exact V2 inputs and token evidence, freezes six prompts/pairings/schedule, runs all relevant offline tests with network and bytecode writes disabled, and freezes all request dependencies. Live checks frozen identity before every send. Only this fixed run directory is accepted; conflicting state is not relocated to a fresh experiment.

After live, first author all stage-1 evidence judgments in `reports/intent_evidence_review.jsonl`, then freeze them with `reporting.freeze_intent_review`. Only base-PASS C intents may receive authored stage-2 judgments in `reports/transfer_judgments.jsonl`. `reporting.final_report` retains missing/invalid/extra outputs, computes declared denominators and paired root-level deltas, and never infers semantic judgments from text. Review is attributed to Codex direct transcript review, not independent blind or human review. Width metrics use frozen scene/session mappings; event count stays null without a reliable event map. No width cap or semantic admission is added.

Stop after six calls and the report. Writer/repair/Solver/Memory/training remain zero even if findings are favorable.
