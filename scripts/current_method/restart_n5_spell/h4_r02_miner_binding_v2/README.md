# P2-2 Miner Binding v2

Formal shared `generation.generate_root_intents` is used by both the original QA entry and this Miner-only runner. The latter returns normally after the shared Miner phase. Original Writer/repair admission, quotas, source eligibility and schema-only common Miner revision remain unchanged.

The v2 binding has exactly five fields. `change_from_source` is a three-field object. No binding `evidence_refs` or `target_distinction` is accepted. `derive_intent_evidence` copies precisely the original validated intent chain, in frozen transcript order. The reusable function does no semantic search, repair, clipping or admission. Guided previews use the same derived sidecar and are `NOT_SENT_TO_WRITER`. Free's future projection excludes binding details and retains the full transcript.

Fixed run: 198480, 122711, 197635, serial; one Miner each, three logical calls including failures. Completed exact cache and observed failures are replayed without sends; unknown reservations stop paid work. Original HTTP 429 retry limit 2 is retained and accounted separately. No Writer, repair, Solver, Memory, training or extra review-model calls are permitted.

```bash
cd /root/autodl-tmp/gxy-selfplay/current_method_github
/root/miniconda3/bin/python -m scripts.current_method.restart_n5_spell.h4_r02_miner_binding_v2.run --phase prepare --run /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_r02_miner_binding_v2_acceptance_v1_live_1
/root/miniconda3/bin/python -m scripts.current_method.restart_n5_spell.h4_r02_miner_binding_v2.run --phase live --run /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_r02_miner_binding_v2_acceptance_v1_live_1
/root/miniconda3/bin/python -m scripts.current_method.restart_n5_spell.h4_r02_miner_binding_v2.run --phase report --run /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_r02_miner_binding_v2_acceptance_v1_live_1
```

Prepare verifies frozen source copies, snapshots code, runs all v1/v2 tests under a network block, verifies tokenizer/service evidence entirely offline, measures exact rendered prompts, and freezes all request dependencies before live. Prechange inventories and source snapshots must be present; existing conflicting run state is rejected rather than moved to a new run. Reporting requires offline authored `reports/miner_binding_content_review.jsonl`, `reports/content_acceptance_judgment.json`, and explanatory `reports/direct_review_narrative.md`. These judgments must identify the reviewer as `Codex direct transcript review`; report code only validates coverage and serializes authored judgments.

No successful readiness verdict schedules Writer. Stop after this report.
