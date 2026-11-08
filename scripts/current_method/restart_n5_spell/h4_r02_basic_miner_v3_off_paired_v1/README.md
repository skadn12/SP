# Basic Miner v3 OFF paired diagnostic

The original shared `generate_root_intents` runs four fresh independent requests.
BASE_OFF exactly rebuilds the prior three-arm A_OFF prompt and matches both its
archived policy and sent prompt. V3_OFF inserts only the literal guidance file
before the unique authoritative transcript marker. Both arms retain the original
intents-only schema, empty active cards, safe history, model and 4096 output cap.

The fixed order is 198480 BASE, 198480 V3, 122711 V3, 122711 BASE. One request per
cell, two per root/arm, four total. Failures count. The original client owns its
HTTP retries; no outer retry exists. Other model stages are forbidden.

```bash
cd /root/autodl-tmp/gxy-selfplay/current_method_github
/root/miniconda3/bin/python -B -m scripts.current_method.restart_n5_spell.h4_r02_basic_miner_v3_off_paired_v1.run --phase prepare --run /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_r02_basic_miner_v3_off_paired_v1_live_1
/root/miniconda3/bin/python -B -m scripts.current_method.restart_n5_spell.h4_r02_basic_miner_v3_off_paired_v1.run --phase live --run /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_r02_basic_miner_v3_off_paired_v1_live_1
/root/miniconda3/bin/python -B -m scripts.current_method.restart_n5_spell.h4_r02_basic_miner_v3_off_paired_v1.run --phase report --run /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_r02_basic_miner_v3_off_paired_v1_live_1
```

Before prepare, preserve the initial workspace status/diff, prior source identity
and four history inventories under `audit/`. Prepare copies exact inputs and
freezes prompts, schema, guidance, configuration, review policy, token evidence,
tests and repository dependencies. Dependency discovery crosses recursive AST
imports with loaded modules from the actual four-cell scheduler, shared Miner,
original client/ledger and error paths under blocked network. Python and library
identities are recorded. Every live request rechecks all frozen bytes and loaded
dependencies; an execution guard rejects unlisted repository Python code.

The new JSON Schema audit uses the original frozen schema. It is independent of
the original canonicalizer and does not change admission. Valid original items
remain reviewable even when a response fails the full schema. Evidence extraction
reuses the unchanged original-chain derivation; no selection or repair occurs.

After live, use `reporting.prepare_review_views(run)` and author all original
usable raw positions in `reports/intent_evidence_review.jsonl`. A record has
`root_id`, `arm_id`, zero-based `raw_index`, `intent_id`, `reviewer` (exactly
`Codex offline diagnostic / direct transcript review`), `reason`, `tags`, and six
`dimensions` entries. Each dimension supplies `judgment`, `reason`, and `quotes`
with exact `turn_ref`/`quote`. Quotes outside the selected chain require
`usage=OUTSIDE_CHAIN_CONTEXT_ONLY`; they never augment accepted evidence.
Optional state/time/cause dimensions may be NOT_APPLICABLE with a reason.
Specificity/support/grouping cannot be NA. Aggregation is fixed by review policy.
Missing/structural slots are preserved as unreviewed; extras never replace slots.
Write the attributed interpretation to `reports/diagnostic_narrative.md`, then
run report. Report performs no model calls and cannot treat missing review as PASS.

Resume only this run, with exact frozen identities. Completed cells use matching
checkpoints/cache; known HTTP rejections are never resent. Timeout, ambiguous
service completion, orphan reservation, identity drift or any sticky guardrail
stops further paid work. `post_miner_check` is mandatory after the shared stage:
normal return can conceal a caught Exception. Preserve all states on failure;
never remove a ledger or create a new run to recover budget. Do not edit frozen
conditions after the first live call. After four attempts and report, stop even
if v3 passes. No automatic three-arm, Writer, Solver, Memory or training run.
