# Stage2 -> Stage3 Self-Play Round

This is the production path for continuing current-method self-play from the
accepted Stage2 solver. `Stage64` remains in a few CLI and artifact names for
historical compatibility; the new experiment is the third solver-training
round, so reports and conclusions should call it Stage3.

## Fixed Roles

- Challenger: configured external API model.
- Frontier evaluator: the accepted Stage2 merged checkpoint, fixed across all batches.
- Stage3 training base: the same accepted Stage2 merged checkpoint.
- Stage3 adapter: a new zero-initialized LoRA.

The Stage2 adapter must not be loaded on the raw Qwen base or on the Stage2
merged model. Its recorded base is the Stage1 merged checkpoint.

```text
Stage2 train rows
  -> local evidence-only canonical graph rebuild
  -> fail-closed audit and stage3_parent_v3
  -> semantic_graph_v2 API delta (semantic_graph_v16 cache/gate namespace)
  -> appended dialogue and API QA
  -> semantic/option/decoding gates
  -> all cyclic positions evaluated by fixed Stage2
  -> easy5 harden / hard0 soften+diagnose / position-bias repair
  -> option reshuffle and independent final cyclic K5
  -> robust frontier accumulation
  -> lineage-group split and post-frontier label balance
  -> Stage3 frontier + replay train mix
  -> Stage2 merged + new LoRA -> Stage3 merged
  -> Stage3 vs Stage2 paired evaluation
```

## 1. Rebuild and Audit Stage3 Parents Locally

Use only the Stage2 rows that actually entered training. Historical heldout
files are exact-identity exclusion manifests. Their raw LoCoMo conversation
overlap is audited because the old split was generated-sample-level, not
conversation-level.

The formal path rebuilds graph semantics from evidence with local Qwen3-4B.
It does not call the paid challenger API. The model request contains only
opaque source-event IDs and evidence turn IDs, speakers, and text; old graph
entity/slot/value/status/time labels, QA, options, and gold never enter the
request.

Terminal events receive a deterministic `depends_on` edge to a prior active event only when strict entity and fact-word matching succeeds; otherwise no dependency is inferred. This tightened rule prevents merely similar facts from being mislinked. Long structured parent atoms and accepted `depends_on` edges remain challenger-readable; only parent-dialogue utterances are omitted.

Run the following actual artifact-path example from the main workspace
`/root/autodl-tmp/gxy-selfplay`:

```bash
python scripts/current_method/rebuild_stage2_canonical_graphs.py \
  --input base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/train_mix/frontier_train.jsonl \
  --output-dir base_restart_current_method_validation/172_stage2_canonical_graph_rebuild_v3_semantic_audited_final \
  --model /root/autodl-tmp/Qwen3-4B \
  --max-rows 0 \
  --minimum-events-per-parent 2 \
  --semantic-exclusion-jsonl data/stage2_semantic_audit_exclusions_v1.jsonl \
  --forbidden-jsonl base_restart_current_method_validation/01_mining500_train/train_mix/frontier_heldout.jsonl \
  --forbidden-jsonl base_restart_current_method_validation/04_stage2_train_from_stage1/train_mix/frontier_heldout.jsonl \
  --forbidden-jsonl base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/train_mix/frontier_heldout.jsonl

python scripts/current_method/audit_stage2_canonical_rebuild.py \
  --source base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/train_mix/frontier_train.jsonl \
  --rebuild-dir base_restart_current_method_validation/172_stage2_canonical_graph_rebuild_v3_semantic_audited_final \
  --forbidden-jsonl base_restart_current_method_validation/01_mining500_train/train_mix/frontier_heldout.jsonl \
  --forbidden-jsonl base_restart_current_method_validation/04_stage2_train_from_stage1/train_mix/frontier_heldout.jsonl \
  --forbidden-jsonl base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/train_mix/frontier_heldout.jsonl \
  --output base_restart_current_method_validation/172_stage2_canonical_graph_rebuild_v3_semantic_audited_final/canonical_rebuild_audit.json
```

The formal artifact schema is `stage2_canonical_graph_rebuild_v3`. The current
semantic-audited local artifact contains 77 canonical graphs, 385 canonical
events, and 77 `stage3_parent_v3` rows. Its 171 exclusions include 18 events
quarantined by the independent semantic-audit manifest. Eligible event
rejection, row-atomic rejection, and exact heldout overlap are all zero. One
of the 78 source rows has no clean linked events and is quarantined as a whole;
no event is synthesized from unlinked dialogue (`fallback_event_count=0`).
Fallback is disabled by default, so do not add the compatibility-only
`--allow-dialogue-fallback` flag. This is an offline parent-artifact result
only: `training_authorized=false`, and no Stage3 training may start from it
yet.

The historical `161_stage2_canonical_graph_rebuild_v2_final` directory passed
its older structural audit, but later semantic review found 11 bad events. It
is not a valid Stage3 parent and must not enter graph-only smoke.

`build_stage3_parent_v2.py` remains only for legacy migration and diagnostics.
Removing QA fields from an old row cannot repair polluted graph semantics, so
its `parent_v2` output is not a formal Stage3 input.

## 2. Run Graph-Only API Smoke

Do not spend dialogue, QA or solver tokens until graph generation passes. This
command makes paid API calls and requires explicit acknowledgement.

```bash
python scripts/current_method/stage3_graph_only_smoke.py \
  --input base_restart_current_method_validation/172_stage2_canonical_graph_rebuild_v3_semantic_audited_final/stage3_parent_v3.jsonl \
  --output-dir /path/to/stage3_graph_smoke \
  --n 7 \
  --max-attempts-per-sample 2 \
  --minimum-pass-rate 0.70 \
  --graph-expansion-max-tokens 4096 \
  --graph-semantic-retries 1 \
  --seed 20260713 \
  --confirm-api-calls
```

Every accepted delta must use `semantic_graph_v2`, contain 5-8 semantic events,
pass operation-specific validation, and link `parent_graph_id` to the supplied
parent snapshot. Reject artifacts retain the raw API graph and metadata.

The smoke freezes the complete target identity (operation, answer type,
reasoning subtype/template, hop count, and hard-negative strategy) before each
request. It stops when the requested pass rate is mathematically unreachable;
with `n=7`, two attempts per slot, and one semantic retry, 28 graph
requests is the hard upper bound. No dialogue, QA, or solver call is made by
this gate.

`semantic_graph_v2` is the payload schema. The graph-cache key and gate
namespace are `semantic_graph_v16`; this prevents older graph responses from
being reused after fixed visibility/relation role layouts and parent-safe
validator-guided correction were added on top of the v15 evidence-isolation
contracts.

Formal targets use `stage64_target_identity_v2`. Before any API call, the
target ledger fixes `qa_operation`, a compatible answer type,
`reasoning_subtype`, `reasoning_template`, `hop_count`, and
`hard_negative_strategy`. Exact operation quotas are interleaved round-robin,
so a partial batch does not consist of one contiguous question type.

For temporal targets, `forbid_max_time_shortcut=true` is machine checked. The
gold is an earlier unique active state and a later same-slot
canceled/outdated/report-or-belief update is a required distractor and
reasoning anchor. Selecting the greatest `time_index` therefore gives the
wrong answer.

Historical run88/run89, v14 `7/7`, and v15 run173 `5/7` are not full v16
evidence. The v16 targeted visibility and relation checks pass, including one
real relation semantic correction, but a fresh seven-operation v16 smoke is
still required in a new output directory before any dialogue or QA call. A
graph rejection is a contract diagnostic, not a reason to continue to the
expensive stages.

## 3. Mine One Small Full Batch

Use a new challenger-generation seed for each batch, but keep
`--solver-eval-seed`, checkpoint and K5 sampling parameters fixed.

```bash
python scripts/run_auto_graph_expansion_round.py \
  --run-name stage3_reliable_mining_batch01_n50 \
  --run-root /path/to/stage3_batch01 \
  --round-id current_method_stage3 \
  --round-number 3 \
  --parent-round-id r2 \
  --parent-round-number 2 \
  --parent-mode round_dataset \
  --parent-dir base_restart_current_method_validation/172_stage2_canonical_graph_rebuild_v3_semantic_audited_final/stage3_parent_v3.jsonl \
  --train-only \
  --generation-backend native_current_method \
  --native-challenger-mode graph_conditioned_api_dialogue \
  --native-curriculum-mode r3_long_easy_v2 \
  --qa-operation-answer-type-diversification \
  --fixed-solver-name solver_base_restart_stage2_accepted \
  --solver-model /path/to/accepted_models/solver_base_restart_stage2_merged \
  --solver-adapter none \
  --accepted-stage2-solver-model /path/to/accepted_models/solver_base_restart_stage2_merged \
  --solver-tokenizer /path/to/Qwen3-4B \
  --base-model /path/to/Qwen3-4B \
  --n-train 50 \
  --min-length-delta-tokens 4000 \
  --seed 20260714 \
  --solver-eval-seed 20260711 \
  --stage3-final-eval-seed 20260712 \
  --stage3-batch-closure execute \
  --stage3-max-hardening-retries 2 \
  --stage3-operation-state-path /path/to/stage3_operation_controller_state_v5.json \
  --max-attempts-per-sample 4 \
  --max-api-failures 100 \
  --generation-concurrency 4 \
  --llm-timeout 180 \
  --max-model-len 32768 \
  --max-length 32000
```

This mode performs semantic filtering, option checks, search cyclic K5 and
decoding-health validation. Closure freezes graph, dialogue, gold and the full
reasoning target, then applies exactly one feedback rewrite action:
`easy5 -> harden`, `hard0 -> soften while retaining a diagnostic copy`, or
`position_sensitive -> repair option-position cues`. Existing robust frontier
is retained without an API rewrite. All rewritten candidates are shuffled and
receive an independent final cyclic K5 verdict.

`--stage3-max-hardening-retries` only retries challenger responses that are
missing or fail local QA validation. It is not a second post-evaluation
difficulty round. Search and final counts are both recorded; only
`final_reliable_batch/frontier.jsonl` is an accepted closure output. The
controller consumes the pre-hardening search signal once and reports metrics
by full reasoning signature without rerunning the solver.

The next production action is one small mining batch, not a large generation
run. As soon as that batch is generated, shuffle/balance its options and run
all cyclic positions with the fixed accepted Stage2 solver at K5. Inspect the
semantic pass rate, decoding health, position sensitivity, and reliable
frontier yield before increasing batch size.

## 4. Accumulate Reliable Frontier

```bash
python scripts/current_method/accumulate_stage64_reliable_frontier.py \
  --recalibration-dir /path/to/stage3_batch01/stage64_mining/batch_closure/final_reliable_batch \
  --recalibration-dir /path/to/stage3_batch02/stage64_mining/batch_closure/final_reliable_batch \
  --output-dir /path/to/stage3_accumulated \
  --min-frontier 300 \
  --heldout-ratio 0.20 \
  --seed 20260711
```

Before 300 reliable rows, only the canonical pool and summary are published.
After the threshold, all descendants of one `lineage_group` stay in one
partition. Train and heldout options are materialized from already evaluated
cyclic positions so each option arity has gold-label spread at most one.
`training_ready_manifest.json` is written only when every gate passes.

## 5. Build the Stage3 Train Mix

```bash
python scripts/current_method/build_stage3_train_mix.py \
  --training-ready-manifest /path/to/stage3_accumulated/training_ready_manifest.json \
  --replay-jsonl /path/to/stage1/train_mix/frontier_train.jsonl \
  --replay-jsonl /path/to/stage1/train_mix/clean_replay.jsonl \
  --replay-jsonl /path/to/stage2/train_mix/frontier_train.jsonl \
  --replay-jsonl /path/to/stage2/train_mix/clean_replay.jsonl \
  --output-dir /path/to/stage3_train_mix \
  --replay-ratio 0.25 \
  --validation-ratio 0.05 \
  --seed 20260711
```

The official Stage3 heldout is exclusion-only. Validation is reserved from
training lineage groups, or explicitly marked `no_valid` when a group split is
impossible. The builder writes paths and SHA256 hashes for all artifacts.

## 6. Train Stage3

When `validation_manifest.json` says `ready`, pass the train-derived validation
SFT file; otherwise replace `--valid-file` with `--no-valid`.

```bash
python scripts/run_round_solver_training_pipeline.py \
  --round-id r3 \
  --round-number 3 \
  --fixed-solver-name solver_base_restart_stage2_accepted \
  --solver-name solver_stage3_reliable_frontier \
  --prepared-train-file /path/to/stage3_train_mix/sft_train.jsonl \
  --prepared-train-summary /path/to/stage3_train_mix/train_mix_summary.json \
  --valid-file /path/to/stage3_train_mix/validation_train_derived_sft.jsonl \
  --base-model /path/to/accepted_models/solver_base_restart_stage2_merged \
  --adapter-output-dir /path/to/stage3_adapter \
  --merged-output-dir /path/to/stage3_merged \
  --max-length 32000 \
  --max-model-len 32768
```

The wrapper verifies the train-mix audit status, path and SHA256. It also
rejects an adapter whose declared base differs from `--base-model`.

## 7. Evaluate Stage3

```bash
python scripts/current_method/cross_stage_eval_panel.py \
  --panel-dir /path/to/stage3_eval_panel \
  --stage3-model /path/to/stage3_merged \
  --stage3-heldout /path/to/stage3_accumulated/splits/frontier_lineage_group_holdout_seed20260711/frontier_heldout.jsonl \
  --stage3-chat-heldout /path/to/stage3_accumulated/splits/frontier_lineage_group_holdout_seed20260711/frontier_heldout_sft.jsonl \
  --run-original \
  --run-chat \
  --k-eval 5 \
  --seed 20260711 \
  --max-model-len 32768
```

The primary comparison is Stage3 versus Stage2 on reliable Stage3 heldout.
Also report Stage1/Stage2 retention, hard0/easy5 diagnostics, clean retention
and external generalization. Training accuracy and the old Stage64 heldout do
not establish Stage3 growth.
