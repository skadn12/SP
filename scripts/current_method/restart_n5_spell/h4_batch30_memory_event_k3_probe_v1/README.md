# H4 Batch30 Memory-event + K=3 Offline Plan

This package contains an isolated plan builder and a safety-gated executor for
the next 30-root integration probe.  The plan builder is offline; the executor
is the only entry point that may invoke the frozen expansion and K=3 QA
commands.  Neither component modifies the cleaned source roots, production
modules, or historical runs.

The expansion flags are frozen from the latest named-speaker successful
sample, `conversation_chronicles_episode_91716_named_speakers_event_bucket_only_expansion_v1_live_1`:
`memory_only_fast=true`, `event_bucket_only=true`, R1, and the bounded-scene
outcome contract.

For cumulative batches, `--cleaned-batch` may also point to a
`h4_root_selection_manifest_v1` produced by
`batch_manifest_v1`.  That manifest references existing cleaned roots by
path and SHA-256; it does not duplicate their files.  Use one frozen selection
manifest per run and never mix two versions of the same root with different
hashes.

## Fixed chain

```text
cleaned unified_root.json (one root)
  -> latest named-speaker, event-bucket-only Memory-event continuation R1 live.py
  -> selector-approved expansion source (promotion first; gated raw fallback)
  -> H4 fresh QA mining, single + multi-select + ordering
  -> K=3 no/full-context Solver and position variants
```

The expansion input is passed directly as `--input <root>/unified_root.json`.
The latest live runner reads `parent["dialogue"]["turns"]`, creates its own
Parent/Child Memory writes, and emits `<expansion_run_root>/expanded_dialogue.json`.
The K=3 runner receives a path returned by the isolated, read-only
`select_successful_expansion_source` selector during its `inventory` phase.
No path written in this preflight plan is itself QA authorization.  The
selector requires the report to say `status=completed` and
`generation_ready=true`, binds the run to the expected cleaned root and input
SHA-256, and verifies the selected dialogue validation and
`complete_dialogue_sha256`.  It prefers
`promotion/promotable_expanded_dialogue.json`; raw output is allowed only as a
success-gated fallback when promotion is absent.  A failed, partial, stale, or
hash-mismatched raw file is quarantined and never sent to QA.

## Offline preflight

```bash
cd /root/autodl-tmp/gxy-selfplay/current_method_github
python -m scripts.current_method.restart_n5_spell.h4_batch30_memory_event_k3_probe_v1.run \
  --cleaned-batch /root/autodl-tmp/gxy-selfplay/restart-n5-cleaned/named_batch_v1_new30_h4_pilot_v2 \
  --run-root /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_batch30_memory_event_k3_probe_v1_offline_5 \
  --expected-count 30 --shard-size 10
```

This command verifies the cleaning manifest, every root hash and unified-root
contract, and cross-root ID/turn uniqueness.  It writes one parent plan and
three `shards/shard-00` through `shard-02` manifests.  Missing runtime assets
are recorded in `runtime_assets.existence_checks`; they never trigger a call.

## Live execution

Run a shard only after reviewing its newly written plan.  The executor runs
one root at a time, records stdout/stderr under the shard, and permits at most
one bounded retry by default.  A retry always receives a new run directory and
Memorax namespace.  A root is eligible for QA only when its report says
`status=completed`, `generation_ready=true`, `child_add_verified=true`, and
`event_bucket_only=true`, with matching input/root hashes and a valid dialogue.
Failed or ambiguous roots are quarantined; they are never silently replaced by
a raw file that merely happens to contain a `turns` list.

```bash
python -m scripts.current_method.restart_n5_spell.h4_batch30_memory_event_k3_probe_v1.execute \
  --phase expansion \
  --plan-root /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_batch30_memory_event_k3_probe_v1_live_1 \
  --shard shard-00 \
  --repo-root /root/autodl-tmp/gxy-selfplay/current_method_github \
  --namespace-prefix gxy-selfplay:h4-batch30-memory-event-k3-v1-live-1
```

After every root in a shard is ready, the QA phase creates a fresh inventory,
passes its smoke gate, and then runs the K=3 batch route with single-choice,
multi-select, and ordering candidates.  It refuses to start when any shard
root is missing unless `--allow-partial` is explicitly supplied.

The optional `--verified-expansion ROOT_ID=ATTEMPT_ROOT` argument lets a new
batch record an already completed, independently verified expansion (for
example the one-root smoke) without reusing its QA candidates.  The executor
still rechecks the report and input hash before accepting it.

## Shard contract

Each shard contains ten roots.  Each root has:

- the exact cleaned input path and SHA-256;
- a fresh per-root Memorax namespace and derived user ID;
- an expansion command with `--memory-only-fast`, `--event-bucket-only`,
  `round_index=1`, and no
  namespace reuse;
- expected raw, promotion, validation, and report artifact paths (locations
  only, not QA authorization);
- a deferred QA source-selection record; only eligible selector outputs can
  populate `--root-source`;
- the K=3 route metadata.

The manifests also define a checkpoint boundary:

```text
ROOT_STARTED
-> EXPANSION_READY
-> QA_INVENTORY_READY
-> QA_SMOKE_PASSED
-> QA_BATCH_COMPLETED
```

The execution wrapper runs the selector after each expansion and skips a root
only when its successful source artifact and hashes still match.  The shard
manifest intentionally has no static QA inventory command containing raw
paths: it has a deferred command template that is materialized from eligible
selector outputs.  A partially completed or ambiguous Memorax request must be
quarantined and must not silently reuse the namespace.

## Current route contracts

Expansion module:

```text
scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.live
```

QA module:

```text
scripts.current_method.restart_n5_spell.h4_qa_multiroot_k3_t08_probe_v1.run
```

QA is fresh generation with all three question types, K=3 rollouts per
variant, no/full-context modes, and cyclic display positions.  Single-choice
uses temperature `0.7`; multi-select and ordering use `0.8`.  Stable paired
seeds use `(root_id, candidate_id, slot_id)` and exclude context mode and
display position.

Smoke and batch commands receive the same explicit `--api-config` path as
expansion; they do not rely on a process default.

## Source selection contract

`plan.py` exposes `select_successful_expansion_source(...)` as a pure,
read-only boundary.  It returns either `eligible=true` with one selected path
and hashes, or `status=quarantined` with reason codes.  Its required checks
are:

```text
report.status == completed
report.generation_ready == true
report.root_id and run_root match the expected root
report/freeze input SHA-256 matches the cleaned input
selected root, counts, validation, and complete dialogue hash match
```

Promotion is authoritative when present.  If promotion exists but fails a
check, the selector quarantines the root rather than falling back to raw.  Raw
fallback is permitted only when promotion is absent and every success gate
passes.  This package remains an offline plan/template unless the explicit
executor is invoked; preflight itself does not perform source selection, call
an API, invoke QA/Solver, or claim that a 30-root batch completed.

This module freezes the connection between the two routes; it intentionally
does not claim that the live batch has run or that any QA is training-ready.
