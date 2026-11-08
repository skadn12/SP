# ROUND0 Ordering Active Feedback Exclusion v1

This independent package removes only authoritative ROUND0 Train ordering
candidates from active feedback. It neither changes Gold nor certifies any
retained question. The original Bank, observations, lifecycle, training/replay
state, duplicate history, live13 cards, current expansion bundle and defaults
remain immutable. No production entrypoint imports this version automatically.

## Run Offline

Run from `/root/autodl-tmp/gxy-selfplay/current_method_github` with
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts`. No service configuration,
tokenizer, credentials, model, Memory or network is needed. Every output
directory below must be new. Paths are explicit; there is no latest resolver.

```bash
python3 -m current_method.restart_n5_spell.round0_ordering_active_feedback_exclusion_v1.bootstrap \
  --handoff /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round1_train_feedback_cleaning_path_handoff_v1_live_1 \
  --output /absolute/new-delivery/source_manifest.json

python3 -m current_method.restart_n5_spell.round0_ordering_active_feedback_exclusion_v1.run \
  --source-manifest /absolute/new-delivery/source_manifest.json \
  --output /absolute/new-delivery/export

python3 -m current_method.restart_n5_spell.round0_ordering_active_feedback_exclusion_v1.loader \
  --export-manifest /absolute/new-delivery/export/artifact_inventory.json \
  --validate-only --output /absolute/new-delivery/load-validation
```

`bootstrap` resolves only HANDOFF final sources and directly named frozen schema
or cooldown sidecars. `run` verifies indexed Train pools and original dialogue
snapshots, joins split/root/candidate identities, cross-checks the bootstrap
corpus, then masks members before old selection functions run. It verifies the
later E0Q0/E1Q0/E1Q1 source manifests separately. Formal data never comes from
test fixtures. The output includes data, code SHA, tests and before/after hashes.

## Counting and Signals

On the bound snapshot: 3062 candidates / 3048 clusters become 2493 candidates /
2482 clusters. The excluded 569 candidates occupy 566 clusters. No actual mixed
excluded/retained cluster exists in this snapshot. Global Targeted positive
eligibility is 251 independent sources / 56 roots; Composition is 927 / 70.
These are different from the reference R03 branch's 98 Targeted sources / 8
roots, 8 selected cards, 273 Composition parents and 788 feasible pairs.
Pair feasibility is lexical/evidence-based, not semantic certification or a
generation schedule. All 902 reference role messages are offline only.

Individual metrics and historical labels are retained. Root full/no aggregates
sum independent correct/total counts. Active family counts, family multipliers,
global/within-root weight denominators, route/pool/pair views are rebuilt after
the mask. Original history has 2554 stem signatures; bounded Fresh history has
700 rows and keeps excluded ordering coverage. Fresh sees only signature and
candidate/cluster counts, never family multipliers, Gold or performance.
Mechanical deduplication continues to use the original full Bank unchanged.

Mixed-cluster tests enforce that an excluded representative's observation never
passes through a later alias. A retained independent representative survives;
without separable observation data, member identities survive but active metrics
and eligibility become UNKNOWN/disabled. The real snapshot has no such unknown
clusters and all 70 root count aggregates are reconstructible. Old capability
narratives, unsupported alias splits and semantic validity are not reconstructed.

`dev_global_source_and_usage.json` preserves the old independent Dev aggregate
and its SHA as audit history. The reused current Bank compiler/shared renderer
does not consume it, so this version adds no new Dev prompt/weight channel and
changes no existing global strategy. The claim is limited to removal of ROUND0
Train ordering active contributions, not all ordering or evaluation signals.

## Feedback Boundary

`feedback_envelope` requires version, root, status, reason and records. An empty
object, wrong source identity, injected metrics or incomplete QA is rejected.
Explicit `NO_ELIGIBLE_FEEDBACK_AFTER_SCOPE_FILTER` allows Expansion to continue
with source dialogue and allows Fresh history-only rendering. Targeted and
Composition without sufficient sources are unschedulable. Fourteen global roots
have no Targeted source; Composition eligibility is assessed separately.
The old running incremental bundle contract is untouched.

Complete single/multi historical QA and observed metrics are preserved in real
ROUND0 examples. Later ordering remains available. Generic ordering stems may
match old question wording; the acceptance report records these collisions and
checks complete QA, candidate identities and independent observation provenance.
Neither matching wording nor matching canonical Gold alone identifies a source.

## Bind After New R03 Closure

The delivered messages use the already authoritative Parent+R01+R02-feedback
reference snapshot to verify the offline chain. They are not a completed new
R03 QA run. Production QA sending remains `NOT_CONFIRMED`; this package has no
sender, budget authorization or production-QA integration.

After the separate R03 closure supplies an authoritative immutable snapshot,
create an explicit JSONL manifest with one row per completed root to bind:

```json
{"root_id":"cc-train-episode-122711","status":"FROZEN_NEW_R03","source_prefix":"Parent+R01+R02-feedback+R03","input_dialogue_path":"/absolute/frozen/new-r03-dialogue.json","input_dialogue_file_sha256":"actual-file-sha256","complete_dialogue_sha256":"actual-content-sha256"}
```

```bash
python3 -m current_method.restart_n5_spell.round0_ordering_active_feedback_exclusion_v1.loader \
  --export-manifest /absolute/new-delivery/export/artifact_inventory.json \
  --r03-manifest /absolute/frozen/new-r03-manifest.jsonl \
  --output /absolute/new-r03-binding-output
```

The loader validates export hashes, source identities, record reconstruction,
role pools/pairs and envelope completeness. It rejects the current R02 manifest,
requires an unchanged R02-feedback prefix followed by unique R03 turns, and
re-renders exported selections without reselecting or importing new questions.
It supports a completed subset of the 8 reference roots. Output is
`QA_SNAPSHOT_BOUND_OFFLINE_ONLY`, always `api_call_authorized=false`.
An authorized future QA runner must explicitly consume those validated messages;
no existing runner or frozen source is switched here. No old 7+7 plan is resumed.

Realizer plan generation, new Memory navigation and production QA dispatch are
outside this offline exporter. No replacement scene plan is fabricated. The
current Realizer chain is not rerendered or modified.

## Tests

```bash
python3 -m unittest discover \
  -s scripts/current_method/restart_n5_spell/round0_ordering_active_feedback_exclusion_v1 \
  -t scripts -p 'test_*.py' -v
```

`run` also executes these tests under a write boundary restricted to its new
output and blocks network/subprocess dispatch. Temporary test fixtures stay in
the new output. Protected model-configuration files are hashed as opaque bytes
solely for preservation checks, never parsed or loaded for credentials.
