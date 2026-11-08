# QA Feedback Policy v2

This policy is human-authored from the agreed experiment design and is frozen
with the intent schema, prompts, implementation and input hashes before live
generation. Models may instantiate intents; they may not modify this policy,
invent strategies, prescribe a curriculum, or diagnose cognitive weaknesses.

## Authority and feedback

The frozen cumulative Parent + R01 + R02 + R03 transcript is the only factual
authority for questions, options, Gold and evidence. Historical QA and Gold are
curriculum/replay metadata, not facts. Planned outcomes and memory summaries
do not establish completed events. Preserve actor, time and state distinctions.

Only the Intent Miner receives historical Solver behavioral observations.
Accuracy, full/no-context differences, error sets and sample counts describe
observed outputs; they do not prove any reasoning deficit. Missing measurements
remain unknown. Do not infer a weakness from an incorrect answer or from a
selected distractor. Do not output model diagnoses or new strategy names.

## Fixed intent slots

Return exactly one record for each slot, in this order. Each record can be READY
or NO_SAFE_EVOLUTION. A NO_SAFE record is a normal result and schedules no QA
generation. Three slots do not imply three safe intents.

1. SUCCESSOR_FRONTIER: reference an eligible historical feedback ID and propose
   a new answer target or reasoning relation to which R03 evidence contributes.
   Do not paraphrase the historical question. Without an eligible feedback seed,
   this slot must be NO_SAFE_EVOLUTION.
2. COVERAGE_GAP: reference a program-supplied coverage gap ID and derive a
   supported relation using R02/R03 evidence. Computed gaps measure historical
   citation coverage only; they are not proof of a semantic coverage gap.
   Inspect the transcript and historical questions before proposing an intent.
3. FRESH_R03: derive novelty from R03 facts or relations without binding the intent
   to a historical QA or Solver observation. Older turns may supply necessary
   context. Do not copy another slot's relation merely to fill this slot.

Only slot-specific strategies in Intent Contract v1 are allowed. Intent IDs are
I001, I002 and I003 respectively. Preserve the first-round three nonempty,
disjoint reasoning evidence groups (before_state, trigger, after_state); use
NO_SAFE_EVOLUTION if the intended relation cannot be expressed safely.

The program validates slot identity, strategy, feedback/gap IDs, canonical turn
references, disjoint evidence groups and required R03 references before scheduling.
The model never grants itself generation/admission permission. A structurally
accepted intent is a proposal, not a semantically verified relationship.

## Generator isolation and batch targets

Generators receive the validated intent, cumulative transcript, evidence anchors,
readable historical question/replay metadata and prior generated stems. They do
not receive raw Solver observations, model scores, training metrics or packet
diagnoses. Feedback IDs are attached locally to candidate provenance.

Retain the first-round generator modes and target type mix:

- contract_guided: 2 single_choice + 1 multi_select + 1 ordering (4 total).
- free_exploration: 4 single_choice + 2 multi_select + 2 ordering (8 total).

Free exploration may choose alternative evidence within the approved slot's
scope. Both modes must implement its novelty/R03 constraints. Targets are not
admission quotas. Return fewer candidates, including an empty candidates array,
when further distinct supported questions are unavailable. Do not pad a batch.
Return a JSON object containing candidates; each candidate uses the first-round
QA field layout: single A-D, multi A-E, ordering I1-I4, explicit Gold and refs.

## Validation, repair and provenance

Use the first-round candidate normalizer, structural validator and permutation
checks. Retain individually valid rows when other rows or count targets fail.
Track batch count deviations separately. Existence of a reference does not
prove semantic support, and citing R03 does not prove R03 necessity.

Apply deterministic exact/normalized stem and content duplicate checks within
the root and against its frozen historical bank. Report lexical similarity as
a review signal. Cross-root wording similarity alone never removes a candidate.
Check full evidence support, unique Gold, distractors, semantic novelty and actual
R03 dependence in subsequent semantic review before final admission.

After primary generation and local filtering, allow at most one targeted repair
for each required question type entirely absent at root level. Do not repair to
reach a numeric quota. Repair candidates undergo the same checks. Transport or
parse failures are recorded and never silently regenerated.

Attach root, input hashes, intent/slot/strategy, generation mode, feedback/gap IDs,
request/batch ID, evidence refs, R03 citation flag and filtering reasons locally.
Keep feedback alignment and semantic novelty UNKNOWN/PENDING until reviewed.

## Experiment boundaries

Use Qwen/Qwen3-14B, temperature 0.1, thinking disabled, 4096 intent output tokens,
8192 QA output tokens, the existing client/ledger and the frozen R03 baseline
root set. Preserve prior runs. Frozen candidate pools are adaptive self-play
data, not independent benchmarks. Freeze questions before Solver evaluation.
Any feedback/no-feedback ablation must share inputs, policy, budgets and
selection rules; counts alone do not demonstrate feedback benefit.
