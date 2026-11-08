# Cross-root feasibility Solver runtime

This is a new local runtime for a 24-slot pilot across four Train roots. Old
sealed runtime and evaluation files remain unchanged. No model inference was
performed while preparing this runtime. `CPU_SELFTEST_V2.json` is the final
selftest report; the earlier `CPU_SELFTEST.json` records the preceding unit-only
check and does not bind the final implementation.

## Frozen input interface

- Public JSON is exactly `{"questions": [...]}`. Each question has only
  `candidate_id`, `root_id`, `question_type`, `question`, `source_path`,
  `source_sha256`, and exactly one of `options` or `items`.
- MC has six distinct, nonempty options in canonical A–F key order. Ordering
  has three to six distinct, nonempty items in canonical I1–In key order.
- Each source is a public dialogue JSON with its matching `root_id` and
  nonempty `turns`. Turns require unique `turn_ref`, `speaker`, and `text`.
  Preparation reads and hashes the original public source, then stages a
  byte-identical public copy. Independent verification reconstructs prompts
  using the staged copy and checks it against `source_sha256`.
- The separate private JSON has `questions`, containing the same complete
  public question fields plus `gold` canonical labels. MC Gold must contain
  at least two unique valid labels; ordering Gold is a complete permutation.
  Extra author/reviewer fields may remain private. Preparation hashes and
  copies these bytes without JSON parsing, and checks the copy against the
  frozen admission SHA. Execution never opens the private sidecar.
- Protocol `solver` must contain the fixed values below plus integer
  `display_seed` and `sampling_seed`. `budget.initial_question_slots` is the
  admitted count; `production_planned_slots` is 24; `max_direct_requests` is
  the exact sum of admitted requests. All 24 `slots` remain in the protocol,
  with unique `slot_id` and `candidate_id` across exactly four planned roots.
  Each slot includes `root_id`, `question_type`, `item_count`, and
  `source: {path, sha256}`. The admitted public questions can be a subset.
- Admission requires status PASS/FROZEN/CONTENT_FROZEN, a namespace beginning
  `train.feedback`, the admitted `questions: [{candidate_id: ...}]`,
  `public_questions.sha256`, `private_questions.sha256`, `planned_slots: 24`,
  and the matching `admitted_count`.

Fixed solver: S0/M-clean, full dialogue, direct boxed answer, temperature 0.8,
top-p 0.9, max output 32 tokens, thinking disabled, context 16384, batch 8,
GPU memory utilization 0.8, no truncation and no retries. The model and
tokenizer paths are enforced by `common.SOLVER_FIXED`.

MC uses six distinct display mappings with three samples each (18 calls).
Ordering uses `min(12, n!)` distinct mappings with one sample each: six calls
for three items and twelve calls for four to six items. Protocol
`ordering_mappings: 12` is this cap. Sampling seed is the frozen starting seed
plus the zero-based request index. All mapping/request reconstruction uses
public content and seeds only.

The native system prompt instructs the model to use the supplied dialogue
and return one box. It does not impose a generic “select all true facts”
criterion: the public question defines the evidence-selection, causal, or
other task. The native finite-choice vocabulary is unchanged for MC6 and
ordering4–6; a local extension supplies all six ordering3 permutations.

## Commands

Use `/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python -B` throughout.
Run preparation only after content review and the execution protocol freeze:

```bash
python -B prepare.py --public PUBLIC.json --private PRIVATE.json \
  --protocol EXECUTION_FREEZE.json --admission CONTENT_ADMISSION.json \
  --run RUN
```

This creates a disjoint `RUN_private` sidecar; an alternative location can be
given with `--private-output`. Existing run or sidecar paths are rejected.
An independent public preflight audit must return status PASS and bind
`RUN/PREFLIGHT_FREEZE.json` in `audited_files` (each binding has absolute
`path`, `sha256`, and `bytes`). It may also bind other public run inputs.

```bash
python -B execute.py verify --run RUN --audit INDEPENDENT_PREFLIGHT.json
python -B execute.py live --run RUN --audit INDEPENDENT_PREFLIGHT.json
python -B score.py --run RUN --private RUN_private
```

`verify` checks hashes and independently reconstructs every display, request
ID, sample slot, seed, mapping, prompt, and complete tokenizer-ID sequence.
`live` additionally checks actual backend token IDs against those frozen IDs.
No automatic rerun is allowed once `STARTED.json` exists. Backend errors
terminalize all remaining planned calls; raw bytes and terminal statuses
are committed before `RAW_RESPONSE_FREEZE.json` is written. The scorer
rejects a missing or incomplete raw freeze before first parsing private Gold.

## Reporting and validation

Executed questions retain all planned calls in denominators. Invalid exact
and ordering pair credit is zero. Invalid MC selection/FP/FN is unknown and
therefore null. Results retain all 24 planned slot outcomes, the four planned
roots, and admitted question/root denominators separately. Root-equal scores
average roots having admitted questions. Exact scores are reported by type
and arity; ordering pair scores include question-equal and pair-weighted
arity results. This pilot provides descriptive feasibility evidence, not an
A/B effect or a controlled claim that new questions are harder.

```bash
python -B selftest.py --report ANOTHER_NEW_CPU_SELFTEST.json
```

The final selftest exercises actual tokenizer/prompt/decoder construction on
CPU, content remapping for every supported arity, native vocabulary equality,
fixed invalid denominators, and a disposable prepare/verify/post-freeze-score
fixture. Only that disposable fixture omits checkpoint bindings and parses
synthetic Gold; production preparation/verification checks real model files.
`MODEL_TOKENIZER_REVALIDATION.json` separately records the real checkpoint and
tokenizer hash revalidation. `RUNTIME_REUSE_REVIEW.json` binds final code and
validation artifacts.
