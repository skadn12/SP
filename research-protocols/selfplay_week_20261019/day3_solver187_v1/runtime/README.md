# Day 3 full Train diagnostic runtime

This version copies the proven Day 2 pilot runtime and preserves the native
full-context prompt, v15 deterministic display rebinding, finite-choice decoder,
S0 M-clean merged weights, Qwen3-4B tokenizer and post-raw-freeze scoring.
The original runtime and prior artifacts remain separate.

Use the repository's `.venv-local4b/bin/python -B`. All commands below assume the
repository root as working directory. Writes are exclusive; no automatic retry
or overwrite is supported.

## Fixed scope and inputs

The diagnostic denominator is exactly 187 content-admitted Train candidates:
74 multi-select and 113 ordering. MC uses 6 display permutations with 3 samples
per permutation; ordering uses 12 permutations with 1 sample each. The request
denominator is exactly 2,688. Production accounting is separately 187 admitted
of 240 planned slots, leaving 53 unadmitted slots; unadmitted slots are not Solver
requests. This run is a Train diagnostic, never a final DEV/TEST evaluation or
an automatic training admission filter.

Public JSON is an object containing `questions`, a list of public question
objects. Each object has `candidate_id`, `root_id`, `question_type`, `question`,
`source_path`, `source_sha256` and exactly one of `options` (A through F) or
`items` (I1 through I6). Gold, arm, design and evidence fields are forbidden in
these question objects and are not passed to the model.

Private JSON contains the same public fields plus `gold` in canonical labels and
optional private metadata. Preparation copies private bytes to a disjoint sidecar
without parsing. Live inference opens only public execution artifacts. Scoring
parses Gold only after every terminal response and the raw freeze are verified.

`EXECUTION_FREEZE.json` must supply:

- `solver`: native decoder/model fields, `display_seed=2026101903` and
  `sampling_seed=2026101904`; CLI seed overrides must match these exact values.
- `budget.initial_question_slots=187`, `budget.max_direct_requests=2688`,
  `budget.production_planned_slots=240`.
- `slots`: exactly the 187 admitted candidate IDs, with `root_id`,
  `question_type`, `item_count=6` and `source.sha256` matching public input.

`CONTENT_FREEZE.json` must supply status PASS/FROZEN/CONTENT_FROZEN, namespace
beginning `train.feedback`, the candidate ID set, public/private file bindings,
`planned_slots=187`, `admitted_count=187`, `production_planned_slots=240`.

## CPU checks and preparation

    .venv-local4b/bin/python -B research-protocols/selfplay_week_20261019/day3_solver187_v1/runtime/selftest.py --output /new/path/CPU_SELFTEST.json

The selftest uses synthetic questions only. It creates no backend or Solver calls.

    .venv-local4b/bin/python -B research-protocols/selfplay_week_20261019/day3_solver187_v1/runtime/prepare.py --public research-protocols/selfplay_week_20261019/day3_solver187_v1/inputs/PUBLIC_QUESTIONS.json --private research-protocols/selfplay_week_20261019/day3_solver187_v1/inputs_private/PRIVATE_QUESTIONS.json --protocol research-protocols/selfplay_week_20261019/day3_solver187_v1/EXECUTION_FREEZE.json --admission research-protocols/selfplay_week_20261019/day3_solver187_v1/CONTENT_ADMISSION.json --run research-protocols/selfplay_week_20261019/day3_solver187_v1/runtime/direct

Default Gold sidecar is `runtime/direct_private`, with directory 0700 and files
0600. The public manifest/preflight has no private Gold file path or binding.

Preparation checks complete source inclusion, exact six-item labels, unique
requests and sampling seeds, frozen model hashes and full prompt lengths. Every
prompt plus its 32-token completion allowance must fit 16,384 tokens. Overflow
fails preparation; truncation is forbidden.

An independent preflight produces a public gate bound to the exact freeze:

    {"status":"PASS","audited_files":[{"path":"/absolute/public/run/PREFLIGHT_FREEZE.json","sha256":"...","bytes":123}]}

All gate bindings must point inside the public run. Code/model/tokenizer hashes
are independently verified from the manifest. Keep private audit information in
a separate private artifact.

## Verify, execute and score

    .venv-local4b/bin/python -B research-protocols/selfplay_week_20261019/day3_solver187_v1/runtime/execute.py verify --run research-protocols/selfplay_week_20261019/day3_solver187_v1/runtime/direct --audit /path/PUBLIC_PREFLIGHT_PASS.json
    .venv-local4b/bin/python -B research-protocols/selfplay_week_20261019/day3_solver187_v1/runtime/execute.py live --run research-protocols/selfplay_week_20261019/day3_solver187_v1/runtime/direct --audit /path/PUBLIC_PREFLIGHT_PASS.json
    .venv-local4b/bin/python -B research-protocols/selfplay_week_20261019/day3_solver187_v1/runtime/score.py --run research-protocols/selfplay_week_20261019/day3_solver187_v1/runtime/direct --private research-protocols/selfplay_week_20261019/day3_solver187_v1/runtime/direct_private

Live preserves RETURNED/BACKEND_ERROR/NOT_SUBMITTED rows, raw outputs, token IDs,
metadata, usage, timing, batches and failures. Invalid returns and terminal
failures stay in the fixed denominator. No retry, best-of selection or source
truncation occurs. `RAW_RESPONSE_FREEZE.json` requires every terminal record.

`SCORES.json` includes all requests/questions, exact rates, validity, prediction
consistency, item TP/FP/FN/TN, ordering pair accuracy, type/root averages and
usage/time. Prediction consistency mixes display permutations and sampling; it
is not an isolated position-bias estimate. Invalid MC item selections and FP/FN
are unknown/null. Invalid ordering responses get zero pair credit. Per-question
strata describe this diagnostic, without deleting easy, hard or invalid cases.
