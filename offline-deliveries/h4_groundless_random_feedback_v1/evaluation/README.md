# Groundless full-only Solver evaluation

This is an independent evaluation version. It does not alter earlier runs.
`preflight` plans only full-context calls; it never invokes a ground/no-context
call. A question is selected only when every planned call is valid and the
equal-weight mean of the per-permutation partial scores satisfies
`0 < S_full < 1`. Parse failures and missing calls remain pending/invalid and
are never treated as zero.

The input pool is JSON with `questions` and `provenance`. Each question has a
`candidate_id`, `root_id`, `question_type` (`multi_select` or `ordering`),
`question`, six `options` or six `items`, a label-space `gold`, complete
`option_evidence`/`item_evidence`, and an absolute `source_path` plus its
`source_sha256`. Multi-select Gold cardinalities 2 through 6 are accepted.
Ordering review proof is optional by default and can be required with
`--require-ordering-review`.

Run with the proven local environment:

```text
python run_solver.py preflight --pool POOL.json --run RUN
python run_solver.py live --run RUN
python run_solver.py report --run RUN
```

Set `--skip-model-check` only for a dry preflight. It cannot be used to make a
live run: live verifies the fixed Qwen3-4B checkpoint audit, source/pool/code
hashes, prompt hashes, and the structured decoder contract.

