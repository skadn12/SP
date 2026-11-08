# One root, one cumulative context

Use the existing expansion runner to generate A and B independently from the
same frozen base, including its frozen previous-round Planner/generator feedback.
Do not feed newly expanded A into B generation or repair. After expansion, use
this version's entry point instead of the historical branch-specific QA runner.

```bash
python offline-deliveries/h4_root_merged_context_v17/merge.py prepare --run <new-run> --expansion <completed-expansion> --pool <historical-pool.jsonl>
python offline-deliveries/h4_root_merged_context_v17/merge.py review --run <new-run>
python offline-deliveries/h4_root_merged_context_v17/merge.py merge --run <new-run>

python offline-deliveries/h4_root_merged_context_v17/generate.py prepare --run <new-question-run> --merge-manifest <new-run>/MERGE_MANIFEST.json
python offline-deliveries/h4_root_merged_context_v17/generate.py verify --run <new-question-run>
python offline-deliveries/h4_root_merged_context_v17/generate.py generate --run <new-question-run>
```

`review` and `generate` use the existing Qwen provider. Preparation and
verification make no API calls. `generate` emits `SOLVER_POOL.json` with one
source hash per root, for the unchanged full-only evaluator. Candidate slots
are a generation budget, not a promised number of selected training examples.
Keep scene boundaries/provenance; count the base once, then add accepted A and B.

For existing-question migration use `merge.py migrate`, then the local Solver
environment for `preflight` and `evaluate`, then `finalize`. The complete run is
independently checked by `audit_v17.py` and sealed by `seal.py`. Historical pool
rows carry archived provenance; the current selection and source bindings are
explicit in `training_pool/CURRENT_PROVENANCE.json` and `MANIFEST.json`.

The latest v15 alternatives were the pilot migration scope. An earlier sibling
expansion is not automatically a previous chronological round. Missing or
conflicting scene evidence is held aside, never silently rewritten or scored
against an unrelated scene sharing the same turn ID. Holding a scene is visible
in `MERGE_MANIFEST.json`; it is not reported as an accepted A+B merge.
