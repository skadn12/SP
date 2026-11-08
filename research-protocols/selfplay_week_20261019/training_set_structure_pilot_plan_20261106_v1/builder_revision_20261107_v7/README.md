# v7 metadata-only builder

The builder prepares a distinct `execution_20261107_v7` from immutable anonymous source/context projections and schedule bindings in v6, then binds Stage B v3 contracts, schemas, state machine and audit rules. It never invokes a model. It refuses to overwrite an existing destination.

## Input interface

The builder is intentionally fail-closed and expects a consolidated selector-safe JSON projection with:

- `schema_version = stage_b_history_projection_v1`, `status = COMPLETE_FOR_SELECTOR`;
- `coverage_set`: exactly 135 independently reviewed, eligible sealed items, each with opaque `history_item_id`, `review_status = INDEPENDENTLY_REVIEWED`, `eligibility = ELIGIBLE_VERIFIED`, and explicit `event`, `natural_information_need`, `query_scope`, `answer_operation`, `evidence_family`;
- `overlap_set`: exactly 77 canonical unique same-root additions after deduplication, 65 fully tuple-annotated (`INDEPENDENTLY_ANNOTATED`) rows and 12 explicit `UNKNOWN_RETAINED` rows. Additions are overlap references; they are not counted as verified training coverage unless a separately audited policy explicitly says otherwise. The builder currently rejects overlap rows marked eligible;
- nonempty `source_bindings` with path and SHA-256 for the full historical inventory and annotation sidecars. The builder verifies every bound raw-byte hash and requires fixed hashes for the coverage index, repaired 966-row inventory, and sealed-135 operation audit.

The projection is not assembled yet. R02/R03 sidecars must be reconciled by canonical candidate identity; 18 sealed overlap representations and 11 derived copies must appear once only. Unknown overlap rows must expose every tuple field as exactly `UNKNOWN`; partial values are rejected. Do not infer novelty from unknown rows or classify drafts as eligible coverage. The builder does not infer any history fields.

## Build and checks

The builder is deliberately blocked even if a projection passes the count checks. The current input contract assumes all 135 sealed rows and 65 operation-coded additions have complete five-dimensional tuples. Independent review found that assumption is false: 49 sealed records have an unresolved natural need, and only 31 of the 65 additional operation-coded rows have a complete five-dimensional tuple. Never invent fields to satisfy the counts. A reviewed amendment for field-level UNKNOWN and source-scoped overlap comparison is needed before this gate can be changed.

The present projection also lacks a reviewed anonymous mapping from each overlap row to the selector's source. The selector cannot identify which UNKNOWN rows are from the current root. Five-dimensional semantic fidelity, non-disclosure of historical questions and answers, exact canonical candidate reconciliation, and source-anchor evidence require independent review; the builder's shape checks and hashes are not that review.

After a reviewed and approved correction of these blockers, the planned invocation was:

```sh
python build_v7.py --history-projection /absolute/path/to/HISTORICAL_COVERAGE_ANON_V7.json --build
```

Then run a v7-specific strict preflight covering raw-byte hashes, schema validity, source and neutral packet bindings, 16 selector history bindings, F/G card assignment (16 F with card, 16 G without), anonymity/access boundaries, immutable schedule counts, and preserved source warnings. Freeze the final manifest only after all checks. Stage A signoff and the combined author/selector-only authorization must bind the SHA-256 of the final frozen manifest; this builder leaves authorization `NONE` and launch flags false.

`preflight_v7.py` is a fail-closed draft diagnostic. It checks packet, schedule, source and authorization bindings and always reports the history gate as incomplete. It does not certify semantic privacy, full provenance or schema/state validity and is not a launch authorization. `python -m unittest -v test_v7` runs synthetic rejection checks without invoking any model or creating an execution.

## Current state

The script is prepared but has not been run. There is no v7 execution directory, author/selector invocation, or generated question. v6 remains unchanged.
