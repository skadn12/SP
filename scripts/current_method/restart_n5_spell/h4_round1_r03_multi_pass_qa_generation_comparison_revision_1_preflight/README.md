# R03 Multi-Pass Revision-1 Preflight

This package prepares, but does not execute, the current revision-1
`MULTI_QA_COMPOSITION` and `FRESH_EXPLORATION` comparison. It reuses the
immutable seven-root selection from the historical `v1` pilot, rebuilds both
schemas and prompts through `qa_output_contract_v1_1_revision_1`, and stops at
an explicit authorization boundary.

The planned budget is one attempt per successful root per pass: seven
Composition plus seven Fresh requests, at most fourteen total. Quarantine root
`cc-train-episode-112812` and Targeted Evolution are excluded.
