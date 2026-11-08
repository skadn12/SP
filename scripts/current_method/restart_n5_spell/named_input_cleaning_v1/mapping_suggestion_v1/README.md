# Model-assisted mapping suggestion

This is an isolated preprocessing stage.  It receives a frozen sequential
selection manifest, asks the configured Challenger for two concrete names per
root, and validates the response through the deterministic cleaner contract.
It never edits source JSONL or cleaned roots and does not invoke expansion, QA,
Memorax, or Solver.

The generated `name_mapping.json` is cleaner-ready only when the report status
is `completed`.  A partial run is retained for diagnosis and must not be sent
to `batch_clean.py`.
