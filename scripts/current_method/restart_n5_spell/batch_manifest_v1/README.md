# Reference-only batch manifests

This package creates a small manifest that references immutable cleaned
`unified_root.json` files.  It does not copy roots, modify cleaning outputs, or
call Challenger, Memorax, or Solver.

## Build a selection

```bash
PYTHONPATH=. python -m scripts.current_method.restart_n5_spell.batch_manifest_v1.build \
  --source-manifest /path/to/cleaned_batch/manifest.json \
  --output /path/to/batch_manifests/h4_root_selection_30_v1/manifest.json \
  --expected-count 30
```

The input may be a cleaning manifest or another selection manifest.  Repeated
references to the same root are collapsed only when their path and SHA-256
match.  A root ID with different content is rejected.

The H4 batch planner accepts the resulting manifest through
`--cleaned-batch`.  Its shard manifests continue to contain the exact root
paths and hashes, so a restarted executor can reuse verified root outputs.
