# v24 sequential question QA adapter

This directory is the downstream QA bridge for a completed v24 sequential expansion. It reads one frozen cumulative source per root (`frozen base -> accepted A -> accepted B`), then reuses the v17 full-only question generation contract. A and B remain provenance diagnostics inside one question context; they are not separate question pools.

Preparation and verification are offline. They require a completed sibling expansion with `MERGE_MANIFEST.json`, `FINAL_AUDIT.json` (`PASS`, complete or explicit shortfall), and the upstream `MANIFEST.json`.

```bash
python generate.py prepare \
  --run runs/<qa-run> \
  --merge-manifest ../h4_sequential_ab_v24/runs/<expansion>/MERGE_MANIFEST.json \
  --slots 4
python generate.py verify --run runs/<qa-run>
python generate.py generate --run runs/<qa-run>
```

The default prior inventory is the frozen 30-question pool at `h4_root_merged_context_v17/runs/merged_latest_02/training_pool/TRAINING_POOL.jsonl`; pass additional `--prior-pool` files only when they are frozen and intended. `generate` is the only command that calls Qwen. It writes the unchanged v17-compatible `SOLVER_POOL.json` and `MECHANICAL_POOL.json`; subsequent evaluation uses the sibling `evaluate_and_pool.py` adapter with the full `M_clean_policy` checkpoint and original Gold.
