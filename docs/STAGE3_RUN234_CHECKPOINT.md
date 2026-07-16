# Stage3 run234 safe-pool checkpoint

Checkpoint date: 2026-07-16

This document records the isolated `run234` result without admitting its output
to training data or copying generated run directories into the source repository.

## Local immutable artifacts

- Runner: `stage3-data-fix/run_234_safe_pool_scene_canary.py`
  - SHA-256: `e0086e83a31c8f5d0b218905e904c6dd931c79a8035d744826ac2a6b9264f104`
- Final report:
  `base_restart_current_method_validation/234_safe_pool_scene_canary_n5/final_report.json`
  - File SHA-256: `33c104506f42cd5b05eb5d568aa4e0ee02e0e0acea057f737c0ddc06a7a8562e`
  - Embedded report SHA-256:
    `0a9ec4ae516e550fac067423fe647d7d54858a2838b3df84a6cfd67d7e9e54ea`
- API call journal:
  `base_restart_current_method_validation/234_safe_pool_scene_canary_n5/api_scene_calls.jsonl`
  - SHA-256: `d5e1e3c0659a4e044168832b9afafd9f48dba8cbfc64b895392d94804aeee0e8`
- Frozen truth plans:
  `base_restart_current_method_validation/234_safe_pool_scene_canary_n5/frozen_truth_plans.json`
  - SHA-256: `b30c4f970dd43a106352d34159b1d27930839cfa65f9475190720490a9970886`

## Fixed outcome

- Final status: `FAIL`
- Physical scene API calls: exactly `5`
- Attempts per canary: `[1, 1, 1, 1, 1]`
- Internal/external retries, repair, continuation, QA API, graph API: `0`
- Training writes: `0`
- Final eligible semantic-clean scenes: `0/5`
- Independent manual core-semantic clean: `2/5`
- Natural dialogue: `0/5`
- Protected-fact violations: `3/5`
- QA shortcuts: `1/5`
- Source parent file and all five source graph hashes remained unchanged.

All five scenes fell below the fixed 150-token lower bound. The dominant
qualitative failure was a repeated record/status-recital discourse pattern.
Specific semantic failures included an added `finalized` state, an added
`last week` time plus `event` type, and an adjacent exchange that directly
confirmed the predetermined gold.

The generated artifacts remain intentionally excluded by `.gitignore`; this
checkpoint records their exact paths and hashes and does not mark them clean.
