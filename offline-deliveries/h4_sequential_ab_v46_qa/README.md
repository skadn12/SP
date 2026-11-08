# v46 same-input adapted-template comparison

Authorized: compare old and current QA template packages on the same complete
156-turn / 4566-body-token candidate source, previous-training-round feedback,
prior inventory and sampling parameters. Each arm has 2 multi-select and 2
ordering slots. No new dialogue, semantic gates, thresholds or repaired Gold.

Old means the historical v15 COMMON+type instructions, minimally adapted for the
current full provisional source and exact-six-statement dedup/paraphrase policy.
Current means the v45 merged-root instructions plus final generic task reminder.
Both use the same v45 independent ordering reviewer, schemas, gates, exact-set
dedup and full-only M_clean_policy selection. This tests complete template
packages, not a tail-position-only causal effect or byte-exact historical replay.

Both arms use identical candidate IDs (records must be indexed by arm+ID), full
source, frozen prior and initial INPUT_JSON. Initial ordering inputs do not gain
this batch's multiselect outputs. Both arms are prepared before live; actual
first requests must match their frozen payloads. Repairs have only their own
failed slots and same-type accepted drafts, at most once/type. No cross-arm
results or current Solver outputs enter generation feedback or dedup inputs.

Total budget: 8 slots, <=12 Qwen HTTP attempts, <=120 full Solver answers.
Qwen/Qwen3.6-35B-A3B only, shared completion gap >=35s and no automatic HTTP retry.
Use exact audited M_clean checkpoint and .venv-local4b, no ground calls/training.
Arm execution order is predeclared by seed20261011. Small-sample/time effects are
limitations; successful selection alone does not certify semantic Gold.

compare.py prepare/verify/live manages bounded generation. evaluate_and_pool.py
runs unchanged selection separately by arm, including empty-pool paths. Report
first-pass generation, repair, reviewer and actual selected yield separately.
Cross-arm exact duplicates are diagnostic then deduplicated for unique total;
never silently count the same question twice or let first arm block the second.
Prior formal pool remains30; provisional candidates are recorded separately.
