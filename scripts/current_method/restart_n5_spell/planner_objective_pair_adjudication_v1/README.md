# Planner Objective Pair Adjudication v2

This is a read-only, zero-API follow-up to the frozen five-root
`planner_objective_probe_v1` batch. It adjudicates incomplete O0/O1 pairs
without changing the raw Planner plans or rerunning any model call.

The only possible waiver is a narrow adjacent `*_value_discontinuity` caused by
surface wording variation. Both boundaries must retain at least three shared
content-bearing tokens, Jaccard similarity >= 0.55, and sequence similarity >=
0.78. Intention-vs-realization, negation, and numeric-value profiles must also
match. Canonical-speaker errors, stage discontinuities, enum errors, and
unrelated state changes remain rejected.

Run against the frozen batch:

```bash
python -m scripts.current_method.restart_n5_spell.planner_objective_pair_adjudication_v1.adjudicator \
  --replay-from /root/autodl-tmp/gxy-selfplay/restart-n5-runs/planner_objective_probe_v1_batch_20260822T110000Z
```

The command writes a new run directory with `effective_records.jsonl`,
`audits.jsonl`, `metrics.json`, `report.md`, and `manifest.json`. The source
run is never overwritten. This probe does not call Planner, Realizer, QA,
Solver, Writer, or Memorax. This v2 run includes intention-vs-realization,
negation, and numeric-value guards; the earlier permissive v1 interpretation
is retained as a non-authoritative diagnostic only.
