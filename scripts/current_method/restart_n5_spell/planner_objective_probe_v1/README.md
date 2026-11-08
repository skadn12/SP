# Planner Objective Probe v1

This isolated Planner-only probe tests whether the Planner's objective, rather
than scene capacity, causes rapid closure. Both conditions use exactly three
scenes and three to seven interaction beats per scene.

- `O0`: current completion-oriented event advancement.
- `O1`: explicit `unresolved -> triggered reconsideration/revision -> final` control.

The same frozen events, model, temperature, schema, validator, and deterministic
scene-complexity analyzer are used. Only the objective instruction changes.
The primary endpoint is the strictly paired difference in complex-chain rate.

The probe never calls Realizer, Solver, QA Writer, or Memorax Search/Add. It
does not mutate production Planner code, Parent/Child data, or historical runs.

Run the two-event pilot:

```bash
python -m scripts.current_method.restart_n5_spell.planner_objective_probe_v1.probe \
  --live --max-events 2
```

Run a no-API smoke by omitting `--live`, or reanalyze a completed run with
`--replay-from /path/to/run` and a new `--run-root`.

Run the deduplicated five-root stability batch:

```bash
python -m scripts.current_method.restart_n5_spell.planner_objective_probe_v1.batch_probe \
  --live --max-roots 5
```

The batch uses one event per frozen `root_id`; repeated experiment snapshots of
the same episode are not counted as independent roots.
