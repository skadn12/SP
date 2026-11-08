# Scene Complexity Diagnostic Probe v1

This is a fully isolated Planner-only diagnostic for the question:

> Does the current scene budget/granularity prevent the generation of complex state evolution?

It runs paired A/B/C settings over the same frozen events and model:

| Setting | Scenes | Interaction beats per scene |
|---|---:|---:|
| A: current capacity | 3 | 3–7 |
| B: increased per-scene capacity | 3 | 7–12 |
| C: more scenes | 5 | 3–7 |

All three settings receive the same planning objective. The probe uses a
probe-local diagnostic annotation so a deterministic analyzer can distinguish
state value changes, state stage changes, and ordinary event progress. It does
not modify the production Planner schema or create a persistent state system.
Live calls use the same configured model and temperature `0` across settings;
the current provider adapter does not expose an explicit sampling seed, which
is recorded as a limitation.

The probe never calls Realizer, Solver, QA Writer, or Memorax Search/Add. It
does not modify Parent/Child data, production code paths, or historical runs.

Run a no-API contract smoke:

```bash
python -m scripts.current_method.restart_n5_spell.scene_complexity_probe_v1.probe \
  --max-events 1 \
  --run-root /root/autodl-tmp/gxy-selfplay/restart-n5-runs/scene_complexity_probe_v1_offline_smoke
```

Run a small paired live diagnostic:

```bash
python -m scripts.current_method.restart_n5_spell.scene_complexity_probe_v1.probe \
  --live --max-events 2
```

Reanalyze an immutable completed run without any new API calls:

```bash
python -m scripts.current_method.restart_n5_spell.scene_complexity_probe_v1.probe \
  --replay-from /path/to/completed-run \
  --run-root /path/to/new-reanalysis-run
```

Each run writes `inputs.jsonl`, `planner_cases.jsonl`, `audits.jsonl`,
`metrics.json`, `report.md`, and `manifest.json` to a new run directory.
