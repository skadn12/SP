# R03 Realizer A/B Input Preparation v1

This module is an offline-only preparation step for root `cc-train-episode-122711`.
It uses the frozen R02 dialogue, the already saved Planner plan, and the saved
text-free continuity shape from the failed R03 debug run.

It produces two complete one-message requests:

- `A_FULL_PARENT`: Planner plan plus the complete Parent/R01/R02 transcript.
- `B_TEXT_FREE`: Planner plan plus the existing text-free continuity shape and
  style profile, with no transcript renderer append.

The module does not call a model, Planner, Memorax, Solver, or training. It does
not change the default Realizer route or the historical failed run. A later,
separately authorized runner may consume the two saved `messages.json` files
for exactly one Realizer call per mode.

Run:

```bash
python -m scripts.current_method.restart_n5_spell.h4_round1_r03_realizer_ab_input_preparation_v1.prepare
```

