# R03 Feedback-Guided Targeted Evolution Pilot v1

This package runs the bounded pilot after the frozen QA Feedback Template v1
preflight. It reads only the six eligible Common Feedback Data rows and their
authoritative R03 cumulative dialogues.

The live contract is one QA Writer request per seed, with no repair, retry,
seed replacement, Solver, training, semantic auditor, or Memorax call. The
offline closure reports structural/schema outcomes separately from semantic
claims; semantic duplicate and Gold/distractor correctness require a later
review and are not inferred here.

Run from the repository root:

```text
PYTHONPATH=/root/autodl-tmp/gxy-selfplay/current_method_github/scripts \
python -m current_method.restart_n5_spell.h4_round1_r03_feedback_guided_targeted_evolution_pilot_v1.run \
  --output /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round1_r03_feedback_guided_targeted_evolution_pilot_v1_live_1
```

The output directory is immutable once populated.
