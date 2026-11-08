# H4 Multi-root K=3/T=0.8 Probe v1

This is an isolated execution probe for the current H4 QA-mining route. It
does not modify the production QA Writer, Planner, Realizer, Memorax, Solver,
or any historical run.

## Stages

```bash
python -m scripts.current_method.restart_n5_spell.h4_qa_multiroot_k3_t08_probe_v1.run \
  --phase inventory \
  --run-root /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_qa_multiroot_k3_t08_probe_v1_live_1

python -m scripts.current_method.restart_n5_spell.h4_qa_multiroot_k3_t08_probe_v1.run \
  --phase smoke --live \
  --run-root /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_qa_multiroot_k3_t08_probe_v1_live_1

python -m scripts.current_method.restart_n5_spell.h4_qa_multiroot_k3_t08_probe_v1.run \
  --phase batch --live \
  --run-root /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_qa_multiroot_k3_t08_probe_v1_live_1

python -m scripts.current_method.restart_n5_spell.h4_qa_multiroot_k3_t08_probe_v1.final_audit \
  --run-root /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_qa_multiroot_k3_t08_probe_v1_live_1
```

`inventory` is read-only with respect to source data and makes no model calls.
`smoke` generates fresh QA for one root and evaluates a type-balanced sample.
`batch` is allowed only after the smoke report passes; it evaluates every
valid fresh candidate on all selected roots, reusing only completed results in
this same new run for crash-safe continuation.

`final_audit` is offline. It verifies all K=3 slots, paired seeds, gold
rebinding, checkpoints, request counts, and named-speaker views, then exports
hard and full-context-zero review queues. It does not make semantic grounding
claims.

## Fixed evaluation contract

- K=3 target slots per display position and context mode.
- Single-choice temperature 0.7; multi-select and ordering temperature 0.8.
- Four single-choice positions, five multi-select positions, and four ordering
  display permutations.
- The same stable seed for a candidate/slot is paired across context and
  display position. Context and position are excluded from the seed.
- Technical/parse failures are recorded separately and make a position
  incomplete; they are never counted as semantic wrong answers.
- `context_helpful` is an auxiliary observation (`full > no`), not a second
  difficulty bucket.
- Structural evidence refs are checked, but semantic grounding is not judged.

The cleaned named-speaker dialogue is written only below the new run. Source
expanded dialogue files are never rewritten.
