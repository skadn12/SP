# H4 Position Permutation A/B Probe v1

This is an isolated, read-only Solver measurement probe. It consumes the
frozen `h4_batch30_memory_event_k3_probe_v1_live_1` QA outputs and never edits
that run, the expansion process, QA Writer, production difficulty, or Solver.

The first phase is an offline preflight. It verifies the exact 71-row hard
pool, recovers every historical K=3 rollout by position and slot, computes a
mechanism profile, deterministically samples 30 candidates, and writes the
permutation manifest. It makes zero model calls.

The later A/B runner will use the frozen manifests to measure cyclic versus
broader permutations and B0/B1/B2 label/content modes. Those measurements are
not started by the preflight command.

Inspect the request plan without loading a model or making a Solver call:

```bash
PYTHONPATH=. /root/miniconda3/bin/python \
  -m scripts.current_method.restart_n5_spell.h4_position_permutation_ab_probe_v1.solver_run \
  --run-root /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_position_permutation_ab_probe_v1_live_2 \
  --scope all --dry-run
```

After explicit authorization, ``--scope a`` runs Experiment A (label output
over all 792 permutations). ``--scope all`` then runs B1 and B2 over the 132
cyclic permutations; B0 is the paired cyclic subset already present in A and
is never regenerated. A live invocation requires local ``--solver-model`` and
``--tokenizer`` paths. The combined request plan is 2,112, below the hard cap
of 2,232. Results are ledger-backed and can be resumed in the same run root.

B1 and B2 use the same finite legal complete-text answer set for guided
decoding, separating content-choice scoring from malformed-output failures.
The offline analysis rebinds ordering slot labels through the immutable
display mapping before scoring and never modifies raw Solver results.

Run from the repository root:

```bash
PYTHONPATH=. /root/miniconda3/bin/python \
  -m scripts.current_method.restart_n5_spell.h4_position_permutation_ab_probe_v1.run \
  --phase preflight \
  --source-run /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_batch30_memory_event_k3_probe_v1_live_1 \
  --run-root /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_position_permutation_ab_probe_v1_live_1
```

For the repaired B2-only condition, use the isolated runner below. It hash-checks
the copied manifests against the immutable source run and writes a separate
Solver ledger and raw-output tree.

```bash
PYTHONPATH=. /root/miniconda3/bin/python \
  -m scripts.current_method.restart_n5_spell.h4_position_permutation_ab_probe_v1.b2_repair_run \
  --run-root /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_position_permutation_ab_probe_v1_b2_repair_live_1 \
  --source-run /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_position_permutation_ab_probe_v1_live_4 \
  --solver-model /root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/training/models/merged \
  --tokenizer /root/autodl-tmp/Qwen3-4B
```
