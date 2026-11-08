# H4 Relation-Contract Refinement Probe v1

This is an isolated execution refinement of H4 state-evolution expansion. It
does not create a new conceptual hypothesis and does not replace H4A or H4B.
It injects a typed `fact_A -> trigger -> fact_B -> resolution` contract into
the scene-local Realizer prompt for one small run.

The probe separates two claims:

1. **Data relation:** the generated transcript contains a same-slot, ordered,
   trigger-backed revision relation. This is measured by `relation_present`,
   `trigger_supported`, `state_chain_valid`, and `relation_audit_pass`.
2. **QA utilization:** the unchanged QA Writer produces atomic questions whose
   evidence covers both relation sides, and the unchanged composition gate
   makes a pair. This is measured later from the Writer/Solver artifacts by
   `atomic_generated`, `pair_candidate`, `composer_ready`, and `admitted`.

The stronger necessity test is a separate ablation over any actual composite
QA response:

```text
full context: correct
remove fact A: incorrect or underdetermined
remove fact B: incorrect or underdetermined
```

The generation probe itself makes no QA or Solver call. It leaves all H4A/H4B,
production Writer/Solver, Memorax, Parent/Child, and historical run files
unchanged.

Run one canary:

```bash
python -m scripts.current_method.restart_n5_spell.h4_relation_contract_refinement_probe_v1.probe \
  --case-id POB03 \
  --run-root /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_relation_contract_refinement_probe_v1_...
```

The isolated canary uses an 8,192-token Realizer response budget because the
relation contract is longer than the temporal-only prompt. It still makes one
request per scene and has no retry.
