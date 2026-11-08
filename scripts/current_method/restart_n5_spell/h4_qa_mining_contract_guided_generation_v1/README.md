# H4 QA Mining Contract-Guided Generation v1

This sibling experiment changes only candidate generation. It reads the frozen
Phase-0 dialogue and already mined reasoning contracts, then asks a separate
Challenger prompt to generate QA whose evidence metadata is exactly bound to
the contract's `before_state`, `trigger`, and `after_state` refs.

It does not modify `h4_qa_mining_probe_v1`, the alignment-audit probe,
production QA, Planner, Realizer, Memorax, Solver code, or historical runs.

The strict gate requires, for every candidate:

```text
required_reasoning = {before_state, trigger, after_state}
evidence_groups[role] = contract.required_evidence_groups[role]
evidence_refs = union of exactly those contract refs
discriminative_variable = contract.discriminative_variable
```

The gate is programmatic. A prompt can request alignment but cannot silently
admit a candidate that violates it. A single schema-repair call per failed
intent is allowed; no semantic repair is performed by the program.

Run:

```bash
python -m scripts.current_method.restart_n5_spell.h4_qa_mining_contract_guided_generation_v1.run --live
```
