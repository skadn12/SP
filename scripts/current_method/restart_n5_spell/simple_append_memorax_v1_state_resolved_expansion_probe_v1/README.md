# State-resolved Expansion Probe v1

This isolated sibling reads the frozen `episode-146700` Parent and frozen
pre-generation Memorax overview, then runs exactly:

```text
one thin State Resolver
-> one slim three-track Planner
-> one Round Realizer
-> append-only Parent + Child assembly
```

Memorax remains non-authoritative. Resolver evidence must be an exact Parent
turn reference, and the program assigns `D01`, `D02`, and subsequent IDs. The
Planner receives frozen latest states rather than anchor selection fields.

Run the zero-API hard preflight first:

```bash
python -m scripts.current_method.restart_n5_spell.simple_append_memorax_v1_state_resolved_expansion_probe_v1.live --preflight-only
```

The same module without `--preflight-only` creates the unique live run and
allows one no-retry request per role. Memorax API, QA, Solver, Repair, R2/R3,
training, and State Ledger persistence are not part of this probe.
