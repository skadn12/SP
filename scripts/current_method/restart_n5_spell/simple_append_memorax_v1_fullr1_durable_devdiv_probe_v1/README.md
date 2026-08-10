# Full-R1 Durable Development-Diversity Local Anchor Span Probe v1

This is an isolated sibling of
`simple_append_memorax_v1_fullr1_devdiv_probe_v1`. The baseline, the stable
`simple_append_v1` contracts, the QA fact-first branch, and historical runs are
read-only inputs to this experiment.

The experiment keeps the existing Fresh R1 architecture:

```text
Parent context selection
-> Memorax overview
-> one Planner request
-> program-owned scene refs and unchanged turn budgets
-> one Round Realizer request
-> local validation
-> append Child at the Parent tail
-> Memorax Child write
```

Only the Planner scene contract changes. Each scene adds:

```text
primary_anchor_span = {start_position, end_position}
durable_outcome
```

The model cannot output a session ID or canonical turn ID. The program maps
the inclusive one-to-six-turn span to authoritative Parent turns and derives
its session. It verifies a contiguous span within one Parent session, unique
derived sessions across scenes, and a non-empty `durable_outcome`. Semantic diversity, outcome
durability, realization quality, and callback takeover remain human audit
questions. No embedding or keyword semantic gate is added.

The copied `probe_live.py` compatibility entry point routes directly to the
same Expansion-only runner as `single_root_live.py`; it cannot load Solver or
invoke QA. Use zero-request preflight first:

```bash
python -m scripts.current_method.restart_n5_spell.simple_append_memorax_v1_fullr1_durable_devdiv_probe_v1.single_root_live \
  --preflight-only --allow-insecure-memorax-http
```

The unique live run uses the same module without `--preflight-only`. It allows
one Planner and one Round Realizer request, with no orchestrator retry. QA,
Solver, Fact Extractor, R2/R3, training, repair, and graph workflows are out of
scope.
