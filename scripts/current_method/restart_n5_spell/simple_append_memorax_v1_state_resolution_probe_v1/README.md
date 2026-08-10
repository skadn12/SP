# Memorax State Resolution Probe v1

This sibling module performs one fixed, offline comparison for
`episode-146700`. It reads the frozen pre/post Memorax overviews and the
authoritative expanded transcript, then reports four manually selected event
lines: soccer, report mistake, health/doctor/work coverage, and new cashier.

It does not call Memorax or any model, infer additional topics, implement a
State Resolver, write a State Ledger, or modify historical artifacts.

Run it from the repository root:

```bash
python -m scripts.current_method.restart_n5_spell.simple_append_memorax_v1_state_resolution_probe_v1.audit
```
