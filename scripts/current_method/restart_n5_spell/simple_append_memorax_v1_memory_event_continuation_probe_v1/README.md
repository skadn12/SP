# Memorax Memory Event Continuation Probe v1

This isolated Fresh R1 probe tests one narrow pipeline:

```text
Fresh Parent Add -> Bucket Search -> transcript-verified Planner
-> one Round Realizer -> append-only Child -> Child Add -> Bucket Search
```

Memorax supplies non-authoritative people and concrete event hints. The full
Parent transcript is authoritative and the Planner must continue after the
verified current state. The branch deliberately excludes Event Chain, State
Ledger, anchors, QA, Solver, R2/R3, and training.

Default input: `cc-train-episode-13836`.

Default namespace:

```text
gxy-selfplay:memory-event-continuation-probe-v1:cc-train-episode-13836
```

Run the zero-API preflight with:

```bash
python -m scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.live \
  --preflight-only --allow-insecure-memorax-http
```
