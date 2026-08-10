# Structural QA Writer v19 Probe

This sibling is an isolated QA-only experiment over the frozen 110-turn
`cc-train-episode-146700` Parent+Child dialogue. It does not run Expansion or
call Memorax. One Structural QA Writer request selects only candidates with a
declared discriminative reasoning structure and minimal evidence; the output is
adapted to the unchanged local Solver and SPELL-style gates.

Run zero-API preflight:

```bash
python -m scripts.current_method.restart_n5_spell.simple_append_memorax_v1_fullr1_devdiv_qa_structural_v19_probe_v1.structural_qa_live
```

The explicit `--live` flag permits the unique Writer call after preflight.

## Copied Full-R1 Reference

# Simple Append + Memorax: Full-R1 Development-Diversity Probe v1

This is an isolated sibling experiment. The protected
`simple_append_memorax_v1` module is not modified.

The probe changes only the input policy and one Planner instruction:

- Fresh R1 uses the complete Parent when raw dialogue is at most 4096 local
  tokenizer tokens, otherwise it selects complete tail sessions.
- QA uses the complete Parent+Child under the same raw-dialogue budget.
- A three-scene plan must contain at least two materially different concrete
  developments.
- Memorax uses the isolated
  `gxy-selfplay:fullr1-devdiv-probe-v1:<root_id>` user namespace.

All scene counts, turn budgets, Realizer/QA schemas, distractor logic, Solver
gates, and training behavior remain inherited from the stable implementation.
`probe_live.py` is preflight-only unless `--live` is explicitly supplied.

# Copied Mainline Reference

This package is an optional, isolated Memory context layer around the stable
`simple_append_v1` contracts. It does not make Memorax authoritative and does
not modify the base Planner, Realizer, assembly, MCQ, or Solver schemas.

The single-root path is:

```text
Parent Add (first round only)
-> Bucket Search
-> bounded non-authoritative Memory overview
-> existing Planner contract
-> existing round Realizer contract
-> existing Parent+Child assembly
-> accepted Child Add
-> Bucket Search
-> complete-dialogue MCQ prompt plus Memory topic hints
```

Only dialogue that passes local Planner, Realizer, and assembly checks is added
as a Child increment. QA gold and evidence must still come from the complete
Parent+Child transcript. Memory keys are never accepted as evidence IDs.

Add requests use `async_mode=true`. An accepted response must return a
`task_id`; the runner polls the documented status endpoint and does not Search
or continue generation until it observes `success`. A failed or exhausted
status check stops without automatically resubmitting the possibly durable Add.

Run the zero-request preflight with:

```bash
python -m scripts.current_method.restart_n5_spell.simple_append_memorax_v1.single_root_live \
  --preflight-only
```

Plain HTTP is rejected by default. The live command requires either an HTTPS
Memorax endpoint or an explicit `--allow-insecure-memorax-http` flag. That flag
is a transport-risk acknowledgement; it must not be enabled implicitly.

When a Parent is already present and has been audited, a later fresh run can
skip only that initialization write with:

```bash
python -m scripts.current_method.restart_n5_spell.simple_append_memorax_v1.single_root_live \
  --allow-insecure-memorax-http \
  --reuse-existing-parent-memory
```

This still performs a pre-generation Search and writes only a newly accepted
Child. It never treats a failed local run as resumable.

The first live canary deliberately calls only Planner and round Realizer. It
materializes the Memory-augmented MCQ prompt but does not call the QA Writer,
Solver, Questioner, training, or any `world_memory_slim_v1` role.

The two-root batch entry point is preflight-only unless `--live` is explicit:

```bash
python -m scripts.current_method.restart_n5_spell.simple_append_memorax_v1.batch_live \
  --preflight-only --allow-insecure-memorax-http
```

It uses one isolated expansion and QA directory per root, reports an exhausted
async Memory poll as `pending_recovery`, and never resubmits that Add. A live
batch requires the additional `--live` flag. The local Solver model is loaded
once and reused across roots. Solver-admitted SPELL frontier items are training
candidates; missing or mismatched evidence auditing remains explicit metadata,
while an executed audit that disproves the answer still quarantines the item.
