# H4 QA Mining Probe v1

Phase 0 only. This sibling probe mines reasoning intents and generates a small
candidate pool from a frozen 119-turn Round 2 expanded dialogue. It performs
full option/item position audit before aggregating Solver difficulty.

It does not modify the production QA Writer, `simple_append_v1`,
`simple_append_memorax_v1`, Planner, Realizer, Memorax, or historical runs.

## Phase-0 contract

Each accepted intent must produce at least `2 single_choice + 1 multi_select
+ 1 ordering`.  The Solver audit is direct: single-choice A-D is rotated over
four placements, multi-select A-E over five placements, and ordering I1-I4
over four display rotations.  Every placement is evaluated with both
no-context and full-context and five rollouts.  Structured questions use the
canonical `structured_qa_box_protocol.py`; position re-binding is performed by
the program, not by the model.

The isolated runner is:

```text
scripts/current_method/restart_n5_spell/h4_qa_mining_probe_v1/run.py
```

The completed Phase-0 artifact is:

```text
/root/autodl-tmp/gxy-selfplay/restart-n5-runs/
conversation_chronicles_episode_178392_h4_qa_mining_probe_v1_phase0_4
```

Important reports in that directory are `report.json`, `summary.md`,
`candidate_pool.json`, `solver_results.json`, `difficulty.json`, and
`critic.json`.  `full_context=0` candidates are retained for manual review;
they are not silently discarded.  Phase-1 question evolution is only an
interface stub and was not executed.
