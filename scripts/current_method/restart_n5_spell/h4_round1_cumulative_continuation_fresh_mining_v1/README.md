# Round-1 cumulative continuation fresh mining

`python -m scripts.current_method.restart_n5_spell.h4_round1_cumulative_continuation_fresh_mining_v1.run`

The preflight reads the frozen Round-0 Train split, canonical R01 promotion
dialogues, and verified R01 Memorax completion records. It writes a new
versioned run directory and makes zero API, QA, Solver, or semantic-auditor
calls.

Inspect a frozen shard without making requests:

```bash
python -m scripts.current_method.restart_n5_spell.h4_round1_cumulative_continuation_fresh_mining_v1.batch \
  --run-root /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round1_cumulative_continuation_fresh_mining_v1_live_2 \
  --shard shard-00 --plan-only
```

Execute one shard serially by removing `--plan-only`. Each root has an isolated
attempt directory and state file. The batch stops at the first quarantined root
and never automatically retries an uncertain R02 Memorax write. The verified
single-root smoke is imported by SHA and is not submitted a second time. This
phase only expands R02; QA and Solver calls remain zero.
