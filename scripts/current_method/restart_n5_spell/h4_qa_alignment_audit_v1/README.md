# H4 QA Alignment Audit v1

This is a read-only, offline audit of an existing
`h4_qa_mining_probe_v1` Phase-0 artifact.  It does not call the Challenger or
local Solver and does not modify the mining probe, production QA, or the
source/run inputs.

The audit checks, per candidate:

- whether the candidate's `required_reasoning` groups are present and
  disjoint;
- whether each candidate evidence group overlaps the corresponding intent
  contract anchor (`before_state`, `trigger`, `after_state`);
- whether evidence refs cover the declared groups and resolve in the frozen
  dialogue;
- whether the question type has the expected relation-chain cues;
- whether the discriminative variable appears lexically in the question (a
  diagnostic only, not a semantic proof);
- whether gold text has any lexical support in the declared evidence turns;
- how alignment categories cross-tab with the previously recorded Solver
  difficulty buckets.

The lexical checks are explicitly labelled heuristics.  This module cannot
prove semantic entailment and never promotes or rejects a production QA.

Run:

```bash
python -m scripts.current_method.restart_n5_spell.h4_qa_alignment_audit_v1.run \
  --input-run /root/autodl-tmp/gxy-selfplay/restart-n5-runs/\
conversation_chronicles_episode_178392_h4_qa_mining_probe_v1_phase0_4
```
