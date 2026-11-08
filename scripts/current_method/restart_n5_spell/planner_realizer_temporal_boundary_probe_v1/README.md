# Planner Realizer Temporal Boundary Probe v1

This isolated A/B probe tests the Realizer bottleneck found by the accepted O1
Planner batch:

- **A**: the already frozen original Realizer canary output;
- **B**: one fresh Realizer call with the same frozen POB03 O1 plan, Parent,
  model, temperature, and output contract, plus explicit scene-local temporal
  boundary constraints.

The B prompt forbids future scene actions and finality in open/provisional
scenes. It does not alter the Planner plan or production code. Each condition
has at most one Realizer request; there is no repair or retry. QA, Solver,
Writer, Auditor, and Memorax calls are zero.

Run:

```bash
python -m scripts.current_method.restart_n5_spell.planner_realizer_temporal_boundary_probe_v1.probe \
  --run-root /root/autodl-tmp/gxy-selfplay/restart-n5-runs/planner_realizer_temporal_boundary_probe_v1_...
```

The existing A run remains immutable. The new run stores the B response and a
zero-API A/B comparison with the strict temporal-boundary audit.

## v2 compact follow-up

`compact_probe_v2.py` is a separate execution variant. It uses the same frozen
 A input and one fresh Realizer request, but replaces the long per-future-scene
 block with compact scene-local rules. It records request attempts separately
 from completed responses, so a timeout is reported as undecided rather than as
 a failed boundary-control result.

The completed v2 canary is:

```text
/root/autodl-tmp/gxy-selfplay/restart-n5-runs/planner_realizer_temporal_boundary_probe_v2_20260822T180000Z
```

Result: `TEMPORAL_BOUNDARY_CONTROL_RECOVERED` (A=false, B=true; one completed
 Realizer response; prompt 5,002 tokens, completion 2,324 tokens). This remains
 a single-root canary and is not a production prompt change.

## Three-root batch

`batch_probe_v1.py` repeats the compact protocol on three already accepted O1
roots (`POB03`, `POB04`, `POS07`). It sends one Realizer request per root and
keeps the source plans/dialogues read-only. The first live batch is preserved
at:

```text
/root/autodl-tmp/gxy-selfplay/restart-n5-runs/planner_realizer_temporal_boundary_batch_v1_20260822T190000Z
```

All three responses were mechanically valid. The initial audit was deliberately
conservative and marked 2 cases down: one explicit future plan was mistaken for
an early action, and `therapist` was not accepted as evidence for `therapy`.
`reanalyze_batch_v1.py` corrected only those audit rules, without API calls or
source edits. The authoritative reanalysis is:

```text
/root/autodl-tmp/gxy-selfplay/restart-n5-runs/planner_realizer_temporal_boundary_batch_reanalysis_v1_20260822T193000Z
```

It passes all three roots under the documented lexical boundary profile. This
is still a Realizer-adherence signal only; it does not establish QA difficulty
or authorize a production prompt change.

## Integration v2 and strict stage audit

`integration_probe_v2.py` appends a plan-derived future-trigger/state reservation lock around the unchanged production builder. The POB04 live canary is at `planner_realizer_temporal_boundary_integration_probe_v2_20260822T210000Z`; it used one Realizer request (prompt 4,749 / completion 2,221 tokens) and passed the strict stage audit after zero-API reanalysis.

The stricter `stage_transition_reanalysis_v2.py` checks not only final-action timing but also whether later scene triggers/states leak into earlier scenes. On the frozen three-root batch it finds 2/3: POB03 and POB04 pass, while POS07 has a real Scene 2 heirloom trigger leak in Scene 1. Therefore the batch remains partial and is not integration-ready for the production prompt.
POS07 was then run through the same v2 wrapper at `planner_realizer_temporal_boundary_integration_probe_v2_POS07_20260822T220000Z`. Its mechanical contract and final-action profile passed, but the strict stage audit failed: Scene 1 still introduced the later heirloom trigger, and the later scene did not preserve the planned trigger ownership. The next probe should release one scene plan at a time instead of adding more text to a single full-plan prompt.
