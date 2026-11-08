# Planner Realizer Adherence Canary v1

This isolated canary asks whether one accepted fixed-capacity O1 plan can be
realized as dialogue while preserving its three-stage state chain.

Frozen input: `POB03:O1` from
`planner_objective_probe_v1_batch_20260822T110000Z`.

The canary reuses the existing production Realizer prompt builder and output
normalizer. It permits exactly one Realizer request, with no repair or retry.
Planner, QA, Solver, Writer, Auditor, and Memorax calls are forbidden.

The offline evidence audit checks the observable sequence:

1. initial glaze proposal and ingredients;
2. a too-thick trigger followed by adding water and reconsidering consistency;
3. final application to the ham and placement in the oven.

The final gate also rejects explicit finality, glaze application, basting, or
timer setup before scene 3. This prevents a lexical-coverage pass from hiding
the same premature-closure behavior under test.

This only tests plan-to-dialogue adherence. It does not establish novelty or
grounding because some frozen event content overlaps the accepted plan.
