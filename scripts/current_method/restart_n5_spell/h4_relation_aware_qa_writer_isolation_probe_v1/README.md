# H4 Relation-aware QA Writer Isolation Probe v1

This is an isolated A/B refinement probe for the H4 relation-contract canary.
It does not modify the production Writer, Planner, Realizer, H4A/H4B modules,
Solver, admission, Memorax, historical runs, or the frozen transcript.

Condition A uses the existing production Writer/Solver artifacts from the same
POB03 relation-contract transcript. Condition B calls one independent Writer
request with an appended relation-aware wrapper, then sends the frozen Writer
JSON through the unchanged QA Solver/grounding/K=5/composition path via the
existing replay interface.

The relation contract is passed as evidence aliases for `fact_A`, `trigger`,
`fact_B`, and `resolution`; the Writer is required to prioritize questions that
ask why/how the state changed. The output schema is unchanged.

The audit keeps separate counts for broad relation evidence coverage, complete
`fact_A + trigger + fact_B` binding, and difficulty-qualified binding after the
unchanged easy-rejection gate. A question citing multiple relation groups is not
treated as a difficult composite unless it survives that gate.

Necessity ablation is not claimed merely from `Composer-ready`; a future
Composer response must pass full-context, remove-A, and remove-B Solver views.
