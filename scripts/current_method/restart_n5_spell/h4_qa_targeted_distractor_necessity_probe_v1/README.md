# H4 QA Targeted Distractor Necessity Probe v1

This is a QA-only isolation probe. It consumes frozen POB03 atomic QA,
expanded dialogue, and relation-contract artifacts. The Composer wrapper asks
for a causal-revision question that explicitly requests the initial state,
trigger, and revised action, then requires near-neighbor distractors that each
mutate exactly one relation slot.

It does not modify Planner, Realizer, production QA Writer, Solver, Memorax,
the expansion chain, or historical runs. One isolated Composer call is used;
Solver evaluation is local-only.
