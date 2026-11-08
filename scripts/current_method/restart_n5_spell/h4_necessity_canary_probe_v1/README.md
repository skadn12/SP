# H4 Necessity Canary Probe v1

This isolated probe reuses the existing scene-local Realizer adapter for one
POB03 canary, but changes only the refinement contract. Its resolution is an
operational follow-through and is forbidden from restating the revised-state
answer (`water`, `thin`, `spreadable`, etc.). It does not modify H4A/H4B,
Planner, production QA Writer/Solver, Memorax, or historical runs.

The canary is eligible for QA only when its source audit confirms:

- fact_A, trigger, and fact_B use disjoint evidence windows;
- S1 does not leak trigger/revision;
- S2 contains trigger and revised state;
- S3 contains no revised-state answer anchor.
