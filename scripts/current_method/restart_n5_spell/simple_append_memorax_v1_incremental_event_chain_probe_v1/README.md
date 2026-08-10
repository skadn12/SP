# Incremental Event Chain Probe v1

This isolated branch tests one Parent-only event-chain initialization followed by
one Fresh R1 Planner/Realizer pass and a Child-only delta update. Memorax is a
frozen non-authoritative input; the branch never calls or writes Memorax.

The live request budget is fixed at one Chain Initializer, one Planner, one
Round Realizer, and one Delta Updater request, with no retries. It does not run
QA, Solver, R2/R3, repair, or training.
