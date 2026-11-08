# H4 Relation-aware Composition Probe v1

This is a fully isolated QA-level probe. It freezes the eligible atomic QA
pool from POB03, asks a Candidate Selector to propose up to five composable
pairs/groups, validates each proposal against the source QA and evidence, and
then asks an isolated Composer for at most one MCQ per candidate.

The selector is not told about Q17+Q21. That pair is recorded only as a hidden
offline sanity-check in the final report. Contribution-level evidence closure
is used for ablation: shared background turns remain, while each source's
unique contribution closure is removed separately.

This probe does not modify Planner, Realizer, atomic QA Writer, production QA,
Solver, Memorax, expansion runs, or historical artifacts. A generated and
strict-admitted composite is evidence about QA-level automatic composition;
it is not by itself proof that H4 state-evolution expansion succeeded.
