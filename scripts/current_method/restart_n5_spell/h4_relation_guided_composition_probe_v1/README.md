# H4 Relation-guided Composition Gate Isolation Probe v1

This isolated probe consumes the frozen POB03 relation-contract transcript and
the frozen Condition B atomic QA batch. Condition A reads the existing pair
compiler artifacts. Condition B selects a pair whose evidence union covers
`fact_A`, `trigger`, and `fact_B`, then adds a relation-guided Composer prompt.

It does not modify or call the production QA Writer, Planner, Realizer, H4
expansion chain, or Memorax. One isolated Composer call is allowed. Solver use
is local-only and limited to the composite full-context K=5 plus remove-A,
remove-trigger, and remove-B ablations.
