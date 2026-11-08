# Expansion / QA Route Contract Freeze

This document is the route boundary for the current self-play experiments. It
is a contract map, not a claim that every historical probe passed.

## Canonical Expansion Route

The current Memory-driven expansion route is:

```text
Memorax event/person context
-> Planner
-> programmatic candidate/materialized-plan checks
-> Semantic Gate
-> selected-candidate isolation
-> one Round Realizer
-> Adherence / State Transition audit
-> optional failed-scene continuation under the existing budget
-> append-only Child
-> Child Add only after the accepted output is frozen
```

The expansion route does not generate QA fields. It must preserve:

- new people/events are allowed when naturally connected;
- exact speaker names are bound by the local speaker registry;
- a selected candidate is the only candidate sent to Realizer;
- rejected candidate beats/outcomes cannot reach Realizer or Child;
- Planner/Gate/Realizer retry budgets are explicit and zero-retry by default;
- complete Parent+Child assembly remains append-only.

The implementation under test is:

```text
simple_append_memorax_v1_memory_event_continuation_probe_v1
```

Its live status must be taken from the individual run report. This freeze does
not convert a failed probe into a success.

## Canonical QA Route

The canonical downstream QA route is direct Spell-style MCQ QA:

```text
complete Parent + Child dialogue (authoritative QA source)
-> one direct MCQ Writer
-> deterministic schema/evidence materialization
-> local no-context Solver
-> full-context Solver frontier K=5
-> admission / training-candidate classification
```

The canonical implementation is:

```text
simple_append_memorax_v1/qa_live.py
simple_append_v1/qa.py
simple_append_v1/prompts.py
```

Its Writer contract uses:

```text
correct_option: "A" | "B" | "C" | "D"
```

The final option position is programmatically rotated by
`assigned_answer_letters`; raw Writer position bias must not be confused with
the final materialized dataset.

Evidence mismatches are recorded and classified by the existing QA admission
logic. They are not grounds for importing the Structural v19 contract.

## Experimental QA Route (not canonical)

`simple_append_memorax_v1_fullr1_devdiv_qa_structural_v19_probe_v1` is an
isolated Structural QA experiment. It uses a different Writer contract:

```text
gold_text = full option text
```

Its `gold_text` field, reasoning-type contract, input override, and completion
budget are not part of direct QA. A v19 result must never be copied into the
canonical direct QA route without an explicit migration and a new preflight.

## Required Preflight Rule

Before any new live call, run:

```bash
python -m scripts.current_method.restart_n5_spell.route_integrity.preflight
```

`decision=needs_baseline_freeze` is a hard stop. It means the working tree is
dirty or a route boundary check failed. It is not permission to repair in
place. First create a clean, explicitly named branch/worktree or freeze the
current file-level SHA manifest; then run the intended route-specific
preflight.

No route may silently change the other route's prompt, schema, solver, memory,
or retry contract.

