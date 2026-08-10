# Simple Append v1

This is an isolated low-cost branch prototype. It does not modify or import
the `world_memory_slim_v1` orchestration path.

The frozen scope is:

```text
authoritative unified parent
-> final one or two complete parent sessions
-> compact 2-3 scene planner outline with distinct developments
-> one round-level Realizer response
-> local structural checks
-> append all child scenes after the parent
-> complete Parent+Child QA source
-> one batched MCQ prompt
```

The first version deliberately excludes historical insertion, parallel
placement, a scene DAG, World Memory, Binder, transactions, and API transport.
New people, events, and locations may appear naturally in planned dialogue;
they are not required to create Memory records.

The complete parent is always preserved in the materialized dialogue and is
used with the child as the QA fact source. Only generation context is narrowed
to the final source sessions.

For three scenes, the soft generation target is 14-18 turns per scene and
42-54 turns overall. Every scene has a hard minimum of 9 turns. The scenes
remain one connected arc, but the second must
shift to a different connected event, person, location, practical problem, or
information state, and the third must produce a result or consequence instead
of replaying the first topic.
