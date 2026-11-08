# QA Memory-only Probe v1

This is an isolated QA-only experiment. It is **not** the main QA route and
does not replace `simple_append_memorax_v1`, `QA-FACT-FIRST`,
`QA-POSITION-FIRST`, or `QA-FACT-BUNDLE-WRITER`.

## Frozen scope

- Input: the existing frozen `episode-41625` Memory snapshot and matching
  complete expanded dialogue from
  `conversation_chronicles_episode_41625_memory_only_fast_expansion_v1_live_1`.
- Writer input: Memory records only. The complete dialogue is deliberately
  excluded from the Writer prompt and is reserved for a later truth audit.
- No Expansion, Planner, Realizer, Memorax Add/Search, or Solver is called by
  the zero-API preflight.
- Historical runs and all other modules remain read-only.

## Why this branch exists

The existing Memory-augmented QA route uses Memory only as a topic hint while
the complete dialogue remains the authoritative Writer source. This branch
tests a different, explicitly diagnostic hypothesis: can Memory alone produce
useful question candidates, and how many require transcript verification later?

Memory-generated candidates are marked `memory_grounded=true` and
`transcript_verified=false`; they must not be treated as final training data
without the later transcript audit.

## Contract

The Writer emits only question text, four options, `gold_text`, a question
type, and local Memory references. The program derives `correct_option`, QA
IDs, and the deferred evidence status. Distractors may be controlled
single-variable near-misses rather than verbatim transcript facts; this is
intentional for this diagnostic and is reported separately from transcript
grounding.

Run zero-API preflight with:

```bash
python -m scripts.current_method.restart_n5_spell.simple_append_memorax_v1_qa_memory_only_probe_v1.preflight
```

The first diagnostic Writer-only live run is recorded at
`conversation_chronicles_episode_41625_qa_memory_only_probe_v1_live_1`; it
used the pre-fix question-type wording and produced no mechanically valid
rows. The isolated follow-up is recorded at
`conversation_chronicles_episode_41625_qa_memory_only_probe_v1_live_2`.
It used one Writer request and zero retries, then stopped before Solver or
admission. `live_2` produced 12 mechanically valid rows; the offline
transcript truth audit retained 6 candidates and rejected/reviewed 6. The
verified candidate file is `live_2/qa/transcript_verified_candidates.json`.
Neither run may be resumed or overwritten.

The six verified candidates were replayed locally through the existing Solver
grounding and full-context K=5 frontier at
`conversation_chronicles_episode_41625_qa_memory_only_probe_v1_solver_replay_1`.
The replay used no paid API requests and admitted zero items: three candidates
were rejected for no-context semantic leakage, and three were `5/5` full-context
easy items.
