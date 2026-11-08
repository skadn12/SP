# QA Fact-Bundle Writer Probe v1

This is an independent, offline QA branch. It does not modify the existing
position-first Writer or any Expansion/Memory/Solver implementation.

## Contract

The future Writer selects only a frozen `bundle_id`, one `gold_fact_id`, and
three `distractor_fact_ids`, together with the question wording and semantic
labels. It does not emit option text, answer letters, evidence IDs, or source
positions. The program derives all option text and provenance from the frozen
fact records, then emits the existing MCQ input shape.

## Frozen shadow preflight

Input:

`/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conversation_chronicles_episode_146700_qa_fact_first_participant_rebundle_v1_offline`

The preflight used 43 facts and 60 participant-connected bundles, including
55 four-fact bundles. Three reference-only Writer rows were materialized with
zero API calls and accepted by the existing focus-writer/Solver input
contract. See `reports/FACT_BUNDLE_WRITER_PREFLIGHT.md` and the JSON report in
the preflight run directory.

No live Writer or Solver probe was run by this branch.
