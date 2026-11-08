# v50: 20 question slots with separate scene and topic preferences

This batch gives each candidate ID a different soft topic preference, covering
three existing A scenes and two retained B scenes. The complete 156-turn,
4566-body-token history remains in every generator, reviewer and Solver request.
No new dialogue is generated, and source scenes are not promoted to training.

Five sequential chunks each request two multi-select and two ordering questions:
c01 targets A:S01, c02 B:S01, c03 A:S02, c04 B:S02, and c05 A:S03. Each scene receives
two different topics for each question type. `slot_focus.py` contains neutral
topic descriptions and preferred scene references, without answers, quotations,
Gold or relation labels. These preferences replace the previous uniform priority
for new B scenes. They are generation guidance only: other source turns remain
usable, and no candidate is rejected for departing from its assigned focus.

`batch.py prepare` freezes `SLOT_FOCUS_PLAN.json` and the complete historical prior.
Each chunk receives its four candidate-ID preferences through
`generate.prepare(..., slot_focus_map=...)`. Generator manifests and
`SLOT_FOCUS.json` bind the map; initial request snapshots bind the actual payload.
`generate.focus_context(manifest, context)` supplies the exact preferences for
pending IDs, including repairs, and sets the old generic `priority_new_scene_refs`
field to an empty list. Repairs keep the same preference for each retained ID.
No focus map is added to independent reviewer requests.

The v48 evidence-first/condition instructions, full frozen prior-round feedback,
Qwen/Qwen3.6-35B-A3B model, temperature 0.25, output schemas, independent ordering
review, exact-six-statement dedup, and full-only M_clean_policy scoring are reused.
The existing thresholds and admission rules are unchanged. One mechanical repair
per question type is allowed; transport retries and replacement slots are not.

The historical prior includes the formal pool and unchanged selected candidate
pools from v44, v47, v48 and v49 (11 prior selected candidate questions in total).
Only public question text enters dedup prompts. Subsequent chunks additionally
receive cumulative public projections of earlier mechanical-pass drafts, with no
Gold, evidence, relation labels, scores, answers or current Solver feedback.

Total budget: 20 candidate slots, at most 30 Qwen HTTP requests and 300 full Solver
answers. All five generation chunks finish before any Solver evaluation begins.
Each stage has a single-execution marker; interruption or failure is recorded
without automatic restart. Finalization combines all five verified candidate
pools, preserves original rows and Gold, and copies contexts and relative
provenance snapshots. The result remains a provisional QA candidate pool.

Run with `/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python`:

```text
batch.py prepare --run runs/root104814_focus20_01
batch.py generate --run runs/root104814_focus20_01
batch.py evaluate --run runs/root104814_focus20_01
batch.py finalize --run runs/root104814_focus20_01
batch.py verify --run runs/root104814_focus20_01
```

Preparation, verification and reporting make no model calls. `generate` makes
Qwen calls; `evaluate` runs the fixed audited Solver.
There are no ground/no-context calls and no training calls. Machine selection does
not certify semantic correctness. Reports distinguish duplicate drafts, rejected
reviews, transport failures, perfect Solver scores and selected candidates.
