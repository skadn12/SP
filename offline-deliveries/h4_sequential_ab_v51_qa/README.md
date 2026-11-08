# v51 offline QA over provisional A/B dialogue

Run only after all three v51 expansion roots have finished and each has
`CANDIDATE_HISTORY.json` plus `COLLECTION_BINDINGS.json`. For each of
`cc-train-episode-138343`, `cc-train-episode-135466`, and
`cc-train-episode-210046`, `batch.py prepare` takes that root's audited history
and `--root`. Each root has its own direct child under this directory's `runs/`.
The source is the complete original+R01 prefix followed, in order, by actually
retained provisional A and B dialogue; `sources/ROOT.json` in a generated chunk
holds its full bound copy. No R02 A/B history is inherited from other runs.

The fixed QA budget per root is five chunks of two multiselect and two ordering
slots: 20 slots total (10+10), at most 30 Qwen HTTP calls and 300 full-context
Solver calls. All generation precedes Solver evaluation. `SLOT_FOCUS_PLAN.json`
records requested A1/B1/A2/B2/A3/B3 scenes, scenes actually present, missing
scenes, and `expansion_complete`. When at least one new scene exists, the 20
soft topic hints cycle over the actual available scenes. Missing expansion
scenes do not trigger replacement generation. With no new scene, prepare stops.
Every prompt still contains the complete source history and frozen prior-round
feedback; scene hints are not evidence or admission gates.

The only citation spelling compatibility removes one exact current-root prefix
from a quoted `turn_ref` when the remaining ref identifies exactly one source
turn, the quote occurs exactly once in that turn, and an optional speaker
matches. Writer output, reviewer output and normalization records remain
bound. This applies to six writer option/item citations and reviewer primary
and alternative quoted anchors. Unquoted reviewer `support_refs` retain the
existing exact-ref check. The independent reviewer may return a single object
inside `reviews`; wrapping it as a one-element list leaves every old schema,
semantic flag, alternative-order and Gold check in force. Multiselect citation
findings remain diagnostic, and raw relations still determine its Gold.

`batch.py generate`, `evaluate`, `finalize`, and `verify` operate on the
prepared run. `training_pool/REPORT.json` records actual selected and unique
counts; `training_pool/TRAINING_POOL.jsonl` contains only provisional QA-selected
questions with the full source context. No scene or question is training-admitted
here. Tests are offline mocks; they do not call Qwen, Solver or the network.
