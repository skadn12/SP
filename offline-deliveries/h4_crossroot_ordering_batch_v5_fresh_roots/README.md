# Fresh-root ordering pilot v5

This independent pilot uses train roots 210046, 231899, 194259, and 96544.
Originally proposed roots 198141 and 232414 are dev in the authoritative
`root_split_manifest.jsonl` and are excluded. The frozen scope is one initial
four-slot generator request per root, with abstention allowed.
The initial request has up to four slots per root. Content-reviewed top-ups
may add two slots twice, with eight total as a ceiling, not a quota. The
generator remains SiliconFlow `Qwen/Qwen3.6-35B-A3B`; no Solver or training
is part of this directory.

Run `python capacity_screen.py` for a read-only source, historical-pool,
and split inventory. Review coherent unused six-turn sequences and evidence
semantics before creating `manifest.json` and `CAPACITY_SCREEN.json`.
The screen must have status `APPROVED`, list precisely the approved train
root IDs, and bind the manifest and split-manifest SHA-256 values.
`generate.py --prepare` refuses dev roots or requests without that approval.
Each prepared prompt includes root-specific screened topic/ref priorities and
the screen's SHA-256. They guide generation but do not establish novelty.

The pilot uses each root's original candidate pool of all question types for
deduplication. No v3/v4 drafts exist for these four roots. A historical
candidate is a novelty warning, not evidence of training admission. Mechanical
validation checks exact source refs and quotes, speaker identity, six distinct
turns including R01, a shuffled display, and Gold recomputed from source turn
order. Human review still decides entailment, topical coherence, and novelty.
