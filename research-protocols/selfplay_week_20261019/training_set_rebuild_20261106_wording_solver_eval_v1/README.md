# Isolated wording Solver evaluation

Research diagnosis only. Five V1 rewrite candidates are bound to their frozen
original drafts and sources. Root 45849 is held for unresolved ordering
ambiguity. Duplicate findings are preserved: a completed dedup comparison does
not mean the candidate is novel or eligible for training. No training export.

`EXTERNAL_REVIEWS.json` must bind the five exact candidate objects, exact source
hashes and real independent review artifacts. Both quality and source checks
require blind review; dedup PASS means comparison performed, not novel. Only
items with all three PASS are evaluated. The comparison has incomplete
historical coverage and review is same-model, not external human review.

The model is the historical exact merged M-clean and its paired Qwen tokenizer.
For each selected item use full/no context, six gold-blind display permutations,
three paired repeats, at most 36 calls per item, 6144 output tokens, answer-only
scoring and auxiliary explanation/citation diagnostics. Four eligible items
reserve 144 calls. No-context is diagnostic. Answer accuracy cannot establish
independent correctness or novelty, and no original-wording Solver comparison
is implied.

Use `/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python`:

1. `python make_review_manifest.py` creates the manifest once from the completed
   review records. It fails if the output already exists.
2. `python adapter.py self-check` runs offline synthetic parser/permutation checks.
3. `python adapter.py check-inputs` verifies review, source, candidate and full
   model/tokenizer identity hashes; this can take time and makes no model calls.
4. `python adapter.py prepare` reserves the single attempt and freezes prompts,
   schemas, inputs and hashes. It does not load the GPU model. A failed or partial
   prepare consumes the attempt; do not overwrite or retry silently.
5. `python adapter.py run` runs the historical local executor and scores on
   completion. Never restart a partial run; STARTED and per-request outputs remain.

Run and score verify prepared inputs via their immutable bindings. Files are
created exclusively, with no automatic retry. A result from this evaluation is
not a training admission decision.
