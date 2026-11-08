# Three-root F retry

This independent exploration was requested for Train roots 189272, 249232 and 59126. It does not change the earlier run's budget or overwrite prior artifacts.

- Generate at most one new F first draft per root, or record no safe candidate.
- Reuse the previous F supplement author, blind review, source qualification, scoring, and semantic deduplication contracts. Authors receive full frozen sources, neutral source information, and the existing abstract feedback only.
- Freeze semantic targets before first drafts. Keep authors blind to historical questions, Gold, reviews, and solver results. Blind reviewers solve before seeing Gold.
- Evaluate only quality- and source-qualified questions with the previously frozen local Qwen3-4B plus M-clean setup, at most 36 calls per candidate, 108 total. Preserve fixed planned denominators and no answer-dependent retries.
- Compare to the existing training export, past candidates, and the separate 18 supplement candidates after authoring. Existing item FS1_59126_01 is not automatically included.
- Record decisions and provenance separately. Do not modify the sealed training export or start training.
