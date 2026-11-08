# v43 candidate dialogue to bounded QA/Solver probe

The user authorized the next proposed step: run a small question-generation and
existing Solver-selection loop over the completed v43 fresh B2 candidate history.
The source is the full 156-turn / 4566-body-token conversation, including a formal
132-turn prefix and two retained provisional scenes. No original dialogue,
reviewer verdict, scene disposition or prior pool is rewritten.

This run has two multi-select and two ordering candidate slots. Generation uses
Qwen/Qwen3.6-35B-A3B, with at most six HTTP calls including one mechanical repair
per question type and one existing independent review per valid ordering question.
There is no multi-select semantic reviewer, automatic transport retry, source
replacement or automatic run restart. The generation phase shares the expansion
lock and durable response-completion clock; HTTP calls remain at least 35 seconds
apart after the preceding response completes. All requests receive the full
source and only frozen previous-training-round feedback.

Solver evaluation reuses the exact audited M_clean_policy checkpoint, the proven
`.venv-local4b` environment and established full-only policy: multi-select has six
display permutations and three answers each; ordering has twelve permutations
and one answer each. At most 60 full-context answers are planned. Selection is
`complete_valid_full_set and 0 < full_partial_mean < 1`; no ground calls, new
difficulty threshold or training run is introduced.

The resulting new pool is explicitly provisional because it uses provisional
source dialogue. Report generated questions, questions ready for Solver,
selected unique questions, source admission and training execution separately.
Selected questions preserve the original question, full Gold, complete source
and raw evaluation receipts. No selected question silently promotes the upstream
dialogue. The existing 30-question formal pool remains a prior deduplication input.

Offline validation and immutable generation inputs must finish before live calls.
Use `generate.py prepare/verify/generate` and
`evaluate_and_pool.py preflight/verify/live/finalize`, with a fresh run directory
for each stage. Live commands are coordinated by the root agent; tests must mock
network and Solver boundaries. Results and content analyses belong in `results/`,
outside the frozen implementation.
