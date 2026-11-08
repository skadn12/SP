# v44 bounded QA prompt repair

The user authorized the next step after v43 QA: improve action-state wording,
support/contradiction labeling and continuous evidence quotes, then run a new
two-multi-select / two-ordering probe on the same complete v43 conversation.

Source: `h4_sequential_ab_v43/runs/fresh_b2_grounded_01/CANDIDATE_HISTORY.json`,
156 turns / 4566 body tokens including 24 retained B turns / 894 body tokens.
No new dialogue or external source is collected by this QA run.

The new prompt guidance is generic. It does not supply scenario answers, repaired
questions or Gold, and does not import recent Solver answers or scores into the
frozen previous-training-round feedback. The actual model responses remain raw.
Keep the current schemas, diagnostic-only multiselect citations, existing
ordering review, exact statement-set deduplication and Solver selection rules.

The live budget is four candidate slots, at most six Qwen HTTP calls, and at most
60 full-context M_clean_policy answers. Qwen/Qwen3.6-35B-A3B remains the only
generator/reviewer model. Generation shares the durable 35-second completion gap
and lock with v43. No automatic transport retry or run replay. Use the exact
audited M_clean checkpoint through `.venv-local4b/bin/python`; no ground calls or
training. The source remains provisional and original scene dispositions stay
unchanged. Report actual selected question count separately from source admission.

Freeze all implementation and tests before preparing the new run. The old v43
code, runs and reports are bound in BASELINE_FILES.json and remain read-only.
New final reports belong under results/root104814_prompt_repair_01.
