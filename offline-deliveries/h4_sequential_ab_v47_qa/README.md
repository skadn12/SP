# v47 evidence before proposition probe

Authorized next step after v46 template comparison: keep the current generator
as the baseline, ask it to output evidence before claims, and run one bounded
2 multi-select + 2 ordering probe on the same complete candidate history.

For each multiselect option, emit turn_ref, speaker, quote before proposition,
relation and rationale. For ordering emit item_evidence before items and gold.
The final field sets and existing gates are unchanged. Raw key order is measured
as a diagnostic only; no extra gate, model answer edit or Gold correction.
The same original full 156-turn/4566-body-token source and frozen prior-training
feedback are retained. Prior inventory remains v46's frozen historical selection
text plus the v44 two selected candidates; no recent Solver answers/scores/Gold.
Both question types start from the same frozen prior-only inventory.

Reviewer instructions/actual request builder stay identical to v46, independent
of writer Gold/evidence. Existing clause anchors, full-only audited M_clean_policy,
exact entire-set dedup and complete_valid_full_set and 0<mean<1 selection remain.
Use .venv-local4b, Qwen/Qwen3.6-35B-A3B only for generator/reviewer. Budget4slots,
max6QwenHTTP, max60fullSolver answers, one existing mechanical repair per type,
no auto HTTP retry, shared durable completion gap >=35s. No ground or training.

Actual run uses prompt_arm=current only (legacy old adapter is retained solely
for compatibility and offline regression tests). It is a single new probe, not
a fresh two-arm comparison. Freeze files before prepare. Prior histories and
formal30-question pool remain immutable, provisional QA yield is reported separately.
