# F Supplement Run 01

Status: frozen scope and scoring protocol; candidate authoring has not started.

## Objective

Search for at most one additional F candidate in each of the 68 frozen Train roots. This is a new, independent exploration round. Revisit roots that reached the previous two-draft cap as well as roots with zero or one previous F draft. Do not stop coverage because an earlier batch yields few or no eligible candidates. A root may produce no safe candidate.

The run does not edit, reopen, or replace any artifact in `training_set_rebuild_20261104_run01` and does not update a training export.

## Frozen budget

- Roots: all 68 entries in the prior run's `ROOT_COVERAGE.json` and `SOURCE_FREEZE_ALL68.json`.
- New first drafts: at most one per root; at most 68 overall.
- Solver requests: at most 36 per quality-passing item: 18 full-context and 18 no-context. Sampling is at most six unique option permutations, three repeats each, per mode; lower option counts may produce fewer calls.
- Maximum solver requests: 2,448. No retries; invalid and truncated answers stay in fixed denominators.
- No expansion beyond these limits without a later explicit user instruction.

## Authoring and review isolation

Authors receive the complete frozen dialogue, neutral source information, and abstract search guidance only. They must not receive previous questions, Gold answers, blind answers, or solver outputs. The author freezes the semantic target and Gold before wording the question. Natural information needs take priority; do not force a question type, Gold size, or difficulty. Preserve a root-level no-safe-candidate result and its reason.

Every candidate must pass mechanical binding checks, blind quality review, and a cited-source qualification review before evaluation. Exact quotes, speaker, turn reference, source version, attributed claims, unresolved conflicts, and proposal/confirmation/completion distinctions must be checked. Whole-root rejection is not inferred from a warning; a restriction blocks only questions that rely on the restricted fact.

## Solver scoring

The primary outcome is task-answer performance in the answer format used for training: one complete `<box>…</box>` final answer. Multi-select and full ordering retain their frozen J metrics; partial ordering uses directed-pair F1. A malformed, incomplete, truncated, or semantically invalid final answer scores zero in the planned-call denominator. No answer repair or retry is allowed.

Structured explanation and citation compliance are recorded as a separate diagnostic dimension and do not veto an otherwise valid final answer. In particular, an explicit `not_supported` judgment may have no positive-turn citation; its rationale must state what evidence was checked and that support was not found. Supported, contradicted, or ambiguous claims still require accurate references when requested. This separation follows the answer-only training target and avoids treating untrained auxiliary JSON behavior as task-answer correctness.

Use the same verified local M-clean weights and sampling settings as the prior run. Keep Full and No-context paired; No-context is diagnostic and cannot independently disqualify an item. Eligibility for the supplement pool requires a recorded outcome for every planned Full-context call and a non-boundary mean task score (strictly between zero and one, with invalid answers scored zero). A single invalid final answer does not veto the entire candidate. Report answer-validity counts alongside scores. Quality, source eligibility, evaluation value, and union-selection status remain separate fields.

## Semantic deduplication and outputs

After quality, source, and solver gates, compare each eligible new candidate against the prior F/G candidate universe, reusable legacy examples, and the sealed training export. Compare task intent, queried relation, entities, evidence, option facts, and answer structure; wording changes alone do not establish novelty. Record near-duplicate families and retain the best coverage representative. Do not use solver score when making the semantic-family judgment.

The final report must include the 68-root coverage ledger, draft and gate counts, issue-level hold reasons, answer-only and auxiliary-protocol metrics, source bindings, semantic-family decisions, type counts, request count, and checksums. No training is started by this run.
