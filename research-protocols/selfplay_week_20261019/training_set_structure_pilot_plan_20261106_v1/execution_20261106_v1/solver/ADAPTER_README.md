# Structure pilot evaluation adapter

This directory contains the new adapter, its seal schema and CPU semantic fixtures. Engineering does not open current drafts, selections or review results, create experimental requests, load a model, call an API, train or export data. Controller execution begins only after every review is frozen. All new runtime artifacts stay in this solver directory; historical code and artifacts are read-only.

## Input seal

Controller supplies `INPUT_REVIEW_SEAL.json` conforming to `INPUT_REVIEW_SEAL.schema.json`. The top-level fields are `schema_version="structure_pilot_input_review_seal_v1"`, `namespace="structure_pilot_20261106_v1"`, `frozen=true`, an actual `frozen_at_utc`, `reviews_frozen={quality:true,gold:true,structure:true,overlap:true}`, `files`, and `items`.

`files` is the complete immutable dependency inventory, with absolute `path` and byte `sha256` for every raw first draft, source snapshot, author/slot freeze, quality and Gold review, independent final structure/overlap audit and relevant protocol/controller artifact. The adapter streams hashes. Each per-item draft/source/review binding must also occur in this inventory. Include the actual underlying review artifacts, not just a controller summary.

Each item has:

- `candidate_id`, `root_id`, and `group`: one of OLD_F, OLD_G, NEW_F, NEW_G; at most one item per root and group.
- `raw_draft`: `{path,sha256}`. A single candidate object, `candidates` wrapper, or candidate list is supported. Exactly one entry must match the ID.
- `source`: `{path,sha256,turn_count}` for the complete immutable source.
- `quality_status`: PASS/HOLD/FAIL, and boolean `gold_agreement`. Only PASS plus true is eligible. EVAL_HOLD is a separate engineering condition.
- `review_artifacts`: a nonempty list of `{path,sha256}` bindings covering that item's external quality/Gold and final independent structure/overlap artifacts.
- `structure_result` and `overlap_result`: objects with `independent=true`, a `status` string and the audit's actual result fields. These must be the independent post-quality audits, not author or provisional quality-review classifications. Ordinary D structure, historical duplication and overlap status do not exclude an otherwise eligible item.
- `protocol_markers`: a list of deviation descriptions, empty for a clean procedure. Marked cases remain descriptive; scoring carries the markers and sets `clean_contrast_eligible=false`.

The adapter does not semantically re-review artifacts or decide which external judgment is correct. The controller certifies completeness and finality of those reviews in the seal. No fields from certificates, feedback cards, Gold or historical results enter the prompt.

## Normalization and visible interface

Normalization occurs in memory without changing first drafts. Supported aliases include `gold`/`gold_option_ids`, `source_hash`/`source_sha256`, `root`/`root_id`, and `full_order`/`ordering`. Options keep their stable `option_id`; historical `label_to_option_id` plus option `id`/`label` is supported. Stable Gold IDs must actually match options; ambiguous label-to-Gold repair is forbidden.

The original question and option text are copied verbatim. Visible `query_scope`, `query_pairs`, `answer_semantics`, `relation_definition`, `pair_domain`, `instructions` and `visible_instructions` are preserved; structured `query_pairs` endpoints map to displayed labels. The prompt contains the permuted displayed labels and item text, with no original option-ID map. If original question/option/scope prose explicitly refers to a stable ID, the candidate becomes `EVAL_HOLD_STABLE_ID_REFERENCE_IN_VISIBLE_PROSE` with quality PASS preserved; the adapter does not rewrite prose or add an inverse map. Multi-character IDs are checked as bounded tokens. Single-character IDs require explicit reference syntax (such as `option A`, `[A]`, `A<B`, `A and B`, or a structured string equal to the ID), so ordinary articles and pronouns do not trigger a naive token-presence hold. Structured `query_pairs` is the sole ID-bearing visible field adapted automatically. Source turns are shown in their original order and in full. No author evidence or audit is shown.

Partial-order raw `gold_option_ids` stays `[]`; `gold_pairs` is separate. Legacy directed-pair conversion is an in-memory proxy used for display generation only. `query_pairs` must be an explicit nonempty list of **directed** stable-ID pairs, with no self/duplicate pair. It is the exact scored domain; if both directions are queried, list both. `query_scope` and `answer_semantics="all_and_only_entailed_pairs_in_query_scope"` are required. Gold must be acyclic and contain every edge entailed by its own transitive closure that lies in the query domain. External semantic review establishes the source entailment; engineering does not invent omitted relations. Predicted edges are not automatically transitively closed or repaired.

All types use the inherited brief `item_evidence`, `answer_reason`, `final_answer` JSON interface. Ordering instructions and their example name all actual k labels and require each exactly once. Multi-select does not hint at Gold cardinality and permits an empty prediction. Partial order requests the complete entailed relation set within the exact displayed query domain, including queried transitive edges, without forcing unconstrained pairs. Schema/auxiliary reason problems remain diagnostic when final_answer is independently valid.

## Frozen execution

After supplying the final seal, controller may run these commands from this directory:

```bash
python -B adapter.py self-check
python -B adapter.py check-inputs
python -B adapter.py prepare
python -B adapter.py run
```

`check-inputs` verifies sealed inputs and the already-bound M-clean/tokenizer files with streaming SHA-256; it generates no requests and loads no model or tokenizer. `prepare` performs CPU source/question/mapping/display/schema/scorer/format/token-length checks, loads only the fixed local tokenizer, reserves a one-time ledger, then freezes exact prompts, schemas, code/input/model/tokenizer bindings, display permutations and request seeds. Any overlength candidate becomes EVAL_HOLD as a whole; its Full/No-context pair is not partly evaluated, its source is never truncated, and the hold is recorded in PREFLIGHT. Other eligible candidates can remain in the same frozen run. A failed preparation consumes its ledger; do not retry it or create replacement runs.

The namespace is `structure_pilot_20261106_v1`. Every type receives min(6,k!) unique Gold-independent displays, each repeated three times in Full and No-context. Display seed is 2026110603. Request seed is 2026110604 + sorted eligible candidate ID index*1000 + display index*3 + repeat index, paired between context modes. Engine seed is 2026110303; temperature .8, top_p .9, bf16, batch 12, total window 16384 and output cap 6144. The conservative potential-call reservation and final request count cannot exceed 1152.

`run` revalidates all frozen bytes before reserving an exclusive RUN_ATTEMPT and invokes the inherited historical local vLLM loop with only a separate tokenizer path and deferred scoring. There is no API, extra model, automatic retry, resume, extra ablation, training or export. Raw results are written per returned request. An interruption preserves those files. Missing/infrastructure records remain explicit and are not model errors. A second run is always blocked, even if importing/loading the backend failed.

On ordinary completion or caught interruption, scoring runs once. After a hard process kill, controller may run `python -B adapter.py score` once to collect the preserved raw files and mark missing requests; this does not execute or resume inference. Scores are exclusive-create and cannot be replaced. Do not remove ledgers to continue unfinished calls.

## Metrics and boundaries

Multi-select/full-order use historical J; partial order uses historical directed-pair precision, recall and F1. Metrics are always reported by type; no J/F1 average is produced. Exact whole-answer accuracy is also reported. Invalid returned final answers receive zero in the fixed executed/returned-request denominator. Missing and infrastructure requests are excluded from that denominator, reported alongside planned counts, and make planned-denominator accuracy unavailable rather than counting as model errors.

The historical ≥2-Gold transport/scorer gate is removed in memory to permit singleton sets. Empty multi-select Gold has undefined J in the existing scorer and becomes EVAL_HOLD, with quality PASS preserved. For all-Gold multi-select the existing scorer explicitly defines J as TP/g, which is retained. Empty partial-order Gold is allowed: the historical pair scorer defines empty-prediction/empty-Gold as exact=true with precision=recall=F1=0. This boundary convention is retained and separately visible; no edge or new metric formula is invented. The fixtures verify these cases.

Runtime outputs are in `evaluation/`: ITEMS_PRIVATE, PERMUTATIONS_PRIVATE, REQUESTS, prompts/schemas, MODEL_CONFIG, PREFLIGHT, INPUT_BINDINGS, PLAN_FREEZE, per-request raw, RAW_ANSWERS and SCORES. ATTEMPT_LEDGER and RUN_ATTEMPT are in the solver root. No training artifact is produced.
