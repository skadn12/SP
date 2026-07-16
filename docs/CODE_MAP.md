# Code Map

This document maps the current-method code paths most often needed for
development and review.

## Main Entry Points

- `scripts/run_auto_graph_expansion_round.py`
  - top-level round runner;
  - parses CLI flags;
  - builds the native current-method policy;
  - forwards API/challenger options into generation;
  - can optionally invoke training/evaluation wrappers.
  - Stage64 mode automatically routes generated rows through semantic validation,
    stratified option balancing, and all-position K5 search mining; optional
    `--stage3-batch-closure execute` adds one feedback rewrite and an independent final K5.

- `scripts/current_method/native_challenger_generation.py`
  - dispatches native generation modes;
  - routes `graph_conditioned_api_dialogue` into
    `native_api_graph_dialogue_expansion.py`.

- `scripts/current_method/native_api_graph_dialogue_expansion.py`
  - core graph-conditioned API self-play generator;
  - API graph expansion with JSON-object response mode and validator-guided
    semantic correction;
  - `semantic_graph_v16` cache/gate isolation for the reasoning-ledger, anti-shortcut, exact formal-evidence, neutral graph-fact isolation, and fixed visibility/relation role contracts;
  - API appended dialogue expansion;
  - Stage64 API QA proposal handling;
  - alignment, leakage, length, and schema validation;
  - JSONL output and run summaries.

- `scripts/current_method/semantic_graph_v2.py`
  - canonical, non-mutating Stage3 graph/event validation;
  - operation-specific temporal, bridge, comparison, visibility, count,
    cause, and relation constraints;
  - `semantic_graph_v2` is the payload schema; `semantic_graph_v16` is the
    separate cache/gate namespace.

- `scripts/current_method/rebuild_stage2_canonical_graphs.py`
  - formal Stage2-to-Stage3 parent construction path;
  - runs local Qwen3-4B evidence extraction without paid API calls;
  - sends only evidence turn IDs, speakers, and text to the model, never old
    graph semantic labels, QA, options, or gold;
  - records every legacy event as re-extracted or quarantined and emits
    `stage2_canonical_graph_rebuild_v3` artifacts,
    `stage2_semantic_graphs.jsonl`, and `stage3_parent_v3.jsonl`;
  - applies an independent semantic-audit exclusion manifest before admission;
  - keeps unlinked-dialogue fallback disabled by default; a row with no clean
    linked evidence is quarantined rather than synthesized;
  - keeps long structured parent atoms and accepted `depends_on` edges visible
    to the challenger; only parent-dialogue utterances are omitted.

- `scripts/current_method/audit_stage2_canonical_rebuild.py`
  - read-only, fail-closed audit of source-event disposition, evidence-only
    requests, canonical graph/parent equality, lineage, and heldout identity;
  - never repairs or admits an artifact;
  - the current semantic-audited v3 artifact has 77 graphs, 385 events, 77
    parents, zero eligible event/row rejection, zero fallback events, and zero
    exact heldout overlap; its 171 exclusions include 18 independent
    semantic-audit quarantines, one source row without clean linked events is
    quarantined, and training remains unauthorized.

The formal artifact directory is
`base_restart_current_method_validation/172_stage2_canonical_graph_rebuild_v3_semantic_audited_final`.
The historical `161_stage2_canonical_graph_rebuild_v2_final` passed its older
structural audit, but semantic review found 11 bad events. It must not be used
as a Stage3 parent or graph-only smoke input.

- `scripts/current_method/build_stage3_parent_v2.py`
  - legacy compatibility migration for old parent payloads and diagnostics;
  - strips old QA surfaces but cannot rebuild polluted graph semantics;
  - is not the formal Stage3 parent-construction path.

- `scripts/current_method/stage3_graph_only_smoke.py`
  - paid-API graph-only gate before dialogue, QA, or solver work;
  - preserves raw graph failures and requires explicit API acknowledgement;
  - freezes a complete v16 reasoning target for every requested operation;
  - stops early when the requested aggregate pass rate is unreachable.

The historical v14 `7/7` smoke and v15 run173 `5/7` are not full v16 evidence.
Targeted v16 visibility and relation checks both pass; a fresh seven-operation
v16 gate is still required. After that gate passes, run one small full mining
batch. Immediately
shuffle/balance options and recalibrate every cyclic position at K5 with the
fixed accepted Stage2 solver before increasing generation volume.

## Prompt Locations

- Graph/event-plan API prompt:
  - `scripts/current_method/native_api_graph_dialogue_expansion.py`
  - function: `call_api_graph_expansion`

- Appended-dialogue plus `qa_proposal` API prompt:
  - `scripts/current_method/native_api_graph_dialogue_expansion.py`
  - function: `call_api_graph_conditioned_dialogue_expansion`

- Stage64 QA schema and prompt constraints:
  - `scripts/current_method/native_api_graph_dialogue_expansion.py`
  - search for `qa_operation_answer_type_diversification`,
    `qa_proposal`, and `stage64_target`.

- Alignment repair prompt:
  - `scripts/current_method/native_api_graph_dialogue_expansion.py`
  - function: `call_api_alignment_repair`

## Stage64 API Validation

- Operation and answer-type constants:
  - `STAGE64_QA_OPERATIONS`
  - `STAGE64_TARGET_OPERATION_COUNTS`
  - `STAGE64_ANSWER_TYPES`
  - `STAGE64_OPERATION_REQUIRED_ANSWER_TYPES`
  - `STAGE64_OPERATION_ANSWER_TYPE_CANDIDATES`
  - `STAGE64_REASONING_TARGET_SPECS`

- Solver-visible / generic-question filters:
  - `STAGE64_BAD_SOLVER_VISIBLE_RE`
  - `STAGE64_GENERIC_QUESTION_RE`

- Per-slot operation target assignment:
  - `_stage64_target_for_slot`
  - exact quotas are expanded by `_stage64_interleaved_quota_sequence`.

- API `qa_proposal` conversion and schema validation:
  - `_build_stage64_api_qa_from_proposal`
  - `build_qa_after_alignment`
  - rejects normalized exact duplicate options and gold text mapped to multiple labels.

- Shared Stage64 semantic quality gate:
  - `scripts/current_method/stage64_quality.py`
  - validates graph plans before cache write/reuse, including cache-version isolation;
  - validates answer surfaces, question/answer type alignment, operation/subtype semantics,
    semantic graph slots, multi-anchor requirements, and related hard-negative events.

## API Request Path

- Request construction:
  - `build_json_request`

- API call wrapper:
  - `call_raw_json_api`

- Low-level curl helper reused from the older gate:
  - `scripts/generate_r2_llm_graph_expansion_challenger.py`
  - function: `post_json_with_curl`

## Dataset and Frontier Helpers

- `scripts/current_method/frontier_selection.py`
  - frontier labeling and filtering.

- `scripts/current_method/report_builder.py`
  - generation reports and solver evaluation summaries.

- `scripts/current_method/cross_stage_eval_panel.py`
  - cross-stage heldout comparison panel;
  - verifies whether Stage1 beats base, Stage2 beats Stage1, and a supplied
    Stage3 model beats Stage2 on reliable Stage3 heldout;
  - can rerun original-format verifier evals or summarize existing JSON into
    `cross_stage_eval_report.md`.

- `scripts/current_method/recalibrate_stage64_shuffled_frontier.py`
  - Stage64 post-shuffle recalibration driver;
  - quality-filters exact/ambiguous duplicate options plus Stage64 semantic failures;
  - locks the resolved accepted base-restart Stage2 checkpoint and seed;
  - evaluates all cyclic option permutations with a strict decoding-health gate;
  - separates robust frontier, hard0, easy5, and position-sensitive diagnostics;
  - supports `lineage_group` so descendants of one root remain together.

- `scripts/current_method/shuffle_stage64_options.py`
  - reproducible option shuffle and label balance;
  - balances within option arity × attack skill;
  - removes stale solver/frontier fields.

- `scripts/current_method/accumulate_stage64_reliable_frontier.py`
  - validates each recalibration batch and fixed-solver provenance;
  - rejects non-robust or semantically invalid rows and duplicate IDs;
  - creates one lineage-group split only after the configured threshold;
  - materializes balanced train/heldout only from evaluated cyclic positions;
  - publishes `training_ready_manifest.json` only when every gate passes.

- `scripts/current_method/stage3_operation_controller.py`
  - consumes existing cyclic K5 aggregates without rerunning the solver;
  - updates next-batch `qa_operation|answer_type` shares with exploration floors;
  - reports rollout-weighted accuracy and frontier yield by full reasoning signature.

- `scripts/current_method/stage3_solver_feedback_mining.py`
  - maps easy5, hard0, position-sensitive, and robust-frontier outcomes to explicit actions;
  - freezes graph, dialogue, gold, target identity, and reasoning signature;
  - emits resumable QA-only challenger queues and diagnostic/retention manifests.

- `scripts/current_method/stage3_batch_closure.py`
  - recoverable one-pass QA feedback closure;
  - merges original robust frontier with accepted rewritten QA, reshuffles, and runs one independent final cyclic K5;
  - publishes `final_reliable_batch` and never authorizes training itself.

- `scripts/current_method/build_stage3_train_mix.py`
  - consumes only a ready balanced-frontier manifest;
  - keeps official heldout exclusion-only, selects replay, derives validation
    from train groups, and writes SHA256 provenance.

- `scripts/current_method/external_generalization_report.py`
  - discovers external eval outputs;
  - validates aligned `record_id` sets and greedy settings;
  - reports paired exact McNemar and gold-label-conditioned accuracy.

- `scripts/current_method/seed_audit.py`
  - provenance checks for train/dev/heldout seed rows.

- `scripts/current_method/build_base_restart_stage2_train_mix.py`
  - train-mix construction for current-method repair/top-up runs.

- `scripts/current_method/build_base_restart_stage2_repair_train_mix.py`
  - repair train-mix construction.

## Optional Training/Evaluation Wrappers

- `scripts/run_round_solver_training_pipeline.py`
  - accepts an audited prepared Stage3 train file;
  - binds it to the train-mix summary and SHA256;
  - trains a zero-initialized LoRA on the accepted Stage2 merged base, merges,
    and evaluates; incompatible init adapters are rejected.

- `train_qwen3_lora_sft.py`
  - base LoRA SFT trainer.

- `scripts/eval_v13_pilot_vllm.py`
  - vLLM-based MCQ evaluation.

- `scripts/merge_custom_lora_adapter.py`
  - LoRA merge helper.

## Data and Model Paths

This package includes only the curated reproducibility subset documented under `data/`; it does
not include model weights or the full workspace. Pass deployment-specific paths explicitly:

- `--parent-dir`
- `--base-model`
- `--solver-adapter`
- `--accepted-stage2-solver-model`
- `--accepted-stage2-solver-adapter`
- `--solver-tokenizer`
- `--output-dir`
- `--report-dir`
