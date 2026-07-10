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

- `scripts/current_method/native_challenger_generation.py`
  - dispatches native generation modes;
  - routes `graph_conditioned_api_dialogue` into
    `native_api_graph_dialogue_expansion.py`.

- `scripts/current_method/native_api_graph_dialogue_expansion.py`
  - core graph-conditioned API self-play generator;
  - API graph expansion;
  - API appended dialogue expansion;
  - Stage64 API QA proposal handling;
  - alignment, leakage, length, and schema validation;
  - JSONL output and run summaries.

## Prompt Locations

- Graph/event-plan API prompt:
  - `scripts/current_method/native_api_graph_dialogue_expansion.py`
  - function: `call_api_current_event_plan`

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

- Solver-visible / generic-question filters:
  - `STAGE64_BAD_SOLVER_VISIBLE_RE`
  - `STAGE64_GENERIC_QUESTION_RE`

- Per-slot operation target assignment:
  - `_stage64_target_for_slot`

- API `qa_proposal` conversion and schema validation:
  - `_build_stage64_api_qa_from_proposal`
  - `build_qa_after_alignment`

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
  - verifies whether Stage1 beats base on Stage1 heldout, Stage2 beats
    Stage1 on Stage2 heldout, and Stage64 beats Stage2 on Stage64 heldout;
  - can rerun original-format verifier evals or summarize existing JSON into
    `cross_stage_eval_report.md`.

- `scripts/current_method/recalibrate_stage64_shuffled_frontier.py`
  - Stage64 post-shuffle recalibration driver;
  - clears stale `solver_eval` / `frontier_label` fields;
  - reruns a fixed Stage2 solver on shuffled clean samples;
  - rebuilds frontier, hard0, easy5, and grouped train/heldout split files.

- `scripts/current_method/seed_audit.py`
  - provenance checks for train/dev/heldout seed rows.

- `scripts/current_method/build_base_restart_stage2_train_mix.py`
  - train-mix construction for current-method repair/top-up runs.

- `scripts/current_method/build_base_restart_stage2_repair_train_mix.py`
  - repair train-mix construction.

## Optional Training/Evaluation Wrappers

- `scripts/run_round_solver_training_pipeline.py`
  - build train data, train LoRA, merge adapter, evaluate.

- `train_qwen3_lora_sft.py`
  - base LoRA SFT trainer.

- `scripts/eval_v13_pilot_vllm.py`
  - vLLM-based MCQ evaluation.

- `scripts/merge_custom_lora_adapter.py`
  - LoRA merge helper.

## Data and Model Paths

This package does not include data or model files. Pass those paths explicitly:

- `--parent-dir`
- `--base-model`
- `--solver-adapter`
- `--solver-tokenizer`
- `--output-dir`
- `--report-dir`
