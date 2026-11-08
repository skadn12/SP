# v89 shard engine inputs

The orchestrator owns actual source selection, independent reviews, parent assets,
scene planning and reporting. This directory supplies immutable code templates.
It does not launch generation or Solver work.

## Installation and shard size

Copy production with python -B engine/install_shard.py /absolute/shard/path.
Execute the installed batch.py, source_admission.py and source_subset.py in that
directory. --home must equal the installed code directory. The shared engine
directory cannot own a production run. Reinstallation or overwriting code fails.

The unchanged v79 engine requires **2–4 distinct root families**. Four is the
target. If division leaves one root, repartition the previous four as three plus
two. If the entire available pool contains only one READY source, admission can
be frozen with SOURCE_READY_AWAITING_SECOND_FAMILY, but input freeze and
generation remain blocked. No runtime AST patch or change to v79 is allowed.

For n roots, budgets are 3n total generation HTTP, 7n candidates and 114n
calls per model. Multiselect sub-budgets remain 2n, 5n, 90n; ordering has two
slots per root and twelve Solver calls per question. All retries and repairs
remain zero, with at most one explicit multiselect supplement per family.

## Admission input

Create ADMISSION_INPUT.json in each shard. Required structure:

    {
      "version": "v89_shard_admission_input_v1",
      "shard_id": "shard_01",
      "split": "train",
      "pilot_plan": {"path": "/absolute/PILOT_PLAN.json", "sha256": "..."},
      "source_gate": {"path": "/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_source_consistency_minfix_v77/source_gate.py", "sha256": "2c5d6b8bd0eeb50de96bb37800f0599ae6db9b5764a7a3e181b542e89c40e58e"},
      "roots": [
        {
          "root_id": "cc-train-episode-123456",
          "split": "train",
          "source": {"path": "/absolute/full/source.json", "sha256": "..."},
          "review": {"path": "/absolute/source/REVIEW.json", "sha256": "..."}
        }
      ]
    }

Every review must pass unchanged v77 validate_review: exact complete source,
all turns and units, all consistency checks, exact findings, recorded exposure,
and decision READY. Root IDs excluded by the pilot cannot be selected. The
validator checks records and bindings, not the truth of newly generated answers.
Run source_admission.py finalize to produce the shard SUMMARY and COMPLETE.

## Remaining input files

Preserve v88 CONFIG and CONFIG_TEMPLATE structure with these substitutions:

- version: v89_full_turn_mixed_shard_v1; unique alphanumeric/underscore batch ID.
- Dynamic root slots, budgets and distributions. target_counts is [2,3,4,5].
- v79 control_count is **6**, meaning six-Gold control, not six questions.
- source_consistency.upstream_complete binds the new shard admission COMPLETE;
  selected roots match both lanes in the same order.
- Parents: parent_pool_path, parent_review_path and parent_provenance_path
  plus SHA fields; independent parent review must precede constructing the private
  six-supported parent. v79 verifies six labels, source references and citations;
  later per-round and complete-question source review remains mandatory.
- Historical source paths may remain external. Source and history bytes must be
  identical complete dialogue; do not truncate sources or feedback.
- Historical public priors retain every unique prior whole question, including
  applicable rejected drafts. All paths are frozen before preparation.
- Registry and actor policy retain the source-name-only contract with unknown
  gender/pronouns; registry adds no facts or Solver context.

SCENE_PLANS.json retains plans: [{root_id, slot_number, scene_id, anchors}].
Each root has slots 1 and 2. Each slot has **six distinct turn_refs in one scene**.
Each anchor contains {turn_ref, speaker, quote}; planning reasons may be retained.
The projected anchor uses only those three fields, and quote must equal the
**entire original turn.text byte-for-character**, including whitespace. No short
excerpt, inferred fact, quote normalization or cross-turn combination is accepted.

Before HTTP the program freezes private labels and Gold, plus the public source-ID
writer plan. The writer returns source-ID predicates only; the program adds source
speaker names and maps to pre-frozen private labels. Independent review still
judges statement, actor, modality, state, unique order and matching semantic anchor.
Its primary anchor must lie in the fixed whole-turn span. A whole turn is an
allowed evidence range, not semantic approval.

Run source_subset.py finalize after these files and CONFIG_TEMPLATE are ready.
It writes SHARD_INPUTS.COMPLETE.json, pinning production, admission, parents,
registry, plans, priors and full sources. Then execute batch.py prepare.
All shard code and inputs are immutable afterward. Request failures are retained;
no automatic retries or fabricated successful outputs are permitted.

## Unchanged lifecycle

prepare → generate → per-round source review → optional bounded supplement
→ finalize-multiselect → final multiselect and ordering source review → freeze
→ prepare-evaluation → trained Solver → base Solver → report.

This engine does not supply a source inventory, author source judgments, migrate
old fixed-root independent auditors, or admit data to training. The orchestrator
supplies independent shard audits and final pilot inventory after actual execution.
