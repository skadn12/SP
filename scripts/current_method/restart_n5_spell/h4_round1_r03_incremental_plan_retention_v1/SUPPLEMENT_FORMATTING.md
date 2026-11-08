# Append-only Supplement Formatting

The default initial Planner uses `legacy_initial_v1`, including the restored
development guidance and current target/progress policy. Existing explicitly
selected shared contracts remain available for their frozen inputs.

The root supplement keeps `ROOT_PROMPT_VERSION` and its original five fields:
`status`, `original_plan_sha256`, `phases`, `new_developments`, `reason`.
Its prompt and model output requirements do not change for formatting.

Select the adapter on a derived bundle with
`root_scope.with_supplement_formatting(bundle)`. This returns a copy with
`supplement_format_version = "root_supplement_formatting_v1"`; it does not
change the frozen bundle, model messages, budget, or call ledger.

The execution sequence is:

1. Preserve the raw supplement response.
2. Move recognized candidate fields into `candidate`; reject conflicting values.
3. Preserve valid phase IDs, assign valid IDs to malformed ones, and update
   explicit dependency references using the same ID map.
4. Run the original append-only merge validation and the selected queue checks.
5. Continue the existing Gate, Realizer, and adherence workflow when executing.

Formatting never creates missing narrative fields, rewrites goals, changes actor
bindings, or infers dependencies. Duplicate JSON keys, unknown fields, ambiguous
IDs, and unknown/self/forward references are rejected. A format pass is not a
semantic approval; the original merger and downstream checks still apply.

`workflow.execute_policy` saves the original response, `formatted_patch.json`,
`supplement_formatting.json` (changes and ID map), and the merge/diff independently.
The adapter is also used by `prepare_saved_resume` for offline validation.
It makes no model calls and grants no additional supplement attempt.

The real failed response from `h4_round1_r03_aligned_prompts_single_root_v1_live_1`
is replayed in `h4_round1_r03_supplement_format_replay_v1_offline_1/replay.py`.
This replay only formats, validates, and selects the saved plan. It does not run
Gate/Realizer, claim a live resume, or modify any historical run or ledger.
