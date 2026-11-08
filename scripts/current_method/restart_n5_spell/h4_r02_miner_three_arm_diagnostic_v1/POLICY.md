# Three-arm Miner diagnostic v1

Inherit `h4_r02_feedback_common_base_v1` and its schema-only Miner example
revision. This version adds `h4_r02_miner_explanation_scope_fix_v1` only in the
new adapter: replace the one default sentence forbidding all explanations with
the contracted sentence forbidding QA answer explanations while permitting
fields required by the active schema. Original/v1/v2 source and requests remain
unchanged. No question examples or new evidence selection instructions are added.

`A_OFF` and `B_OBSERVATION_ONLY` have the same intents-only response schema.
A clears active cards and marks `NO_ACTIVE_FEEDBACK`, while retaining exactly
the same safe historical tracking and fixed policy as B and C. B/C expose the
same original eligible card projection. `C_P2_2_TRANSFER` adds exactly the v2
envelope and `MINER_BINDING_RULES`, with unchanged v2 binding postprocessing.
Arm ID, order and diagnostic hypotheses remain outside the model prompt.

`build_prompt` is the single pure arm-aware constructor. Exact-byte pairing
diagnostics check A/B payload-only differences, B/C schema-and-rules-only
differences, and C versus the historical v2 prompt's one declared replacement.
The original intent item schema, numbered transcript and historical projection
are identical across arms. Whole response schemas are intentionally different.

All arms call the existing `generate_root_intents`, return normally, and reuse
v2 `derive_all_intent_evidence`. A/B raw responses reach the original strict
canonicalizer unchanged; extra fields are errors. A/B never invoke v2 binding
validation or create binding previews. C delegates projection, validation,
derived evidence and offline previews directly to `TransferHooksV2`.

No semantic evidence search, source-ref supplementation, clipping, count cap,
scene gate, intent ranking or extra sampling is performed. All original valid,
invalid and extra items are retained; reporting keeps the 18 planned slots.
Derived evidence is `NOT_SENT_TO_WRITER`. No Writer/repair/Solver/Memory/training
or other model stage is provided by this adapter. Runtime and review policies
remain independently frozen by the runner.
