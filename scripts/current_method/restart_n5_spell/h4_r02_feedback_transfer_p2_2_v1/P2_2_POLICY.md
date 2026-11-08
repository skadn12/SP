# Concrete Feedback Transfer policy v1

Execution authority: `h4_r02_feedback_transfer_p2_2_handoff_v1/CODEX_WORK_ORDER.md`, which supersedes conflicting P2-2 design examples and earlier feedback-template limitations for this explicit mode only. This policy contains no manually chosen content target or reference-position example for live use.

The common base is independently versioned as `h4_r02_feedback_common_base_v1`. It fixes the six P1-selected source identities and eligibility, disables active use of ambiguous sources, and replaces the original Miner concrete format example with a field/type JSON Schema. The original default template remains unchanged. The same common base applies to `LEGACY_COMMON_BASE` and `P2_2_CONCRETE_TRANSFER`. The former retains an intents-only response and generic Writer guidance. `ORIGINAL_HOOK_DISABLED` passes no hooks and retains the original generator behavior.

P2-2 adds a strict envelope consisting of `intents` and `feedback_bindings`. It retains the original intent item, canonicalizer, reasoning contract, and candidate schema. The complete raw response is saved before the adapter projects intents for the original canonicalizer. Additional envelope or binding errors cannot remove valid original intents. No Miner retry, seed substitution, candidate gate, fixed content target, or new difficulty/strategy field is introduced.

Only `ACTIVE_GUIDANCE_ELIGIBLE` non-ordering cards from the current root can enter active prompts. Their scores and canonical predictions are deterministic copies of frozen-source observations; full and no context remain separate, with unknowns preserved. Historical options may be distractors and Gold is a historical annotation, not a model-certified fact. Inactive history projections retain only identity, stem, fingerprint, hash, and status. They contain no Gold, options, correct counts, or prediction distributions. No active source means an explicit empty card list and `NO_ACTIVE_FEEDBACK`.

Fixed observation policy:

1. When full is better than no, explore current evidence supporting jointly distinguishable people, events, states, or constraints; never diagnose a mastered capability.
2. When both contexts perform well, explore distinctions depending on the current dialogue; never infer cheating or leakage.
3. A full-context prediction differing from historical Gold can identify a verifiable confusion only for an eligible source. It cannot establish a missing capability or unconditional need to increase difficulty.
4. Ineligible sources cannot supply score guidance. New question facts and answers must derive from the complete current frozen transcript.
5. Add no score threshold, difficulty grade, strategy enum, curriculum, capability field, or required R02 citation. Positions and rollouts concern the same historical question and are not independent question samples.

A proposed binding requires an explicit, unique raw intent ID that survived the original validation and has a valid reasoning contract. A binding ID must be unique. Source IDs must be unique, from this root, eligible, and at most one for the first acceptance. Target, scope, and change must be nonempty. Evidence references must be unique exact current turn references within that intent's normalized evidence-group union. Unknown or extra fields are errors. `NO_SUPPORTED_TRANSFER` requires empty source/ref arrays and empty target/scope/change strings. Missing, malformed, duplicated, filtered, or unknown associations are retained in sidecar audits. No association is inferred from a list index or repaired by guessing an ID.

Mechanical validation is not semantic validation. Offline direct transcript review separately records binding/intent consistency, evidence, realization, novelty, and Gold support; a `BINDING_INTENT_MISMATCH` diagnostic does not modify intent or QA admission.

Guided with a valid proposed binding receives its original contract, full transcript, that binding, its exact historical cards, and verbatim speaker/text extracted by the binding references. Model-proposed target/scope/change are explicitly proposals, not facts. Guided without a usable binding receives exactly the frozen P1 generic Writer suffix; its fallback reason is preserved separately. Categories include `NO_SUPPORTED_TRANSFER`, `BINDING_MISSING`, `BINDING_INVALID`, `ENVELOPE_INVALID`, and `NO_ACTIVE_FEEDBACK`.

Free receives the original two-field intent label, full transcript, eligible root cards, safe historical tracking, and the fixed observation policy. Its compiler has no binding input and cannot read target/scope/change or binding-derived excerpts. Its effective mode is `ROOT_OBSERVATIONS` or `NO_ACTIVE_FEEDBACK`. Free's absence of binding is not a Guided failure.

Briefs are compiled once per `(root_id, intent_id, generation_source)`. Original primary and original whole-batch repair use the same suffix bytes and source. Each actual Writer prompt has a sidecar recording assigned/effective mode, fallback, root/intent/source, prompt SHA, and brief SHA. Tracking never enters the original QA candidate schema. Repair does not rerun Miner or modify the relationship; the original repaired batch replaces the original primary batch.

The original three-intent request, Guided 2/1/1 and Free 4/2/2 primary targets, shared 2/1/1 minimum acceptance, one whole-batch repair, normalization, validator, and ordering generation remain in the same generator. No transcript rewriting, Memory write, Planner modification, Solver evaluation, or automatic expansion is part of this acceptance. The first three roots have a combined logical-call limit of 39 and a planned 21. All observations, failures, and fallbacks remain visible.
