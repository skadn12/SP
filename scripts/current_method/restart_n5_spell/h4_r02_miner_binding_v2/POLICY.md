# Miner Binding v2

Version: `h4_r02_miner_binding_v2`. Binding schema:
`h4_r02_feedback_binding_schema_v2`; envelope:
`h4_r02_miner_binding_envelope_v2`.

The common base remains `h4_r02_feedback_common_base_v1`, including its unchanged
source eligibility, model observation presentation, fixed policy, and schema-only
Miner example revision. The original intent item fields and position validator
are unchanged. No root-specific question targets or demonstrations are added.

Bindings have exactly `intent_id`, `status`, `source_feedback_ids`, `scope`, and
`change_from_source`. The last field is an exact three-string object:
`old_question_focus`, `feedback_basis`, `added_distinction`. The old independent
`evidence_refs` and `target_distinction` fields are forbidden. Invalid fields and
the v1 string explanation produce `BINDING_SCHEMA_INVALID`. Explicit model
`NO_SUPPORTED_TRANSFER`, missing bindings, and invalid bindings remain separate.

Current evidence comes only from each originally validated intent's original
three nonempty disjoint reasoning groups. `derive_intent_evidence` copies their
union once per ref in frozen transcript order, preserving exact speaker/text and
group membership. It never consults the target, explanation or historical source
refs to select evidence, and does not cap or trim wide chains. Missing or
ambiguous frozen refs stop derivation for that item as an integrity error. Other
valid items survive an invalid item or invalid binding. Intent and transcript
hashes identify the source values; when supplied by the runner, the transcript
hash identifies the exact frozen file bytes as well as its canonical content.

All mechanical checks are structural or exact association checks. Explanation
truth, actual support, causal claims, scope, overly broad evidence and substantive
transfer require separately authored offline review. Counts of refs and scenes
are diagnostic only. The program does not certify explanations by keywords.

The adapter is used with the formally shared Miner stage. It has no Writer
hook. Derived sidecars and Guided previews are `NOT_SENT_TO_WRITER`, and no QA,
Gold, candidate pool, or fake Writer response is created. Future Free projection
accepts only an intent label, root observations/history tracking and the complete
transcript; Guided evidence, scope, and transfer explanations cannot enter it.

Runtime is limited independently to one scheduled Miner call per fixed root,
three calls total including failures. Writer, repair, Solver, Memory, training,
Planner, Realizer and extra review LLM calls are zero. A readiness recommendation
does not authorize Writer execution. Preserve all P1/v1 artifacts and stop after
the Miner-only acceptance report.
