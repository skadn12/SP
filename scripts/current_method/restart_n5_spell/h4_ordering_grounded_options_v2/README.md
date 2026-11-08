# Grounded ordering options v2

Explicit opt-in policy: `{"version":"ordering_grounded_options_v2","node_count":4}`.
Use it with the shared contract-guided Writer and `partial_grounded_v2` admission.
There is no additional extraction, sampler, review, or temporal-reasoning API call.
Free and Memory-only are not enabled for this version.

Writer event nodes retain `item_id`, `item_text`, `actor`, `event_state`, and
`evidence_refs`, and add nonempty `evidence_spans=[{"turn_ref":...,"quote":...}]`.
Several spans and refs may be supplied to preserve pronouns, reported speech,
state qualifiers, and time context. Every cited ref needs at least one exact
substring. A matching quote does **not** prove that it supports the option.

Writer relations contain `ordered_pair=[earlier_id,later_id]`, `evidence_refs`,
and `evidence_spans`. They cannot contain `before`, `after`, `temporal_basis`, or
another independent direction. The program projects the pair to the existing
DAG implementation and derives display relations and Gold. Unknown times,
overlap, and disconnected constraints never receive a turn-order tie break.

Optional `date_anchors=[{"segment_id":...,"conversation_date":...}]` on a node
or relation must match frozen `conversation_date_segments` and the segment of
at least one cited ref. This proves anchor identity only. A conversation date
does not automatically date a remembered action, intended action, or completion.
There is no new required time-expression field.

A transcript batch may explicitly abstain from ordering with
`abstentions=[{"question_type":"ordering","status":"INSUFFICIENT_EVIDENCE","reason":...}]`.
Legal abstention suppresses ordering quota repair; it does not excuse malformed
emitted candidates. If SC/MS triggers the existing single repair, the validated
primary abstention is frozen and an ordering candidate cannot refill it.

Semantic entailment, actor/state attribution, coreference, and true temporal
support remain `NOT_ASSESSED` by the program. Offline content diagnosis must
report them separately. Rejecting an old v1 output for missing new fields is
not evidence of improved generation quality. No model calls accompany these
code and mock tests.

The v1 policy, prompts, normalization, provenance checks, and admission remain
available. Newly created runs must freeze their selected policy and code; do
not modify or resume already-started historical runs under this policy.
