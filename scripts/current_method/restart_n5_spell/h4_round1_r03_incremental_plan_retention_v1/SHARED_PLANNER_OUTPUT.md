# Shared Planner Output v2

This is an explicitly selected historical contract. The current default initial
prompt and append-only formatting path are described in
[SUPPLEMENT_FORMATTING.md](SUPPLEMENT_FORMATTING.md).

Initial and supplemental planning use `shared_planner_output_v2`, defined once in
`shared_planner_contract.py`. Both prompts render that exact schema, with field
descriptions and no filled output examples. Rendering version
`shared_planner_schema_only_v3` removes concrete story examples and all nested
`examples` annotations from actual requests. Historical example values are used
only as offline fixtures. Both compiler entries validate against the schema before
materializing candidate fields. Validation uses `jsonschema` (declared in the
repository requirements). The initial compiler accepts the raw JSON string and
the shared supplement request parser rejects duplicate keys and non-JSON numeric
constants, so parsing cannot silently overwrite a prior field. The HTTP provider's decoding configuration is
unchanged: this is a shared prompt and local validation contract, not a claim that
the provider guarantees schema-constrained generation.

The top-level object always contains exactly `round_goal`,
`selected_continuations`, and `new_development_candidates`. Candidate fields stay
flat. Both candidate types include `source_candidate_ref` and `dependencies`.
There is no model-visible patch wrapper, status, hash, or phase ID.

| Rule | Initial | Append-only |
| --- | --- | --- |
| Array contents | Initial proposed candidates | Only new additions |
| Round goal | Proposed once | Exact frozen original value |
| Continuation source | Null; valid verified track required | Exact original registry ref; track copied or null for an original new development |
| Continuation identity | Existing speaker contract | Exact source goal, target, participants, bindings and declarations |
| Independent development source | Null | Null |
| Dependencies | Earlier rows in this response | Original sources or earlier rows in this response |
| Dependency requirement | Optional | Nonempty for source continuations; optional for independent developments |
| Limits | At most 3 continuations and 8 new developments | At most 8 independent new developments; continuation execution remains budget-limited |

Each dependency has exactly `source_candidate_ref`, `output_ref`, and `condition`.
Exactly one reference is non-null. `output_ref` is a zero-based JSON pointer,
such as `/selected_continuations/0`. Continuations precede new developments in
processing order. Unknown, forward and self references are rejected. Conditions
still require actual retained dialogue evidence at execution time; a reference
does not assert that the source goal was achieved.

## Runtime Integration

For an explicitly shared initial request, use
`prompts.render_initial_planner(planner_contract_version="shared_planner_output_v2", ...)`, then pass
the response to `shared_planning.compile_initial_output(...)`. Persist its
successful selection result as `planner/variant_selection.json`, retaining
`planner_contract_version`, `shared_output`, and the canonical `value`. The raw
response and the sent prompt remain separate request evidence.

`bundle.build_failed_bundle(failed_run=...)` recognizes that explicit selection
version and includes its shared output under the frozen plan-file identity. It
selects `incremental_retention_v1.supplement.shared_planner_v2`. Preparation checks
the original frozen sources and replays the shared compiler, including dependency
mapping. It rejects a shared initial selection paired with a historical supplement
version. It never infers a contract version from an invalid response's shape.

`root_scope.render_for_bundle` renders the shared supplement prompt and immutable
source templates. `root_scope.merge_for_bundle` dispatches to the shared append
compiler. The compiler assigns phase IDs, resolves dependencies, and invokes the
existing scoped candidate materialization and semantic checks. It produces a
separate additions record; the frozen original plan and its hash are preserved.
Both normal execution and saved-output preparation use the same dispatcher.

Source continuations copy the immutable source template and change only the new
phase transition, action-summary beats, and dependencies. Independent developments
receive locally assigned candidate IDs. A changed source goal, target, actor
binding, or declaration is rejected, never silently normalized into acceptance.
Initial dependencies survive execution queue construction and task selection.

The historical root supplement version remains available only when explicitly
selected by historical bundles. Existing completed runs, grants and attempt
ledgers are not rewritten or retried. Historical initial rendering is available
with `planner_contract_version="legacy_initial_v1"` for compatibility checks.

Gate, Realizer, adherence, actual-progress retention, budgets, and request limits
remain enforced downstream. Updating the contract does not approve every planned
scene and does not make unobserved goals into facts.

## Verification

The focused tests are `test_shared_planner_contract.py`,
`test_shared_planning.py`, `test_shared_planner_prompts.py`, and
`test_shared_planner_entry.py`, alongside
the existing root planning, prompt, queue, and Realizer regression tests. Transport
integration uses blocked sockets and temporary mock grants and ledgers. No live
model call is part of this change's verification.

The completed regression run passed 79 tests across these shared-contract tests,
historical prompt restoration, bundle validation, fresh task selection, and policy
execution. The earlier root planning, root CLI, normalization, and Realizer run
passed another 50 tests. Live model adherence to this new output contract has not
been tested.
