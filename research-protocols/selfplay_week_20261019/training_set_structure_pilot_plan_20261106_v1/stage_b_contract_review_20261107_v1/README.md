# Stage B contract review bundle

Status: `DRAFT_FOR_REVIEW_NOT_FROZEN`. This bundle is outside `execution_20261107_v6`; it does not amend that execution, grant authorization, or start an author or selector. The v6 directory and its hashes remain unchanged.

The contract text turns `PLAN_ZH.md` §§2, 4, 5, 8, and 9 into auditable output and state rules while preserving the 8-root, 32-opportunity, OLD/NEW × F/G design. It does not add another generation route, retry, fallback, quota, or training step.

The bundle is not launch-ready. The v6 selector projection has 966 rows but only `question_type`, `answer_operation`, and `operation_status`; 512 are unknown/pending and 454 are provisional. This 966-row projection must not be mistaken for the plan's exact annotation scope. `PLAN_ZH.md` §3 requires independently audited answer-operation coding for the sealed 135 items and tuple annotation (event, natural information need, query scope, answer operation, evidence family) for same-root historical references; other roots/non-training history can remain overlap-only if the complete inventory and hashes are bound. The existing review sidecars report the 135-item operation audit and 106 same-root rows, including 12 explicit unknowns. The sidecar does not yet provide a fully reviewed tuple for every known same-root row: semantic need/evidence certificates are reported for 53 of 94 known rows, and explicit query scope for 12. The remaining known rows need row-level completion/review from their available question, displayed items, anchors, and operation/relation evidence before the selector comparison can be called compliant. Unknowns stay unknown; provisional rows cannot count as verified coverage. No missing labels may be inferred from opaque IDs or silently treated as “uncovered.”

Before Stage B is frozen, the controller must:

1. Complete the remaining row-level tuple audit for eligible same-root references, bind the sealed-135 operation audit and same-root audit, and preserve `UNKNOWN` wherever evidence does not establish a field. Keep the exact plan-defined scope; do not expand it to every row in the 966-row projection.
2. Produce a new anonymous projection and append-only input manifest with a documented eligibility/status policy, complete source inventory/hashes, and raw-byte SHA-256 hashes. Other-root/non-training references may remain overlap-only. Do not edit the v6 projection in place.
3. Rebuild/hash-bind role packets and selector packets to the new contracts and history projection; rerun strict preflight and anonymity/access checks.
4. Freeze the exact author, selector, schema, state-machine, metric, schedule, input, and packet artifacts together.
5. Record the human Stage A signoff and a separate `AUTHOR_ONLY` authorization bound to that freeze hash. The current v6 authorization and freeze both remain `NONE` for generation.

After that freeze, execute at most one attempt for each of the 32 opportunities. OLD authors create one target and at most one item. NEW authors submit up to three semantic slots, freeze them, receive one independent selector result, and may implement only the selected slot. Then stop at frozen first drafts for blind review, structural review, and historical deduplication. Solver, training, export, retry, and fallback remain out of scope under `PLAN_ZH.md`.

Files in this directory are review materials, not packet inputs. `STAGE_B_CONTRACT_FREEZE.json` records the parent hashes and the blockers; its status must not be changed to frozen until the checklist above passes.
