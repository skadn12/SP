# Adherence evidence contract

The default renderer is `adherence.scoped_evidence_v4`. It sends all generated
turns once, with the assigned `turn_ref` directly beside each `generated_turn`.
Raw turn objects are retained verbatim; normalized scene identity and full
authoritative history remain available to the auditor.

The prompt explicitly requires:

- `goal_evidence_refs` to contain unique strings, never evidence objects.
- New `actual_states` references to come only from the current scene.
- `history_evidence_refs` to cite only exact IDs in authoritative history for
  history consistency and novelty comparisons. The field is required in v4;
  an empty list is allowed when no specific historic turn is decisive.
- Goal evidence to use the current scene and relevant authoritative history.
  A partial or realized goal still requires at least one current-scene reference.
- Requested conditions to use current or authoritative historical evidence,
  preserving the specified time, actor, and stage. Historical prerequisites
  cannot be counted as new progress. Plans are requirements, never fact evidence.
- `condition_results` to cover exactly the supplied downstream conditions;
  an empty input list requires an empty output list.

The retention assessor validates these separate reference sets. Sibling scenes
in a pending batch are not authoritative history. Previously retained scenes
become history only when explicitly supplied in the next audit context. Future
audit capacity includes inline reference wrappers and raw segment boundaries;
the final request is counted again before transmission. Explicit
`adherence.raw_response_v2` and `adherence.inline_evidence_v3` rendering remain
available for historical replay. New workflow calls require the v4 audit fields;
saved v3 audit reuse is explicit in the recovery runner.

`adherence_format.format_adherence` creates separate views of saved responses.
It flattens evidence objects, deduplicates goal references, and excludes
unrequested conditions with full transformation records. An invalid reference
can be mapped only through a complete quote matching exactly one turn in the
same scene. Ambiguous, conflicting, or cross-scene references fail conversion.
Judgments, observation stages, statements, and reasons remain unchanged.
Any excluded invented condition sets `semantic_reaudit_required` and persists
its scene reference and full transformation record in `_formatting_provenance`.
`valid` means only that formatting succeeded. The assessor labels affected rows
`NEEDS_SEMANTIC_REAUDIT` and prevents promotion, including after JSON reload or
repeated formatting. Raw extra conditions and malformed provenance also block
promotion. Old unmarked formatting artifacts must be regenerated from raw audits.

This formatter is offline-only and is not an automatic live response repair.
Its output must still pass the existing retention assessor. A recovered audit
does not append dialogue to history. A fresh complete audit must reassess all
judgments from the unchanged dialogue, original goals, and legitimate conditions;
the old audit and its invented criteria are not sent. Later batches use the
history resulting from retention of earlier batches.

The separate run `h4_round1_r03_adherence_contract_v3_probe_live_1` records an
offline review of four saved batches and one new audit of the saved first batch
(two scenes, 28 turns). It does not rerun Planner, supplement, Gate, or Realizer,
and its additional one-request ledger leaves prior execution ledgers intact.
The run's `final_report.json` records the actual outcome after execution.

`h4_round1_r03_scoped_adherence_reaudit_v1_live_1` applies the new boundaries,
reuses that first-batch audit, and reviews S03-S07 in three sequential calls.
It saves a new cumulative snapshot using the existing append/closure checks,
without changing prior run files or regenerating dialogue. Its final report
distinguishes semantic rejection, required re-audit, and malformed responses.
