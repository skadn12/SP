# QA Composition Probe v1

This is a completely isolated, offline QA-level probe. It consumes frozen,
materialized atomic MCQs and frozen evaluation reports, normalizes eligible
questions into an internal IR, compiles exactly-two-QA relation candidates,
and writes future Composer requests.

It does not call the existing QA Writer, a Composer, a distractor service, a
Solver, or Memorax. It does not generate or modify Parent/Child dialogue and
does not change current QA admission logic.

## Accepted source schema

The current adapter reads `simple_append` materialized MCQ rows with:

```text
qa_id
question
question_type
options
reference_answer_letter
reference_answer_text / canonical_answer_text
evidence_turn_ids
source_scope
```

Evidence text and 1-based turn indices are resolved exactly from the supplied
frozen expanded dialogue. Missing gold or evidence makes a row ineligible;
the probe never repairs either field. By default, the corresponding frozen
evaluation must contain `grounding.decision == "pass_grounding"`.
Because this probe targets the existing easy atomic questions, the frozen
full-context result must also be 5/5 correct. Diagnostic flags can relax these
filters explicitly, but relaxed rows must not be reported as the main result.

## Offline run

Use a new, empty run root:

```bash
python -m scripts.current_method.restart_n5_spell.qa_composition_probe_v1.probe \
  --atomic-batch /path/to/frozen/qa/final_mcq_batch.json \
  --expanded-dialogue /path/to/frozen/expanded_dialogue.json \
  --evaluations /path/to/frozen/reports \
  --run-root /path/to/new-offline-run
```

The runner refuses to overwrite a non-empty run root. It writes:

```text
atomic_qa_normalized.jsonl
pair_candidates.jsonl
pair_rejections.jsonl
composer_requests.jsonl
static_audit.jsonl
report.md
manifest.json
```

Only `state_change`, `proposal_vs_outcome`, and `actor_outcome_binding` pairs
enter `composer_requests.jsonl`. `uncertain` pairs remain review-only. The
static audit file contains request preflight records and explicit `not_run`
placeholders for future `no_context`, `remove_E1`, `remove_E2`, and
`full_context` gates. No Composer response is synthesized.
