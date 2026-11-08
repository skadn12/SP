# v48 evidence-first conditional and contradiction coverage probe

One authorized bounded follow-up to v47: generate 2 multi-select + 2 ordering
questions on the same complete 156-turn / 4566-body-token candidate history.
Keep v47 evidence-first output, and add generation-only generic guidance to
preserve conditional premises and intention/commitment levels, prefer supported
and explicitly contradicted propositions when the source permits, and select
different coherent fact combinations. No quotas for relation types; all six
supported options and abstention remain accepted by the original schema.

Reviewer instructions, schema, original Gold derivation, exact full-set dedup,
and Solver selection stay unchanged. Output order and content analysis remain
diagnostic only. No hand edits of model questions, labels, Gold or decisions.
All writer/reviewer calls retain the full cumulative source and frozen previous
training-round feedback. The prior inventory adds v47's three selected question
texts to the v47 baseline inventory solely for exact six-statement-set dedup;
no current scores, answers, Gold or post-run diagnostic feedback enters generation.

Actual run uses current only; old adapter API remains offline compatibility.
Only Qwen/Qwen3.6-35B-A3B for generation/review. Maximum 6 HTTP calls, one
mechanical repair per type, no automatic retries; shared completion gap >=35s.
Fixed audited M_clean_policy checkpoint: at most 60 full-context answers,
MC 6 permutations x3 and ordering 12 permutations x1, no ground. Selection is
complete_valid_full_set and 0 < full_partial_mean < 1.

No new dialogue, source promotion or training in this probe. Formal 30-question
pool and earlier provisional pools remain immutable. Report machine selection
separately from observed semantic limitations. Freeze implementation and tests
before prepare. BASELINE_FILES.json binds 2253 historical files including v47.
