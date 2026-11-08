# v52 bounded QA regression

This adapter reads existing v51 CANDIDATE_HISTORY.json and COLLECTION_BINDINGS.json without rewriting the expansion. Full original+R01 and retained A/B source context reaches every generator and reviewer; frozen prior-round feedback remains present. There are no new question semantic admission gates, no manual Gold edits, no changed full-only Solver score thresholds, and no duplicate-triggered retry.

The writer now sees used_six_statement_sets, a public-only projection of hard-dedup prior inventories with identical whole sets grouped by type. It asks the model to choose evidence first and compare all six statement texts before emission, ignoring display labels/order/case/whitespace. Prior facts and partial overlap remain allowed. The existing dedup code and thresholds still decide actual exact-set duplicates; improved yield is not assumed from offline tests.

Content guidance explicitly checks referent identity and tense. Independent ordering review uses a scope derived only from an explicit recognized public stem scene clause. It retains the complete transcript for context and reference resolution, keeps valid in-scope alternate anchors, and ignores valid quoted out-of-scope alternatives for an explicitly scoped ordering question. Full-dialogue questions retain the full matching scope. The raw review and all ignored alternatives are preserved for replay.

The parent regression driver uses generate.prepare slots=2: each chunk has2MC+2ordering, maximum6Qwen HTTP calls and60full Solver calls. It runs three chunks on each of two v51 root histories (135466 and210046), completing all six generation chunks before Solver. Parent controls all live actions and public prior/focus construction.

Optional batch.py API supports8or20slots (two or five chunks), with --candidate-slots, repeatable --prior-pool and --focus-plan. It freezes an explicit soft focus plan or builds the deterministic cycle over actual available scenes. This API does not authorize additional live work.

Tests use synthetic transport/Solver boundaries and never make live calls. BASELINE_FILES.json binds v51 before changes.
