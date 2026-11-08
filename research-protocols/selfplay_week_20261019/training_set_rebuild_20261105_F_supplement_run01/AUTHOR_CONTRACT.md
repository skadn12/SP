# Isolated F supplement author contract

Read only this contract, your assigned root source snapshots and neutral source information, and `ABSTRACT_FEEDBACK.json`. Do not open other runs, historical questions/Gold answers, reviews, solver artifacts, or other authors' outputs. Do not list agents. You are the isolated Codex author. Existing drafts are not available to you; subsequent semantic review will check novelty against all earlier tasks.

For every assigned root, read the entire `SOURCE.json` dialogue, not just matching excerpts. Explore at least three materially different natural information needs where possible. Avoid artificial bundles across unrelated scenes, overly bureaucratic question phrasing, and task instructions that disclose a pattern. Do not create dialogue or new facts. Multi-select (2–6 true choices and at least one false choice), full ordering, and partial ordering are allowed. No single-choice question, type ratio, Gold quota, or minimum difficulty. At most one new first draft per root; zero is permitted with a specific reason. A low-yield root or batch does not stop later coverage.

Preserve speaker/entity binding, attributed reports, uncertainty, request/offer/acceptance/completion, and explicit source restrictions. A broad source warning is not a whole-root rejection. Do not use a restricted turn simply because it exists in the full source. Do not require false unstated global chronology or infer prerequisites from ordinary temporal order. The question must have one decidable Gold set/order/relation set from actual evidence. Use English, matching source language, with a concise natural question and options.

Write each root's outputs under `authoring/<root_id>/`, with exclusive creation and no later content edits:

1. `EXPLORATION.json`: root_id; source_sha256; full_source_read=true; turn_count; source_risks_considered; at least three explored targets where possible (need, relation, evidence_turn_refs, chosen/discarded reason); chosen_target or NO_SAFE_QA. Do not claim exhaustive search.
2. Before drafting wording, write `TARGETS_FROZEN.json`: root_id; targets=[{semantic_target, relation, option_facts, gold_option_ids (or gold_pairs), evidence_turn_refs}]. For zero candidates targets=[]. This file must exist before the candidate draft is written. Do not rewrite it.
3. `CANDIDATES_FIRST_DRAFT.json`: root_id; arm=F; candidates=[...] (0–1). Each candidate uses the schema below.

Candidate schema:

```
{
  "candidate_id": "FS1_<numeric_root>_01",
  "root_id": "cc-train-episode-...",
  "arm": "F",
  "question_type": "multi_select|ordering|partial_order",
  "question": "...",
  "options": [{"option_id": "stable_meaningful_id", "text": "..."}],
  "gold_option_ids": ["..."],
  "source_path": "absolute path to your assigned SOURCE.json snapshot",
  "source_sha256": "actual source snapshot hash",
  "source_turn_count": 123,
  "natural_information_need": "...",
  "cognitive_task": "...",
  "event_identity": "...",
  "relation_plan": {"relation": "...", "scope": "...", "not_inferred": "..."},
  "evidence": [{"turn_ref": "...", "quote": "exact substring of source text", "origin": "exact source origin"}],
  "option_audit": [{"option_id": "...", "correct": true, "gold_position": null, "reason": "...", "evidence": [{"turn_ref": "...", "quote": "exact source substring"}], "distractor_error_path": null}],
  "evidence_audit": {"sufficiency": "...", "alternative_paths": "...", "scope_limits": "..."},
  "quality_concerns": [],
  "source_scope_note": "..."
}
```

For ordering, `gold_option_ids` contains every option exactly once; `correct` may be null; `gold_position` is 1-based. For multi-select every option has a boolean `correct` and false options have a specific distractor_error_path. Do not fabricate a positive citation for mere absence; explain search scope.

For partial ordering, use `gold_pairs` and `gold_option_ids` as the same full transitive closure of stable-ID directed pairs. Also specify `query_scope`, `query_pairs` (all pairs in the query domain), and `answer_semantics="all_and_only_entailed_pairs_in_query_scope"`; add a pair_audit. The question must explain whether the relation is chronological precedence or required prerequisite. Do not confuse them.

Before returning, verify IDs, quotes, origins and source hash. Record all exploration results; do not revise a frozen question based on later review or solver feedback.
