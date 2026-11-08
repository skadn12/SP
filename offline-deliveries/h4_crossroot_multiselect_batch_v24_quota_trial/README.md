# Multi-select quota trial v24

This is an independent generation/preflight path. Prepare/preflight do not modify v21/v22/v23, call a model, run Solver, export training data, or start training.

The trial quota is three candidate questions per root at first, with at most two supplemental candidates after the first batch is reviewed. The total attempted slots are capped at five per root. A supplement requires a content-review record that approves supplementation and names unused R01 evidence references. It is not triggered by a low pass count alone.

## Why this version changes the workflow

- The model drafts complete, topic-coherent questions from six evidence-backed propositions. It does not need a second model call to assemble a pool, so a pool that cannot be grouped into six is not silently counted as a question.
- Each draft must cite at least one appended R01 turn from the frozen parent-plus-R01 promoted dialogue; other evidence may cite parent (P) or R01 turns.
- A contradiction is allowed only when the same speaker explicitly states the opposite in the cited turn. No false option is manufactured from silence or missing context.
- Options are returned as an unlabeled list. After content review, a deterministic local step may bind A-F labels and vary the Gold positions while keeping proposition/evidence pairs intact. Six-of-six Gold necessarily has no positional variation; that constraint is recorded rather than hidden.
- Raw model output, mechanical checks, content review, any Codex revision, and normalized/bound questions are separate artifacts. A Codex repair never replaces or gets attributed to the model original.

## Configuration

Each root entry in `batch_config.json` identifies the exact promoted P+R01 source and prior candidate pool used only as deduplication context:

```json
{
  "root_id": "cc-train-episode-...",
  "source_path": "restart-n5-runs/.../promotable_expanded_dialogue.json",
  "prior_question_paths": ["SP-820-publish/.../candidate_pool.json"],
  "prior_draft_paths": [],
  "covered_fact_notes": ["reviewed semantic fact key or concise description"]
}
```

Paths are relative to `/root/autodl-tmp/gxy-selfplay` unless absolute. The source must contain a matching `root_id`, a `turns` array, and at least one appended `R01:` turn. The full source enters the prompt. Preparation snapshots every dependency and refuses to overwrite an existing batch directory.

## Commands

```bash
python offline-deliveries/h4_crossroot_multiselect_batch_v24_quota_trial/pipeline.py status
python offline-deliveries/h4_crossroot_multiselect_batch_v24_quota_trial/pipeline.py prepare cc-train-episode-... --batch 01
python offline-deliveries/h4_crossroot_multiselect_batch_v24_quota_trial/pipeline.py preflight cc-train-episode-... --batch 01
```

`prepare` writes the frozen source, dedup context, prompt, provider request body, and a hash manifest with status `PREPARED_NO_API_CALL`. `preflight` recomputes those inputs and hashes; it makes no network call. Model execution is a separate step with one HTTP attempt and no retry.

For a supplement, the first batch must contain `CONTENT_REVIEW.json` with this reviewer-authored shape:

```json
{
  "reviewer": "name or review process",
  "supplement_approved": true,
  "unused_evidence_refs": ["R01:S01:T..."],
  "approved_question_count": 2,
  "reason": "Why additional grounded candidates are justified"
}
```

Only those referenced R01 turns are offered as supplemental evidence. The resulting batch can request no more than the two remaining slots under the five-slot cap. The extra slots are not a promise of five approved or Solver-selected questions.

## Review gates

Mechanical checks cover schema, root/turn membership, quote substring, speaker match, quote length, option count, per-question turn uniqueness, Gold cardinality, candidate IDs, and response metadata. These checks cannot establish entailment, contradiction, topic coherence, or semantic novelty. Each complete question therefore needs content review for every option and the shared stem. Any repaired candidate is written as a separately versioned revision with its source candidate ID and a change log. Solver evaluation is downstream and is not part of this trial skeleton.
