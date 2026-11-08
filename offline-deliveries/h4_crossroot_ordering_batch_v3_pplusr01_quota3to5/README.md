# Cross-root P+R01 ordering generation v3 (3 to 5)

This is an isolated generation batch. Its six roots share the P+R01 promoted-dialogue cohort. The initial request asks for up to three candidates per root. A separately prepared top-up may request up to two more, for a hard maximum of five requested candidates per root. Top-up is an explicit operator action after reviewing the first response and its evidence; the script does not infer that more candidates are warranted. Every candidate must cite at least one appended R01 turn among its six evidence anchors; parent P turns may support its other items. This requirement is checked both in the prompt and by mechanical validation.

## Source manifest

Copy `manifest.template.json` to a new manifest file and populate `sources` with rows of this form:

```json
{
  "root_id": "cc-train-episode-123456",
  "source_path": "/absolute/path/to/promotable_expanded_dialogue.json",
  "source_sha256": "64 lowercase hex characters",
  "prior_candidates_path": "/absolute/path/to/prior_validated_candidates.json",
  "prior_candidates_sha256": "64 lowercase hex characters"
}
```

`prior_candidates_path` and `prior_candidates_sha256` are an optional pair; omit both or set both to `null` when there is no prior pool. Supported candidate containers include a top-level array or an object with `questions`, `candidates`, `valid_candidates`, or `candidates_valid`. Candidate rows can be ordering or multi-select questions.

The source must be a complete promoted dialogue with `root_id`, `turn_count`, and `turns`; each turn must have a unique non-empty `turn_ref` and string `text`. The manifest hash, source bytes, and original prior-pool bytes are frozen during preparation. The prior pool is copied byte-for-byte and never modified. A normalized `PRIOR_CANDIDATE_SUMMARY.json` records candidate IDs, prompts, topics, evidence refs, and ordering-item or multi-select-option text; it is bound into the generation prompt. Do not reuse roots whose evidence has already been exhausted or whose prior candidate coverage has not been checked.

## Prepare initial requests

Preparation makes no network call and does not read credentials. The checked-in manifest pins source and prior-pool bytes for roots 109164, 130221, 160549, 191749, 222937, and 249232:

```bash
python generate.py --prepare --manifest /absolute/path/to/source_manifest.json
```

This creates `BATCH_PREPARED.json`, a frozen `source.json` per root, and an `initial/` request bundle containing prompt and payload hashes. Preparation refuses existing output files.

It also writes `prior_candidates.json` as an untouched copy when a prior pool is supplied, plus the normalized dedup summary. The provider prompt explicitly avoids matching facts/topics regardless of phrasing while allowing uncovered source evidence.

Run the local normalization/prompt checks without network access with:

```bash
python -B -m unittest test_dedup_summary.py
```

## Execute and validate separately

Each explicit execution makes at most one HTTP attempt, with zero HTTP retries, and saves the provider response without overwriting existing files. The imported `ProfileClient` transport has no retry loop and records one transport attempt:

```bash
python generate.py --execute ROOT_ID
python generate.py --validate ROOT_ID
python generate.py --summary
```

Mechanical validation is a separate command. It checks response binding, schema, candidate IDs, six-item structure, complete Gold permutations, distinct evidence turns, verbatim quotes, and—only for `conversation_order`—that Gold agrees with cited transcript order. It cannot verify semantic truth or within-scene action chronology. Valid candidates are labeled `MECHANICAL_CHECKS_ONLY_CONTENT_AUDIT_PENDING`; none is approved, Solver-evaluated, or selected for training.

## Optional top-up

After reviewing the initial output, prepare no more than two extra candidate slots per root:

```bash
python generate.py --prepare-top-up ROOT_ID --reason "specific uncovered evidence remains"
python generate.py --execute-top-up ROOT_ID
python generate.py --validate-top-up ROOT_ID
```

The reason is recorded in a separate immutable top-up decision file. Top-up uses candidate IDs 004 and 005 and a separately frozen prompt/payload. Its dedup summary includes both the prior pool and candidates returned in the initial response. It is never run automatically. If the source has no unused, supportable evidence, do not top up.

Provider, requested and returned model, response ID, usage, raw response text, source/prompt/payload hashes, and transport attempt metadata are retained. The script contains no Solver or training invocation. This corrected P+R01 batch leaves the earlier mixed-cohort prepared directory untouched; that earlier directory must not be used for this request.
