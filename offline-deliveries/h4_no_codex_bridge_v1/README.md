# No-Codex content-review bridge v1

This offline bridge accepts original model drafts and fail-closed model-review artifacts. It does not read or create `CONTENT_REVIEW.json`, call a generator or reviewer, run Solver, export training data, or alter historical results. A missing, changed, incomplete, or inconsistent review fails closed. The review models are recorded as model reviewers, not as human certification.

## Multi-select

`multiselect.py` supports the current `MECHANICAL_CHECKS_ONLY_CONTENT_REVIEW_REQUIRED` draft schema and the `REVIEW_RESULT_POLICY_V3.json` policy from the DeepSeek pilot. It recomputes the policy from the saved reviewer response, checks source/request/response hashes and returned model, then deterministically rebinds the original six propositions and Gold. The caller must explicitly select `multiselect_quote_cues_v3`; a different review policy needs a new bridge version. Quotes with the v3 referent and contradiction cues are treated exactly as in the original pilot, including its known false-rejection risk.

```bash
python multiselect.py \
  --batch-dir /path/to/batch-01 \
  --review-dir /path/to/cc-train-episode-ID \
  --review-policy multiselect_quote_cues_v3 \
  --output /new/output/READY_FOR_EVALUATION.json
```

When at least one question passes, the output uses the existing multi-select Solver v2 `CONTENT_AUDITED_READY_FOR_SOLVER` pool contract, with absolute source paths and source hashes. The output status is `NO_APPROVED_CANDIDATES` when none pass; do not send that empty pool to Solver. The v2 `validate_pool` contract is covered by a regression test.

## Ordering

`ordering.py` accepts the ordering v5 mechanical replay and three saved model reviews: DeepSeek first pass, DeepSeek challenge, and GLM independent pass. It checks candidate/source/old-pool hashes, the exact reviewer requests and responses, returned model IDs, and recomputes all three verdicts. Only their intersection enters its independent output pool. The original paraphrased item text is unchanged; the permitted deterministic operation is label-only rebinding with Gold recomputed from source turn order.

```bash
python ordering.py \
  --batch /path/to/h4_crossroot_ordering_batch_v5_fresh_roots \
  --pilot /path/to/h4_crossroot_ordering_no_codex_pilot_v1 \
  --first-run /path/to/semantic_review_deepseek_v2 \
  --challenge-run /path/to/semantic_challenge_deepseek_v3 \
  --independent-run /path/to/semantic_review_glm_v4 \
  --output /new/output/directory
```

The older ordering Solver runner is bound to a fixed two-question Codex-reviewed pool. The existing no-Codex Solver trial runner is also bound to its frozen one-question build. Thus `ordering.py` produces a verified candidate pool and audit, but **a versioned, parameterized ordering Solver adapter is still required for future batches**. Do not claim Solver admission from the bridge output alone.

## Verification

```bash
python -m unittest -v test_multiselect.py test_ordering.py
```

This bridge is a replayable input gate, not a validation of reviewer accuracy. The frozen four-root trial still yields zero reviewed multi-select questions and one reviewed ordering question; the ordering question was separately Solver-selected and provisionally model-audited for dedup in the earlier no-Codex trial. Per-root target of 4-6 confirmed unique questions is not yet met.
