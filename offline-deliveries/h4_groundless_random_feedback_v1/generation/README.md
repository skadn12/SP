# Groundless generation review interfaces

`ordering_review.py` owns the independent Qwen review route for the unchanged
10 label-rebound ordering candidates. It does not read historical DeepSeek,
GLM, Codex, or Solver verdicts.

```bash
python ordering_review.py prepare --pass-id pass-1
python ordering_review.py live --pass-id pass-1 --root-id cc-train-episode-210046
python ordering_review.py review --pass-id pass-1
python ordering_review.py bridge --pass-id pass-1
```

`prepare` performs four offline request freezes, one per root. `live` reserves
one HTTP attempt for one root and uses the existing SiliconFlow Qwen helper;
it is intentionally not run by this module's tests. A second pass is made with
another `--pass-id` and optionally repeated `--candidate-id` values. The new
request contains only the original candidate/source data and its optional
feedback suffix, never pass-1 verdicts.

The reviewer reports one primary `order_anchor` and separate `support_refs`
per item. The bridge validates all refs, requires distinct anchors and clean
actor/modality/state/stem checks, then recomputes Gold from source positions.
Only the bridged copy is marked
`QWEN_REVIEWED_MECHANICALLY_VERIFIED_READY_FOR_SOLVER`; the raw candidates and
source files remain unchanged. Multiple Qwen passes are diagnostic and are not
treated as independent teachers or correctness proof.

