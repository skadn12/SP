# Ordering generation without Codex content review: frozen v5 replay

Run `python build.py` once to replay the four frozen Qwen3.6-35B-A3B ordering responses. This script makes no model, Solver, or training calls. It writes independent artifacts and does not change the original drafts or reviewed versions.

`REPORT.json` separates three outcomes:

- **Zero-edit raw draft:** use the original mechanical validation as-is.
- **Label-only rebind:** deterministically shuffle item/evidence labels and recompute Gold from source turn indices. The generated paraphrased item text stays unchanged. These candidates still require an independent semantic reviewer for entailment, reference resolution, topical coherence, and novelty. Mechanical eligibility does not authorize Solver or training.
- **Verbatim-excerpt variant:** a distinct task that displays speaker plus exact, source-unique quote rather than the model's paraphrased statement. Exact matching can verify source grounding without judging the paraphrase, but this task may be substantially easier and is not evidence that semantic item generation is fixed.

The replay binds each source and response to the original hashes and returned model. All candidate counts are preliminary; `selected_for_training` stays zero pending appropriate review and Solver selection.

`judge.py` prepared one frozen TJU `tju-llm` review request per root. Its first attempt failed before HTTP during TLS handshake; that error is recorded without retry. `judge_deepseek.py` is a separate reviewer attempt using SiliconFlow `deepseek-ai/DeepSeek-V3.2`, with the same item-level and whole-question audit prompt, one request per root, and fail-closed JSON parsing. The independent review does not edit model-written item text.

The first DeepSeek pass accepted 6 of the 10 label-rebound candidates. Because it missed an unsupported location in one item, `challenge_deepseek.py` made a separately frozen adversarial check of those six, focused on claims absent from the quoted source turn. Its intersection accepted 2 candidates, both under root `194259` (`order-v5-001` and `order-v5-003`). The second pass rejected four candidates for unsupported referents, objects, timing, or modality. These are two prompts to the **same** reviewer model, not two independent reviewers or a measured accuracy guarantee. The excerpt variant is excluded from this original-task yield. Solver selection and training admission are separate downstream stages.

`judge_glm.py` applies the same quote-only entailment standard to all 10 rebound candidates using a different checkpoint, SiliconFlow `zai-org/GLM-5.2`. GLM accepted 2/10 on its own; the intersection of both DeepSeek passes and GLM accepted **one** candidate, `cc-train-episode-194259-order-v5-003`. GLM caught unsupported claims that both DeepSeek prompts missed, including `nervous` and a pronoun-resolved speech reference. The three verdict files retain per-item explanations and request/response hashes. One accepted candidate is only a content-gate result, not a Solver or training result.

| Root | Original drafts | Raw mechanical pass | Deterministic label rebind | DeepSeek first pass | DeepSeek challenge | GLM pass | Three-gate intersection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `210046` | 4 | 0 | 2 | 1 | 0 | 0 | 0 |
| `231899` | 4 | 0 | 3 | 2 | 0 | 0 | 0 |
| `194259` | 4 | 0 | 4 | 3 | 2 | 2 | 1 |
| `96544` | 4 | 0 | 1 | 0 | 0 | 0 | 0 |
| **Total** | **16** | **0** | **10** | **6** | **2** | **2** | **1** |
