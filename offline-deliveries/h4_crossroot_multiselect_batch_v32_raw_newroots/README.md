# v32 raw Qwen multi-select generation trial

Four fresh train roots (`103866`, `104814`, `135466`, `138343`) were each given four question slots in exactly one SiliconFlow request to `Qwen/Qwen3.6-35B-A3B`. Inputs bind the frozen promoted P+R01 dialogue, split manifest, prior candidate pool, prompt, and provider request by SHA-256. All four responses were persisted and parsed without a retry.

The independent `bridge.py` is the admission gate for the raw Solver pool. It checks schema, source turn existence, speaker match, verbatim quote, option uniqueness, and Gold cardinality/mapping. It preserves each Qwen draft's option order and `supports`-derived Gold. It does not perform semantic entailment, topic, or novelty review, and Codex did not edit questions. `pipeline.py check-response` is retained as a legacy diagnostic: its `mechanically_valid_count=0` is caused primarily by the former six-distinct-turn and R01-per-question constraints and is **not** the raw bridge verdict.

| Root | Qwen drafts | Raw bridge pass | Rejected |
| --- | ---: | ---: | ---: |
| 103866 | 4 | 4 | 0 |
| 104814 | 4 | 3 | 1 |
| 135466 | 4 | 4 | 0 |
| 138343 | 4 | 4 | 0 |
| Total | 16 | 15 | 1 |

`104814-qwen36-multi-v32-02` failed because option A's quote is not a verbatim substring of its cited turn. Among the 15 bridge-pass drafts, Gold cardinality is 3 for four questions, 4 for eight, and 6 for three. The 15 are ready **for Solver evaluation only**, not confirmed training examples. No Solver inference, Gold audit, export, or training was performed in this generation task.

The authoritative handoff is `solver_pool/POOL.json`; per-candidate decisions are in `solver_pool/REPORT.json`. A read-only equality check confirmed that all 15 handoff rows retain the original Qwen proposition text, order, source refs, and `supports`-derived Gold.
