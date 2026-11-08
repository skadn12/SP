# Frozen v28 multi-select trial without Codex item review

This independent, offline trial tests whether frozen Qwen drafts can reach the Solver with mechanical checks and an automated content gate. No question was manually repaired. No Solver inference, training, export, or alteration of the historical Codex review was performed. The generator was SiliconFlow `Qwen/Qwen3.6-35B-A3B`; the content reviewer was SiliconFlow `deepseek-ai/DeepSeek-V3.2`.

| Train root | Original drafts | Mechanical candidates | First model gate | Final automated gate |
| --- | ---: | ---: | ---: | ---: |
| `194259` | 4 | 1 | 0 | 0 |
| `210046` | 4 | 4 | 2 | 0 |
| `231899` | 4 | 0 | 0 | 0 |
| `96544` | 4 | 2 | 0 | 0 |
| **Total** | **16** | **7** | **2** | **0** |

`SUMMARY.json` is the machine-readable count manifest. The operative per-root decision is `REVIEW_RESULT_POLICY_V3.json`; the earlier `REVIEW_RESULT.json`, `REVIEW_RESULT_REPARSED.json`, and `REVIEW_RESULT_POLICY_V2.json` preserve intermediate results and must not be used as the final gate. `NO_RESPONSE` for `231899` means that its mechanical candidate count was zero, so no model request was made. The three other DeepSeek requests returned HTTP 200 with matching model IDs. A separate initial TJU AI `tju-llm` attempt failed TLS handshake for `194259` and was not retried; its transport record remains in the v1 directory.

DeepSeek's first decision accepted `210046` questions 01 and 02. Its explanations treated `looking forward to it` as specifically looking forward to meeting the team and `taking this seriously` as specifically taking the course seriously, although the cited quote does not itself identify those referents. It also treated a statement about wanting to help others as direct contradiction of retaining an award for personal use in `194259`. These are concrete examples of an automated reviewer over-reading evidence. The final gate adds a strict negative-evidence cue check and rejects any quote containing one of a fixed set of pronouns or demonstratives. It therefore does not pass the two first-round positives.

That pronoun rule is deliberately conservative: some quotes with `it`, `this`, or `them` are actually understandable, and the rule can reject a valid question. The rule was tightened after inspecting this batch, so the zero-yield result is a calibrated pilot outcome, not an unbiased estimate of future yield or a proof that all seven candidates are factually wrong. The older Codex review judged zero of the seven originals usable unchanged; that comparison is retrospective only and was not provided to DeepSeek or read by the automatic gate.

This pilot establishes that removing Codex from the individual approval path is technically possible, but this gate delivered no Solver-eligible multi-select question in these four roots. Future scale-up needs a separately held-out evaluation of false approvals and false rejections before making the automatic gate authoritative. A useful next test is an independently authored reviewer with explicit claim-to-quote entailment checks, followed by Solver only for the questions it passes. The existing ground and full training selection policy remains separate from this content gate.
