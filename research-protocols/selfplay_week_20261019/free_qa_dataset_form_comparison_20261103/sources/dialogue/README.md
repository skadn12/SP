# Official Dialogue Dataset Form Comparison

This directory contains read-only snapshots and small, non-sensitive extracts used to compare question forms. The LoCoMo repository is pinned to commit `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`; LongMemEval is pinned to commit `9e0b455f4ef0e2ab8f2e582289761153549043fc`. URLs, retrieval times, sizes, and SHA-256 values are in `DOWNLOAD_MANIFEST.json`.

## LoCoMo

The official README (`locomo_README.MD`) describes QA categories corresponding to single-hop, multi-hop, temporal, commonsense/world knowledge, and adversarial reasoning. The released `locomo10.json` stores open question/answer records with category IDs and evidence turn IDs; it does not use the local project's A-F or I1-In display format. `locomo_sample_pool.json` contains four exact source records selected from the official file, including a multi-hop list, a temporal question, an open-domain comparison, and an adversarial item.

## LongMemEval

The official README (`longmemeval_README.md`) describes 500 questions across information extraction, multi-session reasoning, knowledge updates, temporal reasoning, and abstention. The `longmemeval_oracle.json` extract is summarized in `longmemeval_sample_pool.json`; examples remain open short-answer questions and include temporal comparisons/date calculations, knowledge-update questions, and multi-session aggregation. It does not prescribe local multiple-choice or ordering labels.

## Interpretation

The three local free-generation questions are two multi-select questions and one complete-order question. Their labels are stable and useful for scoring, but this batch repeats a narrow response structure and concentrates on plan/travel-state verification and a workflow sequence. The official datasets vary the information need and reasoning category while retaining open answers. Therefore the local batch is structurally narrower than these references, although this comparison does not by itself establish that open-answer questions are harder or better. It supports expanding the discovery target and natural question forms, rather than deleting the existing diagnostic formats.
