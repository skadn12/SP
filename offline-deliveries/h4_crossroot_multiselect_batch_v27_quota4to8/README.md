# Multi-select generation trial v27

Six roots, four initial draft slots per root, with two optional two-slot top-ups after content review. Eight slots per root is a ceiling, not a required output. The generator is SiliconFlow `Qwen/Qwen3.6-35B-A3B`; the Solver and training are outside this batch.

Each `batch-01` directory freezes the source, cross-type dedup summary, prompt, request, and hashes before any HTTP attempt. Dedup context combines the old candidate pool (all question types), ordering v3 parsed model drafts, multi-select v25 parsed model drafts, and the v26 Codex revision for root 191749. Candidate-pool entries are not assumed to have been selected for training.

For a root `cc-train-episode-ID`:

```bash
python pipeline.py preflight cc-train-episode-ID
python pipeline.py execute cc-train-episode-ID
python pipeline.py check-response cc-train-episode-ID
```

`execute` reserves one attempt and does not retry. Mechanical checks do not certify entailment, topic coherence, Gold, or novelty. Review those before binding or top-up. A later top-up requires the preceding batch's `CONTENT_REVIEW.json` to approve it with a reason; it uses the full frozen source and prior model drafts. Run `prepare ... --batch 2` after the first review, then `--batch 3` only if another review justifies it. No selected or training data are emitted by preparation.
