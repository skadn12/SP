# Wording Solver research evaluation: v2

This is a new explicitly recorded CPU preparation after v1 was interrupted
before any requests or model calls. The v1 ledger and partial inputs are
preserved. No Solver output is replayed or overwritten. V2 reserves 144 calls
for the four reviewed candidates and holds 45849 for ordering ambiguity.

The original blind review, semantic comparison and separate cross-arm
addendum are referenced and byte-bound at their v1 paths. Dedup PASS means
comparison completed, not novel or training-eligible. Same-model reviews and
incomplete historical comparison coverage limit the conclusions; duplicates
remain research diagnostics only. No training export is performed.

Uses the v1 adapter's exact M-clean merged weights and tokenizer, historical
executor, 6144 output tokens, full/no context, six Gold-blind displays each
with three repeats, final-answer-only task score and auxiliary explanation
diagnostics. V2 has a distinct display and sampling seed namespace.

From this directory, use `/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python`:

- `python adapter.py manifest` creates the bound review manifest once.
- `python adapter.py self-check` performs offline synthetic fixture checks.
- `python adapter.py check-inputs` verifies review/source/model hashes.
- `python adapter.py prepare` reserves one attempt and freezes prompts and
  schemas on CPU. Any interrupted attempt remains recorded, without retry.
- `python adapter.py run` verifies frozen bindings before GPU model loading;
  the historical executor records per-request outputs and scores on completion.
- `python adapter.py score` requires a completed run and verifies frozen input,
  prompt and schema bindings before scoring; outputs are never overwritten.
