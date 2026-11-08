# Original-wording paired M-clean evaluation

Research-only comparison against the completed wording Solver v2. The four
items, frozen sources, displays, request order, batch size, seeds, model,
tokenizer, sampling parameters, schemas, and scoring are inherited from v2.
Only the question text is restored from its hash-checked first draft. The
45849 ordering item remains on hold. No training export or retry is performed.

From this directory, with `/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python`:

- `adapter.py prepare`: freeze original items and prepare 144 requests on CPU;
  compare all requests and prompt/schema bytes with the bound v2 artifacts.
- `adapter.py verify`: recheck all frozen bindings and pairwise invariants;
  this does not run inference.
- `adapter.py run`: repeat the checks, then use the historical local vLLM
  executor and final-answer-only scorer. An interrupted run is not retried.

Per-request seed and ordering are matched, but different prompts can produce
different token streams; this does not guarantee paired randomness or rule out
the effect of stochastic sampling. Each arm is run at a different time.
