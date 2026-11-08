# Independent M-clean evaluation for the three G retry candidates

This run evaluates only `GS1_189272_01`, `GS1_249232_01`, and `cc-train-episode-59126-G-retry1-001` after the frozen blind quality and source checks. It reuses the established local M-clean prompt, schema, parser, scorer, and executor.

The cap is 36 calls per candidate and 108 total: six unique option permutations, three repeats, in Full and No-context. Invalid or truncated answers score zero in the fixed denominator. There are no retries or repairs. Each new question uses item slot 1 in its root/G stream because the prior G candidate occupied slot 0; exact seeds are frozen in `NEXT_SLOT_SEED_FREEZE.json`.

The run has an independent budget ledger and request list. It does not modify prior runs or the 135-item export and does not start training.
