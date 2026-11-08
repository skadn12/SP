# v56 multiselect diversity collection

This isolated code snapshot emits exactly two multiselect candidate slots per generation run. It retains the six-option schema and accepts 2–6 correct options. A soft prompt preference seeks explicitly evidenced counterexamples first and supplements 2-, 3-, and 4-answer questions. Five- and six-answer questions remain valid; there is no fixed answer quota, additional admission gate, Gold rewrite, or forced abstention solely because an answer count is high.

Each run allows one initial generation request and at most one mechanical repair request, with a 4096-token output cap. The maximum Solver budget is 36 full-context calls; no ordering candidate or ordering-review request is scheduled. Complete source dialogue, frozen prior-round feedback, public-only prior question inventory, random display permutation, and existing Solver selection remain unchanged.

`evaluate_and_pool.py` loads the local `deliver.py`; missing question types display zero counts without fabricated ordering rows or requests. The original v15 delivery and all v55 code remain bound in `BASELINE_FILES.json` and unchanged. The copied `batch.py` and older tests are retained as source history; collection uses the dedicated v56 report orchestrator with `generate.py` and `evaluate_and_pool.py`, not the legacy mixed-type batch helper.

Offline checks cover all allowed answer counts with raw Gold preservation, rejection of unresolved relations and falsified Gold, the actual final request with full source and frozen feedback, two-slot budgets, and MC-only delivery with nonempty selection, empty selection, and no candidates. See `TARGETED_TEST_RESULTS.json` and `TARGETED_TESTS.log`. No model calls are made by these tests.
