# Raw Qwen multiselect Solver trial v31

This independent offline bridge reads the frozen v28 `Qwen/Qwen3.6-35B-A3B` drafts from four train roots. It preserves every option's text and original order. Gold labels come directly from Qwen's `supports` relations. No Codex edits, semantic review, Solver calls, or training occur in this bridge.

`raw_source_and_gold_v1` checks the frozen model/request/source hashes, candidate IDs and schema, six nonempty unique options, Gold cardinality 2-6, source turn existence, exact speaker match, and verbatim quote inclusion. It does not enforce six distinct cited turns, an R01 citation, quote self-containment, topic consistency, novelty, or semantic support/contradiction. Passing this gate does **not** establish that Gold is factually correct.

Run `python bridge.py` to write `POOL.json` and `REPORT.json` once. The pool is compatible with the multiselect Solver v2 input contract. Every one of the 16 original draft slots appears in the report, including any mechanical rejection or generator abstention. Historical v28 records are read only.
