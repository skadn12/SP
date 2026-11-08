# Groundless random feedback V4

This directory contains the offline post-training feedback compiler for the
next self-play round. It is separate from the older V3 delivery.

`compiler_v4.py --compile` reads the fixed previous evaluation run and writes
`compiled_v4/`. Compilation makes no HTTP/model calls and refuses to use a
different source tree or overwrite an existing delivery. `source_lock.json`
records the selected evaluation, model identity, solver contract, formal
training checkpoint, and every candidate-pool SHA.

The delivery covers eight explicit roots and preserves every candidate's
historical M0, clean-policy, and inclusive-policy full/no exact-agreement
counts, primary transition, and training exposure. These are diagnostics, not
semantic Gold certification or current partial scores.

Use `load_feedback_context` for structured data,
`build_feedback_suffix` for a role/model-specific prompt suffix, or
`append_feedback_to_request` to append a suffix to a chat/prompt payload and
receive a provenance receipt. The suffix records the current contracts:
ground removed, multi-select Gold cardinality 2..6, six-item
`conversation_order`, and final selection `0 < S_full < 1` after the complete
full evaluation. Context gain and permutation behavior remain diagnostics.

Run `python test_compiler_v4.py` after compilation for the offline contract
checks.
