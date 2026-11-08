# H4 QA Mining Hybrid Generation Probe v1

This is an isolated Phase-0 refinement probe. It freezes the existing expanded
dialogue and intent contracts, then compares two candidate generators:

* contract-guided: 2 single-choice, 1 multi-select, 1 ordering per intent;
* free exploration: 4 single-choice, 2 multi-select, 2 ordering per intent.

The contract is metadata/guidance, never a semantic admission gate. The basic
validator checks only JSON/schema integrity, option/gold consistency, and that
evidence turn references exist in the frozen dialogue. It deliberately does
not judge whether the cited text semantically supports the answer.

Both sources share the existing isolated Solver adapter, full/no-context K=5
replay, and deterministic position rebinding. Single-choice uses temperature
`0.7`, multi-select uses `0.3`, and ordering uses `0.0`. The difficulty policy is:

```
hard_candidate: 0 < full_context_accuracy < 1 and no_context_accuracy < 1
invalid_or_ambiguous: full_context_accuracy == 0
context_helpful: full_context_accuracy > no_context_accuracy (auxiliary metric)
```

Run with `python -m scripts.current_method.restart_n5_spell.h4_qa_mining_hybrid_generation_v1.run --live`.
The default output is a new run under `restart-n5-runs/`; no production or
historical files are written.
