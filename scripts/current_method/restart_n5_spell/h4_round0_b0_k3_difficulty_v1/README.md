# Full-pool B0 K=3 Difficulty Screening

This isolated route screens the frozen `h4_unexpanded122_qa_generation_only_v1`
candidate pool with base `/root/autodl-tmp/Qwen3-4B`. It stops after mutually
exclusive difficulty buckets and split-specific hard manifests.

The phases are deliberately separate:

```text
preflight -> smoke -> batch -> finalize
```

- `preflight` inventories the read-only source, freezes the exposure-aware root
  split, performs the pure-program candidate/context gate, and freezes the
  exact Solver contract and request plan. It does not load vLLM.
- `smoke` evaluates a deterministic two-root, type-balanced six-candidate
  subset. It requires `--live` and an idle GPU.
- `batch` requires a passing smoke and reuses the same request keys/results.
- `finalize` creates difficulty summaries and train/dev/test hard manifests.

Live phases hold a non-blocking run lock from frozen-contract validation through
backend shutdown. Every request keeps immutable attempt history and an atomic
response envelope. On restart, orphaned requests with a complete envelope are
committed without generation; requests without one retry the exact contract and
paired seed. Cached failures remain technical records but do not retrigger the
current process's consecutive-failure circuit breaker.

The old merged stage2 checkpoint is forbidden. Structured questions freeze the
finite-choice API exposed by the installed vLLM (`guided_decoding` on 0.10.x or
`structured_outputs` on newer releases), and all modes use paired seeds derived
only from `(root_id, candidate_id, slot_id)`.

Prompt length is checked with the frozen tokenizer before execution. Legal
answer vocabularies must fit below `max_tokens`, and any vLLM
`finish_reason=length` response is unscored and routed to technical exclusion.

No phase performs semantic audit, Strict/Broad classification, training,
B1/B2, fresh mining, or another expansion round.

Code-only checks can be run without any Solver calls:

```bash
PYTHONPATH=. python -m unittest \
  scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1.test_round0
```
