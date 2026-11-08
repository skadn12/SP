# B0 K=3 Hard Evidence Stratification v1

This is an isolated, offline classification stage for the frozen
`h4_round0_b0_k3_difficulty_v1_live_1` output. It reads the immutable
`difficulty_manifest.jsonl`, hard split manifests, program gate, root split,
candidate pools, and complete Parent + Child dialogue snapshots.

It makes zero Solver calls, zero external API calls, and zero semantic-auditor
calls. It never modifies the source run, candidate pools, dialogue snapshots,
gold answers, or evidence.

Run:

```bash
PYTHONPATH=. python -m scripts.current_method.restart_n5_spell.h4_round0_hard_evidence_stratification_v1.run
```

The runner fails closed on count, ID, split, lineage, or SHA drift. Outputs are
written once with SHA-256 sidecars to a new versioned run directory.

`EVIDENCE_CLEAN` is an evidence-packaging status, not a semantic truth proof.
`EVIDENCE_RISK` does not mean that the gold answer is wrong. Train clean rows
are marked `strict_eligible`; train risk rows are marked
`broad_only_eligible`. Dev and test rows are never marked training-eligible.
