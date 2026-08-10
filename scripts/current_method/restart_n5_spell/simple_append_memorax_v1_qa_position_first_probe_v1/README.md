# Position-Only Semantic QA Writer Probe

This is an isolated QA-only branch. It reads the frozen `episode-146700`
expanded dialogue and local Memorax overview, but it never calls Expansion,
Planner, Realizer, Memorax, Solver, or any historical Writer output.

The active contract is:

```text
Writer: structure_type, evidence_positions, question, options, gold_text,
        distractor_sources (only the three wrong option letters)
Program: candidate ID, canonical turn ID, D alias, supporting/evidence arrays,
         source scope, correct option, final MCQ schema
```

`evidence_positions` are one-based positions in the complete frozen dialogue.
The program rejects out-of-range, duplicate, empty, or more-than-eight positions
and maps accepted positions deterministically to the existing MCQ/Solver input.
Each wrong option must cite at least one valid dialogue position. The program
checks those references mechanically and records cross-session/scene provenance
as an audit warning. It does not make a local semantic near-miss judgment; the
Writer must skip a candidate unless all three distractors are real facts from the
same local story arc and compete on the same answer dimension.

Zero-API preflight:

```bash
python -m scripts.current_method.restart_n5_spell.simple_append_memorax_v1_qa_position_first_probe_v1.position_preflight
```

The future one-call Writer-only entry is
`position_writer_live.py`; it is not executed by this offline task.
