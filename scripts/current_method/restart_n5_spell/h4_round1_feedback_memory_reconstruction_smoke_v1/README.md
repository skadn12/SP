# Round-1 Feedback Memory Reconstruction Smoke

This package currently implements only the offline preflight. It resolves the
first compiler-selected root, validates the historical R01 Add artifacts and
successful read-only R02 control lineage, and writes deterministic new-namespace
Parent/R01 Add plans. It does not call an API, Memorax, Solver, QA writer, or
auditor. A later live bootstrap must consume these plans explicitly and must
keep the control reconstruction at R01.

```bash
python -m scripts.current_method.restart_n5_spell.h4_round1_feedback_memory_reconstruction_smoke_v1.run
```
