# H4 K=3 Semantic Audit v1

This is a new, isolated semantic review stage for the frozen output of
`h4_qa_multiroot_k3_t08_probe_v1`. It audits the 37 hard candidates and the 20
`full_context=0` candidates against the complete named dialogue.

It does not modify the input run, production QA, Planner, Realizer, Memorax,
Solver, or any historical run. It makes only independent semantic-auditor API
calls; it does not regenerate questions and does not run Solver.

```bash
PYTHONPATH=. python -m scripts.current_method.restart_n5_spell.h4_qa_multiroot_k3_t08_semantic_audit_v1.runner \
  --phase inventory

PYTHONPATH=. python -m scripts.current_method.restart_n5_spell.h4_qa_multiroot_k3_t08_semantic_audit_v1.runner \
  --phase smoke

PYTHONPATH=. python -m scripts.current_method.restart_n5_spell.h4_qa_multiroot_k3_t08_semantic_audit_v1.runner \
  --phase batch
```

The auditor separately reports:

- full-dialogue answer correctness;
- submitted evidence sufficiency;
- question leakage and option ambiguity;
- reasoning type and evidence gaps.

`training_ready` requires a correct, determinate, non-leaking question with a
sufficient submitted evidence set. This is a review export, not production
admission.

