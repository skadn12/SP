# Round-1 strong-model comparison

This package performs an offline R01/R02 lineage preflight and then supports
blind, mirrored evaluation with the independent `TJU AI` `tju-llm` endpoint.
It never invokes QA, Solver, a semantic auditor, or training code.

The private configuration defaults to:

```text
/root/.config/gxy-selfplay/round1_strong_judge.env
```

Run the immutable 70-root offline preflight:

```bash
python -m scripts.current_method.restart_n5_spell.h4_round1_strong_model_comparison_v1.run \
  --phase preflight
```

After a passing preflight, send exactly one first-mirror protocol smoke request:

```bash
python -m scripts.current_method.restart_n5_spell.h4_round1_strong_model_comparison_v1.run \
  --phase smoke-one
```

The smoke result is not a final root decision. A second mirrored request is
required before pairwise adjudication.
