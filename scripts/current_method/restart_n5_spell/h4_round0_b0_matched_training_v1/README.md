# Round-0 B0 Matched Training Dataset v1

This isolated offline route constructs two equal-size answer-only SFT datasets
from the frozen Round-0 train-hard pool:

- `M_clean_policy`: all eligible Clean hard candidates;
- `M_inclusive_policy`: the same shared Clean core with matched Risk
  replacements.

It evaluates both 10% and 20% replacement feasibility and selects 20% only
when 77 unique pairs satisfy exact question type, Gold cardinality, and
position mechanism plus bounded accuracy and token-length deltas. Displayed
targets are assigned per shared/replacement slot, so both policies have exactly
the same label and ordering-target distributions.

The rendered JSONL uses the frozen full Parent + Child B0 chat format and a
single short `<box>...</box>` assistant answer. Evidence, rationales,
difficulty metadata, and audit explanations are not included in model-visible
messages. The repository trainer masks all non-assistant tokens.

The route also freezes common full and M0-hard Dev/Test manifests and the B0
evaluation protocol. It does not load a model, call Solver or an external API,
run a semantic auditor, construct a matched training run, or start training.

The rendered records can exceed the trainer's 4096-token default. The frozen
`training_interface_contract.json` therefore requires `--max-length 8192` for
future training; left-context truncation is forbidden because it would violate
the full Parent + Child input contract.

```bash
PYTHONPATH=. python -m scripts.current_method.restart_n5_spell.h4_round0_b0_matched_training_v1.run
```

The output run is immutable: an existing run directory is rejected, and every
artifact receives a SHA-256 sidecar.
