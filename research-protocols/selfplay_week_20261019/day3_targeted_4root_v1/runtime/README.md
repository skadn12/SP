# Targeted four-root diagnostic runtime

Prepared from the sealed `day3_diversity_crossroot_v1/runtime` source files.
Only source code and this updated documentation were copied; old execution,
raw-response, score, private Gold and review files were excluded.

This runtime supports 24 planned slots across four Train roots, six-option MC,
and ordering with three to six items. The new namespace is
`train.feedback.targeted4root_v1`; display seed is 2026102004 and starting
sampling seed is 2026102040. Inputs and new seeds must be frozen before prepare.

`ADAPTATION_NOTES.json` documents required protocol, source and admission fields,
copy provenance, the one namespace adaptation and the readiness snapshot.
`ADAPTATION_CPU_INTERFACE_CHECK.json` records 30 CPU checks, no inference and no
prepare invocation. The synthetic 260-turn fixture verifies interface capacity;
actual source length remains subject to prompt + 32 <= 16384. Full source is
required and overflow fails without truncation.

Model and decoding inherit S0/M-clean: temperature 0.8, top-p 0.9, max output
32 tokens, thinking disabled, context 16384, batch 8, GPU fraction 0.8,
prefix caching disabled, no retries. MC uses six mappings and three samples
(18 requests); ordering uses min(12, n!) mappings and one sample. All 24 planned
slots remain recorded even if content admission produces fewer questions.

Preparation hashes and copies private bytes to a disjoint private sidecar but
does not parse Gold. Execution reads only public files. The scorer first parses
private Gold only after the exact terminal raw response set is frozen. This is
the scoring process boundary, not a claim of author/reviewer Gold blindness.

No prepare, execution or score has run in this new runtime. After separate
content admission and execution freeze, use the proven environment:

```bash
/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python -B prepare.py \
  --public PUBLIC.json --private PRIVATE.json --protocol EXECUTION_FREEZE.json \
  --admission CONTENT_ADMISSION.json --run direct_v1
```

A separate independent public preflight PASS binding direct_v1/PREFLIGHT_FREEZE
is required before live. Then execute.py verify/live and score.py follow the
unchanged command interface. Do not call old selftest.py yet: its integration
fixture invokes a synthetic prepare, which was outside this adaptation task.
