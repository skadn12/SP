# Difficulty Probe Suite v1

This is a fully isolated, read-only difficulty probe suite. It has exactly
five conceptual hypotheses and six execution probes:

| Hypothesis | Execution probe |
| --- | --- |
| H1: QA paraphrase robustness | H1/P1 |
| H2: evidence perturbation / context augmentation | H2/P2 |
| H3: relation-first QA control | H3/P3 |
| H4: state-evolution expansion | H4A/P4A question-aware; H4B/P4B memory/world-only |
| H5: multi-seed small-world expansion | H5/P5 |

H4A and H4B are two isolated execution variants of H4. They are not two
research directions.

The original `p2_source_paraphrase/` output is retained as the **lexical
robustness baseline**. The independent target experiment is
`p2_evidence_perturbation/`: it adds controlled background, competing-state,
or entity/event-interference context without changing the question, options,
gold, evidence refs, timeline, entity bindings, or final state.

Run it independently with:

```bash
python -m scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.p2_evidence_perturbation.probe \
  --live --max-items 3
```

Its run directory is separate and reports preservation validity, perturbation
type distribution, original/perturbed/no-context/option-only Solver accuracy,
and failure examples. It never calls Planner, Realizer, Writer, or Memorax.

The runner reads the frozen 41625 expanded dialogue, frozen QA batch, and
historical evaluation reports. It writes only to a new
`difficulty_probe_suite_v1_<timestamp>` directory under `restart-n5-runs`.
It never calls the existing QA Writer, never calls Memorax Add, never changes
Parent/Child data, and never edits production source files.

The probe design borrows evaluation ideas rather than training code: HotpotQA
multi-document supporting facts, MuSiQue's compositional connected-hop and
positive-distractor controls, 2WikiMultiHopQA's explicit reasoning paths,
LoCoMo's long temporal/causal conversation setting, and LongMemEval's update
and temporal-reasoning views. SELF-HARMONY is used only as inspiration for
separating generation from verification; no training or GRPO code is called.

Primary references: [HotpotQA](https://arxiv.org/abs/1809.09600),
[MuSiQue](https://arxiv.org/abs/2108.00573),
[2WikiMultiHopQA](https://arxiv.org/abs/2011.01060),
[LoCoMo](https://arxiv.org/abs/2402.17753), and
[LongMemEval](https://arxiv.org/abs/2410.10813).

Run a small local Solver smoke and deterministic fallbacks:

```bash
python -m scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.run_suite \
  --max-items 3 --max-state-roots 2
```

To permit the isolated Planner/Realizer/paraphrase API requests, add
`--live --config /root/autodl-tmp/gxy-selfplay/config.json`. API failures are
recorded and do not silently become successful generations.
