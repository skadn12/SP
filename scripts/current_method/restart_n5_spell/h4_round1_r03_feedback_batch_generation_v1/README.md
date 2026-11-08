# R03 feedback batch generation

R03 now uses the agreed feedback policy and fixed intent slots while retaining
the first-round Qwen3-14B QA generation configuration. The batch targets are
4 guided and 8 free candidates per safe intent, not 5 in both modes. Three safe
intents give a target of 36 candidates per root; this is not a minimum yield.

The editable policy is `qa_feedback_policy_v2.md`; the slot/strategy and output
field contract is `intent_contract_v1.json`. These are versioned source files.
Each run copies them and freezes hashes before calling the model. The miner
instantiates this contract; it cannot modify it. `generation_allowed` is local
validation output only. `NO_SAFE_EVOLUTION` skips the corresponding generators.

`configuration_comparison.json` records inherited settings and intentional
changes. The historical templates and run outputs are preserved. The generator
field layout, structural normalizer, validator and option permutations are
reused from the first round. The intentional policy changes include soft count
targets and root-level missing-type repair, as agreed in the conversation.

Inputs are pinned to the same seven-root cumulative-dialogue manifest used by
the R03 composition/fresh pilot, not selected by latest directory mtime. These
files include Parent, R01, R02 and R03. New Realizer drafts are not silently
substituted. 112812 stays out of this baseline; 96462 has no eligible historical
feedback seed, so only Coverage/Fresh intents can be scheduled for it.

Coverage is the set of currently uncited R02/R03 turns in the frozen active
training bank. It is a navigation signal, not proof of semantic undercoverage.
Readable historical stems reach generators for replay avoidance; Solver
observations reach only the miner. Historical source files and candidate IDs
are pinned to the frozen active-train bank scope.

From the repository root:

```bash
python -m unittest scripts.current_method.restart_n5_spell.h4_round1_r03_feedback_batch_generation_v1.test_workflow
python -m scripts.current_method.restart_n5_spell.h4_round1_r03_feedback_batch_generation_v1.run
python -m scripts.current_method.restart_n5_spell.h4_round1_r03_feedback_batch_generation_v1.run --live --resume
```

The compact intent renderer is available through `run_revision1`. It renders
the same fixed slots with a complete record example and an R03 reference list
at the end, without asking the model to interpret a full JSON Schema. The local
schema remains authoritative. `--reuse-intent ROOT=PATH` can reuse saved raw miner
output after hash verification; the report records zero new miner calls for it.

Reference normalization follows first-round conventions: empty optional metadata
can be canonicalized locally, and valid R03 references can supplement rather than
repeat the reasoning-chain groups. No semantic evidence, IDs or strategies are
invented during normalization. Equal stems with different option content trigger
review, not automatic duplicate rejection.

Preparation runs locally, freezes all inputs and renders the real miner prompts.
The live runner uses one miner per root, two batches per safe intent, and at most
one repair per entirely missing question type. All requests and responses use
the existing request ledger. Resume reuses completed requests under identical
hashes and refuses automatic resend of ambiguous requests. A failed transport
or parse does not trigger a repair request.

Outputs include per-root intents, raw prompts/responses, candidate pool, local
audit, individually accepted candidates and permutation checks. Semantic
alignment, novelty and R03 necessity remain pending. These are adaptive QA
candidates; structural acceptance is not final QA admission or training.
No Solver evaluation, training or memory mutation occurs in this runner.
