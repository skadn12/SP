# Ordering generation, independent v4 trial

This directory freezes a six-root generation batch using SiliconFlow
`Qwen/Qwen3.6-35B-A3B`. Each initial request asks for **up to four** ordering
questions. Subject to content review and source capacity, two further requests
of up to two each may bring the attempted total to eight per root. The
experimental goal of four to six usable questions is not a guaranteed yield.

`build_dedup.py` freezes one combined candidate snapshot per root from the old
candidate pool (all question types), v3 ordering raw drafts, v25 multi-select
drafts, and the v26 `191749` Codex revision. `dedup_inputs/*/PROVENANCE.json`
records input hashes. Candidate presence is a novelty constraint and does not
mean it entered training. `manifest.json` binds each complete source and
combined dedup snapshot by SHA-256. `BATCH_PREPARED.json` then binds the script,
instructions, request parameters, prompts, and payloads.

Commands run from this directory:

```sh
python generate.py --prepare
python generate.py --execute cc-train-episode-109164
python generate.py --validate cc-train-episode-109164
python generate.py --prepare-top-up cc-train-episode-109164 --reason 'content review shortfall'
python generate.py --execute-top-up cc-train-episode-109164
python generate.py --validate-top-up cc-train-episode-109164
python generate.py --prepare-top-up cc-train-episode-109164 --reason 'remaining uncovered source capacity'
python generate.py --execute-top-up-2 cc-train-episode-109164
python generate.py --validate-top-up-2 cc-train-episode-109164
```

Each execution reserves exactly one HTTP attempt, with no retries, and saves
the raw response before validation. Mechanical validation alone does not prove
the chronology, evidence semantics, distinctness, or novelty; review each
question before deciding whether to top up. No Solver or training is run here.
