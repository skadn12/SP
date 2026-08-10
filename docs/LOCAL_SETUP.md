# Local Setup and File Layout

This repository contains the local orchestration, validation, prompts, and test
code. The current expansion path still sends Planner, Gate, Realizer, and QA
Writer requests to the configured remote OpenAI-compatible API; Memorax is also
accessed through its configured HTTP API. Localizing the code does not localize
either service.

## Required Local Files

| Path | Required for | Purpose |
| --- | --- | --- |
| `scripts/` | Any module invocation | Python package root. Keep `scripts/current_method/restart_n5_spell/` and its common helpers. |
| `scripts/current_method/restart_n5_spell/simple_append_memorax_v1_memory_event_continuation_probe_v1/` | Memorax-driven R1 | Current Planner, Gate, candidate selection/isolation, Realizer workflow, contracts, prompts, and live runner. |
| `scripts/current_method/restart_n5_spell/simple_append_memorax_v1/` | QA-only flows | Memorax QA and admission helpers. |
| `scripts/current_method/restart_n5_spell/simple_append_v1/` | Shared QA helpers | Assembly, QA materialization, Solver orchestration, and validation helpers. |
| `scripts/current_method/restart_n5_spell/core.py` | Shared runtime | Atomic files, hashing, and deterministic checks. |
| `scripts/current_method/restart_n5_spell/ledger.py` | API accounting | Request ledger, budget enforcement, and request artifacts. |
| `scripts/current_method/restart_n5_spell/live_backends.py` | Qwen API | OpenAI-compatible Challenger client. |
| `scripts/current_method/restart_n5_spell/backends.py` | Type contracts | Backend request/response dataclasses. |
| `scripts/current_method/restart_n5_spell/unified_dialogue_root.py` | Input validation | Authoritative Parent schema and provenance checks. |
| `requirements.txt` | Python environment | Runtime dependencies. `vllm`, `torch`, and training packages are optional unless local Solver/training is used. |
| `configs/config.example.json` | Configuration template | Safe template for the remote Qwen API. |
| `tests/` | Verification | Unit and regression tests. |

For only the current R1 expansion, the target module plus the common files in
the table is sufficient. Keeping the complete `scripts/current_method` tree is
recommended because several historical QA and expansion entry points share
helpers and makes `python -m` imports predictable.

## Files Supplied Per Machine

These are environment/data inputs, not repository source:

```text
config.json                         # copy from configs/config.example.json and fill API settings
~/.config/memorax/credentials.env   # Memorax API credential, never commit
<input-root>/unified_root.json      # selected authoritative Parent
<tokenizer>/tokenizer.json          # local token counting for preflight
<tokenizer>/tokenizer_config.json
<tokenizer>/vocab.json              # or the tokenizer's equivalent vocab files
<tokenizer>/merges.txt
```

The runners accept explicit paths, so the server's absolute paths do not need
to be reproduced. The tokenizer is used for token accounting only; the remote
Qwen workflow does not require local model weights. The server's Qwen3-4B
directory is about 7.6 GB because it also contains weights. Copying only its
tokenizer files is about 16 MB.

Do not copy these into Git:

```text
config.json
*.env / .env
model-*.safetensors / checkpoints / merged model directories
restart-n5-runs/ and generated reports
raw API responses containing private data unless explicitly curated
```

## Remote API Configuration

Create an untracked `config.json` from the example:

```json
{
  "api": {
    "base_url": "https://<challenger-provider>/v1",
    "model": "Qwen/Qwen3-14B",
    "api_key": "YOUR_KEY"
  }
}
```

Memorax remains configured separately through its credentials file and the
runner's namespace/transport options. Each experiment should use a distinct
namespace; do not reuse a namespace containing a prior Parent or Child.

The API-only process needs no GPU:

```text
single-process RAM:       1–2 GB
recommended RAM:          4 GB
three concurrent runs:    8 GB recommended
disk for code + inputs:   normally below 1 GB
GPU:                      not required
```

## Optional Local Solver

The QA pipeline can run Solver locally through vLLM while Writer and Expansion
continue using remote APIs. The local Solver needs:

```text
scripts/current_method/restart_n5_spell/vllm_backends.py
vllm + torch
solver model directory (all safetensors shards and tokenizer files)
```

The current 4B Solver checkpoint is about 7.6 GB and is comfortable on a
16–24 GB GPU. Run Solver sequentially across a batch so one vLLM engine is
reused rather than loading one engine per process.

## Generated Artifacts

Each run root contains frozen manifests, prompts, raw responses, validation
results, and reports. These are local audit artifacts. Curated summaries can be
copied into `docs/`, but full run directories should stay outside the repo.

## Verification

From the repository root:

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m unittest discover -s scripts/current_method -p 'test_*.py'
python -m compileall -q scripts
git diff --check
```

Use each live module's `--preflight-only` mode before API execution. Launch
relative-import runners with `python -m`, not by executing their `.py` file.
