# Exact-input amendment adapter

This is the user-approved single infrastructure amendment described in `PROTOCOL_AMENDMENT.md`. Input preparation is disabled. Original prepared bytes, request metadata, model/sampling configuration and two EVAL_HOLD items are preserved exactly. New namespace and actual interpreter/package evidence are separate frozen provenance.

Use only `/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python`.

`adapter.py self-check` is a no-write static/evidence check. `adapter.py check-inputs` streams hashes of copied/original inputs and new code/provenance without model/tokenizer load. Both are safe and avoid legacy fixture discovery. `adapter.py prepare` always refuses.

The controller may run `adapter.py run` exactly once after independently reviewing PLAN_FREEZE. An environment/API precheck precedes the exclusive RUN_ATTEMPT reservation; the backend, frozen model/tokenizer, prompts, schemas, displays, seeds, sampling, and inherited local loop are unchanged. Any consumed run attempt stops further inference, even on infrastructure failure. `score` remains the inherited exclusive offline collection/scoring entry for preserved output after a hard kill; ordinary run calls it itself. No training/export is implemented.

`evaluation/MODEL_CONFIG.json` is byte-identical historical preparation metadata. Actual amended runtime packages/namespace are truthfully frozen in `EXECUTION_CONFIG_AMENDMENT.json` and recorded again in RUN_ATTEMPT. Never edit the old model-config bytes to present them as current runtime facts.
