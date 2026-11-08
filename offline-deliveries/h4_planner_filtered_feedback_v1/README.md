# H4 Planner filtered-feedback offline preparation

This is the new explicit preparation entry for a future R03 Planner branch. It reads the delivered filtered export, binds the authoritative Parent+R01+R02 planning history, renders the initial Planner request, and renders a supplement that inherits the exact initial `shared_feedback_view`. It never selects a new card, calls a model, sends HTTP, or changes the current 122711 expansion.

## Reproduce

```bash
cd /root/autodl-tmp/gxy-selfplay/current_method_github
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=scripts
python3 -m current_method.restart_n5_spell.h4_round1_r03_incremental_plan_retention_v1.planner_preparation \
  --config /root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_planner_filtered_feedback_v1/config.json \
  --output /absolute/new-empty-directory
```

The checked run is `prepared_with_mock/`; its `mock_initial.json` is an existing plan transformed only to the new speaker-reference contract for offline compilation. It is not a model response and no supplement response was generated.

The filtered export is `restart-n5-runs/h4_round1_round0_ordering_feedback_exclusion_v1_offline_live_1/export/artifact_inventory.json`. It reports 2493 retained candidates, 1018 active shared records, and one selected Expansion card for this root. The selected E1Q1 ordering identity is retained because it is outside the ROUND0 exclusion scope. `gold_semantics_validated=false` remains explicit.

The authoritative planning history is the export's `reference_identity` entry for `cc-train-episode-122711`: the R02 expanded dialogue has 143 turns and `source_prefix=Parent+R01+R02-feedback`. The current run's memory overview, event candidates, verified tracks, and budget are separately SHA-bound.

## Delivered evidence

- `config.json`: explicit versioned input identities.
- `prepared_with_mock/initial/{prompt.txt,messages.json,bundle.json,capacity.json}`: initial request and exact bundle.
- `prepared_with_mock/supplement/{prompt.txt,messages.json,capacity.json}`: inherited-feedback supplement request.
- `source_to_prompt_audit.json`: source, feedback, prompt, capacity, and zero-call linkage.
- `protected_before.json`: SHA snapshot for current frozen inputs and expansion artifacts.

The request capacities were measured with the local Qwen3 tokenizer and the current 128K budget. Measured final-message tokens are recorded in each capacity file. Actual formal `R03:Sxx` scene IDs are intentionally absent: the production workflow assigns them only after Gate pass.
