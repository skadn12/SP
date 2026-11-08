# H4 partial admission and frozen Memory input modes

Opt-in `admission_policy="partial_v1"` on the existing shared Writer. No global model/config changes.

Offline entry: `python -m scripts.current_method.restart_n5_spell.h4_writer_partial_memory_v1 {prepare,replay,preflight,mock}`. Real generation uses the same entry with `live`, one explicit scope and exact primary/repair/HTTP budget flags. See the prepared run HANDOFF for commands, provenance, unknown service capacity, and authorization boundaries.

Default run: `/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_writer_partial_memory_v1_prepare_1/`.

Sources: `TRANSCRIPT_ONLY`, `TRANSCRIPT_PLUS_MEMORY`, `MEMORY_ONLY`; each requests 2 SC + 1 MS + 1 ordering. Memory-only uses a generic slot without transcript-derived intents. The original TEMPORAL DAG/Gold machinery resolves references through an explicit source envelope.

Replays join frozen semantic diagnostics only after program admission. Runtime uses the existing one-attempt ProfileClient; no Memory client is invoked. No retries/resume after started scopes, and no model calls during preparation or mocks.
