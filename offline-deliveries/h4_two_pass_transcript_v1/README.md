# Full-transcript two-pass expansion

This is the authorized full-transcript fallback for the stopped Memory experiment.
It uses the complete accepted archives for roots 259192, 96462 and 112812. No
Memory API, frozen Memory snapshot, external plot dataset, Miner, Writer, Solver
or training is included.

## Scope

- A / R03: advance existing events, three scenes.
- B / R04: propose autonomous new developments, three scenes, using the actual
  A dialogue only after the original acceptance checks and Codex offline review.
- Per-scene soft target: 500-700 visible dialogue tokens; existing hard bound:
  9-48 utterances. The soft target does not force padding or additional calls.
- Shared repaired Planner, Gate, Outline, Realizer and Adherence path. At most
  one candidate supplement and one continuation with reaudit per phase.
- Existing user authorization: 60 generation HTTP attempts in total, including
  any attempt on the original route. No automatic retry, parameter fallback,
  standalone connectivity test, or refund for unknown completion.
- Technical failure stops all unsent requests; content failure stops that root.
- School provider: tju-llm; requested temperature 0.25, JSON object output and
  existing per-stage output caps. Actual model and usage come from responses.
  Service context capacity, default reasoning and actual parameter adoption
  remain UNKNOWN. Every full request is checked against the frozen local proxy
  guard; no history truncation is allowed.
- Accepted text is stored locally; automatic Memory backfill is disabled.

## Necessary Input Adaptation

This route is not simply the Memory route with its API switch turned off. Without
Memory event tracks, Planner identifies starting circumstances directly from the
full transcript in the same response. Both phases use the existing
`new_development_candidates` container with `source_basis`, `before_state` and
`source_refs`; its array name does not make Phase A events new. Phase A still
allows only existing-event progression and forbids new people. References are
checked for existence, not certified for semantic support.

Realizer receives full history, and Adherence, continuation and reaudit receive
the same history inside their existing calls. All stages preserve available date
annotations and explicit UNKNOWN values. No extra extraction or review API call
is introduced. This is a new input-mode experiment, not a controlled estimate of
the isolated effect of removing Memory.

## Entry

Workspace:
`/root/autodl-tmp/gxy-selfplay/worktrees/h4-two-pass-transcript-v1`

Run these from the workspace using the existing project Python environment:

```bash
python -m scripts.current_method.restart_n5_spell.h4_two_pass_autonomous_expansion_v1.transcript_run prepare
python -m scripts.current_method.restart_n5_spell.h4_two_pass_autonomous_expansion_v1.transcript_run mock
python -m scripts.current_method.restart_n5_spell.h4_two_pass_autonomous_expansion_v1.transcript_run advance --generation-http-budget 60
python -m scripts.current_method.restart_n5_spell.h4_two_pass_autonomous_expansion_v1.transcript_run report
```

`prepare` is single-use. `advance` stops at each content-review boundary. The
review must bind the exact generated turn SHA before the same entry can advance.
It cannot resend an existing phase or resume a technical stop automatically.

## Locations

- `prepared_run/manifest.json`: frozen inputs, code and tokenizer identities.
- `prepared_run/frozen/<root>/A_planner_payload.json`: exact initial requests.
- `prepared_run/phases/<root>/<A-or-B>/execution/`: all requests, raw responses,
  metadata and durable HTTP ledger, including failed and replaced generations.
- `prepared_run/phases/<root>/<A-or-B>/artifacts/`: shared workflow artifacts.
- `prepared_run/reviews/<root>/`: Codex offline content diagnoses, not independent
  human certification.
- `prepared_run/accepted/<root>/dialogue.md`: complete locally accepted text.
- `prepared_run/REPORT.json`: accumulated status, calls and token usage.

The old experiment is preserved at
`/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_two_pass_autonomous_expansion_v1`.
It used two unsuccessful Memory reads and zero generation calls. The active
Memorax credential configuration is `/root/.config/memorax/credentials.env`;
this fallback does not load it.
