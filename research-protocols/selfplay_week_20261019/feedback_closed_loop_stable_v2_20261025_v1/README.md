# Stable closed loop v2

Run from this directory with `python -B pipeline.py`. This is the sole supported
entry point. It acquires a process owner lock and uses three process-shared
semantic-call slots. It stops on a failed pilot and preserves all failures;
running it again does not renew semantic or repair budgets.

`event_ledger.py` reconstructs the original source drafts and strictly classifies
their CLI evidence. `INITIAL_LEDGER_FREEZE.json` binds all 16 original ledgers,
including decoded character spans and verified UTF-8 JSON token spans.
Canonical facts are attributed assertions with negation and modality retained.
They do not certify that an utterance is true or that its native mapping is sound.

`source_engine.py` requires mechanical mapping and two independent accepted
reviews of the same final candidate. A repair returns one scene and its ledger;
assembly verifies other scenes stayed identical. Maximum two repairs per root.
`question_engine.py` freezes each slot plan, then uses separate fresh author,
blind Gold and content sessions. Each call handles one slot. A slot can receive
one targeted revision. Transport attempts are capped at two before any semantic
response; semantic failures are preserved.

`solver_engine.py` uses the fixed B1 checkpoint, FULL THINK_OFF, two deterministic
display permutations, the frozen output validator and scorer, and no submitted
request retries. Generation reads public prompts only. Missing presentations
remain missing. Source and question failures are never solver mistakes.

`PILOT_REPORT.json` is the gate receipt. The automatic full branch is reachable
only at 4/4 sources, 64/64 slots, 128/128 returned presentations and intact
engineering/integrity gates. Full production carries pilot outputs forward,
handles the remaining fixed roots, and reads the sealed four scores without
rerunning them. No training occurs.

Tests: `python -B -m unittest -v test_stability`. Raw calls live in `cli_calls/`;
every call has its prompt, schema, event stream, output, backend metadata and
receipt. `CALL_AUDIT.json` additionally records interrupted duplicate-worker calls.

The `scripts/` and `tests/` directories and `source_pilot.log` were written by a
second pre-existing worker. They are retained as incident evidence and are not
supported launchers. See `CONCURRENCY_INCIDENT.json`. The root-level modules are
the implementation used by `pipeline.py`.

Runtime files and caches stay below `runtime/`. Its Codex authentication files
are private and must not be published. All predecessor data is read-only.
