# Day 2 pilot direct runtime

These new scripts reuse day2/solver/adapter.py, the original native prompt,
v15 deterministic rebinding, finite-choice decoder, S0 M-clean weights and
Qwen3-4B tokenizer. Existing implementation and evidence probe files are unchanged.

Run only with the repo's .venv-local4b/bin/python -B. All commands below assume
the repository root as working directory. Replace placeholder output paths deliberately;
all writes are exclusive and automatic retries/overwrites are disabled.

## Isolation and inputs

- Public JSON: an object containing questions, a list of admitted public question objects.
  Each object has candidate_id, root_id, question_type, question, source_path,
  source_sha256 and exactly one of options (A through F) or items (I1 through I6).
  No design, arm, Gold or evidence fields are admitted into these public questions.
- Private JSON: questions with identical public fields, plus gold (canonical labels)
  and any private evidence/metadata. Preparation copies bytes without parsing.
- EXECUTION_FREEZE.json supplies the 24 planned slots, solver fields, and seed_base.
- CONTENT_FREEZE.json supplies status PASS, namespace beginning train.feedback,
  questions containing candidate_id, public_questions/private_questions bindings,
  planned_slots=24, admitted_count, and optional shortfalls.
- Missing or rejected slots remain in production_denominator=24 and are not inferred.
  Only admitted questions get 18 MC calls or 12 ordering calls, capped at 360.
- DISPLAY_MAPPINGS.json contains public display/canonical content mappings only.
  The private sidecar is disjoint from the public run and never read by live.

## CPU checks

    .venv-local4b/bin/python -B research-protocols/selfplay_week_20261019/day2_pilot_v1/runtime/selftest.py --output /new/path/CPU_SELFTEST.json

The synthetic selftest creates no model backend and reads no pilot questions or Gold.

## Preparation (CPU, explicitly authorized separately)

    .venv-local4b/bin/python -B research-protocols/selfplay_week_20261019/day2_pilot_v1/runtime/prepare.py --public research-protocols/selfplay_week_20261019/day2_pilot_v1/public/PUBLIC_QUESTIONS.json --private research-protocols/selfplay_week_20261019/day2_pilot_v1/private/PRIVATE_QUESTIONS.json --protocol research-protocols/selfplay_week_20261019/day2_pilot_v1/EXECUTION_FREEZE.json --admission research-protocols/selfplay_week_20261019/day2_pilot_v1/CONTENT_FREEZE.json --run research-protocols/selfplay_week_20261019/day2_pilot_v1/runtime/direct

Default Gold copy goes to runtime/direct_private, with directory 0700 and files 0600.
The public manifest and public preflight contain no Gold path or Gold binding.

An independent preflight must produce a minimal public gate:

    {"status":"PASS","audited_files":[{"path":"/absolute/public/run/PREFLIGHT_FREEZE.json","sha256":"...","bytes":123}]}

All gate binding rows must point inside the public run. Gold/private bindings belong
in a separate private audit report. Code/model/tokenizer bindings are independently
verified from the public manifest, so the gate need not repeat those external paths.

## Verify and execute

    .venv-local4b/bin/python -B research-protocols/selfplay_week_20261019/day2_pilot_v1/runtime/execute.py verify --run research-protocols/selfplay_week_20261019/day2_pilot_v1/runtime/direct --audit /path/PUBLIC_PREFLIGHT_PASS.json
    .venv-local4b/bin/python -B research-protocols/selfplay_week_20261019/day2_pilot_v1/runtime/execute.py live --run research-protocols/selfplay_week_20261019/day2_pilot_v1/runtime/direct --audit /path/PUBLIC_PREFLIGHT_PASS.json

Live preserves every RETURNED/BACKEND_ERROR/NOT_SUBMITTED row, raw output, token IDs,
metadata, usage, timing, submit batch and failure. Invalid returned outputs are kept.
It never retries or truncates source. RAW_RESPONSE_FREEZE.json requires one terminal
record for each planned admitted-question request, including all failures.

## Post-freeze score

    .venv-local4b/bin/python -B research-protocols/selfplay_week_20261019/day2_pilot_v1/runtime/score.py --run research-protocols/selfplay_week_20261019/day2_pilot_v1/runtime/direct --private research-protocols/selfplay_week_20261019/day2_pilot_v1/runtime/direct_private

Score first verifies every terminal response and its freeze, then reads Gold JSON.
SCORES.json includes each question, each MC item's TP/FP/FN/TN, each ordering pair,
canonical mapping, all invalids and request failures, fixed denominators, usage/time,
and the separate 24-slot production accounting. Invalid MC predictions have unknown
item FP/FN=null rather than invented zero; invalid ordering gets 0/15 pair credit.
