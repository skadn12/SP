# A candidate records

This directory normalizes the 10 existing A questions into the exact `schemas.candidate_record` schema in `design/PUBLIC_DESIGN_LIBRARY.json`. It does not author questions or replace the existing independent semantic reviews.

- `A-week1-*.json`: ten actual question records, each with existing independent content/target/shortcut QA PASS, disposition HOLD and duplicate_review UNKNOWN.
- `SLOT_STATUS.json`: all 12 inherited slots; the two MC_NEGATION_CONDITIONAL_JOINT shortfalls have no question record or Gold.
- `MECHANICAL_PROVENANCE.json`: original/final public, private, plan, input packet and independent-review hashes; all independent reasonable mentions and 90 ordering-pair comparisons; raw trace, budget, language and provisional family limitations.
- `SCHEMA_VALIDATION.json`: actual Draft 2020-12 validation with a format checker using `.venv-local4b/bin/python`.
- `MECHANICAL_INTEGRITY.json`: exact source spans, internal references, actual artifact hashes, original quotas, budget counters, all source-order positions and all 90 independent pair comparisons.

103866 retains its original English questions with no clarification. 104814 binds the separate English public files from one independently requested translation; original Chinese public and private artifacts remain unchanged. This was a post-initial correction, not a preregistered common language or equal realized compute.

Every INITIAL generation-history path points to the original public artifact; every clarification path points to the clarified public artifact. Each hashes to its candidate-content hash. Private author artifacts remain separately bound. These interactive files are not raw provider API response envelopes; provider request IDs, actual usage and cap evidence remain unavailable.

Rebuild and validate from the repository root:

```sh
.venv-local4b/bin/python research-protocols/selfplay_week_20261019/day3_production_v1/records/A/build_records.py
.venv-local4b/bin/python research-protocols/selfplay_week_20261019/day3_production_v1/records/A/validate_records.py
```

The scripts write only this directory. Cross-pool family adjudication and final acceptance remain the main agent's responsibility. The per-question family reuse enum is provisional; it is not a novelty or deduplication conclusion.
