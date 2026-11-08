# Base QA Generation V2 (offline Codex authoring)

The model work is performed directly by Codex, not by a school API executor. The module is an offline artifact boundary and a bounded AND/OR proof-path calculator. It deliberately imports V1 arithmetic, quotation, option mapping and scoring-boundary checks without changing those files. It does not import inference engines, load Qwen, call HTTP, train, or touch expansion/seed routes or budget ledgers.

## Stage order

```bash
python scripts/current_method/base_qa_pipeline_v2.py prepare --source FULL_SOURCE.json --run-dir NEW_STATE
python scripts/current_method/base_qa_pipeline_v2.py freeze-discovery --run-dir NEW_STATE --input DISCOVERY_INPUT.json
python scripts/current_method/base_qa_pipeline_v2.py freeze-target --run-dir NEW_STATE --input TARGET_INPUT.json
python scripts/current_method/base_qa_pipeline_v2.py freeze-candidate --run-dir NEW_STATE --input CANDIDATE_INPUT.json
# A fresh isolated same-model reviewer sees only BLIND_PACKET.json.
python scripts/current_method/base_qa_pipeline_v2.py freeze-blind --run-dir NEW_STATE --input BLIND_INPUT.json
# Only now may that reviewer see COMPARISON_PACKET.json.
python scripts/current_method/base_qa_pipeline_v2.py finalize --run-dir NEW_STATE --input COMPARISON_INPUT.json
```

Artifact hashes use canonical UTF-8 JSON with sorted keys and compact separators (`v1.legacy.fingerprint`), whereas source bindings hash file bytes. Model output is saved outside the state directory; all state files are exclusive-create. Review uncertainty is frozen too. Implementation and prompt drift requires a new versioned run, never overwriting old frozen files. `NO_SAFE_RELATION` freezes a terminal decision and creates no author packet.

## Required envelope fields

The operator supplies trusted packet/scope hashes and the external isolation receipt. The five .txt files specify semantic payloads; the stage envelopes also require:

- TARGET: `discovery_sha256` and `source_scope_sha256 = fingerprint(INPUT.source_scope)`.
- CANDIDATE: `relation_sha256` and `target_sha256` both equal `fingerprint(TARGET)`, plus the same `source_scope_sha256`.
- BLIND_SOLUTION: `packet_sha256 = fingerprint(BLIND_PACKET)`, same `source_scope_sha256`, `reviewer_mode: same_model_blind`, and `isolation_receipt: {context_mode: fresh_context_no_history, author_material_visible: false, context_id: <actual external context identifier>}`. The receipt is not itself proof of filesystem isolation; the orchestrator must actually use a new context without author material. Exact deployed model ID must not be invented when unavailable.
- COMPARISON: `blind_sha256 = fingerprint(BLIND_SOLUTION)`, `candidate_sha256 = fingerprint(CANDIDATE)`, and the same reviewer identity. A comparison packet binds the actual blinded display order and public snapshot to prevent confusing canonical storage order with what the reviewer saw.

`identity_requirement` is `same_event` (facts share one explicit event ID; confirmed explicit-reference identity links required) or `explicit_cross_event_relation` (e.g. a deadline constrains a separately identified train offer). Event identity is not inferred from matching actor/activity/time. First mention is an introduction. Across scenes, a claimed reference must be explicit and cited; this is a Codex extraction hypothesis that Reviewer must independently inspect. Code checks the declaration and exact quotation, not the semantic antecedent.

## Evidence mechanism and limits

Each fact has alternative source `supports`: each support is an AND set; supports are OR alternatives for that proposition. An inference step is AND over inputs, optionally with OR `input_alternatives`. Exact minimal turn-support antichains are computed across the full declared graph, including alternatives; a path/Cartesian-product bound yields HOLD, never silent truncation. Primary fact evidence must correspond to a declared support. Full-source scans account for every turn without duplication and expose candidate alternatives. Undecided identity paths remain HOLD rather than being treated as confirmed bypasses.

Every primary turn is deleted individually against all graph paths. Semantic premise groups are checked separately, including repeated expressions: `{A,B}` and `{A2,B}` make A individually unnecessary but may leave group `{A,A2}` necessary. No group may be sufficient alone. The certificate must disclose all computed minimal support sets. The mechanism can verify a proposed extraction; it does not automatically discover all natural-language paraphrases, infer entailment, certify naturalness, or prove the source scan complete. Independent full-source review remains necessary. A connected graph is not proof that a question is meaningful.

Source scope preserves safe provenance, file version, known lineage hashes, all inline warnings and explicit per-turn material admission fields. Missing/unknown/empty provenance is never eligible by default. `formal_qa_allowed` requires a traceable explicit PASS declaration; code does not authenticate the external reviewer or review-record content. Admission closure includes proof paths, identity/exclusion evidence, candidate analyses/error paths, ordering edges and independent blind evidence. Unknown material may support a clearly labelled research candidate but cannot become a formal QA PASS. Provenance warnings are not factual premises; no Solver or ablation views are produced in this round.

## Validation

```bash
python -m unittest scripts.current_method.test_base_qa_pipeline_v1 scripts.current_method.test_base_qa_pipeline_v2 scripts.current_method.test_base_qa_v2_historical scripts.current_method.test_base_qa_v2_first_hold -v
```

Controlled fixtures are software regressions, not generated dataset success or semantic quality certification. Historical content-positive records retain their review tiers, hashes and original Gold; schema-only records and the first root's superseded static PASS are excluded as quality positives. The first HOLD stays a negative case. Any actual candidate is generated once under a fixed small budget; no regeneration after quality failure, no Solver invocation, and no claim of independent-model review.
