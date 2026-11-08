# Initial Planner: target eight, hard maximum nine

The cap-nine route accepts a ninth new-development candidate without discarding or rewriting it. The prompt asks for eight, permits fewer, and allows a distinct useful ninth candidate. It never requests nine as a quota. Ten still fails the entire initial response.

Opt in through `candidate_policy_version=initial_candidates_target8_cap9_v1`, `append_enabled=false`, template `planner_minimal_relations_filtered_feedback_target8_cap9_v1`, and relation contract `shared_planner_relations_v2_cap9`. Existing configurations and append paths retain their cap of eight.

The explicit limit passes through schema validation, relation compilation, speaker materialization and the existing plan compiler. The maximum is three continuations plus nine new developments. Gate can review all twelve candidates; the prepared request budget is one Planner plus twelve Gate requests. Final scene quota remains six. Gate approval alone does not materialize a scene or establish a prerequisite in dialogue.

`prepare_cap9.py` copies a previously verified independent input package, updates only local manifest paths and the request budget, and generates a new prompt through the current renderer. History, Memory candidates/tracks and filtered Planner feedback are checked for exact equality with the source. Failed responses never enter the prepared prompt.

`planner_gate_trial.py` selects the cap-nine run version from the sealed input bundle. Before transport it checks source file hashes, template and contract versions, and exact equality with the current renderer output. The entrypoint retains one Planner request and one Gate request per compiled candidate, with no retries, alternatives, append, Realizer, QA, Solver, training or Memory writes.

Offline regression replay of a historical failed response is diagnostic only. It does not relabel the old run as successful, consume another model request or schedule Gate. Cap-nine acceptance of the count does not waive weak transitions, identity errors or broken relations. Gate decisions and final selection quality still require a separately authorized live run.
