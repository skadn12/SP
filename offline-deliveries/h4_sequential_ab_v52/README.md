# v52 A2/A3 diagnostic readiness

This directory fixes A planner aliases and mechanical copy-repair prompts.
It has not made any model, network, question-generation or Solver calls.
v51 and its QA artifacts remain frozen.

Use the bounded entrypoint, with this directory as the working directory:

    /root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python retry_a_diagnostic.py prepare --run /root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_sequential_ab_v52/runs/root210046_a2a3_01
    /root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python retry_a_diagnostic.py verify --run /root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_sequential_ab_v52/runs/root210046_a2a3_01
    /root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python retry_a_diagnostic.py live --run /root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_sequential_ab_v52/runs/root210046_a2a3_01
    /root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python retry_a_diagnostic.py audit --run /root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_sequential_ab_v52/runs/root210046_a2a3_01

Only live invokes the provider. It uses Qwen/Qwen3.6-35B-A3B, the shared live
lock and durable 35-second completion spacing. Budget is at most 8 logical
calls / 16 HTTP attempts: two slots, each one planner, at most two mechanical
realizations and one reviewer. HTTP503/504 have one same-payload retry only.
There are no B/card/source-reservation calls.

Input is exactly 110 original+R01 turns plus 13 retained v51 A1 turns. A2/S02
and A3/S03 run sequentially; retained A2 reaches A3 in full. Other v51 A attempts
and all B scenes are excluded. No new result is merged into v51 or its QA.
Report counts isolate new A output, inherited A1, and cumulative body tokens.
Semantic review/admission policy and duplicate thresholds are unchanged.

Offline verification:

    /root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python -m unittest test_candidate_v52 test_candidate_v51 test_candidate_contract test_candidate_metrics_v41 test_candidate_transport_v41 test_grounded_task_v43 test_generation_regression_v43 -v

Actual saved v51 fixtures cover three recovered A plans, four exact A1 copies
that still reject, precise 1–13 / 111–123 repair locations, one rejected-response
serialization, full history and frozen feedback. Additional tests cover
wrong-root, wrong-position, speaker conflict, ambiguous aliases, legacy literal
compatibility, unchanged B anchor handling, fake two-slot repair/audit flow,
provenance tampering, source immutability and three-root fresh collection.

A usable plan and a more specific repair prompt do not prove future live
generation will stop copying or advance the story. The bounded experiment
measures that next; no success claim is made from the offline fixtures.
