# v45 bounded QA task-tail probe

Authorized next step: move the v44 generic task guidance after complete input and
frozen previous-training-round feedback, then generate 2 multi-select and 2
ordering questions on the unchanged v43 candidate history (156 turns / 4566 body
tokens). No source-specific answers, revised Gold or current Solver feedback.

All prior source/semantic/ordering/score contracts remain unchanged. Requests
stay a single user message. The old feedback receipt binds the intermediate
input+feedback; a new receipt binds the final task tail and final payload. Local
ordering proof parsing accepts exactly this layout. Full source is never truncated.

Budget: 4 slots, at most 6 Qwen HTTP attempts, at most 60 full M_clean_policy
answers. Exact existing checkpoint via .venv-local4b. Only Qwen/Qwen3.6-35B-A3B
for generation/review; shared durable completion gap >=35 seconds, no automatic
transport retry, no ground calls or training. Prior selected question text may
enter exact-set dedup inventory; prior Solver answers/scores and Gold cannot.

Freeze implementation and tests before prepare. BASELINE_FILES.json binds old
v43/v43QA/v44QA files. Provisional QA selections remain separate from the old
formal 30-question pool and source-scene admission. Report true measured yield
and original content errors without modifying outputs to improve scores.
