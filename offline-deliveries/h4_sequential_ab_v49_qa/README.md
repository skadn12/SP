# v49 bulk collection over one complete cumulative A+B history

Authorized task: generate 20 candidate slots (10 multi-select, 10 ordering) on
the existing complete 156-turn / 4566-body-token history. This is one whole bulk
collection task, implemented as five consecutive chunks of 2+2 for bounded JSON
responses. It is not another four-slot experiment. Complete all five chunks,
then evaluate all eligible questions and publish one consolidated provisional
candidate pool. Do not replace duplicate, failed or abstained slots automatically.

Keep v48 generation prompt text, evidence-first schema presentation, reviewer,
Gold derivation, exact whole six-statement-set dedup and scoring unchanged.
All calls receive the full cumulative source and frozen previous-training-round
feedback. Fixed historical prior consists of the default formal/historical
inventory plus selected public question/statement texts from v44, v47 and v48.
Later chunks additionally receive prior chunks' mechanically accepted public
question/statement texts for exact dedup. No relations, evidence, Gold, current
Solver scores/answers, or post-run diagnostics enter generation inventory.
All generation finishes before any Solver evaluation starts.

Overall maximum: 20 slots, 30 Qwen HTTP attempts, 300 full Solver answers.
Each chunk retains 2 initial generator requests, at most one mechanical repair
per type and at most two independent ordering reviews. Shared durable response
completion gap >=35s and no automatic transport retry. Only
Qwen/Qwen3.6-35B-A3B for writer/reviewer; fixed audited M_clean_policy for Solver.
MC 6 permutations x3 full answers, ordering 12 permutations x1 full answer.
Selection: complete_valid_full_set and 0 < full_partial_mean < 1.

No source extension, source-scene promotion, training, altered Gold, or changed
semantic gate in this task. Keep all previously frozen files unchanged, including
the formal30 pool and eight earlier selected QA candidates. Candidate selection
and its documented semantic limitations remain distinguishable. BASELINE_FILES
binds 2732 historical artifacts. New reports and readable bulk results are saved
under results/root104814_bulk20_01. Freeze implementation/tests before prepare.
