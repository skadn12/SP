# Human-defined difficulty objective v1

This addendum implements the user's objective: produce harder, answerable QA
using observed Solver behaviour. It retains Feedback Policy v2's three slots,
fixed strategy enums, actor ownership and transcript-only factual authority.

- Solver accuracies are observations on particular questions, not diagnoses of
  a model's general abilities. Do not invent weaknesses, curricula or strategies.
- For SUCCESSOR_FRONTIER, inspect the current feedback seed and its measured
  full-context and no-context behaviour. Replace an already solved surface target
  with a new relationship requiring multiple independently stated facts. A new
  question must require R03 evidence and differ from the seed's answer target.
- High no-context accuracy is evidence that the observed question can be solved
  without this transcript. Avoid its wording cues and implausible distractors.
  Do not describe this as a proven cause or a general model limitation.
- Use two or more independently supported facts, state updates, actor bindings,
  temporal constraints or cross-stage dependencies when the transcript supports
  them. Do not count several citations to the same fact as several reasoning steps.
- For multi-select, require the exact set of supported propositions. Distractors
  may use a documented earlier state, another participant's action, an unmet
  dependency or an intention that was never completed. Keep each option decidable.
  Require evidence for each option's inclusion or exclusion, with the right actor
  and time. Unrelated personal-life distractors in a work question are not an
  adequate difficulty increase. Two paraphrases of one fact are not two facts.
- For ordering, require four distinct events with explicit chronological evidence.
  Listing events in transcript order is not sufficient when event time differs.
  If chronology is underdetermined, choose NO_SAFE_EVOLUTION, not an invented order.
- Fewer valid questions, including none, are allowed. A requested multi-select or
  ordering batch must never be filled with single-choice questions.
- Do not change Gold or invent facts to make the Solver fail. Difficulty is not
  established until the same Solver protocol has measured the new candidates.

## Operational settings

The next focused batch requests two multi-select and two ordering candidates per
root, in separate Writer calls. These are generation targets, not guaranteed
accepted counts. No single-choice fallback or further quota-filling call is made.
Model, temperature, token limits and fixed intent strategy schema stay unchanged.

## Evidence packaging and evaluation

Generators should output flat evidence equal to the union of the candidate's
actual supporting groups. Do not copy irrelevant intent groups for padding.
For already generated QA, group/flat differences remain visible diagnostics.
An evaluation export may include a question only when the independent audit found
its Gold correct, its options determinate and clean, and its submitted flat evidence
sufficient. Other blockers remain blocking. No refs are added and Gold is unchanged.
Evaluation eligibility does not grant training admission.
An explicit subsequent evidence review overrides an earlier automated acceptance.
Quarantined evidence/Gold/premise issues are QA quality feedback, not Solver ability
observations. Preserve raw evaluations separately and exclude these questions from
difficulty claims and the eligible feedback seeds. Do not silently repair Gold.

## Measuring progress

Report full-context and no-context accuracy by root and question type, plus the
number of independent questions. Report position/rollout sensitivity separately.
Do not treat repeated permutations as independent questions or interpret a changed
type mix as proof of feedback effectiveness. Keep parent/successor source IDs so
later feedback can be traced to the actual model observations that informed it.

Hard-candidate classification retains the first-round integer-count rules in
`h4_round0_b0_k3_difficulty_v1/difficulty.py`: nonzero but imperfect full-context
accuracy, imperfect no-context accuracy, and full-context accuracy no worse than
no-context. Apply these only after semantic support, a new answer target, actual
fact combination and intent alignment have passed review. Require at least one
current trained Solver to remain in that bucket; base-only errors do not establish
the new training frontier. Full-zero, context-worse and technical cases require
review. A hard candidate alone does not prove a causal feedback improvement.
