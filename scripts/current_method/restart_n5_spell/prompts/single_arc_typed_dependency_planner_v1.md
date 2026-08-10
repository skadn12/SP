# Single-arc typed dependency planner v1

Decide whether the assigned outline arc can realize the provided typed future
child event. Use only the listed parent anchor and exact typed key. Do not infer
theme similarity or invent a different relation. If it is valid, return the
typed child event. Otherwise return `{"status":"no_valid_dependency"}`.

The local program owns the arc ID, dependency ID, parent text, evidence,
parent-only/child-only missing items, and canonical joint result. Do not output
questions, answers, explanations, causal prose, distractors, or dialogue.
