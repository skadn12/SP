# Relation-aware Composer contract (isolated probe)

This schema is local to `h4_relation_aware_composition_probe_v1`; it does not
change the production QA pipeline.

The raw Composer response now contains only the question/option layer:

```json
{
  "question": "...",
  "answer_dimension": "...",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "correct_option": "A",
  "distractor_mutations": {"B": "Q06", "C": "Q13", "D": "Q06"},
  "is_mechanical_conjunction": false
}
```

The program then deterministically assembles the envelope consumed by the
reused static audit:

```json
{
  "candidate_id": "C02",
  "required_qa_ids": ["Q06", "Q13"],
  "target": "...",
  "question": "...",
  "answer_dimension": "...",
  "gold_text": "exactly options[correct_option]",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "correct_option": "A",
  "source_contributions": ["copied unchanged from candidate"],
  "required_evidence_groups": [
    {"source_qa_id": "Q06", "turn_refs": ["unique refs for Q06"]},
    {"source_qa_id": "Q13", "turn_refs": ["unique refs for Q13"]}
  ],
  "required_evidence_turn_refs": ["exact union of the two groups"],
  "distractor_mutations": {"B": "Q06", "C": "Q13", "D": "Q06"},
  "is_mechanical_conjunction": false
}
```

`required_evidence_groups` is deliberately the unique contribution closure
used for source ablation. Full grounding closure remains in the frozen
candidate/source artifacts and is not silently substituted into this field.
The raw response is always preserved separately from the assembled response.
