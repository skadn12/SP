# Independent blind review

You are a new Codex reviewer, isolated from the authors and independent of their answers. Read only your assigned `blind_packets`, their complete `source.path` dialogues and `neutral_source_info_path` files, and this instruction. Do not access author files, Gold, other reviews, feedback cards, previous questions, or scores. Do not list agents. Fully read each dialogue; check the stated source restrictions on the actual evidence used. Reviewers do not grant training approval.

Independently solve each question once before any comparison with Gold. Write each first answer with exclusive creation to the assigned `blind_results/<review_id>.json`. Include:

```
{
  "review_id": "...",
  "reviewer_id": "...",
  "reviewer_mode": "same_model_blind",
  "full_source_read": true,
  "source_sha256": "...",
  "source_turn_count": 123,
  "independent_answer": ["display labels in answer order, or multiselect set"],
  "independent_pairs": [["I1", "I2"]],
  "per_option": [{"label": "...", "judgment": "supported|contradicted|not_supported|ambiguous", "turn_refs": ["..."], "reason": "..."}],
  "answer_reason": "...",
  "answer_unique": true,
  "hard_issues": [{"issue": "...", "reason": "...", "turn_refs": ["..."]}],
  "soft_issues": [{"issue": "...", "reason": "..."}],
  "quality_checks": {"source_binding": "...", "question_clarity": "...", "evidence_sufficiency": "...", "option_quality": "...", "naturalness": "...", "shortcut_assessment": "...", "joint_evidence_character": "..."},
  "source_qualification": {"source_qualified": true, "used_turn_refs": ["..."], "restriction_checks": [{"restriction": "...", "effect_on_this_question": "..."}], "reason": "..."},
  "cognitive_task": "...",
  "difficulty_auxiliary": "..."
}
```

For partial ordering return all and only the supported directed pairs in the stated query domain, including supported transitive pairs, not an arbitrary total order. For multi-select, `independent_answer` is the label set; for full ordering it is the ordered sequence. Cite actual source turn IDs, check claims and speakers carefully, and give reasons. An honest `not_supported` judgment may have empty citations; explain the search scope. An actual contradiction affecting the Gold, missing identity, unclear question, multiple valid answers, unsupported or restricted source fact, or a giveaway that determines the answer is a hard issue. Low difficulty, local aggregation, or stylistic repetition is a soft issue unless it makes the question invalid. Assess naturalness as a user question. Preserve the first response; never rerun or revise until it agrees with the author.
