# Relation Candidate Probe v2

This is a completely independent, offline probe that mines exactly one
relation type from a frozen real Parent + Child transcript:

```text
proposal -> later outcome
```

It does not read the previous frozen QA pool as its relation source. It does
not modify `qa_composition_probe_v1`, QA Writer, promotion, grounding, Solver,
Memorax, Parent/Child data, or historical runs.

## Input contract

The primary input is the canonical `expanded_dialogue.json` containing
`root_id` and ordered turns with `turn_ref`, `speaker`, `text`, and `origin`.
Evidence in all outputs is copied only from these transcript turns. Optional
planner/materialization/promotion JSON files can be supplied as structured
annotations; they are recorded as `source_structured_refs` and are never used
as transcript evidence.

## Run

Use a new empty run root:

```bash
python -m scripts.current_method.restart_n5_spell.relation_candidate_probe_v2.probe \
  --transcript /path/to/frozen/expanded_dialogue.json \
  --structured-source /path/to/planner/materialized_plan.json \
  --structured-source /path/to/planner/output.json \
  --structured-source /path/to/promotion/promotable_child.json \
  --run-root /path/to/new-relation-probe-run
```

The runner refuses to overwrite a non-empty run root. It writes:

```text
normalized_turns.jsonl
proposal_facts.jsonl
outcome_facts.jsonl
relation_candidates.jsonl
relation_rejections.jsonl
paired_qa_targets.jsonl
report.md
manifest.json
```

The miner is rule-based. It requires a specific closed event slot such as
`event_type:charity_event`, `event_format:music_festival`,
`collaborator:elena`, or `planning_tool:shared_google_doc`; shared people,
scene, `charity`, `environment`, or `event` themes alone are rejected.

`paired_qa_targets.jsonl` contains only eligible relations. It stores the two
atomic target roles, exact required transcript refs, target answer values, and
a deterministic composite intent. It does not generate natural-language
questions.

