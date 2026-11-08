"""Full-history Gate output description using the existing validator fields."""
from .contracts import PARENT_RELATIONS, semantic_gate_review_fields

LEGACY_PROMPT_VERSION = "realization_aware_semantic_gate_prompt_v1"
FULL_PROMPT_VERSION = "realization_aware_semantic_gate_full_history_prompt_v2"
FULL_CONTRACT_VERSION = "semantic_gate_full_history_output_contract_v1"


def full_history_output_contract(context):
    if context.get("memory_only_fast") is not False:
        raise ValueError("full_history_gate_requires_non_fast_context")
    history = context.get("authoritative_complete_history")
    candidates = context.get("expected_candidate_refs")
    if not isinstance(history, list) or not history or not candidates:
        raise ValueError("full_history_gate_requires_history_and_candidates")
    turn_ref = str(history[0]["turn_id"])
    fields = semantic_gate_review_fields(strict_memory_contract=False)
    evidence = fields["evidence_key"]
    examples = []
    for relation in ("PLANNED_OR_INTENDED", "REALIZED"):
        reviews = []
        for candidate in candidates:
            review = {"candidate_ref": candidate, "parent_relation": relation,
                      "decision": "reject" if relation == "REALIZED" else "pass",
                      "duplicate_with_parent": relation == "REALIZED",
                      evidence: [turn_ref] if relation == "REALIZED" else []}
            assert fields["required"] <= review.keys() <= fields["allowed"]
            reviews.append(review)
        examples.append({"candidate_reviews": reviews})
    rules = f"""Do not repair, rewrite, improve, or suggest an alternative development.
The only output evidence field is {evidence}. Its values must be exact turn_id
values from this request's authoritative_complete_history, or corresponding
turn_ref values in authoritative_parent_evidence from that same root/snapshot.
Memory identifiers and prose descriptions are not transcript references.
For REALIZED/reject, cite at least one such turn proving the completed duplicate.
For every other relation/pass, an empty evidence list is allowed. Do not invent
past evidence for a genuinely new development. reason is optional in this mode;
omitting it preserves the existing locally supplied audit reason.
Use one review per candidate exactly once. Only REALIZED maps to reject/true;
all other allowed relations map to pass/false. Allowed relations are:
{', '.join(sorted(PARENT_RELATIONS))}.
Judge each candidate independently; do not select, rank, repair, or rewrite it.
The two examples illustrate ALTERNATIVE output shapes using input identities,
not semantic judgments. Their reference does not assert that a duplicate exists.
Return one object, not both examples, and do not copy their decisions blindly.
"""
    properties = {key: {"type": "string"} for key in sorted(fields["allowed"])}
    properties.update({"decision": {"enum": ["pass", "reject"]},
                       "parent_relation": {"enum": sorted(PARENT_RELATIONS)},
                       "duplicate_with_parent": {"type": "boolean"},
                       evidence: {"type": "array", "items": {"type": "string"}, "uniqueItems": True}})
    schema = {"type": "object", "required": ["candidate_reviews"], "additionalProperties": False,
              "properties": {"candidate_reviews": {"type": "array", "items": {
                  "type": "object", "required": sorted(fields["required"]),
                  "additionalProperties": False, "properties": properties}}}}
    return {"prompt_version": FULL_PROMPT_VERSION, "output_contract_version": FULL_CONTRACT_VERSION,
            "input_mode": "authoritative_full_history", "evidence_field": evidence,
            "validator_options": {"memory_only_fast": False,
                                  "minimum_pass_candidates": int(context["minimum_pass_candidates"])},
            "rules": rules, "examples": examples, "structural_schema": schema,
            "schema_scope": "field description only; existing validator enforces coverage, refs and decisions",
            "decoding_constraint": "PROMPT_JSON_ONLY_NO_RESPONSE_FORMAT_OR_SCHEMA_DECODING"}
