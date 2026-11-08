"""Locate exact copied output without repeating the rejected response twice."""
from copy import deepcopy

POLICY = "v52_located_copy_repair_single_rejected_response"


def build(raw, failures, duplicate):
    duplicate = duplicate or {}
    spans = deepcopy(duplicate.get("consecutive_spans", []))
    matches = [{key: deepcopy(row[key]) for key in
                ("new_position", "new_turn_ref", "speaker", "sources")}
               for row in duplicate.get("matches", [])]
    instruction = (
        "Correct the listed mechanical failures for the current target. The previous_response below "
        "is rejected output, not history or a template to reproduce. Copied output positions are "
        "1-based and their source refs locate the actual old turns in the complete history. "
        "Replace copied spans with a genuinely new interaction: choose a concrete next action, "
        "new information, decision, or outcome compatible with the complete current history "
        "and supplied source card. Do not merely rename the scene, rephrase old lines, replay "
        "a completed milestone, or repeat old farewells. Preserve established roles, facts, "
        "commitments and action status. A draft plan cannot establish historical facts; use "
        "it only where faithful. Return only the corrected NEW scene, or no_scene if a "
        "coherent continuation cannot be written."
    )
    return {"policy": POLICY, "instruction": instruction, "failures": deepcopy(failures),
            "duplicate_reasons": deepcopy(duplicate.get("reasons", [])),
            "copied_spans": spans, "copied_output_positions": matches,
            "previous_response": raw}


def context_reference(repair, digest):
    if repair is None:
        return None
    return {"policy": POLICY, "repair_sha256": digest(repair),
            "location": "final TARGET_AND_ORIGINAL_CARD_JSON.mechanical_repair",
            "previous_response_occurrences": 1}
