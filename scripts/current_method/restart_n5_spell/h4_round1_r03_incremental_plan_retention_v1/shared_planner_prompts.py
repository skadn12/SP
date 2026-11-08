"""Append-only task instructions around the shared Planner output contract."""
import hashlib
import json

from ..h4_round1_r03_feedback_view_v2.templates import (
    SHARED_FEEDBACK_RULES_V3, PLANNER_DEVELOPMENT_GUIDANCE_V1,
)
from .input_views import project_supplement_inputs
from .planning import sha
from .prompts import TARGET_RULES, package
from .shared_planner_contract import (
    VERSION, RENDER_VERSION, ROOT_SHARED_PROMPT_VERSION, output_schema, render_output_contract,
)


APPEND_TASK = """ROOT-SCOPED ONE-TIME INCREMENTAL DEVELOPMENT
Execution mode: append_only. Propose only additions to the supplied frozen plan.
Use the complete authoritative history, frozen Memory navigation and shared
feedback. Memory summaries and original plans are navigation, not observed facts.
You may extend any eligible original event or introduce compatible independent
events, people, places and side stories. Diagnostic candidate IDs are observations,
not a permission list or a prescribed storyline. Budgets are ceilings, not quotas.
This is the sole pre-Gate supplement opportunity for this root and round.

Return the SAME shared Planner output object used by initial planning. Copy
round_goal exactly from original_plan.round_goal. Both output arrays contain only
new additions, never copies or replacements of the original arrays. If no suitable
addition exists, return both arrays empty and the unchanged round_goal.

For selected_continuations, choose a source_candidate_ref from the supplied
continuation_source_templates. Each template is a complete source declaration.
Its required_transition and interaction_beats describe the original planned phase;
they are reference context, not the new action to output. Its empty dependencies
is not a valid continuation dependency list. Derive the added phase and actual
prerequisites from the selected source and the supplied history.
Copy its verified_track_ref, goal, after_state, speaker_refs, actor_bindings,
new_people and transition_from_previous exactly. Change only required_transition,
interaction_beats and dependencies to describe the added phase. Never introduce
or reassign a person in a continuation. Existing new-development sources can
retain their already declared people; their verified_track_ref is null.
Multiple added phases may use the same source. An eligible source need not have
been selected for execution; eligibility does not prove that it happened.

For independent new_development_candidates, source_candidate_ref must be null.
Use one to four concrete source_basis facts grounded in the existing world.
At least one canonical Parent speaker must participate. speaker_refs and
actor_ref use only the supplied canonical SP refs. New people are declared only
in new_people using their exact name and relationship_basis, at most two per
candidate; never invent SP refs. New goals and states remain planned targets.
There may be at most eight new-development candidates, including zero.

DEPENDENCIES
Process selected_continuations first, then new_development_candidates, in array
order. Each continuation requires at least one concrete prerequisite. Independent
new developments may use an empty dependencies array. Each dependency references
either an original source_candidate_ref or an earlier output_ref, with the other
field null. output_ref is the exact zero-based JSON pointer of the earlier row,
for example /selected_continuations/0. No forward, self or unknown references.
State the concrete condition that actual retained dialogue must establish.
Dependency existence or source retention alone does not prove that condition.
The program assigns execution phase IDs and freezes hashes; do not output them.

Use three to seven third-person action-summary interaction_beats per addition,
never dialogue scripts. Do not replay old events, invent past actions, or force
completion. Goals can remain partial or not_realized. Only actually retained
dialogue becomes later history; unexecuted plans are not mandatory next-round tasks.
"""


def render_shared_supplement(inputs, *, continuation_source_templates):
    projected, mapping = project_supplement_inputs(
        inputs, extra_reference_fields=("memory_navigation",))
    template = (SHARED_FEEDBACK_RULES_V3 + "\n" + TARGET_RULES + "\n"
                + APPEND_TASK + "\n" + PLANNER_DEVELOPMENT_GUIDANCE_V1
                + "\n" + render_output_contract())
    built = package("supplement.shared_planner_v2", template
                    + "\nCONTINUATION SOURCE TEMPLATES\n"
                    + json.dumps(continuation_source_templates, ensure_ascii=False, indent=2),
                    projected)
    built["prompt_version"] = ROOT_SHARED_PROMPT_VERSION
    built["input_projection_mapping"] = mapping
    identity = {"output_contract": VERSION, "schema_sha256": sha(output_schema()),
                "render_version": RENDER_VERSION,
                "execution_mode": "append_only", "strict_decoding_added": False}
    built["planner_contract_identity"] = identity
    built["supplement_contract_identity"] = {
        **identity, "prompt_version": ROOT_SHARED_PROMPT_VERSION,
        "version_source": "bundle.supplement_prompt_version",
        "source_scope": inputs["source_scope"],
        "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "allowed_source_view": continuation_source_templates,
        "allowed_source_view_sha256": sha(continuation_source_templates),
        "development_guidance_version": "planner_development_guidance_v1"}
    return built
