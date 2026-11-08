"""Fixed root-level task and an explicit existing/new-development output union."""
import hashlib
import json

from ..h4_round1_r03_feedback_view_v2.templates import SHARED_FEEDBACK_RULES_V3, PLANNER_DEVELOPMENT_GUIDANCE_V1
from .input_views import project_supplement_inputs, _resolve_at
from .prompts import TARGET_RULES, package
from .root_scope import ROOT_POLICY_ID, ROOT_SCOPE, ROOT_PROMPT_VERSION
from .planning import sha

ROOT_TASK = r"""ROOT-SCOPED ONE-TIME INCREMENTAL DEVELOPMENT v1
Use this root's complete authoritative history, frozen Memory navigation,
original plan and shared feedback to propose one valuable incremental development.
Memory summaries are navigation with their original provenance, not verified facts.
Original plans are reference material, not a checklist that must be completed.
You may extend any suitable event in the eligible original registry, or introduce
compatible new events, people, places and side stories. Diagnostic candidate IDs
are reference observations only: they do not restrict source permission or require
prioritizing a previously flagged event. Do not target a prescribed storyline.
Return only additions to the single existing plan, not another full plan or variants.
One incremental Planner request is allowed per root/round; budgets are ceilings,
never output quotas. NO_SUPPLEMENT is legitimate when no suitable addition exists.

Existing-event phases must name a true source_candidate_id from the ROOT SOURCE
NAVIGATION table. original_goal must copy exactly the parsed source scene.goal:
original_goal = stable_candidates[source_candidate_id].scene.goal
Resolve existing $input_ref pointers. Do not translate, summarize, trim or substitute
required_transition, after_state, another source's goal, or a phase-specific goal.
JSON escaping is fine only if the parsed string stays equal. Multiple phases may
use one source. Original candidates and goals stay immutable. Eligibility only
establishes scoped identity; Gate and adherence must still check content/binding.
The source need not have been flagged by prior diagnostics, nor selected initially.

New independent developments use new_developments, not an invented old source ID.
Each candidate uses the existing new-development speaker contract. speaker_refs
come only from speaker_registry; actor_bindings use actor_ref from those existing
SP refs and a nonempty binding. New people use new_people declarations with exact
name and relationship_basis; at most two per candidate. They do not get invented
SP refs. The existing materializer appends declared people to participants and
assigns NP refs locally. At least one existing Parent speaker must participate.
Use one to four concrete source_basis statements grounded in the existing world.
New goal/before_state/after_state are planned fields, not proof of observed history.
New places/events are described within the existing fields, not new output keys.
New candidates are at most eight, may be fewer or zero, and are not a quota.

PHASE ID AND DEPENDENCY CONTRACT
All new phase_id values fully match phase:[A-Za-z0-9_-]+ and must be unique across
both arrays. Legal examples: phase:followup_1 and phase:new-event_2.
phase:ND01:C01 is illegal: a second colon is forbidden after 'phase:'. No dots,
whitespace, normalization or implicit repair. Source IDs and new phase IDs differ.
Process phases first, then new_developments, preserving order within each array.
Existing-event phases require a nonempty dependencies list. New developments may
use [] if independent, otherwise the same dependency rule applies. Dependencies
contain exactly phase_id and a nonempty concrete condition. A dependency may name
initial:<candidate_id> from the original registry, or an earlier new phase in the
processing order. For example phase:new-event_2 may depend on:
{"phase_id":"phase:followup_1","condition":"The earlier phase established the required arrangement."}
This illustrates syntax, not a task condition to copy. No unknown, forward or self
references. A dependency's existence is not proof it holds. Mere source selection
or retained content does not prove a goal/after_state was realized. Only actual
retained dialogue can establish prerequisites for downstream execution.

Use 3-7 third-person action-summary interaction_beats per phase/candidate, never
prewritten dialogue. Do not paraphrase old events as new progress, create historical
contradictions to meet targets, or claim unobserved completion. Targets may remain
partial/not_realized; actual progress and content retention are judged separately.
Unexecuted or incomplete plans stay experiment records, not next-round facts or
mandatory to-dos. Later history is frozen text plus actually retained dialogue.

OUTPUT CONTRACT (root_incremental_v1 ONLY)
Return exactly these five top-level fields:
{"status":"PATCH" or "NO_SUPPLEMENT","original_plan_sha256":"...",
 "phases":[{"phase_id":"phase:followup_1","source_candidate_id":"an eligible ID",
   "original_goal":"exact source scene.goal","required_transition":"new action",
   "interaction_beats":["action 1","action 2","action 3"],
   "dependencies":[{"phase_id":"initial:<candidate_id>","condition":"actual prerequisite"}]}],
 "new_developments":[{"phase_id":"phase:new-event_2","dependencies":[],
   "candidate":{"source_basis":["existing-world basis"],
     "new_people":[{"name":"new person's name","relationship_basis":"connection to an existing person"}],
     "goal":"new proposed goal","speaker_refs":["SP01"],
     "transition_from_previous":"connection with prior context","before_state":"grounded starting situation",
     "required_transition":"new action","after_state":"desired result, not an observed fact",
     "actor_bindings":[{"actor_ref":"SP01","binding":"existing participant owns the core action"}],
     "interaction_beats":["action 1","action 2","action 3"]}}],
 "reason":"brief reason"}
Arrays may be empty individually. PATCH requires at least one addition across them;
NO_SUPPLEMENT requires both arrays empty. Do not emit source_candidate_id or
original_goal inside independent new developments. Do not emit the navigation
table, registry, scope hashes, full original plan, or extra fields as output.
"""


def render_root_supplement(inputs):
    projected, mapping = project_supplement_inputs(inputs, extra_reference_fields=("memory_navigation",))
    rows = []
    for candidate_id in inputs["eligible_source_candidate_ids"]:
        escaped = candidate_id.replace("~", "~0").replace("/", "~1")
        pointer = "/stable_candidates/" + escaped + "/scene/goal"
        goal, _ = _resolve_at(projected, pointer)
        rows.append({"source_candidate_id": candidate_id, "original_goal": goal,
                     "goal_source": pointer})
    template = (SHARED_FEEDBACK_RULES_V3 + "\n" + TARGET_RULES + ROOT_TASK
                + "\n" + PLANNER_DEVELOPMENT_GUIDANCE_V1)
    built = package("supplement.root_incremental_v1", template + "\nROOT SOURCE NAVIGATION\n"
                    + json.dumps(rows, ensure_ascii=False, indent=2), projected)
    built["input_projection_mapping"] = mapping
    built["prompt_version"] = ROOT_PROMPT_VERSION
    built["supplement_contract_identity"] = {"prompt_version": ROOT_PROMPT_VERSION,
        "version_source": "bundle.supplement_prompt_version", "policy_id": ROOT_POLICY_ID,
        "scope_mode": ROOT_SCOPE, "source_scope": inputs["source_scope"],
        "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "allowed_source_view": rows, "allowed_source_view_sha256": sha(rows),
        "output_contract": "root_incremental_patch_v1", "strict_decoding_added": False}
    built["supplement_contract_identity"]["development_guidance_version"] = "planner_development_guidance_v1"
    return built
