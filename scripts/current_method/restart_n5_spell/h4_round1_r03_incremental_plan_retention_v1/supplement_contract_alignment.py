"""Opt-in input explanations for the unchanged incremental patch contract."""
import hashlib
import json

from .input_views import _resolve_at

LEGACY_VERSION = "incremental_retention_v1.supplement.readiness_v2"
ALIGNED_VERSION = "incremental_retention_v1.supplement.contract_alignment_v1"
VERSION_FIELD = "supplement_prompt_version"

CONTRACT_ALIGNMENT_RULES = r"""INCREMENTAL PATCH CONTRACT EXPLANATION v1
The following clarifies the existing output fields; it adds no output fields.

PHASE IDENTITY AND DEPENDENCIES
A new phase_id must fully match this exact regex: phase:[A-Za-z0-9_-]+
After the single 'phase:' prefix, use one or more ASCII letters, digits,
underscores or hyphens. No second colon, whitespace or dots. No trimming or
case normalization is performed. IDs must be unique and not already known.
Two legal new IDs are phase:followup_1 and phase:followup-2.
phase:ND01:C01 is illegal because the suffix contains a second colon.
The old output illustration 'phase:...' is a placeholder, not a legal value.
source_candidate_id and phase_id are different identity types. Do not turn a
colon-containing candidate ID into a phase ID by simply adding a prefix/suffix.

Every phase must have a nonempty dependencies list. Each dependency has exactly
phase_id and condition; condition is a nonempty string naming an actual prerequisite.
A dependency phase_id can be initial:<candidate_id> for ANY candidate present
in stable_candidates, or the ID of an earlier added phase in this phases array.
No self-reference, forward reference or unknown reference is permitted.
For example, after phase:followup_1, phase:followup-2 can contain:
{"dependencies":[{"phase_id":"phase:followup_1","condition":"The prerequisite established by the earlier phase actually holds."}]}
This fragment explains reference syntax only, not a complete patch or a task
condition to copy. Supply concrete task conditions; the reference existing does
not prove the condition holds. An initial reference retains the entire candidate
ID, including its colon, unlike a newly generated phase ID. Execution follows
input order and actual dependency checks, not lexicographic ordering of IDs.

ALLOWED SOURCES AND BACKGROUND
For every added phase: source_candidate_id must be in affected_candidate_ids.
The ALLOWED SUPPLEMENT SOURCES table is derived from exactly that list; it is
the complete whitelist for new phase sources. The full original plan and other
candidates are background, not permission to add phases for those candidates.
One allowed source may have multiple follow-on phases, subject to the existing
policy and execution budget. The number of sources is not a phase quota.
Background mentions, dependency references and source_candidate_id are distinct.
Referencing another existing initial phase as a dependency does not grant
permission to use its candidate as a new phase source, and is not itself a
source-scope violation. Unrelated candidates remain unchanged.

EXACT ORIGINAL GOAL AND PHASE CONTENT
For each new phase, use this formula after resolving existing $input_ref values:
original_goal = stable_candidates[source_candidate_id].scene.goal
The table's original_goal is that exact parsed string, not a generated summary.
Copy it from the same legal source without translating, summarizing, changing
punctuation, trimming or rewording. Do not substitute required_transition,
after_state, another candidate's goal, or this new phase's particular action.
Normal JSON escaping is allowed only if the parsed string remains identical.
The goal_source pointer identifies the authoritative scene.goal within INPUT DATA.

original_goal preserves the association with the original event objective.
Describe the phase-specific action or intended intermediate result in the existing
required_transition field, supported by interaction_beats and dependencies.
There is no additional phase-result output field. Never add or remove fields
from the existing output contract. Do not output this table or template identity.
Copying the original goal correctly does not require forcing its completion.
Original target completion and retention of actual qualified progress remain
separate under the existing policy. Plans and dependency conditions are not facts.
Return NO_SUPPLEMENT with an empty phases array when no safe supplement exists.
"""


def select_supplement_version(bundle):
    version = bundle.get(VERSION_FIELD, LEGACY_VERSION)
    if version not in (LEGACY_VERSION, ALIGNED_VERSION):
        raise ValueError("unknown_supplement_prompt_version")
    return version


def allowed_source_view(inputs):
    """Resolve exact source goals using the existing lossless reference API."""
    allowed = inputs["affected_candidate_ids"]
    registry = inputs["stable_candidates"]
    rows = []
    for candidate_id in allowed:
        if candidate_id not in registry:
            raise ValueError("allowed_source_missing_from_registry")
        escaped = candidate_id.replace("~", "~0").replace("/", "~1")
        pointer = "/stable_candidates/" + escaped + "/scene/goal"
        goal, _ = _resolve_at(inputs, pointer)
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("allowed_source_goal_string_required")
        rows.append({"source_candidate_id": candidate_id, "original_goal": goal,
                     "goal_source": pointer})
    return rows


def contract_sidecar(version, template, rows):
    return {"prompt_version": version, "version_source": "bundle." + VERSION_FIELD,
            "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
            "allowed_source_view": rows,
            "allowed_source_view_sha256": hashlib.sha256(json.dumps(
                rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "output_contract": "UNCHANGED_merge_patch", "strict_decoding_added": False}
