"""Versioned initial candidate counts; historical and append defaults stay unchanged."""
from copy import deepcopy

VERSION = "initial_candidates_target8_cap9_v1"
TARGET_NEW_CANDIDATES = 8
MAX_NEW_CANDIDATES = 9
MAX_CONTINUATIONS = 3
SCENE_QUOTA = 6
GATE_REQUEST_LIMIT = MAX_CONTINUATIONS + MAX_NEW_CANDIDATES

def candidate_limit(config):
    version = config.get("candidate_policy_version")
    if version is None:
        return 8
    if version != VERSION or config.get("append_enabled") is not False:
        raise ValueError("candidate_cap9_requires_explicit_initial_only_policy")
    return MAX_NEW_CANDIDATES

def gate_budget(budget, *, max_new_candidates=8):
    if type(max_new_candidates) is not int or max_new_candidates not in (8, 9):
        raise ValueError("unsupported_initial_candidate_limit")
    result = deepcopy(budget)
    limit = MAX_CONTINUATIONS + max_new_candidates
    result["role_requests"] = {r: (1 if r == "planner" else limit if r == "gate" else 0)
        for r in ("planner", "supplement", "gate", "realizer", "adherence", "closure")}
    result["total_requests"] = 1 + limit
    result["max_generated_tokens"] = result["max_output_tokens"]["planner"] + limit * result["max_output_tokens"]["gate"]
    return result
