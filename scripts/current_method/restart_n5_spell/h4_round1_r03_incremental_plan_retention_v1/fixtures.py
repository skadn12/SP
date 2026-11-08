"""Explicitly synthetic responses. No fixture is an actual R03 observation."""
from copy import deepcopy
import json
import re

from .budget import ROLES
from .planning import prepare_original
from .bundle import PROJECT, FAILED, PRIOR, build_failed_bundle


def example_budget():
    return {"total_requests": 45,
            "role_requests": {"planner": 0, "supplement": 1, "gate": 20, "realizer": 10, "adherence": 10, "closure": 1},
            "max_input_tokens": 200000, "max_output_tokens": {role: 8192 for role in ROLES},
            "context_window_tokens": 220000, "max_generated_tokens": 400000,
            "max_cumulative_tokens": 200000, "scene_output_reserve_tokens": 20000}


def failed_bundle():
    bundle = build_failed_bundle(branch_id="offline-fixture-incremental-v1-122711")
    bundle["fixture_notice"] = "Historical plan/history real and immutable; all new responses are mocks."
    return bundle


def fixture_patch(bundle):
    prepared = prepare_original(bundle["original_plan"], parent=bundle["parent"], candidates=bundle["candidates"], verified_tracks=bundle["verified_tracks"])
    ref = prepared["affected_candidate_ids"][0]
    goal = prepared["candidate_registry"][ref]["scene"]["goal"]
    return {"status": "PATCH", "original_plan_sha256": prepared["original_sha256"],
            "reason": "OFFLINE MOCK: test two same-event followups without changing goal.",
            "phases": [{"phase_id": "phase:followup_" + str(n), "source_candidate_id": ref,
                        "original_goal": goal,
                        "required_transition": "James finalizes a concrete invitation draft phase " + str(n),
                        "interaction_beats": ["James selects a specific meeting date.", "Emily checks the prepared invitation.", "James saves the agreed revision."],
                        "dependencies": [{"phase_id": "initial:" + ref, "condition": "invitation draft prepared"}]}
                       for n in (1, 2)]}


class FixtureBackend:
    offline_mock = True
    def __init__(self, bundle, *, fail_role=None, no_supplement=False, establish=True, bad_audit=None):
        self.bundle = bundle
        self.fail_role = fail_role
        self.no_supplement = no_supplement
        self.establish = establish
        self.bad_audit = bad_audit
        self.requests = []

    def request(self, request):
        self.requests.append(deepcopy(request))
        role = request["role"]
        if role == self.fail_role:
            raise TimeoutError("MOCK ambiguous timeout")
        text = request["messages"][0]["content"]
        data = json.loads(text.rsplit("\n\nINPUT DATA\n", 1)[1])
        if role == "supplement":
            patch = fixture_patch(self.bundle)
            if self.no_supplement:
                patch.update(status="NO_SUPPLEMENT", phases=[])
            return patch
        if role == "gate":
            # This entry intentionally audits one phase/candidate at a time;
            # later refs in the legacy prompt are contract examples, not inputs.
            refs = re.findall(r"(?:ME|ND)\d+:C\d+", text)[:1]
            return {"candidate_reviews": [{"candidate_ref": ref, "decision": "pass",
                "duplicate_with_parent": False, "evidence_turn_refs": [],
                "parent_relation": "NONE", "reason": "Explicit synthetic Gate PASS fixture, not semantic evidence."} for ref in refs]}
        if role == "realizer":
            return {"scenes": [{"dialogue_turns": [
                {"speaker": scene["required_participants"][i % len(scene["required_participants"])],
                 "text": f"OFFLINE MOCK {scene['phase_id']} detail {i}: We prepared a draft but have not sent any invitation."}
                for i in range(14)]} for scene in data["planned_phases_not_facts"]]}
        if role == "adherence":
            rows = []
            generated = json.loads(request["messages"][1]["content"])
            scenes = []
            if "actual_generated_dialogue_metadata" in data:
                for actual in generated["scenes"]:
                    scenes.append({**actual, "dialogue_turns": [
                        {**turn["generated_turn"], "turn_ref": turn["turn_ref"]}
                        for turn in actual["dialogue_turns"]]})
            else:
                for ident, actual in zip(data["actual_generated_dialogue_identity"]["scenes"], generated["scenes"]):
                    scenes.append({**ident, "dialogue_turns": [
                        {**turn, "turn_ref": ref} for turn, ref in zip(actual["dialogue_turns"], ident["dialogue_turn_refs_in_output_order"])]})
            for scene in scenes:
                ref = scene["dialogue_turns"][0]["turn_ref"]
                row = {"scene_ref": scene["scene_ref"], "goal_status": "partial", "content_decision": "retain",
                       "participant_adherence": "pass", "history_consistency": "pass", "novel_progress": "pass",
                       "actual_states": [{"statement": scene["dialogue_turns"][0]["text"], "stage": "arrangement", "turn_refs": [ref]}],
                       "goal_evidence_refs": [ref], "condition_results": [
                           {"condition": c, "status": "established" if self.establish else "UNKNOWN",
                            "turn_refs": [ref] if self.establish else []}
                           for c in data["downstream_conditions"][scene["scene_ref"]]],
                       "reason": "MOCK partial progress judgment; no objective semantic claim."}
                if request.get("prompt_version", "").endswith("adherence.scoped_evidence_v4"):
                    row["history_evidence_refs"] = []
                if self.bad_audit:
                    row.update(self.bad_audit)
                rows.append(row)
            return {"scene_audits": rows}
        raise AssertionError("unexpected mock role:" + role)
