from __future__ import annotations

import json
from typing import Any

from .backends import BackendResponse, QuestionerRequest, RealizerRequest, SolverRequest
from .core import normalize_option


def _session_turns(session: int) -> list[dict[str, str]]:
    turns = []
    for index in range(12):
        speaker = "John" if index % 2 == 0 else "Maria"
        detail = (
            f"During session {session}, step {index + 1}, {speaker} reviewed a distinct part of the "
            "neighborhood priorities brief, compared resident observations with the current plan, "
            "and explained what still needed confirmation before the next meeting. The discussion "
            "connected education repairs with infrastructure safety while keeping the advisory "
            "application realistic, specific, and grounded in the information they had collected."
        )
        turns.append({"speaker": speaker, "text": detail})
    if session == 1:
        turns[0]["text"] += " John's brief status was drafting."
    if session == 4:
        turns[-1]["text"] += " John's brief status is submitted."
    return turns


class ScriptedMockRealizer:
    model_identifier = "mock-realizer-v1"

    def generate_session(self, request: RealizerRequest) -> BackendResponse:
        if request.session_index == 2 and request.attempt == 1:
            payload = {"dialogue_turns": [{"speaker": "Unauthorized Stranger", "text": "I decide everything."}],
                "used_parent_anchor_ids": [], "advanced_event_arc_ids": [],
                "continued_unresolved_hook_ids": [],
                "durable_assertion_candidates": [], "local_scene_facts": [],
                "resolved_hook_ids": [], "new_unresolved_hooks": []}
            return BackendResponse(json.dumps(payload), self.model_identifier, {"script": "reject_attempt"})
        session = request.session_index
        prior_hook = [] if session == 1 else [f"H{session-1:04d}"]
        payload = {"dialogue_turns": _session_turns(session),
            "used_parent_anchor_ids": ["P1"] if session == 1 else [],
            "advanced_event_arc_ids": ["E1", "E2", "E3", "E4"],
            "continued_unresolved_hook_ids": prior_hook,
            "durable_assertion_candidates": (
                [{"subject":"John's brief", "predicate":"status", "object_or_state":"drafting",
                  "supporting_turn_indices":[0]}] if session == 1 else
                [{"subject":"John", "predicate":"outcome", "object_or_state":"won",
                  "supporting_turn_indices":[0]}] if session == 2 else
                [{"subject":"John's brief", "predicate":"status", "object_or_state":"submitted",
                  "supporting_turn_indices":[11]},
                 {"subject":"John's brief", "predicate":"status", "object_or_state":"withdrawn",
                  "supporting_turn_indices":[11]}] if session == 4 else []),
            "local_scene_facts": [f"Session {session} advanced the brief."],
            "resolved_hook_ids": [],
            "new_unresolved_hooks": [] if session == 4 else [{"description":f"Continue planned step after session {session}."}]}
        return BackendResponse(json.dumps(payload), self.model_identifier, {"script": "accepted"})


def _qa(qa_id: str, gold: str, turn_ids: tuple[str, ...], session_ids: tuple[str, ...],
        *, question: str | None = None, invalid: bool = False) -> dict[str, Any]:
    correct = f"education and infrastructure priorities for {qa_id} are confirmed"
    distractors = ["the neighborhood priorities brief", "resident observations", "the current plan"]
    distractor_values = ["neighborhood priorities brief", "resident observations", "current plan"]
    number = int(qa_id[-2:])
    child_turn_ids = [x for x in turn_ids if str(x).startswith("C")]
    child_session_ids = [x for x in session_ids if str(x).startswith("session_")]
    if number >= 7 and len(child_turn_ids) >= 14 and len(child_session_ids) >= 2:
        evidence = [child_turn_ids[0], child_turn_ids[12], child_turn_ids[13]]
        sessions = list(child_session_ids[:2])
    else:
        evidence = list(child_turn_ids[:2])
        sessions = list(child_session_ids[:1])
    item = {"qa_id": qa_id, "question": question or f"Which conclusion follows for scenario {qa_id}?",
        "correct_answer_text": correct, "distractors": distractors,
        "correct_answer_explanation": "The cited turns jointly establish the conclusion.",
        "distractor_explanations": [f"Distractor {index} follows a distinct incorrect path." for index in range(1, 4)],
        "distractor_values_from_evidence":distractor_values,
        "distractor_supporting_quotes":[
            [{"turn_id":evidence[0], "exact_quote":"neighborhood priorities brief"}],
            [{"turn_id":evidence[0], "exact_quote":"resident observations"}],
            [{"turn_id":evidence[0], "exact_quote":"current plan"}]],
        "evidence_turn_ids": evidence, "evidence_session_ids": sessions,
        "answer_support_type": "multi_hop",
        "answer_evidence_values":[
            {"turn_id":turn_id, "value":("education" if position == 0 else
                "infrastructure" if position == 1 else "resident observations"),
             "exact_quote":("education repairs" if position == 0 else
                "infrastructure safety" if position == 1 else "resident observations"),
             "role":"answer_value" if position < 2 else "supporting_fact"}
            for position, turn_id in enumerate(evidence)],
        "final_reasoning_conclusion": correct,
        "reasoning_steps": ["Combine the first fact.", "Apply the second fact.", "Derive the conclusion."],
        "reasoning_type": f"mock_path_{qa_id}",
        "difficulty_level": "L2" if number >= 7 else "L1"}
    if invalid:
        item["distractors"][2] = item["distractors"][0]
    return item


class ScriptedMockQuestioner:
    model_identifier = "mock-questioner-v1"

    def __init__(self, shortfall_batch: int | None = None, reject_all: bool = False):
        self.shortfall_batch = shortfall_batch
        self.reject_all = reject_all

    def generate_batch(self, request: QuestionerRequest) -> BackendResponse:
        mapping = {0: ("qa01", "qa02", "qa03"), 1: ("qa04", "qa05", "bad01"),
                   2: ("qa06", "qa07", "dup01"), 3: ("qa08", "qa09", "qa10")}
        ids = mapping[request.batch_index]
        rows = []
        for index, qa_id in enumerate(ids):
            if qa_id == "bad01":
                rows.append(_qa(qa_id, request.assigned_gold_letters[index], request.valid_turn_ids,
                                request.valid_session_ids, invalid=True))
            elif qa_id == "dup01":
                rows.append(_qa(qa_id, request.assigned_gold_letters[index], request.valid_turn_ids,
                                request.valid_session_ids,
                                question="Which conclusion follows for scenario qa01?"))
            else:
                rows.append(_qa(qa_id, request.assigned_gold_letters[index], request.valid_turn_ids,
                                request.valid_session_ids))
        if self.shortfall_batch == request.batch_index:
            rows = rows[:2]
        if self.reject_all:
            for row in rows:
                row["distractors"][2] = row["distractors"][0]
        return BackendResponse(json.dumps(rows), self.model_identifier,
                               {"batch_index": request.batch_index, "object_count": len(rows)})


class ScriptedMockSolver:
    model_identifier = "mock-solver-v1"

    @staticmethod
    def _letters(request: SolverRequest) -> tuple[str, str]:
        correct = next(k for k, value in request.options.items()
                       if normalize_option(value) == normalize_option(request.canonical_answer_text))
        wrong = next(k for k in "ABCD" if k != correct)
        return correct, wrong

    def _desired_reward(self, r: SolverRequest) -> int:
        number = int(r.qa_id[-2:])
        if r.evaluation_mode == "no_context":
            if number == 1:
                return 1
            # Exercise the adaptive boundary exactly: initial slots produce
            # [1, 1, 0], then the two extension slots produce [0, 0].  The
            # orchestrator must preserve K3, request only slots 3 and 4, and
            # decide pass_grounding from the merged K5 rewards.
            if number == 4:
                return int(r.target_slot < 2)
            return 0
        if r.evaluation_mode == "full_context":
            patterns = {2: [1]*5, 3: [0]*5, 4: [1,0,1,0,0], 5: [1,1,0,1,0],
                        6: [1,0,1,0,0], 7: [1,0,1,0,0], 8: [1,0,1,0,0],
                        9: [1,0,1,0,0], 10: [1,0,1,0,0]}
            return patterns[number][r.target_slot]
        if r.evaluation_mode == "evidence_only":
            return 0 if number == 7 else (1 if r.target_slot < 2 else 0)
        if r.evaluation_mode == "evidence_masked":
            return 1 if number == 7 else 0
        if r.evaluation_mode == "option_permutation":
            if number == 8:
                rates = {"perm1": 4, "perm2": 2}
            elif number == 9:
                rates = {"perm1": 5, "perm2": 0}
            else:
                rates = {"perm1": 2, "perm2": 2}
            return int(r.target_slot < rates[r.order_id])
        return 0

    def generate(self, request: SolverRequest) -> BackendResponse:
        number = int(request.qa_id[-2:])
        if number == 5 and request.evaluation_mode == "full_context" and request.target_slot == 1 and request.retry_index == 0:
            raw = "Answer: A and B"
        elif number == 6 and request.evaluation_mode == "full_context" and request.target_slot == 4:
            raw = ""
        elif number == 10 and request.evaluation_mode == "option_permutation" and request.order_id == "perm2" and request.target_slot == 4:
            raw = "A or B"
        else:
            correct, wrong = self._letters(request)
            reward = self._desired_reward(request)
            letter = correct if reward else wrong
            raw = f"<box>{letter}</box>"
        return BackendResponse(raw, self.model_identifier,
                               {"qa_id": request.qa_id, "mode": request.evaluation_mode})
