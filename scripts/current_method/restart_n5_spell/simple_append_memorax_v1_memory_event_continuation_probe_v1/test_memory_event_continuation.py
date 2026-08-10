from __future__ import annotations

import json
import unittest
from pathlib import Path

from .contracts import (
    build_event_candidates,
    build_semantic_gate_context,
    materialize_planner_output,
    normalize_realizer_output,
    validate_planner_output,
    validate_realizer_plan_isolation,
    validate_semantic_gate_output,
)
from .live_runtime import PER_ROLE_REQUEST_BUDGET, TOTAL_REQUEST_BUDGET
from .payloads import build_add_intent, build_bucket_search_payload, memorax_user_id
from .prompts import build_realizer_prompt, build_semantic_continuation_gate_prompt
from .workflow import WorkflowError, execute_round

PARENT_PATH = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/conversation_chronicles_v1/"
    "roots/cc-train-episode-13836/unified_root.json")


def _parent():
    turns=[]
    for i in range(1,7):
        turns.append({"turn_id":f"root:P01:T{i:03d}","session_id":"root:parent_session_01",
                      "speaker":"A" if i%2 else "B","text":f"parent {i}"})
    return {"root_id":"root","dialogue":{"dialogue_hash":"a"*64,"turns":turns}}


def _next_fields(index: int) -> dict[str, object]:
    return {"next_step_candidates":[
        {"new_development":f"next {index}",
         "durable_outcome":f"next outcome {index}",
         "interaction_beats":[f"rejected-only-{index}","next beat 2","next beat 3"]},
        {"new_development":f"later {index}",
         "durable_outcome":f"later outcome {index}",
         "interaction_beats":[f"selected-only-{index}","later beat 2","later beat 3"]},
    ]}


def _gate_reviews(*, event_count: int = 3, reject_event: int | None = None,
                  parent_ref: str = "root:P01:T001") -> dict[str, object]:
    reviews=[]
    for event_index in range(1,event_count+1):
        for candidate_index in range(1,3):
            reject=event_index==reject_event
            reviews.append({
                "candidate_ref":f"ME{event_index:02d}:C{candidate_index:02d}",
                "decision":"reject" if reject else "pass",
                "duplicate_with_parent":reject,
                "evidence_turn_refs":[parent_ref] if reject else [],
                "reason":"duplicate" if reject else "new",
            })
    return {"candidate_reviews":reviews}


class MemoryEventContinuationContractTest(unittest.TestCase):
    def test_live_request_budget_allows_one_planner_gate_and_realizer(self):
        self.assertEqual(PER_ROLE_REQUEST_BUDGET, 1)
        self.assertEqual(TOTAL_REQUEST_BUDGET, 3)

    def test_namespace_isolated_and_payload_secret_free(self):
        self.assertEqual(
            memorax_user_id("root"),
            "gxy-selfplay:memory-event-continuation-semantic-gate-probe-v1:root")
        intent=build_add_intent(root_id="root",round_index=0,source_dialogue_sha256="a"*64,turns=_parent()["dialogue"]["turns"],async_mode=True)
        serialized=json.dumps(intent)
        self.assertFalse(intent["api_key_in_artifact"])
        self.assertNotIn("Authorization",serialized)
        self.assertNotIn("MEMORAX_API_KEY",serialized)

    def test_candidates_are_stable(self):
        overview={"events":[{"canonical_key":"event-a","display_key":"A","content":["fact-a"]},{"canonical_key":"event-b","display_key":"B","content":["fact-b"]},{"canonical_key":"event-c","display_key":"C","content":["fact-c"]}]}
        self.assertEqual([x["memory_event_id"] for x in build_event_candidates(overview)], ["ME01","ME02","ME03"])

    def test_candidates_include_explicit_recent_events_from_people(self):
        overview={
            "events":[
                {"canonical_key":"event-a","display_key":"A","content":["fact-a"]},
                {"canonical_key":"event-b","display_key":"B","content":["fact-b"]},
            ],
            "people":[{
                "canonical_key":"person-a","display_key":"Person A",
                "content":["[关系] lives nearby",
                           "[事件 @近期] Person A scheduled the next meeting."]}],
        }
        candidates=build_event_candidates(overview)
        self.assertEqual(len(candidates),3)
        self.assertEqual(candidates[-1]["memory_event_key"],
                         "person-a:recent_event:01")
        self.assertIn("scheduled the next meeting",candidates[-1]["memory_event_content"][0])

    def test_duplicate_event_bucket_and_person_event_is_not_doubled(self):
        text="[事件 @近期] Person A scheduled the next meeting."
        overview={"events":[{"canonical_key":"event-a","display_key":"A",
                             "content":[text]}],
                  "people":[{"canonical_key":"person-a","display_key":"Person A",
                              "content":[text]}]}
        candidates=build_event_candidates(overview)
        self.assertEqual(len(candidates),1)

    def test_planner_requires_three_distinct_known_events_and_refs(self):
        parent=_parent(); candidates=build_event_candidates({"events":[{"canonical_key":f"e{i}","display_key":f"E{i}","content":[f"f{i}"]} for i in range(3)]})
        base={"round_goal":"g","scenes":[]}
        for i,c in enumerate(candidates,1):
            base["scenes"].append({"memory_event_id":c["memory_event_id"],"verified_current_state":"old",
                "evidence_turn_refs":["root:P01:T001"],"goal":"g","participants":["A","B"],
                **_next_fields(i),
                "transition_from_previous":"t"})
        self.assertTrue(validate_planner_output(base,candidates=candidates,parent=parent)["valid"])
        base["scenes"][2]["memory_event_id"]="ME99"
        self.assertFalse(validate_planner_output(base,candidates=candidates,parent=parent)["valid"])

    def test_planner_requires_candidates_and_selected_development(self):
        parent=_parent(); candidates=build_event_candidates({
            "events":[{"canonical_key":"e","display_key":"E","content":["f"]}]})
        scene={"memory_event_id":"ME01","verified_current_state":"old",
               "evidence_turn_refs":["root:P01:T001"],"goal":"g",
               "participants":["A","B"],**_next_fields(1),
               "transition_from_previous":"t"}
        plan={"round_goal":"g","scenes":[scene,dict(scene,memory_event_id="ME02"),
                                               dict(scene,memory_event_id="ME03")]}
        # Use three known candidates for the three-scene contract.
        candidates=build_event_candidates({"events":[
            {"canonical_key":f"e{i}","display_key":f"E{i}","content":[f"f{i}"]}
            for i in range(3)]})
        self.assertTrue(validate_planner_output(plan,candidates=candidates,parent=parent)["valid"])
        plan["scenes"][0]["next_step_candidates"][0]["candidate_id"]="C01"
        self.assertFalse(validate_planner_output(
            plan,candidates=candidates,parent=parent)["valid"])
        plan["scenes"][0]["next_step_candidates"][0].pop("candidate_id")
        materialized=materialize_planner_output(
            plan,passed_candidate_refs=["ME01:C02","ME02:C01","ME03:C01"])
        self.assertEqual(materialized["scenes"][0]["new_development"],"later 1")
        self.assertEqual(materialized["scenes"][0]["durable_outcome"],
                         "later outcome 1")
        self.assertEqual(materialized["scenes"][0]["interaction_beats"],
                         ["selected-only-1","later beat 2","later beat 3"])
        self.assertNotIn("next_step_candidates", materialized["scenes"][0])
        plan["scenes"][0]["selected_candidate_index"]=1
        self.assertFalse(validate_planner_output(plan,candidates=candidates,parent=parent)["valid"])

    def test_rejected_candidate_is_absent_from_realizer_input_and_prompt(self):
        raw_plan={"round_goal":"g","scenes":[{
            "memory_event_id":f"ME{i:02d}","verified_current_state":"state",
            "evidence_turn_refs":["root:P01:T001"],"goal":"g",
            "participants":["A","B"],**_next_fields(i),
            "transition_from_previous":"t"}
            for i in range(1,4)]}
        passed=["ME01:C02","ME02:C01","ME03:C01"]
        clean=materialize_planner_output(raw_plan,passed_candidate_refs=passed)
        isolation=validate_realizer_plan_isolation(
            raw_plan=raw_plan,materialized_plan=clean,
            passed_candidate_refs=passed)
        self.assertTrue(isolation["valid"])
        self.assertEqual(isolation["rejected_content_leak_count"],0)
        self.assertEqual(clean["scenes"][0]["interaction_beats"],
                         ["selected-only-1","later beat 2","later beat 3"])
        prompt=build_realizer_prompt(parent=_parent(),plan=clean)["prompt"]
        self.assertIn("selected-only-1",prompt)
        self.assertNotIn("rejected-only-1",prompt)
        self.assertNotIn("next_step_candidates",prompt)

    def test_realizer_program_assigns_scene_local_turn_refs(self):
        parent=_parent(); plan={"scenes":[{"memory_event_id":"ME01","participants":["A","B"]}]}
        raw={"scenes":[{"dialogue_turns":[{"speaker":"A","text":f"child {i}"} for i in range(9)]}]}
        result=normalize_realizer_output(raw,plan=plan,parent=parent)
        self.assertTrue(result["valid"]); self.assertEqual(result["child"]["scenes"][0]["dialogue_turns"][0]["turn_ref"],"R01:S01:T001")

    def test_semantic_gate_pass_and_reject_contracts(self):
        parent=_parent()
        plan={"scenes":[{"memory_event_id":"ME01",**_next_fields(1)},
                        {"memory_event_id":"ME02",**_next_fields(2)}]}
        passed=_gate_reviews(event_count=2)
        result=validate_semantic_gate_output(passed,plan=plan,parent=parent)
        self.assertTrue(result["valid"]); self.assertTrue(result["realizer_allowed"])

        first_rejected=_gate_reviews(event_count=2)
        first_rejected["candidate_reviews"][0]={
            "candidate_ref":"ME01:C01","decision":"reject",
            "duplicate_with_parent":True,
            "evidence_turn_refs":["root:P01:T002"],
            "reason":"The first candidate repeats a Parent action."}
        result=validate_semantic_gate_output(first_rejected,plan=plan,parent=parent)
        self.assertTrue(result["valid"])
        self.assertTrue(result["realizer_allowed"])
        self.assertEqual(result["overall_decision"],"pass")
        self.assertIn("ME01:C02",result["passed_candidate_refs"])

        rejected=_gate_reviews(event_count=2,reject_event=2,
                               parent_ref="root:P01:T002")
        result=validate_semantic_gate_output(rejected,plan=plan,parent=parent)
        self.assertTrue(result["valid"]); self.assertFalse(result["realizer_allowed"])
        self.assertEqual(result["overall_decision"],"reject")

    def test_semantic_gate_rejects_non_parent_evidence(self):
        reviews=_gate_reviews(event_count=1)
        reviews["candidate_reviews"][0]={
            "candidate_ref":"ME01:C01","decision":"reject",
            "duplicate_with_parent":True,"evidence_turn_refs":["R01:S01:T001"],
            "reason":"duplicate"}
        result=validate_semantic_gate_output(reviews,
            plan={"scenes":[{"memory_event_id":"ME01",**_next_fields(1)}]},
            parent=_parent())
        self.assertFalse(result["valid"])
        self.assertIn("gate_review_1_evidence_not_parent",result["errors"])

    def test_gate_context_includes_complete_parent_session(self):
        parent=_parent()
        context=build_semantic_gate_context(
            parent=parent,candidates=[{"memory_event_id":"ME01"}],
            plan={"scenes":[{"memory_event_id":"ME01",
                             "evidence_turn_refs":["root:P01:T001"],
                             "verified_current_state":"state",
                             **_next_fields(1)}]})
        refs=[row["turn_ref"] for row in context["scenes"][0]["authoritative_parent_evidence"]]
        self.assertEqual(refs,[f"root:P01:T{i:03d}" for i in range(1,7)])
        self.assertEqual(
            [row["candidate_ref"] for row in context["scenes"][0]["candidates"]],
            ["ME01:C01","ME01:C02"])
        self.assertEqual(context["scenes"][0]["candidates"][1]["interaction_beats"],
                         ["selected-only-1","later beat 2","later beat 3"])

    def test_gate_prompt_is_narrow_and_non_repairing(self):
        prompt=build_semantic_continuation_gate_prompt(
            gate_context={"scenes":[]})["prompt"]
        for category in ("decision","intention","action","agreement","resolution","result"):
            self.assertIn(f"- {category}",prompt)
        self.assertIn("Do not repair, rewrite",prompt)

    def test_complete_zero_api_workflow(self):
        parent=json.loads(PARENT_PATH.read_text(encoding="utf-8"))
        speakers=list(dict.fromkeys(row["speaker"] for row in parent["dialogue"]["turns"]))
        refs=[row["turn_id"] for row in parent["dialogue"]["turns"]]
        events={f"event-{i}":{"display_key":f"Event {i}","content":[f"event fact {i}"]} for i in range(1,4)}

        class Gateway:
            def __init__(self): self.calls=[]; self.search_count=0
            def add(self,payload,*,request_id):
                self.calls.append({"role":"add","request_id":request_id})
                return {"success":True,"data":{"status":"completed"}}
            def search_buckets(self,payload,*,request_id):
                self.calls.append({"role":"search","request_id":request_id}); self.search_count+=1
                current={} if self.search_count==1 else events
                return {"success":True,"data":{"buckets":{"person":{},"location":{},"event":current,"item":{},"dreaming":{}}}}
            def add_status(self,*args,**kwargs): raise AssertionError("sync fixture must not poll")

        def planner_call(prompt):
            return {"round_goal":"continue three events","scenes":[{
                "memory_event_id":f"ME{i:02d}","verified_current_state":f"state {i}",
                "evidence_turn_refs":[refs[i-1]],"goal":f"goal {i}",
                "participants":speakers[:2],**_next_fields(i),
                "transition_from_previous":"continue"}
                for i in range(1,4)]}

        def realizer_call(prompt):
            return {"scenes":[{"dialogue_turns":[{
                "speaker":speakers[j%2],"text":f"scene {i} generated turn {j}"}
                for j in range(1,10)]} for i in range(1,4)]}

        def semantic_gate_call(prompt):
            return _gate_reviews(event_count=3,parent_ref=refs[0])

        gateway=Gateway()
        result=execute_round(parent=parent,memorax=gateway,planner_call=planner_call,
            semantic_gate_call=semantic_gate_call,realizer_call=realizer_call,
            token_counter=lambda text:max(1,len(text)//4),
            poll_delays_seconds=(0.0,),sleep_fn=lambda _:None)
        self.assertTrue(result["generation_ready"])
        self.assertEqual(len(result["child_dialogue"]["scenes"]),3)
        self.assertEqual([row["role"] for row in gateway.calls],
                         ["search","add","search","add","search"])

    def test_memory_guard_receives_contract_roles_not_artifact_labels(self):
        parent=json.loads(PARENT_PATH.read_text(encoding="utf-8"))
        speakers=list(dict.fromkeys(row["speaker"] for row in parent["dialogue"]["turns"]))
        refs=[row["turn_id"] for row in parent["dialogue"]["turns"]]
        events={f"event-{i}":{"display_key":f"Event {i}","content":[f"event fact {i}"]} for i in range(1,4)}
        seen_roles=[]

        class Gateway:
            def __init__(self):
                self.calls=[]
                self.search_count=0
            def search_buckets(self,payload,*,request_id):
                self.search_count += 1
                current = {} if self.search_count == 1 else events
                return {"success":True,"data":{"buckets":{"person":{},"location":{},"event":current,"item":{},"dreaming":{}}}}
            def add(self,payload,*,request_id):
                return {"success":True,"data":{"status":"completed"}}
            def add_status(self,*args,**kwargs):
                raise AssertionError("sync fixture must not poll")

        def planner_call(prompt):
            return {"round_goal":"g","scenes":[{
                "memory_event_id":f"ME{i:02d}","verified_current_state":"state",
                "evidence_turn_refs":[refs[i-1]],"goal":"g",
                "participants":speakers[:2],**_next_fields(i),
                "transition_from_previous":"t"}
                for i in range(1,4)]}

        def realizer_call(prompt):
            return {"scenes":[{"dialogue_turns":[{
                "speaker":speakers[j%2],"text":f"scene {i} turn {j}"}
                for j in range(1,10)]} for i in range(1,4)]}

        def semantic_gate_call(prompt):
            return _gate_reviews(event_count=3,parent_ref=refs[0])

        execute_round(
            parent=parent, memorax=Gateway(), planner_call=planner_call,
            semantic_gate_call=semantic_gate_call,realizer_call=realizer_call,
            before_memory_request=lambda role, request_id: seen_roles.append(role),
            token_counter=lambda text:max(1,len(text)//4),
            poll_delays_seconds=(0.0,),sleep_fn=lambda _:None)
        self.assertEqual(seen_roles, [
            "isolation_search", "parent_add", "pre_generation_search",
            "child_add", "post_generation_search"])

    def test_gate_reject_stops_before_realizer_and_child_add(self):
        parent=json.loads(PARENT_PATH.read_text(encoding="utf-8"))
        speakers=list(dict.fromkeys(row["speaker"] for row in parent["dialogue"]["turns"]))
        refs=[row["turn_id"] for row in parent["dialogue"]["turns"]]
        events={f"event-{i}":{"display_key":f"Event {i}","content":[f"fact {i}"]}
                for i in range(1,4)}
        class Gateway:
            def __init__(self): self.calls=[]; self.search_count=0
            def add(self,payload,*,request_id):
                self.calls.append("add"); return {"success":True,"data":{"status":"completed"}}
            def search_buckets(self,payload,*,request_id):
                self.calls.append("search"); self.search_count+=1
                current={} if self.search_count==1 else events
                return {"success":True,"data":{"buckets":{"person":{},"location":{},"event":current,"item":{},"dreaming":{}}}}
            def add_status(self,*args,**kwargs): raise AssertionError("unexpected poll")
        def planner_call(prompt):
            return {"round_goal":"g","scenes":[{
                "memory_event_id":f"ME{i:02d}","verified_current_state":"state",
                "evidence_turn_refs":[refs[i-1]],"goal":"g",
                "participants":speakers[:2],**_next_fields(i),
                "transition_from_previous":"t"}
                for i in range(1,4)]}
        def gate_call(prompt):
            return _gate_reviews(event_count=3,reject_event=2,
                                 parent_ref=refs[1])
        gateway=Gateway()
        with self.assertRaisesRegex(WorkflowError,"semantic_continuation_gate"):
            execute_round(parent=parent,memorax=gateway,planner_call=planner_call,
                semantic_gate_call=gate_call,
                realizer_call=lambda prompt: self.fail("Realizer must not run"),
                token_counter=lambda text:max(1,len(text)//4),
                poll_delays_seconds=(0.0,),sleep_fn=lambda _:None)
        self.assertEqual(gateway.calls,["search","add","search"])


if __name__ == "__main__": unittest.main()
