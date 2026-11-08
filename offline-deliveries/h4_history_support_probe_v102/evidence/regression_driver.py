"""Frozen history-only claim probes and at most one fixed-plan semantic re-review."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

BASE = Path(__file__).resolve().parents[1]
RUN = BASE / "regression"
OLD = BASE.parent / "h4_review_wrapper_correction_v101"
OLDER = BASE.parent / "h4_bound_card_review_feedback_v100"
sys.path.insert(0, str(BASE / "expansion_engine"))
import runner as r
import history_support as hs


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    r.base.write(Path(path), value)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def now():
    return datetime.now(timezone.utc).isoformat()


def claim(identifier, statement, status, evidence, basis="reported", **optional):
    value = {"claim_id": identifier, "statement": statement, "status": status,
             "evidence_basis": basis, "evidence": deepcopy(evidence), **optional}
    value["claim_sha256"] = hs.claim_digest(value)
    return value


def actual_claims(review):
    return [claim("timeline:" + row["constraint_id"], row["statement"], row["status"],
                  row["evidence"], row["evidence_basis"],
                  **({"uncertainty": row["uncertainty"]} if "uncertainty" in row else {}))
            for row in review["timeline"]]


def cite(history, ref, quote):
    original = next(turn for turn in history if turn["turn_ref"] == ref)
    assert quote and quote in original["text"]
    return {"turn_ref": ref, "quote": quote}


def prepare():
    assert not RUN.exists()
    RUN.mkdir()
    a = read(OLD / "regression/INPUT.json")
    b = read(OLDER / "regression/inputs/25652.json")
    candidate_path = next((OLD / "regression/requests").glob("001-*/CANDIDATE_PLAN.json"))
    review_b_path = next((OLDER / "regression/requests").glob("003-*/RESPONSE.json"))
    candidate = read(candidate_path)
    review_a = read(OLD / "regression/POST_REPLAN_REVIEW.json")
    review_b = json.loads(read(review_b_path)["text"])
    write(RUN / "inputs/19053.json", a)
    write(RUN / "inputs/25652.json", b)
    write(RUN / "FIXED_PLAN.json", candidate)
    write(RUN / "BASELINE_REVIEW.json", review_a)
    write(RUN / "BASELINE_REVIEW_25652.json", review_b)
    fixtures, expectations = {}, {}
    for short, data, review in [("19053", a, review_a), ("25652", b, review_b)]:
        context = data["context"]
        history = context["cumulative_history"]
        claims = actual_claims(review)
        actual_ids = [row["claim_id"] for row in claims]
        expected = {}
        def add(identifier, statement, status, evidence, allowed):
            claims.append(claim("probe:" + identifier, statement, status, evidence))
            expected["probe:" + identifier] = allowed
        if short == "19053":
            expected["timeline:c3"] = ["unsupported", "uncertain"]
            add("website_checked", "Emily checked the animal shelter website this morning and found volunteer slots for next Saturday.",
                "completed", [cite(history, "R01:S01:T001", "I checked their website this morning and there are some volunteer slots open next Saturday.")], ["supported"])
            add("snacks_planned", "Emily said she will bring snacks after the volunteer session.", "planned",
                [cite(history, "R01:S01:T011", "I’ll also bring some snacks for after the session.")], ["supported"])
            add("volunteer_session_completed", "Emily and James have already completed the volunteer session scheduled for next Saturday.",
                "completed", [cite(history, "R01:S01:T014", "Let’s make sure we’re both there and ready to help. Saturday can’t come soon enough.")], ["unsupported", "uncertain"])
        else:
            add("willing_to_help", "Noel expressed willingness to help draft the project proposal and suggested starting by outlining its goals and structure.",
                "completed", [cite(history, "R01:S01:T004", "Absolutely, I'd be happy to. Let's start by outlining the main goals and structure of the project.")], ["supported"])
            add("past_project_completed", "Milan completed a difficult project on time last year.", "completed",
                [cite(history, "cc-train-episode-25652:P01:T010", "Remember when you took on that difficult project last year and completed it on time?")], ["supported"])
            add("boss_meeting_completed", "Milan has already met with the boss to discuss leading the new project.", "completed",
                [cite(history, "cc-train-episode-25652:P01:T007", "Not yet, but I plan to set up a meeting with her soon to discuss it.")], ["unsupported", "uncertain"])
            add("coffee_interest", "Elena expressed interest in getting coffee with Milan and said they would need to plan it.", "completed",
                [cite(history, "R01:S03:T012", "That sounds nice. I'd like that. I'll make sure to get your number so we can plan it.")], ["supported"])
            add("coffee_completed", "Milan and Elena have already had coffee together.", "completed",
                [cite(history, "R01:S03:T012", "That sounds nice. I'd like that. I'll make sure to get your number so we can plan it.")], ["unsupported", "uncertain"])
        fixture = {"root_id": context["root_id"], "history": history, "claims": claims,
                   "actual_review_claim_ids": actual_ids, "review_source_sha256": r.base.digest(review)}
        fixtures[short] = fixture
        write(RUN / f"fixtures/{short}.json", fixture)
        expectations[short] = {"required_verdicts": expected,
                               "diagnostic_only": [identifier for identifier in actual_ids if identifier not in expected]}
    write(RUN / "EXPECTATIONS.json", {"version": "v102_frozen_probe_expectations_v1",
          "scope": "Analyst diagnostic expectations only; not generated question gold and NEVER included in requests.",
          "cases": expectations, "all_clear_controls_required_before_new_review": True})
    profile, transport, tokenizer = r.transport_inputs()
    write(RUN / "provider_profile.json", profile)
    _profile, _factory, annotate, capacity, dev = r.base.transport()
    capacities = []
    for short, fixture in fixtures.items():
        payload = hs.build_request(fixture["root_id"], fixture["history"], fixture["claims"])
        target = RUN / f"prepared/{short}_initial_support.json"
        write(target, payload)
        cap = annotate(capacity(profile, dev.tokenizer_from(str(r.base.TOKENIZER)), payload["messages"], 4096))
        assert cap["status"] == "PASS"
        capacities.append({"case": short, "prepared_request": binding(target), "capacity": cap})
    write(RUN / "CAPACITY_PREFLIGHT.json", {"status": "PASS", "provider_calls": 0, "requests": capacities})
    paths = set((BASE / "expansion_engine").glob("*.py")) | set((BASE / "qa_engine").glob("*.py"))
    paths.update([Path(__file__), BASE / "PRECALL_PLAN_ZH.md", BASE / "PRESERVATION_BEFORE.json",
                  OLD / "COMPLETE_MANIFEST.json", OLD / "regression/INPUT.json", candidate_path,
                  OLD / "regression/POST_REPLAN_REVIEW.json", OLDER / "regression/inputs/25652.json", review_b_path])
    paths.update(BASE / "expansion_engine" / n for n in ["SOURCE_REGISTRY.json", "SOURCE_SELECTION.json"])
    paths.update(q for q in RUN.rglob("*") if q.is_file())
    paths.update(transport)
    paths.update(tokenizer)
    paths.update(r.LEGACY.glob("*.py"))
    for module in tuple(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if file:
            q = Path(file).resolve()
            if q.is_file() and q.suffix == ".py" and q.is_relative_to(r.base.PROJECT) and ".venv" not in str(q):
                paths.add(q)
    write(RUN / "REGRESSION_MANIFEST.json", {"version": "v102_history_claim_probe_v1", "created_at_utc": now(),
          "scope": "Two exposed histories; frozen claims and clear controls; at most one fixed-plan semantic re-review.",
          "max_logical_requests": 8, "http_max": 16, "max_tokens": 4096, "max_stage_structure_attempts": 2,
          "fixed_candidate": binding(RUN / "FIXED_PLAN.json"), "expectations": binding(RUN / "EXPECTATIONS.json"),
          "checker_input_excludes": ["scene_plan", "dataset_cards", "expected_verdicts", "original_plan_verdicts"],
          "new_sources": 0, "realization_allowed": False, "training_allowed": False,
          "files": [binding(q) for q in sorted(paths)]})
    print(json.dumps({"status": "PREPARED", "frozen_files": len(paths), "probe_claims": {s: len(f["claims"]) for s, f in fixtures.items()}}), flush=True)


class Caller(r.Caller):
    def __init__(self):
        self.run = RUN
        self.limit = 16
        profile, self.factory, self.annotate, self.capacity, dev = r.base.transport()
        self.profile = profile()
        assert self.profile == read(RUN / "provider_profile.json")
        self.tokenizer = dev.tokenizer_from(str(r.base.TOKENIZER))
        self.request_counter = self.http_counter = 0
        self.stage = "initial_support_19053"

    def call(self, label, payload, proof):
        assert self.request_counter < 8, "logical_request_budget_exhausted"
        return super().call(label, payload, {**proof, "regression_stage": self.stage,
                             "regression_only": True, "fresh_source": False,
                             "fixed_candidate_sha256": sha(RUN / "FIXED_PLAN.json")})


def expectations_result(short, result):
    expected = read(RUN / "EXPECTATIONS.json")["cases"][short]
    by_id = {row["claim_id"]: row for row in result["checks"]} if result else {}
    rows = [{"claim_id": identifier, "expected_allowed": allowed,
             "actual": by_id.get(identifier, {}).get("verdict"),
             "pass": by_id.get(identifier, {}).get("verdict") in allowed}
            for identifier, allowed in expected["required_verdicts"].items()]
    return {"all_required_pass": all(row["pass"] for row in rows), "required_controls": rows,
            "diagnostic_only": [by_id.get(identifier) for identifier in expected["diagnostic_only"]]}


def support(caller, short, fixture, label):
    caller.stage = label
    out = hs.run_check(caller, RUN / label, fixture["root_id"], fixture["history"], fixture["claims"], max_attempts=2)
    write(RUN / (label + ".json"), out)
    return out


def review_fixed_candidate(caller, support_result):
    data = read(RUN / "inputs/19053.json")["context"]
    fixture = read(RUN / "fixtures/19053.json")
    scene = read(RUN / "FIXED_PLAN.json")["scenes"][0]
    ids = set(fixture["actual_review_claim_ids"])
    semantic = {"kind": "historical_claim_evidence_support_feedback",
        "previous_review": read(RUN / "BASELINE_REVIEW.json"),
        "actual_historical_claims": [row for row in fixture["claims"] if row["claim_id"] in ids],
        "actual_claim_support_checks": [row for row in support_result["result"]["checks"] if row["claim_id"] in ids],
        "complete_checker_result_sha256": r.base.digest(support_result["result"]),
        "checker_receipt": binding(RUN / "initial_support_19053.json"),
        "instruction": "Reassess the entire unchanged plan against the full unchanged history and assigned card. "
            "These checker records are fallible semantic feedback about original historical claims, not verified truth or an instruction to pass. "
            "Reconsider every unsupported or uncertain claim and its citations; do not retain a claim as historical fact merely because the previous review said so. "
            "Do not rewrite the history, plan or card. Introduce no new historical assertion without adequate actual historical evidence. "
            "A coherent new event can establish new facts in new dialogue: absence from old history alone is not a plan defect. "
            "Preserve or state any genuine negative finding; return the full review, not a patch."}
    write(RUN / "SEMANTIC_REVIEW_FEEDBACK.json", semantic)
    caller.stage = "fixed_plan_event_rereview"
    repair = semantic
    attempts = []
    for attempt in [1, 2]:
        payload = r.event_contract.build_request(data["root_id"], data["route"], scene,
            data["cumulative_history"], data["dataset_cards"], data["latest_state"], repair=repair)
        proof = {"role": "event_state_extractor", "root_id": data["root_id"], "scene_id": scene["scene_id"],
                 "scene_plan_sha256": r.base.digest(scene), "history_sha256": r.base.digest(data["cumulative_history"]),
                 "semantic_feedback_sha256": r.base.digest(semantic), "structure_attempt": attempt}
        raw, directory = caller.call(f"{data['root_id']}-fixed-plan-event-rereview-{attempt}", payload, proof)
        # The first call is a new semantic review. Only a format retry applies the old correction guard.
        correction = r.event_contract.correction_progress(raw, repair if attempt > 1 else None)
        write(directory / "CORRECTION_PROGRESS.json", correction)
        try:
            value = r.event_contract.validate(raw, data["root_id"], data["route"], scene,
                                             data["cumulative_history"], cards=data["dataset_cards"])
            r.event_contract.require_correction_progress(correction)
            write(directory / "DERIVED_COVERAGE.json", r.event_contract.derive_coverage(value, data["cumulative_history"]))
            write(directory / "VALIDATION.json", {"status": "PASS", "semantic_certified": False,
                  "requires_replan": r.event_contract.requires_replan(value)})
            attempts.append({"directory": str(directory), "status": "PASS"})
            write(RUN / "NEW_EVENT_REVIEW.json", value)
            return {"result": value, "attempts": attempts}
        except (ValueError, TypeError) as error:
            repair = r.event_contract.build_repair(error, raw, history=data["cumulative_history"])
            repair["original_semantic_support_feedback"] = semantic
            write(directory / "VALIDATION.json", {"status": "REJECT", "reason": str(error), "errors": repair["errors"]})
            attempts.append({"directory": str(directory), "status": "REJECT", "errors": repair["errors"]})
    return {"result": None, "attempts": attempts}


def live():
    manifest = read(RUN / "REGRESSION_MANIFEST.json")
    for row in manifest["files"]:
        assert sha(row["path"]) == row["sha256"], row["path"]
    gate = read(BASE / "evidence/REGRESSION_PREFLIGHT_AUDIT.json")
    assert gate["status"] == "PASS" and gate["manifest_sha256"] == sha(RUN / "REGRESSION_MANIFEST.json")
    with (RUN / "LIVE_STARTED.json").open("x") as stream:
        json.dump({"at_utc": now()}, stream)
    caller = Caller()
    out = {"initial_support": {}, "control_results": {}, "event_rereview": None,
           "post_rereview_support": None, "support_endpoint_passed": False, "terminal_failure": None}
    try:
        for short in ["19053", "25652"]:
            fixture = read(RUN / f"fixtures/{short}.json")
            checked = support(caller, short, fixture, "initial_support_" + short)
            out["initial_support"][short] = checked
            evaluated = expectations_result(short, checked["result"])
            out["control_results"][short] = evaluated
            write(RUN / ("CONTROL_RESULT_" + short + ".json"), evaluated)
        if not all(value["all_required_pass"] for value in out["control_results"].values()):
            out["terminal_failure"] = "initial_support_or_clear_control_gate_failed"
        else:
            reviewed = review_fixed_candidate(caller, out["initial_support"]["19053"])
            out["event_rereview"] = reviewed
            if reviewed["result"] is None:
                out["terminal_failure"] = "event_rereview_structure_failed_after_limit"
            else:
                original = read(RUN / "fixtures/19053.json")
                fixture = {"root_id": original["root_id"], "history": original["history"],
                           "claims": actual_claims(reviewed["result"])}
                write(RUN / "POST_REVIEW_CLAIMS.json", fixture)
                checked = support(caller, "19053", fixture, "post_rereview_support")
                out["post_rereview_support"] = checked
                if checked["result"] is None:
                    out["terminal_failure"] = "post_rereview_support_structure_failed"
                elif not checked["assessment"]["all_supported"]:
                    out["terminal_failure"] = "new_review_historical_claim_not_supported"
                elif r.event_contract.requires_replan(reviewed["result"]):
                    out["terminal_failure"] = "new_review_requires_plan_revision"
                else:
                    out["support_endpoint_passed"] = True
    except (RuntimeError, ValueError, TypeError) as error:
        out["terminal_failure"] = f"{type(error).__name__}:{error}"
    for row in manifest["files"]:
        assert sha(row["path"]) == row["sha256"], row["path"]
    write(RUN / "REPORT.json", {"version": "v102_history_claim_probe_results_v1", "status": "COMPLETE",
          "result": out, "logical_requests": caller.request_counter, "http_attempts": caller.http_counter,
          "semantic_certified": False, "fresh_yield_measured": False,
          "new_sources": 0, "new_dialogue_turns": 0, "ready_sources": 0, "new_questions": 0,
          "solver_calls": 0, "training_jobs": 0})
    print(json.dumps({"status": "COMPLETE", "logical_requests": caller.request_counter,
          "http_attempts": caller.http_counter, "support_endpoint_passed": out["support_endpoint_passed"],
          "terminal_failure": out["terminal_failure"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "live"])
    args = parser.parse_args()
    prepare() if args.action == "prepare" else live()
