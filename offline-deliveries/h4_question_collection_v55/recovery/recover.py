"""Historical v54 raw-draft recovery under v55 cardinality; no new generation."""
from __future__ import annotations

import argparse
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
DELIVERIES = HERE.parent.parent
CODE = DELIVERIES / "h4_sequential_ab_v55_qa"
OLD = DELIVERIES / "h4_sequential_ab_v54_qa/runs/root135466_rules4_01_generation"
ROOT = "cc-train-episode-135466"
REQUEST = OLD / "requests/001-135466-merged-multi-generate-1"
SOURCE = OLD / "sources" / (ROOT + ".json")
SOLVER = HERE / "solver"
sys.path.insert(0, str(CODE))
import evaluate_and_pool as ep

read, require = ep.read, ep.require
sha = ep.assembler.sha
VERSION = "v55_historical_v54_cardinality_recovery_v1"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def scope():
    return {"historical_re_evaluation": True, "new_generation_calls": 0,
            "source_admission_status": "PROVISIONAL_CANDIDATE_HISTORY",
            "training_allowed": False, "training_started": False,
            "training_calls": 0, "ground_calls": 0,
            "codex_question_edits": False, "gold_rewritten": False}


def replay():
    response, request, source = read(REQUEST / "RESPONSE.json"), read(REQUEST / "REQUEST.json"), read(SOURCE)
    raw = json.loads(response["text"])
    ids = [q["candidate_id"] for q in raw["questions"]]
    require(len(ids) == 2 and all(i.startswith(ROOT) for i in ids), "wrong_historical_drafts")
    context = json.JSONDecoder().raw_decode(request["messages"][0]["content"].split("\nINPUT_JSON\n", 1)[1])[0]
    require(context["source_turns"] == source["turns"] and context["candidate_ids"] == ids,
            "historical_request_context_changed")
    require(response["metadata"]["raw_response"]["choices"][0]["message"]["content"] == response["text"],
            "historical_raw_http_response_changed")
    questions, rejected, abstentions, error = ep.generation.validate_draft(
        response["text"], ids, ROOT, "multi_select", source, SOURCE, prompt_arm="current")
    require(len(questions) == 2 and not rejected and not abstentions and error is None,
            "corrected_replay_failed")
    report = read(OLD / "REPORT.json")
    old_rejects = {r["candidate_id"]: r for group in report["roots"] if group["question_type"] == "multi_select"
                   for r in group["mechanical_rejected"]}
    prior = read(OLD / "prior" / (ROOT + ".json"))
    kept, duplicates = ep.v15.dedup.deduplicate(questions, prior, [])
    require(kept == questions and not duplicates, "historical_exact_set_duplicate")
    for q, draft in zip(questions, raw["questions"]):
        require(old_rejects[q["candidate_id"]]["errors"] == ["current_multi_requires_2_to_4_raw_supports:5"],
                "historical_rejection_not_only_wrong_cardinality")
        require(q["raw_model_candidate"] == draft and len(q["gold"]) == 5,
                "raw_question_or_gold_changed")
        q.update(generator_model=ep.generation.MODEL,
                 generation_request_path=str(REQUEST / "REQUEST.json"),
                 generation_response_path=str(REQUEST / "RESPONSE.json"),
                 raw_question_content_preserved=True, source_context_version=read(OLD / "MANIFEST.json")["version"],
                 historical_recovery_version=VERSION, historical_generation_run=str(OLD),
                 source_training_pool_eligible=False, source_semantic_certified=False,
                 upstream_scene_promotion=False, upstream_admitted_scenes=0, **scope())
    return questions


def prepare():
    require(not (HERE / "MANIFEST.json").exists(), "immutable_recovery_already_prepared")
    # Historical prompt verification runs in a clean interpreter so its frozen
    # v54 modules cannot accidentally be replaced by the v55 imports above.
    historical_check = """import sys,json
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,sys.argv[1])
import evaluate_and_pool as old
run=Path(sys.argv[2]); request=Path(sys.argv[3])
old.generation.verify(run)
chain=old.Inspector().feedback_request(request,run,'cc-train-episode-135466')
print(json.dumps({'status':'PASS','historical_generation_verified':True,'historical_actual_http_payload_verified':True,'historical_frozen_feedback_verified':True,'chain':chain}))
"""
    checked = subprocess.run([sys.executable, "-B", "-c", historical_check,
                              str(OLD.parent.parent), str(OLD), str(REQUEST / "REQUEST.json")],
                             capture_output=True, text=True, check=True)
    historical = json.loads(checked.stdout)
    write(HERE / "HISTORICAL_PROVENANCE_VERIFICATION.json", historical)
    questions = replay()
    provenance = {"requested_model": ep.generation.MODEL, "returned_model": ep.generation.MODEL,
                  "origin": "historical_v54_raw_response_re_evaluation",
                  "original_generation_run": str(OLD), "corrected_cardinality": [2, 3, 4, 5, 6], **scope()}
    write(HERE / "REPLAY_POOL.json", {"version": VERSION, "questions": questions, "provenance": provenance,
                                       "candidate_count": 2, **scope()})
    write(HERE / "RECOVERY_POOL.json", {"version": VERSION, "questions": questions[:1], "provenance": provenance,
                                         "candidate_count": 1, **scope()})
    notes = [
        {"candidate_id": questions[0]["candidate_id"], "mechanical_replay": "PASS", "correct_options": 5,
         "recovery_decision": "ELIGIBLE_FOR_SOLVER", "scope": "call_to_friend_and_immediate_aftermath",
         "supported_options": {"A": "A:S01:T001 explicitly reports the call last night.",
             "B": "A:S01:T003 explicitly reports no longer keeping the secret.",
             "C": "A:S01:T005 reports anger, yelling and betrayal accusation.",
             "D": "A:S01:T007 reports selfishness accusation.",
             "E": "A:S01:T009 explicitly expresses worry friendship is over."},
         "false_option": {"F": "A:S01:T005 anger and yelling contradict a calm accepting reaction."},
         "nonblocking_wording_note": "Female pronouns are not independently established by the cited first-person turns; core speaker/action attribution is clear. Original wording is preserved.",
         "core_semantic_findings": []},
        {"candidate_id": questions[1]["candidate_id"], "mechanical_replay": "PASS", "correct_options": 5,
         "recovery_decision": "EXCLUDED_KNOWN_OVERBROAD_SUPPORT", "problem_option": "B",
         "proposition": questions[1]["options"]["B"],
         "source_turn": "CANDIDATE:B:S03:T005",
         "source_text": next(t["text"] for t in read(SOURCE)["turns"] if t["turn_ref"] == "CANDIDATE:B:S03:T005"),
         "reason": "Any area is fine does not establish willingness to visit any campus; source also asks for a good place. The campus quantifier is stronger than the source.",
         "not_excluded_for_five_correct": True, "manual_rewrite": False, "solver_calls_planned": 0}]
    write(HERE / "CONTENT_AUDIT.json", {"reviewer": "independent Codex source inspection", "questions": notes,
                                         "new_global_gate": False, **scope()})
    paths = {Path(__file__), OLD / "MANIFEST.json", OLD / "REPORT.json", SOURCE,
             OLD / "prior" / (ROOT + ".json"), REQUEST / "REQUEST.json", REQUEST / "RESPONSE.json",
             REQUEST / "PROOF.json", REQUEST / "HTTP_ATTEMPT.json"}
    paths.update(Path(r["path"]) for r in read(OLD / "MANIFEST.json")["files"])
    paths.update(CODE.glob("*.py"))
    paths.update(HERE.glob("*.json"))
    write(HERE / "MANIFEST.json", {"version": VERSION, "status": "RECOVERY_PREPARED_NO_NEW_GENERATION",
        "historical_drafts_replayed": 2, "eligible_for_solver": 1, "excluded_content_issue": 1,
        "full_calls_planned": 18, "files": [{"path": str(p), "sha256": sha(p)} for p in sorted(paths)], **scope()})
    return verify()


def verify():
    manifest = read(HERE / "MANIFEST.json")
    ep.base.verify_bindings(manifest["files"])
    questions = replay()
    require(read(HERE / "REPLAY_POOL.json")["questions"] == questions, "replay_pool_changed")
    require(read(HERE / "RECOVERY_POOL.json")["questions"] == questions[:1], "eligible_pool_changed")
    if (HERE / "RESULT_BINDINGS.json").exists():
        ep.base.verify_bindings(read(HERE / "RESULT_BINDINGS.json")["files"])
    return {"status": "PASS", "historical_drafts": 2, "solver_eligible": 1, "full_calls_planned": 18, **scope()}


def preflight():
    verify()
    ep.evaluator.preflight(HERE / "RECOVERY_POOL.json", SOLVER,
                           require_ordering_review=True, require_model=True, solver_model=ep.assembler.TARGET)
    ep._scope(read(SOLVER / "manifest.json"))
    return {"status": "PREFLIGHT_PASS", "questions": 1, "full_calls": 18, **scope()}


def live():
    verify()
    ep._scope(read(SOLVER / "manifest.json"))
    ep.evaluator.live(SOLVER)
    return read(SOLVER / "COMPLETE.json")


def finalize():
    verify()
    require((SOLVER / "COMPLETE.json").exists(), "solver_incomplete")
    if not (SOLVER / "RESULTS.json").exists():
        ep.evaluator.report(SOLVER)
    questions = read(HERE / "RECOVERY_POOL.json")["questions"]
    manifest, records, _ = ep.assembler.Inspector().solver(SOLVER)
    results = read(SOLVER / "RESULTS.json")
    selected = ep.verify_scores(questions, manifest, records, results)
    write(HERE / "SELECTED_QUESTIONS.json", {"version": VERSION, "questions": selected,
                                             "selected_count": len(selected), **scope()})
    views = ep.display_export.write_views(HERE, selected)
    scores = []
    for q in questions:
        rows = [r for r in records if r["candidate_id"] == q["candidate_id"]]
        partial = sum((Fraction(r["score"]["partial_score_fraction"]) for r in rows), Fraction()) / len(rows)
        exact = sum(r["score"]["exact_match"] for r in rows)
        scores.append({"candidate_id": q["candidate_id"], "correct_options": len(q["gold"]),
                       "full_calls": len(rows), "exact_correct": exact, "exact_accuracy": exact / len(rows),
                       "partial_mean_fraction": str(partial), "partial_mean": float(partial),
                       "selected_by_policy": q["candidate_id"] in results["selected_questions"]})
    summary = {"version": VERSION, "historical_drafts": 2, "eligible": 1, "excluded_content_issue": 1,
               "evaluated": len(questions), "selected": len(selected), "solver_calls": len(records),
               "scores": scores, "display_stats": ep.display_export.display_stats(views), **scope()}
    write(HERE / "RESULT_SUMMARY.json", summary)
    write(HERE / "RESULT_BINDINGS.json", {"files": [{"path": str(p), "sha256": sha(p)}
          for p in sorted(HERE.rglob("*")) if p.is_file() and p.name not in {"RESULT_BINDINGS.json"}
          and p.suffix != ".log"]})
    verify()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify", "preflight", "live", "finalize"))
    args = parser.parse_args()
    print(json.dumps(globals()[args.action](), ensure_ascii=False, indent=2))
