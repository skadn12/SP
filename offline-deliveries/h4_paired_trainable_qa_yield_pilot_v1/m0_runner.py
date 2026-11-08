"""Run the frozen six-display M0 probe on an untouched reviewed candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import a_arm
import b_arm
from common import sha_bytes, sha_file, write_once
from prepare import M0_SCRIPT, RUN, verify
import review_gate


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _same_or_write(path: Path, value: Any) -> None:
    if path.exists():
        if _read(path) != value:
            raise RuntimeError(f"existing_artifact_differs:{path}")
    else:
        write_once(path, value)


def _source_parts(manifest: dict[str, Any], root_id: str) -> tuple[dict[str, Any], Path, Path]:
    frozen = manifest["roots"][root_id]
    surface = _read(Path(frozen["surface_path"]))
    if surface["root_id"] != root_id or surface["source_sha256"] != frozen["source_sha256"]:
        raise RuntimeError("frozen_surface_binding_invalid")
    parent = RUN / "frozen" / root_id / "m0_parent.json"
    child = RUN / "frozen" / root_id / "m0_child.json"
    split = frozen["parent_turn_count"]
    if _read(parent) != {"turns": surface["turns"][:split]}:
        raise RuntimeError("frozen_m0_parent_changed")
    if _read(child) != {"turns": surface["turns"][split:]}:
        raise RuntimeError("frozen_m0_child_changed")
    if not surface["turns"][split:]:
        raise RuntimeError("frozen_m0_child_empty")
    return surface, parent, child


def _artifact(stage: Path, arm: str, question_type: str, surface: dict[str, Any]
              ) -> tuple[bytes, dict[str, Any] | None]:
    if arm == "A":
        artifact = (stage / "response_raw.txt").read_bytes()
        audit = _read(stage / "shape_audit.json")
        if audit != a_arm.validate(artifact.decode("utf-8"), surface, question_type):
            raise RuntimeError("writer_shape_audit_changed")
        return artifact, audit
    pool = _read(stage.parent / "pool.json")
    if pool["root_id"] != surface["root_id"] or pool["source_sha256"] != surface["source_sha256"]:
        raise RuntimeError("relation_pool_source_mismatch")
    raw = (stage / "response_raw.txt").read_text(encoding="utf-8")
    rendered = _read(stage / "rendered.json")
    if rendered != b_arm.render_selection(raw, pool, question_type):
        raise RuntimeError("rendered_candidate_differs_from_selection")
    return (stage / "rendered.json").read_bytes(), None


def admitted_candidate(manifest: dict[str, Any], root_id: str, arm: str,
                       question_type: str, candidate_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Revalidate every stage review and return only the original passing subtree."""
    if root_id not in manifest["roots"] or arm not in {"A", "B"} or question_type not in {"multi_select", "ordering"}:
        raise ValueError("stage_not_frozen")
    surface, parent, child = _source_parts(manifest, root_id)
    stage = RUN / "roots" / root_id / arm / question_type
    artifact_bytes, shape_audit = _artifact(stage, arm, question_type, surface)
    generated = review_gate.prepare_review_inputs(
        artifact_bytes, surface, manifest["question_bank_sha256"], shape_audit)
    aggregate_path = stage / "candidate_review_validated.json"
    aggregate = _read(aggregate_path)
    if (not isinstance(aggregate, dict) or set(aggregate) != {"reviews", "skipped", "summary", "training_promoted"}
            or aggregate["training_promoted"] != 0 or aggregate["skipped"] != generated["skipped"]
            or not isinstance(aggregate["reviews"], list)):
        raise RuntimeError("candidate_review_aggregate_invalid")
    submitted = _read(stage / "candidate_review.json")
    if (not isinstance(submitted, dict) or set(submitted) != {"reviews", "skipped"}
            or submitted["skipped"] != generated["skipped"]
            or len(submitted["reviews"]) != len(generated["packets"])):
        raise RuntimeError("candidate_review_submission_invalid")
    packets = {packet["anonymous_id"]: packet for packet in generated["packets"]}
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, entry in enumerate(aggregate["reviews"]):
        if not isinstance(entry, dict) or set(entry) != {"packet", "review", "result"}:
            raise RuntimeError("candidate_review_entry_invalid")
        packet, review, result = entry["packet"], entry["review"], entry["result"]
        anonymous_id = packet.get("anonymous_id") if isinstance(packet, dict) else None
        if anonymous_id not in packets or anonymous_id in seen or packet != packets[anonymous_id]:
            raise RuntimeError("candidate_review_packet_mismatch")
        if submitted["reviews"][index] != review:
            raise RuntimeError("candidate_review_form_changed")
        verified = review_gate.validate_review(
            packet, review, artifact_bytes, surface, manifest["question_bank_sha256"], shape_audit)
        if verified != result or result["question_type"] != question_type or result["root_id"] != root_id:
            raise RuntimeError("candidate_review_result_mismatch")
        seen.add(anonymous_id)
        results.append(result)
    if seen != set(packets):
        raise RuntimeError("candidate_reviews_incomplete")
    if aggregate != review_gate.aggregate_reviews(aggregate["reviews"], generated["skipped"]):
        raise RuntimeError("candidate_review_summary_mismatch")
    matches = [result for result in results if result["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise ValueError("candidate_id_not_unique_or_unreviewed")
    result = matches[0]
    if result["decision"] != "pass_to_m0" or result["status"] != "PASS_TO_M0":
        raise ValueError("candidate_not_admitted_to_m0")
    if result["artifact_sha256"] != sha_bytes(artifact_bytes):
        raise RuntimeError("candidate_artifact_hash_mismatch")
    candidates = json.loads(artifact_bytes)["questions"]
    question = candidates[result["index"]]
    if question.get("candidate_id") != candidate_id or review_gate._digest(question) != result["question_sha256"]:
        raise RuntimeError("candidate_subtree_hash_mismatch")
    return question, {
        "root_id": root_id, "arm": arm, "question_type": question_type,
        "candidate_id": candidate_id, "candidate_index": result["index"],
        "question_subtree_sha256": result["question_sha256"],
        "artifact_sha256": result["artifact_sha256"],
        "candidate_review_sha256": sha_file(aggregate_path),
        "review_sha256": result["review_sha256"],
        "source_sha256": surface["source_sha256"],
        "surface_sha256": sha_file(Path(manifest["roots"][root_id]["surface_path"])),
        "m0_parent_sha256": sha_file(parent), "m0_child_sha256": sha_file(child),
        "training_promoted": False,
    }


def score_summary(summary: dict[str, Any], plan: dict[str, Any], output: Path,
                  manifest: dict[str, Any]) -> dict[str, Any]:
    """Verify all 12 saved responses before applying the registered band."""
    config = manifest["m0"]
    if config != {"displays": 6, "modes": ["no_context", "full_context"],
                  "seed": 20261004, "no_context_max_correct": 1,
                  "full_context_min_correct": 2, "full_context_max_correct": 4,
                  "minimum_gain": 2}:
        raise RuntimeError("m0_policy_changed")
    requests = plan["requests"]
    results = summary.get("results")
    if (summary.get("status") != "COMPLETE" or plan.get("display_count") != 6
            or plan.get("request_count") != 12 or summary.get("request_count") != 12
            or not isinstance(results, list) or len(results) != 12 or len(requests) != 12):
        raise RuntimeError("m0_requests_incomplete")
    if (summary.get("question_sha256") != plan["question_sha256"]
            or summary.get("historical_source_sha256") != plan["historical_source_sha256"]
            or summary.get("realized_dialogue_sha256") != plan["realized_dialogue_sha256"]
            or plan.get("seed") != config["seed"] or list(plan.get("modes", [])) != config["modes"]
            or plan.get("model") != manifest["m0_identity"]["model_path"]):
        raise RuntimeError("m0_plan_or_source_mismatch")
    correct = {mode: 0 for mode in config["modes"]}
    parse_failures = {mode: 0 for mode in config["modes"]}
    displayed: dict[str, set[str]] = {mode: set() for mode in config["modes"]}
    for index, (request, result) in enumerate(zip(requests, results)):
        path = output / f"solver_probe_response_{index:02d}.json"
        saved = _read(path)
        mode, variant_id = request["mode"], request["variant_id"]
        if (mode not in displayed or request["request_id"] != result.get("request_id")
                or result.get("mode") != mode or result.get("variant_id") != variant_id
                or saved.get("request_id") != request["request_id"] or saved.get("mode") != mode
                or saved.get("variant_id") != variant_id or saved.get("score") != result.get("score")
                or saved.get("display_to_canonical") != request["display_to_canonical"]
                or saved.get("question_sha256") != plan["question_sha256"]
                or saved.get("historical_source_sha256") != plan["historical_source_sha256"]
                or saved.get("realized_dialogue_sha256") != plan["realized_dialogue_sha256"]
                or saved.get("prompt_sha256") != request["prompt_sha256"]
                or sha_bytes(saved["prompt"].encode("utf-8")) != request["prompt_sha256"]
                or variant_id in displayed[mode]):
            raise RuntimeError("m0_response_binding_invalid")
        displayed[mode].add(variant_id)
        score = result["score"]
        if not isinstance(score, dict) or type(score.get("scored")) is not bool:
            raise RuntimeError("m0_score_shape_invalid")
        if score["scored"]:
            if type(score.get("correct")) is not bool:
                raise RuntimeError("m0_scored_correct_invalid")
        elif score.get("correct") is not None:
            raise RuntimeError("m0_parse_failure_correct_not_null")
        correct[mode] += int(score["correct"] is True)
        parse_failures[mode] += int(not score["scored"])
    if (displayed["no_context"] != displayed["full_context"] or len(displayed["no_context"]) != 6
            or any(summary["rates"][mode] != {"correct": correct[mode],
                                               "parse_failures": parse_failures[mode], "total": 6}
                   for mode in displayed)):
        raise RuntimeError("m0_summary_rates_invalid")
    no_context, full_context = correct["no_context"], correct["full_context"]
    reasons = []
    if no_context > config["no_context_max_correct"]:
        reasons.append("no_context_too_easy")
    if full_context < config["full_context_min_correct"]:
        reasons.append("full_context_too_hard")
    if full_context > config["full_context_max_correct"]:
        reasons.append("full_context_too_easy")
    if full_context - no_context < config["minimum_gain"]:
        reasons.append("context_gain_too_small")
    return {"m0_band_pass": not reasons, "failure_reasons": reasons,
            "no_context_correct": no_context, "full_context_correct": full_context,
            "context_gain": full_context - no_context,
            "parse_failures": parse_failures, "display_count": 6,
            "training_promoted": False}


def run(root_id: str, arm: str, question_type: str, candidate_id: str, *, live: bool = False) -> dict[str, Any]:
    manifest = verify()
    question, binding = admitted_candidate(manifest, root_id, arm, question_type, candidate_id)
    output = RUN / "roots" / root_id / arm / question_type / "m0" / ("q_" + sha_bytes(candidate_id.encode("utf-8"))[:16])
    if live and (output / "m0_live_reserved.json").exists():
        raise RuntimeError("m0_live_already_reserved_no_rerun")
    output.mkdir(parents=True, exist_ok=True)
    _same_or_write(output / "m0_question.json", question)
    binding["m0_question_sha256"] = sha_file(output / "m0_question.json")
    _same_or_write(output / "m0_admission.json", binding)
    command = [sys.executable, str(M0_SCRIPT), "--question", str(output / "m0_question.json"),
               "--historical-source", str(RUN / "frozen" / root_id / "m0_parent.json"),
               "--dialogue", str(RUN / "frozen" / root_id / "m0_child.json"),
               "--output-dir", str(output)]
    if live:
        write_once(output / "m0_live_reserved.json", {"question_sha256": binding["m0_question_sha256"],
                                                    "accepted_question_subtree_sha256": binding["question_subtree_sha256"],
                                                    "training_promoted": False})
        command.extend(["--live", "--accepted-sha256", binding["m0_question_sha256"]])
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    process_result = {"exit_code": process.returncode, "stdout": process.stdout,
                      "stderr": process.stderr, "training_promoted": False}
    write_once(output / ("m0_live_process.json" if live else "m0_dry_process.json"), process_result)
    if process.returncode:
        raise RuntimeError(f"m0_probe_failed:{process.returncode}")
    plan = _read(output / "solver_probe_manifest.json")
    if (plan["question_sha256"] != binding["m0_question_sha256"]
            or plan["historical_source_sha256"] != binding["m0_parent_sha256"]
            or plan["realized_dialogue_sha256"] != binding["m0_child_sha256"]
            or plan.get("request_count") != 12):
        raise RuntimeError("m0_plan_binding_invalid")
    if not live:
        return {"status": "DRY_RUN", "root_id": root_id, "arm": arm,
                "candidate_id": candidate_id, "request_count": 12,
                "output_dir": str(output), "training_promoted": False}
    summary = _read(output / "solver_probe_summary.json")
    outcome = score_summary(summary, plan, output, manifest)
    outcome.update({"status": "M0_BAND_PASS" if outcome["m0_band_pass"] else "M0_BAND_FAIL",
                    "root_id": root_id, "arm": arm, "question_type": question_type,
                    "candidate_id": candidate_id, "question_subtree_sha256": binding["question_subtree_sha256"],
                    "m0_question_sha256": binding["m0_question_sha256"]})
    write_once(output / "m0_outcome.json", outcome)
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--arm", required=True, choices=("A", "B"))
    parser.add_argument("--type", required=True, choices=("multi_select", "ordering"))
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--live", action="store_true", help="Run the frozen local GPU model")
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.arm, args.type, args.candidate_id, live=args.live),
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
