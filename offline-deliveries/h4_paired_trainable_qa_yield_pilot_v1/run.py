"""Execute the frozen paired pilot without repair or retries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import a_arm
import b_arm
import review_gate
from common import HERE, SOURCES, call_qwen, write_once, sha_bytes
from prepare import RUN, verify


def _view(root_id: str) -> dict[str, Any]:
    return json.loads((RUN / "frozen" / root_id / "surface.json").read_text(encoding="utf-8"))


def _failed_parse(path: Path, error: Exception) -> None:
    write_once(path, {"status": "INVALID_RESPONSE_NO_RETRY", "error_type": type(error).__name__,
                      "reason": str(error)[:300], "training_promoted": False})


def generate() -> None:
    manifest = verify()
    for root_id in manifest["roots"]:
        view = _view(root_id)
        for arm in manifest["roots"][root_id]["arm_order"]:
            if arm == "A":
                for question_type in ("multi_select", "ordering"):
                    stage = RUN / "roots" / root_id / "A" / question_type
                    prompt = manifest["roots"][root_id]["prompts"][f"A_{question_type}"]
                    print(json.dumps({"root": root_id, "arm": arm, "stage": question_type}), flush=True)
                    try:
                        response = call_qwen(stage, Path(prompt["path"]).read_text(encoding="utf-8"),
                                             frozen_prompt_sha256=prompt["sha256"])
                    except Exception as error:
                        print(json.dumps({"root": root_id, "arm": arm, "stage": question_type,
                                          "status": "CALL_FAILED", "error_type": type(error).__name__}), flush=True)
                        continue
                    audit = a_arm.validate(response["raw"], view, question_type)
                    write_once(stage / "shape_audit.json", audit)
            else:
                stage = RUN / "roots" / root_id / "B" / "extraction"
                prompt = manifest["roots"][root_id]["prompts"]["B_extraction"]
                print(json.dumps({"root": root_id, "arm": arm, "stage": "extraction"}), flush=True)
                try:
                    response = call_qwen(stage, Path(prompt["path"]).read_text(encoding="utf-8"),
                                         frozen_prompt_sha256=prompt["sha256"])
                except Exception as error:
                    print(json.dumps({"root": root_id, "arm": arm, "stage": "extraction",
                                      "status": "CALL_FAILED", "error_type": type(error).__name__}), flush=True)
                    continue
                try:
                    validation = b_arm.validate_extraction(response["raw"], view)
                except (TypeError, ValueError, KeyError) as error:
                    _failed_parse(stage / "validation_failure.json", error)
                    continue
                write_once(stage / "validated.json", validation)
                write_once(stage / "review_template.json", b_arm.review_template(validation))


def compose_b(root_id: str) -> None:
    manifest = verify()
    if root_id not in manifest["roots"]:
        raise ValueError("root_not_frozen")
    view = _view(root_id)
    root = RUN / "roots" / root_id / "B"
    extraction = root / "extraction"
    validated = json.loads((extraction / "validated.json").read_text(encoding="utf-8"))
    raw = (extraction / "response_raw.txt").read_text(encoding="utf-8")
    if b_arm.validate_extraction(raw, view) != validated:
        raise RuntimeError("validated_extraction_differs_from_raw_response")
    review = json.loads((extraction / "review.json").read_text(encoding="utf-8"))
    accepted = b_arm.select_accepted(validated, review)
    pool = b_arm.build_pool(accepted, view)
    write_once(root / "accepted_records.json", accepted)
    write_once(root / "pool.json", pool)
    for question_type in ("multi_select", "ordering"):
        stage = root / question_type
        reason = pool[f"{question_type}_abstain_reason"]
        if reason:
            write_once(stage / "abstention.json", {"status": "ABSTAIN_BEFORE_PAID_COMPOSITION",
                                                     "reason": reason, "questions": [],
                                                     "training_promoted": False})
            print(json.dumps({"root": root_id, "arm": "B", "stage": question_type,
                              "status": "ABSTAIN_BEFORE_PAID_COMPOSITION", "reason": reason}), flush=True)
            continue
        prompt = b_arm.composition_prompt(pool, question_type)
        digest = sha_bytes(prompt.encode("utf-8"))
        write_once(root / f"{question_type}_composition_freeze.json", {
            "pool_sha256": sha_bytes((root / "pool.json").read_bytes()),
            "prompt_sha256": digest, "source_sha256": view["source_sha256"],
            "review_sha256": sha_bytes((extraction / "review.json").read_bytes()),
            "maximum_http_attempts": 1, "training_promoted": False,
        })
        print(json.dumps({"root": root_id, "arm": "B", "stage": question_type}), flush=True)
        try:
            response = call_qwen(stage, prompt, frozen_prompt_sha256=digest)
        except Exception as error:
            print(json.dumps({"root": root_id, "arm": "B", "stage": question_type,
                              "status": "CALL_FAILED", "error_type": type(error).__name__}), flush=True)
            continue
        try:
            candidate = b_arm.render_selection(response["raw"], pool, question_type)
        except (TypeError, ValueError, KeyError) as error:
            _failed_parse(stage / "render_failure.json", error)
            continue
        write_once(stage / "rendered.json", candidate)


def _review_source(stage: Path, arm: str, surface: dict[str, Any],
                   question_type: str) -> tuple[bytes, dict[str, Any] | None] | None:
    source = stage / ("response_raw.txt" if arm == "A" else "rendered.json")
    if not source.exists():
        return None
    artifact = source.read_bytes()
    if arm == "A":
        audit = json.loads((stage / "shape_audit.json").read_text(encoding="utf-8"))
        if a_arm.validate(artifact.decode("utf-8"), surface, question_type) != audit:
            raise RuntimeError("a_shape_audit_differs_from_raw_response")
    else:
        pool_path = stage.parent / "pool.json"
        pool = json.loads(pool_path.read_text(encoding="utf-8"))
        response = (stage / "response_raw.txt").read_text(encoding="utf-8")
        if b_arm.render_selection(response, pool, question_type) != json.loads(artifact):
            raise RuntimeError("b_rendered_differs_from_raw_selection")
        audit = None
    return artifact, audit


def prepare_reviews() -> None:
    manifest = verify()
    for root_id in manifest["roots"]:
        surface = _view(root_id)
        for arm in ("A", "B"):
            for question_type in ("multi_select", "ordering"):
                stage = RUN / "roots" / root_id / arm / question_type
                source = _review_source(stage, arm, surface, question_type)
                if source is None:
                    continue
                artifact, audit = source
                inputs = review_gate.prepare_review_inputs(
                    artifact, surface, manifest["question_bank_sha256"], audit)
                write_once(stage / "review_packets.json", inputs)
                write_once(stage / "candidate_review_template.json", {
                    "reviews": [review_gate.review_template(packet) for packet in inputs["packets"]],
                    "skipped": inputs["skipped"],
                })
                print(json.dumps({"root": root_id, "arm": arm, "stage": question_type,
                                  "review_packets": len(inputs["packets"]),
                                  "shape_skipped": len(inputs["skipped"])}), flush=True)


def finalize_reviews() -> None:
    manifest = verify()
    for root_id in manifest["roots"]:
        surface = _view(root_id)
        for arm in ("A", "B"):
            for question_type in ("multi_select", "ordering"):
                stage = RUN / "roots" / root_id / arm / question_type
                source = _review_source(stage, arm, surface, question_type)
                if source is None:
                    continue
                artifact, audit = source
                inputs = review_gate.prepare_review_inputs(
                    artifact, surface, manifest["question_bank_sha256"], audit)
                if inputs != json.loads((stage / "review_packets.json").read_text(encoding="utf-8")):
                    raise RuntimeError(f"review_packets_changed:{root_id}:{arm}:{question_type}")
                if not inputs["packets"]:
                    reviewed = []
                else:
                    submitted = json.loads((stage / "candidate_review.json").read_text(encoding="utf-8"))
                    if not isinstance(submitted, dict) or set(submitted) != {"reviews", "skipped"}:
                        raise ValueError("candidate_review_container_invalid")
                    if submitted["skipped"] != inputs["skipped"] or len(submitted["reviews"]) != len(inputs["packets"]):
                        raise ValueError("candidate_review_count_or_skips_invalid")
                    reviewed = []
                    for packet, review in zip(inputs["packets"], submitted["reviews"]):
                        result = review_gate.validate_review(packet, review, artifact, surface,
                                                              manifest["question_bank_sha256"], audit)
                        reviewed.append({"packet": packet, "review": review, "result": result})
                write_once(stage / "candidate_review_validated.json",
                           review_gate.aggregate_reviews(reviewed, inputs["skipped"]))
                print(json.dumps({"root": root_id, "arm": arm, "stage": question_type,
                                  "reviewed": len(reviewed), "pass_to_m0": sum(
                                      row["result"]["status"] == "PASS_TO_M0" for row in reviewed)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "compose-b", "prepare-reviews", "finalize-reviews"))
    parser.add_argument("--root", choices=tuple(SOURCES))
    args = parser.parse_args()
    if args.command == "generate":
        if args.root:
            parser.error("generate runs both frozen roots in the predeclared order")
        generate()
    elif args.command == "compose-b":
        if not args.root:
            parser.error("compose-b requires --root")
        compose_b(args.root)
    elif args.command == "prepare-reviews":
        if args.root:
            parser.error("prepare-reviews runs both frozen roots")
        prepare_reviews()
    else:
        if args.root:
            parser.error("finalize-reviews runs both frozen roots")
        finalize_reviews()


if __name__ == "__main__":
    main()
