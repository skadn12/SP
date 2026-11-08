#!/usr/bin/env python3
"""Unified NEW-protocol source runner. Historical OET1 outputs are read-only.

Run --help for preflight, read-only review validation, scope receipts, or future
production. Actual model execution requires --execute-model-calls and a NEW run
directory under this repair directory. No model is called by other subcommands.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import traceback

from production_core import (CHECKS, SCENES, EvidenceResolver, ProductionError, bind,
                             build_cumulative, make_turns, now, read, require,
                             safe_log, validate_review, verify, write)

HERE = Path(__file__).resolve().parent
OET1 = HERE.parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    previous_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_setting
    return module


def owned_output(path):
    path = Path(path).resolve()
    require(path != HERE and path.is_relative_to(HERE), "OUTPUT_OUTSIDE_NEW_REPAIR_DIRECTORY", path=str(path))
    return path


def seal_equal(path, value):
    """Resume only exact deterministic artifacts, without replacing old bytes."""
    path = Path(path)
    if path.exists():
        require(read(path) == value, "EXISTING_SEALED_ARTIFACT_CHANGED", path=str(path))
        return bind(path)
    return write(path, value)


class CodexCallBackend:
    """Frozen transport + provenance guard; output/log destination is new only."""
    def __init__(self, run_dir):
        self.run_dir = owned_output(run_dir)
        self.driver = load_module("oet1_followup_frozen_transport", OET1 / "scripts/production_driver.py")
        self.integrity = load_module("oet1_followup_provenance", OET1 / "scripts/production_integrity.py")
        self.driver.BASE = self.run_dir
        self.driver.log = lambda event_type, **fields: safe_log(self.run_dir / "PROGRESS.jsonl", event_type, **fields)

    def __call__(self, call_id, prompt):
        response, result = self.driver.call(call_id, prompt)
        path = self.run_dir / "cli_calls" / call_id / "RESULT.json"
        checked, _ = self.integrity.validate_call_result(path, prompt)
        require(checked == result and verify(result["parsed_response"]) == response,
                "CALL_RETURN_OR_PARSED_PROVENANCE_DRIFT")
        return response, result


def read_scope_receipt(paths):
    """A controller read/hash receipt is NOT an independent semantic audit."""
    artifacts = []
    for path in paths:
        value = bind(path)
        raw = Path(value["path"]).read_bytes()
        import hashlib
        require(len(raw) == value["bytes"] and hashlib.sha256(raw).hexdigest() == value["sha256"],
                "READ_SCOPE_BYTES_CHANGED")
        item = {"binding": value, "read_all_bytes": True, "readable": True}
        try:
            data = json.loads(raw)
            item["json_parsed"] = True
            if isinstance(data, dict) and isinstance(data.get("turns"), list):
                item["turn_count"] = len(data["turns"])
        except (ValueError, UnicodeDecodeError):
            item["json_parsed"] = False
        artifacts.append(item)
    return {"status": "MECHANICAL_READ_SCOPE_RECEIPT_ONLY", "artifacts": artifacts,
            "independent_review_completed": False, "semantic_audit_pass": False}


class ProductionPipeline:
    """One entrypoint owns prompts, cumulative metadata, validation and admission."""
    def __init__(self, run_dir, call):
        self.run_dir = owned_output(run_dir)
        self.call = call
        self.prompt_driver = load_module("oet1_followup_prompt_templates", OET1 / "scripts/production_driver.py")
        self.integrity = load_module("oet1_followup_result_guard", OET1 / "scripts/production_integrity.py")

    def _checked_call(self, call_id, prompt):
        response, result = self.call(call_id, prompt)
        result_path = self.run_dir / "cli_calls" / call_id / "RESULT.json"
        checked, _ = self.integrity.validate_call_result(result_path, prompt)
        require(checked == result and verify(result["parsed_response"]) == response,
                "CALL_RETURN_OR_PARSED_PROVENANCE_DRIFT")
        return response, result

    def _freeze(self):
        artifacts = [bind(path) for path in (HERE / "production_core.py", Path(__file__),
                     OET1 / "scripts/production_driver.py", OET1 / "scripts/production_integrity.py")]
        value = {"schema_version": "oet1_followup_production_protocol_v1", "code_bindings": artifacts,
                 "source_protocol": "NEW_PROMPT_VERSION_NOT_RETROACTIVE_RECOVERY",
                 "turns_per_scene": 14, "scenes": SCENES, "semantic_attempts_per_scene_role": 1,
                 "semantic_retry": False, "old_frozen_outputs_are_read_only": True,
                 "question_generation_included": False}
        path = self.run_dir / "PROTOCOL.json"
        if path.exists():
            require(read(path) == value, "PROTOCOL_OR_CODE_BINDING_CHANGED_START_NEW_RUN")
        else:
            write(path, value)

    def _prepare_packet(self, root, arm):
        require(arm in ("GENERIC", "TARGETED"), "UNKNOWN_CONDITION")
        require(root in read(OET1 / "PLAN.json")["new_train_roots"], "ROOT_NOT_IN_FROZEN_INPUT_PLAN")
        source_path = OET1 / "sources" / root / arm / "SOURCE_AUTHOR_INPUT.json"
        packet = read(source_path)
        require(packet["root_id"] == root and packet["condition"] == arm, "PACKET_SCOPE_MISMATCH")
        initial = verify(packet["source_binding"])
        require(initial == packet["full_initial_source"], "PACKET_INITIAL_SOURCE_DIFFERS_FROM_BINDING")
        for scene in SCENES[3:]:
            EvidenceResolver(packet, initial, scene)
        unit = self.run_dir / "sources" / root / arm
        packet_path = unit / "SOURCE_AUTHOR_INPUT.json"
        if packet_path.exists():
            require(packet_path.read_bytes() == source_path.read_bytes(), "COPIED_PACKET_CHANGED")
        else:
            packet_path.parent.mkdir(parents=True, exist_ok=True)
            with packet_path.open("xb") as stream:
                stream.write(source_path.read_bytes())
        input_receipt = {"packet_input": bind(source_path), "packet_copy": bind(packet_path),
                         "initial_source": packet["source_binding"],
                         "native_events": [n["native_event_binding"] for n in packet["native_events"]]}
        receipt_path = unit / "INPUT_BINDINGS.json"
        if receipt_path.exists():
            require(read(receipt_path) == input_receipt, "INPUT_BINDINGS_CHANGED")
        else:
            write(receipt_path, input_receipt)
        return packet, unit, packet_path

    def reviewer_prompt(self, packet, source, draft, scene, proof):
        # Both this prompt and validate_review use exactly the same resolver.
        resolver = EvidenceResolver(packet, source, scene)
        return self.prompt_driver.source_reviewer(packet, source, draft, scene) + "\n" + json.dumps({
            "protocol_version": "oet1_followup_production_v1",
            "mechanical_prefix_and_count_proof": proof,
            "reference_contract": resolver.prompt_contract(),
            "review_scope": "Read every cumulative source turn; native evidence is a separate domain, excluded from source_turn_count.",
            "mechanical_proof_is_not_semantic_review": True,
        }, ensure_ascii=False)

    def _validate_accepted(self, sd, packet, previous_path, scene):
        accepted = read(sd / "ACCEPTED.json")
        require(accepted["source"] == bind(sd / "CUMULATIVE_SOURCE.json")
                and accepted["review"] == bind(sd / "REVIEW.json")
                and accepted["validation"] == bind(sd / "REVIEW_VALIDATION.json"), "ACCEPTED_BINDING_CHANGED")
        draft = read(sd / "DRAFT.json")
        source, proof = build_cumulative(packet, read(previous_path), draft, scene,
                                         bind(previous_path), bind(sd / "DRAFT.json"))
        require(source == read(sd / "CUMULATIVE_SOURCE.json") and proof == read(sd / "PREFIX_PROOF.json"),
                "ACCEPTED_SOURCE_OR_PROOF_CHANGED")
        review = read(sd / "REVIEW.json")
        validation = validate_review(review, packet, source, scene)
        require(validation == read(sd / "REVIEW_VALIDATION.json"), "ACCEPTED_VALIDATION_CHANGED")
        require(review["reviewer_id"] != draft["author_id"], "SOURCE_SELF_REVIEW")
        require(review["source_binding"] == accepted["source"] and review["draft_binding"] == bind(sd / "DRAFT.json"),
                "ACCEPTED_REVIEW_BINDING_CHANGED")
        require(accepted["author_result"] == bind(sd / "AUTHOR_RESULT.json")
                and accepted["review_result"] == bind(sd / "REVIEW_RESULT.json"), "ACCEPTED_CALL_RECORD_CHANGED")
        for role in ("AUTHOR", "REVIEW"):
            result = read(sd / f"{role}_RESULT.json")
            raw = verify(result["parsed_response"])
            expected_prompt = (self.prompt_driver.source_author(packet, read(previous_path), scene) if role == "AUTHOR"
                               else self.reviewer_prompt(packet, source, draft, scene, proof))
            result_path = self.run_dir / "cli_calls" / f"{packet['root_id']}__{packet['condition']}__{scene.replace(':', '_')}__{role}" / "RESULT.json"
            checked_result, _ = self.integrity.validate_call_result(result_path, expected_prompt)
            require(checked_result == result, "ACCEPTED_CALL_PROVENANCE_CHANGED")
            if role == "AUTHOR":
                require(make_turns(raw, packet, read(previous_path), scene) == draft["turns"]
                        and result["backend"]["thread_id"] == draft["author_id"], "ACCEPTED_AUTHOR_RAW_CHANGED")
            else:
                require(all(review.get(key) == value for key, value in raw.items())
                        and result["backend"]["thread_id"] == review["reviewer_id"], "ACCEPTED_REVIEW_RAW_CHANGED")

    def run_source(self, root, arm, *, through_scene=SCENES[-1]):
        require(through_scene in SCENES, "UNKNOWN_SCENE")
        self._freeze()
        packet, unit, packet_path = self._prepare_packet(root, arm)
        previous_path = Path(packet["source_binding"]["path"])
        for scene in SCENES[:SCENES.index(through_scene) + 1]:
            sd = unit / scene.replace(":", "_")
            if (sd / "ACCEPTED.json").exists():
                self._validate_accepted(sd, packet, previous_path, scene)
                previous_path = sd / "CUMULATIVE_SOURCE.json"
                continue
            if (sd / "FAILURE.json").exists():
                return {"status": "STOP_EXISTING_FAILURE_NO_RETRY", "scene": scene, "failure": bind(sd / "FAILURE.json")}
            try:
                previous = read(previous_path)
                call_prefix = f"{root}__{arm}__{scene.replace(':', '_')}"
                author_response, author_result = self._checked_call(call_prefix + "__AUTHOR",
                    self.prompt_driver.source_author(packet, previous, scene))
                turns = make_turns(author_response, packet, previous, scene)
                draft = {"schema_version": "oet1_followup_scene_draft_v1", "root_id": root, "scene_ref": scene,
                         "author_id": author_result["backend"]["thread_id"], "created_at_utc": author_result["finished_at_utc"],
                         "input_source_binding": bind(previous_path), "author_input_binding": bind(packet_path),
                         "turns": turns, "cli_raw_binding": author_result["raw_response"], "controller_metadata_only": True}
                if SCENES.index(scene) >= 3:
                    draft["native_event_binding"] = packet["native_events"][SCENES.index(scene) - 3]["native_event_binding"]
                draft_path = sd / "DRAFT.json"
                if draft_path.exists():
                    require(read(draft_path) == draft, "EXISTING_DRAFT_CHANGED")
                else:
                    write(draft_path, draft)
                seal_equal(sd / "AUTHOR_RESULT.json", author_result)
                source, proof = build_cumulative(packet, previous, draft, scene, bind(previous_path), bind(draft_path))
                for name, value in (("CUMULATIVE_SOURCE.json", source), ("PREFIX_PROOF.json", proof)):
                    path = sd / name
                    if path.exists():
                        require(read(path) == value, "EXISTING_MECHANICAL_OUTPUT_CHANGED", path=str(path))
                    else:
                        write(path, value)
                response, result = self._checked_call(call_prefix + "__REVIEW",
                    self.reviewer_prompt(packet, source, draft, scene, proof))
                require(result["backend"]["thread_id"] != draft["author_id"], "SOURCE_SELF_REVIEW")
                # Preserve every raw semantic field even when admission will fail.
                reserved = {"schema_version", "reviewer_id", "source_binding", "draft_binding", "cli_raw_binding"}
                require(not reserved.intersection(response), "REVIEW_RAW_RESERVED_FIELD_COLLISION",
                        fields=sorted(reserved.intersection(response)))
                review = {**response, "schema_version": "oet1_followup_source_review_v1", "reviewer_id": result["backend"]["thread_id"],
                          "source_binding": bind(sd / "CUMULATIVE_SOURCE.json"), "draft_binding": bind(draft_path),
                          "cli_raw_binding": result["raw_response"]}
                for name, value in (("REVIEW.json", review), ("REVIEW_RESULT.json", result)):
                    path = sd / name
                    if path.exists():
                        require(read(path) == value, "EXISTING_REVIEW_OUTPUT_CHANGED", path=str(path))
                    else:
                        write(path, value)
                validation = validate_review(response, packet, source, scene)
                seal_equal(sd / "REVIEW_VALIDATION.json", validation)
                receipt = {"status": "READY", "created_at_utc": now(), "source": bind(sd / "CUMULATIVE_SOURCE.json"),
                           "review": bind(sd / "REVIEW.json"), "validation": bind(sd / "REVIEW_VALIDATION.json"),
                           "author_result": bind(sd / "AUTHOR_RESULT.json"), "review_result": bind(sd / "REVIEW_RESULT.json")}
                write(sd / "ACCEPTED.json", receipt)
                safe_log(self.run_dir / "PROGRESS.jsonl", "SCENE_ACCEPTED", root=root, arm=arm, scene=scene)
                previous_path = sd / "CUMULATIVE_SOURCE.json"
            except Exception as error:
                failure = {"status": "FAIL", "at_utc": now(), "error": str(error),
                           "error_code": getattr(error, "code", type(error).__name__), "traceback": traceback.format_exc(),
                           "semantic_retry": False, "descendants_stopped": True}
                if not (sd / "FAILURE.json").exists():
                    write(sd / "FAILURE.json", failure)
                safe_log(self.run_dir / "PROGRESS.jsonl", "SCENE_FAILED", root=root, arm=arm, scene=scene, **failure)
                return {"status": "STOP_FAILURE", "scene": scene, "failure": bind(sd / "FAILURE.json")}
        if through_scene == SCENES[-1]:
            final = unit / "FINAL_SOURCE.json"
            if final.exists():
                require(final.read_bytes() == previous_path.read_bytes(), "FINAL_SOURCE_CHANGED")
            else:
                with final.open("xb") as stream:
                    stream.write(previous_path.read_bytes())
            return {"status": "SOURCE_COMPLETE", "final_source": bind(final)}
        return {"status": "SOURCE_PREFIX_COMPLETE", "last_scene": through_scene, "source": bind(previous_path)}


def preflight(root, arm):
    require(root in read(OET1 / "PLAN.json")["new_train_roots"] and arm in ("GENERIC", "TARGETED"), "UNKNOWN_INPUT_UNIT")
    packet_path = OET1 / "sources" / root / arm / "SOURCE_AUTHOR_INPUT.json"
    packet = read(packet_path)
    initial = verify(packet["source_binding"])
    require(initial == packet["full_initial_source"], "INITIAL_SOURCE_PACKET_MISMATCH")
    native = [EvidenceResolver(packet, initial, scene).prompt_contract() for scene in SCENES[3:]]
    return {"status": "PASS_MECHANICAL_PREFLIGHT_ONLY", "root_id": root, "condition": arm,
            "source_turn_count": len(initial["turns"]), "packet_binding": bind(packet_path),
            "native_reference_contracts": native,
            "read_scope": read_scope_receipt([packet_path, packet["source_binding"]["path"]] +
                [event["native_event_binding"]["path"] for event in packet["native_events"]]),
            "model_call_executed": False, "sandbox_connectivity_tested": False,
            "independent_semantic_review_completed": False, "gpu_required": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("preflight", "run-source"):
        command = sub.add_parser(name)
        command.add_argument("--root", required=True)
        command.add_argument("--arm", required=True, choices=("GENERIC", "TARGETED"))
        if name == "run-source":
            command.add_argument("--run-dir", required=True, type=Path)
            command.add_argument("--through-scene", choices=SCENES, default=SCENES[-1])
            command.add_argument("--execute-model-calls", action="store_true")
    review = sub.add_parser("inspect-review")
    review.add_argument("--packet", type=Path, required=True)
    review.add_argument("--source", type=Path, required=True)
    review.add_argument("--review", type=Path, required=True)
    review.add_argument("--scene", choices=SCENES, required=True)
    review.add_argument("--evidence-only", action="store_true")
    scope = sub.add_parser("read-scope")
    scope.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "preflight":
            value = preflight(args.root, args.arm)
        elif args.action == "inspect-review":
            value = validate_review(read(args.review), read(args.packet), read(args.source), args.scene,
                                    require_ready=not args.evidence_only)
        elif args.action == "read-scope":
            value = read_scope_receipt(args.paths)
        else:
            require(args.execute_model_calls, "MODEL_EXECUTION_FLAG_REQUIRED")
            run_dir = owned_output(args.run_dir)
            value = ProductionPipeline(run_dir, CodexCallBackend(run_dir)).run_source(args.root, args.arm,
                        through_scene=args.through_scene)
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0 if value.get("status") not in ("STOP_FAILURE", "STOP_EXISTING_FAILURE_NO_RETRY") else 1
    except Exception as error:
        print(json.dumps({"status": "REJECT", "error_code": getattr(error, "code", type(error).__name__),
                          "error": str(error)}, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
