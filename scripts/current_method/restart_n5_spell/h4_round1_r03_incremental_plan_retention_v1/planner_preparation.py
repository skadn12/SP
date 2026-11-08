"""Explicit local preparation of new Planner requests; no model transport."""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path

from . import planner_relations, planner_templates
from .planner_candidate_policy import candidate_limit, gate_budget
from .capacity import local_counter, measure, VERSION as CAPACITY_VERSION
from .planner_feedback_source import load_planner_feedback, FILTER_VERSION
from .planning import sha
from .root_scope import ROOT_SCOPE
from .run import offline_network_guard
from .shared_planning import source_continuation_rows
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import build_speaker_registry
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.incremental_round import build_round_history_parent


VERSION = "planner_filtered_feedback_preparation_v1"


def file_identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_verified(identity):
    if not isinstance(identity, dict) or set(identity) != {"path", "sha256"}:
        raise ValueError("planner_preparation_file_identity_required")
    path = Path(identity["path"]).resolve()
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != identity["sha256"]:
        raise ValueError("planner_preparation_file_sha_mismatch:" + str(path))
    return (list(map(json.loads, filter(str.strip, content.decode().splitlines())))
            if path.suffix == ".jsonl" else json.loads(content))


def _seal(value):
    result = deepcopy(value)
    result["bundle_sha256"] = sha(result)
    return result


def _validate_bundle(bundle):
    expected = bundle.get("bundle_sha256")
    if expected != sha({key: value for key, value in bundle.items() if key != "bundle_sha256"}):
        raise ValueError("planner_preparation_bundle_sha_mismatch")
    limit = candidate_limit(bundle)
    template_version = planner_templates.CAP9_VERSION if limit == 9 else planner_templates.VERSION
    if bundle.get("version") != VERSION or bundle.get("template_version") != template_version:
        raise ValueError("planner_preparation_bundle_version_mismatch")
    if limit == 9 and bundle.get("output_contract_version") != planner_relations.CAP9_VERSION:
        raise ValueError("planner_preparation_cap9_output_contract_mismatch")
    if bundle.get("feedback_version") != FILTER_VERSION:
        raise ValueError("planner_preparation_feedback_version_mismatch")
    if not bundle.get("shared_feedback_view"):
        raise ValueError("planner_preparation_inherited_feedback_required")
    if (bundle["shared_feedback_view"] != bundle["feedback_source"]["shared_feedback_view"]
            or bundle["feedback_sha256"] != sha(bundle["shared_feedback_view"])):
        raise ValueError("planner_preparation_feedback_inheritance_mismatch")
    for identity in bundle["input_files"].values():
        read_verified(identity)
    return bundle


def _capacity(built, budget, counter, role):
    check = measure(built["messages"], counter=counter, budget=budget, role=role)
    if not check["fits"]:
        raise ValueError("planner_preparation_capacity_exceeded:" + role)
    return check


def prepare_initial(config, *, counter=None):
    """Build a new request from explicit versioned sources, without selecting anew."""
    with offline_network_guard():
        limit = candidate_limit(config)
        template_version = planner_templates.CAP9_VERSION if limit == 9 else planner_templates.VERSION
        contract_version = planner_relations.CAP9_VERSION if limit == 9 else planner_relations.VERSION
        if (config.get("version") != VERSION
                or config.get("template_version") != template_version
                or config.get("output_contract_version") != contract_version
                or config.get("feedback_version") != FILTER_VERSION):
            raise ValueError("planner_preparation_explicit_versions_required")
        files = deepcopy(config["input_files"])
        required = {"export_manifest", "history_manifest", "run_freeze", "memory_overview",
                    "candidates", "verified_tracks", "capacity_budget"}
        if set(files) != required:
            raise ValueError("planner_preparation_explicit_inputs_required")
        data = {key: read_verified(identity) for key, identity in files.items()}
        root = config["root_id"]
        histories = [row for row in data["history_manifest"] if row["root_id"] == root]
        if len(histories) != 1:
            raise ValueError("planner_preparation_history_not_unique")
        history = histories[0]
        files["history"] = {"path": history["input_dialogue_path"],
                            "sha256": history["input_dialogue_file_sha256"]}
        dialogue = read_verified(files["history"])
        freeze = data["run_freeze"]
        if (freeze["root_id"] != root or freeze["round_index"] != 3
                or Path(freeze["input_path"]).resolve() != Path(files["history"]["path"]).resolve()
                or freeze["input_sha256"] != files["history"]["sha256"]):
            raise ValueError("planner_preparation_memory_history_binding_mismatch")
        source_run = Path(files["run_freeze"]["path"]).resolve().parent
        for key, relative in {"memory_overview": "memory/pre_generation_overview.json",
                              "candidates": "memory/event_candidates.json",
                              "verified_tracks": "event_verifier/materialized_tracks.json"}.items():
            if Path(files[key]["path"]).resolve() != source_run / relative:
                raise ValueError("planner_preparation_memory_run_association:" + key)
        if data["memory_overview"].get("root_id") != root:
            raise ValueError("planner_preparation_memory_root_mismatch")
        feedback = load_planner_feedback(export_manifest=files["export_manifest"]["path"],
            root_id=root, planning_history_identity=history)
        budget = data["capacity_budget"]
        if limit == 9:
            budget = gate_budget(budget, max_new_candidates=limit)
        if budget.get("readiness_version") != CAPACITY_VERSION or "role_capacity" not in budget:
            raise ValueError("planner_preparation_current_capacity_required")
        for role in ("planner", "supplement"):
            if budget["max_output_tokens"][role] != 8192:
                raise ValueError("planner_preparation_output_contract_capacity_mismatch")
        counter = counter or local_counter()
        parent = build_round_history_parent(dialogue, round_index=3)
        parent.pop("prior_expanded_dialogue", None)
        candidates, tracks = data["candidates"]["candidates"], data["verified_tracks"]["verified_tracks"]
        event_ids = [row["memory_event_id"] for row in candidates]
        track_ids = [row["verified_track_ref"] for row in tracks]
        if (len(set(event_ids)) != len(event_ids) or len(set(track_ids)) != len(track_ids)
                or any(row["memory_event_ref"] not in event_ids for row in tracks)):
            raise ValueError("planner_preparation_memory_registry_invalid")
        built = planner_templates.render_initial_planner(
            feedback_view=feedback["shared_feedback_view"], dialogue=dialogue,
            memory_overview=data["memory_overview"], memory_candidates=candidates,
            verified_tracks=tracks, speaker_registry=build_speaker_registry(parent), max_new_candidates=limit)
        append_enabled = config.get("append_enabled", True)
        if type(append_enabled) is not bool:
            raise ValueError("planner_preparation_append_enabled_boolean_required")
        bundle = _seal({"version": VERSION, "template_version": template_version,
            **({"candidate_policy_version": config["candidate_policy_version"]} if limit == 9 else {}),
            "output_contract_version": contract_version, "feedback_version": FILTER_VERSION,
            "planner_entry_mode": "initial_only" if not append_enabled else "initial_plus_optional_append",
            "append_enabled": append_enabled,
            "identity": {"root_id": root, "round_index": 3}, "input_files": files,
            "parent": parent, "dialogue": dialogue, "candidates": candidates,
            "verified_tracks": tracks, "memory_overview": data["memory_overview"],
            "capacity_budget": budget, "feedback_source": feedback,
            "shared_feedback_view": deepcopy(feedback["shared_feedback_view"]),
            "feedback_sha256": sha(feedback["shared_feedback_view"]),
            "request_sha256": sha(built["messages"]), "api_call_authorized": False,
            "status": "INITIAL_REQUEST_PREPARED_NOT_GENERATED"})
        return {"bundle": bundle, "request": built,
                "capacity": _capacity(built, budget, counter, "planner"), "external_calls": 0}


def accept_initial_output(initial_bundle, raw):
    """Bind a supplied response to the new bundle; this function never generates it."""
    _validate_bundle(initial_bundle)
    compiled = planner_relations.compile_initial_output(raw, parent=initial_bundle["parent"],
        candidates=initial_bundle["candidates"], verified_tracks=initial_bundle["verified_tracks"],
        max_new_candidates=candidate_limit(initial_bundle))
    if not compiled["valid"]:
        raise ValueError("planner_preparation_initial_output_invalid:" + str(compiled["errors"]))
    prepared = compiled["prepared"]
    prepared.update(eligible_source_candidate_ids=list(prepared["candidate_registry"]),
        source_scope={"scope_mode": ROOT_SCOPE, **initial_bundle["identity"],
            "history_sha256": sha(initial_bundle["parent"]),
            "original_plan_sha256": prepared["original_sha256"],
            "candidate_registry_sha256": sha(prepared["candidate_registry"]),
            "semantic_source_alignment": "UNKNOWN"},
        _root_materialization_inputs={key: deepcopy(initial_bundle[key])
                                      for key in ("parent", "candidates", "verified_tracks")})
    return compiled


def prepare_supplement(initial_bundle, raw_initial_output, *, counter=None):
    """Inherit frozen feedback exactly; never reload/reselect it in the supplement."""
    with offline_network_guard():
        if candidate_limit(initial_bundle) == 9:
            raise ValueError("candidate_cap9_initial_only_no_supplement")
        compiled = accept_initial_output(initial_bundle, raw_initial_output)
        prepared = compiled["prepared"]
        inputs = {"identity": initial_bundle["identity"], "original_plan_version": "original_v1",
            "original_plan": prepared["original"], "original_plan_sha256": prepared["original_sha256"],
            "stable_candidates": prepared["candidate_registry"],
            "authoritative_history": initial_bundle["parent"]["dialogue"]["turns"],
            "retained_actual_r03_dialogue": [], "source_scope": prepared["source_scope"],
            "eligible_source_candidate_ids": prepared["eligible_source_candidate_ids"],
            "speaker_registry": build_speaker_registry(initial_bundle["parent"]),
            "memory_navigation": initial_bundle["candidates"],
            "budget": initial_bundle["capacity_budget"],
            "shared_feedback_view": deepcopy(initial_bundle["shared_feedback_view"]),
            "initial_relations": compiled["relation_sidecar"]}
        built = planner_templates.render_supplement(inputs,
            continuation_source_templates=source_continuation_rows(prepared))
        counter = counter or local_counter()
        return {"initial_bundle_sha256": initial_bundle["bundle_sha256"], "request": built,
                "inputs": inputs, "initial_compilation": compiled,
                "feedback_sha256": sha(built["feedback_view"]), "feedback_reselected": False,
                "capacity": _capacity(built, initial_bundle["capacity_budget"], counter, "supplement"),
                "external_calls": 0}


def prepare_from_config(config, *, raw_initial_output=None, raw_supplement_output=None, counter=None):
    with offline_network_guard():
        initial = prepare_initial(config, counter=counter)
        append_enabled = config.get("append_enabled", True)
        result = {"version": VERSION, "initial": initial, "external_calls": 0,
                  "planner_entry_mode": initial["bundle"]["planner_entry_mode"],
                  "status": "OFFLINE_REQUESTS_PREPARED_NOT_GENERATED"}
        if not append_enabled and (raw_initial_output is not None or raw_supplement_output is not None):
            raise ValueError("planner_preparation_append_disabled")
        if raw_supplement_output is not None and raw_initial_output is None:
            raise ValueError("planner_preparation_initial_response_required")
        if raw_initial_output is not None:
            supplement = prepare_supplement(initial["bundle"], raw_initial_output, counter=counter)
            result["supplement"] = supplement
            if raw_supplement_output is not None:
                merged = planner_relations.merge_output(supplement["initial_compilation"]["prepared"],
                                                        raw_supplement_output)
                if not merged["valid"]:
                    raise ValueError("planner_preparation_supplement_output_invalid:" + str(merged["errors"]))
                result["supplement_compilation"] = merged
        return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def write_preparation(output, result):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "preparation.json", result)
    write_json(output / "initial/bundle.json", result["initial"]["bundle"])
    for role in ("initial", "supplement"):
        if role not in result:
            continue
        built = result[role]["request"]
        write_json(output / role / "messages.json", built["messages"])
        write_json(output / role / "capacity.json", result[role]["capacity"])
        with (output / role / "prompt.txt").open("x", encoding="utf-8") as stream:
            stream.write(built["prompt"])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mock-initial-output", type=Path)
    parser.add_argument("--mock-supplement-output", type=Path)
    args = parser.parse_args()
    with offline_network_guard():
        config = json.loads(args.config.read_text())
        supplied = {key: path.read_text() if path else None for key, path in (
            ("raw_initial_output", args.mock_initial_output),
            ("raw_supplement_output", args.mock_supplement_output))}
        result = prepare_from_config(config, **supplied)
        result["mock_response_inputs"] = {str(path): file_identity(path) for path in
            (args.mock_initial_output, args.mock_supplement_output) if path}
        write_preparation(args.output, result)
        print(json.dumps({"output": str(args.output.resolve()), "status": result["status"],
                          "external_calls": 0}))


if __name__ == "__main__":
    main()
