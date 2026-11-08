"""Versioned source evidence-domain validator; original prompts are unchanged.

No work runs at import. CLI calls remain owned by the caller's finite scheduler.
An exact native reference is accepted only for the assigned scene's pre-frozen
native record. Source full-read metadata never includes native reference IDs.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import traceback

BASE = Path(__file__).resolve().parents[1]
SCENES = [f"OET1:{route}:S{i:02}" for route in "AB" for i in range(1, 4)]
CHECKS = ["time", "location", "actor_identity", "plan_vs_completion", "cross_scene_continuity", "semantic_repetition", "naturalness", "event_fidelity"]


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve(); data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def verify(value):
    assert bind(value["path"]) == {k: value[k] for k in ("path", "sha256", "bytes")}, ("BINDING_CHANGED", value["path"])
    return read(value["path"])


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2); stream.write("\n")
    return bind(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def evidence_domains(packet, source, scene):
    """Return source turns and this scene's unique authorized native utterances."""
    assert scene in SCENES
    source_turns = {t["turn_ref"]: t for t in source["turns"]}
    assert len(source_turns) == len(source["turns"]), "DUPLICATE_SOURCE_REFS"
    native_turns = {}; native_binding = None
    index = SCENES.index(scene)
    if index >= 3:
        native = packet["native_events"][index - 3]
        record = verify(native["native_event_binding"])
        assert record == native["native_record"], "PACKET_NATIVE_RECORD_DIFFERS_FROM_BOUND_RECORD"
        native_binding = native["native_event_binding"]
        dialogue_id = record["dialogue_id"]
        assert isinstance(dialogue_id, str) and dialogue_id
        for turn in record["turns"]:
            ref = dialogue_id + ":" + str(turn["turn_id"])
            assert ref not in native_turns and ref not in source_turns, "NONUNIQUE_NATIVE_REFERENCE"
            assert isinstance(turn["utterance"], str) and turn["utterance"]
            native_turns[ref] = turn
    return source_turns, native_turns, native_binding


def validate_evidence(response, packet, source, scene):
    """Check every exact quote/ref without rewriting any semantic JSON value."""
    assert response["full_source_read"] is True
    assert response["source_turn_count"] == len(source["turns"])
    assert response["independent_of_source_author"] is True
    assert set(CHECKS) <= set(response["checks"])
    source_turns, native_turns, native_binding = evidence_domains(packet, source, scene)
    checked = []
    for dimension, check in response["checks"].items():
        assert check["reason"] and check["evidence"], ("EMPTY_REVIEW_CHECK", dimension)
        for e in check["evidence"]:
            ref = e["turn_ref"]; quote = e["quote"]
            assert isinstance(quote, str) and quote, ("EMPTY_QUOTE", dimension)
            if ref in source_turns:
                domain = "cumulative_source"; turn = source_turns[ref]; text = turn["text"]
            elif ref in native_turns:
                domain = "assigned_native_record"; turn = native_turns[ref]; text = turn["utterance"]
            else:
                raise ValueError("UNKNOWN_OR_UNAUTHORIZED_EVIDENCE_REF:" + str(ref))
            assert quote in text, ("QUOTE_NOT_EXACT_CONTIGUOUS_SUBSTRING", ref)
            if "speaker" in e:
                assert e["speaker"] == turn["speaker"], ("EVIDENCE_SPEAKER_MISMATCH", ref)
            checked.append({"dimension": dimension, "turn_ref": ref, "domain": domain,
                            "quote_sha256": hashlib.sha256(quote.encode()).hexdigest(),
                            "exact_contiguous_substring": True})
    for finding in response.get("findings", []):
        for ref in finding.get("turn_refs", []):
            assert ref in source_turns or ref in native_turns, ("UNKNOWN_FINDING_REFERENCE", ref)
    return {"schema_version": "oet1_source_native_evidence_validation_v1", "status": "PASS_EXACT_EVIDENCE_DOMAINS_ONLY",
            "scene_ref": scene, "source_turn_count": len(source_turns), "native_turn_count": len(native_turns),
            "native_event_binding": native_binding,
            "native_reference_grammar": "<assigned_native_record.dialogue_id>:<str(turn_id)>",
            "checked_quotes": checked, "quote_counts_by_domain": dict(Counter(r["domain"] for r in checked)),
            "source_read_scope_unchanged": True, "semantic_admission_inferred": False,
            "checker_binding": bind(__file__)}


def materialize_review(driver, root, arm, scene, response, result, *, require_ready=False):
    """Materialize a previously absent REVIEW from its one immutable raw return."""
    u = BASE / "sources" / root / arm; sd = u / scene.replace(":", "_")
    path = sd / "REVIEW.json"; assert not path.exists(), "REVIEW_ALREADY_SEALED"
    packet_path = u / "SOURCE_AUTHOR_INPUT.json"; packet = read(packet_path)
    source_path = sd / "CUMULATIVE_SOURCE.json"; draft_path = sd / "DRAFT.json"
    source = read(source_path); draft = read(draft_path)
    assert packet["root_id"] == root and packet["condition"] == arm
    assert draft["author_input_binding"] == bind(packet_path)
    assert draft["root_id"] == root and draft["scene_ref"] == scene
    index = SCENES.index(scene)
    previous = u / SCENES[index-1].replace(":", "_") / "CUMULATIVE_SOURCE.json" if index else Path(packet["source_binding"]["path"])
    assert draft["input_source_binding"] == bind(previous)
    assert source["turns"] == read(previous)["turns"] + draft["turns"], "SOURCE_PREFIX_OR_APPEND_CHANGED"
    assert source["draft_binding"] == bind(draft_path)
    if index >= 3:
        assert draft["native_event_binding"] == packet["native_events"][index-3]["native_event_binding"]
    result_path = BASE / "cli_calls" / f"{root}__{arm}__{scene.replace(':','_')}__REVIEW" / "RESULT.json"
    integrity = load_module("oet1_native_evidence_provenance", BASE / "scripts/production_integrity.py")
    author_result_path = result_path.parent.with_name(f"{root}__{arm}__{scene.replace(':','_')}__AUTHOR") / "RESULT.json"
    author_result, _ = integrity.validate_call_result(author_result_path, driver.source_author(packet, read(previous), scene))
    author_response = verify(author_result["parsed_response"])
    assert draft["cli_raw_binding"] == author_result["raw_response"] and draft["author_id"] == author_result["backend"]["thread_id"], "DRAFT_AUTHOR_PROVENANCE_MISMATCH"
    assert author_response["all_source_turns_read"] is True and author_response["source_turn_count"] == len(read(previous)["turns"])
    assert len(author_response["turns"]) == len(draft["turns"]) == 14
    assert all(a["speaker"] == d["speaker"] and a["text"] == d["text"] for a, d in zip(author_response["turns"], draft["turns"])), "DRAFT_DIFFERS_FROM_AUTHOR_RAW"
    checked_result, _ = integrity.validate_call_result(result_path, driver.source_reviewer(packet, source, draft, scene))
    assert checked_result == result and verify(result["parsed_response"]) == response, "RESPONSE_OR_RECEIPT_DRIFT"
    assert result["backend"]["thread_id"] != draft["author_id"], "SOURCE_SELF_REVIEW"
    if require_ready:
        assert response["decision"] == "READY", "RECOVERY_CANNOT_RESCUE_MODEL_FAIL_OR_HOLD"
        assert response.get("source_or_gold_ambiguity_failure") is False, "RECOVERY_CANNOT_RESCUE_AMBIGUITY"
        assert all(response["checks"][k]["status"] in ("PASS", "WARNING", "PASS_WITH_WARNING") for k in CHECKS)
        assert not any(f.get("severity") == "BLOCKING" for f in response.get("findings", []))
    validation = validate_evidence(response, packet, source, scene)
    validation.update(created_at_utc=now(), root_id=root, condition=arm,
                      source_binding=bind(source_path), draft_binding=bind(draft_path),
                      author_packet_binding=bind(packet_path), review_result_binding=bind(result_path),
                      raw_review_binding=result["raw_response"])
    validation_path = sd / "NATIVE_EVIDENCE_VALIDATION_V1.json"
    if validation_path.exists():
        old = read(validation_path)
        for key in validation:
            if key != "created_at_utc": assert old[key] == validation[key], ("VALIDATION_RECEIPT_DRIFT", key)
    else:
        save(validation_path, validation)
    review = {**response, "schema_version": "oet1_source_content_review_v1",
              "reviewer_id": result["backend"]["thread_id"], "reviewer_type": "codex_cli_ai",
              "source_binding": bind(source_path), "draft_binding": bind(draft_path),
              "turn_refs_read": [t["turn_ref"] for t in source["turns"]],
              "controller_expanded_turn_refs": True, "created_at_utc": result["finished_at_utc"],
              "cli_raw_binding": result["raw_response"], "blinding": "condition_visible_no_counterpart_no_solver",
              "native_evidence_validation": bind(validation_path)}
    for key, value in response.items():
        assert review[key] == value, ("RAW_REVIEW_SEMANTIC_VALUE_CHANGED", key)
    save(path, review)
    return bind(path), bind(validation_path)


def source_unit(driver, root, arm, control_path=None):
    """Original source-unit flow with the evidence resolver/materializer fixed.

    Original driver.source_author/source_reviewer prompt functions are called
    byte-for-byte. The injected driver.call retains its existing provenance guard.
    Existing failures remain terminal unless a separate explicit engineering
    recovery has already produced a hash-bound acceptance receipt.
    """
    control = driver.module("oet_control_v3_"+root+arm, Path(control_path) if control_path else BASE / "scripts/control_v3.py")
    u = BASE / "sources" / root / arm; packet = driver.read(u / "SOURCE_AUTHOR_INPUT.json")
    for i, scene in enumerate(SCENES):
        sd = u / scene.replace(":", "_")
        if (sd / "ACCEPTED.json").exists():
            r = driver.read(sd / "ACCEPTED.json")
            assert r["source"] == driver.bind(sd / "CUMULATIVE_SOURCE.json") and r["review"] == driver.bind(sd / "REVIEW.json")
            continue
        if (sd / "FAILURE.json").exists(): return False
        if driver.stop_generation(): return False
        previous_path = u / SCENES[i-1].replace(":", "_") / "CUMULATIVE_SOURCE.json" if i else Path(packet["source_binding"]["path"])
        previous = driver.read(previous_path)
        try:
            if not (sd / "DRAFT.json").exists():
                response, result = driver.call(f"{root}__{arm}__{scene.replace(':','_')}__AUTHOR", driver.source_author(packet, previous, scene))
                assert response["all_source_turns_read"] is True and response["source_turn_count"] == len(previous["turns"])
                assert len(response["turns"]) == 14
                if i >= 3: assert response["native_record_read"] is True
                turns = []
                for n, t in enumerate(response["turns"], 1):
                    assert t["speaker"] in packet["speakers"] and isinstance(t["text"], str) and t["text"].strip()
                    turns.append(dict(turn_ref=f"{scene}:T{n:03}", scene_ref=scene, session_ref=scene, speaker=t["speaker"], text=t["text"]))
                draft = dict(schema_version="oet1_scene_draft_v1", root_id=root, scene_ref=scene,
                             author_id=result["backend"]["thread_id"], created_at_utc=result["finished_at_utc"],
                             input_source_binding=driver.bind(previous_path), author_input_binding=driver.bind(u / "SOURCE_AUTHOR_INPUT.json"),
                             turns=turns, cli_raw_binding=result["raw_response"], controller_metadata_only=True)
                if i >= 3:
                    draft.update(native_event_binding=packet["native_events"][i-3]["native_event_binding"], source_id=packet["native_events"][i-3]["source_id"])
                driver.write(sd / "DRAFT.json", draft)
                driver.write(sd / "AUTHOR_CONSUMPTION.json", dict(full_source_read=True, scope_attestation=response,
                             turn_refs_read=[t["turn_ref"] for t in previous["turns"]], controller_expanded_turn_refs=True,
                             input=driver.bind(previous_path), packet=driver.bind(u / "SOURCE_AUTHOR_INPUT.json"),
                             draft=driver.bind(sd / "DRAFT.json"), call_receipt=result))
            if not (sd / "CUMULATIVE_SOURCE.json").exists(): control.commit(root, arm, scene)
            if not (sd / "REVIEW.json").exists():
                source = driver.read(sd / "CUMULATIVE_SOURCE.json"); draft = driver.read(sd / "DRAFT.json")
                response, result = driver.call(f"{root}__{arm}__{scene.replace(':','_')}__REVIEW", driver.source_reviewer(packet, source, draft, scene))
                materialize_review(driver, root, arm, scene, response, result)
            control.accept(root, arm, scene)
            driver.log("SCENE_ACCEPTED_V3", root=root, arm=arm, scene=scene)
        except Exception as error:
            if not (sd / "FAILURE.json").exists():
                driver.write(sd / "FAILURE.json", dict(status="FAIL", at_utc=driver.now(), error=str(error), traceback=traceback.format_exc(), semantic_retry=False, descendants_stopped=True))
            driver.log("SCENE_FAILED_V3", root=root, arm=arm, scene=scene, error=str(error)); return False
    return (u / "FINAL_SOURCE.json").exists()
