"""Offline v99 planner schema regression using frozen v98 request/context bytes.

No provider call is made.  v98 planner requests are retained as controls; v99
make_request builds A/B treatment payloads by reading the sealed v98 run's
complete history and frozen feedback.  This checks prompt and
context contracts only, never model compliance or semantic quality.
"""
from __future__ import annotations
import copy
import hashlib
import json
import pathlib
import sys
from json import JSONDecoder

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENGINE = ROOT / "expansion_engine"
V98 = ROOT.parent / "h4_event_identity_correction_pilot_v98"
V98RUN = V98 / "expansion_engine/runs/four_roots_B_scene01"
V99 = ROOT
sys.path.insert(0, str(ENGINE))
import runner  # noqa: E402

ROOT_ID = "cc-train-episode-19053"
SCENE_ID = "S01"
A = runner.ROUTES[0]
B = runner.ROUTES[1]


def digest(value):
    if isinstance(value, bytes):
        data = value
    else:
        data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text())


def parse_response_text(path):
    return json.loads(read(path)["text"])


def parse_input(content):
    marker = "\nINPUT_JSON\n"
    at = content.rfind(marker)
    if at < 0:
        raise AssertionError("INPUT_JSON marker missing")
    raw = content[at + len(marker):]
    result, _ = JSONDecoder().raw_decode(raw)
    return result


def v98_request(number):
    return V98RUN / "requests" / number / "REQUEST.json"


def control(number):
    payload = read(v98_request(number))
    text = payload["messages"][0]["content"]
    return {
        "label": "control_v98_" + number,
        "source": str(v98_request(number).relative_to(V98)),
        "payload_sha256": digest(payload),
        "prompt_sha256": digest(text.encode()),
        "input": parse_input(text),
        "prompt": text,
    }


def treatment_payloads():
    # Production make_request is a pure builder: it reads the sealed v98 run's
    # full history/feedback and performs no write or network call.  The B card,
    # latest state and targeted repair are copied from the actual request JSON.
    initial = control("006-cc-train-episode-19053-B-S01-planner-1")["input"]
    retry = control("007-cc-train-episode-19053-B-S01-planner-2")["input"]
    history = initial["cumulative_history"]
    state = initial["latest_state"]
    cards = initial["dataset_cards"]
    state_a = copy.deepcopy(state)
    state_a["route"] = A
    outputs = {}
    cases = [
        (A, [], state_a, None, "treatment_A_v99"),
        (B, cards, state, None, "treatment_B_v99"),
        (B, cards, state, retry["targeted_repair"], "treatment_B_retry_v99"),
    ]
    for route, assigned, latest, repair, label in cases:
        payload, proof = runner.make_request(V98RUN, ROOT_ID, route, "planner", history,
                                             assigned, SCENE_ID, latest_state=latest, repair=repair)
        text = payload["messages"][0]["content"]
        outputs[label] = {
            "label": label,
            "payload": payload,
            "payload_sha256": digest(payload),
            "prompt_sha256": digest(text.encode()),
            "input": parse_input(text),
            "proof": proof,
            "prompt": text,
        }
    return outputs


def complete_schema(prompt):
    marker = "COMPLETE OUTPUT SCHEMA"
    at = prompt.find(marker)
    if at < 0:
        raise AssertionError("authoritative schema marker missing")
    return JSONDecoder().raw_decode(prompt[prompt.find("{", at):])[0]


def assertions(controls, treatments):
    rows = []
    def check(name, ok, detail):
        rows.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    # The frozen v98 failure is the baseline: both actual B planner contexts
    # carried a card but the old introductory shape did not make event_identity
    # part of the complete scene field list.
    c1, c2 = controls
    check("control_v98_route_B", c1["input"].get("route") == B and c2["input"].get("route") == B,
          f"routes={[c1['input'].get('route'), c2['input'].get('route')]}")
    check("control_v98_card_context", bool(c1["input"].get("dataset_cards")) and bool(c1["input"].get("card_mapping_context")),
          "v98 B context retains assigned card and mapping context")
    c_prompt = c1["prompt"]
    check("control_v98_stale_intro_observed", "event_identity" not in c_prompt[c_prompt.find("The scene has"):c_prompt.find("};", c_prompt.find("The scene has")) + 2],
          "v98 introductory scene field list omits event_identity (historical control)")

    a = treatments["treatment_A_v99"]
    b = treatments["treatment_B_v99"]
    ai, bi = a["input"], b["input"]
    check("treatment_A_route", ai.get("route") == A, f"route={ai.get('route')}")
    check("treatment_B_route", bi.get("route") == B, f"route={bi.get('route')}")
    check("A_requires_no_card", ai.get("dataset_cards") == [], f"dataset_cards={len(ai.get('dataset_cards', []))}")
    check("B_requires_one_card", len(bi.get("dataset_cards", [])) == 1 and "card_mapping_context" in bi,
          f"cards={len(bi.get('dataset_cards', []))}, mapping={'card_mapping_context' in bi}")
    check("A_scene_plan_no_event_identity_requirement", "event_identity" not in a["prompt"].split("COMPLETE OUTPUT SCHEMA", 1)[1].split("INPUT_JSON", 1)[0],
          "route A authoritative schema excludes dataset event_identity")
    b_schema_block = b["prompt"].split("COMPLETE OUTPUT SCHEMA", 1)[1].split("INPUT_JSON", 1)[0]
    check("B_scene_schema_requires_event_identity", "event_identity" in b_schema_block,
          "route B authoritative schema includes event_identity")
    check("A_scene_schema_has_empty_card_fields", '"dataset_card_id": null' in a["prompt"] and '"card_development_links": []' in a["prompt"],
          "A schema explicitly fixes null/empty card fields")
    check("B_scene_schema_has_card_fields", complete_schema(b["prompt"])["scenes"][0]["dataset_card_id"] == bi["dataset_cards"][0]["card_id"] and "card_development_links" in b_schema_block,
          "B schema explicitly fixes card mapping fields")
    # Initial and retry route builders share the complete schema; a synthetic
    # exact missing-path error must occur immediately before the final schema.
    missing = [{"code": "event_identity_missing", "path": "scene.event_identity", "actual": None,
                "allowed": "complete object"}]
    retry_b = runner.planner_contract.final_planner_instruction({"errors": missing}, route=B, card=bi["dataset_cards"][0])
    retry_a = runner.planner_contract.final_planner_instruction({"errors": [{"code": "scene_fields_invalid", "path": "scene", "actual": ["x"]}]}, route=A)
    b_tail_schema = runner.planner_schema.schema_text(B, bi["dataset_cards"][0])
    a_tail_schema = runner.planner_schema.schema_text(A)
    check("retry_B_exact_error_tail", "scene.event_identity" in retry_b and retry_b.rstrip().endswith(b_tail_schema),
          "B retry ends with exact missing path then complete B schema")
    check("retry_A_exact_error_tail", "scene" in retry_a and retry_a.rstrip().endswith(a_tail_schema),
          "A retry ends with exact validation error then complete A schema")
    check("initial_retry_B_schema_same", complete_schema(b["prompt"]) == json.loads(b_tail_schema) and retry_b.rstrip().endswith(b_tail_schema),
          "B initial and retry complete schemas are byte-identical")
    check("initial_retry_A_schema_same", complete_schema(a["prompt"]) == json.loads(a_tail_schema)
          and retry_a.rstrip().endswith(a_tail_schema), "A initial and retry complete schemas are byte-identical")
    actual_retry = treatments["treatment_B_retry_v99"]
    errors = c2["input"]["targeted_repair"]["errors"]
    actual_retry_text = actual_retry["prompt"]
    exact_json = json.dumps(errors, ensure_ascii=False, indent=2)
    check("actual_v98_retry_errors_repeat_at_treatment_tail", exact_json in actual_retry_text[actual_retry_text.rfind("PLANNER RETRY"):] and actual_retry_text.rstrip().endswith(b_tail_schema),
          "actual v98 validator error JSON preserved after frozen feedback and before final B schema")
    check("B_context_initial_equal_v98", bi == c1["input"], "B treatment preserves entire v98 actual initial context")
    check("B_context_retry_equal_v98", actual_retry["input"] == c2["input"], "B treatment preserves entire v98 actual retry context")
    a_tail = runner.planner_contract.final_planner_instruction(route=A)
    check("A_tail_no_dataset_assignment_requirements", all(term not in a_tail for term in ["assigned card", "card's actual", "card step and dependency", "MAIN new event from the card"]),
          "A tail must not impose dataset-card mapping/business requirements")
    check("control_treatment_prompt_changed", c1["prompt_sha256"] != b["prompt_sha256"],
          "v99 treatment prompt differs from frozen v98 control")
    return rows


def main():
    controls = [control("006-cc-train-episode-19053-B-S01-planner-1"),
                control("007-cc-train-episode-19053-B-S01-planner-2")]
    treatments = treatment_payloads()
    rows = assertions(controls, treatments)
    out = {
        "schema_version": "h4.v99.planner_schema_regression.v1",
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
        "live": False,
        "provider_calls": 0,
        "logical_requests": 0,
        "preview_payloads": 3,
        "planned_max_tokens": 4096,
        "root_id": ROOT_ID,
        "scene_id": SCENE_ID,
        "controls": [{k: v for k, v in c.items() if k not in ("prompt",)} for c in controls],
        "treatments": {k: {kk: vv for kk, vv in v.items() if kk not in ("prompt", "payload")} for k, v in treatments.items()},
        "checks": rows,
        "engine_bindings": {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in [ENGINE / "runner.py", ENGINE / "planner_schema.py", ENGINE / "planner_contract.py", ENGINE / "latest_state.py"]},
        "test_script_sha256": digest(pathlib.Path(__file__).read_bytes()),
        "fixture_notes": ["A latest_state is a copied structural fixture with route changed from B to A; it is not an actual new extraction or semantic result", "B latest_state, history, card and complete targeted_repair are exactly copied from v98 actual request contexts", "feedback is read from sealed v98 run by the production feedback adapter; no network or model call"],
        "source_bindings": {
            "v98_run": str(V98RUN),
            "v98_control_requests": [str(v98_request("006-cc-train-episode-19053-B-S01-planner-1")), str(v98_request("007-cc-train-episode-19053-B-S01-planner-2"))],
            "v98_history_sha256": digest(read(V98RUN / "sources" / f"{ROOT_ID}.json")["turns"]),
            "v98_card_sha256": digest(controls[0]["input"]["dataset_cards"][0]),
        },
        "assertion_scope": "prompt/context/schema contract only; no model compliance, semantic review, or fresh yield",
    }
    preview_dir = ROOT / "evidence/schema_regression_previews"
    preview_dir.mkdir(exist_ok=True)
    for label, value in treatments.items():
        (preview_dir / (label + ".json")).write_text(json.dumps({"payload": value["payload"], "proof": value["proof"]}, ensure_ascii=False, indent=2) + "\n")
        out["treatments"][label]["preview_path"] = str((preview_dir / (label + ".json")).relative_to(ROOT))
    path = ROOT / "evidence/REGRESSION_SCHEMA_TEST_RESULTS.json"
    if path.exists():
        prior = ROOT / "evidence/REGRESSION_SCHEMA_TEST_RESULTS_INITIAL_INTERFACE_MISMATCH.json"
        if not prior.exists():
            prior.write_bytes(path.read_bytes())
            (ROOT / "evidence/REGRESSION_SCHEMA_TEST_INITIAL_INTERFACE_MISMATCH.log").write_bytes((ROOT / "evidence/REGRESSION_SCHEMA_TEST.log").read_bytes())
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "evidence/REGRESSION_SCHEMA_TEST.log").write_text("\n".join(f"{r['status']} {r['check']}: {r['detail']}" for r in rows) + "\n")
    print(json.dumps({"status": out["status"], "checks": len(rows), "output": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
