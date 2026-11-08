"""Offline proxy budget rehearsal on exposed v97 full requests and responses.

Comparison text is manually authored annotation on a copy, never a new model
output or semantic success. Historical original texts and failures are unchanged.
"""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sys

HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME / "expansion_engine"))
import runner
import event_state
import realization_review
from event_identity_test_fixtures import comparison, identity

OLD = HOME.parent / "h4_semantic_review_mixed_pilot_v97"
REQUESTS = OLD / "expansion_engine/runs/four_roots_B_scene01/requests"


def bind(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def run():
    profile, _client, annotate, capacity, dev = runner.base.transport()
    tokenizer = dev.tokenizer_from(runner.base.TOKENIZER)
    rows = []
    for folder in sorted(REQUESTS.iterdir()):
        if not ("event-state" in folder.name or "review-" in folder.name): continue
        is_scene = "review-" in folder.name
        original_request = json.loads((folder / "REQUEST.json").read_text())
        original_response = json.loads((folder / "RESPONSE.json").read_text())
        marker = "REVIEW INPUT DATA:\n" if is_scene else "EVENT_STATE_INPUT_JSON\n"
        context, _ = json.JSONDecoder().raw_decode(original_request["messages"][0]["content"].split(marker, 1)[1])
        scene = deepcopy(context["scene_plan"])
        history = context["branch_history"] if is_scene else context["cumulative_history"]
        card = context["dataset_card_guidance"] if is_scene else context["dataset_cards"][0]
        comp = comparison(card, history, scene)
        root = context["root_id"].rsplit("-", 1)[1]
        if scene.get("source_anchors"):
            comp["nearest_old_event"]["history_evidence"] = [deepcopy(scene["source_anchors"][0])]
        if root == "19053":
            comp["purpose"] = {"source": "Enjoy a shared completed outing and plan another", "actual": "Confirm a future volunteer schedule"}
            comp["start_state"] = {"source": "Outing experienced", "actual": "Volunteer shift still pending"}
            comp["end_state"] = {"source": "Experience appreciated and another agreed", "actual": "Future monthly slot agreed"}
            comp["experience_state"].update(source="completed_experience", actual="future_arrangement", verdict="revise")
            comp["nearest_old_event"].update(summary="Already arranged shelter volunteering", difference="Future scheduling only; no completed outing")
        elif root == "9557":
            comp["purpose"] = {"source": "Request and accept a new service", "actual": "Advance an old risk task"}
            comp["start_state"] = {"source": "Service not requested", "actual": "Risk help requested and accepted earlier"}
            comp["end_state"] = {"source": "New request accepted", "actual": "Risk list discussed; old acceptance repeated"}
            comp["nearest_old_event"].update(summary="Old risk-help request and acceptance", difference="Risk list is real progress, not a new mapped request")
            comp["request_acceptance"] = {"required": True, "verdict": "revise", "request_evidence": [], "acceptance_evidence": []}
            if is_scene:
                comp["request_acceptance"]["acceptance_evidence"] = [{"turn_ref": "NEW:T006",
                    "quote": "I formally accept the request to help lead the risk mitigation strategy for these areas."}]
            else:
                comp["request_acceptance"]["acceptance_evidence"] = [{"development_index": 2, "quote": scene["developments"][1]}]
        else:
            comp["purpose"] = {"source": "Learn soil preparation from an advisor", "actual": "Prepare for kitten adoption" if folder.name.startswith("018") else "Prepare garden ground around the kitten"}
            comp["nearest_old_event"].update(summary="Old kitten discovery and offer of adoption help",
                difference="First plan substitutes adoption; second adds soil work but purpose remains a review risk")
        scene["event_identity"] = identity(card, history, scene)
        response = json.loads(original_response["text"])
        target = response["card_checks"][0] if is_scene else response["card_checks"][0]["function_check"]
        target["comparison"] = comp
        # This deliberately preserves every old response judgment and any bad
        # quote. It measures shape size, not validity or a repaired old output.
        serializations = {"compact": json.dumps(response, ensure_ascii=False, separators=(",", ":")),
                          "indent2": json.dumps(response, ensure_ascii=False, indent=2)}
        counts = {key: len(tokenizer.encode(text, add_special_tokens=False)) for key, text in serializations.items()}
        if is_scene:
            text = realization_review.build_review_prompt(context["root_id"], context["route"], scene,
                history, context["new_scene"], context["other_branch_added_turns_for_review_only"], dataset_card=card,
                latest_state=context["latest_state"], event_state=context["event_state"])
            text += realization_review.final_review_instruction(scene, context["new_scene"])
            messages = [{"role": "user", "content": text}]
        else:
            request = event_state.build_request(context["root_id"], context["route"], scene, history,
                [card], context["latest_state"])
            messages = request["messages"]
            assert request["max_tokens"] == 4096
        cap = annotate(capacity(profile(), tokenizer, messages, 4096))
        rows.append({"request_id": folder.name, "original_request": bind(folder / "REQUEST.json"),
            "original_response": bind(folder / "RESPONSE.json"), "history_turns": len(history),
            "local_old_raw_output_tokens": len(tokenizer.encode(original_response["text"], add_special_tokens=False)),
            "authored_copy_output_tokens": counts,
            "authored_copy_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in serializations.items()},
            "fixture_max_margin_to_4096": 4096 - max(counts.values()),
            "whole_v98_review_request_capacity": cap,
            "actual_future_output_or_provider_limit_certified": False})
    result = {"version": "v98_event_identity_exposed_fixture_budget_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_EXPOSED_OFFLINE_FIXTURE_ONLY" if all(r["fixture_max_margin_to_4096"] >= 0 and
            r["whole_v98_review_request_capacity"]["status"] == "PASS" for r in rows) else "HOLD_FIXTURE_OVER_BUDGET",
        "tokenizer_path": str(runner.base.TOKENIZER), "max_output_tokens_unchanged": 4096,
        "rows": rows, "script": bind(Path(__file__).resolve()), "network_calls": 0,
        "limitations": ["Local tokenizer is a proxy for the remote model; no provider hard limit is certified.",
            "Manually authored compact comparisons on known failures test output shape size, not model fidelity, semantic improvement or independent success.",
            "All old checks and invalid quotes remain in the size-only copies; they are not silently repaired or promoted.",
            "Future actual requests still need the unchanged exact capacity gate; future full model responses may be longer than these fixtures."]}
    out = HOME / "evidence/EVENT_IDENTITY_BUDGET_REHEARSAL.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"path": str(out), "status": result["status"],
        "rows": len(rows), "max_authored_indent_tokens": max(r["authored_copy_output_tokens"]["indent2"] for r in rows),
        "min_fixture_margin": min(r["fixture_max_margin_to_4096"] for r in rows)}, ensure_ascii=False))
    assert result["status"] == "PASS_EXPOSED_OFFLINE_FIXTURE_ONLY"


if __name__ == "__main__": run()
