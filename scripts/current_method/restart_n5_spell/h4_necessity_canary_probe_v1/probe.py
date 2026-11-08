"""Live isolated H4 necessity canary generation."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import prepare_empty, write_json
from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.live_adapter import LiveJSONClient
from scripts.current_method.restart_n5_spell.planner_realizer_temporal_boundary_probe_v1.batch_probe_v1 import _adapt_plan, _find_source
from scripts.current_method.restart_n5_spell.planner_realizer_temporal_boundary_probe_v1.scene_release_probe_v2 import _renumber, build_single_scene_prompt_v2
from scripts.current_method.restart_n5_spell.planner_realizer_temporal_boundary_probe_v1.scene_release_probe_v3 import _safe_extract
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import normalize_realizer_output


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
PROBE_VERSION = "h4_necessity_canary_probe_v1"
MAX_TOKENS = 8192
FORBIDDEN_RESOLUTION_TERMS = ("water", "thin", "thinner", "spreadable", "more fluid", "fluid")
TRIGGER_TERMS = ("too thick", "thick", "spread evenly", "coat the ham", "coat")
FACT_B_TERMS = ("water", "thin", "thinner")
OPERATIONAL_TERMS = ("roast", "oven", "timer", "baste", "prepare", "ham")


def _canary_plan(plan: dict[str, Any]) -> dict[str, Any]:
    scenes = [dict(scene) for scene in plan["scenes"]]
    # Keep A and B from the accepted source, but make S3 an operational
    # follow-through so the full transcript does not repeat B's answer.
    resolution = scenes[2]
    resolution.update({
        "before_state": "the prepared ham is ready for the next cooking step",
        "verified_current_state": "the prepared ham is ready for the next cooking step",
        "required_transition": "resolution / resolution; requested=none; trigger=operational_followthrough: begin roasting and coordinate the cooking timer",
        "after_state": "the team begins roasting the prepared ham and coordinates the cooking timer",
        "durable_outcome": "the team begins roasting the prepared ham and coordinates the cooking timer",
        "new_development": "resolution / resolution; requested=none; trigger=operational_followthrough: begin roasting and coordinate the cooking timer",
        "actor_bindings": [
            {"actor": "Classmates A", "responsibilities": ["Classmates A places the prepared ham in the oven."]},
            {"actor": "Classmates B", "responsibilities": ["Classmates B sets the cooking timer and prepares the side dish."]},
        ],
        "interaction_beats": [
            "Classmates A places the prepared ham in the oven.",
            "Classmates B sets the cooking timer.",
            "They coordinate the remaining cooking steps.",
        ],
    })
    return {**plan, "scenes": scenes, "plan_version": "h4_necessity_canary_plan_v1"}


def _contract(plan: dict[str, Any]) -> dict[str, Any]:
    first, middle, final = plan["scenes"]
    trigger = middle.get("trigger") or {}
    return {
        "contract_version": "h4_necessity_canary_contract_v1",
        "hypothesis": "H4",
        "execution_scope": "isolated_necessity_canary",
        "relation_id": "RELCHAIN:NECESSITY:S01-S02-S03",
        "relation_type": "causal_revision",
        "state_slot": str(first.get("before_state") or first.get("after_state") or ""),
        "fact_A": {
            "role": "initial_state_or_plan",
            "scene_index": 1,
            "value": str(first.get("after_state") or ""),
            "source_plan_field": "scenes[0].after_state",
        },
        "trigger": {
            "role": "causal_trigger",
            "scene_index": 2,
            "type": str(trigger.get("type") or "new_information"),
            "content": str(trigger.get("content") or middle.get("required_transition") or ""),
            "source_plan_field": "scenes[1].trigger",
        },
        "fact_B": {
            "role": "revised_state_or_plan",
            "scene_index": 2,
            "value": str(middle.get("after_state") or ""),
            "source_plan_field": "scenes[1].after_state",
        },
        "resolution": {
            "role": "operational_followthrough_without_state_restatement",
            "scene_index": 3,
            "value": str(final.get("after_state") or ""),
            "source_plan_field": "scenes[2].after_state",
        },
        "answer_components": {
            "fact_A": ["two parts honey", "one part mustard"],
            "trigger": ["too thick", "spread evenly"],
            "fact_B": ["water", "thin"],
        },
        "forbidden_resolution_terms": list(FORBIDDEN_RESOLUTION_TERMS),
        "typed_edges": [
            {"source": "fact_A", "relation": "constrained_or_revised_by", "target": "trigger"},
            {"source": "trigger", "relation": "causes_revision_to", "target": "fact_B"},
            {"source": "fact_B", "relation": "followed_by", "target": "resolution"},
        ],
        "qa_target": {
            "composite_intent": "identify the initial recipe, the reason for revision, and the revised addition",
            "required_evidence_groups": ["fact_A", "trigger", "fact_B"],
        },
    }


def _prompt(parent: dict[str, Any], prior: list[dict[str, Any]], current: dict[str, Any], index: int, contract: dict[str, Any]) -> dict[str, Any]:
    base = build_single_scene_prompt_v2(parent=parent, prior=prior, current=current, index=index)
    if index == 1:
        rules = (
            "Expose only the initial recipe state. Do not mention or foreshadow the later thickness objection, water, thinning, or any revised state."
        )
        payload = {"relation_type": contract["relation_type"], "fact_A": contract["fact_A"], "stage_rules": rules}
    elif index == 2:
        rules = (
            "State the thickness/spreading problem as the trigger, then state the water addition as fact_B. Keep the causal order explicit. Do not describe the later operational resolution."
        )
        payload = {"relation_type": contract["relation_type"], "fact_A": contract["fact_A"], "trigger": contract["trigger"], "fact_B": contract["fact_B"], "stage_rules": rules}
    else:
        rules = (
            "Show only operational follow-through: roasting, oven, timer, or preparation. Do not restate or paraphrase fact_B. Do not use any forbidden resolution terms: "
            + ", ".join(FORBIDDEN_RESOLUTION_TERMS)
            + "."
        )
        payload = {"relation_type": contract["relation_type"], "resolution": contract["resolution"], "stage_rules": rules, "forbidden_resolution_terms": contract["forbidden_resolution_terms"]}
    prompt = str(base["prompt"]) + (
        "\n\nNECESSITY-FIRST RELATION CONTRACT (isolated H4 canary)\n"
        "The purpose is evidence necessity, not extra plot. Use the same named people and object.\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n\nSTAGE INSTRUCTION\n" + rules
    )
    return {**base, "prompt_version": "h4_necessity_canary_prompt_v1", "prompt": prompt, "relation_contract_version": contract["contract_version"]}


def _scene_text(scene: dict[str, Any]) -> str:
    return " ".join(str(row.get("text") or "") for row in scene.get("dialogue_turns") or [])


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.casefold()
    return any(re.search(r"\b" + re.escape(term.casefold()) + r"\b", lowered) for term in terms)


def _audit(child: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    scenes = list(child.get("scenes") or [])
    texts = [_scene_text(scene) for scene in scenes]
    s1, s2, s3 = (texts + [""] * 3)[:3]
    checks = {
        "three_scenes": len(texts) == 3,
        "fact_A_in_s1": "honey" in s1.casefold() and "mustard" in s1.casefold(),
        "no_revision_leakage_in_s1": not _has_any(s1, FACT_B_TERMS + ("too thick", "spread evenly")),
        "trigger_in_s2": _has_any(s2, TRIGGER_TERMS),
        "fact_B_in_s2": _has_any(s2, FACT_B_TERMS),
        "operational_resolution_in_s3": _has_any(s3, OPERATIONAL_TERMS),
        "no_answer_leakage_in_s3": not _has_any(s3, FORBIDDEN_RESOLUTION_TERMS),
    }
    return {
        "audit_version": "h4_necessity_canary_audit_v1",
        "relation_id": contract["relation_id"],
        "relation_type": contract["relation_type"],
        "checks": checks,
        "relation_present": bool(checks["fact_A_in_s1"] and checks["trigger_in_s2"] and checks["fact_B_in_s2"]),
        "temporal_leakage_free": bool(checks["no_revision_leakage_in_s1"] and checks["no_answer_leakage_in_s3"]),
        "resolution_leakage_terms": [term for term in FORBIDDEN_RESOLUTION_TERMS if _has_any(s3, (term,))],
        "relation_source_turns": {
            "fact_A": ["R01:S01:T001"],
            "trigger": ["R01:S02:T001"],
            "fact_B": ["R01:S02:T002", "R01:S02:T003"],
            "resolution": [
                str(row.get("turn_ref") or "")
                for row in (scenes[2].get("dialogue_turns") or [])
                if row.get("turn_ref")
            ] if len(scenes) == 3 else [],
        },
        "required_role_refs_disjoint": True,
        "necessity_source_ready": bool(all(checks.values())),
    }


def run(*, run_root: Path, config: Path, case_id: str = "POB03") -> dict[str, Any]:
    prepare_empty(run_root)
    source, provenance = _find_source(case_id)
    parent = provenance["parent"]
    plan = _canary_plan(_adapt_plan(source))
    contract = _contract(plan)
    write_json(run_root / "source_record.json", source)
    write_json(run_root / "adapted_plan.json", plan)
    write_json(run_root / "relation_contract.json", contract)
    client = LiveJSONClient(config, max_tokens=MAX_TOKENS, temperature=0.0)
    prior: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    transport_error = None
    for index, current in enumerate(plan["scenes"], 1):
        scene_root = run_root / f"scene_{index:02d}"
        scene_root.mkdir(parents=True, exist_ok=True)
        prompt = _prompt(parent, prior, current, index, contract)
        write_json(scene_root / "prompt.json", prompt)
        raw, meta, parsed, error, mode = "", {}, {}, None, None
        calls_before = client.calls
        try:
            raw, meta = client.complete(prompt["prompt"])
            (scene_root / "raw_response.txt").write_text(raw, encoding="utf-8")
            parsed, mode = _safe_extract(raw)
            write_json(scene_root / "parsed_response.json", parsed)
        except Exception as exc:
            error = f"{type(exc).__name__}:{exc}"
            transport_error = error
            (scene_root / "raw_response.txt").write_text(raw, encoding="utf-8")
        mechanical = normalize_realizer_output(parsed, plan={"scenes": [current]}, parent=parent, round_index=1) if error is None else {"valid": False, "errors": [error]}
        if mechanical.get("valid") and mechanical.get("child", {}).get("scenes"):
            scene = _renumber(mechanical["child"]["scenes"][0], index, str(parent["root_id"]), str(current.get("memory_event_id")))
            prior.append(scene)
            write_json(scene_root / "child_scene.json", scene)
        record = {
            "scene_index": index,
            "request_attempts": 1,
            "realizer_calls_completed": client.calls - calls_before,
            "response_meta": meta,
            "mechanical_contract_valid": bool(mechanical.get("valid")),
            "envelope_mode": mode,
            "transport_error": error,
        }
        write_json(scene_root / "mechanical_audit.json", mechanical)
        write_json(scene_root / "metrics.json", record)
        records.append(record)
        if error or not mechanical.get("valid"):
            break
    child = {"schema_version": "memory_event_continuation_child_v1", "root_id": parent["root_id"], "round_index": 1, "scenes": prior}
    write_json(run_root / "child_dialogue.json", child)
    audit = _audit(child, contract)
    write_json(run_root / "relation_audit.json", audit)
    conclusion = "NECESSITY_CANARY_SOURCE_READY" if audit["necessity_source_ready"] else "NECESSITY_CANARY_LEAKAGE_OR_REALIZATION_FAIL"
    metrics = {
        "probe_version": PROBE_VERSION,
        "hypothesis": "H4",
        "case_id": case_id,
        "root_id": parent["root_id"],
        "scene_records": records,
        "scenes_completed": len(prior),
        "realizer_calls_completed": client.calls,
        "relation_present": audit["relation_present"],
        "temporal_leakage_free": audit["temporal_leakage_free"],
        "necessity_source_ready": audit["necessity_source_ready"],
        "relation_audit": str((run_root / "relation_audit.json").resolve()),
        "conclusion": conclusion,
        "analysis_execution_api_calls": 0,
        "planner_calls": 0,
        "qa_writer_calls": 0,
        "solver_calls": 0,
        "composer_calls": 0,
        "memorax_calls": 0,
    }
    write_json(run_root / "metrics.json", metrics)
    write_json(run_root / "run_freeze_manifest.json", {
        "freeze_version": PROBE_VERSION,
        "case_id": case_id,
        "isolated": True,
        "api_role_allowlist": ["realizer"],
        "one_request_per_scene": True,
        "mainline_modified": False,
        "qa_writer_calls": 0,
        "solver_calls": 0,
        "composer_calls": 0,
        "memorax_calls": 0,
    })
    (run_root / "request_ledger.json").write_text(json.dumps({"probe_version": PROBE_VERSION, "requests": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_root / "report.md").write_text("\n".join([
        "# H4 Necessity Canary Probe v1", "", f"Verdict: **{conclusion}**", "",
        f"- Realizer calls: `{client.calls}`", f"- Scenes completed: `{len(prior)}/3`",
        f"- relation_present: `{audit['relation_present']}`", f"- temporal_leakage_free: `{audit['temporal_leakage_free']}`",
        f"- necessity_source_ready: `{audit['necessity_source_ready']}`", "",
        "This run is an isolated Realizer canary. QA Writer, Composer, Solver, Memorax, Planner, and historical runs were not called or modified.", "",
        "Only a `NECESSITY_CANARY_SOURCE_READY` result may proceed to the QA pipeline.", "",
    ]), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("/root/autodl-tmp/gxy-selfplay/config.json"))
    parser.add_argument("--case-id", default="POB03")
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = (args.run_root or RUN_BASE / f"{PROBE_VERSION}_{stamp}").resolve()
    print(json.dumps(run(run_root=root, config=args.config, case_id=args.case_id), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
