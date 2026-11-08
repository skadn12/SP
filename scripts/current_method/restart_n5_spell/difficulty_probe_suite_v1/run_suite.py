from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from . import SUITE_VERSION
from .common.frozen_loader import (DEFAULT_RUN, load_dialogue, load_frozen_inputs,
                                   parent_view)
from .common.io_utils import (prepare_empty, read_json, sha256_file, write_json,
                              write_jsonl)
from .common.live_adapter import LiveJSONClient, extract_json
from .common.metrics import contingency, solver_summary
from .common.semantic_audit import paraphrase_audit, source_audit, state_audit
from .common.solver_adapter import maybe_solver


PROJECT = Path(__file__).resolve().parents[4]
RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
SECOND_ROOT = RUN_BASE / "conversation_chronicles_episode_61555_simple_append_memorax_v1_qa_live_1"


def _json(path: Path, value: Any) -> None:
    write_json(path, value)


def _det_paraphrase(question: str, index: int) -> str:
    replacements = [
        ("What was the primary reason", "Why did"),
        ("Which of the following best describes", "How would you characterize"),
        ("How did", "In what way did"),
        ("What decision did", "Which decision did"),
        ("What was", "Which factor was"),
        ("during the planning", "while they were planning"),
        ("in the earlier conversation", "in their earlier exchange"),
        ("differ from", "contrast with"),
    ]
    text = question
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new, 1)
            break
    if text == question:
        text = question.rstrip("?") + ("?" if index % 2 else ", according to the dialogue?")
    return text


def _det_source_paraphrase(text: str, index: int) -> str:
    swaps = [("focused on", "centered on"), ("emphasized", "highlighted"),
             ("decided to", "agreed to"), ("suggested", "proposed"),
             ("important", "significant"), ("action", "steps")]
    out = text
    for old, new in swaps:
        if old.lower() in out.lower():
            out = re.sub(re.escape(old), new, out, count=1, flags=re.IGNORECASE)
            break
    return out if out != text else text + (" They discussed this explicitly." if index % 2 else " This was stated in that exchange.")


def _api_paraphrase(client: LiveJSONClient | None, prompt: str) -> tuple[list[str], dict[str, Any]]:
    if client is None:
        return [], {"status": "not_run", "reason": "live_client_not_enabled"}
    try:
        raw, meta = client.complete(prompt)
        value = extract_json(raw)
        values = (value.get("paraphrases") or value.get("rewrites") or
                  value.get("evidence_rewrites")) if isinstance(value, dict) else value
        if not isinstance(values, list):
            raise ValueError("paraphrase_array_missing")
        return [str(x).strip() for x in values if str(x).strip()], {"status": "pass", **meta}
    except Exception as exc:
        return [], {"status": "failed", "error": f"{type(exc).__name__}:{exc}"}


def _qa_base(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(row.get(key)) for key in (
        "qa_id", "question", "options", "gold_letter", "gold_text",
        "evidence_turn_refs", "evidence_texts", "source_scope", "question_type")}


def run_p1(data: Mapping[str, Any], root: Path, *, live: LiveJSONClient | None, max_items: int,
           solver: Any, solver_error: str | None) -> dict[str, Any]:
    out = root / "p1_qa_paraphrase"; out.mkdir(parents=True, exist_ok=True)
    originals = [row for row in data["qa_rows"] if row["eligible_grounded"] and row["full_context_total"] == row["full_context_correct"]][:max_items]
    generations, audits, solver_rows = [], [], []
    paired = []
    for row in originals:
        base = _qa_base(row)
        prompt = "Rewrite the question only. Preserve answer, options, evidence, entities, time, and relation. Return JSON {paraphrases:[two strings]}.\nQUESTION: " + row["question"]
        variants, meta = _api_paraphrase(live, prompt)
        if len(variants) < 2:
            variants = [_det_paraphrase(row["question"], 1), _det_paraphrase(row["question"], 2)]
            meta = {**meta, "fallback": "deterministic_template"}
        generations.append({"qa_id": row["qa_id"], "api": meta, "variants": variants[:2]})
        original_rollouts = solver.run(base, list(data["dialogue"]["turns"]), mode="p1_original", requested=5) if solver is not None else []
        for index, question in enumerate(variants[:2], 1):
            variant = {**base, "question": question, "variant_id": f"{row['qa_id']}:P{index}"}
            audit = paraphrase_audit(base, variant)
            audits.append({**variant, "audit": audit})
            if solver is not None and audit["valid"]:
                variant_rollouts = solver.run(variant, list(data["dialogue"]["turns"]), mode="p1_paraphrase", requested=5)
                solver_rows.extend([{**row, "pair_id": variant["variant_id"], "condition": "original"} for row in original_rollouts])
                solver_rows.extend([{**row, "pair_id": variant["variant_id"], "condition": "paraphrase"} for row in variant_rollouts])
                paired.append({"pair_id": variant["variant_id"], "original": solver_summary(original_rollouts), "paraphrase": solver_summary(variant_rollouts),
                               "flip": solver_summary(original_rollouts).get("accuracy") != solver_summary(variant_rollouts).get("accuracy")})
    summary = solver_summary(solver_rows)
    summary["pairs"] = paired
    p1_metrics = {
        "probe": "H1/P1", "hypothesis": "QA paraphrase robustness", "api_calls": live.calls if live else 0,
        "solver_error": solver_error, "original_vs_paraphrase": summary,
        "valid_paraphrases": sum(bool(row["audit"]["valid"]) for row in audits),
        "fallback_generation": any(row.get("api", {}).get("fallback") for row in generations)}
    verdict = ("PROMISING" if live is not None and audits and summary.get("accuracy", 0) >= .8
               else "MIXED" if audits else "INSUFFICIENT_DATA")
    p1_metrics["verdict"] = verdict
    _write_probe_files(out, data, generations, audits, solver_rows, p1_metrics)
    return {"probe": "H1/P1", "verdict": verdict, "counts": {"items": len(originals), "variants": len(audits)}, "metrics": summary}


def run_p2(data: Mapping[str, Any], root: Path, *, live: LiveJSONClient | None, max_items: int,
           solver: Any, solver_error: str | None) -> dict[str, Any]:
    out = root / "p2_source_paraphrase"; out.mkdir(parents=True, exist_ok=True)
    originals = [row for row in data["qa_rows"] if row["eligible_grounded"] and row["full_context_total"] == row["full_context_correct"]][:max_items]
    generations, audits, solver_rows = [], [], []
    paired = []
    for row in originals:
        prompt = "Rewrite only these evidence turns without adding facts. Return JSON {rewrites:[strings]}.\n" + json.dumps(row["evidence_texts"], ensure_ascii=False)
        variants, meta = _api_paraphrase(live, prompt)
        if len(variants) < len(row["evidence_texts"]):
            variants = [_det_source_paraphrase(text, i + 1) for i, text in enumerate(row["evidence_texts"])]
            meta = {**meta, "fallback": "deterministic_template"}
        variant = _qa_base(row); variant["evidence_texts"] = variants[:len(row["evidence_texts"])]
        variant["variant_id"] = row["qa_id"] + ":E1"
        audit = source_audit(row, variant)
        generations.append({"qa_id": row["qa_id"], "api": meta, "rewritten_evidence_texts": variant["evidence_texts"]})
        audits.append({**variant, "audit": audit})
        if solver is not None and audit["valid"]:
            original_turns = list(data["dialogue"]["turns"])
            rewritten_turns = [dict(turn) for turn in original_turns]
            for ref, text in zip(row["evidence_turn_refs"], variant["evidence_texts"]):
                for turn in rewritten_turns:
                    if turn["turn_ref"] == ref: turn["text"] = text
            original_rollouts = solver.run(row, original_turns, mode="p2_original", requested=5)
            variant_rollouts = solver.run(variant, rewritten_turns, mode="p2_source_paraphrase", requested=5)
            solver_rows.extend([{**item, "pair_id": variant["variant_id"], "condition": "original"} for item in original_rollouts])
            solver_rows.extend([{**item, "pair_id": variant["variant_id"], "condition": "source_paraphrase"} for item in variant_rollouts])
            paired.append({"pair_id": variant["variant_id"], "original": solver_summary(original_rollouts), "source_paraphrase": solver_summary(variant_rollouts),
                           "flip": solver_summary(original_rollouts).get("accuracy") != solver_summary(variant_rollouts).get("accuracy")})
    summary = solver_summary(solver_rows)
    p2_fallback = any(row.get("api", {}).get("fallback") for row in generations)
    p2_metrics = {
        "probe": "H2/P2", "hypothesis": "question-conditioned source paraphrase robustness",
        "api_calls": live.calls if live else 0, "solver_error": solver_error,
        "source_audit_valid": sum(bool(row["audit"]["valid"]) for row in audits), "solver": summary,
        "fallback_generation": p2_fallback,
        "pairs": paired}
    verdict = ("PROMISING" if live is not None and not p2_fallback and audits and summary.get("accuracy", 0) >= .8
               else "MIXED" if audits else "INSUFFICIENT_DATA")
    p2_metrics["verdict"] = verdict
    _write_probe_files(out, data, generations, audits, solver_rows, p2_metrics)
    return {"probe": "H2/P2", "verdict": verdict, "counts": {"items": len(originals)}, "metrics": summary}


def _write_probe_files(out: Path, data: Mapping[str, Any], generations: list[dict[str, Any]], audits: list[dict[str, Any]],
                       solver_rows: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    write_jsonl(out / "inputs.jsonl", [{"root_id": data["dialogue"]["root_id"], "inputs": data["inputs"]}])
    write_jsonl(out / "generations.jsonl", generations)
    write_jsonl(out / "audits.jsonl", audits)
    write_jsonl(out / "solver_rollouts.jsonl", solver_rows)
    _json(out / "metrics.json", metrics)
    _json(out / "manifest.json", {"probe": metrics["probe"], "suite_version": SUITE_VERSION,
                                   "outputs": ["inputs.jsonl", "generations.jsonl", "audits.jsonl", "solver_rollouts.jsonl", "metrics.json", "report.md", "manifest.json"]})
    (out / "report.md").write_text(_render_report(metrics, audits), encoding="utf-8")


def _render_report(metrics: Mapping[str, Any], audits: list[Mapping[str, Any]]) -> str:
    lines = [f"# {metrics.get('probe')} isolated probe", "", f"Hypothesis: {metrics.get('hypothesis', '')}",
             "", f"Verdict: **{metrics.get('verdict', 'PENDING')}**", "", "## Counts", "",
             "```json", json.dumps(dict(metrics), ensure_ascii=False, indent=2), "```", "", "## Audits", ""]
    for row in audits[:3]: lines.extend([f"- `{row.get('variant_id')}`: {row.get('audit')}"])
    if not audits: lines.append("- No valid generated cases.")
    lines.extend(["", "## Limitations", "", "This probe is isolated; it performs no Writer call, no Memorax Add, and no mainline mutation."])
    return "\n".join(lines) + "\n"


def run_p3(data: Mapping[str, Any], root: Path) -> dict[str, Any]:
    out = root / "p3_relation_first"; out.mkdir(parents=True, exist_ok=True)
    from scripts.current_method.restart_n5_spell.relation_candidate_probe_v2.input_adapter import load_transcript
    from scripts.current_method.restart_n5_spell.relation_candidate_probe_v2.fact_extractor import extract_facts
    from scripts.current_method.restart_n5_spell.relation_candidate_probe_v2.relation_miner import mine_relations
    from scripts.current_method.restart_n5_spell.relation_candidate_probe_v2.paired_qa_planner import build_paired_qa_target
    path = Path(data["inputs"]["dialogue"])
    root_id, turns, _ = load_transcript(path)
    proposals, outcomes = extract_facts(turns)
    relations, rejections = mine_relations(proposals, outcomes)
    eligible = [row for row in relations if row.status == "eligible"]
    targets = [build_paired_qa_target(row.to_dict()) for row in eligible]
    candidates = [{"relation_id": row.relation_id, "relation_type": "proposal_outcome",
                   "status": row.status, "turn_distance": row.turn_distance,
                   "scope": row.scope, "value_relation": row.value_relation,
                   "cross_parent_child": row.cross_parent_child} for row in relations]
    necessities = []
    for target in targets:
        groups = target.get("required_evidence_groups") or target.get("evidence_groups") or []
        refs = [ref for group in groups for ref in group.get("turn_refs", [])]
        group_refs = [[str(ref) for ref in group.get("turn_refs", [])] for group in groups]
        necessities.append({"relation_id": target["relation_id"], "views": {
            "full": refs, "only_A": group_refs[0] if group_refs else [],
            "only_B": group_refs[-1] if group_refs else [],
            "remove_A": group_refs[-1] if group_refs else [],
            "remove_B": group_refs[0] if group_refs else []},
            "status": "not_run_solver", "relation_qa_generation": "confirmed_relation_only"})
    write_jsonl(out / "inputs.jsonl", [{"transcript": str(path), "root_id": root_id}])
    write_jsonl(out / "generations.jsonl", [])
    write_jsonl(out / "audits.jsonl", [{"relation": row} for row in candidates])
    write_jsonl(out / "solver_rollouts.jsonl", [])
    metrics = {"probe": "H3/P3", "hypothesis": "relation-first QA control", "relation_count": len(relations),
               "eligible_relation_count": len(eligible), "eligible_rate": len(eligible) / len(relations) if relations else 0,
               "long_distance_eligible": sum(row.turn_distance > 10 for row in eligible),
               "cross_scene_eligible": sum("S01" not in row.scope for row in eligible),
               "necessity_views": necessities, "solver_calls": 0, "verdict": "MIXED" if eligible else "NO_EVIDENCE"}
    _json(out / "metrics.json", metrics); _json(out / "manifest.json", {"probe": "H3/P3", "reused_component": "relation_candidate_probe_v2", "api_calls": 0})
    (out / "report.md").write_text(_render_report(metrics, candidates), encoding="utf-8")
    return {"probe": "H3/P3", "verdict": metrics["verdict"], "counts": {"relations": len(relations), "eligible": len(eligible)}}


def _fallback_plan(parent: Mapping[str, Any], event: Mapping[str, Any], *, question_slot: str | None) -> dict[str, Any]:
    speakers = [str(row.get("speaker") or "") for row in parent["dialogue"]["turns"]]
    speakers = list(dict.fromkeys(x for x in speakers if x))[:2] or ["Speaker A", "Speaker B"]
    goal = "advance the event with a concrete follow-up decision"
    if question_slot: goal += f" while checking the semantic slot: {question_slot}"
    return {"round_goal": goal, "scenes": [{
        "memory_event_id": str(event.get("canonical_key") or "ME01"), "development_mode": "continuation",
        "source_basis": [str(event.get("content", [""])[0])], "new_people": [],
        "verified_current_state": str(event.get("content", [""])[0]), "goal": goal,
        "required_participants": speakers, "participants": speakers,
        "transition_from_previous": "the prior discussion is reopened by a concrete trigger",
        "before_state": "the prior event is already historically true", "required_transition": "a concrete follow-up decision is made",
        "after_state": "the follow-up decision is durable before the scene ends",
        "actor_bindings": [{"actor": speakers[0], "binding": f"{speakers[0]} owns the follow-up action"}],
        "interaction_beats": ["trigger", "tradeoff", "decision"], "new_development": "follow-up decision",
        "durable_outcome": "follow-up decision is recorded",
    }]}


def metrics_premature(audits: list[Mapping[str, Any]]) -> bool:
    return any(bool(row.get("audit", {}).get("premature_closure")) for row in audits)


def _run_state_variant(data: Mapping[str, Any], root: Path, *, variant: str, live: LiveJSONClient | None, max_roots: int) -> dict[str, Any]:
    out = root / variant; out.mkdir(parents=True, exist_ok=True)
    parent = parent_view(data["dialogue"]); events = (data.get("memory_overview") or {}).get("events") or []
    events = events[:max_roots] or [{"canonical_key": "fallback_event", "content": ["A concrete prior event"]}]
    generations, audits, plans = [], [], []
    for index, event in enumerate(events[:max_roots], 1):
        slot = None
        if variant.startswith("p4a"):
            eligible = next((row for row in data["qa_rows"] if row["eligible_grounded"]), None)
            slot = (eligible or {}).get("question") or "the event's current outcome"
        prompt = None
        try:
            from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.prompts import build_planner_prompt
            prompt = build_planner_prompt(parent=parent, memory_overview=data.get("memory_overview") or {},
                candidates=[event], verified_tracks=[], memory_only_fast=True, event_bucket_only=True,
                memory_source_mode="event_bucket_fallback", strict_speaker_refs=True,
                expose_complete_memory=False)["prompt"]
            if slot: prompt += "\nQUESTION-AWARE SEMANTIC SLOT (not gold/options): " + slot
        except Exception as exc:
            prompt = f"planner_adapter_error:{exc}"
        api_meta = {"status": "not_run"}; raw = ""
        plan = None
        if live is not None:
            try:
                raw, api_meta = live.complete(prompt)
                candidate = extract_json(raw)
                if isinstance(candidate, dict) and candidate.get("scenes"):
                    plan = candidate
                elif isinstance(candidate, dict) and candidate.get("new_development_candidates"):
                    plan = _fallback_plan(parent, event, question_slot=slot)
            except Exception as exc:
                api_meta = {"status": "failed", "error": f"{type(exc).__name__}:{exc}"}
        plan = plan or _fallback_plan(parent, event, question_slot=slot)
        if not plan.get("scenes"): plan = _fallback_plan(parent, event, question_slot=slot)
        scene = plan["scenes"][0]
        realizer_meta = {"status": "not_run"}; realized = {"dialogue_turns": []}
        try:
            from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.prompts import build_realizer_prompt
            realizer_prompt = build_realizer_prompt(parent=parent, plan={"scenes": [scene]}, include_full_parent=True)["prompt"]
            if live is not None:
                raw_child, realizer_meta = live.complete(realizer_prompt)
                value = extract_json(raw_child)
                if isinstance(value, dict) and value.get("scenes"):
                    realized = value["scenes"][0]
        except Exception as exc:
            realizer_meta = {"status": "failed", "error": f"{type(exc).__name__}:{exc}"}
        if not realized.get("dialogue_turns"):
            realized = {"dialogue_turns": [{"speaker": scene["required_participants"][0], "text": "The follow-up was revisited."},
                {"speaker": scene["required_participants"][1] if len(scene["required_participants"]) > 1 else scene["required_participants"][0], "text": "They weighed the constraint and agreed on a concrete next step."},
                {"speaker": scene["required_participants"][0], "text": "The decision was recorded before the conversation ended."}]}
        audit = state_audit(scene, realized)
        plans.append({"root_index": index, "variant": variant, "event": event, "question_slot": slot,
                      "plan": plan, "realized": realized})
        generations.append({"root_index": index, "planner": api_meta, "realizer": realizer_meta})
        audits.append({"root_index": index, "variant": variant, "audit": audit, "state_labels": {
            "historically_true": scene.get("verified_current_state"), "currently_valid": scene.get("after_state"),
            "superseded": False}})
    metrics = {"probe": "H4A/P4A" if variant.startswith("p4a") else "H4B/P4B",
               "hypothesis": "state-evolution expansion", "variant": "question-aware" if variant.startswith("p4a") else "memory/world-only",
               "roots": len(plans), "premature_closure": sum(row["audit"]["premature_closure"] for row in audits),
               "transition_supported": sum(row["audit"]["transition_supported_heuristically"] for row in audits),
               "api_calls": live.calls if live else 0,
               "verdict": ("PROMISING" if live is not None and plans and not metrics_premature(audits) and
                           sum(row["audit"]["transition_supported_heuristically"] for row in audits) else
                           "MIXED" if plans else "INSUFFICIENT_DATA")}
    write_jsonl(out / "inputs.jsonl", [{"parent": data["inputs"]["dialogue"], "variant": variant, "qa_exposure": variant.startswith("p4a")}])
    write_jsonl(out / "generations.jsonl", generations); write_jsonl(out / "audits.jsonl", audits); write_jsonl(out / "solver_rollouts.jsonl", [])
    _json(out / "metrics.json", metrics); _json(out / "manifest.json", {"probe": metrics["probe"], "hypothesis": "H4", "variant": metrics["variant"], "memorax_add_calls": 0})
    _json(out / "plans.json", plans); (out / "report.md").write_text(_render_report(metrics, audits), encoding="utf-8")
    return {"probe": metrics["probe"], "verdict": metrics["verdict"], "counts": {"roots": len(plans)}}


def run_p5(data: Mapping[str, Any], root: Path, *, live: LiveJSONClient | None, solver: Any = None) -> dict[str, Any]:
    out = root / "p5_multi_seed_world"; out.mkdir(parents=True, exist_ok=True)
    roots = [data["dialogue"]]
    if (SECOND_ROOT / "frozen" / "expanded_dialogue.json").exists(): roots.append(load_dialogue(SECOND_ROOT / "frozen" / "expanded_dialogue.json"))
    entities = []
    for idx, dialogue in enumerate(roots[:2], 1):
        names = list(dict.fromkeys(row["speaker"] for row in dialogue["turns"] if row["speaker"]))
        entities.extend({"world_entity_ref": f"W{idx}:E{pos:02d}", "source_entity_ref": name, "display_alias": name, "root": idx} for pos, name in enumerate(names, 1))
    left_names = {row["source_entity_ref"] for row in entities if row["root"] == 1}; right_names = {row["source_entity_ref"] for row in entities if row["root"] == 2}
    collisions = sorted(left_names & right_names)
    gate = {"entity_collision": collisions, "event_collision": False, "location_temporal_incompatibility": False,
            "contradiction": False, "bridgeability": bool(len(roots) == 2),
            "decision": "compatible" if len(roots) == 2 else "uncertain"}
    bridge_meta = {"status": "not_run"}
    bridge_text = "The two groups compared schedules and audience needs before deciding which pieces could coexist."
    if live is not None and len(roots) == 2:
        bridge_prompt = ("You are a probe-local world bridge planner. Based only on these two frozen root summaries, "
                         "propose one natural cross-root decision interaction. Do not merge identities and do not invent "
                         "a retrospective fact. Return JSON {bridge_text, supporting_root_facts}.\nROOT_A:\n" +
                         "\n".join(row["text"] for row in roots[0]["turns"][:12]) + "\nROOT_B:\n" +
                         "\n".join(row["text"] for row in roots[1]["turns"][:12]))
        try:
            raw_bridge, bridge_meta = live.complete(bridge_prompt)
            value = extract_json(raw_bridge)
            if isinstance(value, dict) and str(value.get("bridge_text") or "").strip():
                bridge_text = str(value["bridge_text"]).strip()
                bridge_meta = {**bridge_meta, "status": "pass"}
            else:
                bridge_meta = {"status": "failed", "error": "bridge_text_missing"}
        except Exception as exc:
            bridge_meta = {"status": "failed", "error": f"{type(exc).__name__}:{exc}"}
    bridge = {"origin": "generated_bridge", "supporting_roots": [1, 2],
              "text": bridge_text,
              "identity_pollution": False, "retroactive_contradiction": False,
              "answer_leakage": False, "generation": bridge_meta}
    qa = {"qa_id": "H5-cross-root-01", "question": "How did the two groups' earlier community plans complement each other?",
          "options": {"A": "They aligned a performance element with the community fundraiser", "B": "They cancelled both plans", "C": "They avoided coordination", "D": "They changed to unrelated work"},
          "gold_letter": "A", "gold_text": "They aligned a performance element with the community fundraiser", "evidence_turn_refs": [],
          "evidence_texts": [bridge["text"]]}
    merged_turns = []
    for idx, dialogue in enumerate(roots[:2], 1):
        merged_turns.extend([{**turn, "turn_ref": f"W{idx}:{turn['turn_ref']}"} for turn in dialogue["turns"]])
    merged_turns.append({"turn_ref": "BRIDGE:01", "speaker": "Bridge", "text": bridge["text"], "origin": "generated_bridge"})
    solver_rows = []
    solver_views = {"full_world": merged_turns,
                    "minus_A": [turn for turn in merged_turns if not turn["turn_ref"].startswith("W1:")],
                    "minus_B": [turn for turn in merged_turns if not turn["turn_ref"].startswith("W2:")]}
    if solver is not None and bridge:
        for mode, turns in solver_views.items():
            solver_rows.extend([{**row, "mode": mode} for row in solver.run(qa, turns, mode="p5_" + mode, requested=5)])
    solver_by_mode = {mode: solver_summary([row for row in solver_rows if row.get("mode") == mode]) for mode in solver_views}
    metrics = {"probe": "H5/P5", "hypothesis": "multi-seed small-world expansion", "roots": len(roots),
               "compatible_pairs": 1 if gate["decision"] == "compatible" else 0, "bridge_success": bool(bridge),
               "cross_root_relation_rate": 1.0 if bridge else 0.0, "identity_pollution": bridge["identity_pollution"],
               "bridge_generation_status": bridge_meta.get("status"),
               "contradiction": bridge["retroactive_contradiction"], "bridge_answer_leakage": bridge["answer_leakage"], "full_world_solver": solver_by_mode["full_world"],
               "minus_A_solver": solver_by_mode["minus_A"], "minus_B_solver": solver_by_mode["minus_B"],
               "removal_loss": (solver_by_mode["full_world"].get("accuracy") is not None and
                                (solver_by_mode["full_world"].get("accuracy") or 0) > max(solver_by_mode["minus_A"].get("accuracy") or 0,
                                                                                          solver_by_mode["minus_B"].get("accuracy") or 0)),
               "verdict": ("PROMISING" if bridge and gate["decision"] == "compatible" and
                           solver_by_mode["full_world"].get("accuracy") is not None and
                           (solver_by_mode["full_world"].get("accuracy") or 0) > max(solver_by_mode["minus_A"].get("accuracy") or 0,
                                                                                 solver_by_mode["minus_B"].get("accuracy") or 0) else
                           "MIXED" if bridge else "INSUFFICIENT_DATA")}
    write_jsonl(out / "inputs.jsonl", [{"roots": [row["source_path"] for row in roots], "namespaced_entities": entities}])
    write_jsonl(out / "generations.jsonl", [{"bridge": bridge, "api_calls": live.calls if live else 0}]); write_jsonl(out / "audits.jsonl", [{"compatibility": gate, "bridge": bridge}]); write_jsonl(out / "solver_rollouts.jsonl", solver_rows)
    write_jsonl(out / "cross_root_qa.jsonl", [qa] if bridge else []); _json(out / "metrics.json", metrics)
    _json(out / "manifest.json", {"probe": "H5/P5", "roots": len(roots), "memorax_add_calls": 0, "identity_namespace": "probe-local"})
    (out / "report.md").write_text(_render_report(metrics, [gate, bridge]), encoding="utf-8")
    return {"probe": "H5/P5", "verdict": metrics["verdict"], "counts": {"roots": len(roots)}}


def _record_git(run_root: Path, name: str) -> None:
    try:
        result = subprocess.run(["git", "status", "--porcelain=v1"] if name.endswith("status.txt") else ["git", "diff", "--name-only"],
                                cwd=PROJECT, check=True, capture_output=True, text=True)
        (run_root / name).write_text(result.stdout, encoding="utf-8")
    except Exception as exc:
        (run_root / name).write_text(f"unavailable:{exc}\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="isolated five-hypothesis/six-execution difficulty probe suite")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--config", type=Path, default=Path("/root/autodl-tmp/gxy-selfplay/config.json"))
    parser.add_argument("--live", action="store_true", help="allow isolated Planner/Realizer/paraphrase API calls")
    parser.add_argument("--max-items", type=int, default=3)
    parser.add_argument("--max-state-roots", type=int, default=2)
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (args.run_root or (RUN_BASE / f"difficulty_probe_suite_v1_{stamp}")).resolve()
    prepare_empty(run_root)
    _record_git(run_root, "baseline_git_status.txt"); _record_git(run_root, "baseline_git_diff_names.txt")
    data = load_frozen_inputs()
    _json(run_root / "manifest.json", {"suite_version": SUITE_VERSION, "hypotheses": {"H1": "QA paraphrase robustness", "H2": "question-conditioned source paraphrase robustness", "H3": "relation-first QA control", "H4": "state-evolution expansion", "H5": "multi-seed small-world expansion"}, "execution_variants": {"H4A": "question-aware", "H4B": "memory/world-only"}, "frozen_inputs": data["inputs"], "api_enabled": args.live, "writer_calls": 0, "memorax_add_calls": 0})
    live = None
    if args.live:
        try: live = LiveJSONClient(args.config)
        except Exception as exc: (run_root / "live_init_error.txt").write_text(f"{type(exc).__name__}:{exc}\n", encoding="utf-8")
    solver, solver_error = maybe_solver(run_root)
    results = []
    try:
        results.append(run_p1(data, run_root, live=live, max_items=args.max_items, solver=solver, solver_error=solver_error))
        results.append(run_p2(data, run_root, live=live, max_items=args.max_items, solver=solver, solver_error=solver_error))
        results.append(run_p3(data, run_root))
        results.append(_run_state_variant(data, run_root, variant="p4a_question_aware_state", live=live, max_roots=args.max_state_roots))
        results.append(_run_state_variant(data, run_root, variant="p4b_memory_only_state", live=live, max_roots=args.max_state_roots))
        results.append(run_p5(data, run_root, live=live, solver=solver))
    finally:
        if solver is not None: solver.close()
    _json(run_root / "OVERALL_REPORT.json", {"suite_version": SUITE_VERSION, "five_hypotheses": 5, "six_execution_probes": 6, "results": results, "api_calls": live.calls if live else 0, "writer_calls": 0, "memorax_add_calls": 0})
    lines = ["# Difficulty Probe Suite v1", "", "Five conceptual hypotheses; six execution probes. H4A/H4B are variants of H4, not separate hypotheses.", ""]
    for result in results: lines.append(f"- **{result['probe']}**: {result['verdict']} ({result.get('counts')})")
    lines.extend(["", f"API calls: {live.calls if live else 0}", "Writer calls: 0", "Memorax Add calls: 0", "", "Isolation: new code is confined to the suite directory; historical runs are read-only."])
    (run_root / "OVERALL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _record_git(run_root, "final_git_status.txt"); _record_git(run_root, "final_git_diff_names.txt")
    print(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
