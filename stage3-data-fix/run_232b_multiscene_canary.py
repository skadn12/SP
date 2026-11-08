from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.current_method import native_api_graph_dialogue_expansion as native
from scripts.current_method import stage3_claim_ir
from scripts.current_method.stage3_truth_first_discourse import canonical_sha256


RUN_ID = "232b_qwen32b_typed_semantic_multiscene_canary_n1"
DEFAULT_OUTPUT = (
    ROOT / "base_restart_current_method_validation" / RUN_ID
)
SOURCE_ARTIFACT = (
    ROOT
    / "base_restart_current_method_validation"
    / "230l_truth_first_temporal_n1_retry11"
    / "generation"
    / "graph_artifacts"
    / "c8e83824a84b81d1baa0d7ffcd52cd852b1e7c66254764b34c30ead42ef07d71.json"
)

EXPECTED_VALUES = ["Cedar Hall", "London", "the Midwest", "community gym"]
EXPECTED_STATUSES = ["belief", "active", "outdated", "canceled"]

TYPED_FACT_SPECS = [
    {
        "fact_ref": "S1.1",
        "predicate": "believes_destination",
        "status": "belief_unconfirmed",
        "must_mean": (
            "John currently considers Cedar Hall a possible destination, but it "
            "remains unconfirmed; it is not an old, replaced, deferred, or definite "
            "plan."
        ),
    },
    {
        "fact_ref": "S1.2",
        "predicate": "current_destination",
        "status": "active",
        "must_mean": (
            "London is John's current active destination or committed destination "
            "plan; it is not rumored, canceled, outdated, or merely former."
        ),
    },
    {
        "fact_ref": "S2.1",
        "predicate": "formerly_considered_destination",
        "status": "outdated_no_longer_applicable",
        "must_mean": (
            "John previously considered the Midwest as a destination, but it is now "
            "outdated and no longer an applicable option; it was not canceled."
        ),
    },
    {
        "fact_ref": "S2.2",
        "predicate": "canceled_destination_plan",
        "status": "canceled_before_effect",
        "must_mean": (
            "John's plan to use the community gym as a destination was canceled "
            "before taking effect; it was not completed, active, merely outdated, "
            "or only rumored canceled."
        ),
    },
]

MODE_SPECS = [
    {
        "name": "inquiry_then_confirmation",
        "instruction": (
            "One participant asks cautiously about current and uncertain travel "
            "information; John answers naturally without turning uncertainty into "
            "confirmation."
        ),
    },
    {
        "name": "recollection_then_correction",
        "instruction": (
            "One participant recalls older travel possibilities; John corrects which "
            "item is outdated and which was canceled without inventing a replacement "
            "destination."
        ),
    },
]


def _load_frozen_claims() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    artifact = json.loads(SOURCE_ARTIFACT.read_text(encoding="utf-8"))
    graph = artifact["current_event_plan"]["_truth_first_materialized"]["graph"]
    claim_bundle = stage3_claim_ir.compile_claim_ir(graph)
    claims = []
    for claim in claim_bundle["claims"]:
        canonical = claim.get("canonical_support_claim") or {}
        if (
            claim.get("claim_type") == "node"
            and canonical.get("subject") == "John"
            and canonical.get("predicate") == "destination"
        ):
            claims.append(copy.deepcopy(claim))
    claims.sort(
        key=lambda claim: int(
            (
                claim["canonical_support_claim"].get("time") or {}
            ).get("global_time_index")
            or -1
        )
    )
    claims = claims[:4]
    values = [claim["canonical_support_claim"]["objects"][0] for claim in claims]
    statuses = [claim["canonical_support_claim"]["status"] for claim in claims]
    if values != EXPECTED_VALUES or statuses != EXPECTED_STATUSES:
        raise RuntimeError(
            "source frozen claims changed: "
            f"values={values!r}, statuses={statuses!r}"
        )
    return graph, claims


def _typed_facts(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    facts = []
    for claim, spec in zip(claims, TYPED_FACT_SPECS):
        canonical = claim["canonical_support_claim"]
        facts.append(
            {
                **copy.deepcopy(spec),
                "claim_id_local_only": claim["claim_id"],
                "subject": canonical["subject"],
                "value": canonical["objects"][0],
            }
        )
    return facts


def _api_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in fact.items() if key != "claim_id_local_only"}
        for fact in facts
    ]


def _checkpoint(last_speaker: str) -> dict[str, Any]:
    return {
        "participants": ["Tim", "John"],
        "last_speaker": last_speaker,
        "previous_scene_topic": "travel possibilities",
        "tone": "calm_casual",
        "dialogue_goal": (
            "Continue to older alternatives without reopening or summarizing the "
            "closed prior exchange."
        ),
        "prior_exchange_state": "closed_do_not_reopen",
        "expressed_claim_ids": ["S1.1", "S1.2"],
        "ancillary_facts_carried_forward": False,
        "raw_prior_turns_included": 0,
    }


def _checkpoint_audit(
    checkpoint: dict[str, Any], scene1_turns: list[dict[str, Any]]
) -> dict[str, Any]:
    serialized = json.dumps(checkpoint, ensure_ascii=False, sort_keys=True)
    forbidden = [
        "Cedar Hall",
        "London",
        "believes_destination",
        "current_destination",
        "belief_unconfirmed",
        "active",
    ]
    leaks = [value for value in forbidden if value.casefold() in serialized.casefold()]
    exact_turn_matches = [
        turn.get("text")
        for turn in scene1_turns
        if str(turn.get("text") or "")
        and str(turn.get("text")) in serialized
    ]
    return {
        "pass": not leaks and not exact_turn_matches,
        "forbidden_prior_semantic_matches": leaks,
        "exact_prior_turn_matches": exact_turn_matches,
        "prior_fact_value_count": sum(
            serialized.casefold().count(value.casefold())
            for value in ("Cedar Hall", "London")
        ),
        "raw_prior_turn_count": 0,
        "ancillary_fact_count": 0,
    }


def _prompt(
    scene_index: int,
    facts: list[dict[str, Any]],
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any]:
    rules = [
        "Write one natural direct conversation scene of 4-8 turns using only Tim and John.",
        "Express every typed semantic fact naturally in your own words; first-person I and second-person you are allowed when grounded by speaker roles.",
        "Preserve each typed predicate, subject, value, and status exactly. The must_mean field is immutable truth, including every stated exclusion.",
        "Light greetings, reactions, and non-factual social context are allowed, but added context is ancillary and must not change, resolve, cause, replace, or add a destination fact.",
        "Return exactly one JSON object whose sole top-level key is turns. Do not return fact refs, claim IDs, supports, quotes, evidence, explanations, or sidecars.",
    ]
    if scene_index == 2:
        rules.insert(
            4,
            "The checkpoint contains only discourse state. Never mention its IDs, never infer missing prior facts, and never reopen or restate the closed prior exchange.",
        )
    return {
        "mode": "stage3_truth_first_typed_semantic_multiscene_live_canary_v2",
        "scene_index": scene_index,
        "scene_count": 2,
        "discourse_mode": copy.deepcopy(MODE_SPECS[scene_index - 1]),
        "speaker_roster": ["Tim", "John"],
        "setting": "John and Tim continue one conversation about John's travel information.",
        "semantic_facts": _api_facts(facts),
        "discourse_checkpoint": copy.deepcopy(checkpoint),
        "prior_scene_raw_text_exposed": False,
        "rules": rules,
        "response_schema": {
            "turns": [
                {"speaker": "Tim or John", "text": "natural visible dialogue"}
            ]
        },
    }


def _basic_response_audit(response: Any) -> dict[str, Any]:
    issues = []
    if not isinstance(response, dict) or set(response) != {"turns"}:
        issues.append("response_must_contain_only_turns")
        turns = []
    else:
        turns = response.get("turns")
        if not isinstance(turns, list):
            issues.append("turns_must_be_array")
            turns = []
    if not 4 <= len(turns) <= 8:
        issues.append("turn_count_must_be_4_to_8")
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            issues.append(f"turn_{index}_must_be_object")
            continue
        if turn.get("speaker") not in {"Tim", "John"}:
            issues.append(f"turn_{index}_speaker_invalid")
        if not str(turn.get("text") or "").strip():
            issues.append(f"turn_{index}_text_empty")
    return {"pass": not issues, "issues": issues, "turn_count": len(turns)}


def _prompt_sha256(prompt: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            prompt,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-live", action="store_true")
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or (
        ROOT / "base_restart_current_method_validation" / args.run_id
    )

    graph, claims = _load_frozen_claims()
    facts = _typed_facts(claims)
    scene1_facts = facts[:2]
    scene2_facts = facts[2:]
    preflight_checkpoint = _checkpoint("John")
    preflight_audit = _checkpoint_audit(preflight_checkpoint, [])
    if not preflight_audit["pass"]:
        raise RuntimeError(f"checkpoint preflight failed: {preflight_audit}")

    if not args.execute_live:
        print(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "preflight": "pass",
                    "model": "Qwen/Qwen3-32B",
                    "typed_predicates": [fact["predicate"] for fact in facts],
                    "typed_statuses": [fact["status"] for fact in facts],
                    "checkpoint_audit": preflight_audit,
                    "output_dir_fresh": not output_dir.exists(),
                },
                sort_keys=True,
            )
        )
        return 0

    if output_dir.exists():
        raise RuntimeError(f"refusing to reuse output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))["api"]
    api_args = argparse.Namespace(
        challenger_api_key=config["api_key"],
        challenger_base_url=config["base_url"],
        challenger_model="Qwen/Qwen3-32B",
        challenger_temperature=0.25,
        challenger_max_tokens=768,
        max_llm_retries=0,
        llm_timeout=120,
        api_yield_normalizers=False,
    )
    system_prompt = (
        "Write one short scene in a continuing natural conversation from immutable "
        "typed semantics. Preserve epistemic and lifecycle meaning exactly. Return "
        "JSON containing only a turns array and no evidence metadata."
    )
    graph_hash_before = canonical_sha256(graph)
    records = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    checkpoint = None

    for scene_index, scene_facts in enumerate((scene1_facts, scene2_facts), start=1):
        prompt = _prompt(scene_index, scene_facts, checkpoint)
        checkpoint_audit = None
        if scene_index == 2:
            scene1_turns = records[0]["response"]["turns"]
            checkpoint_audit = _checkpoint_audit(checkpoint or {}, scene1_turns)
            if not checkpoint_audit["pass"]:
                raise RuntimeError(
                    f"Scene 2 checkpoint leaked prior content: {checkpoint_audit}"
                )

        started = time.monotonic()
        response, meta = native.call_raw_json_api(
            api_args,
            prompt,
            system_prompt=system_prompt,
            required_keys=("turns",),
            call_kind="api_truth_single_scene_probe",
            max_tokens_override=768,
            max_retries_override=0,
            temperature_override=0.25,
            include_raw_response=True,
        )
        audit = _basic_response_audit(response)
        record = {
            "run_id": args.run_id,
            "scene_id": f"MS1:S{scene_index}",
            "scene_index": scene_index,
            "discourse_mode": MODE_SPECS[scene_index - 1]["name"],
            "model": "Qwen/Qwen3-32B",
            "semantic_facts_local": copy.deepcopy(scene_facts),
            "discourse_checkpoint": copy.deepcopy(checkpoint),
            "checkpoint_audit": copy.deepcopy(checkpoint_audit),
            "prior_scene_raw_text_exposed": False,
            "prompt_payload": prompt,
            "system_prompt": system_prompt,
            "prompt_sha256": _prompt_sha256(prompt),
            "response": response,
            "api_meta": meta,
            "basic_response_audit": audit,
            "latency_ms": int(round((time.monotonic() - started) * 1000)),
            "api_retry_count": 0,
            "local_repair_applied": False,
            "manual_review_required": True,
        }
        records.append(record)
        with (output_dir / "scenes.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        usage = meta.get("usage") if isinstance(meta, dict) else {}
        for key in usage_total:
            usage_total[key] += int((usage or {}).get(key) or 0)
        print(
            json.dumps(
                {
                    "scene": scene_index,
                    "api_used": bool(meta.get("used")),
                    "finish_reason": meta.get("finish_reason"),
                    "basic_response_pass": audit["pass"],
                    "usage": usage,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not meta.get("used") or not audit["pass"]:
            break
        if scene_index == 1:
            checkpoint = _checkpoint(response["turns"][-1]["speaker"])

    summary = {
        "schema_version": "stage3_typed_semantic_multiscene_canary_v2",
        "run_id": args.run_id,
        "model": "Qwen/Qwen3-32B",
        "scene_count_requested": 2,
        "scene_count_completed": len(records),
        "physical_api_calls": len(records),
        "api_retries": 0,
        "local_repairs": 0,
        "usage": usage_total,
        "graph_sha256_before": graph_hash_before,
        "graph_sha256_after": canonical_sha256(graph),
        "graph_mutated": canonical_sha256(graph) != graph_hash_before,
        "typed_predicates": [fact["predicate"] for fact in facts],
        "typed_statuses": [fact["status"] for fact in facts],
        "scene2_checkpoint_audit": (
            records[1].get("checkpoint_audit") if len(records) == 2 else None
        ),
        "prior_raw_turns_exposed_to_scene2": 0,
        "ancillary_facts_carried_to_scene2": False,
        "manual_review_required": True,
        "proceed_to_more_multiscene": False,
        "scenes": records,
    }
    _write_json(output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "completed_scenes": len(records),
                "usage": usage_total,
                "graph_mutated": summary["graph_mutated"],
                "summary": str(output_dir / "summary.json"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if len(records) == 2 else 2


if __name__ == "__main__":
    raise SystemExit(main())
