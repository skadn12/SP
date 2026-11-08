"""One offline-scoped live prompt-layout diagnostic; never a fresh collection run."""
from __future__ import annotations
import json, shutil
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import candidate_runner as c
from candidate_transport import Caller

SOURCE_RUN = HERE / "runs" / "fresh_b2_candidates_01"
OUT = HERE / "diagnostics" / "prompt_tail_b2_s01_01"
ROOT = c.ROOT
ROUTE = c.r.ROUTES[1]
SCENE = "S01"

def write(path, value):
    c.r.base.write(path, value)

def prepare():
    if OUT.exists():
        raise RuntimeError("diagnostic_output_already_exists")
    OUT.mkdir(parents=True)
    shutil.copytree(SOURCE_RUN / "feedback", OUT / "feedback")
    for name in ("provider_profile.json", "FORMAL_BASELINE.json"):
        shutil.copyfile(SOURCE_RUN / name, OUT / name)
    card = c.r.base.read(SOURCE_RUN / "extractions" / "01.json")["card"]
    history = c.r.base.read(SOURCE_RUN / "FORMAL_BASELINE.json")["turns"]
    guidance = c.r.base.dataset_cards.build_realization_guidance(card)
    manifest = {
        "version": "h4_candidate_prompt_tail_diagnostic_v41",
        "scope": "offline_diagnostic",
        "model": c.r.MODEL,
        "logical_max": 1,
        "http_max": 2,
        "training_pool_eligible": False,
        "training_started": False,
        "formal_history_sha256": c.r.base.digest(history),
        "source_run": str(SOURCE_RUN),
        "source_id": card["source_id"],
        "source_card_sha256": c.r.base.digest(card),
        "provenance": "diagnostic_replay_reuses_consumed_source_no_fresh_yield",
    }
    write(OUT / "MANIFEST.json", manifest)
    write(OUT / "INPUT.json", {"root_id": ROOT, "route": ROUTE, "scene_id": SCENE,
        "history": history, "formal_baseline_turns": len(history), "card": guidance,
        "source_card_raw": card, "source_run": str(SOURCE_RUN),
        "policy": "full_history_and_consumed_card_diagnostic_only"})
    return history, guidance, card

def run():
    history, card, raw_card = prepare()
    # Use the production request builder so provider parameters, full history and
    # frozen feedback remain identical. Add one explicit final instruction after
    # the old feedback suffix, which is the variable under test.
    payload, proof = c.request(OUT, ROOT, ROUTE, SCENE, "realizer", history, history, card)
    final = """

FINAL DIAGNOSTIC TASK — READ THIS LAST AND FOLLOW IT OVER EARLIER EXAMPLES:
This is an offline prompt-layout diagnostic, not formal training collection.
Return exactly one JSON Realizer object for root_id cc-train-episode-104814,
route dataset_new_event, scene_id S01. Use only Emily and James as direct speakers.
Create a genuinely new occasion after the complete supplied history. Do not copy,
paraphrase, summarize, or continue the existing R02:A:S03 workshop rollout,
Friday break, email pause, or farewell dialogue. Any turn that repeats an existing
history turn is a failure for this diagnostic.
The assigned source card is the college-information event with its exact card_id
and six mapped actions: request a Cambridge-centre college, propose a college,
accept/inquire about cost, report free admission, request postcode, provide postcode.
Adapt those source actors to Emily and James in this root; source words, source
speakers, and source outcome are not already root facts. Realize a new conversation
that uses the card core, with concrete new information or a decision. Keep plans,
intentions, attempts, and completed actions distinct. Do not invent an old-history
fact. Return 12-16 turns if coherent, status scene, the exact card_id, and no review
or summary. If this cannot be done, return status no_scene with an honest reason;
do not output old dialogue.
END FINAL DIAGNOSTIC TASK.
"""
    payload = json.loads(json.dumps(payload))
    payload["messages"][0]["content"] += final
    proof = {**proof, "diagnostic_scope": "prompt_tail_after_frozen_feedback",
             "final_instruction_sha256": c.r.base.digest(final),
             "formal_history_sha256": c.r.base.digest(history),
             "source_card_sha256": c.r.base.digest(raw_card),
             "training_pool_eligible": False, "semantic_certified": False}
    write(OUT / "REQUEST_INPUT.json", {"payload": payload, "proof": proof})
    caller = Caller(OUT)
    raw, directory = caller.call("cc-train-episode-104814-B-S01-prompt-tail-diagnostic", payload, proof)
    write(OUT / "RESPONSE_BINDING.json", {"request": str(directory), "raw_response": raw,
        "raw_response_sha256": c.r.base.digest(raw), "source_run": str(SOURCE_RUN),
        "provenance": "diagnostic_replay", "training_pool_eligible": False})
    try:
        parsed = c.r.parse_object(raw)
    except Exception as error:
        parsed = None
        write(OUT / "PARSE.json", {"status": "REJECT", "error": str(error)})
    else:
        write(OUT / "PARSE.json", {"status": "PARSED", "parsed": parsed,
            "scene_id": parsed.get("scene_id"), "source_card_id": parsed.get("dataset_card_id"),
            "turns": len(parsed.get("turns", [])) if isinstance(parsed.get("turns"), list) else 0})
    write(OUT / "DIAGNOSTIC_REPORT.json", {"scope": "offline_diagnostic",
        "provenance": "diagnostic_replay", "fresh_collection_body_tokens": 0,
        "training_admitted_body_tokens": 0, "semantic_certified": False,
        "source_reservation_created": False, "source_run": str(SOURCE_RUN),
        "request": str(directory), "parsed": parsed is not None})
    print(json.dumps({"status": "COMPLETE", "request": str(directory),
        "parsed": parsed is not None, "turns": len(parsed.get("turns", [])) if isinstance(parsed, dict) and isinstance(parsed.get("turns"), list) else 0,
        "scene_id": parsed.get("scene_id") if isinstance(parsed, dict) else None,
        "source_card_id": parsed.get("dataset_card_id") if isinstance(parsed, dict) else None,
        "fresh_collection_body_tokens": 0, "training_admitted_body_tokens": 0}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    run()
