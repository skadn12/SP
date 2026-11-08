"""Record the immutable rejection of the 31457 GENERIC B2 READY review.

The reviewer cited a native state JSON fragment as if it were the utterance for
turn 14.  The fragment is not a contiguous substring of the assigned native
utterance, so this READY response cannot be mechanically materialized or
accepted.  This script only writes a rejection receipt and never calls CLI.
"""
from pathlib import Path
import hashlib, json
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parents[1]
ROOT = "cc-train-episode-31457"; ARM = "GENERIC"; SCENE = "OET1:B:S02"

def bind(path):
    p = Path(path).resolve(); b = p.read_bytes()
    return {"path": str(p), "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}

def read(path): return json.loads(Path(path).read_text())

def main():
    sd = BASE / "sources" / ROOT / ARM / SCENE.replace(":", "_")
    out = sd / "NATIVE_EVIDENCE_REJECTION.json"
    assert not out.exists(), "REJECTION_ALREADY_RECORDED"
    failure = read(sd / "FAILURE.json")
    review_dir = BASE / "cli_calls" / f"{ROOT}__{ARM}__{SCENE.replace(':','_')}__REVIEW"
    parsed = read(review_dir / "PARSED_RESPONSE.json")
    packet = read(BASE / "sources" / ROOT / ARM / "SOURCE_AUTHOR_INPUT.json")
    native = packet["native_events"][1]
    native_record = read(native["native_event_binding"]["path"])
    turn14 = next(t for t in native_record["turns"] if str(t["turn_id"]) == "14")
    cited = next(e for e in parsed["checks"]["location"]["evidence"]
                 if e["turn_ref"] == "MUL0870.json:turn_id=14")
    assert cited["quote"] not in turn14["utterance"]
    assert failure["semantic_retry"] is False
    value = {
        "schema_version": "oet1_native_evidence_rejection_v1",
        "status": "FAIL_EVIDENCE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "root_id": ROOT, "condition": ARM, "scene_ref": SCENE,
        "classification": "READY_RESPONSE_REJECTED_NONCONTIGUOUS_NATIVE_QUOTE",
        "reason": "The location evidence cites a native state JSON fragment, but the quote is not a contiguous substring of the assigned native turn 14 utterance. Native state annotations are outside the permitted evidence text domain.",
        "original_failure_binding": bind(sd / "FAILURE.json"),
        "original_review_result_binding": bind(review_dir / "RESULT.json"),
        "original_review_parsed_binding": bind(review_dir / "PARSED_RESPONSE.json"),
        "native_event_binding": native["native_event_binding"],
        "native_turn_ref": "MUL0870.json:turn_id=14",
        "cited_quote": cited["quote"],
        "cited_quote_sha256": hashlib.sha256(cited["quote"].encode()).hexdigest(),
        "actual_native_turn_14_utterance": turn14["utterance"],
        "actual_native_utterance_sha256": hashlib.sha256(turn14["utterance"].encode()).hexdigest(),
        "exact_contiguous_substring": False,
        "review_decision": parsed["decision"],
        "source_or_gold_ambiguity_failure": parsed.get("source_or_gold_ambiguity_failure"),
        "semantic_retry": False,
        "review_materialization": "FORBIDDEN",
        "accepted": False,
        "planned_slot_preserved": True,
        "descendants_stopped": True,
        "no_new_cli_calls": True,
        "cli_artifacts_unchanged_binding": [bind(p) for p in sorted(review_dir.rglob("*")) if p.is_file()],
        "validator_rule": "quote must be a contiguous substring of assigned native record utterance; state/frames are not readable evidence text",
        "rejection_code_binding": bind(__file__),
    }
    with out.open("x") as f:
        json.dump(value, f, ensure_ascii=False, indent=2); f.write("\n")
    print(json.dumps({"status": value["status"], "binding": bind(out)}, ensure_ascii=False))

if __name__ == "__main__": main()
