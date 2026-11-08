"""Build citation-only packets after the raw response freeze is committed.

These packets omit final predictions, canonical Gold and reviewer anchor refs.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE / "runs/B1"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert read(RUN / "RAW_RESPONSE_FREEZE.json")["status"] == "COMMITTED"
    requests = read(HERE / "REQUESTS.json")["requests"]
    public = {r["candidate_id"]: r for r in read(HERE / "PUBLIC_QUESTIONS.json")["questions"]}
    packets = []
    for req in requests:
        response = read(RUN / "responses" / (req["request_id"] + ".json"))
        assert response["status"] == "RETURNED"
        answer = json.loads(response["raw_output"])
        source_path = Path(req["source_path"])
        assert sha(source_path) == req["source_sha256"]
        source = read(source_path)
        by_ref = {turn["turn_ref"]: turn for turn in source["turns"]}
        item_rows = []
        for display_label in req["labels"]:
            canonical = req["display_to_canonical"][display_label]
            evidence = answer["item_evidence"][display_label]
            item_rows.append({
                "display_label": display_label,
                "statement": public[req["candidate_id"]]["items"][canonical],
                "judgment": evidence["judgment"],
                "selected_citation_refs": evidence["turn_refs"],
                "selected_citations": [by_ref[ref] for ref in evidence["turn_refs"]],
            })
        packets.append({
            "request_id": req["request_id"], "candidate_id": req["candidate_id"],
            "root_id": req["root_id"], "variant_id": req["variant_id"],
            "question": req["question"], "source_path": str(source_path),
            "source_sha256": req["source_sha256"], "source_turn_count": len(source["turns"]),
            "items": item_rows,
        })
    output = {"status": "BLIND_MODEL_SELECTED_CITATION_SEMANTIC_REVIEW",
              "rubric": read(HERE / "PROTOCOL.json")["citation_rubric"],
              "instruction": "Review each statement and its actual model-selected citations in full source context. Do not substitute another turn. For each item classify citation_sufficient_for_first_full_report, identify any incorrect judgment (the model judgment here is expected supported), distinguish later-only/plan-vs-completion/speaker/object/predicate/qualifier omissions. Do not inspect or infer the final order. Return an item-level review with reasons.",
              "requests": packets}
    out = HERE / "CITATION_ONLY_REVIEW_PACKET.json"
    with out.open("x") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"WROTE {out} requests={len(packets)} citation_sets={sum(len(x['items']) for x in packets)}")


if __name__ == "__main__":
    main()
