"""Package fresh direct-Codex drafts with the unchanged V2.2.1 answer/freezer helpers.

This is a research-run adapter, not a new generator or a material approval.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(PROJECT / "scripts/current_method"))
import base_qa_answers_v2_2_1 as answers
import base_qa_option_freeze_v2_2_1 as freezer
import base_qa_pipeline_v2_1 as legacy


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(value, *keys):
    if isinstance(value, list):
        return value
    for key in keys:
        if isinstance(value.get(key), list):
            return value[key]
    raise ValueError("Unrecognized author envelope")


def package(letter):
    review_letter = "a" if letter == "d" else letter
    base = HERE / "authors" / letter
    candidates = rows(read(base / "CANDIDATES.json"), "candidates", "items")
    targets_doc = read(base / "TARGETS.json")
    targets = rows(targets_doc, "targets", "items")
    discovery = read(base / "DISCOVERY.json")
    receipt = read(base / "TARGETS_FREEZE.json")
    # Keep exact author files and expose no private target/Gold to the reviewer.
    author_files = {name: sha(base / name) for name in ("DISCOVERY.json", "TARGETS.json", "TARGETS_FREEZE.json", "CANDIDATES.json")}
    frozen_target_hash = next(receipt[key] for key in ("targets_sha256", "targets_file_sha256", "target_file_sha256") if key in receipt)
    frozen_discovery_hash = next(receipt[key] for key in ("discovery_sha256", "discovery_file_sha256") if key in receipt)
    assert frozen_target_hash == author_files["TARGETS.json"], "Frozen semantic target drift"
    assert frozen_discovery_hash == author_files["DISCOVERY.json"], "Discovery changed after target freeze"
    index = []
    for original in candidates:
        candidate = answers.normalize_candidate(original)
        qid, root = candidate["candidate_id"], candidate["root_id"]
        target = next(row for row in targets if row.get("root_id") == root)
        assert answers.typed_equal(candidate["semantic_answer"], answers.validate_semantic(target["semantic_answer"])), "Candidate changed frozen semantic target"
        source_row = next(row for row in read(HERE / "INPUT_FREEZE.json")["roots"] if row["root_id"] == root)
        source_path = Path(source_row["source"]["path"])
        assert sha(source_path) == source_row["source"]["sha256_bytes"]
        source = read(source_path)
        source_index = {turn["turn_ref"]: turn for turn in source["turns"]}
        assert 6 <= len(candidate["options"]) <= 8
        g = len(candidate["gold_answer"]["option_ids"])
        assert g in range(2, 7)
        quote_errors = []
        evidence_refs = set()
        for oid, analysis in candidate["item_analysis"].items():
            assert analysis["should_select"] == (oid in candidate["gold_answer"]["option_ids"])
            for ev in analysis["evidence"]:
                turn = source_index.get(ev["turn_ref"])
                evidence_refs.add(ev["turn_ref"])
                if not turn or turn["speaker"] != ev["speaker"] or ev["quote"] not in turn["text"]:
                    quote_errors.append({"option_id": oid, "evidence": ev})
        binding = {"path": str(source_path), "sha256": sha(source_path), "bytes": source_path.stat().st_size}
        # This manifest records missing approval; it grants no permission.
        scope = legacy.source_manifest(source, binding, challenger_id=candidate["author_id"])
        scope["source_warnings"] = source.get("warnings", [])
        scope["scope_interpretation"] = "Every source turn remains visible. This run records actual item evidence and does not approve synthetic material or reject an entire dialogue based on unrelated turns."
        bundle_dir = HERE / "bundles" / qid
        freezer.freeze_bundle(bundle_dir, candidate, source, scope,
                              reviewer_id="fresh-cardinality-reviewer-" + review_letter,
                              context_id="/root/cardinality_blind_" + review_letter,
                              issuer_id="fresh-cardinality-orchestrator",
                              receipt_id="fresh-cardinality-" + qid,
                              extra_paths=(base / "TARGETS.json", base / "TARGETS_FREEZE.json", base / "DISCOVERY.json", base / "CANDIDATES.json"))
        packet = freezer.frozen_packet(bundle_dir)
        assert [row["turn_ref"] for row in packet["packet"]["source"]["turns"]] == [row["turn_ref"] for row in source["turns"]]
        public = {**packet, "reviewer_id": "fresh-cardinality-reviewer-" + review_letter,
                  "isolation_receipt": freezer.reviewer_receipt_projection(bundle_dir)}
        new(HERE / "blind_input" / review_letter / (qid + ".json"), public)
        mapping = freezer.frozen_mapping(bundle_dir)
        display_gold = answers.map_answer(candidate["gold_answer"], "multi_select", {v: k for k, v in mapping.items()})
        style = legacy.style_audit({**candidate, "gold": candidate["gold_answer"]["option_ids"]})
        words = {oid: len(text.split()) for oid, text in candidate["options"].items()}
        gold = set(candidate["gold_answer"]["option_ids"])
        wrong = set(words) - gold
        long_gold = bool(wrong) and min(words[oid] for oid in gold) > max(words[oid] for oid in wrong)
        short_gold = bool(wrong) and max(words[oid] for oid in gold) < min(words[oid] for oid in wrong)
        evidence_origins = {ref: source_index[ref].get("origin", "UNKNOWN") for ref in sorted(evidence_refs)}
        new(HERE / "preaudit" / (qid + ".json"), {
            "candidate_id": qid, "answer_contract": "PASS", "exact_quotes": "FAIL" if quote_errors else "PASS", "quote_errors": quote_errors,
            "target_binding": {"target_sha256": freezer.fingerprint(target), "targets_file_sha256": author_files["TARGETS.json"], "author_freeze_receipt": receipt},
            "author_files": author_files, "display_gold": display_gold,
            "option_count": len(candidate["options"]), "gold_count": g,
            "all_true_control": g == len(candidate["options"]),
            "display_to_option_id": mapping, "option_words": words,
            "gold_strictly_longest_k": long_gold, "gold_strictly_shortest_k": short_gold,
            "style_flags_for_review": style, "evidence_origin_by_turn": evidence_origins,
            "evidence_scope": legacy.formal_evidence_gate(scope, sorted(evidence_refs)),
            "formal_qa_approved": False,
            "full_typed_discovery_graph_validation": "NOT_CLAIMED: direct semantic exploration artifacts are preserved, not fabricated into an executable graph",
        })
        index.append({"candidate_id": qid, "root_id": root, "option_count": len(candidate["options"]), "gold_count": g, "display_gold": display_gold,
                      "bundle_dir": str(bundle_dir), "blind_input": str(HERE / "blind_input" / review_letter / (qid + ".json")), "review_group": review_letter,
                      "difficulty_label": candidate.get("difficulty_label"), "task_operation": candidate.get("task_operation"),
                      "all_true_control": g == len(candidate["options"]), "formal_qa_approved": False})
    new(HERE / ("INDEX_PRIVATE_" + letter + ".json"), index)
    print(json.dumps(index, ensure_ascii=False), flush=True)


def compare(letter):
    index = read(HERE / ("INDEX_PRIVATE_" + letter + ".json"))
    result = []
    for entry in index:
        qid = entry["candidate_id"]
        bundle_dir = Path(entry["bundle_dir"])
        raw_path = HERE / "blind_outputs" / entry.get("review_group", letter) / (qid + ".json")
        raw = raw_path.read_bytes()
        # Freeze exact received bytes BEFORE looking at the private answer.
        formal_parse = freezer.freeze_blind_response(bundle_dir, raw)
        bundle = freezer.load_bundle(bundle_dir)
        response = freezer._loads(raw)
        material_prefixes = ("EVIDENCE:NOT_FORMALLY_ADMITTED:", "EVIDENCE:TURN_NOT_FORMALLY_ALLOWED:")
        protocol_errors = [error for error in formal_parse["errors"] if not error.startswith(material_prefixes)]
        protocol_valid = not protocol_errors
        canonical, matches = None, None
        if protocol_valid and response["decision"] == "SOLVED":
            canonical = answers.map_answer(response["independent_answer"], "multi_select", bundle["render_contract"]["display_to_canonical"])
            matches = answers.typed_equal(canonical, bundle["candidate"]["gold_answer"])
        review = {"candidate_id": qid, "review_mode": "same_model_blind_fresh_context",
                  "isolation": "fresh fork without author history; reviewer restricted to public inputs; no claim of OS filesystem access control",
                  "raw_output_path": str(raw_path), "raw_output_sha256": hashlib.sha256(raw).hexdigest(),
                  "protocol_valid_ignoring_material_eligibility": protocol_valid,
                  "protocol_errors": protocol_errors, "reviewer_decision": response.get("decision"),
                  "reviewer_display_answer": response.get("independent_answer"), "reviewer_canonical_answer": canonical,
                  "matches_author": matches, "formal_parser_status": formal_parse["status"],
                  "formal_parser_errors": formal_parse["errors"], "formal_qa_approved": False,
                  "content_disposition": "GOLD_AGREEMENT_REQUIRES_QUALITY_REVIEW" if matches else "HOLD",
                  "source_disposition": "ITEM_MATERIAL_ADMISSION_PENDING"}
        new(HERE / "comparisons" / (qid + ".json"), review)
        result.append({"candidate_id": qid, "protocol_valid": protocol_valid, "matches_author": matches, "reviewer_decision": response.get("decision")})
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("package", "compare"))
    parser.add_argument("letter", choices=("a", "b", "c", "d"))
    args = parser.parse_args()
    {"package": package, "compare": compare}[args.mode](args.letter)
