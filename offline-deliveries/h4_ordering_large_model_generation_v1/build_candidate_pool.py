"""Build a source-audited candidate pool with explicit evidence-ref normalization."""

from copy import deepcopy
import json

import generate as engine


def main() -> None:
    audit = engine.read("AUDIT.json")
    supported = {row["candidate_id"] for row in audit["candidate_reviews"] if row["source_supported"]}
    draft = engine.read("repair_01/CANDIDATES.json")
    source = engine.source()
    exact_refs = {row["turn_ref"] for row in source["turns"]}
    normalizations, checks, pool = [], [], []
    for original in draft["questions"]:
        row = deepcopy(original)
        for label, evidence in row["item_evidence"].items():
            ref = evidence["turn_ref"]
            if ref in exact_refs:
                continue
            alternatives = {engine.ROOT_ID + ":" + ref}
            if ref.startswith(engine.ROOT_ID + ":"):
                alternatives.add(ref[len(engine.ROOT_ID) + 1:])
            matches = alternatives & exact_refs
            if len(matches) != 1:
                raise ValueError("reference_normalization_not_unique")
            resolved = matches.pop()
            evidence["turn_ref"] = resolved
            normalizations.append({"candidate_id": row["candidate_id"], "item_id": label,
                                   "original_ref": ref, "canonical_ref": resolved,
                                   "operation": "EXACT_UNIQUE_ROOT_PREFIX_NORMALIZATION"})
        restored = deepcopy(row)
        for label in restored["item_evidence"]:
            restored["item_evidence"][label]["turn_ref"] = original["item_evidence"][label]["turn_ref"]
        if restored != original:
            raise ValueError("model_content_changed_beyond_reference_normalization")
        errors = engine.check_candidate(row, source["turns"])
        checks.append({"candidate_id": row["candidate_id"], "mechanical_errors_after_normalization": errors,
                       "source_supported_by_independent_audit": row["candidate_id"] in supported})
        if row["candidate_id"] in supported and not errors:
            pool.append(row)
    if len(pool) != len(supported):
        raise ValueError("audited_supported_pool_incomplete")
    provenance = deepcopy(draft["provenance"])
    provenance.update(
        question_items_and_gold_modified_by_codex=False,
        evidence_ref_normalization_applied=bool(normalizations),
        source_response_path=str(engine.HERE / "repair_01/response.raw.txt"),
        source_response_sha256=engine.sha((engine.HERE / "repair_01/response.raw.txt").read_bytes()),
        content_audit_path=str(engine.HERE / "AUDIT.json"),
        content_audit_sha256=engine.sha((engine.HERE / "AUDIT.json").read_bytes()),
    )
    engine.write_new("REF_NORMALIZATION.json", {
        "scope": "EVIDENCE_TURN_REF_ONLY", "original_model_output_preserved": True,
        "question_items_and_gold_unchanged": True, "normalizations": normalizations, "checks": checks,
    })
    engine.write_new("READY_FOR_EVALUATION.json", {
        "status": "SOURCE_SUPPORTED_CANDIDATES_NOT_SOLVER_SCREENED", "provenance": provenance,
        "candidate_count": len(pool), "questions": pool,
        "excluded_candidate_ids": [row["candidate_id"] for row in draft["questions"] if row["candidate_id"] not in supported],
        "solver_calls": 0, "selected_by_ground_full_policy": None,
        "display_permutations_created": False, "sft_exported": False, "training_started": False,
        "diagnostics": ["Model ignored the requested shuffled display: all repaired Gold arrays are I1 through I6.",
                        "Use the established mapped display permutations for later Solver screening; none have been run here.",
                        "Transcript-order questions do not establish dates of recalled or planned events."],
    })
    lines = ["# Source-Supported Ordering Candidates", "",
             f"Generator: `{provenance['returned_model']}` via SiliconFlow.",
             f"Root: `{engine.ROOT_ID}`; frozen complete R03 source, 188 turns.",
             "Question text, items, and Gold below are unchanged remote-model output.",
             "Two evidence IDs in question 06 were transparently normalized in the JSON pool.",
             "These are candidates, not training selections. No Solver evaluation or training was run.",
             "The model returned chronological canonical labels; later evaluation must use mapped shuffled displays.", ""]
    for row in pool:
        lines += [f"## {row['candidate_id']}: {row['topic']}", "", row["question"], ""]
        lines += [f"- {label}: {text}" for label, text in row["items"].items()]
        lines += ["", "Gold: `" + ", ".join(row["gold"]) + "`", "", row["order_reason"], ""]
    engine.write_new("READY_QUESTIONS.md", "\n".join(lines), raw=True)
    print(json.dumps({"ready_candidate_count": len(pool), "reference_normalizations": len(normalizations),
                      "model": provenance["returned_model"], "solver_calls": 0, "training_started": False}))


if __name__ == "__main__":
    main()
