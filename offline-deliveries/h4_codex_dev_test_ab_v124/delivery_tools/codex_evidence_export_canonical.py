"""Export reviewed evidence without dropping citations or inventing judgments.

This module checks recorded evidence against source text; it does not establish
semantic entailment. A reviewer must identify ``primary_citation`` when a
statement has multiple citations. The full citation list and the reviewer's
relation/reason are preserved alongside that native-compatible primary span.
No model, Solver, training process, or output-file writer is invoked here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


VERSION = "source_review_lossless_evidence_export_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _turns(source: Mapping[str, Any]) -> dict[str, dict]:
    rows = source.get("turns")
    _require(isinstance(rows, list) and bool(rows), "source_turns_required")
    _require(all(isinstance(t, dict) and all(_text(t.get(k)) for k in
                 ("turn_ref", "speaker", "text")) for t in rows), "source_turn_schema")
    result = {t["turn_ref"]: t for t in rows}
    _require(len(result) == len(rows), "duplicate_source_turn_ref")
    return result


def _citation(citation: Any, turns: dict[str, dict], where: str,
              *, speaker_required: bool = True) -> dict:
    _require(isinstance(citation, dict), where + ":citation_not_object")
    turn = turns.get(citation.get("turn_ref"))
    _require(turn is not None, where + ":citation_turn_ref_invalid")
    _require(_text(citation.get("quote")) and citation["quote"] in turn["text"],
             where + ":citation_quote_not_exact")
    if speaker_required or "speaker" in citation:
        _require(citation.get("speaker") == turn["speaker"],
                 where + ":citation_speaker_mismatch")
    return {**deepcopy(citation), "speaker": turn["speaker"]}


def _primary(judgment: dict, citations: list[dict], override: Any, where: str) -> tuple[dict, str]:
    recorded = judgment.get("primary_citation")
    if recorded is not None and override is not None:
        _require(recorded == override, where + ":primary_designations_conflict")
    selected = recorded if recorded is not None else override
    origin = "source_judgment" if recorded is not None else "explicit_reviewer_designation"
    if selected is None:
        _require(len(citations) == 1, where + ":reviewed_primary_citation_required")
        selected, origin = citations[0], "sole_reviewed_citation"
    # An explicit primary must select a saved citation, not supply a new span.
    _require(isinstance(selected, dict) and selected in citations,
             where + ":primary_not_in_reviewed_citations")
    return deepcopy(selected), origin


def export_question_evidence(
    question: dict,
    private_gold: dict,
    source: dict,
    source_judgment: dict,
    *,
    asset_source_judgments: dict | None = None,
    design: dict | None = None,
    primary_citations: Mapping[str, dict] | None = None,
) -> dict:
    """Return a new native-compatible candidate with lossless reviewed evidence.

    ``source_judgment`` is the candidate's entry from PUBLIC_SOURCE_JUDGMENTS,
    not its Gold. Optional ``design`` and ``asset_source_judgments`` associate
    option labels with reviewed statement IDs. Both must be supplied together.
    ``primary_citations`` is only for an explicit, separately recorded reviewer
    designation when an older review lacks ``primary_citation``; the exporter
    never chooses a multi-citation primary by list order or lexical guessing.
    Inputs are not mutated. Bad or missing references fail before export.
    """
    turns = _turns(source)
    cid = question.get("candidate_id")
    _require(_text(cid) and private_gold.get("candidate_id") == cid, "candidate_gold_binding")
    _require(source_judgment.get("candidate_id", cid) == cid, "candidate_judgment_binding")
    _require(question.get("root_id") == source.get("root_id"), "source_root_binding")
    _require((asset_source_judgments is None) == (design is None), "asset_design_pair_required")
    if design is not None:
        _require(design.get("candidate_id") == cid, "candidate_design_binding")
    qtype = question.get("question_type")
    _require(qtype in ("multi_select", "ordering"), "unsupported_question_type")
    field = "options" if qtype == "multi_select" else "items"
    labels = tuple("ABCDEF") if qtype == "multi_select" else tuple(f"I{i}" for i in range(1, 7))
    values = question.get(field)
    # v124 delivery adapter: normalize JSON object insertion order only.
    # Label-to-text mappings, private Gold and every reviewed citation stay exact.
    _require(isinstance(values, dict) and set(values) == set(labels), "six_canonical_labels_required")
    original_mapping = deepcopy(values)
    question = deepcopy(question)
    question[field] = {label: values[label] for label in labels}
    values = question[field]
    _require(values == original_mapping, "presentation_normalization_changed_mapping")
    _require(isinstance(values, dict) and tuple(values) == labels, "six_canonical_labels_required")
    _require(all(_text(v) for v in values.values()), "proposition_text_required")
    gold = private_gold.get("gold")
    _require(isinstance(gold, list) and 2 <= len(gold) <= 6 and
             len(set(gold)) == len(gold) and set(gold) <= set(labels), "gold_labels_invalid")
    evidence = private_gold.get("evidence")
    _require(isinstance(evidence, dict) and set(evidence) == set(labels), "private_evidence_incomplete")
    judgments = source_judgment.get(field)
    _require(isinstance(judgments, dict) and set(judgments) == set(labels), "source_judgments_incomplete")
    overrides = dict(primary_citations or {})
    _require(set(overrides) <= set(labels), "unknown_primary_designation_label")
    exported = {}
    for label in labels:
        j = judgments[label]
        where = cid + ":" + label
        _require(isinstance(j, dict) and j.get("proposition") == values[label], where + ":proposition_mismatch")
        _require(_text(j.get("reason")), where + ":review_reason_required")
        entry = deepcopy(j)
        if qtype == "multi_select":
            _require(j.get("relation") in ("supports", "contradicts"), where + ":review_relation_invalid")
            _require(j.get("scope_valid") is True, where + ":review_scope_invalid")
            _require((j["relation"] == "supports") == (label in gold), where + ":review_gold_conflict")
            citations = j.get("citations")
            _require(isinstance(citations, list) and bool(citations), where + ":citations_required")
            _require(evidence[label] == citations, where + ":private_review_citations_mismatch")
            for index, citation in enumerate(citations):
                _citation(citation, turns, where + f":citation{index}")
            selected, primary_origin = _primary(j, citations, overrides.get(label), where)
            primary = _citation(selected, turns, where + ":primary")
            if design is not None:
                sid = (design.get("labels_to_statement_ids") or {}).get(label)
                asset = (asset_source_judgments.get("statements") or {}).get(sid)
                _require(isinstance(asset, dict), where + ":asset_judgment_missing")
                _require(asset.get("proposition") == values[label], where + ":asset_proposition_mismatch")
                _require(asset.get("relation") == j["relation"], where + ":asset_relation_mismatch")
                asset_citations = asset.get("citations")
                _require(isinstance(asset_citations, list) and bool(asset_citations), where + ":asset_citations_required")
                for index, citation in enumerate(asset_citations):
                    _citation(citation, turns, where + f":asset_citation{index}")
                _require(_text(asset.get("reason")), where + ":asset_reason_required")
                entry["statement_id"] = sid
                entry["asset_source_judgment"] = deepcopy(asset)
        else:
            _require(label not in overrides, where + ":ordering_primary_is_fixed_anchor")
            _require(all(j.get(key) is True for key in ("supported", "actor_ok", "modality_ok", "state_ok",
                         "neutral_stem_ok", "single_event_ok")), where + ":ordering_review_invalid")
            anchor = j.get("order_anchor")
            _require(evidence[label] == anchor, where + ":private_review_anchor_mismatch")
            primary = _citation(anchor, turns, where + ":anchor", speaker_required=False)
            _require(turns[primary["turn_ref"]]["text"].count(primary["quote"]) == 1,
                     where + ":ordering_anchor_not_unique")
            alternatives = j.get("alternative_matching_anchors")
            _require(isinstance(alternatives, list), where + ":ordering_alternatives_required")
            for index, citation in enumerate(alternatives):
                _citation(citation, turns, where + f":alternative{index}", speaker_required=False)
            refs = j.get("support_refs")
            _require(isinstance(refs, list) and bool(refs) and all(ref in turns for ref in refs),
                     where + ":ordering_support_refs_invalid")
            _require(primary["turn_ref"] in refs, where + ":ordering_anchor_not_in_support_refs")
            if design is not None:
                _require((design.get("fixed_turns") or {}).get(label) == primary["turn_ref"],
                         where + ":ordering_outside_fixed_anchor")
            entry["citations"] = [deepcopy(anchor), *deepcopy(alternatives)]
            primary_origin = "reviewed_order_anchor"
        # Compatibility fields point at the reviewed primary; all original
        # citations/judgment metadata remain in the entry, without truncation.
        entry.update({key: primary[key] for key in ("turn_ref", "speaker", "quote")})
        entry["primary_citation"] = deepcopy(primary)
        entry["primary_citation_origin"] = primary_origin
        exported[label] = entry
    if qtype == "ordering":
        _require(len(gold) == 6 and source_judgment.get("derived_order") == gold,
                 "ordering_review_gold_conflict")
        positions = {ref: index for index, ref in enumerate(turns)}
        observed_order = sorted(labels, key=lambda label: (
            positions[exported[label]["turn_ref"]],
            turns[exported[label]["turn_ref"]]["text"].index(exported[label]["quote"])))
        _require(observed_order == gold, "ordering_source_gold_conflict")
    out = deepcopy(question)
    out.update(gold=deepcopy(gold), role=private_gold.get("role"))
    out["option_evidence" if qtype == "multi_select" else "item_evidence"] = exported
    if qtype == "ordering":
        # Fixed order spans remain exactly the reviewed anchors, with source
        # speaker resolved for metadata. Other evidence is stored separately.
        out["order_anchor_spans"] = {
            label: _citation(judgments[label]["order_anchor"], turns, cid + ":" + label,
                             speaker_required=False) for label in labels}
    out["evidence_export"] = {"version": VERSION, "semantic_entailment_checked_by_exporter": False,
                              "all_reviewed_citations_preserved": True}
    return out


def export_root_questions(root_folder: Path | str) -> list[dict]:
    """Read a finalized root and export CONTENT_PASS candidates; never write.

    File hashes bind source, public questions, Gold, final semantic review,
    asset judgments and label-to-statement design before evidence is exported.
    Older multi-citation reviews without an explicit primary fail closed.
    """
    folder = Path(root_folder)
    names = ("CUMULATIVE_SOURCE.json", "PUBLIC_QUESTIONS.json", "PRIVATE_GOLD.json",
             "PUBLIC_SOURCE_JUDGMENTS.json", "ASSET_SOURCE_JUDGMENTS.json", "PRIVATE_DESIGN.json",
             "FINAL_CROSSCHECK.json", "PUBLIC_ASSETS.json")
    raw = {name: (folder / name).read_bytes() for name in names}
    docs = {name: json.loads(value) for name, value in raw.items()}
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()}
    g = docs["PRIVATE_GOLD.json"]; review = docs["PUBLIC_SOURCE_JUDGMENTS.json"]
    design = docs["PRIVATE_DESIGN.json"]; crosscheck = docs["FINAL_CROSSCHECK.json"]
    expected = {"source_sha256": "CUMULATIVE_SOURCE.json", "public_sha256": "PUBLIC_QUESTIONS.json",
                "source_judgments_sha256": "PUBLIC_SOURCE_JUDGMENTS.json", "original_design_sha256": "PRIVATE_DESIGN.json"}
    for key, name in expected.items():
        _require(g.get(key) == hashes[name], "private_gold_hash_binding:" + key)
    for key, name in (("source_sha256", "CUMULATIVE_SOURCE.json"), ("public_sha256", "PUBLIC_QUESTIONS.json"),
                      ("asset_judgments_sha256", "ASSET_SOURCE_JUDGMENTS.json")):
        _require(review.get(key) == hashes[name], "public_review_hash_binding:" + key)
    for key, name in (("source_sha256", "CUMULATIVE_SOURCE.json"), ("public_assets_sha256", "PUBLIC_ASSETS.json"),
                      ("asset_judgments_sha256", "ASSET_SOURCE_JUDGMENTS.json")):
        _require(design.get(key) == hashes[name], "design_hash_binding:" + key)
    _require(crosscheck.get("private_sha256") == hashes["PRIVATE_GOLD.json"], "crosscheck_private_binding")
    for key, name in (("public_sha256", "PUBLIC_QUESTIONS.json"),
                      ("source_judgments_sha256", "PUBLIC_SOURCE_JUDGMENTS.json")):
        _require(crosscheck.get(key) == hashes[name], "crosscheck_hash_binding:" + key)
    assets = docs["ASSET_SOURCE_JUDGMENTS.json"]
    _require(assets.get("source_sha256") == hashes["CUMULATIVE_SOURCE.json"] and
             assets.get("public_sha256") == hashes["PUBLIC_ASSETS.json"], "asset_review_hash_binding")
    by_gold = {q["candidate_id"]: q for q in g["questions"]}
    by_design = {q["candidate_id"]: q for q in design["design"]}
    by_check = {q["candidate_id"]: q for q in crosscheck["results"]}
    public = docs["PUBLIC_QUESTIONS.json"]["questions"]
    public_ids = [q["candidate_id"] for q in public]
    _require(len(public_ids) == len(set(public_ids)), "duplicate_public_candidate_id")
    _require(len(by_gold) == len(g["questions"]) and len(by_design) == len(design["design"]) and
             len(by_check) == len(crosscheck["results"]), "duplicate_bound_candidate_id")
    _require(set(public_ids) == set(by_gold) == set(by_design) == set(by_check) ==
             set(review["questions"]), "candidate_coverage_mismatch")
    out = []
    for question in public:
        cid = question["candidate_id"]
        if by_check[cid]["status"] != "CONTENT_PASS":
            continue
        _require(by_check[cid]["source_derived_answer"] == by_gold[cid]["gold"], "crosscheck_gold_conflict:" + cid)
        _require(question.get("source_sha256") == hashes["CUMULATIVE_SOURCE.json"] and
                 Path(question["source_path"]).read_bytes() == raw["CUMULATIVE_SOURCE.json"],
                 "question_source_file_binding:" + cid)
        candidate = export_question_evidence(question, by_gold[cid], docs["CUMULATIVE_SOURCE.json"],
                    review["questions"][cid], asset_source_judgments=docs["ASSET_SOURCE_JUDGMENTS.json"],
                    design=by_design[cid])
        candidate["evidence_export"]["input_sha256"] = deepcopy(hashes)
        out.append(candidate)
    return out
