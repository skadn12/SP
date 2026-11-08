"""Reproduce automatic-only baselines from the frozen v5 ordering responses."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "h4_crossroot_ordering_batch_v5_fresh_roots"
LABELS = [f"I{i}" for i in range(1, 7)]
MODEL = "Qwen/Qwen3.6-35B-A3B"
EXPECTED_ROOTS = ["210046", "231899", "194259", "96544"]


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def shuffled_old_labels(candidate_id: str) -> list[str]:
    labels = LABELS.copy()
    rng = random.Random(int(digest(candidate_id.encode("utf-8")), 16))
    while True:
        rng.shuffle(labels)
        if sum(label != f"I{index}" for index, label in enumerate(labels, 1)) >= 3:
            return labels


def anchored(row: dict, root_id: str, turns: list[dict]) -> tuple[dict, list[str]]:
    errors = []
    if (row.get("root_id") != root_id or row.get("question_type") != "ordering"
            or row.get("chronology_kind") != "conversation_order"):
        errors.append("identity_or_type_invalid")
    items, evidence = row.get("items"), row.get("item_evidence")
    if not isinstance(items, dict) or set(items) != set(LABELS):
        errors.append("items_invalid")
    if not isinstance(evidence, dict) or set(evidence) != set(LABELS):
        return {}, errors + ["evidence_invalid"]
    by_ref = {turn["turn_ref"]: turn for turn in turns}
    if len(by_ref) != len(turns):
        raise ValueError("source_refs_not_unique")
    positions = {turn["turn_ref"]: index for index, turn in enumerate(turns)}
    refs = {}
    for label in LABELS:
        entry = evidence[label]
        if not isinstance(entry, dict):
            errors.append(f"{label}:evidence_not_object")
            continue
        ref, quote = entry.get("turn_ref"), entry.get("quote")
        if ref not in by_ref:
            errors.append(f"{label}:ref_invalid")
            continue
        refs[label] = ref
        if entry.get("speaker") != by_ref[ref]["speaker"]:
            errors.append(f"{label}:speaker_mismatch")
        if not isinstance(quote, str) or not quote.strip() or quote not in by_ref[ref]["text"]:
            errors.append(f"{label}:quote_not_verbatim")
    if len(refs) == 6:
        if len(set(refs.values())) != 6:
            errors.append("duplicate_anchor_turns")
        if not any(ref.startswith("R01:") for ref in refs.values()):
            errors.append("r01_anchor_required")
    return {"positions": positions, "refs": refs, "by_ref": by_ref}, errors


def rebind(row: dict, anchor: dict) -> dict:
    result = deepcopy(row)
    old_order = shuffled_old_labels(row["candidate_id"])
    result["items"] = {new: row["items"][old] for new, old in zip(LABELS, old_order)}
    result["item_evidence"] = {new: row["item_evidence"][old] for new, old in zip(LABELS, old_order)}
    refs = {new: anchor["refs"][old] for new, old in zip(LABELS, old_order)}
    result["gold"] = sorted(LABELS, key=lambda label: anchor["positions"][refs[label]])
    result["order_reason"] = "Source turn appearance order: " + ", ".join(
        f"{label} ({refs[label]})" for label in result["gold"]
    ) + "."
    return result


def quote_question(row: dict, anchor: dict) -> tuple[dict | None, list[str]]:
    errors = []
    for label in LABELS:
        quote = row["item_evidence"][label]["quote"]
        matches = [ref for ref, turn in anchor["by_ref"].items() if quote in turn["text"]]
        if len(matches) != 1:
            errors.append(f"{label}:quote_not_unique_in_source")
        if len(re.findall(r"[A-Za-z0-9]+", quote)) < 4:
            errors.append(f"{label}:quote_too_short")
    if errors:
        return None, errors
    result = rebind(row, anchor)
    result["candidate_id"] += "-excerpt"
    result["question"] = "In what order do these exact utterance excerpts APPEAR IN THE SUPPLIED DIALOGUE?"
    result["items"] = {
        label: f'{result["item_evidence"][label]["speaker"]}: "{result["item_evidence"][label]["quote"]}"'
        for label in LABELS
    }
    result["item_evidence"] = {
        label: {
            "turn_ref": entry["turn_ref"], "speaker": entry["speaker"],
            "quote": entry["quote"], "quote_unique_in_source": True,
        }
        for label, entry in result["item_evidence"].items()
    }
    result["task_variant"] = "verbatim_excerpt_ordering"
    return result, []


def main() -> None:
    manifest = load(SOURCE / "BATCH_PREPARED.json")
    roots = manifest["roots"]
    if [root["root_id"].split("-")[-1] for root in roots] != EXPECTED_ROOTS:
        raise ValueError("unexpected_root_batch")
    report = {
        "status": "AUTOMATIC_PILOT_NO_SOLVER_NO_TRAINING",
        "source_batch": str(SOURCE), "source_batch_sha256": digest((SOURCE / "BATCH_PREPARED.json").read_bytes()),
        "generator_model": MODEL, "roots": [],
        "semantic_reviewer_used": False, "solver_calls": 0, "selected_for_training": 0,
    }
    label_rows, excerpt_rows = [], []
    for root in roots:
        root_id = root["root_id"]
        directory = SOURCE / root_id
        source_raw = (directory / "source.json").read_bytes()
        if digest(source_raw) != root["source_sha256"]:
            raise ValueError(f"source_hash_mismatch:{root_id}")
        turns = json.loads(source_raw)["turns"]
        response_raw = (directory / "initial/RESPONSE.json").read_bytes()
        response = json.loads(response_raw)
        if (response["source_sha256"] != root["source_sha256"] or
                response["prompt_sha256"] != root["initial"]["prompt_sha256"] or
                response["payload_sha256"] != root["initial"]["payload_sha256"] or
                response["requested_model"] != MODEL or response["returned_model"] != MODEL):
            raise ValueError(f"response_binding_mismatch:{root_id}")
        parsed = json.loads(response["raw_text"])
        if parsed != load(directory / "initial/PARSED_RESPONSE.json"):
            raise ValueError(f"parsed_response_mismatch:{root_id}")
        validation = load(directory / "initial/VALIDATION.json")
        if validation["global_errors"] or validation["candidate_count"] != len(parsed["questions"]):
            raise ValueError(f"global_validation_error:{root_id}")
        checks = {check["candidate_id"]: check["errors"] for check in validation["checks"]}
        root_report = {
            "root_id": root_id, "source_sha256": digest(source_raw),
            "response_sha256": digest(response_raw), "response_id": response["response_id"],
            "generated_count": len(parsed["questions"]),
            "raw_zero_edit_mechanical_pass": validation["mechanically_valid_count"],
            "label_only_mechanical_candidates": 0, "verbatim_excerpt_candidates": 0,
            "checks": [],
        }
        seen_sets = set()
        for row in parsed["questions"]:
            candidate_id = row["candidate_id"]
            anchor, errors = anchored(row, root_id, turns)
            event_set = tuple(sorted(anchor.get("refs", {}).values()))
            if len(event_set) == 6 and event_set in seen_sets:
                errors.append("duplicate_six_turn_set")
            seen_sets.add(event_set)
            eligible = not errors and checks[candidate_id] == ["display_not_sufficiently_shuffled"]
            check = {
                "candidate_id": candidate_id, "raw_mechanical_errors": checks[candidate_id],
                "anchor_errors": errors, "label_only_mechanical_candidate": eligible,
                "verbatim_excerpt_candidate": False, "excerpt_errors": [],
            }
            if eligible:
                label = rebind(row, anchor)
                if label["gold"] == LABELS or sum(x != y for x, y in zip(label["gold"], LABELS)) < 3:
                    raise ValueError("rebind_not_shuffled")
                label_rows.append(label)
                root_report["label_only_mechanical_candidates"] += 1
            if not errors:
                quote, quote_errors = quote_question(row, anchor)
                check["excerpt_errors"] = quote_errors
                if quote is not None:
                    excerpt_rows.append(quote)
                    check["verbatim_excerpt_candidate"] = True
                    root_report["verbatim_excerpt_candidates"] += 1
            root_report["checks"].append(check)
        report["roots"].append(root_report)
    report["totals"] = {
        "generated": sum(root["generated_count"] for root in report["roots"]),
        "raw_zero_edit_mechanical_pass": sum(root["raw_zero_edit_mechanical_pass"] for root in report["roots"]),
        "label_only_mechanical_candidates": len(label_rows),
        "verbatim_excerpt_candidates": len(excerpt_rows),
    }
    for name, value in (("REPORT.json", report), ("LABEL_REBOUND_UNREVIEWED.json", label_rows),
                        ("VERBATIM_EXCERPT_VARIANT.json", excerpt_rows)):
        path = HERE / name
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["totals"]))


if __name__ == "__main__":
    main()
