"""Close the frozen full Dev/Test ordering review without touching Solver results."""

from __future__ import annotations

import csv
import json
from pathlib import Path


OUT = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_dev_test_ordering_review_v1_live_1")
SUMMARY = OUT / "ordering_full_audit_summary.json"
OLD_SAMPLE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_ordering_candidate_review_v1_live_1/review_decisions_v2.tsv")
OLD_FULL = OUT / "remaining_195_manual_review.tsv"
FINAL = OUT / "final_manual_review.tsv"


# These are the 47 rows that were still blank after the 50-row sample and the
# first 148-row manual pass.  The judgments are based on the frozen dialogue
# turns and the item/evidence mapping, not on model outputs.
NEW: dict[str, tuple[str, str, str, str, str]] = {
    # 151581
    "cc-train-episode-151581:I001-C-Q04": ("valid", "mismatch", "evidence_mismatch", "I1,I2,I4,I3", "The order is correct; the supplied boss ref supports the later improvement only partially, not the full combined item."),
    "cc-train-episode-151581:I001-F-Q07": ("valid", "clean", "valid", "I2,I1,I3,I4", "Jail, review, neighborhood-watch proposal, and boss presentation are direct and chronological."),
    "cc-train-episode-151581:I001-F-Q08": ("valid", "clean", "valid", "I2,I1,I3,I4", "Jail, missed work, apology, and neighborhood-watch proposal are direct and chronological."),
    "cc-train-episode-151581:I002-C-Q04": ("invalid_gold", "clean", "invalid_gold", "I1,I4,I2,I3", "The apology and watch proposal precede the later boss explanation and business-improvement turns."),
    "cc-train-episode-151581:I002-F-Q05": ("valid", "clean", "valid", "I1,I2,I3,I4", "Review, integrity correction, guilt, and apology are directly stated in order."),
    "cc-train-episode-151581:I002-F-Q06": ("invalid_gold", "mismatch", "invalid_gold", "I2,I1,I3,I4", "Police experience is earlier in P02; Lena, suspicious activity, and the watch proposal follow in R01:S02."),
    "cc-train-episode-151581:I003-C-Q03": ("valid", "clean", "valid", "I2,I4,I3,I1", "Apology, customer feedback, watch proposal, and boss presentation are direct and chronological."),
    "cc-train-episode-151581:I003-F-Q07": ("invalid_gold", "clean", "invalid_gold", "I1,I2,I4,I3", "The apology precedes the later neighborhood-watch proposal; the gold swaps those two events."),
    "cc-train-episode-151581:I003-F-Q08": ("invalid_gold", "mismatch", "invalid_gold", "I1,I4,I2,I3", "Warning, jail disclosure, review-apology advice, and job discussion occur in that order; supplied refs are partly shifted."),
    # 174027
    "cc-train-episode-174027:I001-C-Q04": ("invalid_gold", "mismatch", "invalid_gold", "I1,I2,I4,I3", "The offer of literary contacts precedes Tatum's later request for agent help; the supplied P01 bakery refs are unrelated."),
    "cc-train-episode-174027:I001-F-Q05": ("valid", "mismatch", "evidence_mismatch", "I1,I3,I4,I2", "The order is correct, but the pandemic/coping item is not supported by the supplied publishing ref."),
    "cc-train-episode-174027:I002-C-Q04": ("valid", "clean", "valid", "I1,I2,I3,I4", "Close call, realization, course signup, and joint plan are direct and chronological."),
    "cc-train-episode-174027:I002-F-Q07": ("valid", "clean", "valid", "I1,I2,I4,I3", "Novel, driving incident, garden work, and joint course plan are direct and chronological."),
    "cc-train-episode-174027:I002-F-Q08": ("invalid_gold", "clean", "invalid_gold", "I1,I4,I3,I2", "The bakery mention precedes garden work, while the publishing meeting is last; gold swaps I3 and I4."),
    "cc-train-episode-174027:I003-F-Q05": ("ambiguous", "mismatch", "ambiguous", "I1,I2,I3,I4", "The final attention-to-detail claim is not directly stated by Tatum; the supplied refs also mix speakers and occurrences."),
    "cc-train-episode-174027:I003-F-Q06": ("valid", "clean", "valid", "I1,I2,I3,I4", "Sharing, agent help, delivery, and agreement to read are direct and chronological."),
    # 183408
    "cc-train-episode-183408:I001-C-Q03": ("invalid_gold", "mismatch", "invalid_gold", "I2,I4,I1,I3", "Gallery, Central Park discussion, investigation, and tribute painting occur in that order; evidence refs are shifted."),
    "cc-train-episode-183408:I001-F-Q07": ("valid", "mismatch", "evidence_mismatch", "I1,I3,I4,I2", "The gold order is correct, but the supplied determination ref is a focus statement rather than the investigation event."),
    "cc-train-episode-183408:I001-F-Q08": ("invalid_gold", "mismatch", "invalid_gold", "I2,I1,I3,I4", "Gallery, contest result, New York trip, and investigation are chronological; gold puts the contest result last."),
    "cc-train-episode-183408:I002-C-Q03": ("invalid_gold", "mismatch", "invalid_gold", "I4,I2,I3,I1", "Questioning people precedes determination, archive planning, and the tribute; gold is reversed and refs are misassigned."),
    "cc-train-episode-183408:I002-F-Q07": ("invalid_gold", "mismatch", "invalid_gold", "I1,I2,I3,I4", "Gallery precedes investigation, library planning, and tribute painting; the supplied calabash ref is unrelated."),
    "cc-train-episode-183408:I002-F-Q08": ("valid", "clean", "valid", "I1,I2,I4,I3", "Gallery, contest entry, New York visit, and library plan are direct and chronological."),
    "cc-train-episode-183408:I003-C-Q04": ("invalid_gold", "mismatch", "invalid_gold", "I4,I3,I1,I2", "Contest entry and gallery inspiration precede the tribute decision and sketching; supplied refs do not match those items."),
    "cc-train-episode-183408:I003-F-Q08": ("invalid_gold", "mismatch", "invalid_gold", "I2,I1,I3,I4", "Gallery discussion, contest entry, New York trip, and dance-class recollection occur in that order; gold is wrong."),
    # 196614
    "cc-train-episode-196614:I001-C-Q03": ("valid", "clean", "valid", "I1,I2,I3,I4", "Stress, mindfulness advice, routine, and study breaks are direct and chronological."),
    "cc-train-episode-196614:I001-F-Q08": ("valid", "mismatch", "evidence_mismatch", "I1,I2,I3,I4", "The order is correct, but the supplied final ref is a welcome rather than Elena's acceptance/community statement."),
    "cc-train-episode-196614:I003-C-Q04": ("valid", "clean", "valid", "I1,I2,I3,I4", "Neighbor, invitation, acceptance/community interest, and project discussion are direct and chronological."),
    "cc-train-episode-196614:I003-F-Q07": ("valid", "clean", "valid", "I1,I2,I3,I4", "Speeding, call-ahead advice, invitation call, and school-board interest are direct and chronological."),
    "cc-train-episode-196614:I003-F-Q08": ("invalid_gold", "clean", "invalid_gold", "I3,I1,I4,I2", "The road-trip recollection is in P02, honesty and mindfulness are in P03, and evidence discussion is in P05."),
    # 233309
    "cc-train-episode-233309:I001-C-Q04": ("valid", "mismatch", "evidence_mismatch", "I1,I2,I3,I4", "The order is correct, but the supplied last ref is James's distress rather than encouragement."),
    "cc-train-episode-233309:I001-F-Q07": ("valid", "mismatch", "evidence_mismatch", "I3,I1,I2,I4", "Wedding, contest win, surname change, and trig difficulty are chronological; the first three refs are shifted."),
    "cc-train-episode-233309:I001-F-Q08": ("invalid_gold", "mismatch", "invalid_gold", "I1,I4,I2,I3", "Updo, library anticipation, trig difficulty, and contest-growth reflection occur in that order; gold is wrong."),
    "cc-train-episode-233309:I002-C-Q04": ("ambiguous", "mismatch", "ambiguous", "I4,I1,I3,I2", "Several claims occur in the same P04:T007 retrospective turn, so a unique order among them is not defensible."),
    "cc-train-episode-233309:I002-F-Q07": ("invalid_gold", "mismatch", "invalid_gold", "I1,I4,I2,I3", "Contest win, work mistake, surname change, and photography entry occur in that order; gold swaps the tail."),
    "cc-train-episode-233309:I002-F-Q08": ("valid", "mismatch", "evidence_mismatch", "I1,I2,I4,I3", "Wedding, work mistake, photography entry, and surname change are chronological; the photography ref is an essay-contest ref."),
    "cc-train-episode-233309:I003-C-Q04": ("valid", "mismatch", "evidence_mismatch", "I3,I1,I2,I4", "Wedding appearance, contest win, surname change, and essay submission are chronological; the first three evidence refs are shifted."),
    "cc-train-episode-233309:I003-F-Q07": ("invalid_gold", "mismatch", "invalid_gold", "I4,I3,I1,I2", "Library meeting, photography entry, essay submission, and surname update occur in that order; all supplied refs are shifted."),
    "cc-train-episode-233309:I003-F-Q08": ("invalid_gold", "mismatch", "invalid_gold", "I1,I2,I4,I3", "Updo, essay-topic question, non-winning result, and surname discussion occur in that order; gold is wrong."),
    # 86416
    "cc-train-episode-86416:I001-C-Q03": ("valid", "mismatch", "evidence_mismatch", "I1,I2,I3,I4", "The order is correct, but the supplied apology ref is a later reflection rather than the apology event."),
    "cc-train-episode-86416:I001-F-Q07": ("valid", "mismatch", "evidence_mismatch", "I4,I3,I1,I2", "The order is correct; the first two supplied refs are retrospective/general rather than direct event turns."),
    "cc-train-episode-86416:I001-F-Q08": ("ambiguous", "mismatch", "ambiguous", "I1,I2,I3,I4", "The final claim that James would keep Emily's advice in mind is not directly stated in the supplied dialogue."),
    "cc-train-episode-86416:I002-C-Q04": ("invalid_gold", "mismatch", "invalid_gold", "I1,I2,I4,I3", "The breakup occurs before the later patience discussion; gold places the later reflection too early."),
    "cc-train-episode-86416:I002-F-Q07": ("valid", "clean", "valid", "I4,I2,I3,I1", "Sister, friend, apology, and breakup are direct and chronological."),
    "cc-train-episode-86416:I002-F-Q08": ("invalid_gold", "clean", "invalid_gold", "I1,I4,I2,I3", "Admission, one-week check-in, honesty advice, and later reflection occur in that order."),
    "cc-train-episode-86416:I003-C-Q04": ("valid", "mismatch", "evidence_mismatch", "I1,I2,I4,I3", "The order is correct, but the supplied apology ref is the earlier conflict description, not the apology."),
    "cc-train-episode-86416:I003-F-Q07": ("valid", "mismatch", "evidence_mismatch", "I4,I3,I2,I1", "The order is correct; the supplied first three refs are retrospective or mismatched to the stated events."),
    "cc-train-episode-86416:I003-F-Q08": ("invalid_gold", "mismatch", "invalid_gold", "I3,I1,I4,I2", "Corner support, regret, breakup sympathy, and availability occur in that order; gold is wrong and I2's ref is shifted."),
}

# Two legacy rows were marked invalid_gold even though one item is not stated
# in the dialogue.  They are unsupported and must not enter any accuracy
# denominator.
NEW_STATUS_OVERRIDES = {
    "cc-train-episode-232414:I002-C-Q04": ("unsupported", "mismatch", "unsupported", "", "I4 (resolving by rescheduling) is not stated; the dialogue says brunch then moving help."),
    "cc-train-episode-97446:I001-F-Q08": ("ambiguous", "clean", "ambiguous", "", "I4 combines distinct weekly-check-in and feeling-better claims; no unique occurrence."),
}


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    meta = {row["candidate_id"]: row for row in summary["candidates"]}
    decisions: dict[str, dict[str, str]] = {}

    for row in load_tsv(OLD_SAMPLE):
        decisions[row["candidate_id"]] = {
            "candidate_id": row["candidate_id"],
            "split": meta[row["candidate_id"]]["split"],
            "root_id": meta[row["candidate_id"]]["root_id"],
            "gold_status": row["gold_status"],
            "evidence_status": row["evidence_status"],
            "primary_status": row["primary_status"],
            "corrected_order": row["corrected_order"],
            "note": row["note"],
            "review_basis": "full_dialogue_manual",
        }
    for row in load_tsv(OLD_FULL):
        decisions[row["candidate_id"]] = row
    for cid, values in NEW.items():
        gold, evidence, primary, order, note = values
        m = meta[cid]
        decisions[cid] = {
            "candidate_id": cid,
            "split": m["split"],
            "root_id": m["root_id"],
            "gold_status": gold,
            "evidence_status": evidence,
            "primary_status": primary,
            "corrected_order": order,
            "note": note,
            "review_basis": "full_dialogue_manual",
        }
    # Apply legacy-row corrections last so older partial tables cannot
    # overwrite the final adjudication.
    for cid, values in NEW_STATUS_OVERRIDES.items():
        gold, evidence, primary, order, note = values
        m = meta[cid]
        decisions[cid] = {
            "candidate_id": cid,
            "split": m["split"],
            "root_id": m["root_id"],
            "gold_status": gold,
            "evidence_status": evidence,
            "primary_status": primary,
            "corrected_order": order,
            "note": note,
            "review_basis": "full_dialogue_manual",
        }

    expected = set(meta)
    actual = set(decisions)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra or len(NEW) != 47:
        raise SystemExit(f"closure failure missing={missing} extra={extra} new={len(NEW)}")

    fields = ["review_index", "candidate_id", "split", "root_id", "gold_status", "evidence_status", "primary_status", "corrected_order", "note", "review_basis"]
    with FINAL.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for index, cid in enumerate(sorted(decisions), 1):
            row = dict(decisions[cid])
            row["review_index"] = f"R{index:03d}"
            writer.writerow({field: row.get(field, "") for field in fields})

    by_split: dict[str, dict[str, int]] = {}
    for row in decisions.values():
        counts = by_split.setdefault(row["split"], {"total": 0})
        counts["total"] += 1
        for key in ("gold_status", "evidence_status", "primary_status"):
            value = f"{key}:{row[key]}"
            counts[value] = counts.get(value, 0) + 1
    report = {
        "audit_version": "h4_round0_ordering_full_manual_review_v1",
        "offline_only": True,
        "solver_calls": 0,
        "external_api_calls": 0,
        "memorax_calls": 0,
        "candidate_count": len(decisions),
        "split_counts": by_split,
        "source_summary": str(SUMMARY),
        "source_summary_sha256": __import__("hashlib").sha256(SUMMARY.read_bytes()).hexdigest(),
        "source_sample_review": str(OLD_SAMPLE),
        "source_first_full_review": str(OLD_FULL),
        "new_manual_rows": 47,
        "final_review": str(FINAL),
        "status": "COMPLETE_MANUAL_REVIEW_READY_FOR_REEVALUATION",
    }
    (OUT / "final_ordering_manual_review_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
