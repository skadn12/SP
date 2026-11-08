#!/usr/bin/env python3
"""Apply three frozen warning-policy adjudications without changing original data.

Only quality receipts, the frozen adjudication, and existing first blind reviews
are inputs. Answers and Gold are opaque inherited values; this program performs
no answering, source qualification, training admission, scoring, or model calls.
The fixed scope and pinned input hashes deliberately fail closed on input drift.
Run without arguments to create new views, or with --check to verify them.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
POLICY_PATH = ROOT / "PUBLIC_SOURCE_WARNING_POLICY_ADJUDICATION.json"
POLICY_SHA256 = "369af3a1c5dbad644942096147a2bef41f336a92bd8930acf6db3a777ef96fed"
SCOPE = (
    {
        "batch": "batch_02",
        "root_id": "cc-train-episode-24719",
        "arm": "F",
        "candidate_id": "24719_F_Q02",
        "review_id": "Q02",
        "original_quality_sha256": "10821822bfc4b53d3379700c1f25e51e585d3ad4c0f13ad1f485528de3261305",
        "inherited_quality_sha256": "1e7fa738f5ab16838c46f6db269116419ef9635730c084206b75131b8c2fe4e9",
        "view_parent": "target_hash_interface_view",
    },
    {
        "batch": "batch_02",
        "root_id": "cc-train-episode-45849",
        "arm": "F",
        "candidate_id": "45849_F_Q02",
        "review_id": "Q02",
        "original_quality_sha256": "29a64242e79bfd9cf019e8bd5720e9e65ce9d7c2d619db707a4dbcd059ab37aa",
        "inherited_quality_sha256": "a87d0aedcfcbd254cf500a5faf07403716603330a8ec513057a38897bd100632",
        "view_parent": "target_hash_interface_view",
    },
    {
        "batch": "batch_05",
        "root_id": "cc-train-episode-246750",
        "arm": "G",
        "candidate_id": "G246750_Q01",
        "review_id": "Q01",
        "original_quality_sha256": "9046558227bfe872e8fb46fe2cd647cfd86690fc242a5604c69756f6b7f097d0",
        "inherited_quality_sha256": "9046558227bfe872e8fb46fe2cd647cfd86690fc242a5604c69756f6b7f097d0",
        "view_parent": "",
    },
)
CHANGED_EXISTING_FIELDS = {
    "hard_issues", "soft_issues", "hold_reasons", "status", "content_pass",
    "solver_eligible",
}
ADDED_FIELDS = {
    "original_hard_issues", "original_quality_path", "original_quality_sha256",
    "inherited_quality_path", "inherited_quality_sha256",
    "warning_policy_adjudication",
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


class Inputs:
    def __init__(self):
        self.raw = {}
        self.records = {}

    def load(self, path, expected_sha256, kind):
        path = Path(path).resolve()
        assert path.is_relative_to(ROOT), f"Input outside fixed run root: {path}"
        raw = path.read_bytes()
        assert digest(raw) == expected_sha256, f"Input hash mismatch: {path}"
        if path in self.raw:
            assert raw == self.raw[path], f"Input changed while reading: {path}"
        self.raw[path] = raw
        self.records[path] = {
            "path": str(path), "sha256": expected_sha256,
            "bytes": len(raw), "kind": kind,
        }
        return json.loads(raw)

    def unchanged(self):
        for path, raw in self.raw.items():
            assert path.read_bytes() == raw, f"Original input changed: {path}"


def build_views():
    inputs = Inputs()
    policy = inputs.load(POLICY_PATH, POLICY_SHA256, "frozen_policy_adjudication")
    cases = {case["candidate_id"]: (i, case)
             for i, case in enumerate(policy["cases"])}
    assert len(cases) == len(policy["cases"]) == len(SCOPE) == 3
    assert set(cases) == {spec["candidate_id"] for spec in SCOPE}
    script_binding = {
        "path": str(Path(__file__).resolve()),
        "sha256": digest(Path(__file__).resolve().read_bytes()),
    }
    outputs = []
    summaries = []

    for spec in SCOPE:
        batch = ROOT / spec["batch"] / spec["root_id"] / spec["arm"] / "batch"
        original_path = batch / "QUALITY_FINAL.json"
        inherited_path = batch / spec["view_parent"] / "QUALITY_FINAL.json"
        view = batch / "warning_policy_view"
        quality_path = view / "QUALITY_FINAL.json"
        application_path = view / "POLICY_APPLICATION.json"
        read_before = set(inputs.records)
        original = inputs.load(original_path, spec["original_quality_sha256"], "original_quality")
        inherited = inputs.load(inherited_path, spec["inherited_quality_sha256"], "inherited_quality")
        original_rows = {row["candidate_id"]: row for row in original["items"]}
        inherited_rows = {row["candidate_id"]: row for row in inherited["items"]}
        assert len(original_rows) == len(original["items"])
        assert len(inherited_rows) == len(inherited["items"])
        assert list(original_rows) == list(inherited_rows)
        assert spec["candidate_id"] in inherited_rows
        case_index, case = cases[spec["candidate_id"]]
        for key in ("batch", "root_id", "arm", "review_id"):
            assert case[key] == spec[key], f"Policy scope mismatch: {key}"
        assessment = case["policy_assessment"]
        assert assessment["adjudicated_issue_severity"] == "soft"
        assert assessment["retain_original_hard_issue_verbatim_as_history"] is True
        assert assessment["challenged_hard_issue_is_supported_under_frozen_policy"] is False
        disputed = case["original_review_quality_fields_only"]["hard_issues"]
        assert isinstance(disputed, list) and len(disputed) == 1
        disputed = disputed[0]
        first_review_receipts = []

        # Read only existing first-review records and verify their frozen bindings.
        # Their answer values are never replaced, normalized, or used for selection.
        for candidate_id, row in inherited_rows.items():
            baseline = original_rows[candidate_id]
            original_review_path = batch / "blind_results" / f"{row['review_id']}.json"
            assert Path(baseline["review_path"]) == original_review_path
            assert baseline["review_sha256"] == row["review_sha256"]
            first_review = inputs.load(
                original_review_path, baseline["review_sha256"], "original_first_blind_review"
            )
            bound_review = inputs.load(
                row["review_path"], row["review_sha256"], "bound_first_blind_review"
            )
            assert first_review == bound_review
            assert first_review["review_id"] == row["review_id"]
            assert first_review["hard_issues"] == row["hard_issues"]
            assert first_review["soft_issues"] == row["soft_issues"]
            assert baseline["blind_answer"] == row["blind_answer"]
            assert baseline["gold_labels"] == row["gold_labels"]
            first_review_receipts.append({
                "candidate_id": candidate_id,
                "original_first_review_path": str(original_review_path),
                "original_first_review_sha256": baseline["review_sha256"],
                "inherited_review_path": row["review_path"],
                "inherited_review_sha256": row["review_sha256"],
                "first_answer_and_gold_opaque_values_preserved": True,
            })

        before = inherited_rows[spec["candidate_id"]]
        baseline = original_rows[spec["candidate_id"]]
        assert case["original_review"]["path"] == baseline["review_path"]
        assert case["original_review"]["sha256"] == baseline["review_sha256"]
        assert baseline["hard_issues"] == before["hard_issues"]
        assert before["hard_issues"].count(disputed) == 1
        assert not (ADDED_FIELDS & before.keys())

        derived = copy.deepcopy(inherited)
        after = next(row for row in derived["items"]
                     if row["candidate_id"] == spec["candidate_id"])
        after["original_hard_issues"] = copy.deepcopy(before["hard_issues"])
        after["original_quality_path"] = str(original_path)
        after["original_quality_sha256"] = spec["original_quality_sha256"]
        after["inherited_quality_path"] = str(inherited_path)
        after["inherited_quality_sha256"] = spec["inherited_quality_sha256"]
        adjudication_record = {
            "path": str(POLICY_PATH),
            "sha256": POLICY_SHA256,
            "case_pointer": f"/cases/{case_index}",
            "candidate_id": spec["candidate_id"],
            "original_first_review_path": case["original_review"]["path"],
            "original_first_review_sha256": case["original_review"]["sha256"],
            "adjudicated_issue_severity": "soft",
            "classification": assessment["classification"],
            "finding_zh": assessment["finding_zh"],
            "applicable_policy_ids": copy.deepcopy(assessment["applicable_policy_ids"]),
            "soft_issue_summary": copy.deepcopy(case["soft_issue_to_append_in_derived_policy_record_only"]),
            "scope": "Only this historical-source-warning issue is reclassified; no source qualification or formal training approval is created.",
        }
        after["warning_policy_adjudication"] = copy.deepcopy(adjudication_record)
        remaining = copy.deepcopy(before["hard_issues"])
        remaining.remove(disputed)
        after["hard_issues"] = remaining
        moved_issue = copy.deepcopy(disputed)
        moved_issue["policy_adjudication"] = copy.deepcopy(adjudication_record)
        after["soft_issues"].append(moved_issue)

        # PASS is allowed only when this issue was the sole inherited HOLD cause.
        # The F inputs already carry their independently created target-hash view.
        assert before["hold_reasons"] == ["blind_hard_issue"]
        assert before["mechanical_pass"] is True
        assert before["review_protocol_valid"] is True
        assert before["blind_gold_agreement"] is True
        assert before["review_protocol_errors"] == []
        assert before["status"] == "HOLD"
        assert before["content_pass"] is False
        assert before["solver_eligible"] is False
        assert remaining == []
        after["hold_reasons"] = []
        after["status"] = "PASS"
        after["content_pass"] = True
        after["solver_eligible"] = True

        # Every other pre-existing field and every other row stays identical.
        assert set(after) - set(before) == ADDED_FIELDS
        actual_changes = {key for key in before if after[key] != before[key]}
        assert actual_changes == CHANGED_EXISTING_FIELDS
        assert after["original_hard_issues"] == before["hard_issues"]
        assert after["soft_issues"][:-1] == before["soft_issues"]
        assert {key: value for key, value in moved_issue.items()
                if key != "policy_adjudication"} == disputed
        unchanged_candidates = []
        for old_row, new_row in zip(inherited["items"], derived["items"]):
            if old_row["candidate_id"] != spec["candidate_id"]:
                assert old_row == new_row
                unchanged_candidates.append(old_row["candidate_id"])
        assert {key: value for key, value in inherited.items() if key != "items"} == {
            key: value for key, value in derived.items() if key != "items"
        }
        assert after["mechanical_pass"] is True
        assert after["review_protocol_valid"] is True
        assert after["blind_gold_agreement"] is True
        assert after["hard_issues"] == after["hold_reasons"] == []
        quality_raw = encode(derived)
        relevant_paths = (set(inputs.records) - read_before) | {
            POLICY_PATH, original_path, inherited_path,
        }
        application = {
            "schema_version": "historical_source_warning_policy_application_v1",
            "operation": "Deterministic application of the existing nonblind policy adjudication; no reanswer or blind review retry.",
            "script": script_binding,
            "batch": spec["batch"], "root_id": spec["root_id"], "arm": spec["arm"],
            "candidate_ids_inherited_in_original_order": list(inherited_rows),
            "candidate_ids_reclassified": [spec["candidate_id"]],
            "candidate_ids_unchanged": unchanged_candidates,
            "original_quality": inputs.records[original_path],
            "inherited_quality": inputs.records[inherited_path],
            "adjudication": adjudication_record,
            "original_hard_issues": copy.deepcopy(before["hard_issues"]),
            "remaining_hard_issues": copy.deepcopy(after["hard_issues"]),
            "original_baseline_hold_reasons": copy.deepcopy(baseline["hold_reasons"]),
            "inherited_hold_reasons": copy.deepcopy(before["hold_reasons"]),
            "effective_hold_reasons": copy.deepcopy(after["hold_reasons"]),
            "inherited_status": before["status"], "effective_status": after["status"],
            "changed_existing_fields": sorted(actual_changes),
            "added_provenance_fields": sorted(ADDED_FIELDS),
            "first_review_bindings": first_review_receipts,
            "input_files": [inputs.records[path] for path in sorted(relevant_paths)],
            "output_quality": {
                "path": str(quality_path), "sha256": digest(quality_raw),
                "bytes": len(quality_raw),
            },
            "verification": {
                "mechanical_pass_asserted_true": True,
                "review_protocol_valid_asserted_true": True,
                "blind_gold_agreement_asserted_true": True,
                "inherited_only_hold_reason_asserted_blind_hard_issue": True,
                "all_remaining_hard_issues_asserted_empty": True,
                "uncontested_rows_exactly_equal_to_inherited_rows": True,
                "all_other_existing_fields_exactly_inherited": True,
                "original_hard_issues_preserved_verbatim": True,
                "original_first_answers_and_gold_preserved": True,
                "all_input_bytes_verified_unchanged_before_write": True,
                "model_or_api_calls": 0, "new_answers": 0,
                "review_retries": 0, "scores_read_or_used": False,
                "source_qualification_changed": False,
                "formal_training_approval_created": False,
                "original_files_modified": False,
            },
        }
        outputs.extend(((quality_path, quality_raw), (application_path, encode(application))))
        summaries.append({
            "candidate_id": spec["candidate_id"], "effective_status": after["status"],
            "unchanged_candidate_ids": unchanged_candidates,
            "quality_path": str(quality_path), "quality_sha256": digest(quality_raw),
            "application_path": str(application_path),
            "application_sha256": digest(encode(application)),
        })

    inputs.unchanged()
    assert len(outputs) == 6
    assert len({path for path, _ in outputs}) == 6
    assert all(path.parent.name == "warning_policy_view" for path, _ in outputs)
    return inputs, outputs, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify exact derived bytes; write nothing.")
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError("Assertions are mandatory; do not run Python with -O.")
    inputs, outputs, summaries = build_views()
    # Preflight every output before any write. Existing files may only be checked.
    if args.check:
        for path, expected in outputs:
            assert path.read_bytes() == expected, f"Derived output differs: {path}"
    else:
        for path, _ in outputs:
            assert not path.exists(), f"Refusing to overwrite: {path}; use --check"
        inputs.unchanged()
        for path, raw in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(raw)
        for path, expected in outputs:
            assert path.read_bytes() == expected, f"Output write mismatch: {path}"
    inputs.unchanged()
    print(json.dumps({
        "mode": "check" if args.check else "create",
        "script_sha256": digest(Path(__file__).resolve().read_bytes()),
        "all_original_input_bytes_unchanged": True,
        "results": summaries,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
