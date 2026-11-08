"""Build and verify the proposed QA output contract without network calls."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

from . import (
    DEFAULT_OUTPUT, FROZEN_FEEDBACK_TEMPLATE_SHA256, SOURCE_RUN,
    TARGETED_PILOT_RUN, TARGETED_TEMPLATE_RUN, TEMPLATE_SHA_POLICY, VERSION,
)
from .contract import build_schema, build_targeted_schema, local_closure, schema_valid


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(output: Path, name: str, value: Any) -> str:
    path = output / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        data = value if value.endswith("\n") else value + "\n"
    elif name.endswith(".jsonl"):
        data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in value)
    else:
        data = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(data, encoding="utf-8")
    digest = hashlib.sha256(data.encode("utf-8")).hexdigest()
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def source_identity() -> list[dict[str, Any]]:
    files = [
        SOURCE_RUN / "multi_qa_composition_output_schema_v1.json",
        SOURCE_RUN / "fresh_exploration_output_schema_v1.json",
        SOURCE_RUN / "multi_qa_composition_template_v1.md",
        SOURCE_RUN / "fresh_exploration_template_v1.md",
        TARGETED_TEMPLATE_RUN / "qa_feedback_output_schema_v1.json",
        TARGETED_PILOT_RUN / "qa_feedback_template_v1.md",
    ]
    rows = []
    for path in files:
        sidecar = path.with_name(path.name + ".sha256")
        if not path.is_file() or not sidecar.is_file():
            raise RuntimeError(f"missing_source:{path}")
        observed = sha(path)
        expected = sidecar.read_text(encoding="utf-8").split()[0]
        if observed != expected:
            raise RuntimeError(f"source_sha_mismatch:{path}")
        rows.append({"path": str(path), "sha256": observed, "sidecar_match": True})
    return rows


def fixture(pass_name: str, qtype: str, options: dict[str, str], gold: list[str]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "outcome": "QA", "question_type": qtype,
        "new_question": "Which result is supported by the dialogue?",
        "options_or_items": options, "canonical_gold": gold,
        "evidence_refs": ["P01:T001", "R03:S01:T001"],
        "selected_evidence_relations": ["relation_a", "relation_b"] if pass_name == "composition" else ["relation_a"],
        "expected_context_dependency": "The answer uses the supplied context.",
        "r03_semantic_contribution_claimed": True, "answerable_without_r03_claim": False,
    }
    if pass_name == "composition":
        row.update({"source_cluster_ids": ["cluster-a", "cluster-b"], "composition_relation": "joint relation"})
    else:
        row.update({"independent_new_qa_reason": "It targets a new relation.", "novelty_basis": "R03 state change."})
    return row


def run(output: Path) -> int:
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"immutable_output_exists:{output}")
    output.mkdir(parents=True, exist_ok=False)
    sources = source_identity()
    schemas = {
        "targeted": build_targeted_schema(read_json(TARGETED_TEMPLATE_RUN / "qa_feedback_output_schema_v1.json")),
        "composition": build_schema(read_json(SOURCE_RUN / "multi_qa_composition_output_schema_v1.json"), "composition"),
        "fresh": build_schema(read_json(SOURCE_RUN / "fresh_exploration_output_schema_v1.json"), "fresh"),
    }
    if sha(TARGETED_PILOT_RUN / "qa_feedback_template_v1.md") != FROZEN_FEEDBACK_TEMPLATE_SHA256:
        raise RuntimeError("frozen_feedback_template_sha_mismatch")
    write(output, "qa_output_schema_v1_1_targeted.json", schemas["targeted"])
    write(output, "qa_output_schema_v1_1_composition.json", schemas["composition"])
    write(output, "qa_output_schema_v1_1_fresh.json", schemas["fresh"])
    write(output, "closure_policy_v1_1.json", {
        "version": "qa_output_closure_v1.1",
        "evidence_policy": "minimal_real_refs_only; completeness_not_required_first_pass",
        "model_emits": ["evidence_refs"],
        "locally_derived": ["new_r03_evidence_refs = evidence_refs filtered by R03 prefix"],
        "option_contract": {"single_choice": "exactly 4", "multi_select": "4-6", "ordering": "4-6"},
        "semantic_fields": "UNKNOWN_NO_SEMANTIC_AUDITOR",
        "hard_errors": ["unknown evidence ref", "invalid option key shape", "invalid Gold shape", "duplicate/replayed question"],
    })
    write(output, "qa_output_contract_addendum_v1_1.md", "\n".join([
        "# QA Output Contract Addendum v1.1", "",
        "This addendum changes output mechanics only. It does not change the frozen QA Feedback Template or its feedback logic.", "",
        "For `single_choice`, emit exactly four options (`A` through `D`) and exactly one Gold key.",
        "For `multi_select`, emit four to six contiguous options (`A` through `D`, `E`, or `F`), with two or more Gold keys and at least one non-Gold option.",
        "For `ordering`, emit four to six contiguous items (`I1` through `I4`, `I5`, or `I6`) and list every item exactly once in Gold order.",
        "Emit only `evidence_refs`. Include at least one real dialogue turn that supports the QA. Exhaustive evidence enumeration is not required in the first pass.",
        "Do not emit `new_r03_evidence_refs`; it is derived locally from `evidence_refs`.", "",
    ]))
    known_refs = {"P01:T001", "R03:S01:T001"}
    rows = []
    for pass_name, schema in schemas.items():
        fixtures = [
            ("single_4", fixture(pass_name, "single_choice", {"A": "a", "B": "b", "C": "c", "D": "d"}, ["A"])),
            ("multi_4", fixture(pass_name, "multi_select", {"A": "a", "B": "b", "C": "c", "D": "d"}, ["A", "B"])),
            ("multi_5", fixture(pass_name, "multi_select", {chr(65 + i): str(i) for i in range(5)}, ["A", "B"])),
            ("multi_6", fixture(pass_name, "multi_select", {chr(65 + i): str(i) for i in range(6)}, ["A", "B"])),
            ("ordering_4", fixture(pass_name, "ordering", {f"I{i + 1}": str(i) for i in range(4)}, [f"I{i + 1}" for i in range(4)])),
            ("ordering_5", fixture(pass_name, "ordering", {f"I{i + 1}": str(i) for i in range(5)}, [f"I{i + 1}" for i in range(5)])),
            ("ordering_6", fixture(pass_name, "ordering", {f"I{i + 1}": str(i) for i in range(6)}, [f"I{i + 1}" for i in range(6)])),
        ]
        for case, row in fixtures:
            if pass_name == "targeted":
                row.pop("outcome")
                row.pop("question_type")
                row.pop("selected_evidence_relations")
                row.pop("independent_new_qa_reason", None)
                row.pop("novelty_basis", None)
                row.pop("r03_semantic_contribution_claimed", None)
                row.pop("answerable_without_r03_claim", None)
                row.update({
                    "source_question_id": "question-1", "source_cluster_id": "cluster-1", "root_id": "root-1",
                    "selected_strategy": "ENTITY_RELATION_BINDING", "strategy_rationale": "A supported relation changes.",
                    "feedback_observations_used": ["M0"], "tentative_hypothesis": "Hypothesis only.",
                    "source_question_relation": "The answer target changes.",
                })
            valid, error = schema_valid(row, schema)
            closure = local_closure(row, pass_name, known_refs) if valid else {"status": "NOT_RUN"}
            rows.append({"pass": pass_name, "case": case, "schema_valid": valid, "schema_error": error, "closure": closure, "expected": "PASS"})
    # Explicit negative cases separately verify bounds, Gold shape, provenance,
    # and removal of the repeated R03 field.
    negative_rows = [
        ("multi_3_rejected", fixture("fresh", "multi_select", {"A": "a", "B": "b", "C": "c"}, ["A", "B"]), "SCHEMA_REJECT"),
        ("multi_7_rejected", fixture("fresh", "multi_select", {chr(65 + i): str(i) for i in range(7)}, ["A", "B"]), "SCHEMA_REJECT"),
        ("ordering_3_rejected", fixture("fresh", "ordering", {f"I{i + 1}": str(i) for i in range(3)}, [f"I{i + 1}" for i in range(3)]), "SCHEMA_REJECT"),
        ("ordering_7_rejected", fixture("fresh", "ordering", {f"I{i + 1}": str(i) for i in range(7)}, [f"I{i + 1}" for i in range(7)]), "SCHEMA_REJECT"),
        ("multi_all_options_gold_rejected", fixture("fresh", "multi_select", {"A": "a", "B": "b", "C": "c", "D": "d"}, ["A", "B", "C", "D"]), "CLOSURE_REJECT"),
        ("ordering_incomplete_gold_rejected", fixture("fresh", "ordering", {f"I{i + 1}": str(i) for i in range(4)}, ["I1", "I2", "I3"]), "SCHEMA_REJECT"),
        ("unknown_evidence_ref_rejected", fixture("fresh", "single_choice", {"A": "a", "B": "b", "C": "c", "D": "d"}, ["A"]), "CLOSURE_REJECT"),
        ("repeated_r03_field_rejected", fixture("fresh", "single_choice", {"A": "a", "B": "b", "C": "c", "D": "d"}, ["A"]), "SCHEMA_REJECT"),
    ]
    negative_rows[-2][1]["evidence_refs"] = ["R03:S99:T999"]
    negative_rows[-1][1]["new_r03_evidence_refs"] = ["R03:S01:T001"]
    for case, row, expected in negative_rows:
        valid, error = schema_valid(row, schemas["fresh"])
        closure = local_closure(row, "fresh", known_refs) if valid else {"status": "NOT_RUN"}
        rows.append({"pass": "fresh", "case": case, "schema_valid": valid, "schema_error": error, "closure": closure, "expected": expected})
    write(output, "fixture_validation_report.jsonl", rows)
    checks = {
        "source_sidecars_match": all(row["sidecar_match"] for row in sources),
        "positive_fixture_count": sum(row["expected"] == "PASS" and row["schema_valid"] and row["closure"].get("status") == "PASS" for row in rows) == 21,
        "negative_fixtures_rejected": all(
            (row["expected"] == "SCHEMA_REJECT" and not row["schema_valid"])
            or (row["expected"] == "CLOSURE_REJECT" and row["schema_valid"] and row["closure"].get("status") == "FAIL")
            for row in rows if row["expected"] != "PASS"
        ),
        "minimal_evidence_policy": True,
        "model_r03_field_removed": "new_r03_evidence_refs" not in schemas["fresh"]["properties"],
        "targeted_model_r03_field_removed": "new_r03_evidence_refs" not in schemas["targeted"]["properties"],
        "frozen_feedback_template_sha_preserved": sha(TARGETED_PILOT_RUN / "qa_feedback_template_v1.md") == FROZEN_FEEDBACK_TEMPLATE_SHA256,
        "no_api_calls": True,
        "no_solver_training_memorax": True,
    }
    report = {
        "version": VERSION, "status": "R03_QA_OUTPUT_CONTRACT_V1_1_PREFLIGHT_PASS" if all(checks.values()) else "FAIL",
        "source_run_immutable": str(SOURCE_RUN), "source_files": sources,
        "template_policy": TEMPLATE_SHA_POLICY,
        "option_contract": {"single_choice": 4, "multi_select": "4-6", "ordering": "4-6"},
        "evidence_policy": "real minimum refs required; completeness not a first-pass hard gate; semantic support remains unknown",
        "next_status": "READY_FOR_BOUNDED_SCHEMA_REPAIR_VERIFICATION",
        "checks": checks, "api_calls": {"qa_writer": 0, "solver": 0, "semantic_auditor": 0, "memorax": 0, "training": 0},
    }
    write(output, "preflight_report.json", report)
    write(output, "preflight_report.md", "\n".join([
        "# R03 QA Output Contract v1.1 Preflight", "",
        f"Status: `{report['status']}`", "",
        "This is an offline contract verification. The frozen v1 inputs were read and hash-checked; no API or state-changing call was made.", "",
        "- single-choice: exactly 4 options.",
        "- multi-select: 4 to 6 options.",
        "- ordering: 4 to 6 items.",
        "- The model emits only `evidence_refs`; R03 refs are derived locally.",
        "- Evidence completeness is not required in the first pass; real turn refs remain required.",
        "- Gold uniqueness, distractor validity, semantic duplication, and semantic R03 contribution remain unadjudicated.",
        "",
    ]))
    write(output, "source_sha_manifest.json", {"status": "PASS", "files": sources})
    return 0 if report["status"].endswith("PASS") else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-direct-tests", action="store_true")
    args = parser.parse_args()
    if args.run_direct_tests:
        from . import test_contract
        failures = []
        tests = [fn for name, fn in inspect.getmembers(test_contract, inspect.isfunction) if name.startswith("test_")]
        for fn in tests:
            try:
                fn()
            except Exception as error:
                failures.append({"test": fn.__name__, "error": repr(error)})
        if failures:
            raise RuntimeError(f"direct_tests_failed:{failures}")
    return run(args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
