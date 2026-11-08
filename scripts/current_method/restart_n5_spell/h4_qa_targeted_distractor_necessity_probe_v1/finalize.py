"""Zero-API strict-admission finalization of an existing replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import write_json

from .admission import strict_necessity_admission


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def finalize(source: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    metrics = _load(source / "metrics.json")
    solver = _load(source / "solver_evaluation.json")
    static = _load(source / "composer_static_audit.json")
    targeted = _load(source / "targeted_slot_audit.json")
    response = _load(source / "composer_response_envelope.json")
    passed, admission = strict_necessity_admission(solver)
    result = dict(metrics)
    result.update({
        "probe_version": "h4_qa_targeted_distractor_necessity_probe_v1_strict_final",
        "analysis_execution_api_calls": 0,
        "generic_static_valid": static.get("valid"),
        "targeted_audit_valid": targeted.get("valid"),
        "necessity_pass": passed,
        "admission": admission,
        "admitted": bool(targeted.get("valid") and passed),
        "conclusion": (
            "TARGETED_QA_SHAPE_PASSES_BUT_DIFFICULTY_NOT_VALIDATED"
            if targeted.get("valid") and not passed else
            "TARGETED_QA_AND_NECESSITY_PASS"
            if targeted.get("valid") and passed else
            "TARGETED_QA_AUDIT_FAILED"
        ),
        "composer_question": response.get("question"),
        "composer_options": response.get("options"),
    })
    write_json(output / "metrics.json", result)
    write_json(output / "POST_RUN_INTEGRITY.json", {
        "status": "complete", "analysis_execution_api_calls": 0,
        "source_replay": str(source.resolve()), "source_artifacts_mutated": False,
        "admitted": result["admitted"], "necessity_pass": result["necessity_pass"],
    })
    (output / "report.md").write_text("\n".join([
        "# H4 QA Targeted Distractor Necessity Probe v1 strict finalization", "",
        f"Conclusion: **{result['conclusion']}**", "",
        "This is a zero-API reclassification of the completed local replay.", "",
        f"- Targeted QA audit: `{result['targeted_audit_valid']}`",
        f"- No-context accuracy: `{result.get('no_context_accuracy')}`",
        f"- Full-context accuracy: `{result.get('full_context_accuracy')}`",
        f"- Remove fact_A: `{result.get('remove_A_accuracy')}`",
        f"- Remove trigger: `{result.get('remove_trigger_accuracy')}`",
        f"- Remove fact_B: `{result.get('remove_B_accuracy')}`",
        f"- Necessity pass: `{result['necessity_pass']}`",
        f"- Admitted: `{result['admitted']}`", "",
        "Strict rule: full K=5 must pass, no-context must not be all-correct, "
        "and each declared role ablation must fail or be uncertain.", "",
    ]) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.source.resolve(), args.output.resolve()), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
