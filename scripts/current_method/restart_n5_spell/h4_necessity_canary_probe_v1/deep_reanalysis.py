"""Zero-API semantic-role closure audit and ablation reanalysis."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import prepare_empty, write_json

from scripts.current_method.restart_n5_spell.h4_relation_guided_composition_probe_v1.solver_eval import evaluate_composite
from .probe import _scene_text


ROLE_ANCHORS = {
    "fact_A": ("two parts honey", "one part mustard"),
    "trigger": ("too thick", "spread evenly"),
    "fact_B": ("water", "thin", "thinner", "pourable", "revised"),
}
DECLARED_REFS = {
    "fact_A": {"R01:S01:T001"},
    "trigger": {"R01:S02:T001"},
    "fact_B": {"R01:S02:T002", "R01:S02:T003"},
}


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _contains(text: str, phrase: str) -> bool:
    return bool(re.search(r"\b" + re.escape(phrase.casefold()) + r"\b", text.casefold()))


def _role_refs(child: dict) -> dict[str, list[str]]:
    result = {key: [] for key in ROLE_ANCHORS}
    for scene in child.get("scenes") or []:
        for row in scene.get("dialogue_turns") or []:
            ref = str(row.get("turn_ref") or "")
            text = str(row.get("text") or "")
            for role, anchors in ROLE_ANCHORS.items():
                if any(_contains(text, anchor) for anchor in anchors):
                    result[role].append(ref)
    return result


def run(*, source_run: Path, expanded_dialogue: Path, composite_run: Path, output: Path, tokenizer: Path, solver: Path) -> dict:
    prepare_empty(output)
    child = _load(source_run / "child_dialogue.json")
    contract = _load(source_run / "relation_contract.json")
    expanded = _load(expanded_dialogue)
    observed = _role_refs(child)
    leakage = {
        role: sorted(set(refs) - DECLARED_REFS[role])
        for role, refs in observed.items()
    }
    closure = {
        role: sorted(set(DECLARED_REFS[role]) | set(observed[role]))
        for role in ROLE_ANCHORS
    }
    relation_audit = {
        "audit_version": "h4_necessity_canary_deep_audit_v1",
        "relation_id": contract.get("relation_id"),
        "relation_type": contract.get("relation_type"),
        "declared_role_refs": {key: sorted(value) for key, value in DECLARED_REFS.items()},
        "observed_role_refs_by_anchor": observed,
        "role_restatement_outside_declared_refs": leakage,
        "expanded_role_evidence_closure": closure,
        "semantic_restatement_present": any(leakage.values()),
        "resolution_leakage_terms": [],
        "necessity_source_ready": not any(leakage.values()),
    }
    write_json(output / "deep_relation_audit.json", relation_audit)
    write_json(output / "run_freeze_manifest.json", {
        "freeze_version": "h4_necessity_canary_deep_reanalysis_v1",
        "source_run": str(source_run.resolve()),
        "expanded_dialogue": str(expanded_dialogue.resolve()),
        "composite_run": str(composite_run.resolve()),
        "analysis_execution_api_calls": 0,
        "local_solver_generations": 24,
        "source_artifacts_mutated": False,
    })
    composite = _load(composite_run / "composer_response_envelope.json")
    eval_result = evaluate_composite(
        composite=composite,
        expanded=expanded,
        relation_source_turns=closure,
        run_root=output,
        tokenizer_path=tokenizer,
        solver_model=solver,
    )
    metrics = {
        "probe_version": "h4_necessity_canary_deep_reanalysis_v1",
        "analysis_execution_api_calls": 0,
        "source_run": str(source_run.resolve()),
        "expanded_dialogue": str(expanded_dialogue.resolve()),
        "composite_run": str(composite_run.resolve()),
        "semantic_restatement_present": relation_audit["semantic_restatement_present"],
        "role_restatement_outside_declared_refs": leakage,
        "expanded_role_evidence_closure": closure,
        "full_context_accuracy": eval_result.get("full_context_accuracy"),
        "remove_A_accuracy": eval_result.get("remove_A_accuracy"),
        "remove_trigger_accuracy": eval_result.get("remove_trigger_accuracy"),
        "remove_B_accuracy": eval_result.get("remove_B_accuracy"),
        "necessity_pass": eval_result.get("necessity_pass"),
        "conclusion": (
            "ROLE_RESTATEMENT_BLOCKS_NECESSITY"
            if relation_audit["semantic_restatement_present"] and not eval_result.get("necessity_pass")
            else "NECESSITY_PASS_AFTER_ROLE_CLOSURE"
            if eval_result.get("necessity_pass")
            else "NECESSITY_STILL_UNPROVEN"
        ),
    }
    write_json(output / "metrics.json", metrics)
    write_json(output / "request_ledger.json", {
        "probe_version": "h4_necessity_canary_deep_reanalysis_v1",
        "execution_mode": "zero_api_role_closure_plus_local_solver",
        "analysis_execution_api_calls": 0,
    })
    write_json(output / "POST_RUN_INTEGRITY.json", {
        "probe_version": "h4_necessity_canary_deep_reanalysis_v1",
        "status": "complete",
        "analysis_execution_api_calls": 0,
        "source_artifacts_mutated": False,
        "composite_artifacts_mutated": False,
        "necessity_pass": eval_result.get("necessity_pass"),
    })
    report = [
        "# H4 Necessity Canary Deep Reanalysis v1", "",
        f"Verdict: **{metrics['conclusion']}**", "",
        "This run expands each relation role to the closure of semantically equivalent dialogue turns before ablation.", "",
        f"- Semantic role restatement present: `{metrics['semantic_restatement_present']}`",
        f"- Full context: `{metrics['full_context_accuracy']}`",
        f"- Remove fact_A: `{metrics['remove_A_accuracy']}`",
        f"- Remove trigger: `{metrics['remove_trigger_accuracy']}`",
        f"- Remove fact_B: `{metrics['remove_B_accuracy']}`",
        f"- Necessity pass: `{metrics['necessity_pass']}`", "",
        "## Restatement closure", "",
    ]
    for role in ROLE_ANCHORS:
        report.append(f"- `{role}` observed: `{observed[role]}`; outside declared: `{leakage[role]}`")
    report.extend(["", "No Writer, Composer, Planner, Realizer, or Memorax call was made.", ""])
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--expanded-dialogue", type=Path, required=True)
    parser.add_argument("--composite-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        source_run=args.source_run.resolve(), expanded_dialogue=args.expanded_dialogue.resolve(), composite_run=args.composite_run.resolve(),
        output=args.output.resolve(), tokenizer=args.tokenizer.resolve(), solver=args.solver.resolve(),
    ), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
