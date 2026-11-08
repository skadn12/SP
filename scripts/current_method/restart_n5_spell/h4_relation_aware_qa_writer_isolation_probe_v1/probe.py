"""H4 relation-aware QA Writer isolation probe, Condition B runner."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import prepare_empty, write_json
from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.live_adapter import load_config
from scripts.current_method.restart_n5_spell.simple_append_v1.live_runtime import load_json_object_strict
from scripts.current_method.restart_n5_spell.simple_append_v1.prompts import build_mcq_batch_prompt

from .attribution import compare, summarize_condition
from .prompt import build_relation_aware_writer_prompt


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
PROBE_VERSION = "h4_relation_aware_qa_writer_isolation_probe_v1"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _call_writer(*, prompt: str, config_path: Path, run_root: Path) -> dict[str, Any]:
    import urllib.request
    api = load_config(config_path)
    payload = {
        "model": api["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.25,
        "max_tokens": 8192,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        str(api["base_url"]).rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Authorization": "Bearer " + str(api["api_key"]), "Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(request, timeout=600) as response:
        body = json.loads(response.read().decode())
    choice = (body.get("choices") or [{}])[0]
    raw = str((choice.get("message") or {}).get("content") or "")
    parsed = load_json_object_strict(raw)
    write_json(run_root / "writer_response_envelope.json", parsed)
    (run_root / "writer_raw_response.txt").write_text(raw, encoding="utf-8")
    write_json(run_root / "writer_response_meta.json", {
        "model": api["model"], "usage": body.get("usage"),
        "finish_reason": choice.get("finish_reason"), "condition": "B_relation_aware",
    })
    return parsed


def _run_replayed_qa(*, source_run: Path, writer_output: Path, qa_run: Path, config: Path, tokenizer: Path, solver: Path) -> None:
    cmd = [
        "python", "-m", "scripts.current_method.restart_n5_spell.simple_append_v1.qa_live",
        "--source-run", str(source_run), "--run-root", str(qa_run),
        "--api-config", str(config), "--tokenizer", str(tokenizer),
        "--solver-model", str(solver), "--focus-round-index", "1",
        "--writer-output-replay", str(writer_output),
    ]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    (qa_run.parent / "qa_runner_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (qa_run.parent / "qa_runner_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0 and not (qa_run / "qa/final_mcq_batch.json").exists():
        raise RuntimeError(f"relation_aware_replayed_qa_failed:{completed.returncode}")


def _materialize_evaluation_aliases(*, qa_run: Path, output: Path) -> Path:
    """Adapt frozen QA filter names to the composition probe's input contract."""
    # The composition CLI refuses a non-empty output root. Keep this adapter as
    # a sibling so the output directory itself is empty when the CLI starts.
    evaluation_dir = output.parent / f"{output.name}__evaluation_aliases"
    evaluation_dir.mkdir(parents=True, exist_ok=False)
    for source in sorted((qa_run / "filter").glob("*.json")):
        target = evaluation_dir / f"{source.stem}_evaluation.json"
        shutil.copy2(source, target)
    return evaluation_dir


def _run_composition(*, qa_run: Path, source_run: Path, output: Path, evaluation_dir: Path | None = None) -> None:
    if evaluation_dir is None:
        evaluation_dir = _materialize_evaluation_aliases(qa_run=qa_run, output=output)
    cmd = [
        "python", "-m", "scripts.current_method.restart_n5_spell.qa_composition_probe_v1.probe",
        "--atomic-batch", str(qa_run / "qa/final_mcq_batch.json"),
        "--expanded-dialogue", str(source_run / "expanded_dialogue.json"),
        "--evaluations", str(evaluation_dir), "--run-root", str(output),
    ]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    (output.parent / "composition_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output.parent / "composition_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"relation_aware_composition_failed:{completed.returncode}")


def run(*, relation_probe_run: Path, baseline_qa_run: Path, baseline_composition: Path, config: Path, tokenizer: Path, solver: Path, run_root: Path, relation_audit_path: Path | None = None) -> dict[str, Any]:
    prepare_empty(run_root)
    source_run = relation_probe_run / "qa_source"
    relation_contract = _load(relation_probe_run / "relation_contract.json")
    relation_audit = _load(relation_audit_path or (relation_probe_run / "relation_audit.json"))
    expanded = _load(source_run / "expanded_dialogue.json")
    prompt = build_relation_aware_writer_prompt(
        expanded=expanded, relation_contract=relation_contract,
        relation_audit=relation_audit)
    write_json(run_root / "relation_aware_writer_prompt.json", prompt)
    writer_output = _call_writer(prompt=prompt["prompt"], config_path=config, run_root=run_root)
    writer_output_path = run_root / "writer_response_envelope.json"
    qa_run = run_root / "qa_run"
    _run_replayed_qa(source_run=source_run, writer_output=writer_output_path,
                     qa_run=qa_run, config=config, tokenizer=tokenizer, solver=solver)
    baseline_composition_recomputed = run_root / "baseline_composition_offline"
    _run_composition(qa_run=baseline_qa_run, source_run=source_run,
                     output=baseline_composition_recomputed)
    composition = run_root / "composition_offline"
    _run_composition(qa_run=qa_run, source_run=source_run, output=composition)
    baseline_metrics = summarize_condition(
        qa_run=baseline_qa_run, composition_run=baseline_composition_recomputed,
        relation_audit=relation_audit, condition="A_baseline")
    relation_metrics = summarize_condition(
        qa_run=qa_run, composition_run=composition,
        relation_audit=relation_audit, condition="B_relation_aware")
    comparison = compare(baseline=baseline_metrics, relation_aware=relation_metrics)
    write_json(run_root / "baseline_metrics.json", baseline_metrics)
    write_json(run_root / "relation_aware_metrics.json", relation_metrics)
    write_json(run_root / "ab_comparison.json", comparison)
    metrics = {
        "probe_version": PROBE_VERSION,
        "hypothesis": "H4",
        "condition_a": "production_writer_baseline",
        "condition_b": "relation_aware_writer_wrapper",
        "relation_probe_run": str(relation_probe_run.resolve()),
        "baseline_qa_run": str(baseline_qa_run.resolve()),
        "baseline_composition_run": str(baseline_composition_recomputed.resolve()),
        "relation_aware_qa_run": str(qa_run.resolve()),
        "relation_aware_composition_run": str(composition.resolve()),
        "writer_api_calls_condition_b": 1,
        "solver_api_calls": 0,
        "planner_calls": 0,
        "realizer_calls": 0,
        "memorax_calls": 0,
        "analysis_execution_api_calls": 0,
        "baseline": baseline_metrics,
        "relation_aware": relation_metrics,
        "ab_comparison": comparison,
    }
    write_json(run_root / "metrics.json", metrics)
    report = [
        "# H4 Relation-aware QA Writer Isolation Probe v1", "",
        f"Conclusion: **{comparison['conclusion']}**", "",
        "Condition A uses the unchanged production Writer artifacts from the same "
        "relation-contract transcript. Condition B changes only the Writer prompt "
        "through an isolated wrapper; Solver, grounding, K=5, composition, and "
        "admission remain unchanged.", "",
        "## Condition A", "",
        f"- Atomic generated: `{baseline_metrics['atomic_generated']}`",
        f"- Relation-targeted: `{baseline_metrics['relation_targeted_questions']}`",
        f"- Grounding + K=5: `{baseline_metrics['relation_targeted_grounding_k5']}`",
        f"- Relation-targeted easy rejected: `{baseline_metrics['relation_targeted_easy_rejected']}`",
        f"- Composer-ready: `{baseline_metrics['composer_ready']}`",
        f"- Admitted: `{baseline_metrics['admitted']}`", "",
        "## Condition B", "",
        f"- Atomic generated: `{relation_metrics['atomic_generated']}`",
        f"- Relation-targeted: `{relation_metrics['relation_targeted_questions']}`",
        f"- Relation-targeted grounding + K=5: `{relation_metrics['relation_targeted_grounding_k5']}`",
        f"- Relation-targeted easy rejected: `{relation_metrics['relation_targeted_easy_rejected']}`",
        f"- Relation utilization rate: `{relation_metrics['relation_utilization_rate']:.3f}`",
        f"- Difficulty-qualified relation utilization: `{relation_metrics['difficulty_qualified_relation_utilization_rate']:.3f}`",
        f"- Composer-ready: `{relation_metrics['composer_ready']}`",
        f"- Admitted: `{relation_metrics['admitted']}`", "",
        "## Necessity", "",
        "Composer response and remove-fact Solver ablations are recorded as not run "
        "until a Composer-ready pair exists; no necessity claim is made without them.", "",
    ]
    (run_root / "report.md").write_text("\n".join(report), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="H4 relation-aware QA Writer isolation probe")
    parser.add_argument("--relation-probe-run", type=Path, required=True)
    parser.add_argument("--baseline-qa-run", type=Path, required=True)
    parser.add_argument("--baseline-composition", type=Path, required=True)
    parser.add_argument("--relation-audit", type=Path)
    parser.add_argument("--config", type=Path, default=Path("/root/autodl-tmp/gxy-selfplay/config.json"))
    parser.add_argument("--tokenizer", type=Path, default=Path("/root/autodl-tmp/Qwen3-4B"))
    parser.add_argument("--solver", type=Path, default=Path("/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/training/models/merged"))
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(relation_probe_run=args.relation_probe_run.resolve(), baseline_qa_run=args.baseline_qa_run.resolve(), baseline_composition=args.baseline_composition.resolve(), config=args.config.resolve(), tokenizer=args.tokenizer.resolve(), solver=args.solver.resolve(), run_root=args.run_root.resolve(), relation_audit_path=args.relation_audit.resolve() if args.relation_audit else None)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
