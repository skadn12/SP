"""Fresh-run path validation for formal Stage3 generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any


STATE_BEARING_OUTPUT_NAMES = (
    "samples.partial.jsonl",
    "samples.jsonl",
    "graph_conditioned_api_dialogue_raw.jsonl",
    "graph_conditioned_api_dialogue_validated.jsonl",
    "graph_conditioned_api_dialogue_rejects.jsonl",
    "graph_artifacts",
    "graph_proposal_cache",
)


def _within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def validate_fresh_stage3_run(
    *,
    run_root: str | Path,
    output_dir: str | Path,
    partial_samples: str | Path,
    failed_api_calls: str | Path,
    graph_proposal_cache_dir: str | Path | None,
    fixed_stage3_graph_artifact: str | Path | None,
    stage3_operation_state_path: str | Path | None,
    no_resume: bool,
) -> dict[str, Any]:
    """Reject any path that could read, overwrite, or reuse a prior Stage3 run."""

    run = Path(run_root).resolve()
    output = Path(output_dir).resolve()
    partial = Path(partial_samples).resolve()
    failed = Path(failed_api_calls).resolve()
    issues: list[dict[str, Any]] = []
    if not no_resume:
        issues.append(
            {
                "code": "fresh_stage3_requires_no_resume",
                "message": "A fresh Stage3 run must pass --no-resume.",
            }
        )
    if not _within(output, run):
        issues.append(
            {
                "code": "fresh_stage3_output_outside_run_root",
                "path": str(output),
                "run_root": str(run),
            }
        )
    for label, path in (("partial_samples", partial), ("failed_api_calls", failed)):
        if not _within(path, output):
            issues.append(
                {
                    "code": "fresh_stage3_state_path_outside_output",
                    "field": label,
                    "path": str(path),
                    "output_dir": str(output),
                }
            )
    cache = None
    if graph_proposal_cache_dir:
        cache = Path(graph_proposal_cache_dir).resolve()
        if not _within(cache, output):
            issues.append(
                {
                    "code": "fresh_stage3_cache_outside_output",
                    "path": str(cache),
                    "output_dir": str(output),
                }
            )
    if fixed_stage3_graph_artifact:
        issues.append(
            {
                "code": "fresh_stage3_fixed_artifact_forbidden",
                "path": str(Path(fixed_stage3_graph_artifact).resolve()),
            }
        )
    operation_state = (
        Path(stage3_operation_state_path).resolve()
        if stage3_operation_state_path
        else None
    )
    if operation_state is None or not _within(operation_state, run):
        issues.append(
            {
                "code": "fresh_stage3_operation_state_outside_run_root",
                "path": str(operation_state) if operation_state else None,
                "run_root": str(run),
            }
        )
    elif operation_state.exists():
        issues.append(
            {
                "code": "fresh_stage3_operation_state_already_exists",
                "path": str(operation_state),
            }
        )

    existing_state_paths = [
        str(output / name)
        for name in STATE_BEARING_OUTPUT_NAMES
        if (output / name).exists()
    ]
    if existing_state_paths:
        issues.append(
            {
                "code": "fresh_stage3_output_contains_state",
                "paths": existing_state_paths,
            }
        )

    # A formal fresh run is intentionally one-shot.  Reject a pre-existing
    # non-empty run directory even when its generation subdirectory happens to
    # be absent, because policy/reports from a prior attempt are still run
    # state and would otherwise be overwritten or silently mixed.
    existing_run_entries = (
        sorted(str(path) for path in run.iterdir()) if run.exists() else []
    )
    if existing_run_entries:
        issues.append(
            {
                "code": "fresh_stage3_run_root_not_empty",
                "paths": existing_run_entries,
            }
        )

    audit = {
        "schema_version": "stage3_fresh_run_isolation_v1",
        "pass": not issues,
        "run_root": str(run),
        "output_dir": str(output),
        "partial_samples": str(partial),
        "failed_api_calls": str(failed),
        "graph_proposal_cache_dir": str(cache) if cache else str(output / "graph_proposal_cache"),
        "fixed_stage3_graph_artifact": None,
        "stage3_operation_state_path": (
            str(operation_state) if operation_state else None
        ),
        "no_resume": bool(no_resume),
        "existing_state_paths": existing_state_paths,
        "existing_run_entries": existing_run_entries,
        "issues": issues,
    }
    if issues:
        raise RuntimeError(f"Fresh Stage3 run isolation failed: {issues}")
    return audit
