"""Zero-API route contract preflight.

This module deliberately does not repair or rewrite any route.  It catches
cross-branch contract drift before a live run is started.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
SPELL = ROOT / "scripts/current_method/restart_n5_spell"
DIRECT_QA = SPELL / "simple_append_memorax_v1"
QA_CORE = SPELL / "simple_append_v1"
EXPANSION = SPELL / "simple_append_memorax_v1_memory_event_continuation_probe_v1"
STRUCTURAL_QA = SPELL / "simple_append_memorax_v1_fullr1_devdiv_qa_structural_v19_probe_v1"
BASELINE_MANIFEST = SPELL / "route_integrity/reports/ROUTE_BASELINE_MANIFEST.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _git_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


def _has_assignment(path: Path, name: str, expected: Any) -> bool:
    tree = ast.parse(_text(path), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not any(isinstance(target, ast.Name) and target.id == name
                       for target in targets):
                continue
            value = node.value
            if isinstance(value, ast.Constant):
                return value.value == expected
    return False


def _check(path: Path, predicate: bool, reason: str) -> dict[str, Any]:
    return {"path": str(path), "ok": bool(predicate), "reason": reason}


def _route_source_files() -> list[Path]:
    roots = (DIRECT_QA, QA_CORE, EXPANSION, STRUCTURAL_QA,
             SPELL / "route_integrity")
    files: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if (path.is_file() and "__pycache__" not in path.parts and
                    "reports" not in path.parts and
                    path.suffix in {".py", ".md"}):
                files.add(path)
    for path in (SPELL / "core.py", SPELL / "ledger.py",
                 SPELL / "live_backends.py", SPELL / "minimal_admission.py",
                 SPELL / "questioner_contract.py"):
        if path.is_file():
            files.add(path)
    return sorted(files)


def _baseline_matches() -> tuple[bool, str]:
    if not BASELINE_MANIFEST.is_file():
        return False, "route_baseline_manifest_missing"
    try:
        baseline = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
        expected = baseline.get("source_files")
        if not isinstance(expected, dict) or not expected:
            return False, "route_baseline_manifest_invalid"
        current = {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in _route_source_files()
        }
        if current != expected:
            missing = sorted(set(expected) - set(current))
            added = sorted(set(current) - set(expected))
            changed = sorted(
                key for key in set(expected) & set(current)
                if expected[key] != current[key]
            )
            return False, "route_baseline_mismatch:" + json.dumps(
                {"missing": missing, "added": added, "changed": changed},
                ensure_ascii=False, sort_keys=True,
            )
        return True, "route_baseline_matches"
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        return False, f"route_baseline_read_failed:{type(error).__name__}"


def preflight() -> dict[str, Any]:
    direct_live = DIRECT_QA / "qa_live.py"
    direct_prompt = DIRECT_QA / "prompts.py"
    qa_core = QA_CORE / "qa.py"
    qa_core_prompt = QA_CORE / "prompts.py"
    expansion_live = EXPANSION / "live.py"
    expansion_workflow = EXPANSION / "workflow.py"
    structural_live = STRUCTURAL_QA / "structural_qa_live.py"

    checks = [
        _check(direct_live, direct_live.is_file(), "direct QA entry exists"),
        _check(direct_prompt, direct_prompt.is_file(), "direct QA prompt exists"),
        _check(qa_core, qa_core.is_file(), "shared direct QA contract exists"),
        _check(expansion_live, expansion_live.is_file(), "expansion entry exists"),
        _check(expansion_workflow, expansion_workflow.is_file(),
               "expansion workflow exists"),
        _check(structural_live, structural_live.is_file(),
               "Structural v19 remains an explicit sibling"),
    ]

    if all(item["ok"] for item in checks):
        direct_text = _text(direct_live)
        prompt_text = _text(direct_prompt)
        core_text = _text(qa_core)
        core_prompt_text = _text(qa_core_prompt)
        expansion_text = _text(expansion_live) + _text(expansion_workflow)
        structural_text = _text(structural_live)

        checks.extend([
            _check(
                direct_live,
                'RUNNER_VERSION = "simple_append_memorax_qa_live_v2_spell_frontier_k5"'
                in direct_text,
                "canonical Memorax QA runner is direct Spell/frontier-k5",
            ),
            _check(
                qa_core,
                '"correct_option"' in core_text and
                "materialize_solver_candidate" in core_text and
                "assigned_answer_letters" in core_text,
                "direct QA keeps letter binding and programmatic answer rotation",
            ),
            _check(
                qa_core_prompt,
                '"correct_option": "A"' in core_prompt_text and
                "gold_text" not in core_prompt_text,
                "direct Writer contract uses correct_option, not v19 gold_text",
            ),
            _check(
                direct_live,
                "structural_qa_writer_v19" not in direct_text and
                "structural_contract" not in direct_text,
                "direct QA does not import Structural v19 contract",
            ),
            _check(
                direct_prompt,
                "build_mcq_batch_prompt" in prompt_text and
                "gold_text" not in prompt_text,
                "Memorax direct QA prompt preserves direct MCQ output",
            ),
            _check(
                expansion_live,
                "Planner" in expansion_text and "Realizer" in expansion_text,
                "expansion remains Planner -> Realizer route",
            ),
            _check(
                structural_live,
                "WRITER_COMPLETION_TOKENS" in structural_text,
                "Structural v19 owns its Writer-only budget locally",
            ),
        ])

    dirty = _git_dirty()
    route_files = [direct_live, direct_prompt, qa_core, qa_core_prompt,
                   expansion_live, expansion_workflow, structural_live]
    hashes = {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in route_files if path.is_file()
    }
    baseline_ok, baseline_reason = _baseline_matches()
    checks.append(_check(BASELINE_MANIFEST, baseline_ok, baseline_reason))
    failures = [item["reason"] for item in checks if not item["ok"]]
    if dirty and not BASELINE_MANIFEST.is_file():
        failures.append("working_tree_dirty_baseline_not_frozen")
    return {
        "report_version": "route_contract_preflight_v1",
        "api_calls": 0,
        "solver_calls": 0,
        "expansion_calls": 0,
        "memorax_calls": 0,
        "working_tree_dirty": dirty,
        "decision": "pass" if not failures else "needs_baseline_freeze",
        "failures": failures,
        "checks": checks,
        "route_file_sha256": hashes,
        "canonical_routes": {
            "expansion": str(EXPANSION.relative_to(ROOT)),
            "qa": str(DIRECT_QA.relative_to(ROOT)),
            "qa_core": str(QA_CORE.relative_to(ROOT)),
            "structural_qa_experimental": str(STRUCTURAL_QA.relative_to(ROOT)),
        },
    }


def main() -> int:
    result = preflight()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["decision"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
