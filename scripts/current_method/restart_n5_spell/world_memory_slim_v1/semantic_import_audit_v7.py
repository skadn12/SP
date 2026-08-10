"""Recursive static import-graph audit for the v7 live semantic boundary."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


SEMANTIC_IMPORT_AUDIT_VERSION = "slim_v7_recursive_semantic_import_audit_v1"
FORBIDDEN_V6_SEMANTIC_LEAVES = frozenset({
    "live_fresh_e2e", "affordance_compiler", "binder", "memory",
    "realizer_claims", "schemas", "structural", "prompts",
    "stateful_graph_expansion", "stateful_outline_planner",
    "stateful_graph_orchestrator", "dynamic_generation_canary", "validation",
})


def audit_v7_recursive_import_graph(
    *, project_root: Path, runner_path: Path,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    runner_path = Path(runner_path).resolve()
    if not runner_path.is_file():
        return {"audit_version": SEMANTIC_IMPORT_AUDIT_VERSION,
                "valid": False, "errors": ["v7_runner_missing"],
                "visited_files": [], "edges": [], "forbidden_findings": []}
    package_init = runner_path.parent / "__init__.py"
    queue = [runner_path, package_init]
    visited: set[Path] = set()
    edges: list[dict[str, str]] = []
    findings: list[dict[str, str]] = []
    errors: list[str] = []
    while queue:
        path = queue.pop()
        if path in visited or not path.is_file():
            continue
        visited.add(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as error:
            errors.append(f"import_audit_syntax:{path}:{error}")
            continue
        current_module = _module_name(project_root, path)
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                target = _resolve_import_from(current_module, node.module, node.level,
                                              is_package=path.name == "__init__.py")
                if target:
                    targets.append(target)
            for module in targets:
                resolved = _resolve_module_file(project_root, module)
                edges.append({"from": current_module, "to": module,
                              "resolved_file": str(resolved) if resolved else "external_or_missing"})
                leaf = module.rsplit(".", 1)[-1]
                if (leaf in FORBIDDEN_V6_SEMANTIC_LEAVES and
                        "world_memory_slim_v1" in module):
                    findings.append({"from": current_module, "forbidden_module": module,
                                     "source_file": str(path)})
                if resolved and project_root in resolved.parents:
                    queue.append(resolved)
    return {
        "audit_version": SEMANTIC_IMPORT_AUDIT_VERSION,
        "runner_path": str(runner_path),
        "forbidden_semantic_leaves": sorted(FORBIDDEN_V6_SEMANTIC_LEAVES),
        "visited_files": [str(path.relative_to(project_root)) for path in sorted(visited)],
        "edges": edges, "forbidden_findings": findings,
        "forbidden_import_count": len(findings), "errors": errors,
        "valid": not findings and not errors,
    }


def _module_name(project_root: Path, path: Path) -> str:
    relative = path.relative_to(project_root)
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_import_from(current: str, module: str | None, level: int,
                         *, is_package: bool) -> str | None:
    if level == 0:
        return module
    parts = current.split(".")
    package = parts if is_package else parts[:-1]
    ascend = max(0, level - 1)
    if ascend > len(package):
        return None
    base = package[:len(package) - ascend]
    if module:
        base.extend(module.split("."))
    return ".".join(base)


def _resolve_module_file(project_root: Path, module: str) -> Path | None:
    relative = Path(*module.split("."))
    file_path = project_root / relative.with_suffix(".py")
    if file_path.is_file():
        return file_path.resolve()
    init_path = project_root / relative / "__init__.py"
    if init_path.is_file():
        return init_path.resolve()
    return None
