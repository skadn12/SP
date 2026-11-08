"""Repository import closure plus observed offline execution coverage."""
from __future__ import annotations

import ast
from contextlib import contextmanager
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import sys
from unittest.mock import patch

from ..core import atomic_json, sha256_file

CODE = Path(__file__).resolve().parent
REPO = CODE.parents[3]


def record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha256_file(path), 'size': path.stat().st_size}


def loaded_repository_files():
    files = set()
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if not name:
            continue
        path = Path(name).resolve()
        if path.suffix == '.pyc':
            try:
                path = Path(importlib.util.source_from_cache(str(path)))
            except ValueError:
                continue
        if path.is_relative_to(REPO) and path.is_file():
            files.add(path)
    return files


def module_file(name):
    base = REPO.joinpath(*name.split('.'))
    for path in (base.with_suffix('.py'), base / '__init__.py'):
        if path.is_file():
            return path.resolve()
    return None


def static_closure(seeds):
    """Trace all local imports, including function-local and error-path imports."""
    pending = list(seeds)
    files, edges, dynamic = set(), [], []
    while pending:
        path = Path(pending.pop()).resolve()
        if path in files or not path.is_relative_to(REPO) or not path.is_file():
            continue
        files.add(path)
        if path.suffix != '.py':
            continue
        relative = path.relative_to(REPO).with_suffix('')
        parts = list(relative.parts)
        if parts[-1] == '__init__':
            parts.pop()
            package = '.'.join(parts)
        else:
            package = '.'.join(parts[:-1])
        names = []
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                try:
                    base = (importlib.util.resolve_name('.' * node.level + (node.module or ''), package)
                            if node.level else node.module or '')
                except (ImportError, ValueError):
                    continue
                names.append(base)
                names.extend(base + '.' + alias.name for alias in node.names if alias.name != '*')
            elif isinstance(node, ast.Call) and (
                    isinstance(node.func, ast.Name) and node.func.id == '__import__' or
                    isinstance(node.func, ast.Attribute) and node.func.attr == 'import_module'):
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    name = node.args[0].value
                    if not name.startswith('.'):
                        names.append(name)
                else:
                    dynamic.append({'path': str(path), 'line': node.lineno,
                                    'coverage': 'Observed modules and repository execution guard supplement static tracing.'})
        for name in names:
            components = name.split('.')
            for length in range(1, len(components) + 1):
                target = module_file('.'.join(components[:length]))
                if target is not None:
                    edges.append({'source': str(path), 'target': str(target)})
                    if target not in files:
                        pending.append(target)
    return files, edges, dynamic


def environment_identity():
    libraries = {}
    for package in ('transformers', 'tokenizers', 'huggingface-hub', 'jsonschema',
                    'referencing', 'attrs', 'rpds-py', 'numpy', 'torch', 'requests'):
        try:
            libraries[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            libraries[package] = None
    return {'python': sys.version, 'executable': record(Path(sys.executable)),
            'platform': platform.platform(), 'libraries': libraries,
            'bytecode_writes_disabled': sys.dont_write_bytecode}


@contextmanager
def forbid_network():
    with patch('socket.socket.connect', side_effect=AssertionError('NETWORK_FORBIDDEN')), \
         patch('socket.create_connection', side_effect=AssertionError('NETWORK_FORBIDDEN')), \
         patch('urllib.request.urlopen', side_effect=AssertionError('NETWORK_FORBIDDEN')):
        yield


def collect_coverage(run, dry_run_report):
    run = Path(run)
    observed = loaded_repository_files() | {Path(p) for p in dry_run_report['repository_modules']}
    # Include all new source/tests and every branch imported by the runner.
    seeds = set(CODE.glob('*.py')) | observed
    traced, edges, dynamic = static_closure(seeds)
    required = {CODE.parent / name for name in
                ('backends.py', 'simple_append_v1/live_runtime.py', 'live_backends.py', 'core.py')}
    checks = {'dry_run_pass': dry_run_report['status'] == 'PASS',
              'observed_modules_covered': observed <= traced,
              'explicit_dependencies_covered': required <= traced,
              'explicit_dependencies_observed': required <= observed,
              'python_bytecode_disabled': sys.dont_write_bytecode}
    result = {'version': 'basic_miner_v3_dependency_coverage_v1',
              'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
              'method': 'AST recursive repository import closure, including local imports, unioned with modules loaded during network-blocked shared Miner plus original HTTP client dry-runs. Before and after each live stage all loaded repository files must belong to this pre-frozen set; a Python call profile guard rejects execution of any unfrozen repository file.',
              'repository_files': [record(p) for p in sorted(traced)],
              'observed_repository_modules': [record(p) for p in sorted(observed)],
              'static_import_edges': edges, 'nonliteral_dynamic_import_sites': dynamic,
              'dry_run': dry_run_report, 'environment': environment_identity()}
    atomic_json(run / 'audit/dependency_coverage_report.json', result)
    return result


def check_loaded_coverage(frozen_paths):
    unknown = loaded_repository_files() - {Path(p) for p in frozen_paths}
    if unknown:
        from .runtime import GuardrailStop
        raise GuardrailStop('unfrozen_loaded_repository_dependency:' + ','.join(map(str, sorted(unknown))))


@contextmanager
def repository_execution_guard(records):
    """Reject unlisted repository Python execution before it can send a request."""
    allowed = {row['path'] for row in records}
    previous = sys.getprofile()
    def profile(frame, event, arg):
        if event != 'call':
            return
        filename = frame.f_code.co_filename
        if not filename.startswith(str(REPO) + '/'):
            return
        if str(Path(filename).resolve()) not in allowed:
            from .runtime import GuardrailStop
            raise GuardrailStop('unfrozen_executed_repository_dependency:' + filename)
    sys.setprofile(profile)
    try:
        yield
    finally:
        sys.setprofile(previous)
