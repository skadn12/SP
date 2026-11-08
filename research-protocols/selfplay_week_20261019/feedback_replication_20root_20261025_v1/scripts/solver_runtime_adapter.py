#!/usr/bin/env python3
"""Use the locked Solver under the user-required current Python interpreter.

The interpreter-prefix guard and implementation-binding path resolver are
adapted in memory to use the original read-only dependency environment.  The
locked Solver and all original experiment/model/dependency files remain read-only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parents[1]
FROZEN_RUNNER = BASE / "solver_runner.py"
EXPECTED_CURRENT_PREFIX = Path("/root/miniconda3")
RECEIPT = BASE / "RUNTIME_INTERPRETER_ADAPTATION.json"
_RUNNER: Any = None
_ORIGINAL_PREFIX: str | None = None
READONLY_DEPENDENCIES = BASE.parents[2] / ".venv-local4b" / "lib" / "python3.10" / "site-packages"
_CACHE_ENV_BEFORE: dict[str, str | None] | None = None
_CACHE_ENV = {name: str(BASE / "runtime_cache" / suffix) for name, suffix in (
    ("TMPDIR", "tmp"), ("VLLM_CACHE_ROOT", "vllm"), ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
    ("TRITON_CACHE_DIR", "triton"), ("XDG_CACHE_HOME", "xdg"))}


def configure_readonly_dependencies() -> None:
    """Route imports to the existing locked-method runtime and caches to P."""
    global _CACHE_ENV_BEFORE
    if not READONLY_DEPENDENCIES.is_dir():
        raise ValueError("Original read-only dependency site-packages is missing")
    for name in ("torch", "transformers", "vllm", "tokenizers", "xgrammar"):
        loaded = sys.modules.get(name)
        if loaded is not None and getattr(loaded, "__file__", None):
            Path(loaded.__file__).resolve().relative_to(READONLY_DEPENDENCIES.resolve())
    if _CACHE_ENV_BEFORE is None:
        _CACHE_ENV_BEFORE = {name: os.environ.get(name) for name in _CACHE_ENV}
    for name, value in _CACHE_ENV.items():
        Path(value).mkdir(parents=True, exist_ok=True)
        os.environ[name] = value
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    path = str(READONLY_DEPENDENCIES.resolve())
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bind(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def verify(value: dict[str, Any]) -> None:
    if not isinstance(value, dict) or bind(value.get("path", "")) != {
            key: value.get(key) for key in ("path", "sha256", "bytes")}:
        raise ValueError(f"Immutable binding drift: {value}")


def checked_input_lock() -> dict[str, Any]:
    lock = read(BASE / "INPUT_LOCK.json")
    rows = lock.get("bindings")
    if not isinstance(rows, list) or not rows:
        raise ValueError("INPUT_LOCK lacks immutable bindings")
    for row in rows:
        verify(row)
    if bind(FROZEN_RUNNER) not in rows:
        raise ValueError("Frozen Solver is not bound by INPUT_LOCK")
    return lock


def get_runner():
    """Return the exact locked runner with explicit dependency-path adaptation."""
    global _RUNNER, _ORIGINAL_PREFIX
    checked_input_lock()
    if Path(sys.prefix).resolve() != EXPECTED_CURRENT_PREFIX.resolve():
        raise ValueError("Current Python must resolve to /root/miniconda3")
    configure_readonly_dependencies()
    if _RUNNER is None:
        spec = importlib.util.spec_from_file_location("fgr20_current_python_frozen_runner", FROZEN_RUNNER)
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot import the frozen Solver")
        runner = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner
        spec.loader.exec_module(runner)
        expected_legacy = BASE.parents[2] / ".venv-local4b"
        if Path(runner.orig.PYTHON_PREFIX).resolve() != expected_legacy.resolve():
            raise ValueError("Unexpected legacy interpreter-prefix guard")
        _ORIGINAL_PREFIX = str(Path(runner.orig.PYTHON_PREFIX).resolve())
        runner.orig.PYTHON_PREFIX = Path(sys.prefix).resolve()
        original_api_paths = runner.v2.structured_api_paths

        def actual_imported_api_paths():
            # The frozen resolver uses sys.prefix.  With an explicit read-only
            # dependency path, bind the identical relative implementation files
            # under the package that is actually imported instead.
            original_package = (Path(sys.prefix) / "lib" / "python3.10" / "site-packages" / "vllm")
            spec = importlib.util.find_spec("vllm")
            if spec is None or spec.origin is None:
                raise ValueError("Cannot locate the original dependency vllm package")
            actual_package = Path(spec.origin).resolve().parent
            actual_package.relative_to(READONLY_DEPENDENCIES.resolve())
            return [actual_package / Path(path).relative_to(original_package)
                    for path in original_api_paths()]

        runner.v2.structured_api_paths = actual_imported_api_paths
        _RUNNER = runner
    if Path(_RUNNER.__file__).resolve() != FROZEN_RUNNER.resolve():
        raise ValueError("Runtime adapter no longer wraps the frozen Solver")
    if Path(_RUNNER.orig.PYTHON_PREFIX).resolve() != EXPECTED_CURRENT_PREFIX.resolve():
        raise ValueError("Interpreter-prefix adaptation drift")
    if (_RUNNER.MAX_TOKENS, _RUNNER.MODEL_LEN, _RUNNER.v2.BATCH) != (6144, 16384, 8):
        raise ValueError("Frozen Solver budget/batch drift")
    if str(_RUNNER.orig.MODEL.resolve()) != str((BASE.parents[2] / "restart-n5-runs/day5_B1_schedule_v1/merged_model").resolve()):
        raise ValueError("Frozen B1 model path drift")
    if _RUNNER.orig.TOKENIZER.resolve() != Path("/root/autodl-tmp/Qwen3-4B").resolve():
        raise ValueError("Frozen tokenizer path drift")
    if _RUNNER.v2.STRUCTURED_CONFIG != {"reasoning_parser": "qwen3", "enable_in_reasoning": False}:
        raise ValueError("Frozen THINK_OFF structured settings drift")
    return _RUNNER


def check_environment() -> dict[str, Any]:
    """Check config/API objects on CPU, without constructing an LLM or writing."""
    runner = get_runner()
    endpoint, config = runner.orig.endpoint_receipt()
    api = runner.v2.cpu_api_check()
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU compatibility check initialized a CUDA context")
    code = [Path(__file__).resolve(), FROZEN_RUNNER,
            BASE.parent / "thinking_switch_20261024_structured_v2/run.py",
            *runner.orig.code_paths()]
    bindings = [bind(path) for path in code]
    bindings.extend(api["implementation_bindings"])
    bindings = sorted({row["path"]: row for row in bindings}.values(), key=lambda row: row["path"])
    for row in bindings:
        verify(row)
    return {
        "schema_version": "fgr20_current_python_solver_adapter_v1",
        "status": "PASS_CPU_ONLY_EXPLICIT_RUNTIME_PATH_ADAPTATION",
        "adaptation": {"target": "imported_frozen_runner.orig.PYTHON_PREFIX",
                       "original_expected_prefix": _ORIGINAL_PREFIX,
                       "actual_current_prefix": str(Path(sys.prefix).resolve()),
                       "python_executable": str(Path(sys.executable).resolve()),
                       "scope": ["interpreter-prefix guard adapted in memory",
                                 "original read-only site-packages prepended before runtime imports",
                                 "structured API implementation-binding paths redirected to the actual imported package",
                                 "new runtime caches routed under this task directory"],
                       "readonly_dependency_path": str(READONLY_DEPENDENCIES.resolve()),
                       "dependency_install_or_update": False,
                       "scientific_model_or_decode_settings_changed": False,
                       "frozen_files_modified": False},
        "cache_environment": {"actual": _CACHE_ENV, "before": _CACHE_ENV_BEFORE,
                              "PYTHONDONTWRITEBYTECODE": "1", "sys_dont_write_bytecode": sys.dont_write_bytecode},
        "input_lock_binding": bind(BASE / "INPUT_LOCK.json"),
        "code_bindings": bindings,
        "endpoint": endpoint,
        "compatible_config_receipt": config,
        "cpu_api_check": api,
        "runtime_versions": runner.orig.versions(),
        "settings": {"model": str(runner.orig.MODEL.resolve()),
                     "tokenizer": str(runner.orig.TOKENIZER.resolve()),
                     "max_output_tokens": runner.MAX_TOKENS,
                     "max_model_length": runner.MODEL_LEN,
                     "batch_size": runner.v2.BATCH,
                     "condition": "FULL", "mode": "THINK_OFF",
                     "temperature": 0.8, "top_p": 0.9, "display_variants": 2,
                     "training": False, "original_four_rerun": False},
        "gpu_model_started": False,
        "cuda_context_initialized": False,
        "model_calls": 0,
    }


def ensure_receipt() -> dict[str, Any]:
    value = check_environment()
    if RECEIPT.exists():
        old = read(RECEIPT)
        comparable = {key: item for key, item in old.items() if key != "created_at_utc"}
        if comparable != value:
            raise RuntimeError("Existing runtime adaptation receipt does not match current bindings/environment")
        return old
    value["created_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    with RECEIPT.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "prepare", "run", "analyze"))
    args = parser.parse_args()
    if (BASE / "STOP_REQUEST").exists():
        raise RuntimeError("STOP_REQUEST is present")
    receipt = ensure_receipt()
    if args.action == "check":
        print(json.dumps({"status": receipt["status"], "receipt_binding": bind(RECEIPT),
                          "model_calls": 0, "gpu_model_started": False}, ensure_ascii=False))
        return 0
    runner = get_runner()
    getattr(runner, args.action)()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
