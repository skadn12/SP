from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "old_improved_gate400_sample0_5"
SEED_SCOPE = "sample3_5_train_only"
FIXED_SOLVER_NAME = "solver_r3_ctx15360_full"
DEFAULT_SEED_DATA = str(ROOT / "eval_sets/graph_expansion_challenger_v1/r3_auto_full/samples.jsonl")
DEFAULT_SOLVER = str(ROOT / "outputs/qwen3_4b_lora_r3_frontier_plus_clean_replay_from_r2_ctx15360_full")
DEFAULT_OUTPUT_DIR = f"outputs/{EXPERIMENT}"
DEFAULT_MODEL = "/root/autodl-tmp/Qwen3-4B"
DEFAULT_CONFIG = str(ROOT / "config.json")
DEFAULT_CHALLENGER_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_CHALLENGER_MODEL = "Qwen/Qwen3-14B"
ALLOWED_SAMPLE_NAMES = {f"sample{i}" for i in range(3, 6)}
REQUIRED_CHALLENGER_KEYS = ("attack_plan", "graph_update", "dialogue", "qa")
RAW_RESPONSE_PREVIEW_CHARS = 800
ACTIVE_ARGS: argparse.Namespace | None = None

BASE_SCRIPT = Path(__file__).with_name("run_old_improved_gate200.py")
BASE_SPEC = importlib.util.spec_from_file_location("old_improved_gate200_base", BASE_SCRIPT)
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise RuntimeError(f"Failed to load base script: {BASE_SCRIPT}")
base = importlib.util.module_from_spec(BASE_SPEC)
sys.modules[BASE_SPEC.name] = base
BASE_SPEC.loader.exec_module(base)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the old_method_improved gate400 sample3_5 generation/eval pipeline with a fixed r3 solver and API challenger."
    )
    parser.add_argument("--solver_checkpoint", "--solver-checkpoint", default=DEFAULT_SOLVER)
    parser.add_argument("--seed_data", "--seed-data", default=DEFAULT_SEED_DATA)
    parser.add_argument("--output_dir", "--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num_samples", "--num-samples", type=int, default=400)
    parser.add_argument("--k_eval", "--k-eval", type=int, default=5)
    parser.add_argument("--max_length", "--max-length", type=int, default=15360)
    parser.add_argument("--max_largest_question_type_share", "--max-largest-question-type-share", type=float, default=0.35)
    parser.add_argument("--max_largest_attack_skill_share", "--max-largest-attack-skill-share", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--eval_backend", "--eval-backend", choices=("auto", "vllm", "transformers", "heuristic"), default="auto")
    parser.add_argument("--base_model", "--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--max_model_len", "--max-model-len", type=int, default=16384)
    parser.add_argument("--max_new_tokens", "--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", "--top-p", type=float, default=0.9)
    parser.add_argument("--gpu_memory_utilization", "--gpu-memory-utilization", type=float, default=0.86)
    parser.add_argument("--limit_seed_rows", "--limit-seed-rows", type=int, default=None)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Project config.json with api.base_url/api.api_key/api.model.")
    parser.add_argument("--challenger_base_url", "--challenger-base-url", default=None)
    parser.add_argument("--challenger_model", "--challenger-model", default=None)
    parser.add_argument("--challenger_api_key", "--challenger-api-key", default=None)
    parser.add_argument("--llm_timeout", "--llm-timeout", type=int, default=180)
    parser.add_argument("--max_llm_retries", "--max-llm-retries", type=int, default=4)
    parser.add_argument("--max_api_failures", "--max-api-failures", type=int, default=50)
    parser.add_argument("--challenger_temperature", "--challenger-temperature", type=float, default=0.25)
    parser.add_argument("--challenger_max_tokens", "--challenger-max-tokens", type=int, default=4096)
    parser.add_argument("--checkpoint_every", "--checkpoint-every", type=int, default=1)
    parser.add_argument("--partial_samples", "--partial-samples", default=None)
    parser.add_argument("--no_resume", "--no-resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def source_dialogue_path(row: dict[str, Any]) -> str:
    for value in (
        row.get("source_dialogue_path"),
        row.get("metadata", {}).get("source_dialogue_path") if isinstance(row.get("metadata"), dict) else None,
        row.get("state_chain", {}).get("source_dialogue_path") if isinstance(row.get("state_chain"), dict) else None,
    ):
        if value:
            return str(value)
    return ""


def extract_seed_source(row: dict[str, Any]) -> dict[str, Any]:
    split = str(row.get("split") or "")
    heldout_split = row.get("heldout_split")
    source_path = source_dialogue_path(row)
    source_split = "unknown"
    sample_name = "unknown"
    match = re.search(r"/(train|dev|test)/(sample\d+)/", source_path)
    if match:
        source_split = match.group(1)
        sample_name = match.group(2)
    else:
        sample_match = re.search(r"/(sample\d+)/", source_path)
        if sample_match:
            sample_name = sample_match.group(1)
    forbidden_reasons: list[str] = []
    if split != "train":
        forbidden_reasons.append(f"split={split or 'missing'}")
    if heldout_split not in (None, "", "None"):
        forbidden_reasons.append(f"heldout_split={heldout_split}")
    if source_split != "train":
        forbidden_reasons.append(f"source_split={source_split}")
    if sample_name not in ALLOWED_SAMPLE_NAMES:
        forbidden_reasons.append(f"sample_name={sample_name}")
    return {
        "split": split or "unknown",
        "heldout_split": None if heldout_split in (None, "", "None") else str(heldout_split),
        "source_split": source_split,
        "sample_name": sample_name,
        "source_dialogue_path": source_path,
        "allowed_train_seed": not forbidden_reasons,
        "forbidden_reasons": forbidden_reasons,
    }


def build_seed_source_audit(rows: list[dict[str, Any]], seed_data_path: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_split_counts: Counter[str] = Counter()
    raw_heldout_counts: Counter[str] = Counter()
    raw_source_split_counts: Counter[str] = Counter()
    raw_sample_name_counts: Counter[str] = Counter()
    for row in rows:
        source = extract_seed_source(row)
        raw_split_counts[source["split"]] += 1
        raw_heldout_counts[str(source["heldout_split"])] += 1
        raw_source_split_counts[source["source_split"]] += 1
        raw_sample_name_counts[source["sample_name"]] += 1

    train_rows = [row for row in rows if str(row.get("split") or "") == "train"]
    selected_meta = [extract_seed_source(row) for row in train_rows]
    forbidden = [meta for meta in selected_meta if not meta["allowed_train_seed"]]
    selected_source_split_counts = Counter(meta["source_split"] for meta in selected_meta)
    selected_sample_name_counts = Counter(meta["sample_name"] for meta in selected_meta)

    audit = {
        "seed_source_path": seed_data_path,
        "allowed_sample_names": sorted(ALLOWED_SAMPLE_NAMES),
        "raw_source_total": len(rows),
        "raw_split_counts": dict(sorted(raw_split_counts.items())),
        "raw_heldout_counts": dict(sorted(raw_heldout_counts.items())),
        "raw_source_split_counts": dict(sorted(raw_source_split_counts.items())),
        "raw_sample_name_counts": dict(sorted(raw_sample_name_counts.items())),
        "total_seed_candidates": len(train_rows),
        "train_seed_count": len(train_rows),
        "dev_seed_count": 0,
        "heldout_seed_count": 0,
        "source_split_counts": dict(sorted(selected_source_split_counts.items())),
        "sample_name_counts": dict(sorted(selected_sample_name_counts.items())),
        "forbidden_seed_count": len(forbidden),
        "forbidden_reason_counts": dict(sorted(Counter(reason for meta in forbidden for reason in meta["forbidden_reasons"]).items())),
        "pass": len(train_rows) > 0 and len(forbidden) == 0,
    }
    if audit["pass"]:
        allowed_rows = train_rows
    else:
        allowed_rows = []
    return audit, allowed_rows


def load_config_payload(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def apply_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    payload = load_config_payload(args.config)
    api = payload.get("api") if isinstance(payload.get("api"), dict) else {}
    if args.challenger_base_url is None:
        args.challenger_base_url = str(api.get("base_url") or DEFAULT_CHALLENGER_BASE_URL)
    if args.challenger_model is None:
        args.challenger_model = str(api.get("model") or DEFAULT_CHALLENGER_MODEL)
    if args.challenger_api_key is None:
        key = api.get("api_key") if isinstance(api, dict) else None
        args.challenger_api_key = str(key) if key else None
    args.config_payload = payload
    return args


def json_from_text(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(text[start : end + 1])
                return payload if isinstance(payload, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def json_candidate_from_text(text: str) -> tuple[Any, bool]:
    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1]), True
            except json.JSONDecodeError:
                return None, False
    return None, False


def raw_response_preview(text: Any, limit: int = RAW_RESPONSE_PREVIEW_CHARS) -> str:
    preview = str(text or "").replace("\r", " ").replace("\n", "\\n")
    return preview[:limit]


def validate_challenger_payload(payload: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not isinstance(payload, dict):
        return None, {
            "error_type": "missing_required_keys",
            "missing_keys": list(REQUIRED_CHALLENGER_KEYS),
        }
    missing = [key for key in REQUIRED_CHALLENGER_KEYS if not isinstance(payload.get(key), dict)]
    if missing:
        return None, {
            "error_type": "missing_required_keys",
            "missing_keys": missing,
        }
    return payload, None


def retry_instruction_for_error(error: dict[str, Any] | None) -> str:
    if not error:
        return ""
    error_type = error.get("error_type")
    if error_type == "missing_required_keys":
        missing_keys = ", ".join(str(key) for key in error.get("missing_keys") or REQUIRED_CHALLENGER_KEYS)
        return "\n".join(
            [
                f"Previous response was invalid because it missed required keys: {missing_keys}.",
                "Return exactly one JSON object with keys:",
                ", ".join(REQUIRED_CHALLENGER_KEYS) + ".",
                "Do not omit any key.",
            ]
        )
    if error_type == "invalid_json":
        return "\n".join(
            [
                "Previous response was not valid JSON.",
                "Return exactly one JSON object with keys:",
                ", ".join(REQUIRED_CHALLENGER_KEYS) + ".",
                "Do not omit any key.",
            ]
        )
    return "\n".join(
        [
            "Previous API attempt did not yield a valid challenger payload.",
            "Return exactly one JSON object with keys:",
            ", ".join(REQUIRED_CHALLENGER_KEYS) + ".",
            "Do not omit any key.",
        ]
    )


def build_challenger_request_payload(
    args: argparse.Namespace,
    prompt_payload: dict[str, Any],
    previous_error: dict[str, Any] | None,
) -> dict[str, Any]:
    attempt_prompt = copy.deepcopy(prompt_payload)
    retry_instruction = retry_instruction_for_error(previous_error)
    if retry_instruction:
        attempt_prompt["previous_response_error"] = retry_instruction
    user_content = json.dumps(attempt_prompt, ensure_ascii=False)
    if retry_instruction:
        user_content += "\n\n" + retry_instruction
    return {
        "model": args.challenger_model,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict JSON-only challenger rewriter for graph-expansion QA samples.",
            },
            {"role": "user", "content": user_content},
        ],
        "temperature": args.challenger_temperature,
        "max_tokens": args.challenger_max_tokens,
    }


def classify_api_exception(exc: BaseException) -> str:
    if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)):
        return "timeout"
    text = str(exc).lower()
    if "timed out" in text or "timeout" in text or "operation timed out" in text:
        return "timeout"
    if isinstance(exc, json.JSONDecodeError):
        return "invalid_json"
    return "api_request_error"


def post_json_with_curl(url: str, payload: dict[str, Any], api_key: str, timeout_seconds: int) -> dict[str, Any]:
    curl = shutil.which("curl")
    if not curl:
        raise urllib.error.URLError("curl not available")
    config_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as config:
            config_path = config.name
            config.write(f'url = "{url}"\n')
            config.write('request = "POST"\n')
            config.write(f'header = "Authorization: Bearer {api_key}"\n')
            config.write('header = "Content-Type: application/json"\n')
            config.write('data-binary = "@-"\n')
            config.write("silent\n")
            config.write("show-error\n")
            config.write("fail-with-body\n")
            config.write(f"max-time = {timeout_seconds}\n")
        proc = subprocess.run(
            [curl, "--config", config_path],
            input=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds + 5,
            check=False,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            stdout = proc.stdout.decode("utf-8", errors="replace").strip()
            raise urllib.error.URLError(stderr or stdout or f"curl exited {proc.returncode}")
        return json.loads(proc.stdout.decode("utf-8"))
    finally:
        if config_path:
            try:
                os.unlink(config_path)
            except FileNotFoundError:
                pass


def call_challenger_api(args: argparse.Namespace, prompt_payload: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    api_key = args.challenger_api_key or os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key or not args.challenger_base_url or not args.challenger_model:
        return None, {"used": False, "reason": "missing_api_config"}
    url = str(args.challenger_base_url).rstrip("/") + "/chat/completions"
    errors: list[dict[str, Any]] = []
    previous_error: dict[str, Any] | None = None
    for attempt in range(1, args.max_llm_retries + 2):
        request_payload = build_challenger_request_payload(args, prompt_payload, previous_error)
        try:
            if shutil.which("curl"):
                data = post_json_with_curl(url, request_payload, api_key, max(1, int(args.llm_timeout)))
            else:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(request_payload, ensure_ascii=False).encode("utf-8"),
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=args.llm_timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
            content = str(data["choices"][0]["message"]["content"])
            candidate, parsed_json = json_candidate_from_text(content)
            if not parsed_json:
                previous_error = {
                    "attempt": attempt,
                    "error_type": "invalid_json",
                    "missing_keys": [],
                    "raw_response_preview": raw_response_preview(content),
                }
                errors.append(previous_error)
            else:
                parsed_payload, validation_error = validate_challenger_payload(candidate)
                if parsed_payload is not None:
                    return parsed_payload, {
                        "used": True,
                        "attempts": attempt,
                        "model": args.challenger_model,
                        "base_url": args.challenger_base_url,
                        "previous_errors": errors,
                    }
                previous_error = {
                    "attempt": attempt,
                    "error_type": validation_error.get("error_type") if validation_error else "missing_required_keys",
                    "missing_keys": (validation_error or {}).get("missing_keys", []),
                    "raw_response_preview": raw_response_preview(content),
                }
                errors.append(previous_error)
        except (KeyError, json.JSONDecodeError, urllib.error.URLError, TimeoutError, ValueError, subprocess.TimeoutExpired) as exc:
            previous_error = {
                "attempt": attempt,
                "error_type": classify_api_exception(exc),
                "missing_keys": [],
                "error": str(exc),
                "raw_response_preview": "",
            }
            errors.append(previous_error)
        if attempt <= args.max_llm_retries:
            time.sleep(min(2**attempt, 20))
    last_error = errors[-1] if errors else {}
    return None, {
        "used": False,
        "attempts": len(errors) or args.max_llm_retries + 1,
        "model": args.challenger_model,
        "base_url": args.challenger_base_url,
        "error_type": last_error.get("error_type", "challenger_api_failed"),
        "missing_keys": last_error.get("missing_keys", []),
        "raw_response_preview": last_error.get("raw_response_preview", ""),
        "errors": errors,
    }


def require_challenger_payload(payload: dict[str, Any] | None, meta: dict[str, Any]) -> dict[str, Any]:
    parsed_payload, validation_error = validate_challenger_payload(payload)
    if parsed_payload is None:
        raise RuntimeError(f"challenger API payload validation failed {validation_error}: {meta}")
    return parsed_payload


class ChallengerApiFailure(RuntimeError):
    def __init__(self, meta: dict[str, Any]):
        super().__init__(str(meta))
        self.meta = meta


class ApiFailureLimitExceeded(RuntimeError):
    pass


def sample_resume_key(sample: dict[str, Any]) -> str:
    value = sample.get("resume_key")
    if value:
        return str(value)
    sample_id = str(sample.get("sample_id") or "")
    generation_index = sample.get("generation_index")
    if generation_index is None:
        match = re.match(rf"^{re.escape(EXPERIMENT)}_(\d{{4}})_", sample_id)
        if match:
            generation_index = int(match.group(1))
    source_id = sample.get("source_id")
    qtype = sample.get("qa", {}).get("question_type") if isinstance(sample.get("qa"), dict) else ""
    skill = sample.get("attack_plan", {}).get("attack_skill") if isinstance(sample.get("attack_plan"), dict) else ""
    if generation_index is not None and source_id:
        return f"{generation_index}|{source_id}|{qtype}|{skill}"
    if generation_index is not None:
        return f"index:{generation_index}"
    return sample_id


def plan_resume_key(index: int, bundle: Any, qtype: str, skill: str) -> str:
    source_id = str(bundle.row.get("sample_id") or bundle.row.get("record_id") or index)
    return f"{index}|{source_id}|{qtype}|{skill}"


def sample_generation_index(sample: dict[str, Any]) -> int | None:
    value = sample.get("generation_index")
    if isinstance(value, int):
        return value
    try:
        if value is not None:
            return int(value)
    except (TypeError, ValueError):
        pass
    match = re.match(rf"^{re.escape(EXPERIMENT)}_(\d{{4}})_", str(sample.get("sample_id") or ""))
    if match:
        return int(match.group(1))
    return None


def read_partial_samples(path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.no_resume or not path.exists():
        return []
    samples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_keys: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON in partial checkpoint {path}:{line_no}: {exc}") from exc
            if not isinstance(sample, dict):
                continue
            if sample.get("challenger_api_used") is not True:
                continue
            if sample.get("generation_config", {}).get("seed") not in (None, args.seed):
                continue
            sample_id = str(sample.get("sample_id") or "")
            if not sample_id.startswith(f"{EXPERIMENT}_"):
                continue
            key = sample_resume_key(sample)
            if sample_id in seen_ids or key in seen_keys:
                continue
            seen_ids.add(sample_id)
            seen_keys.add(key)
            samples.append(sample)
    return samples


def summarize_partial_checkpoint(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(path),
        "count": 0,
        "max_generation_index": -1,
        "generation_config_num_samples": [],
    }
    if not path.exists():
        return summary
    num_samples_values: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(sample, dict):
                continue
            if sample.get("challenger_api_used") is not True:
                continue
            if sample.get("generation_config", {}).get("seed") not in (None, args.seed):
                continue
            sample_id = str(sample.get("sample_id") or "")
            if not sample_id.startswith(f"{EXPERIMENT}_"):
                continue
            summary["count"] += 1
            index = sample_generation_index(sample)
            if index is not None:
                summary["max_generation_index"] = max(int(summary["max_generation_index"]), index)
            configured_num_samples = sample.get("generation_config", {}).get("num_samples")
            if isinstance(configured_num_samples, int):
                num_samples_values.add(configured_num_samples)
    summary["generation_config_num_samples"] = sorted(num_samples_values)
    return summary


def resolve_partial_samples_path(args: argparse.Namespace, output_dir: Path) -> Path:
    partial_path = Path(args.partial_samples) if args.partial_samples else output_dir / "samples.partial.jsonl"
    if args.no_resume or args.partial_samples or partial_path.exists():
        return partial_path
    outputs_root = ROOT / "outputs"
    if not outputs_root.exists():
        return partial_path
    candidates: list[tuple[int, float, Path, dict[str, Any]]] = []
    for candidate in outputs_root.glob("*/samples.partial.jsonl"):
        try:
            if candidate.resolve() == partial_path.resolve():
                continue
        except FileNotFoundError:
            pass
        summary = summarize_partial_checkpoint(candidate, args)
        if int(summary.get("count") or 0) > 0:
            candidates.append((int(summary["count"]), candidate.stat().st_mtime, candidate, summary))
    if not candidates:
        return partial_path
    count, _mtime, source_path, summary = sorted(candidates, key=lambda item: (item[0], item[1], str(item[2])), reverse=True)[0]
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, partial_path)
    if not args.quiet:
        print(
            f"[resume] copied discovered checkpoint with {count} samples from {source_path} to {partial_path}",
            flush=True,
        )
        print(f"[resume] discovered checkpoint summary: {json.dumps(summary, ensure_ascii=False)}", flush=True)
    return partial_path


def append_partial_sample(path: Path, sample: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_failed_api_calls(path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.no_resume or not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            if record.get("experiment") not in (None, EXPERIMENT):
                continue
            records.append(record)
    return records


def api_failure_summary(records: list[dict[str, Any]]) -> dict[str, int]:
    error_counts = Counter(str(record.get("error_type") or "unknown") for record in records)
    return {
        "api_failure_count": len(records),
        "api_missing_keys_count": int(error_counts.get("missing_required_keys", 0)),
        "api_invalid_json_count": int(error_counts.get("invalid_json", 0)),
        "api_timeout_count": int(error_counts.get("timeout", 0)),
        "skipped_seed_count": len(records),
    }


def build_failed_api_call_record(
    *,
    index: int,
    resume_key: str,
    bundle: Any,
    qtype: str,
    skill: str,
    api_meta: dict[str, Any],
) -> dict[str, Any]:
    source_id = str(bundle.row.get("sample_id") or bundle.row.get("record_id") or index)
    return {
        "experiment": EXPERIMENT,
        "generation_index": index,
        "resume_key": resume_key,
        "seed_id": source_id,
        "qtype": qtype,
        "attack_skill": skill,
        "error_type": str(api_meta.get("error_type") or "challenger_api_failed"),
        "missing_keys": list(api_meta.get("missing_keys") or []),
        "raw_response_preview": raw_response_preview(api_meta.get("raw_response_preview") or ""),
        "attempts": int(api_meta.get("attempts") or 0),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def infer_planning_num_samples(args: argparse.Namespace, resumed_samples: list[dict[str, Any]]) -> int:
    planning_num_samples = args.num_samples
    for sample in resumed_samples:
        configured_num_samples = sample.get("generation_config", {}).get("num_samples")
        if isinstance(configured_num_samples, int):
            planning_num_samples = max(planning_num_samples, configured_num_samples)
    return planning_num_samples


def build_shuffled_bundles(rows: list[dict[str, Any]], rng: random.Random) -> list[Any]:
    bundles = [bundle for row in rows if (bundle := base.build_source_bundle(row))]
    if not bundles:
        raise ValueError("No seed rows contained enough dialogue turns and graph/event-chain nodes.")
    rng.shuffle(bundles)
    return bundles


def bundle_at_index(bundles: list[Any], index: int) -> Any:
    return bundles[index % len(bundles)]


def label_at_index(primary_labels: list[str], extra_labels: list[str], index: int) -> str:
    if index < len(primary_labels):
        return primary_labels[index]
    extra_index = index - len(primary_labels)
    if extra_index >= len(extra_labels):
        raise RuntimeError(f"Generation plan exhausted before reaching requested sample count at index {index}.")
    return extra_labels[extra_index]


def compact_scaffold_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "attack_plan": copy.deepcopy(sample.get("attack_plan")),
        "graph_update": copy.deepcopy(sample.get("graph_update")),
        "dialogue": copy.deepcopy(sample.get("dialogue")),
        "qa": copy.deepcopy(sample.get("qa")),
    }


def build_rewrite_prompt(
    bundle: Any,
    scaffold: dict[str, Any],
    index: int,
    qtype: str,
    skill: str,
    seed_text: str,
) -> dict[str, Any]:
    source_turns = [
        {
            "turn_id": turn.get("turn_id"),
            "speaker": turn.get("speaker"),
            "text": turn.get("text"),
        }
        for turn in bundle.turns
    ]
    source_nodes = [
        {
            "role": node.get("role"),
            "label": node.get("label"),
            "value": node.get("value"),
            "turn_ids": base.node_turn_ids(node) if hasattr(base, "node_turn_ids") else [],
        }
        for node in bundle.nodes
    ]
    return {
        "task": "Rewrite this validated challenger scaffold into an API-authored sample.",
        "sample_index": index,
        "seed_text": seed_text,
        "requested_attack_skill": skill,
        "requested_question_type": qtype,
        "constraints": [
            "return JSON only",
            "return exactly the top-level keys attack_plan, graph_update, dialogue, qa",
            "keep attack_skill exactly as requested",
            "keep question_type exactly as requested",
            "keep dialogue turn_ids, qa option ids, evidence_turn_ids, and gold labels unchanged",
            "you may rewrite attack_plan wording, graph_update wording, dialogue texts, question wording, and option texts",
            "do not mention dev, heldout, or sample6",
            "use only facts supported by the provided source turns and source nodes",
        ],
        "seed_source": extract_seed_source(bundle.row),
        "source_turns": source_turns,
        "source_nodes": source_nodes,
        "scaffold": compact_scaffold_sample(scaffold),
        "output_schema": {
            "attack_plan": "object",
            "graph_update": "object",
            "dialogue": {"turns": [{"turn_id": "string", "speaker": "string", "text": "string"}]},
            "qa": "question object with unchanged ids/gold",
        },
    }


def merge_api_sample(api_payload: dict[str, Any], scaffold: dict[str, Any], qtype: str, skill: str) -> dict[str, Any]:
    merged = copy.deepcopy(scaffold)
    if isinstance(api_payload.get("attack_plan"), dict):
        merged_attack = copy.deepcopy(scaffold.get("attack_plan") or {})
        merged_attack.update(api_payload["attack_plan"])
        merged_attack["attack_skill"] = skill
        merged["attack_plan"] = merged_attack
    if isinstance(api_payload.get("graph_update"), dict):
        merged_graph = copy.deepcopy(scaffold.get("graph_update") or {})
        if api_payload["graph_update"].get("events") or api_payload["graph_update"].get("state_changes"):
            merged_graph.update(api_payload["graph_update"])
        merged["graph_update"] = merged_graph
    if isinstance(api_payload.get("dialogue"), dict):
        api_turns = api_payload["dialogue"].get("turns")
        scaffold_turns = list((scaffold.get("dialogue") or {}).get("turns") or [])
        if isinstance(api_turns, list) and scaffold_turns:
            merged_turns = []
            for idx, scaffold_turn in enumerate(scaffold_turns):
                turn = copy.deepcopy(scaffold_turn)
                api_turn = api_turns[idx] if idx < len(api_turns) and isinstance(api_turns[idx], dict) else {}
                if api_turn.get("speaker"):
                    turn["speaker"] = api_turn["speaker"]
                if api_turn.get("text"):
                    turn["text"] = api_turn["text"]
                turn["turn_id"] = scaffold_turn.get("turn_id")
                merged_turns.append(turn)
            merged["dialogue"] = {"turns": merged_turns}
    if isinstance(api_payload.get("qa"), dict):
        merged_qa = copy.deepcopy(scaffold.get("qa") or {})
        merged_qa.update({k: v for k, v in api_payload["qa"].items() if k != "options"})
        merged_qa["question_type"] = qtype
        if "question" in api_payload["qa"] and api_payload["qa"]["question"]:
            merged_qa["question"] = api_payload["qa"]["question"]
        scaffold_options = list((scaffold.get("qa") or {}).get("options") or [])
        api_options = api_payload["qa"].get("options")
        if isinstance(api_options, list) and scaffold_options:
            merged_options = []
            for idx, scaffold_option in enumerate(scaffold_options):
                option = copy.deepcopy(scaffold_option)
                api_option = api_options[idx] if idx < len(api_options) and isinstance(api_options[idx], dict) else {}
                if api_option.get("text"):
                    option["text"] = api_option["text"]
                merged_options.append(option)
            merged_qa["options"] = merged_options
        merged["qa"] = merged_qa
    return merged


def make_api_sample(bundle: Any, index: int, qtype: str, skill: str, rng: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    qa, graph_update, turns, evidence_turn_ids = base.build_qa_and_graph(bundle, qtype, skill, rng)
    scaffold = {
        "attack_plan": base.attack_plan(skill),
        "graph_update": graph_update,
        "dialogue": base.make_dialogue_payload(turns),
        "qa": qa,
    }
    seed_source = extract_seed_source(bundle.row)
    source_id = str(bundle.row.get("sample_id") or bundle.row.get("record_id") or index)
    sample_id = f"{EXPERIMENT}_{index:04d}_{base.stable_id(source_id + qtype + skill + str(index), 10)}"
    seed = base.seed_identity(bundle.row)
    seed["evidence_window_turn_ids"] = sorted(set(evidence_turn_ids), key=base.turn_sort_value)
    prompt = build_rewrite_prompt(bundle, scaffold, index, qtype, skill, seed_text=sample_id)
    api_payload, api_meta = call_challenger_api(ACTIVE_ARGS or argparse.Namespace(), prompt)
    if api_payload is None:
        raise ChallengerApiFailure(api_meta)
    api_payload = require_challenger_payload(api_payload, api_meta)
    merged = merge_api_sample(api_payload, scaffold, qtype, skill)
    sample = {
        "sample_id": sample_id,
        "generation_index": index,
        "resume_key": plan_resume_key(index, bundle, qtype, skill),
        "seed": seed,
        "seed_source": seed_source,
        "frontier_from_split": seed_source["split"],
        "frontier_from_sample_name": seed_source["sample_name"],
        "attack_plan": merged["attack_plan"],
        "graph_update": merged["graph_update"],
        "dialogue": merged["dialogue"],
        "qa": merged["qa"],
        "challenger_api_used": True,
        "challenger_api_meta": api_meta,
        "challenger_payload_keys": sorted(api_payload.keys()),
        "generation_source": "api_rewrite",
        "generation_config": {
            "seed": (ACTIVE_ARGS.seed if ACTIVE_ARGS is not None else None),
            "num_samples": (ACTIVE_ARGS.num_samples if ACTIVE_ARGS is not None else None),
            "seed_scope": SEED_SCOPE,
        },
        "validation": {
            "graph_dialogue_consistent": False,
            "gold_derivable": False,
            "option_valid": False,
            "anti_shortcut_pass": False,
            "warnings": [],
        },
        "solver_eval": {
            "k": 5,
            "predictions": [],
            "exact_correct_count": 0,
            "soft_accuracy": 0.0,
            "partial_credit": 0.0,
            "bucket": "0/5",
        },
        "frontier_label": {
            "is_frontier": False,
            "category": "",
            "reason": "",
        },
    }
    sample["validation"] = base.validate_sample(sample)
    return sample, scaffold, api_meta


def build_sample(bundle: Any, index: int, qtype: str, skill: str, rng: Any) -> dict[str, Any]:
    sample, scaffold, api_meta = make_api_sample(bundle, index, qtype, skill, rng)
    source_id = str(bundle.row.get("sample_id") or bundle.row.get("record_id") or index)
    sample["source_id"] = source_id
    sample.setdefault("challenger_api_meta", api_meta)
    return sample


def generate_samples(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = random.Random(args.seed)
    bundles = build_shuffled_bundles(rows, rng)
    output_dir = Path(args.output_dir)
    partial_path = resolve_partial_samples_path(args, output_dir)
    failed_api_calls_path = Path(getattr(args, "failed_api_calls", None) or output_dir / "failed_api_calls.jsonl")
    resumed_samples = read_partial_samples(partial_path, args)
    failed_api_calls = read_failed_api_calls(failed_api_calls_path, args)
    args.api_failure_records = failed_api_calls
    args.failed_api_calls_path = str(failed_api_calls_path)
    if len(failed_api_calls) > args.max_api_failures:
        raise ApiFailureLimitExceeded(
            f"failed_api_calls count {len(failed_api_calls)} exceeds max_api_failures={args.max_api_failures}"
        )
    resumed_by_key = {sample_resume_key(sample): sample for sample in resumed_samples}
    resumed_by_index = {
        index: sample
        for sample in resumed_samples
        if (index := sample_generation_index(sample)) is not None
    }
    failed_by_key = {str(record.get("resume_key")): record for record in failed_api_calls if record.get("resume_key")}
    failed_by_index: dict[int, dict[str, Any]] = {}
    for record in failed_api_calls:
        try:
            failed_by_index[int(record.get("generation_index"))] = record
        except (TypeError, ValueError):
            continue
    planning_num_samples = infer_planning_num_samples(args, resumed_samples)
    attack_plan_labels = base.expand_plan(base.ATTACK_SKILL_QUOTAS, planning_num_samples)
    qtype_labels = base.expand_plan(base.QUESTION_TYPE_QUOTAS, planning_num_samples)
    rng.shuffle(attack_plan_labels)
    rng.shuffle(qtype_labels)
    extra_plan_count = max(8, args.max_api_failures + len(failed_api_calls) + 5)
    extra_attack_plan_labels = base.expand_plan(base.ATTACK_SKILL_QUOTAS, extra_plan_count)
    extra_qtype_labels = base.expand_plan(base.QUESTION_TYPE_QUOTAS, extra_plan_count)
    rng.shuffle(extra_attack_plan_labels)
    rng.shuffle(extra_qtype_labels)
    generation_index_limit = planning_num_samples + extra_plan_count
    samples: list[dict[str, Any]] = []
    used_sample_ids: set[str] = set()
    if resumed_samples and not args.quiet:
        print(f"[resume] loaded {len(resumed_samples)} successful samples from {partial_path}", flush=True)
    if failed_api_calls and not args.quiet:
        print(f"[resume] loaded {len(failed_api_calls)} failed API calls from {failed_api_calls_path}", flush=True)
    idx = 0
    while len(samples) < args.num_samples:
        if idx >= generation_index_limit:
            raise RuntimeError(
                f"Generation plan exhausted at index {idx}; successes={len(samples)}, requested={args.num_samples}."
            )
        bundle = bundle_at_index(bundles, idx)
        qtype = label_at_index(qtype_labels, extra_qtype_labels, idx)
        skill = label_at_index(attack_plan_labels, extra_attack_plan_labels, idx)
        resume_key = plan_resume_key(idx, bundle, qtype, skill)
        if resume_key in resumed_by_key or idx in resumed_by_index:
            sample = resumed_by_key.get(resume_key) or resumed_by_index[idx]
            samples.append(sample)
            used_sample_ids.add(str(sample.get("sample_id")))
            if not args.quiet:
                print(f"[resume] reused {len(samples)}/{args.num_samples}: {sample.get('sample_id')}", flush=True)
            idx += 1
            continue
        if resume_key in failed_by_key or idx in failed_by_index:
            failed_record = failed_by_key.get(resume_key) or failed_by_index[idx]
            if not args.quiet:
                print(
                    f"[api-skip] skipped failed seed index={idx} error_type={failed_record.get('error_type')}",
                    flush=True,
                )
            idx += 1
            continue
        seed_source = extract_seed_source(bundle.row)
        if seed_source.get("allowed_train_seed") is not True:
            raise RuntimeError(f"Forbidden seed reached generation: {seed_source}")
        sample_rng = random.Random(args.seed + idx * 7919)
        try:
            sample = build_sample(bundle, idx, qtype, skill, sample_rng)
        except ChallengerApiFailure as exc:
            failed_record = build_failed_api_call_record(
                index=idx,
                resume_key=resume_key,
                bundle=bundle,
                qtype=qtype,
                skill=skill,
                api_meta=exc.meta,
            )
            append_jsonl_record(failed_api_calls_path, failed_record)
            failed_api_calls.append(failed_record)
            args.api_failure_records = failed_api_calls
            failed_by_key[resume_key] = failed_record
            failed_by_index[idx] = failed_record
            if not args.quiet:
                missing = ",".join(failed_record.get("missing_keys") or [])
                suffix = f" missing_keys={missing}" if missing else ""
                print(
                    f"[api-skip] index={idx} resume_key={resume_key} error_type={failed_record['error_type']}{suffix}",
                    flush=True,
                )
            if len(failed_api_calls) > args.max_api_failures:
                raise ApiFailureLimitExceeded(
                    f"failed_api_calls count {len(failed_api_calls)} exceeds max_api_failures={args.max_api_failures}"
                )
            idx += 1
            continue
        if sample.get("challenger_api_used") is not True:
            raise RuntimeError(f"Challenger API was not used for sample {sample.get('sample_id')}: {sample.get('challenger_api_meta')}")
        sample_id = str(sample.get("sample_id"))
        if sample_id in used_sample_ids:
            raise RuntimeError(f"Duplicate sample_id during generation: {sample_id}")
        append_partial_sample(partial_path, sample)
        samples.append(sample)
        used_sample_ids.add(sample_id)
        if not args.quiet:
            print(f"generated {len(samples)}/{args.num_samples} via challenger API", flush=True)
        idx += 1
    args.api_failure_records = failed_api_calls
    args.skipped_seed_count = len(failed_api_calls)
    return samples


def assign_initial_frontier_labels(samples: list[dict[str, Any]]) -> None:
    for sample in samples:
        validation = sample.get("validation", {})
        soft = float(sample["solver_eval"].get("soft_accuracy") or 0.0)
        partial = float(sample["solver_eval"].get("partial_credit") or 0.0)
        seed_source = sample.get("seed_source", {})
        if seed_source.get("allowed_train_seed") is not True:
            category, reason = "forbidden_seed", "seed source was not train-only sample3-5"
        elif validation.get("option_valid") is not True:
            category, reason = "format_issue", "option validation failed"
        elif validation.get("gold_derivable") is not True:
            category, reason = "gold_issue", "gold was not derivable from graph/dialogue"
        elif validation.get("graph_dialogue_consistent") is not True:
            category, reason = "ambiguous_issue", "graph/dialogue evidence mismatch"
        elif validation.get("anti_shortcut_pass") is not True:
            category, reason = "shortcut_issue", "anti-shortcut validation failed"
        elif not base.is_clean(sample):
            category, reason = "ambiguous_issue", "validation warnings present"
        elif 0.2 <= soft <= 0.6 and partial >= 0.4:
            category, reason = "clean_near_miss", "soft accuracy in frontier band with sufficient partial credit"
        elif soft == 0.8 and partial >= 0.4 and base.rare_mechanism(sample, samples):
            category, reason = "hard_correct", "soft accuracy 0.8 with rare reasoning mechanism"
        elif soft == 0.0:
            if partial >= 0.4:
                category, reason = "clean_near_miss", "zero exact accuracy retained as a frontier candidate after clean audit"
            elif partial >= 0.2:
                category, reason = "true_too_hard", "zero exact accuracy with limited partial credit"
            else:
                category, reason = "bad_question", "zero exact accuracy and insufficient partial credit"
        elif soft > 0.6:
            category, reason = "easy", "solver soft accuracy above frontier band"
        else:
            category, reason = "true_too_hard", "below frontier band or insufficient partial credit"
        sample["frontier_label"] = {"is_frontier": False, "category": category, "reason": reason}


def candidate_priority(sample: dict[str, Any]) -> tuple[float, float, str]:
    soft = float(sample["solver_eval"].get("soft_accuracy") or 0.0)
    partial = float(sample["solver_eval"].get("partial_credit") or 0.0)
    zero_exact_penalty = 0.0 if soft > 0.0 else 1.0
    target_distance = abs(soft - 0.4)
    return (zero_exact_penalty, target_distance, -partial, sample["sample_id"])


def balanced_select_frontier(samples: list[dict[str, Any]], max_type_share: float, max_skill_share: float) -> list[dict[str, Any]]:
    candidates = [
        sample
        for sample in samples
        if sample["frontier_label"]["category"] in {"clean_near_miss", "hard_correct"}
        and sample.get("seed_source", {}).get("allowed_train_seed") is True
        and sample.get("frontier_from_split") == "train"
    ]
    candidates.sort(key=candidate_priority)
    selected = list(candidates)
    changed = True
    while changed and selected:
        changed = False
        type_counts = Counter(sample["qa"]["question_type"] for sample in selected)
        skill_counts = Counter(sample["attack_plan"]["attack_skill"] for sample in selected)
        n = len(selected)
        largest_type, type_count = type_counts.most_common(1)[0]
        largest_skill, skill_count = skill_counts.most_common(1)[0]
        remove_group: tuple[str, str] | None = None
        if type_count / n > max_type_share:
            remove_group = ("type", largest_type)
        elif skill_count / n > max_skill_share:
            remove_group = ("skill", largest_skill)
        if remove_group:
            group_kind, group_value = remove_group
            group_items = [
                sample
                for sample in selected
                if (sample["qa"]["question_type"] if group_kind == "type" else sample["attack_plan"]["attack_skill"]) == group_value
            ]
            group_items.sort(key=candidate_priority, reverse=True)
            selected.remove(group_items[0])
            changed = True
    selected_ids = {sample["sample_id"] for sample in selected}
    for sample in samples:
        if sample["sample_id"] in selected_ids:
            sample["frontier_label"]["is_frontier"] = True
        elif sample["frontier_label"]["category"] in {"clean_near_miss", "hard_correct"}:
            sample["frontier_label"]["reason"] += "; excluded by balanced frontier cap"
    return selected


def build_failure_matrix(samples: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = base.build_failure_matrix(samples)
    matrix["by_seed_split"] = base.count_map([sample.get("frontier_from_split", "unknown") for sample in samples])
    matrix["by_seed_sample_name"] = base.count_map([sample.get("frontier_from_sample_name", "unknown") for sample in samples])
    matrix["frontier_by_seed_sample_name"] = base.count_map(
        [sample.get("frontier_from_sample_name", "unknown") for sample in samples if sample["frontier_label"]["is_frontier"]]
    )
    return matrix


def largest_share(counter: dict[str, int]) -> float:
    return base.largest_share(counter)


def recommendation(report: dict[str, Any]) -> str:
    if report["generated_count"] < 400:
        return "dry_run_only_confirm_seed_audit_and_frontier_split"
    success = (
        report["clean_count"] >= 320
        and report["frontier_count"] >= 40
        and report["frontier_yield"] >= 0.10
        and report["near_miss_count"] > report["true_too_hard_count"]
        and report["largest_type_share"] <= 0.35
        and report["largest_skill_share"] <= 0.25
        and report["shortcut_warning_count"] <= max(5, report["generated_count"] // 50)
        and report["dev_or_heldout_frontier_count"] == 0
    )
    if success:
        return "proceed_to_gate400_quality_audit_before_r6_train_small"
    if 10 <= report["frontier_count"] <= 40:
        return "revise_challenger_prompt_or_attack_quota_and_rerun_gate400"
    return "do_not_train_r6_challenger_saturation_suspected"


def build_report(
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
    solver_model_path: str,
    adapter_path: str | None,
    eval_backend: str,
    seed_audit: dict[str, Any],
) -> dict[str, Any]:
    frontier = [sample for sample in samples if sample["frontier_label"]["is_frontier"]]
    soft_buckets = {f"{idx}/{args.k_eval}": 0 for idx in range(args.k_eval + 1)}
    for sample in samples:
        bucket = str(sample["solver_eval"].get("bucket") or f"0/{args.k_eval}")
        soft_buckets[bucket] = soft_buckets.get(bucket, 0) + 1
    frontier_by_question_type = base.count_map([sample["qa"]["question_type"] for sample in frontier])
    frontier_by_attack_skill = base.count_map([sample["attack_plan"]["attack_skill"] for sample in frontier])
    first_correct = sum(1 for sample in samples if sample["solver_eval"]["predictions"] and sample["solver_eval"]["predictions"][0]["exact_correct"])
    warning_samples = sum(1 for sample in samples if sample.get("validation", {}).get("warnings"))
    dev_or_heldout_frontier_count = sum(1 for sample in frontier if sample.get("frontier_from_split") != "train")
    api_failure_records = getattr(args, "api_failure_records", None)
    if api_failure_records is None:
        failed_path = Path(getattr(args, "failed_api_calls_path", Path(args.output_dir) / "failed_api_calls.jsonl"))
        api_failure_records = read_failed_api_calls(failed_path, args)
    api_stats = api_failure_summary(api_failure_records)
    report = {
        "experiment": EXPERIMENT,
        "fixed_solver": FIXED_SOLVER_NAME,
        "seed_scope": SEED_SCOPE,
        "solver_model_path": solver_model_path,
        "solver_adapter_path": adapter_path,
        "eval_backend": eval_backend,
        "max_length": args.max_length,
        "max_model_len": args.max_model_len,
        "generated_count": len(samples),
        "valid_count": sum(1 for sample in samples if base.base_valid(sample)),
        "clean_count": sum(1 for sample in samples if base.is_clean(sample)),
        "warning_count": warning_samples,
        "solver_exact_accuracy_k1": round(first_correct / len(samples), 6) if samples else 0.0,
        "solver_soft_accuracy_k5": round(
            sum(float(sample["solver_eval"].get("soft_accuracy") or 0.0) for sample in samples) / len(samples), 6
        )
        if samples
        else 0.0,
        "wrong_count_k1": len(samples) - first_correct,
        "soft_buckets": soft_buckets,
        "near_miss_count": sum(1 for sample in samples if sample["frontier_label"]["category"] == "clean_near_miss"),
        "hard_correct_count": sum(1 for sample in samples if sample["frontier_label"]["category"] == "hard_correct"),
        "true_too_hard_count": sum(1 for sample in samples if sample["frontier_label"]["category"] == "true_too_hard"),
        "bad_question_count": sum(
            1
            for sample in samples
            if sample["frontier_label"]["category"]
            in {"bad_question", "format_issue", "gold_issue", "ambiguous_issue", "shortcut_issue", "forbidden_seed"}
        ),
        "frontier_count": len(frontier),
        "frontier_yield": round(len(frontier) / len(samples), 6) if samples else 0.0,
        "frontier_by_question_type": frontier_by_question_type,
        "frontier_by_attack_skill": frontier_by_attack_skill,
        "largest_type_share": largest_share(frontier_by_question_type),
        "largest_skill_share": largest_share(frontier_by_attack_skill),
        "shortcut_warning_count": sum(1 for sample in samples if sample.get("validation", {}).get("anti_shortcut_pass") is not True),
        "dev_or_heldout_frontier_count": dev_or_heldout_frontier_count,
        "generated_by_question_type": base.count_map([sample["qa"]["question_type"] for sample in samples]),
        "generated_by_attack_skill": base.count_map([sample["attack_plan"]["attack_skill"] for sample in samples]),
        "frontier_by_seed_sample_name": base.count_map([sample.get("frontier_from_sample_name", "unknown") for sample in frontier]),
        "seed_source_audit_pass": bool(seed_audit.get("pass")),
        "challenger_config_path": args.config,
        "challenger_base_url": args.challenger_base_url,
        "challenger_model": args.challenger_model,
        "challenger_api_key_present": bool(args.challenger_api_key or os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("OPENAI_API_KEY")),
        "challenger_api_called_count": sum(1 for sample in samples if sample.get("challenger_api_used") is True),
        "api_failure_count": api_stats["api_failure_count"],
        "api_missing_keys_count": api_stats["api_missing_keys_count"],
        "api_invalid_json_count": api_stats["api_invalid_json_count"],
        "api_timeout_count": api_stats["api_timeout_count"],
        "skipped_seed_count": api_stats["skipped_seed_count"],
        "challenger_generation_sources": base.count_map([sample.get("generation_source", "unknown") for sample in samples]),
        "recommendation": "",
    }
    report["recommendation"] = recommendation(report)
    return report


def build_markdown_report(report: dict[str, Any], failure_matrix: dict[str, Any], seed_audit: dict[str, Any]) -> str:
    lines = [
        f"# {EXPERIMENT} Report",
        "",
        f"- fixed_solver: {report['fixed_solver']}",
        f"- seed_scope: {report['seed_scope']}",
        f"- seed_source_audit_pass: {report['seed_source_audit_pass']}",
        f"- challenger_model: {report.get('challenger_model')}",
        f"- challenger_base_url: {report.get('challenger_base_url')}",
        f"- challenger_api_called_count: {report.get('challenger_api_called_count')}",
        f"- api_failure_count: {report.get('api_failure_count')}",
        f"- api_missing_keys_count: {report.get('api_missing_keys_count')}",
        f"- api_invalid_json_count: {report.get('api_invalid_json_count')}",
        f"- api_timeout_count: {report.get('api_timeout_count')}",
        f"- skipped_seed_count: {report.get('skipped_seed_count')}",
        f"- eval_backend: {report.get('eval_backend')}",
        f"- generated_count: {report['generated_count']}",
        f"- valid_count: {report['valid_count']}",
        f"- clean_count: {report['clean_count']}",
        f"- frontier_count: {report['frontier_count']}",
        f"- frontier_yield: {report['frontier_yield']}",
        f"- solver_exact_accuracy_k1: {report['solver_exact_accuracy_k1']}",
        f"- solver_soft_accuracy_k5: {report['solver_soft_accuracy_k5']}",
        f"- largest_type_share: {report['largest_type_share']}",
        f"- largest_skill_share: {report['largest_skill_share']}",
        f"- dev_or_heldout_frontier_count: {report['dev_or_heldout_frontier_count']}",
        f"- recommendation: {report['recommendation']}",
        "",
        "## Seed Audit",
        "",
        f"- total_seed_candidates: {seed_audit['total_seed_candidates']}",
        f"- forbidden_seed_count: {seed_audit['forbidden_seed_count']}",
        f"- sample_name_counts: {json.dumps(seed_audit['sample_name_counts'], ensure_ascii=False)}",
        "",
        "## Soft Buckets",
        "",
    ]
    for bucket, count in report["soft_buckets"].items():
        lines.append(f"- {bucket}: {count}")
    lines.extend(["", "## Frontier By Question Type", ""])
    for key, count in report["frontier_by_question_type"].items():
        lines.append(f"- {key}: {count}")
    lines.extend(["", "## Frontier By Attack Skill", ""])
    for key, count in report["frontier_by_attack_skill"].items():
        lines.append(f"- {key}: {count}")
    lines.extend(["", "## Failure Categories", ""])
    for key, count in failure_matrix.get("by_frontier_category", {}).items():
        lines.append(f"- {key}: {count}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    global ACTIVE_ARGS
    args = parse_args()
    args = apply_config_defaults(args)
    ACTIVE_ARGS = args
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = base.load_seed_rows(args.seed_data)
    seed_audit, allowed_rows = build_seed_source_audit(rows, args.seed_data)
    base.write_json(output_dir / "seed_source_audit.json", seed_audit)
    if not seed_audit["pass"]:
        print(json.dumps(seed_audit, ensure_ascii=False, indent=2))
        print(f"[stop] Seed audit failed. See: {output_dir / 'seed_source_audit.json'}")
        return 2

    if args.limit_seed_rows is not None:
        allowed_rows = allowed_rows[: args.limit_seed_rows]

    base.build_sample = build_sample
    base.generate_samples = generate_samples
    solver_model_path, adapter_path = base.resolve_solver_checkpoint(args.solver_checkpoint)
    try:
        samples = base.generate_samples(allowed_rows, args)
    except ApiFailureLimitExceeded as exc:
        print(f"[stop] {exc}", flush=True)
        return 4
    if len(samples) < args.num_samples:
        partial_path = Path(args.partial_samples) if args.partial_samples else output_dir / "samples.partial.jsonl"
        print(f"[stop] Generated {len(samples)}/{args.num_samples}; partial checkpoint remains at {partial_path}")
        return 3

    base.write_jsonl(output_dir / "samples.jsonl", samples)
    valid_samples = [sample for sample in samples if base.base_valid(sample)]
    base.write_jsonl(output_dir / "valid_samples.jsonl", valid_samples)

    eval_rows, used_backend = base.run_solver_eval(samples, args, solver_model_path)
    assign_initial_frontier_labels(samples)
    frontier = balanced_select_frontier(
        samples,
        args.max_largest_question_type_share,
        args.max_largest_attack_skill_share,
    )
    failure_matrix = build_failure_matrix(samples)
    report = build_report(samples, args, solver_model_path, adapter_path, used_backend, seed_audit)

    base.write_jsonl(output_dir / "samples.jsonl", samples)
    base.write_jsonl(output_dir / "valid_samples.jsonl", [sample for sample in samples if base.base_valid(sample)])
    base.write_jsonl(output_dir / "solver_k5_eval.jsonl", eval_rows)
    base.write_jsonl(output_dir / "frontier.jsonl", frontier)
    base.write_json(output_dir / "failure_matrix.json", failure_matrix)
    base.write_json(output_dir / "report.json", report)
    (output_dir / "report.md").write_text(build_markdown_report(report, failure_matrix, seed_audit), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved {EXPERIMENT} outputs to: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
