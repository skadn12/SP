"""Run the independent TJU strong API Solver against frozen Dev prompts.

This package deliberately does not import QA generation, semantic auditing, or
local model backends. It reads the already-frozen prompt/ledger artifacts from
the common Dev run, and writes only to a new isolated run directory.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..core import canonical_json, sha256_bytes, sha256_file
from ..h4_round0_b0_k3_difficulty_v1.contract import (
    position_variants,
    score_output,
)
from ..h4_round0_b0_k3_difficulty_v1.io_utils import stable_seed
from ..h4_round0_b0_k3_difficulty_v1 import (
    PARSER_SINGLE,
    PARSER_STRUCTURED,
    ROLLOUTS,
    REQUESTS_PER_CANDIDATE,
    SEED_SCHEDULE_VERSION,
    SINGLE_TEMPERATURE,
    STRUCTURED_TEMPERATURE,
    TOP_P,
    MAX_TOKENS,
)
from . import VERSION


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DEV_RUN = RUN_BASE / "h4_round0_common_dev_eval_v1_live_3"
DATA_RUN = RUN_BASE / "h4_round0_b0_matched_training_v1_live_2"
DIFFICULTY_RUN = RUN_BASE / "h4_round0_b0_k3_difficulty_v1_live_1"
DEFAULT_CONFIG = Path("/root/.config/gxy-selfplay/round1_strong_judge.env")
DEFAULT_RUN_ROOT = RUN_BASE / "h4_round0_strong_api_common_dev_eval_v1_live_1"
EXPECTED = {"candidates": 485, "roots": 15, "single_choice": 243,
            "multi_select": 123, "ordering": 119, "requests": 12378}
RETRY_BACKOFF = (5, 20)
MAX_ATTEMPTS = 3
TIMEOUT_SECONDS = 300
FORBIDDEN_CONFIG_PATH = Path("/root/autodl-tmp/gxy-selfplay/config.json")


class ContractError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_text(sha256_file(path) + "\n", encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2,
                                sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    write_text(path, "".join(json.dumps(dict(row), ensure_ascii=False,
                                           sort_keys=True) + "\n" for row in rows))


def verify_sidecar(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise ContractError(f"sidecar_missing:{path}")
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    observed = sha256_file(path)
    if expected != observed:
        raise ContractError(f"sidecar_mismatch:{path}")
    return observed


class RunLock:
    def __init__(self, run_root: Path):
        self.path = run_root / ".run.lock"
        self.handle = None

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.handle.close()
            self.handle = None
            if error.errno in {errno.EACCES, errno.EAGAIN}:
                raise ContractError("run_lock_busy") from error
            raise
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(f"pid={os.getpid()}\nstarted_at={now()}\n")
        self.handle.flush()
        return self

    def __exit__(self, *_: Any) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


def read_config(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ContractError(f"config_missing:{path}")
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        raise ContractError(f"config_not_private:{oct(mode)}")
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ContractError("config_invalid_line")
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    expected = {
        "provider": "TJU AI", "model": "tju-llm",
        "base_url": "https://ai.tju.edu.cn/api/v3",
        "api_key_env": "ROUND1_STRONG_JUDGE_API_KEY",
    }
    for key, value in expected.items():
        if values.get(key) != value:
            raise ContractError(f"config_identity_mismatch:{key}")
    key_name = values["api_key_env"]
    api_key = os.environ.get(key_name) or values.get(key_name, "")
    if not api_key or api_key in {"密码", "<YOUR_API_KEY>", "你的完整Key", "YOUR_API_KEY"}:
        raise ContractError(f"api_key_missing:{key_name}")
    values["_api_key"] = api_key
    values["_credential_source"] = "environment" if os.environ.get(key_name) else "private_env_file"
    return values


def identity(config: Mapping[str, str], config_path: Path) -> dict[str, Any]:
    return {
        "identity_version": VERSION + "_model_identity_v1",
        "provider": config["provider"], "model": config["model"],
        "base_url": config["base_url"],
        "endpoint": config["base_url"].rstrip("/") + "/chat/completions",
        "api_key_env": config["api_key_env"], "api_key_present": True,
        "credential_file": str(config_path.resolve()),
        "credential_source": config["_credential_source"],
        "credential_serialized": False,
        "forbidden_config": str(FORBIDDEN_CONFIG_PATH),
        "generation": {"thinking": False, "top_p": TOP_P, "max_tokens": MAX_TOKENS,
                        "temperatures": {"single_choice": SINGLE_TEMPERATURE,
                                         "multi_select": STRUCTURED_TEMPERATURE,
                                         "ordering": STRUCTURED_TEMPERATURE},
                        "max_attempts_total": MAX_ATTEMPTS,
                        "retry_backoff_seconds": list(RETRY_BACKOFF),
                        "timeout_seconds": TIMEOUT_SECONDS},
        "seed_schedule": SEED_SCHEDULE_VERSION,
        "original_expansion_model": "SiliconFlow/Qwen/Qwen3-14B",
    }


def solver_contract(source: Mapping[str, Any], contract_sha: str) -> dict[str, Any]:
    return {
        "contract_version": VERSION + "_solver_contract_v1",
        "source_dev_evaluation_contract": str(DEV_RUN / "dev_evaluation_contract.json"),
        "source_sha256": contract_sha,
        "verbatim_contract": dict(source),
        "model_role": "M_strong_api_solver",
        "input_policy": {
            "frozen_clean_prompt_text": True,
            "inclusive_prompt_sha_must_match": True,
            "gold_sent": False, "difficulty_sent": False, "evidence_sent": False,
            "old_answers_sent": False, "correctness_sent": False,
        },
        "retry_policy": {
            "network_rate_limit_server_only": True,
            "same_payload_seed": True, "max_attempts_total": MAX_ATTEMPTS,
            "backoff_seconds": list(RETRY_BACKOFF),
            "parse_failure_retry": False,
            "finish_reason_length": "technical_exclusion",
        },
    }


def _load_request_ledgers(scope: str) -> dict[str, list[dict[str, Any]]]:
    root = DEV_RUN / scope / "request_ledger"
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(root.glob("*/*.json")):
        value = load(path)
        if value.get("status") != "completed":
            raise ContractError(f"historical_ledger_not_completed:{scope}:{path}")
        result[str(value["request_contract"]["candidate_id"])].append(value)
    return result


def load_inputs() -> dict[str, Any]:
    manifest_path = DATA_RUN / "dev_full_eval_manifest.jsonl"
    contract_path = DEV_RUN / "dev_evaluation_contract.json"
    difficulty_path = DIFFICULTY_RUN / "difficulty_manifest.jsonl"
    for path in (manifest_path, contract_path, difficulty_path):
        verify_sidecar(path)
    manifest = load_jsonl(manifest_path)
    if len(manifest) != EXPECTED["candidates"]:
        raise ContractError(f"candidate_count:{len(manifest)}")
    candidate_ids = {str(row.get("candidate_id")) for row in manifest}
    if len(candidate_ids) != EXPECTED["candidates"]:
        raise ContractError("candidate_ids_not_unique")
    qcounts = Counter(str(row.get("question_type")) for row in manifest)
    if dict(qcounts) != {k: EXPECTED[k] for k in ("single_choice", "multi_select", "ordering")}:
        raise ContractError(f"question_type_counts:{dict(qcounts)}")
    roots = {str(row.get("root_id")) for row in manifest}
    if len(roots) != EXPECTED["roots"]:
        raise ContractError(f"root_count:{len(roots)}")
    if any(str(row.get("split")) != "dev" for row in manifest):
        raise ContractError("non_dev_candidate")
    source_contract = load(contract_path)
    if source_contract.get("test_accessed") is not False:
        raise ContractError("dev_contract_test_access")
    if int(source_contract.get("max_tokens")) != MAX_TOKENS or not source_contract.get("enable_thinking") is False:
        raise ContractError("dev_contract_generation_mismatch")
    if source_contract.get("seed_schedule", {}).get("version") != SEED_SCHEDULE_VERSION:
        raise ContractError("seed_schedule_mismatch")
    for qtype, count in (("single_choice", 4), ("multi_select", 5), ("ordering", 4)):
        if int(source_contract.get("position_counts", {}).get(qtype, -1)) != count:
            raise ContractError(f"position_count_mismatch:{qtype}")
    difficulty = load_jsonl(difficulty_path)
    difficulty_dev = {str(row["candidate_id"]): row for row in difficulty if row.get("split") == "dev"}
    if set(difficulty_dev) != candidate_ids:
        raise ContractError("difficulty_candidate_identity_mismatch")
    clean_ledgers = _load_request_ledgers("clean_dev")
    inclusive_ledgers = _load_request_ledgers("inclusive_dev")
    if set(clean_ledgers) != candidate_ids or set(inclusive_ledgers) != candidate_ids:
        raise ContractError("historical_ledger_candidate_identity_mismatch")
    prompt_records: list[dict[str, Any]] = []
    candidate_by_id = {str(row["candidate_id"]): row for row in manifest}
    for cid in sorted(candidate_ids):
        clean = clean_ledgers[cid]
        inclusive = inclusive_ledgers[cid]
        qtype = str(candidate_by_id[cid]["question_type"])
        expected_count = REQUESTS_PER_CANDIDATE[qtype]
        if len(clean) != expected_count or len(inclusive) != expected_count:
            raise ContractError(f"historical_request_count:{cid}")
        clean_by_key = {}
        inc_by_key = {}
        for ledger, target in ((clean, clean_by_key), (inclusive, inc_by_key)):
            for item in ledger:
                c = item["request_contract"]
                key = (str(c["variant_id"]), str(c["context_mode"]), int(c["slot_id"]))
                if key in target:
                    raise ContractError(f"historical_request_duplicate:{cid}:{key}")
                target[key] = item
        if set(clean_by_key) != set(inc_by_key):
            raise ContractError(f"inclusive_request_key_mismatch:{cid}")
        for key in sorted(clean_by_key):
            c = clean_by_key[key]["request_contract"]
            i = inc_by_key[key]["request_contract"]
            if c.get("prompt_sha256") != i.get("prompt_sha256"):
                raise ContractError(f"inclusive_prompt_sha_mismatch:{cid}:{key}")
            prompt_path = Path(str(clean_by_key[key].get("prompt_path") or ""))
            inclusive_prompt_path = Path(str(inc_by_key[key].get("prompt_path") or ""))
            if not prompt_path.is_file():
                raise ContractError(f"frozen_prompt_missing:{cid}:{key}")
            if not inclusive_prompt_path.is_file():
                raise ContractError(f"inclusive_frozen_prompt_missing:{cid}:{key}")
            prompt_sha = sha256_file(prompt_path)
            if prompt_sha != c.get("prompt_sha256"):
                raise ContractError(f"frozen_prompt_sha_mismatch:{cid}:{key}")
            if sha256_file(inclusive_prompt_path) != prompt_sha:
                raise ContractError(f"inclusive_prompt_file_sha_mismatch:{cid}:{key}")
            expected_seed = stable_seed(c["root_id"], c["candidate_id"], c["slot_id"])
            if int(c.get("seed")) != expected_seed:
                raise ContractError(f"historical_seed_mismatch:{cid}:{key}")
            expected_temp = SINGLE_TEMPERATURE if qtype == "single_choice" else STRUCTURED_TEMPERATURE
            generation_config = c.get("generation_config") or {}
            if float(generation_config.get("temperature", -1)) != expected_temp:
                raise ContractError(f"historical_temperature_mismatch:{cid}:{key}")
            prompt_records.append({
                "candidate_id": cid, "root_id": c["root_id"],
                "question_type": qtype, "variant_id": c["variant_id"],
                "context_mode": c["context_mode"], "slot_id": int(c["slot_id"]),
                "seed": expected_seed, "temperature": expected_temp,
                "prompt_path": str(prompt_path.resolve()),
                "prompt_sha256": prompt_sha,
                "request_cache_key": str(clean_by_key[key]["request_cache_key"]),
                "display_to_canonical": c.get("display_to_canonical"),
                "canonical_gold": c.get("canonical_gold"),
            })
    if len(prompt_records) != EXPECTED["requests"]:
        raise ContractError(f"planned_request_records:{len(prompt_records)}")
    return {
        "manifest": manifest, "candidate_by_id": candidate_by_id,
        "difficulty": difficulty_dev, "source_contract": source_contract,
        "source_contract_sha": sha256_file(contract_path),
        "prompt_records": prompt_records,
        "clean_ledgers": clean_ledgers, "inclusive_ledgers": inclusive_ledgers,
    }


def build_plan(inputs: Mapping[str, Any]) -> dict[str, Any]:
    records = list(inputs["prompt_records"])
    qcounts = Counter(str(row["question_type"]) for row in inputs["manifest"])
    request_by_type = {q: qcounts[q] * REQUESTS_PER_CANDIDATE[q]
                       for q in ("single_choice", "multi_select", "ordering")}
    return {
        "plan_version": VERSION + "_plan_v1", "split": "dev",
        "candidate_count": len(inputs["manifest"]),
        "root_count": len({str(row["root_id"]) for row in inputs["manifest"]}),
        "question_type_counts": dict(qcounts),
        "request_count_by_question_type": request_by_type,
        "planned_request_count": len(records),
        "requests_per_candidate": dict(REQUESTS_PER_CANDIDATE),
        "positions": {"single_choice": 4, "multi_select": 5, "ordering": 4},
        "rollouts_per_position": ROLLOUTS, "context_modes": ["no_context", "full_context"],
        "seed_schedule": SEED_SCHEDULE_VERSION,
        "temperature_by_question_type": {"single_choice": SINGLE_TEMPERATURE,
                                          "multi_select": STRUCTURED_TEMPERATURE,
                                          "ordering": STRUCTURED_TEMPERATURE},
        "max_tokens": MAX_TOKENS, "thinking": False,
        "source_prompt_policy": "clean_dev_frozen_prompt_text",
        "inclusive_prompt_sha_verified": True,
        "records": records,
    }


def prepare_run(run_root: Path, inputs: Mapping[str, Any], config: Mapping[str, str]) -> None:
    if run_root.exists():
        raise ContractError(f"run_root_already_exists:{run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    for name in ("strong_dev/raw_prompts", "strong_dev/raw_responses",
                 "strong_dev/request_ledger"):
        (run_root / name).mkdir(parents=True)
    write_json(run_root / "strong_api_model_identity.json", identity(config, DEFAULT_CONFIG))
    write_json(run_root / "strong_api_solver_contract.json",
               solver_contract(inputs["source_contract"], inputs["source_contract_sha"]))
    write_json(run_root / "strong_api_solver_plan.json", build_plan(inputs))


def _request_body(config: Mapping[str, str], prompt: str, record: Mapping[str, Any]) -> dict[str, Any]:
    # The frozen chat-template text is sent byte-for-byte as the user content.
    # This keeps the prompt SHA contract independent of provider-side templating.
    return {
        "model": config["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(record["temperature"]), "top_p": TOP_P,
        "max_tokens": MAX_TOKENS, "seed": int(record["seed"]),
        "stream": False,
    }


def _post(config: Mapping[str, str], body: bytes) -> tuple[int, bytes]:
    request = urllib.request.Request(
        config["base_url"].rstrip("/") + "/chat/completions", data=body,
        method="POST", headers={"Content-Type": "application/json",
                                 "Authorization": "Bearer " + config["_api_key"]})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as error:
        return int(error.code), error.read()


def _response_content(raw: bytes) -> tuple[str, dict[str, Any]]:
    try:
        envelope = json.loads(raw.decode("utf-8"))
    except Exception as error:
        raise ContractError(f"api_envelope_invalid:{error}") from error
    choices = envelope.get("choices") if isinstance(envelope, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ContractError("api_choices_missing")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise ContractError("api_content_missing")
    return content, {
        "response_id": envelope.get("id"), "response_model": envelope.get("model"),
        "finish_reason": choices[0].get("finish_reason"), "usage": envelope.get("usage"),
    }


def _candidate_context(inputs: Mapping[str, Any], candidate_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_row = inputs["candidate_by_id"][candidate_id]
    gate = manifest_row
    pool_path = Path(str(gate["candidate_pool_path"]))
    dialogue_path = Path(str(gate["dialogue_path"]))
    if not pool_path.is_file() or not dialogue_path.is_file():
        raise ContractError(f"candidate_source_missing:{candidate_id}")
    if sha256_file(pool_path) != gate["candidate_pool_sha256"] or sha256_file(dialogue_path) != gate["dialogue_sha256"]:
        raise ContractError(f"candidate_source_sha_drift:{candidate_id}")
    pool = {str(row["candidate_id"]): row for row in load(pool_path)}
    if candidate_id not in pool:
        raise ContractError(f"candidate_not_in_pool:{candidate_id}")
    return pool[candidate_id], load(dialogue_path)


def _smoke_records(inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    chosen: dict[str, str] = {}
    for row in sorted(inputs["manifest"], key=lambda x: str(x["candidate_id"])):
        qtype = str(row["question_type"])
        chosen.setdefault(qtype, str(row["candidate_id"]))
    return [row for row in inputs["prompt_records"] if row["candidate_id"] in set(chosen.values())]


def _run_record(config: Mapping[str, str], run_root: Path, record: Mapping[str, Any],
                candidate: Mapping[str, Any], variant: Mapping[str, Any],
                *, smoke: bool) -> dict[str, Any]:
    request_id = (f"{record['candidate_id']}__{record['variant_id']}__"
                  f"{record['context_mode']}__s{int(record['slot_id']):02d}")
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
    prompt = Path(str(record["prompt_path"])).read_text(encoding="utf-8")
    if sha256_bytes(prompt.encode("utf-8")) != record["prompt_sha256"]:
        raise ContractError(f"prompt_changed:{request_id}")
    body = _request_body(config, prompt, record)
    body_bytes = canonical_json(body).encode("utf-8")
    prompt_out = run_root / "strong_dev/raw_prompts" / f"{digest}.txt"
    if prompt_out.exists() and prompt_out.read_text(encoding="utf-8") != prompt:
        raise ContractError(f"output_prompt_drift:{request_id}")
    if not prompt_out.exists():
        write_text(prompt_out, prompt)
    response_paths: list[Path] = []
    ledger_path = run_root / "strong_dev/request_ledger" / f"{digest}.json"
    if ledger_path.exists():
        existing = load(ledger_path)
        if existing.get("status") == "completed":
            return existing["result"]
        raise ContractError(f"incomplete_ledger_requires_manual_review:{request_id}")
    attempts = []
    result = None
    terminal_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        started = now(); status = None; raw = b""; metadata: dict[str, Any] = {}
        parse_result = None; error_text = None
        try:
            status, raw = _post(config, body_bytes)
            if status != 200:
                raise ContractError(f"http_status:{status}")
            content, metadata = _response_content(raw)
            finish_reason = metadata.get("finish_reason")
            if finish_reason == "length":
                parse_result = {
                    "parse_status": "output_truncated", "scored": False,
                    "correct": None, "display_prediction": [],
                    "canonical_prediction": [], "failure_type": "solver_output_truncated",
                    "parser_version": PARSER_SINGLE if record["question_type"] == "single_choice" else PARSER_STRUCTURED,
                    "parse_reason": "finish_reason_length", "metrics": {},
                }
            else:
                parse_result = score_output(content, candidate, variant)
            result = {
                "request_id": request_id, "candidate_id": record["candidate_id"],
                "root_id": record["root_id"], "question_type": record["question_type"],
                "variant_id": record["variant_id"], "context_mode": record["context_mode"],
                "slot_id": record["slot_id"], "seed": record["seed"],
                "temperature": record["temperature"], "top_p": TOP_P,
                "max_tokens": MAX_TOKENS, "prompt_sha256": record["prompt_sha256"],
                "request_body_sha256": sha256_bytes(body_bytes),
                "parser_version": parse_result.get("parser_version"),
                "parse_status": parse_result.get("parse_status"),
                "display_prediction": parse_result.get("display_prediction"),
                "canonical_prediction": parse_result.get("canonical_prediction"),
                "canonical_gold": record.get("canonical_gold"),
                "scored": parse_result.get("scored"), "correct": parse_result.get("correct"),
                "failure_type": parse_result.get("failure_type"),
                "parse_reason": parse_result.get("parse_reason"),
                "metrics": parse_result.get("metrics") or {},
                "finish_reason": finish_reason, "http_status": status,
                "api_metadata": metadata,
                "attempt_count": attempt, "smoke": smoke,
            }
        except Exception as error:
            error_text = f"{type(error).__name__}:{error}"
            terminal_error = error_text
            result = None
        response_path = (run_root / "strong_dev/raw_responses" /
                         f"{digest}__attempt-{attempt:02d}.json")
        response_paths.append(response_path)
        response_envelope = {
            "request_id": request_id, "attempt": attempt,
            "http_status": status, "raw_response": raw.decode("utf-8", errors="replace"),
            "error": error_text, "api_metadata": metadata,
        }
        # Preserve every provider response, including errors, without auth data.
        write_json(response_path, response_envelope)
        attempts.append({"attempt": attempt, "started_at": started,
                         "finished_at": now(), "http_status": status,
                         "raw_response_sha256": sha256_bytes(raw),
                         "error": error_text,
                         "finish_reason": metadata.get("finish_reason")})
        if result is not None:
            break
        retryable = status is None or status == 429 or (status is not None and status >= 500)
        if not retryable or attempt >= MAX_ATTEMPTS:
            break
        time.sleep(RETRY_BACKOFF[attempt - 1])
    if result is None:
        result = {
            "request_id": request_id, "candidate_id": record["candidate_id"],
            "root_id": record["root_id"], "question_type": record["question_type"],
            "variant_id": record["variant_id"], "context_mode": record["context_mode"],
            "slot_id": record["slot_id"], "seed": record["seed"],
            "temperature": record["temperature"], "top_p": TOP_P,
            "max_tokens": MAX_TOKENS, "prompt_sha256": record["prompt_sha256"],
            "request_body_sha256": sha256_bytes(body_bytes),
            "parser_version": PARSER_SINGLE if record["question_type"] == "single_choice" else PARSER_STRUCTURED,
            "parse_status": "technical_failure", "display_prediction": [],
            "canonical_prediction": [], "canonical_gold": record.get("canonical_gold"),
            "scored": False, "correct": None, "failure_type": "technical_failure",
            "parse_reason": terminal_error, "metrics": {},
            "finish_reason": attempts[-1].get("finish_reason"),
            "http_status": attempts[-1].get("http_status"), "attempt_count": len(attempts),
            "smoke": smoke,
        }
    ledger = {
        "ledger_version": VERSION + "_request_ledger_v1", "status": "completed",
        "request_id": request_id, "candidate_id": record["candidate_id"],
        "root_id": record["root_id"], "question_type": record["question_type"],
        "variant_id": record["variant_id"], "context_mode": record["context_mode"],
        "slot_id": record["slot_id"], "seed": record["seed"],
        "temperature": record["temperature"], "prompt_sha256": record["prompt_sha256"],
        "request_body_sha256": sha256_bytes(body_bytes), "api_key_env": config["api_key_env"],
        "authorization_header_recorded": False, "attempts": attempts,
        "raw_prompt_path": str(prompt_out.resolve()), "raw_prompt_sha256": sha256_file(prompt_out),
        "raw_response_paths": [str(path.resolve()) for path in response_paths],
        "raw_response_sha256": [sha256_file(path) for path in response_paths],
        "result": result,
    }
    write_json(ledger_path, ledger)
    return result


def run_smoke(*, run_root: Path, config_path: Path, inputs: Mapping[str, Any]) -> dict[str, Any]:
    config = read_config(config_path)
    records = _smoke_records(inputs)
    by_candidate: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for record in records:
        by_candidate.setdefault(record["candidate_id"], _candidate_context(inputs, record["candidate_id"]))
    results = []
    for record in sorted(records, key=lambda x: (x["question_type"], x["candidate_id"], x["variant_id"], x["context_mode"], x["slot_id"])):
        candidate, _dialogue = by_candidate[record["candidate_id"]]
        variant = next(v for v in position_variants(candidate)
                       if str(v["variant_id"]) == str(record["variant_id"]))
        results.append(_run_record(config, run_root, record, candidate, variant, smoke=True))
    qcounts = Counter(row["question_type"] for row in results)
    technical = sum(row.get("failure_type") == "technical_failure" for row in results)
    parse_fail = sum(row.get("failure_type") in {"parse_failure", "solver_output_truncated"} for row in results)
    scored = sum(bool(row.get("scored")) for row in results)
    report = {
        "smoke_version": VERSION + "_smoke_v1", "created_at": now(),
        "status": "PASS" if len(results) == 78 and technical == 0 and parse_fail == 0 and scored == 78 else "BLOCKED",
        "planned_requests": 78, "completed_requests": len(results),
        "request_count_by_question_type": dict(qcounts),
        "scored_requests": scored, "technical_failure_count": technical,
        "parser_failure_or_length_count": parse_fail,
        "all_context_modes_and_positions": True, "all_k3_slots": True,
        "inclusive_prompt_sha_verified": True,
        "api_calls": sum(int(row.get("attempt_count", 0)) for row in results),
        "qa_generation_calls": 0, "semantic_auditor_calls": 0,
        "solver_local_model_calls": 0,
        "candidate_results": results,
    }
    write_json(run_root / "api_smoke_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "smoke"), required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    inputs = load_inputs()
    config = read_config(args.config)
    if args.phase == "preflight":
        prepare_run(args.run_root, inputs, config)
        report = {
            "preflight_version": VERSION + "_preflight_v1", "created_at": now(),
            "status": "PASS", "dev_candidates": EXPECTED["candidates"],
            "dev_roots": EXPECTED["roots"],
            "question_type_counts": {k: EXPECTED[k] for k in ("single_choice", "multi_select", "ordering")},
            "planned_requests": EXPECTED["requests"], "inclusive_prompt_sha_verified": True,
            "m0_reused": True, "clean_reused": True, "inclusive_reused": True,
            "test_accessed": False, "qa_generation_calls": 0,
            "semantic_auditor_calls": 0, "training_calls": 0,
            "adapter_modified": False, "api_calls": 0,
            "checks": {
                "dev_candidates_485": True, "dev_roots_15": True,
                "single_choice_243": True, "multi_select_123": True,
                "ordering_119": True, "planned_requests_12378": True,
                "prompt_sha_inclusive_equal": True, "b0_contract_frozen": True,
                "seed_schedule_frozen": True, "no_test_access": True,
                "no_old_model_rerun": True,
            },
            "run_root": str(args.run_root.resolve()),
        }
        write_json(args.run_root / "preflight_report.json", report)
    else:
        if not args.run_root.exists():
            prepare_run(args.run_root, inputs, config)
            preflight = {
                "preflight_version": VERSION + "_preflight_v1", "created_at": now(),
                "status": "PASS", "dev_candidates": EXPECTED["candidates"],
                "dev_roots": EXPECTED["roots"], "planned_requests": EXPECTED["requests"],
                "inclusive_prompt_sha_verified": True, "api_calls": 0,
            }
            write_json(args.run_root / "preflight_report.json", preflight)
        else:
            verify_sidecar(args.run_root / "preflight_report.json")
        with RunLock(args.run_root):
            report = run_smoke(run_root=args.run_root, config_path=args.config, inputs=inputs)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
