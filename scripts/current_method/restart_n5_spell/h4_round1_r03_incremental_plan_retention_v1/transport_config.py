"""Explicit complete-URL contract and scoped legacy credential handoff."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit

VERSION = "incremental_connection_config_v1"


def chat_completion_endpoint(value):
    """Convert an explicit API base when building a NEW configuration only."""
    if not isinstance(value, str) or not value or value != value.strip() or any(c.isspace() for c in value):
        raise ValueError("invalid_endpoint")
    parsed = urlsplit(value)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment):
        raise ValueError("https_endpoint_without_credentials_query_fragment_required")
    path = parsed.path.rstrip("/")
    if not path.endswith("/chat/completions"):
        path += "/chat/completions"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def validate_complete_endpoint(value):
    if chat_completion_endpoint(value) != value:
        raise ValueError("complete_chat_completions_endpoint_required_no_implicit_runtime_rewrite")
    return value


def model_config_from_historical(execution_plan, *, tokenizer_path, tokenizer_sha256,
                                 api_key_env="SILICONFLOW_API_KEY"):
    config = {key: execution_plan[key] for key in ("model", "temperature", "timeout_seconds")}
    config.update(endpoint=chat_completion_endpoint(execution_plan["endpoint"]),
                  api_key_env=api_key_env, tokenizer_path=str(tokenizer_path),
                  tokenizer_json_sha256=tokenizer_sha256)
    return config


@contextmanager
def credential_environment(model_config, source=None):
    """No discovery/fallback; optional file must have a separately pinned SHA.

    Only the project API key enters the process environment. Neither the parsed
    credential object nor its value is returned, copied to artifacts or logged.
    """
    endpoint = validate_complete_endpoint(model_config["endpoint"])
    variable = model_config["api_key_env"]
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", variable) or variable in {"HOME", "CODEX_HOME", "PATH"}:
        raise ValueError("invalid_credential_environment_variable")
    previous = os.environ.get(variable)
    key = None
    if source is not None:
        if not isinstance(source, dict) or set(source) != {"path", "sha256"}:
            raise ValueError("pinned_credential_source_required")
        try:
            content = Path(source["path"]).read_bytes()
        except OSError:
            raise ValueError("credential_source_unreadable") from None
        if hashlib.sha256(content).hexdigest() != source["sha256"]:
            raise ValueError("credential_source_sha_mismatch")
        try:
            record = json.loads(content)
        except (ValueError, UnicodeError):
            raise ValueError("credential_source_invalid_json") from None
        api = record.get("api") if isinstance(record, dict) else None
        if not isinstance(api, dict) or api.get("model") != model_config["model"]:
            raise ValueError("credential_source_model_mismatch")
        if chat_completion_endpoint(api.get("base_url")) != endpoint:
            raise ValueError("credential_source_provider_endpoint_mismatch")
        key = api.get("api_key")
        if not isinstance(key, str) or not key.strip() or key != key.strip() or any(c in key for c in "\r\n"):
            raise ValueError("credential_source_key_missing_or_invalid")
        if previous and previous != key:
            raise ValueError("credential_environment_and_explicit_source_conflict")
    if not previous and key is None:
        raise ValueError("credential_environment_missing_explicit_source_required")
    injected = not previous and key is not None
    if injected:
        os.environ[variable] = key
    try:
        yield {"source": "explicit_pinned_project_config" if source is not None else "existing_environment",
               "environment_present": True, "temporary_injection": injected,
               "key_value_logged": False}
    finally:
        if injected:
            if previous is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = previous
