"""Small Memorax HTTP client with explicit transport and secret boundaries."""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import httpx


class MemoraxContractError(RuntimeError):
    """The service returned a response that violates the documented contract."""


class MemoraxTransportError(RuntimeError):
    """A sanitized network or HTTP failure."""


@dataclass(frozen=True)
class MemoraxConfig:
    base_url: str
    api_key: str = field(repr=False)
    platform: str = "memory"
    connect_timeout_seconds: float = 15.0
    read_timeout_seconds: float = 180.0
    allow_insecure_http: bool = False

    def __post_init__(self) -> None:
        parsed = urlparse(str(self.base_url).rstrip("/"))
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("memorax_base_url_invalid")
        if parsed.scheme == "http" and not self.allow_insecure_http:
            raise ValueError("memorax_plaintext_http_forbidden")
        if not str(self.api_key).strip():
            raise ValueError("memorax_api_key_required")
        if self.platform != "memory":
            raise ValueError("memorax_platform_must_be_memory")

    @property
    def normalized_base_url(self) -> str:
        return str(self.base_url).rstrip("/")

    def redacted(self) -> dict[str, Any]:
        return {
            "base_url": self.normalized_base_url,
            "api_key_present": bool(self.api_key),
            "api_key_recorded": False,
            "platform": self.platform,
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "read_timeout_seconds": self.read_timeout_seconds,
            "allow_insecure_http": self.allow_insecure_http,
        }


def _read_env_file(path: Path) -> dict[str, str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"memorax_credentials_missing:{path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise PermissionError("memorax_credentials_permissions_too_open")
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError("memorax_credentials_line_invalid")
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def load_memorax_config(
    *, credentials_path: Path | None = None,
    allow_insecure_http: bool = False,
) -> MemoraxConfig:
    values = dict(os.environ)
    if credentials_path is not None:
        values.update(_read_env_file(credentials_path))
    return MemoraxConfig(
        base_url=values.get("MEMORAX_BASE_URL", ""),
        api_key=values.get("MEMORAX_API_KEY", ""),
        allow_insecure_http=allow_insecure_http,
    )


class MemoraxClient:
    def __init__(
        self, config: MemoraxConfig, *, transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.config = config
        self.transport = transport
        self.calls: list[dict[str, Any]] = []

    def _post(
        self, path: str, *, payload: Mapping[str, Any], request_id: str,
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Token {self.config.api_key}",
            "X-Platform": self.config.platform,
            "Content-Type": "application/json",
            "X-Request-ID": str(request_id),
        }
        timeout = httpx.Timeout(
            connect=self.config.connect_timeout_seconds,
            read=self.config.read_timeout_seconds,
            write=self.config.read_timeout_seconds,
            pool=self.config.connect_timeout_seconds,
        )
        call_record = {
            "path": path,
            "request_id": str(request_id),
            "http_status": None,
            "request_attempted": True,
            "authorization_header_recorded": False,
        }
        self.calls.append(call_record)
        try:
            with httpx.Client(
                base_url=self.config.normalized_base_url,
                headers=headers,
                timeout=timeout,
                transport=self.transport,
            ) as client:
                response = client.post(path, json=dict(payload))
        except httpx.HTTPError as error:
            call_record["transport_error"] = type(error).__name__
            raise MemoraxTransportError(
                f"memorax_transport_failed:{path}:{type(error).__name__}") from error
        call_record["http_status"] = response.status_code
        try:
            value = response.json()
        except ValueError as error:
            raise MemoraxContractError(
                f"memorax_response_not_json:{path}:{response.status_code}") from error
        if response.status_code >= 400:
            detail = value.get("detail") if isinstance(value, Mapping) else None
            error_value = value.get("error") if isinstance(value, Mapping) else None
            code = error_value.get("code") if isinstance(error_value, Mapping) else None
            raise MemoraxTransportError(
                f"memorax_http_error:{path}:{response.status_code}:"
                f"{code or detail or 'unknown'}")
        if not isinstance(value, dict) or value.get("success") is not True:
            raise MemoraxContractError(f"memorax_success_envelope_invalid:{path}")
        return value

    def _get(self, path: str, *, request_id: str) -> dict[str, Any]:
        headers = {
            "Authorization": f"Token {self.config.api_key}",
            "X-Platform": self.config.platform,
            "Content-Type": "application/json",
            "X-Request-ID": str(request_id),
        }
        timeout = httpx.Timeout(
            connect=self.config.connect_timeout_seconds,
            read=self.config.read_timeout_seconds,
            write=self.config.read_timeout_seconds,
            pool=self.config.connect_timeout_seconds,
        )
        call_record = {
            "path": path,
            "request_id": str(request_id),
            "http_status": None,
            "request_attempted": True,
            "authorization_header_recorded": False,
        }
        self.calls.append(call_record)
        try:
            with httpx.Client(
                base_url=self.config.normalized_base_url,
                headers=headers,
                timeout=timeout,
                transport=self.transport,
            ) as client:
                response = client.get(path)
        except httpx.HTTPError as error:
            call_record["transport_error"] = type(error).__name__
            raise MemoraxTransportError(
                f"memorax_transport_failed:{path}:{type(error).__name__}") from error
        call_record["http_status"] = response.status_code
        try:
            value = response.json()
        except ValueError as error:
            raise MemoraxContractError(
                f"memorax_response_not_json:{path}:{response.status_code}") from error
        if response.status_code >= 400:
            detail = value.get("detail") if isinstance(value, Mapping) else None
            raise MemoraxTransportError(
                f"memorax_http_error:{path}:{response.status_code}:"
                f"{detail or 'unknown'}")
        if not isinstance(value, dict):
            raise MemoraxContractError(f"memorax_status_response_invalid:{path}")
        return value

    def add(self, payload: Mapping[str, Any], *, request_id: str) -> dict[str, Any]:
        value = self._post(
            "/v1/memories/add", payload=payload, request_id=request_id)
        data = value.get("data")
        if not isinstance(data, Mapping) or data.get("status") not in {
                "completed", "accepted"}:
            raise MemoraxContractError("memorax_add_status_invalid")
        return value

    def search_buckets(
        self, payload: Mapping[str, Any], *, request_id: str,
    ) -> dict[str, Any]:
        value = self._post(
            "/v1/memory-buckets/search", payload=payload,
            request_id=request_id)
        data = value.get("data")
        if not isinstance(data, Mapping) or not isinstance(
                data.get("buckets"), Mapping):
            raise MemoraxContractError("memorax_bucket_response_invalid")
        return value

    def add_status(self, task_id: str, *, request_id: str) -> dict[str, Any]:
        task = str(task_id or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", task):
            raise ValueError("memorax_task_id_invalid")
        value = self._get(
            f"/v1/memories/add/status/{task}", request_id=request_id)
        data = value.get("data")
        status = (
            data.get("status") if isinstance(data, Mapping) else
            value.get("status"))
        if status not in {"processing", "success", "failed", "completed"}:
            raise MemoraxContractError("memorax_add_status_invalid")
        return value
