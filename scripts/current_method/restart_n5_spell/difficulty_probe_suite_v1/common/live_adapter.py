from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    api = payload.get("api") if isinstance(payload, dict) else {}
    if not isinstance(api, dict) or not api.get("base_url") or not api.get("api_key") or not api.get("model"):
        raise ValueError("api_config_requires_base_url_api_key_model")
    return api


class LiveJSONClient:
    def __init__(self, config_path: Path, *, max_tokens: int = 1200, temperature: float = .25):
        api = load_config(config_path)
        self.base_url = str(api["base_url"]).rstrip("/")
        self.api_key = str(api["api_key"])
        self.model = str(api["model"])
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.calls = 0

    def complete(self, prompt: str, *, json_object: bool = True) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                                   "temperature": self.temperature, "max_tokens": self.max_tokens}
        if json_object:
            payload["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=180) as response:
            body = json.loads(response.read().decode("utf-8"))
        self.calls += 1
        choice = body.get("choices", [{}])[0]
        message = choice.get("message") or {}
        return str(message.get("content") or ""), {"model": self.model, "usage": body.get("usage"),
            "finish_reason": choice.get("finish_reason")}


def extract_json(raw: str) -> Any:
    text = str(raw or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min([pos for pos in (text.find("{"), text.find("[")) if pos >= 0], default=-1)
        end = max(text.rfind("}"), text.rfind("]"))
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise

