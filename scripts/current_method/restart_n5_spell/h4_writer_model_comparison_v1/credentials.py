"""Interactive local Key handoff; never used during preparation or model calls."""
from __future__ import annotations
import getpass
import os
import tempfile
from pathlib import Path


def configure_key(config_path):
    from .config import read
    api = read(config_path)["api"]
    name = api.get("api_key_env")
    destination = api.get("credential_file")
    if not name or not destination or api.get("credential_format") != "raw":
        raise ValueError("an environment-variable name and raw credential_file path must be configured")
    credential = os.environ.get(name) or getpass.getpass("请输入 TJU API Key（输入不显示）: ")
    if not credential or any(c.isspace() for c in credential):
        raise ValueError("API Key is empty or contains whitespace")
    target = Path(destination).expanduser()
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".h4-key-", dir=target.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            handle.write(credential + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"status": "KEY_STORED_LOCALLY", "path": str(target), "key_printed": False, "external_calls": 0}
