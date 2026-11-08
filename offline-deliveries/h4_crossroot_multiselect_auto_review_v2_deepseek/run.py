"""Independent SiliconFlow DeepSeek transport for the frozen multi-select audit."""

import argparse
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "h4_crossroot_multiselect_auto_review_v1/run.py"
spec = importlib.util.spec_from_file_location("multiselect_auto_review_v1", BASE)
assert spec and spec.loader
reviewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reviewer)
reviewer.HERE = HERE
reviewer.MODEL = "deepseek-ai/DeepSeek-V3.2"
reviewer.ENDPOINT = "https://api.siliconflow.cn/v1/chat/completions"


def config_key() -> str:
    config = json.loads(Path("/root/autodl-tmp/gxy-selfplay/config.json").read_text())
    api = config["api"]
    if api["base_url"].rstrip("/") != "https://api.siliconflow.cn/v1":
        raise ValueError("review_provider_identity_invalid")
    key = api["api_key"]
    if not isinstance(key, str) or not key or any(char.isspace() for char in key):
        raise ValueError("review_credential_invalid")
    return key


reviewer.config_key = config_key

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "execute", "review", "reparse", "policy_recheck", "policy_v3"))
    parser.add_argument("root", choices=reviewer.ROOTS)
    args = parser.parse_args()
    if args.action == "reparse":
        value = reviewer.review(args.root, output_name="REVIEW_RESULT_REPARSED.json")
    elif args.action == "policy_recheck":
        value = reviewer.review(args.root, output_name="REVIEW_RESULT_POLICY_V2.json")
    elif args.action == "policy_v3":
        value = reviewer.review(args.root, output_name="REVIEW_RESULT_POLICY_V3.json")
    else:
        value = getattr(reviewer, args.action)(args.root)
    print(json.dumps(value, ensure_ascii=False))
