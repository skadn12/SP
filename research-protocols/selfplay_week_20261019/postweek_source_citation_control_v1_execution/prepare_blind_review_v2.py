"""CPU-only ID normalization fix; preserves the executed and bound v1 code.

Source review packets contain both fully prefixed original references and local
extension references. Frozen prompts/schema apply first_source_ref to both;
blind-review source lookup must use that identical normalization.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / "prepare_blind_review.py"


def main() -> None:
    sys.dont_write_bytecode = True
    import json
    execution = json.loads((HERE / "EXECUTION_MANIFEST.json").read_text())
    original_binding = next(r for r in execution["bindings"] if Path(r["path"]).resolve() == ORIGINAL.resolve())
    if hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() != original_binding["sha256"]:
        raise RuntimeError("BOUND_REVIEW_BUILDER_CHANGED")
    spec = importlib.util.spec_from_file_location("bound_blind_review_builder_v1", ORIGINAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_read = module.read

    def normalized_read(path: Path):
        packet = original_read(path)
        if path.parent.name == "source_review_packets" and isinstance(packet, dict):
            prefix = packet["root_id"] + ":"
            for turn in packet["source_turns"]:
                if not turn["turn_ref"].startswith(prefix):
                    turn["turn_ref"] = prefix + turn["turn_ref"]
            refs = [t["turn_ref"] for t in packet["source_turns"]]
            if len(refs) != len(set(refs)):
                raise RuntimeError("DUPLICATE_SOURCE_REF_AFTER_NORMALIZATION")
        return packet

    module.read = normalized_read
    module.main()


if __name__ == "__main__":
    main()
