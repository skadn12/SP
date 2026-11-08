"""Request a model-authored correction in a separate immutable attempt folder."""

import argparse
import json
from pathlib import Path

import generate as engine


PARENT = Path(__file__).resolve().parent
HERE = PARENT / "repair_01"
engine.HERE = HERE
engine.VERSION = PARENT.name + "_repair_01"


def dependencies() -> dict:
    names = ["generate.py", "repair.py", "prompt.txt", "response.raw.txt", "RESPONSE_METADATA.json"]
    return {name: engine.sha((PARENT / name).read_bytes()) for name in names}


def prepare() -> None:
    engine.source()
    instructions = (HERE / "instructions.txt").read_bytes()
    prompt = ((PARENT / "prompt.txt").read_text(encoding="utf-8")
              + "\nPrevious model-generated draft (unmodified):\n"
              + (PARENT / "response.raw.txt").read_text(encoding="utf-8")
              + "\nCorrection instructions:\n" + instructions.decode("utf-8"))
    payload = engine.build_payload(engine.profile(), prompt)
    engine.write_new("PREPARED.json", {
        "status": "PREPARED_MODEL_AUTHORED_CORRECTION", "version": engine.VERSION,
        "root_id": engine.ROOT_ID, "source_path": str(engine.SOURCE),
        "source_sha256": engine.SOURCE_SHA, "source_turns": 188,
        "instructions_sha256": engine.sha(instructions),
        "script_sha256": engine.sha((PARENT / "generate.py").read_bytes()),
        "prompt_sha256": engine.sha(prompt.encode("utf-8")),
        "payload_sha256": engine.sha(engine.canonical(payload)),
        "dependencies_sha256": dependencies(), "requested_model": engine.MODEL,
        "profile": engine.profile(), "target_candidates": 6,
        "maximum_http_attempts": 1, "solver_calls": 0, "training_started": False,
    })
    engine.write_new("prompt.txt", prompt, raw=True)
    engine.write_new("request.json", payload)
    print(json.dumps({"status": "REPAIR_PREPARED", "model": engine.MODEL, "http_calls": 0}))


def execute() -> None:
    frozen = engine.read("PREPARED.json")
    if dependencies() != frozen["dependencies_sha256"]:
        raise ValueError("repair_dependencies_changed")
    engine.execute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    execute() if args.execute else prepare()
