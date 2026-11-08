"""Exercise the typed appointment composer on one frozen R03 source."""

from __future__ import annotations

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
WORKTREE = Path("/root/autodl-tmp/gxy-selfplay/worktrees/h4-writer-coverage-v1")
sys.path.insert(0, str(WORKTREE))
from scripts.current_method.restart_n5_spell.h4_typed_appointment_binding_v1 import compose  # noqa: E402

SOURCE = Path("/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_memory_light_gate_siliconflow_qa_v1/prepared_run/frozen/qa_source.json")
OUT = HERE / "candidate_v3.json"


def main() -> None:
    ledger = json.loads((HERE / "reviewed_ledger.json").read_text(encoding="utf-8"))
    result = compose(ledger=ledger, frozen_source_path=SOURCE)
    if OUT.exists():
        if json.loads(OUT.read_text(encoding="utf-8")) != result:
            raise RuntimeError("candidate_drift")
    else:
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "candidate_id": result["candidate_id"],
                      "gold": result["gold"], "options": result["options"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
