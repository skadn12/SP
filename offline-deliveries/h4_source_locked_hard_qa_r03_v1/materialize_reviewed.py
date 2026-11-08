"""Materialize the reviewed record selection and verify the tested item matches."""

from __future__ import annotations

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
WORKTREE = Path("/root/autodl-tmp/gxy-selfplay/worktrees/h4-writer-coverage-v1")
sys.path.insert(0, str(WORKTREE))
from scripts.current_method.restart_n5_spell.h4_source_locked_multiselect_v1 import materialize  # noqa: E402

SOURCE = Path("/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_memory_light_gate_siliconflow_qa_v1/prepared_run/frozen/qa_source.json")
OUT = HERE / "materialized_candidate.json"


def main() -> None:
    bank = json.loads((HERE / "reviewed_bank.json").read_text(encoding="utf-8"))
    selection = json.loads((HERE / "selected_record_ids.json").read_text(encoding="utf-8"))
    result = materialize(bank=bank, selection=selection, frozen_source_path=SOURCE)
    tested = json.loads((HERE / "candidates.json").read_text(encoding="utf-8"))["candidates"][0]
    for key in ("candidate_id", "root_id", "question_type", "question", "options", "gold"):
        if result[key] != tested[key]:
            raise RuntimeError(f"materialized_item_differs_from_tested:{key}")
    if OUT.exists():
        if json.loads(OUT.read_text(encoding="utf-8")) != result:
            raise RuntimeError("materialized_candidate_drift")
    else:
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "candidate_id": result["candidate_id"],
                      "gold": result["gold"], "option_count": len(result["options"])}))


if __name__ == "__main__":
    main()
