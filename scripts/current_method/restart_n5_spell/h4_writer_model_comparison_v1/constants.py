"""Fixed scope: existing mixed TEMPORAL path, two roots and six original intents."""
from pathlib import Path
VERSION = "h4_writer_model_comparison_v1"
CODE = Path(__file__).resolve().parent
METHOD = CODE.parent
REPO = CODE.parents[3]
WORKSPACE = REPO.parent
RUNS = WORKSPACE / "restart-n5-runs"
DEFAULT_RUN = RUNS / (VERSION + "_live_1")
SOURCE_RUN = RUNS / "h4_r02_ordering_temporal_contract_v1_live_1"
CONFIG_DIR = REPO / "configs" / VERSION
STRONG_CONFIG = CONFIG_DIR / "strong_model.json"
QWEN_CONFIG = CONFIG_DIR / "qwen14b.json"
ROOTS = ("cc-train-episode-198480", "cc-train-episode-122711")
ARMS = ("QWEN14B_TEMPORAL", "STRONG_TEMPORAL")
SOURCE_ARM = "TEMPORAL_CONTRACT_ORDERING"
REQUESTED_COUNTS = {"single_choice": 2, "multi_select": 1, "ordering": 1}
PRIMARY_LIMIT = 12
REPAIR_LIMIT = 12
TOTAL_LIMIT = 24
SCHEDULE = []
for root_index, root in enumerate(ROOTS):
    for slot in range(1, 4):
        first = (root_index + slot - 1) % 2
        for arm in (ARMS[first], ARMS[1-first]):
            iid = f"intent_{slot}"
            rid = f"{root}-contract_guided-generator-{iid}"
            SCHEDULE.append(dict(order=len(SCHEDULE)+1, root_id=root, arm_id=arm,
                raw_slot=slot, raw_index=slot-1, intent_id=iid, slot_id=iid,
                request_id=rid, global_request_id=f"{arm}::{root}::{iid}::primary::{rid}"))

def arm_policy(arm):
    if arm not in ARMS:
        raise ValueError("UNKNOWN_ARM")
    return {"version": "ordering_temporal_contract_v1", "node_count": 4}
