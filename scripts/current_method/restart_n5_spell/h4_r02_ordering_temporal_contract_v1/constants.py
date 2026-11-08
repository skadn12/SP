"""Fixed offline preparation and subsequent explicitly authorized comparison."""
from pathlib import Path

VERSION = 'h4_r02_ordering_temporal_contract_v1'
CODE = Path(__file__).resolve().parent
METHOD = CODE.parent
REPO = CODE.parents[3]
WORKSPACE = REPO.parent
RUNS = WORKSPACE / 'restart-n5-runs'
DEFAULT_RUN = RUNS / (VERSION + '_live_1')
SOURCE_WRITER = RUNS / 'h4_r02_writer_scms_quality_ablation_v1_live_1'
WORK_ORDER = RUNS / (VERSION + '_CODEX_TASK.md')
CONFIG = WORKSPACE / 'config.json'
ROOTS = ('cc-train-episode-198480', 'cc-train-episode-122711')
ARMS = ('BASE_ORDERING', 'TEMPORAL_CONTRACT_ORDERING')
SCHEDULE = []
for root_index, root in enumerate(ROOTS):
    for slot in range(1, 4):
        first = (root_index + slot - 1) % 2
        for arm in (ARMS[first], ARMS[1-first]):
            iid = f'intent_{slot}'
            rid = f'{root}-contract_guided-generator-{iid}'
            SCHEDULE.append(dict(order=len(SCHEDULE)+1, root_id=root, arm_id=arm,
                raw_slot=slot, raw_index=slot-1, intent_id=iid, slot_id=iid,
                request_id=rid, global_request_id=f'{arm}::{root}::{iid}::primary::{rid}'))
GENERATION = dict(model='Qwen/Qwen3-14B', base_url='https://api.siliconflow.cn/v1',
    temperature=0.1, enable_thinking=False, generation_seed_policy='not_sent_by_original_client',
    primary_max_output_tokens=8192, repair_max_output_tokens=8192, timeout_seconds=180,
    http_429_retry_limit=2, requested_counts=dict(single_choice=2, multi_select=1, ordering=1),
    repair='one_original_whole_batch_replacement_only_if_below_minimum')


def arm_policy(arm):
    if arm not in ARMS:
        raise ValueError('UNKNOWN_ARM')
    return {'version': 'legacy' if arm == ARMS[0] else 'ordering_temporal_contract_v1',
            'node_count': 4}
